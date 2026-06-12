# """
# database.py — PostgreSQL connection factory.
# Every router/service calls get_connection() to obtain a fresh connection.
# Connections are closed in the finally block of each endpoint.
# """

# import psycopg2
# from config import PG_HOST, PG_DB, PG_USER, PG_PASS, PG_PORT


# def get_connection():
#     """Return a new psycopg2 connection to the int-doc-class database."""
#     return psycopg2.connect(
#         host=PG_HOST,
#         dbname=PG_DB,
#         user=PG_USER,
#         password=PG_PASS,
#         port=int(PG_PORT),
#     )









"""
database.py — PostgreSQL connection pool for the Document Intelligence platform.

WHAT CHANGED vs previous version
──────────────────────────────────
OLD (v1): get_connection() opened a brand-new psycopg2 connection on every call.
          Under concurrent load this exhausted PostgreSQL's max_connections limit.

OLD (v2): Used monkey-patching to redirect conn.close() → pool.putconn().
          Failed with:
          AttributeError: 'psycopg2.extensions.connection' object attribute
          'close' is read-only
          psycopg2 is a C extension — its connection object does not allow
          replacing built-in methods at runtime.

NEW (v3): PooledConnection wrapper class.
          get_connection() returns a PooledConnection that wraps the real
          psycopg2 connection. PooledConnection forwards every attribute
          access to the underlying connection EXCEPT .close(), which it
          overrides to call pool.putconn() instead.
          This works because Python allows overriding methods on pure-Python
          classes freely — only C extension types are read-only.

          ALL existing routers keep their conn.close() calls unchanged.
          Zero changes needed anywhere else in the project.

──────────────────────────────────────────────────────────────────────────────
Pool sizing
────────────
  MIN_POOL_CONN (default 2)  — connections kept open even when idle.
  MAX_POOL_CONN (default 20) — hard cap on simultaneous connections.

  Azure Database for PostgreSQL Flexible Server default max_connections is
  50–100 depending on the SKU. Keep MAX_POOL_CONN well below that to leave
  headroom for pgAdmin, migrations, and monitoring tools.

  Override via environment variables without touching code:
    DB_POOL_MIN_CONN=2
    DB_POOL_MAX_CONN=20

Usage pattern in all routers (unchanged — works as-is)
────────────────────────────────────────────────────────
  conn = get_connection()
  try:
      # ... database work ...
      conn.commit()
  except Exception:
      conn.rollback()
      raise
  finally:
      conn.close()   ← PooledConnection.close() → puts back into pool

Optional context manager for new code
───────────────────────────────────────
  from database import with_connection

  with with_connection() as conn:
      with conn.cursor() as cur:
          cur.execute("SELECT 1")
      # commits automatically, releases on exit
"""

import atexit
import logging
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extensions
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor   # re-exported for convenience

from config import PG_HOST, PG_DB, PG_USER, PG_PASS, PG_PORT

log = logging.getLogger(__name__)

# ── Pool configuration ────────────────────────────────────────────────────────

_MIN_CONN = int(os.environ.get("DB_POOL_MIN_CONN", "2"))
_MAX_CONN = int(os.environ.get("DB_POOL_MAX_CONN", "20"))


# ── Pool initialisation ───────────────────────────────────────────────────────

def _create_pool() -> pg_pool.ThreadedConnectionPool:
    """
    Create the connection pool at module import time.
    Raises immediately if the database is unreachable so the problem is
    visible at startup rather than on the first request.
    """
    try:
        p = pg_pool.ThreadedConnectionPool(
            minconn=_MIN_CONN,
            maxconn=_MAX_CONN,
            host=PG_HOST,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS,
            port=int(PG_PORT),
            # Keep idle connections alive through Azure's 4-minute TCP idle timeout
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=10,
            keepalives_count=5,
            # Fail fast if the server is unreachable
            connect_timeout=10,
        )
        log.info(
            "PostgreSQL connection pool created — min=%d, max=%d, host=%s, db=%s",
            _MIN_CONN, _MAX_CONN, PG_HOST, PG_DB,
        )
        return p
    except psycopg2.OperationalError as exc:
        log.critical("Failed to create PostgreSQL connection pool: %s", exc)
        raise


_pool: pg_pool.ThreadedConnectionPool = _create_pool()

# Close the pool gracefully when the process exits
atexit.register(lambda: _pool.closed or _pool.closeall())


# ── PooledConnection wrapper ──────────────────────────────────────────────────

class PooledConnection:
    """
    A thin wrapper around a real psycopg2 connection that returns the
    connection to the pool when .close() is called, instead of destroying it.

    Why a wrapper class and not monkey-patching?
    ────────────────────────────────────────────
    psycopg2 is implemented as a C extension. Its connection type does not
    allow replacing built-in methods at runtime:
        conn.close = my_func  →  AttributeError: attribute 'close' is read-only

    Python classes have no such restriction — overriding a method in a
    subclass or wrapper is always allowed. __getattr__ forwards every
    attribute that PooledConnection itself does not define directly to the
    underlying psycopg2 connection, so callers see a fully transparent object.

    All existing routers call conn.close() in their finally blocks.
    With this wrapper, conn.close() silently returns the connection to the
    pool instead. No router needs to be changed.
    """

    __slots__ = ("_conn", "_pool", "_closed")

    def __init__(self, conn: psycopg2.extensions.connection, pool: pg_pool.ThreadedConnectionPool):
        # Use object.__setattr__ to bypass our own __setattr__ for slot init
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_closed", False)

    # ── Core override: close() returns to pool ────────────────────────────────

    def close(self) -> None:
        """Return this connection to the pool instead of closing the socket."""
        if not object.__getattribute__(self, "_closed"):
            object.__setattr__(self, "_closed", True)
            pool = object.__getattribute__(self, "_pool")
            conn = object.__getattribute__(self, "_conn")
            try:
                pool.putconn(conn)
                log.debug("DB connection returned to pool")
            except Exception as exc:
                log.warning("Failed to return connection to pool: %s", exc)

    # ── Transparent proxy: forward everything else to the real connection ─────

    def __getattr__(self, name: str):
        """Forward any attribute not defined on PooledConnection to the real conn."""
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name: str, value) -> None:
        """Forward attribute sets to the real connection (e.g. autocommit)."""
        if name in ("_conn", "_pool", "_closed"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_conn"), name, value)

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return object.__getattribute__(self, "_conn").__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        conn = object.__getattribute__(self, "_conn")
        result = conn.__exit__(exc_type, exc_val, exc_tb)
        self.close()
        return result


# ── Public API ────────────────────────────────────────────────────────────────

def get_connection() -> PooledConnection:
    """
    Borrow a connection from the pool and return it wrapped in PooledConnection.

    All existing routers call conn.close() in their finally blocks.
    PooledConnection.close() returns the connection to the pool so it can
    be reused — no router changes required.

    Raises psycopg2.pool.PoolError if all MAX_CONN connections are checked out.
    """
    raw_conn = _pool.getconn()
    log.debug("DB connection borrowed from pool")
    return PooledConnection(raw_conn, _pool)


def release_connection(conn: PooledConnection) -> None:
    """
    Explicitly return a connection to the pool.
    Equivalent to calling conn.close() — provided as a named alternative
    for clarity in new code.
    """
    conn.close()


@contextmanager
def with_connection():
    """
    Context manager: borrows a connection, commits on success,
    rolls back on exception, and always releases back to the pool.

    Usage:
        from database import with_connection

        with with_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()