"""auth/password.py — bcrypt password hashing using the bcrypt library directly.

Uses bcrypt >= 4.x directly instead of passlib (passlib 1.7.4 is incompatible
with bcrypt >= 4.0.0 due to the removal of bcrypt.__about__).
"""

import bcrypt


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False
