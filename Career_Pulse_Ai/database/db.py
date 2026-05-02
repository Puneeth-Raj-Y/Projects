"""
database/db.py
--------------
All database access lives here.  app.py should import only from this module.

Schema
------
  users           (id, username, password_hash)
  jobs            (id, title, skills_json)
  recommendations (id, user_id, job_id, score, created_at)

Security
--------
  Passwords are stored as bcrypt hashes.  Plain-text storage is removed.
"""

import os
import sqlite3
import json
from typing import Optional, Dict, Any, List

import bcrypt

DB_PATH: str = os.getenv("DB_PATH", "users.db")


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return an open connection with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not already exist."""
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            skills_json TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
            job_id     INTEGER NOT NULL REFERENCES jobs(id)   ON DELETE CASCADE,
            score      REAL    NOT NULL DEFAULT 0.0,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        );
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

def create_user(username: str, plain_password: str) -> str:
    """
    Insert a new user with a bcrypt-hashed password.

    Returns:
        "success"  — user created.
        "exists"   — username already taken.
        "error"    — unexpected database error.
    """
    password_hash = bcrypt.hashpw(
        plain_password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return "success"
    except sqlite3.IntegrityError:
        return "exists"
    except Exception as exc:
        print(f"[db] create_user error: {exc}")
        return "error"
    finally:
        conn.close()


def verify_user(username: str, plain_password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user.

    Returns:
        A dict with {"id": int, "username": str} on success, or None on failure.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    stored_hash: str = row["password_hash"]

    # Support legacy plain-text passwords during transition
    try:
        is_valid = bcrypt.checkpw(
            plain_password.encode("utf-8"), stored_hash.encode("utf-8")
        )
    except Exception:
        # If the stored value is not a valid bcrypt hash (legacy plain text)
        is_valid = plain_password == stored_hash

    if not is_valid:
        return None

    return {"id": row["id"], "username": row["username"]}


def get_user_by_id(uid: int) -> Optional[Dict[str, Any]]:
    """Return a dict for the given user id, or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT id, username FROM users WHERE id = ?", (uid,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Job cache operations
# ---------------------------------------------------------------------------

def upsert_job(title: str, skills: List[str]) -> int:
    """
    Insert a job or update its skills if a row with the same title exists.
    Returns the job row id.
    """
    skills_json = json.dumps(skills)
    conn = get_db()
    c = conn.cursor()

    existing = c.execute(
        "SELECT id FROM jobs WHERE title = ?", (title,)
    ).fetchone()

    if existing:
        job_id = existing["id"]
        c.execute(
            "UPDATE jobs SET skills_json = ? WHERE id = ?",
            (skills_json, job_id),
        )
    else:
        c.execute(
            "INSERT INTO jobs (title, skills_json) VALUES (?, ?)",
            (title, skills_json),
        )
        job_id = c.lastrowid

    conn.commit()
    conn.close()
    return job_id


# ---------------------------------------------------------------------------
# Recommendation operations
# ---------------------------------------------------------------------------

def save_recommendation(user_id: int, job_id: int, score: float) -> None:
    """Persist a recommendation (user ↔ job ↔ score)."""
    conn = get_db()
    conn.execute(
        """
        INSERT INTO recommendations (user_id, job_id, score)
        VALUES (?, ?, ?)
        """,
        (user_id, job_id, score),
    )
    conn.commit()
    conn.close()
