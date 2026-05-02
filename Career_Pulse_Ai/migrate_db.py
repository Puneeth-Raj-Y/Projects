"""
migrate_db.py
-------------
One-shot migration:
  1. Rename `users.password`  →  `users.password_hash`
  2. Create `jobs` table if missing
  3. Create `recommendations` table if missing

Safe to re-run (all statements are idempotent).
"""

import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "users.db")


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def migrate():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── 1. Rename password → password_hash ─────────────────────────
    if column_exists(c, "users", "password") and not column_exists(c, "users", "password_hash"):
        print("[migrate] Renaming column: users.password -> users.password_hash")
        c.execute("ALTER TABLE users RENAME COLUMN password TO password_hash")
    else:
        print("[migrate] users.password_hash already exists — skipping rename")

    # ── 2. Create jobs table ────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            skills_json TEXT NOT NULL DEFAULT '[]'
        )
    """)

    # ── 3. Create recommendations table ────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
            job_id     INTEGER NOT NULL REFERENCES jobs(id)   ON DELETE CASCADE,
            score      REAL    NOT NULL DEFAULT 0.0,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print("[migrate] Done — schema is up to date.")


if __name__ == "__main__":
    migrate()
