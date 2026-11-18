"""Load jobs from a CSV file into a SQLite database.

Usage: python load_jobs.py sample_jobs.csv
CSV columns: id, title, company, location, description
"""

import csv
import os
import sqlite3
import sys

DB_PATH = os.getenv("JOBS_DB_PATH", "jobs.db")


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            location TEXT,
            description TEXT NOT NULL
        )
        """)


def load_csv(path: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    count = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # INSERT OR REPLACE makes the load idempotent: re-running never duplicates rows
            conn.execute(
                "INSERT OR REPLACE INTO jobs (id, title, company, location, description) "
                "VALUES (?, ?, ?, ?, ?)",
                (int(row["id"]), row["title"], row["company"], row["location"], row["description"]),
            )
            count += 1
    conn.commit()
    conn.close()
    return count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python load_jobs.py <jobs.csv>")
    print(f"Loaded {load_csv(sys.argv[1])} jobs into {DB_PATH}")
