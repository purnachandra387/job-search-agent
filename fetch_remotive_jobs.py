"""Fetch real job postings from the public Remotive and Arbeitnow APIs into SQLite.

Job ids are derived from (company, title, location), so re-running this script updates
existing rows instead of shifting ids around. After fetching, rebuild the search index:

    python fetch_remotive_jobs.py
    python search.py build

Note: eval_search.py expects ids 1-8 from sample_jobs.csv. Loading hundreds of extra
postings makes that benchmark much harder (more distractors), which is a useful stress test.
"""

import hashlib
import re
import sqlite3

import httpx

import load_jobs

REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"
ID_OFFSET = 1_000_000  # keeps fetched ids clear of the hand-written sample ids
ID_RANGE = 1_000_000_000
TAG_RE = re.compile(r"<[^>]+>")

Record = tuple[int, str, str, str, str]  # id, title, company, location, description


def strip_html(html_text: str) -> str:
    text = TAG_RE.sub(" ", html_text or "")
    return " ".join(text.split())[:1500]


def stable_id(title: str, company: str, location: str) -> int:
    digest = hashlib.sha1(f"{company}|{title}|{location}".lower().encode("utf-8")).hexdigest()
    return ID_OFFSET + int(digest[:8], 16) % ID_RANGE


def make_record(title: str, company: str, location: str, description_html: str) -> Record | None:
    title, company, location = title.strip(), company.strip(), (location or "Remote").strip()
    desc = strip_html(description_html)
    if not (title and company and desc):
        return None
    return stable_id(title, company, location), title, company, location, desc


def fetch_remotive() -> list[Record]:
    resp = httpx.get(REMOTIVE_API_URL, timeout=30.0)
    resp.raise_for_status()
    records = (
        make_record(
            j.get("title") or "",
            j.get("company_name") or "",
            j.get("candidate_required_location") or "Remote",
            j.get("description") or "",
        )
        for j in resp.json().get("jobs", [])
    )
    return [r for r in records if r]


def fetch_arbeitnow(pages: int = 5) -> list[Record]:
    out: list[Record] = []
    for page in range(1, pages + 1):
        resp = httpx.get(ARBEITNOW_API_URL, params={"page": page}, timeout=30.0)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        for j in data:
            rec = make_record(
                j.get("title") or "",
                j.get("company_name") or "",
                j.get("location") or "Remote",
                j.get("description") or "",
            )
            if rec:
                out.append(rec)
    return out


def fetch_and_load(target_count: int = 350) -> int:
    unique: dict[int, Record] = {}
    for name, fetcher in (("Remotive", fetch_remotive), ("Arbeitnow", fetch_arbeitnow)):
        if len(unique) >= target_count:
            break
        try:
            for rec in fetcher():
                unique.setdefault(rec[0], rec)  # dedupe identical postings
        except (httpx.HTTPError, ValueError) as exc:
            print(f"{name} fetch warning: {exc}")
    records = list(unique.values())[:target_count]

    conn = sqlite3.connect(load_jobs.DB_PATH)
    load_jobs.init_db(conn)
    conn.executemany(
        "INSERT OR REPLACE INTO jobs (id, title, company, location, description) VALUES (?, ?, ?, ?, ?)",
        records,
    )
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    conn.commit()
    conn.close()
    print(
        f"Fetched {len(records)} jobs (total in {load_jobs.DB_PATH}: {total}). Now run: python search.py build"
    )
    return total


if __name__ == "__main__":
    fetch_and_load()
