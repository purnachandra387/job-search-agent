"""Fetch 350+ real job postings from public Remotive and Arbeitnow Job APIs into SQLite.
Usage:
    python fetch_remotive_jobs.py
"""
import re
import sqlite3
import httpx
import load_jobs
REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"
TAG_RE = re.compile(r"<[^>]+>")
def strip_html(html_text: str) -> str:
    text = TAG_RE.sub(" ", html_text or "")
    return " ".join(text.split())[:1500]
def fetch_and_load(target_count: int = 350) -> int:
    records: list[tuple[str, str, str, str]] = []
    try:
        r_resp = httpx.get(REMOTIVE_API_URL, timeout=30.0)
        r_resp.raise_for_status()
        for job in r_resp.json().get("jobs", []):
            title = (job.get("title") or "").strip()
            company = (job.get("company_name") or "").strip()
            location = (job.get("candidate_required_location") or "Remote").strip()
            desc = strip_html(job.get("description") or "")
            if title and company and desc:
                records.append((title, company, location, desc))
    except Exception as exc:
        print(f"Remotive fetch warning: {exc}")
    for page in range(1, 6):
        if len(records) >= target_count:
            break
        try:
            a_resp = httpx.get(ARBEITNOW_API_URL, params={"page": page}, timeout=30.0)
            a_resp.raise_for_status()
            page_jobs = a_resp.json().get("data", [])
            if not page_jobs:
                break
            for job in page_jobs:
                title = (job.get("title") or "").strip()
                company = (job.get("company_name") or "").strip()
                location = (job.get("location") or "Remote").strip()
                desc = strip_html(job.get("description") or "")
                if title and company and desc:
                    records.append((title, company, location, desc))
                if len(records) >= target_count:
                    break
        except Exception as exc:
            print(f"Arbeitnow page {page} warning: {exc}")
            break
    conn = sqlite3.connect(load_jobs.DB_PATH)
    load_jobs.init_db(conn)
    for idx, (title, company, location, desc) in enumerate(records, start=9):
        conn.execute(
            "INSERT OR REPLACE INTO jobs (id, title, company, location, description) VALUES (?, ?, ?, ?, ?)",
            (idx, title, company, location, desc),
        )
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    conn.commit()
    conn.close()
    print(f"Fetched {len(records)} real jobs from public APIs (Total in {load_jobs.DB_PATH}: {total} jobs)")
    return total
if __name__ == "__main__":
    fetch_and_load()
