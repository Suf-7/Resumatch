"""Scrape public Pakistan-based job postings from Rozee.pk (academic use only).

Policy:
  * robots.txt is checked for every URL; disallowed URLs are skipped.
  * One request every --delay seconds (default 2s) with a descriptive user agent;
    the delay doubles (up to 30s) whenever the site answers 403/429 and eases back
    toward the base delay after successful requests.
  * Only public posting text is stored (title, company, location, description,
    skills, experience, dates). No applicant or recruiter personal data.
  * Job details are read from the schema.org JobPosting JSON-LD block on each page.

Usage:
    python data_scripts/scrape_jobs.py --max-jobs 400
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common import LOGS, RAW, save_stats

BASE = "https://www.rozee.pk"
USER_AGENT = (
    "ResumatchAcademicBot/0.1 (FAST-NUCES Lahore AI semester project; "
    "non-commercial; respects robots.txt)"
)
QUERIES = [
    "software engineer", "python", "data analyst", "data scientist", "machine learning",
    "web developer", "react", "android", "devops", "qa engineer", "ui ux designer",
    "graphic designer", "accountant", "finance", "banking", "hr", "sales", "marketing",
    "digital marketing", "business development", "teacher", "civil engineer",
    "mechanical engineer", "electrical engineer", "customer service", "chef", "doctor",
    "nurse", "pharmacist", "lawyer", "content writer", "project manager", "internship",
]
JOB_URL = re.compile(r"//www\.rozee\.pk/([a-z0-9-]+-jobs-(\d+))")
MAX_CONSECUTIVE_BLOCKS = 5
OUT = RAW / "jobs_rozee_pk.jsonl"
LOG = LOGS / "scrape_rozee.log"


class PoliteSession:
    def __init__(self, delay: float):
        self.base_delay = delay
        self.delay = delay
        self.consecutive_blocks = 0
        self.last = 0.0
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.robots = robotparser.RobotFileParser(BASE + "/robots.txt")
        r = self.s.get(BASE + "/robots.txt", timeout=30)
        if r.ok:
            self.robots.parse(r.text.splitlines())
        else:
            self.robots.disallow_all = True

    def get(self, url: str) -> str | None:
        if not self.robots.can_fetch(USER_AGENT, url):
            self._log(f"robots-disallowed {url}")
            return None
        wait = self.delay - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            r = self.s.get(url, timeout=30)
        except requests.RequestException as e:
            self._log(f"error {url} {e}")
            return None
        self._log(f"{r.status_code} {url}")
        if r.status_code in (403, 429):
            self.delay = min(30.0, self.delay * 2)
            self.consecutive_blocks += 1
            return None
        self.consecutive_blocks = 0
        if r.ok:
            self.delay = max(self.base_delay, self.delay * 0.75)
        return r.text if r.ok else None

    @staticmethod
    def _log(msg: str) -> None:
        with open(LOG, "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")


def search_urls(sess: PoliteSession, query: str, pages: int) -> dict[str, str]:
    found: dict[str, str] = {}
    for page in range(pages):
        url = f"{BASE}/job/jsearch/q/{quote(query)}"
        if page:
            url += f"/fpn/{page * 20}"
        html = sess.get(url)
        if not html:
            break
        new = {jid: f"https://www.rozee.pk/{slug}" for slug, jid in JOB_URL.findall(html)}
        if not set(new) - set(found):
            break
        found.update(new)
    return found


DETAIL_FIELDS = {
    "Total Positions:": "positions", "Job Shift:": "job_shift", "Job Type:": "job_type",
    "Gender:": "gender_preference", "Age:": "age_range", "Career Level:": "career_level",
    "Experience:": "experience_text", "Minimum Education:": "min_education",
    "Industry:": "industry_text", "Functional Area:": "functional_area",
}


def parse_page_details(soup: BeautifulSoup) -> dict:
    """Key/value 'Job Details' block and the 'Skills' tag list shown on the page."""
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text("\n").split("\n") if l.strip()]
    out: dict = {}
    for i, line in enumerate(lines[:-1]):
        if line in DETAIL_FIELDS and DETAIL_FIELDS[line] not in out:
            out[DETAIL_FIELDS[line]] = lines[i + 1]
    if "Skills" in lines and "Job Details" in lines:
        s, e = lines.index("Skills"), lines.index("Job Details")
        if s < e:
            out["skills_tags"] = ", ".join(lines[s + 1 : e])
    return out


def parse_job(html: str, url: str, query: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    posting = None
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                posting = item
    if not posting:
        return None

    def first(x):
        return x[0] if isinstance(x, list) and x else x

    org = first(posting.get("hiringOrganization")) or {}
    loc = first(posting.get("jobLocation")) or {}
    addr = loc.get("address", {}) if isinstance(loc, dict) else {}
    exp = posting.get("experienceRequirements")
    if isinstance(exp, dict):
        exp = exp.get("monthsOfExperience")
    skills = posting.get("skills")
    if isinstance(skills, list):
        skills = ", ".join(map(str, skills))
    country = addr.get("addressCountry")
    if isinstance(country, dict):
        country = country.get("name")
    emp = posting.get("employmentType")
    if isinstance(emp, list):
        emp = ", ".join(map(str, emp))
    details = parse_page_details(soup)
    return {
        "job_id": url.rsplit("-", 1)[-1],
        "source": "rozee.pk",
        "query": query,
        "url": url,
        "title": posting.get("title"),
        "company": org.get("name") if isinstance(org, dict) else None,
        "city": None if str(addr.get("addressLocality", "")).isdigit() else addr.get("addressLocality"),
        "region": addr.get("addressRegion"),
        "country": country,
        "description_html": posting.get("description"),
        "skills": skills or details.pop("skills_tags", None),
        "industry": posting.get("industry"),
        "employment_type": emp,
        "experience": exp,
        "education": json.dumps(posting.get("educationRequirements")) if posting.get("educationRequirements") else None,
        "date_posted": posting.get("datePosted"),
        "valid_through": posting.get("validThrough"),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in details.items() if k != "skills_tags"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-jobs", type=int, default=400)
    ap.add_argument("--pages", type=int, default=2, help="search result pages per query")
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--per-query", type=int, default=15, help="max new postings per query")
    ap.add_argument("--stats-only", action="store_true", help="recompute stats, no requests")
    args = ap.parse_args()
    if args.stats_only:
        write_stats()
        return

    done, covered = set(), set()
    if OUT.exists():
        for line in OUT.open():
            rec = json.loads(line)
            done.add(rec["job_id"])
            covered.add(rec["query"])
    sess = PoliteSession(args.delay)

    queue: dict[str, tuple[str, str]] = {}
    searched = 0
    for q in sorted(QUERIES, key=lambda q: q in covered):
        searched += 1
        added = 0
        for jid, url in search_urls(sess, q, args.pages).items():
            if jid not in done and jid not in queue and added < args.per_query:
                queue[jid] = (url, q)
                added += 1
        print(f"[search] {q!r}: +{added} queue={len(queue)}")
        if len(queue) + len(done) >= args.max_jobs:
            break

    n_ok = n_fail = 0
    with OUT.open("a") as f:
        for jid, (url, q) in list(queue.items())[: max(0, args.max_jobs - len(done))]:
            html = sess.get(url)
            rec = parse_job(html, url, q) if html else None
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                n_fail += 1
            if sess.consecutive_blocks >= MAX_CONSECUTIVE_BLOCKS:
                print(f"[stop] {sess.consecutive_blocks} consecutive 403/429 responses; respecting the block")
                break
            if (n_ok + n_fail) % 25 == 0:
                print(f"[detail] ok={n_ok} fail={n_fail}")

    write_stats({"new_this_run": n_ok, "failed_this_run": n_fail,
                 "queries_searched_this_run": searched, "delay_s": args.delay})


def write_stats(run: dict | None = None) -> None:
    df = pd.read_json(OUT, lines=True, dtype={"job_id": str})
    log = LOG.read_text().splitlines() if LOG.exists() else []
    stats = {
        "file": OUT.name,
        "postings_total": len(df),
        **(run or {}),
        "queries_defined": len(QUERIES),
        "queries_covered_total": int(df["query"].nunique()),
        "postings_by_query": df["query"].value_counts().to_dict(),
        "requests_logged": len(log),
        "requests_blocked_403_429": sum(" 403 " in l or " 429 " in l for l in log),
        "robots_disallowed": sum("robots-disallowed" in l for l in log),
        "user_agent": USER_AGENT,
        "top_cities": df["city"].value_counts().head(8).to_dict(),
    }
    save_stats("scrape_rozee", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
