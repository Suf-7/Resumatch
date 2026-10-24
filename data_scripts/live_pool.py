"""Build the pool of open, applyable postings that users are matched against.

Only Rozee.pk postings qualify: they are current, Pakistan-based and keep the apply URL
and closing date. LinkedIn (Kaggle) postings closed in 2023-24 and are used for training
only; the closing-date filter excludes them automatically.

Rozee.pk postings close within days, so run `make refresh` (scrape + rebuild) regularly.

Outputs:
  data/processed/jobs_live.parquet   open postings + row index into emb_jobs.npy
  report/stats/live_pool.json
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common import PROCESSED, save_stats

LIVE_SOURCES = {"rozee_pk_scraped"}
KEEP = ["job_id", "title", "company", "location", "url", "date_posted", "valid_through",
        "seniority", "job_category", "skills", "n_skills", "years_required", "emb_row"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", help="reference time (default: now, UTC)")
    args = ap.parse_args()
    now = pd.Timestamp(args.as_of or pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    jobs = pd.read_parquet(PROCESSED / "jobs_features.parquet")
    jobs["emb_row"] = range(len(jobs))
    cand = jobs[jobs["source"].isin(LIVE_SOURCES)]
    has_url = cand["url"].notna()
    is_open = cand["valid_through"].notna() & (cand["valid_through"] >= now)
    live = cand[has_url & is_open].sort_values("valid_through")
    live[KEEP].to_parquet(PROCESSED / "jobs_live.parquet", index=False)

    days_left = (live["valid_through"] - now).dt.total_seconds() / 86400
    stats = {
        "as_of": now.isoformat(),
        "candidates": int(len(cand)),
        "open": int(len(live)),
        "expired": int((cand["valid_through"] < now).sum()),
        "missing_url_or_date": int((~has_url | cand["valid_through"].isna()).sum()),
        "closing_within_7_days": int((days_left <= 7).sum()),
        "median_days_left": round(float(days_left.median()), 1) if len(live) else None,
        "open_with_years_required": round(float(live["years_required"].notna().mean()), 3) if len(live) else None,
        "open_mean_skills": round(float(live["n_skills"].mean()), 2) if len(live) else None,
        "open_by_category": live["job_category"].value_counts().to_dict(),
        "open_by_city": live["location"].value_counts().head(8).to_dict(),
    }
    save_stats("live_pool", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
