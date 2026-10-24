"""Pre-label CV-job pairs as poor / partial / good with an LLM (logged).

Inputs: data/processed/pairs.parquet (built by build_features.py). Only
anonymized text is sent to the API. Labels are written to
data/processed/pairs_llm_labels.parquet and are *never* used for the human gold set.

Usage:
    python data_scripts/label_pairs.py --n 2000 [--dry-run]
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import INTERIM, PROCESSED, SEED
from llm_client import LLMClient, parse_json

SYSTEM = (
    "You are an experienced technical recruiter. You judge how well a candidate's resume "
    "fits a job posting. Be strict and consistent. Ignore names, gender, age, nationality "
    "and university prestige; judge only skills, experience and role fit."
)
PROMPT = """JOB POSTING
Title: {title}
{job_text}

RESUME
{resume_text}

Rate the fit as one of: "poor", "partial", "good".
- good: meets most required skills and the seniority level
- partial: relevant background but clear skill or seniority gaps
- poor: different role or most required skills missing
Return JSON: {{"label": "...", "matched_skills": [...], "missing_skills": [...], "reason": "<=30 words"}}"""

MAX_CHARS = 3500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs = pd.read_parquet(PROCESSED / "pairs.parquet")
    pairs = pairs[pairs["split"] == args.split]
    pairs = pairs.groupby("weak_label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.n // 2), random_state=SEED)
    )
    resumes = pd.read_parquet(INTERIM / "resumes_clean.parquet").set_index("resume_id")
    jobs = pd.read_parquet(INTERIM / "jobs_clean.parquet").set_index("job_id")

    llm = LLMClient("pair_labeling", dry_run=args.dry_run, temperature=0.0)
    rows = []
    for _, p in pairs.iterrows():
        r, j = resumes.loc[p.resume_id], jobs.loc[p.job_id]
        prompt = PROMPT.format(title=j.title, job_text=j.text[:MAX_CHARS],
                               resume_text=r.text_anon[:MAX_CHARS])
        out = parse_json(llm.chat(prompt, SYSTEM, json_mode=True,
                                  meta={"resume_id": p.resume_id, "job_id": p.job_id}))
        if out and out.get("label") in {"poor", "partial", "good"}:
            rows.append({"resume_id": p.resume_id, "job_id": p.job_id, "llm_label": out["label"],
                         "matched_skills": out.get("matched_skills"),
                         "missing_skills": out.get("missing_skills"), "reason": out.get("reason")})
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(PROCESSED / "pairs_llm_labels.parquet", index=False)
        print(df["llm_label"].value_counts())
    else:
        print("no labels produced (dry run or API failure); prompts logged to logs/api_calls.jsonl")


if __name__ == "__main__":
    main()
