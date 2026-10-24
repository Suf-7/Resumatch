"""Generate synthetic CVs for under-represented roles with an LLM (logged).

Under-represented roles are taken from the cleaned resume category counts, plus a
fixed list of Pakistan-market entry-level roles missing from the Kaggle data.
Names, emails, phones and addresses are always placeholders, so no real PII is
ever produced. Each CV is tagged `synthetic=True` so it can be excluded from
evaluation sets.

Usage:
    python data_scripts/generate_synthetic_cvs.py --per-role 20 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from common import INTERIM, RAW, SEED
from llm_client import LLMClient, parse_json

EXTRA_ROLES = [
    "Junior Data Scientist", "Machine Learning Intern", "Flutter Developer",
    "SQA Engineer", "DevOps Engineer", "Cyber Security Analyst", "Business Analyst",
    "Supply Chain Officer", "Content Writer", "Customer Support Representative",
]
SENIORITY = ["intern", "fresh graduate", "junior (1-2 years)", "mid-level (3-5 years)"]
STYLE = ["concise bullet points", "short paragraphs", "skills-first layout", "projects-heavy"]

SYSTEM = (
    "You write realistic, varied resumes in plain text for training a resume-job matching "
    "model. Never use real people or companies' confidential data. Use placeholders "
    "[NAME], [EMAIL], [PHONE], [ADDRESS] for personal details."
)
PROMPT = """Write one resume for this candidate: {role}, {seniority}, based in {city}, Pakistan.
Style: {style}. Include sections: Summary, Skills, Experience (or Projects), Education.
Mention 8-15 concrete skills a recruiter would search for. Vary wording; do not copy a template.
Return JSON: {{"category": "{category}", "title": "...", "resume_text": "..."}}"""


def under_represented(threshold_quantile: float = 0.25) -> list[str]:
    path = INTERIM / "resumes_clean.parquet"
    if not path.exists():
        return []
    counts = pd.read_parquet(path)["category"].value_counts()
    return counts[counts <= counts.quantile(threshold_quantile)].index.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-role", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(SEED)
    roles = [(c.replace("-", " ").title(), c) for c in under_represented()]
    roles += [(r, r.upper().replace(" ", "-")) for r in EXTRA_ROLES]
    llm = LLMClient("synthetic_cv", dry_run=args.dry_run, temperature=0.9)
    out = RAW / "resumes_synthetic.jsonl"
    n = 0
    with out.open("a") as f:
        for role, category in roles:
            for i in range(args.per_role):
                params = dict(role=role, category=category, seniority=rng.choice(SENIORITY),
                              style=rng.choice(STYLE),
                              city=rng.choice(["Lahore", "Karachi", "Islamabad", "Faisalabad", "Peshawar"]))
                resp = parse_json(llm.chat(PROMPT.format(**params), SYSTEM, json_mode=True,
                                           meta={"role": role, "i": i}))
                if resp and resp.get("resume_text"):
                    f.write(json.dumps({**params, **resp, "synthetic": True}) + "\n")
                    n += 1
    print(f"wrote {n} synthetic CVs to {out} ({len(roles)} roles)")


if __name__ == "__main__":
    main()
