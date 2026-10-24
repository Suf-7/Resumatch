"""Baseline ATS score (0-100) of one resume for one specific posting.

Score = weighted mean of the components that can be computed for the pair:
  skills      0.50  share of the posting's ESCO skills found in the resume
  experience  0.20  resume years / required years (capped at 1); skipped if not stated
  title       0.15  share of the posting title's content words found in the resume
  structure   0.15  contact details, experience, education and skills sections present
Missing skills are returned for the skill-gap report. The skills component is the part
the Phase 3 skill-extraction model replaces; the weights are tuned on the gold set.

Usage:
  python data_scripts/ats_score.py evaluate          # sanity check on the test pairs
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import INTERIM, PROCESSED, save_stats
from skills import extract_skills, load_gazetteer

WEIGHTS = {"skills": 0.50, "experience": 0.20, "title": 0.15, "structure": 0.15}
SECTIONS = {
    "contact": re.compile(r"\[(EMAIL|PHONE)\]|[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{8,}\d", re.I),
    "experience": re.compile(r"\b(experience|employment|work history|career history)\b", re.I),
    "education": re.compile(r"\b(education|university|college|bachelor|master'?s?|degree)\b", re.I),
    "skills": re.compile(r"\b(skills|competencies|highlights|proficienc(y|ies)|expertise)\b", re.I),
}
TITLE_STOP = {
    "a", "an", "and", "the", "of", "for", "in", "at", "to", "with", "or", "senior", "sr", "junior",
    "jr", "lead", "head", "intern", "internship", "trainee", "associate", "assistant", "i", "ii",
    "iii", "iv", "level", "entry", "mid", "remote", "hybrid", "onsite", "full", "part", "time",
    "urgent", "required", "needed", "job", "jobs", "position", "role", "staff", "officer",
}


@dataclass
class ATSResult:
    score: float
    components: dict = field(default_factory=dict)
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)


def section_score(text: str) -> float:
    return sum(bool(rx.search(text)) for rx in SECTIONS.values()) / len(SECTIONS)


def title_score(title: str, resume_lower: str) -> float | None:
    words = [w for w in re.findall(r"[a-z][a-z+#.]*", (title or "").lower())
             if w not in TITLE_STOP and len(w) > 1]
    if not words:
        return None
    return sum(bool(re.search(rf"\b{re.escape(w)}", resume_lower)) for w in words) / len(words)


def experience_score(resume_years: float, required: float) -> float | None:
    if required is None or np.isnan(required):
        return None
    if required <= 0:
        return 1.0
    if resume_years is None or np.isnan(resume_years):
        return 0.0
    return float(min(1.0, resume_years / required))


def ats_score(resume_text: str, resume_skills, resume_years: float,
              job_title: str, job_skills, required_years: float) -> ATSResult:
    job_skills = list(dict.fromkeys(job_skills))
    have = set(resume_skills)
    matched = [s for s in job_skills if s in have]
    missing = [s for s in job_skills if s not in have]
    comp = {
        "skills": len(matched) / len(job_skills) if job_skills else None,
        "experience": experience_score(resume_years, required_years),
        "title": title_score(job_title, resume_text.lower()),
        "structure": section_score(resume_text),
    }
    used = {k: v for k, v in comp.items() if v is not None}
    total = sum(WEIGHTS[k] for k in used)
    score = 100 * sum(WEIGHTS[k] * v for k, v in used.items()) / total
    return ATSResult(round(score, 1), {k: (None if v is None else round(v, 3)) for k, v in comp.items()},
                     matched, missing)


def score_text(resume_text: str, job: pd.Series, gaz=None, resume_years: float = np.nan) -> ATSResult:
    """Score a raw resume against one row of jobs_live.parquet / jobs_features.parquet."""
    gaz = gaz or load_gazetteer()
    return ats_score(resume_text, extract_skills(resume_text, gaz), resume_years,
                     job["title"], job["skills"], job["years_required"])


def evaluate() -> None:
    pairs = pd.read_parquet(PROCESSED / "pairs.parquet")
    pairs = pairs[pairs["split"] == "test"]
    res = pd.read_parquet(PROCESSED / "resumes_features.parquet").set_index("resume_id")
    text = pd.read_parquet(INTERIM / "resumes_clean.parquet").set_index("resume_id")["text_anon"]
    jobs = pd.read_parquet(PROCESSED / "jobs_features.parquet").set_index("job_id")
    rows = []
    for p in pairs.itertuples():
        r, j = res.loc[p.resume_id], jobs.loc[p.job_id]
        out = ats_score(text[p.resume_id], r["skills"], r["years_exp"],
                        j["title"], j["skills"], j["years_required"])
        rows.append({"pair_type": p.pair_type, "score": out.score, **out.components})
    df = pd.DataFrame(rows)
    pos = df["pair_type"] == "positive"

    def auc(neg: str) -> float:
        m = pos | (df["pair_type"] == neg)
        return round(float(roc_auc_score(pos[m], df.loc[m, "score"])), 3)

    stats = {
        "test_pairs": len(df),
        "mean_score_by_type": df.groupby("pair_type")["score"].mean().round(1).to_dict(),
        "component_mean_by_type": df.groupby("pair_type")[list(WEIGHTS)].mean().round(3).to_dict(),
        "component_coverage": df[list(WEIGHTS)].notna().mean().round(3).to_dict(),
        "auc_pos_vs_hard": auc("hard_negative"),
        "auc_pos_vs_random": auc("random_negative"),
        "weights": WEIGHTS,
    }
    save_stats("ats_baseline", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evaluate"])
    args = ap.parse_args()
    evaluate()
