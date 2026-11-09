"""Hand-crafted pair features shared by the baselines and MatchNet.

All of them are interpretable and are also what the user sees in the ATS breakdown.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import PROCESSED
from ats_score import ats_score, experience_score, section_score, title_score
from build_features import required_years, resume_years, tfidf_normalize
from skills import extract_skills, load_gazetteer

SCALAR_FEATURES = [
    "tfidf_cos", "skill_cov", "skill_jacc", "log_job_skills", "log_res_skills",
    "years_fit", "years_missing", "title_match", "structure", "ats_score",
]


def add_side_features(df: pd.DataFrame) -> pd.DataFrame:
    """Skills and years for pairs that do not come from the Deliverable 1 pipeline."""
    gaz = load_gazetteer()
    df = df.copy()
    for col, out in (("resume_text", "resume_skills"), ("job_text", "job_skills")):
        if out not in df:
            uniq = df[col].unique()
            found = {t: extract_skills(t, gaz) for t in uniq}
            df[out] = df[col].map(found)
    if "years_exp" not in df:
        uniq = df["resume_text"].unique()
        df["years_exp"] = df["resume_text"].map({t: resume_years(t) for t in uniq})
    if "years_required" not in df:
        uniq = df["job_text"].unique()
        df["years_required"] = df["job_text"].map({t: required_years(t) for t in uniq})
    return df


def scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    vec = joblib.load(PROCESSED / "tfidf_vectorizer.joblib")
    r_text = df["resume_text"].map(tfidf_normalize)
    j_text = df["job_text"].map(tfidf_normalize)
    R, J = vec.transform(r_text), vec.transform(j_text)
    tfidf_cos = np.asarray(R.multiply(J).sum(axis=1)).ravel()

    rows = []
    for t, rs, js, ry, jy, title in zip(df["resume_text"], df["resume_skills"], df["job_skills"],
                                       df["years_exp"], df["years_required"], df["job_title"]):
        rs, js = set(rs), list(dict.fromkeys(js))
        inter = sum(s in rs for s in js)
        union = len(rs | set(js))
        exp = experience_score(ry, jy)
        tm = title_score(title, t.lower())
        rows.append({
            "skill_cov": inter / len(js) if js else 0.0,
            "skill_jacc": inter / union if union else 0.0,
            "log_job_skills": np.log1p(len(js)),
            "log_res_skills": np.log1p(len(rs)),
            "years_fit": exp if exp is not None else 1.0,
            "years_missing": float(exp is None),
            "title_match": tm if tm is not None else 0.0,
            "structure": section_score(t),
            "ats_score": ats_score(t, rs, ry, title, js, jy).score / 100,
        })
    out = pd.DataFrame(rows, index=df.index)
    out.insert(0, "tfidf_cos", tfidf_cos)
    return out[SCALAR_FEATURES].astype(np.float32)
