"""Pair datasets for Phase 3/4.

Task A ("weak"): the 15,755 resume-posting pairs built in Deliverable 1. Label 1 = same
occupation category (positive), 0 = hard or random negative. Split by resume and by posting.

Task B ("human"): the external resume-job description fit pairs, labelled by people as
No Fit / Potential Fit / Good Fit. Binary label 1 = Potential or Good Fit. The official test
split is kept as the test set; validation is 15% of the training job descriptions (grouped by
job description, so no description appears in two splits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import INTERIM, PROCESSED, RAW, SEED
from common import anonymize, clean_text  # noqa: E402  (data_scripts on sys.path)

FIT_LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


def load_weak_pairs() -> pd.DataFrame:
    """One row per pair with texts and the Deliverable 1 features of both sides."""
    pairs = pd.read_parquet(PROCESSED / "pairs.parquet")
    res = pd.read_parquet(PROCESSED / "resumes_features.parquet")[
        ["resume_id", "skills", "years_exp", "category"]]
    res = res.merge(pd.read_parquet(INTERIM / "resumes_clean.parquet")[["resume_id", "text_anon"]],
                    on="resume_id").rename(columns={"text_anon": "resume_text",
                                                    "skills": "resume_skills"})
    jobs = pd.read_parquet(PROCESSED / "jobs_features.parquet")[
        ["job_id", "title", "skills", "years_required"]]
    jobs = jobs.merge(pd.read_parquet(INTERIM / "jobs_clean.parquet")[["job_id", "text"]],
                      on="job_id").rename(columns={"text": "job_text", "skills": "job_skills",
                                                   "title": "job_title"})
    df = pairs.merge(res.drop(columns="category"), on="resume_id").merge(jobs, on="job_id")
    df["label"] = df["weak_label"].astype(int)
    return df.reset_index(drop=True)


def load_human_pairs() -> pd.DataFrame:
    """External human-labelled pairs with train/val/test split and anonymized text."""
    tr = pd.read_csv(RAW / "pairs_resume_jd_fit_train.csv")
    te = pd.read_csv(RAW / "pairs_resume_jd_fit_test.csv")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, va_idx = next(gss.split(tr, groups=tr["job_description_text"]))
    tr["split"] = "train"
    tr.loc[tr.index[va_idx], "split"] = "val"
    te["split"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    texts = pd.unique(pd.concat([df["resume_text"], df["job_description_text"]]))
    cleaned = {t: anonymize(clean_text(t)) for t in texts}
    df["resume_text"] = df["resume_text"].map(cleaned)
    df["job_text"] = df["job_description_text"].map(cleaned)
    df["fit3"] = df["label"].map(FIT_LABELS)
    df["label"] = (df["fit3"] > 0).astype(int)
    df["job_title"] = df["job_text"].str.split().str[:8].str.join(" ")
    df["resume_id"] = "h_" + pd.factorize(df["resume_text"])[0].astype(str)
    df["job_id"] = "hj_" + pd.factorize(df["job_text"])[0].astype(str)
    return df.drop(columns=["job_description_text"])


def split_summary(df: pd.DataFrame) -> dict:
    return {s: {"pairs": int((df.split == s).sum()),
                "positive_share": round(float(df.loc[df.split == s, "label"].mean()), 3)}
            for s in ("train", "val", "test")}


if __name__ == "__main__":
    a, b = load_weak_pairs(), load_human_pairs()
    print("weak", len(a), split_summary(a))
    print("human", len(b), split_summary(b))
    print(b.groupby("split")["fit3"].value_counts().unstack())
    assert np.isfinite(a["label"]).all()
