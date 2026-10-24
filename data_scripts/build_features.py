"""Feature engineering + leakage-free splits + weakly-labelled CV-job pairs.

Outputs (data/processed/):
  resumes_features.parquet, jobs_features.parquet   ids, split, encodings, numeric features
  tfidf_{resumes,jobs}.npz, tfidf_vectorizer.joblib  TF-IDF baseline representation
  skills_{resumes,jobs}.npz, skill_vocab.json        multi-hot ESCO-normalized skills
  scaler.joblib, label_encoder.joblib                fitted on the TRAIN split only
  pairs.parquet                                      positives / hard negatives / random negatives
  emb_{resumes,jobs}.npy                             384-d MiniLM embeddings (--embeddings)
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

from common import INTERIM, PROCESSED, SEED, save_stats
from skills import extract_skills, load_gazetteer

# --------------------------------------------------------------------------- #
# Job title -> resume category (ordered; first match wins)
# --------------------------------------------------------------------------- #
TITLE_RULES = [
    ("AVIATION", r"pilot|aviation|aircraft|avionics|flight (attendant|instructor)|airline|a&p mechanic"),
    ("CHEF", r"\bchef\b|\bcook\b|culinary|kitchen|sous|baker|pastry"),
    ("ADVOCATE", r"attorney|lawyer|paralegal|legal|counsel\b|advocate|litigation"),
    ("TEACHER", r"teacher|tutor|instructor|professor|educator|teaching|lecturer"),
    ("FITNESS", r"fitness|personal trainer|\bgym\b|yoga|pilates|strength coach"),
    ("AGRICULTURE", r"agricultur|\bfarm|agronom|\bcrop|ranch|greenhouse|horticultur|veterinar"),
    ("AUTOMOBILE", r"automotive|auto (technician|mechanic)|vehicle|\bcar\b|collision|diesel mechanic"),
    ("APPAREL", r"apparel|fashion|garment|textile|stylist|footwear"),
    ("HEALTHCARE", r"\bnurse|\brn\b|\blpn\b|\bcna\b|physician|medical|clinical|pharmac|therap|"
                   r"health ?care|dental|caregiver|patient|surgeon|radiolog|phlebotom|technologist|"
                   r"patholog|laboratory|sonograph"),
    ("ACCOUNTANT", r"accountant|accounting|bookkeep|auditor|\btax\b|accounts (payable|receivable)|"
                   r"controller|\bcpa\b|billing"),
    ("BANKING", r"\bbank|teller|loan officer|mortgage|underwriter|credit union"),
    ("FINANCE", r"financ|investment|fp&a|treasury|credit analyst|actuar|wealth|portfolio"),
    ("HR", r"human resources|\bhr\b|recruit|talent acquisition|payroll|benefits|people operations"),
    ("DESIGNER", r"designer|\bux\b|\bui\b|graphic|interior design|cad drafter"),
    ("INFORMATION-TECHNOLOGY", r"software|developer|programmer|data (scientist|engineer|analyst)|devops|"
                               r"cloud|cyber|network|system(s)? administrator|\bit\b|web|full.?stack|"
                               r"front.?end|back.?end|\bqa\b|machine learning|database|salesforce|sap\b|"
                               r"scrum master|product (owner|manager)"),
    ("CONSTRUCTION", r"construction|civil engineer|carpenter|electrician|plumb|superintendent|"
                     r"foreman|hvac|estimator|welder"),
    ("ENGINEERING", r"engineer"),
    ("PUBLIC-RELATIONS", r"public relations|\bpr\b|communications|media relations"),
    ("DIGITAL-MEDIA", r"digital|social media|content|\bseo\b|video|marketing"),
    ("BPO", r"call cent|customer (service|support|care)|contact cent|\bbpo\b"),
    ("BUSINESS-DEVELOPMENT", r"business development|partnership|growth|operations manager"),
    ("CONSULTANT", r"consult|advisor|business analyst|project manager|program manager|\bpmo\b"),
    ("SALES", r"sales|account (executive|manager)|retail|cashier|store|real estate agent|territory manager"),
    ("ARTS", r"artist|music|photograph|art director|animator|illustrat"),
]
_TITLE_RULES = [(c, re.compile(p, re.I)) for c, p in TITLE_RULES]


def job_category(title: str) -> str:
    for cat, rx in _TITLE_RULES:
        if rx.search(title or ""):
            return cat
    return "unknown"


# --------------------------------------------------------------------------- #
# Years of experience (regex)
# --------------------------------------------------------------------------- #
_YEARS = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-|to|–)?\s*(?:\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b(?:[^.\n]{0,40}?experience)?",
    re.I)
_RANGE = re.compile(
    r"(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*|\b\d{1,2}/)((?:19|20)\d{2})"
    r"\s*(?:-|–|to)\s*"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*|\d{1,2}/)?((?:19|20)\d{2}|current|present|now|till date)",
    re.I)
CURRENT_YEAR = date.today().year


def explicit_years(text: str) -> float:
    vals = [int(m.group(1)) for m in _YEARS.finditer(text) if "experience" in m.group(0).lower()]
    vals = [v for v in vals if 0 < v <= 40]
    return float(max(vals)) if vals else np.nan


def reference_year(texts) -> int:
    """'Current' in a CV means the year it was written; estimate it per source as the
    99th percentile of explicit end years (Kaggle resumes were collected ~2019)."""
    ends = [int(m.group(2)) for t in texts for m in _RANGE.finditer(t) if m.group(2).isdigit()]
    ends = [e for e in ends if e <= CURRENT_YEAR]
    return int(np.percentile(ends, 99)) if ends else CURRENT_YEAR


def career_span_years(text: str, ref_year: int = CURRENT_YEAR) -> float:
    spans = []
    for m in _RANGE.finditer(text):
        start = int(m.group(1))
        end = m.group(2).lower()
        end = ref_year if not end.isdigit() else int(end)
        if 1960 <= start <= end <= max(ref_year, start):
            spans.append((start, end))
    if not spans:
        return np.nan
    return float(min(45, max(e for _, e in spans) - min(s for s, _ in spans)))


def resume_years(text: str, ref_year: int = CURRENT_YEAR) -> float:
    y = explicit_years(text)
    return y if not np.isnan(y) else career_span_years(text, ref_year)


def required_years(text: str) -> float:
    vals = [int(m.group(1)) for m in _YEARS.finditer(text)
            if "experience" in m.group(0).lower() or "exp" in text[m.end(): m.end() + 30].lower()]
    vals = [v for v in vals if 0 < v <= 25]
    return float(min(vals)) if vals else np.nan


_LISTED = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d{1,2}\s*)?(years?|yrs?|months?)", re.I)


def listed_years(field) -> float:
    """Minimum years from a structured experience field such as '3 Years - 5 Years'."""
    m = _LISTED.search(str(field or ""))
    if not m:
        return np.nan
    return 0.0 if m.group(2).lower().startswith("month") else float(min(int(m.group(1)), 25))


# --------------------------------------------------------------------------- #
# TF-IDF normalization (baseline only; transformer inputs keep raw casing)
# --------------------------------------------------------------------------- #
def light_lemma(token: str) -> str:
    if len(token) <= 4:
        return token
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("s", "")):
        if token.endswith(suf) and len(token) - len(suf) >= 3 and not token.endswith("ss"):
            return token[: -len(suf)] + rep
    return token


def tfidf_normalize(text: str) -> str:
    text = re.sub(r"\[(EMAIL|PHONE|URL|ADDRESS|NAME)\]", " ", text.lower())
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    return " ".join(light_lemma(t) for t in text.split())


# --------------------------------------------------------------------------- #
def stratified_split(ids: pd.Series, strata: pd.Series, seed: int = SEED) -> pd.Series:
    strata = strata.where(strata.map(strata.value_counts()) >= 10, "rare")
    train, rest = train_test_split(ids, test_size=0.30, stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata.loc[rest.index], random_state=seed)
    split = pd.Series("train", index=ids.index)
    split.loc[val.index] = "val"
    split.loc[test.index] = "test"
    return split


def build_pairs(res: pd.DataFrame, jobs: pd.DataFrame, R, J, k_pos=3, k_hard=2, k_rand=1) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for split in ("train", "val", "test"):
        ri = np.where(res["split"].values == split)[0]
        jmask = (jobs["split"].values == split) & (jobs["job_category"].values != "unknown")
        ji = np.where(jmask)[0]
        if len(ri) == 0 or len(ji) == 0:
            continue
        sims = (R[ri] @ J[ji].T).toarray()
        jcats = jobs["job_category"].values[ji]
        for row, r_idx in enumerate(ri):
            cat = res["category"].values[r_idx]
            s = sims[row]
            same = np.where(jcats == cat)[0]
            diff = np.where(jcats != cat)[0]
            if len(same) == 0:
                continue
            pos = same[np.argsort(-s[same])[:k_pos]]
            hard = diff[np.argsort(-s[diff])[:k_hard]]
            rand = rng.choice(np.setdiff1d(diff, hard), size=k_rand, replace=False)
            for kind, idxs, lab in (("positive", pos, 1), ("hard_negative", hard, 0),
                                    ("random_negative", rand, 0)):
                for j in idxs:
                    rows.append({"resume_id": res["resume_id"].values[r_idx],
                                 "job_id": jobs["job_id"].values[ji[j]], "split": split,
                                 "pair_type": kind, "weak_label": lab,
                                 "tfidf_cosine": round(float(s[j]), 4),
                                 "resume_category": cat, "job_category": jcats[j]})
    return pd.DataFrame(rows)


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true", help="compute 384-d MiniLM embeddings")
    args = ap.parse_args()
    stats: dict = {}

    res = pd.read_parquet(INTERIM / "resumes_clean.parquet")
    jobs = pd.read_parquet(INTERIM / "jobs_clean.parquet")

    gaz = load_gazetteer()
    res["skills"] = res["text"].map(lambda t: extract_skills(t, gaz))
    jobs["skills"] = jobs["text"].map(lambda t: extract_skills(t, gaz))
    res["n_skills"] = res["skills"].str.len()
    jobs["n_skills"] = jobs["skills"].str.len()
    ref_years = {src: reference_year(g["text"]) for src, g in res.groupby("source")}
    stats["resume_reference_year"] = ref_years
    res["years_exp"] = [resume_years(t, ref_years[s]) for t, s in zip(res["text"], res["source"])]
    jobs["years_required"] = jobs["text"].map(required_years)
    if "experience_text" in jobs:
        jobs["years_required"] = jobs["years_required"].fillna(jobs["experience_text"].map(listed_years))
    jobs["job_category"] = jobs["title"].map(job_category)
    stats["years_exp_coverage_resumes"] = round(float(res["years_exp"].notna().mean()), 3)
    stats["years_required_coverage_jobs"] = round(float(jobs["years_required"].notna().mean()), 3)
    stats["job_category_coverage"] = round(float((jobs["job_category"] != "unknown").mean()), 3)
    stats["job_category_counts"] = jobs["job_category"].value_counts().to_dict()

    # splits by entity (not by pair) to avoid leakage
    res["split"] = stratified_split(res["resume_id"], res["category"])
    jobs["split"] = stratified_split(jobs["job_id"], jobs["job_category"])
    stats["resume_splits"] = res["split"].value_counts().to_dict()
    stats["job_splits"] = jobs["split"].value_counts().to_dict()

    # label encoding (shared label space; 'unknown' for unmapped jobs)
    le = LabelEncoder().fit(sorted(set(res["category"]) | set(jobs["job_category"])))
    res["category_id"] = le.transform(res["category"])
    jobs["category_id"] = le.transform(jobs["job_category"])
    joblib.dump(le, PROCESSED / "label_encoder.joblib")

    # numeric features: log1p for heavy-tailed counts, impute median (train) for years, scale (train)
    for df, ycol in ((res, "years_exp"), (jobs, "years_required")):
        df["log_n_words"] = np.log1p(df["n_words"])
        df["log_n_skills"] = np.log1p(df["n_skills"])
        df[f"{ycol}_missing"] = df[ycol].isna().astype(int)
        med = df.loc[df["split"] == "train", ycol].median()
        df[f"{ycol}_filled"] = df[ycol].fillna(med)
    num_r = ["log_n_words", "log_n_skills", "years_exp_filled"]
    num_j = ["log_n_words", "log_n_skills", "years_required_filled"]
    sc_r = StandardScaler().fit(res.loc[res.split == "train", num_r])
    sc_j = StandardScaler().fit(jobs.loc[jobs.split == "train", num_j])
    res[[c + "_z" for c in num_r]] = sc_r.transform(res[num_r])
    jobs[[c + "_z" for c in num_j]] = sc_j.transform(jobs[num_j])
    joblib.dump({"resumes": sc_r, "jobs": sc_j}, PROCESSED / "scaler.joblib")

    # multi-hot skills (vocabulary = skills seen in >= 5 training documents)
    train_skills = pd.concat([res.loc[res.split == "train", "skills"],
                              jobs.loc[jobs.split == "train", "skills"]]).explode().value_counts()
    vocab = sorted(train_skills[train_skills >= 5].index)
    mlb = MultiLabelBinarizer(classes=vocab, sparse_output=True)
    mlb.fit([vocab])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sparse.save_npz(PROCESSED / "skills_resumes.npz", mlb.transform(res["skills"]).tocsr())
        sparse.save_npz(PROCESSED / "skills_jobs.npz", mlb.transform(jobs["skills"]).tocsr())
    with open(PROCESSED / "skill_vocab.json", "w") as f:
        json.dump(vocab, f)
    stats["skill_vocab_size"] = len(vocab)
    stats["mean_skills_resume"] = round(float(res["n_skills"].mean()), 2)
    stats["mean_skills_job"] = round(float(jobs["n_skills"].mean()), 2)

    # TF-IDF baseline (fit on training resumes + jobs only)
    res["tfidf_text"] = res["text"].map(tfidf_normalize)
    jobs["tfidf_text"] = jobs["text"].map(tfidf_normalize)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, max_features=50_000,
                          sublinear_tf=True, stop_words="english", dtype=np.float32)
    vec.fit(pd.concat([res.loc[res.split == "train", "tfidf_text"],
                       jobs.loc[jobs.split == "train", "tfidf_text"]]))
    R = vec.transform(res["tfidf_text"])
    J = vec.transform(jobs["tfidf_text"])
    sparse.save_npz(PROCESSED / "tfidf_resumes.npz", R)
    sparse.save_npz(PROCESSED / "tfidf_jobs.npz", J)
    joblib.dump(vec, PROCESSED / "tfidf_vectorizer.joblib")
    stats["tfidf_vocab_size"] = len(vec.vocabulary_)

    pairs = build_pairs(res, jobs, R, J)
    pairs.to_parquet(PROCESSED / "pairs.parquet", index=False)
    stats["pairs"] = len(pairs)
    stats["pairs_by_type"] = pairs["pair_type"].value_counts().to_dict()
    stats["pairs_by_split"] = pairs["split"].value_counts().to_dict()
    stats["pair_label_balance"] = round(float(pairs["weak_label"].mean()), 3)
    stats["mean_cosine_by_type"] = pairs.groupby("pair_type")["tfidf_cosine"].mean().round(4).to_dict()

    if args.embeddings:
        try:
            np.save(PROCESSED / "emb_resumes.npy", embed(res["text"].tolist()))
            np.save(PROCESSED / "emb_jobs.npy", embed((jobs["title"] + ". " + jobs["text"]).tolist()))
            stats["embeddings"] = "all-MiniLM-L6-v2, 384-d, L2-normalized"
        except ImportError:
            print("sentence-transformers not installed; skipping embeddings")

    drop = ["text", "text_anon", "tfidf_text"]
    res.drop(columns=[c for c in drop if c in res], errors="ignore").to_parquet(
        PROCESSED / "resumes_features.parquet", index=False)
    jobs.drop(columns=[c for c in drop if c in jobs], errors="ignore").to_parquet(
        PROCESSED / "jobs_features.parquet", index=False)
    save_stats("features", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
