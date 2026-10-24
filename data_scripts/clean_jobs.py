"""Clean LinkedIn (Kaggle) and Rozee.pk (scraped) postings into one schema.

Same steps as clean_resumes.py; step counts go to report/stats/clean_jobs.json.
Salary is never used as a model input; it is normalized to yearly values for
analysis only and missing values are left as NaN (not imputed).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (INTERIM, RAW, anonymize, clean_text, english_ratio, length_outliers,
                    near_duplicate_mask, save_stats, text_hash, truncate_words, word_count)

MIN_WORDS = 30
EN_THRESHOLD = 0.3

LINKEDIN_LEVEL = {
    "Internship": "intern", "Entry level": "entry", "Associate": "associate",
    "Mid-Senior level": "mid-senior", "Director": "director", "Executive": "executive",
}
ROZEE_LEVEL = {
    "Intern/Student": "intern", "Entry Level": "entry", "Mid-Level": "associate",
    "Experienced Professional": "mid-senior", "Manager": "mid-senior",
    "Department Head": "director", "Executive": "executive",
    "Gm / CEO / Country Head / President": "executive",
}
PERIOD_TO_YEAR = {"HOURLY": 2080, "WEEKLY": 52, "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1}


def load_linkedin() -> pd.DataFrame:
    j = pd.read_csv(RAW / "jobs_linkedin_arshkon.csv")
    mult = j["pay_period"].map(PERIOD_TO_YEAR)
    sal = j[["min_salary", "med_salary", "max_salary"]].mean(axis=1, skipna=True)
    desc = j["description"].fillna("")
    extra = j["skills_desc"].fillna("")
    return pd.DataFrame({
        "job_id": "li_" + j["job_id"].astype(str),
        "source": "linkedin_kaggle",
        "title": j["title"],
        "company": j["company_id"].astype("Int64").astype(str),
        "location": j["location"],
        "country": "US",
        "seniority_raw": j["formatted_experience_level"],
        "seniority": j["formatted_experience_level"].map(LINKEDIN_LEVEL),
        "work_type": j["formatted_work_type"],
        "remote": j["remote_allowed"].fillna(0).astype(int),
        "salary_yearly": sal * mult,
        "currency": j["currency"],
        "skills_listed": j["skills_desc"],
        "url": j["application_url"].fillna(j["job_posting_url"]),
        "date_posted": pd.to_datetime(j["listed_time"], unit="ms", utc=True),
        "valid_through": pd.to_datetime(j["expiry"], unit="ms", utc=True),
        "raw_text": np.where(extra.str.len() > 0, desc + "\n\nSkills: " + extra, desc),
    })


def load_rozee() -> pd.DataFrame:
    path = RAW / "jobs_rozee_pk.jsonl"
    if not path.exists():
        return pd.DataFrame()
    r = pd.read_json(path, lines=True, dtype={"job_id": str})
    skills = r.get("skills", pd.Series(index=r.index, dtype=object)).fillna("")
    city = r["city"].astype(object)
    city = city.mask(city.isna() | city.astype(str).str.fullmatch(r"\d+"), r.get("region"))
    return pd.DataFrame({
        "job_id": "rz_" + r["job_id"].astype(str),
        "source": "rozee_pk_scraped",
        "title": r["title"],
        "company": r.get("company"),
        "location": city,
        "country": "PK",
        "seniority_raw": r.get("career_level"),
        "seniority": r.get("career_level", pd.Series(index=r.index, dtype=object)).map(ROZEE_LEVEL),
        "work_type": r.get("job_type"),
        "remote": 0,
        "salary_yearly": np.nan,
        "currency": "PKR",
        "skills_listed": skills.replace("", np.nan),
        "experience_text": r.get("experience_text"),
        "gender_preference": r.get("gender_preference"),
        "url": r["url"],
        "date_posted": pd.to_datetime(r["date_posted"], utc=True, errors="coerce"),
        "valid_through": pd.to_datetime(r["valid_through"], utc=True, errors="coerce"),
        "raw_text": np.where(skills.str.len() > 0,
                             r["description_html"].fillna("") + "\n\nSkills: " + skills,
                             r["description_html"].fillna("")),
    })


def main() -> None:
    steps = {}
    df = pd.concat([load_linkedin(), load_rozee()], ignore_index=True)
    # newest closing date first, so deduplication keeps the posting that is still open
    df = df.sort_values("valid_through", ascending=False, na_position="last", kind="stable")
    df = df.reset_index(drop=True)
    steps["loaded"] = len(df)
    steps["loaded_by_source"] = df["source"].value_counts().to_dict()
    steps["missing_share_raw"] = df.isna().mean().round(3).to_dict()

    df["raw_words"] = df["raw_text"].fillna("").map(word_count)
    df[["job_id", "source", "raw_words"]].to_parquet(INTERIM / "jobs_raw_wordcount.parquet")

    df = df[df["raw_text"].fillna("").str.strip().astype(bool) & df["title"].notna()]
    steps["after_drop_empty"] = len(df)
    for col in ["location", "seniority", "work_type", "company"]:
        df[col] = df[col].fillna("unknown").replace({"<NA>": "unknown", "": "unknown"})

    df["text"] = df["raw_text"].map(clean_text)
    df["title"] = df["title"].map(clean_text)

    df["en_ratio"] = df["text"].map(english_ratio)
    steps["non_english_dropped"] = int((df["en_ratio"] < EN_THRESHOLD).sum())
    df = df[df["en_ratio"] >= EN_THRESHOLD]
    steps["after_language_filter"] = len(df)

    df["hash"] = df["text"].map(text_hash)
    steps["exact_duplicates"] = int(df["hash"].duplicated().sum())
    df = df.drop_duplicates("hash").reset_index(drop=True)
    steps["after_exact_dedup"] = len(df)

    dup, pairs = near_duplicate_mask(df["text"].tolist(), threshold=0.95)
    steps["near_duplicates"] = int(dup.sum())
    with open(INTERIM / "jobs_near_dup_pairs.json", "w") as f:
        json.dump([(df.job_id[i], df.job_id[j], round(s, 4)) for i, j, s in pairs[:5000]], f)
    df = df[~dup].reset_index(drop=True)
    steps["after_near_dedup"] = len(df)

    df["n_words"] = df["text"].map(word_count)
    bounds = length_outliers(df["n_words"])
    steps["length_bounds"] = bounds
    low = df["n_words"] < bounds["p_low"]
    high = df["n_words"] > bounds["p_high"]
    steps["flagged_short"] = int(low.sum())
    steps["flagged_long"] = int(high.sum())
    steps["iqr_flagged"] = int(((df.n_words < bounds["iqr_low"]) | (df.n_words > bounds["iqr_high"])).sum())
    junk = low & (df["n_words"] < MIN_WORDS)
    steps["dropped_junk"] = int(junk.sum())
    df = df[~junk].copy()
    cap = int(bounds["p_high"])
    df.loc[df["n_words"] > cap, "text"] = df.loc[df["n_words"] > cap, "text"].map(
        lambda t: truncate_words(t, cap))
    df["length_flag"] = np.select([df.n_words < bounds["p_low"], df.n_words > cap], ["short", "capped"], "ok")
    df["n_words"] = df["text"].map(word_count)
    steps["after_outliers"] = len(df)

    before = df["text"].copy()
    df["text"] = df["text"].map(anonymize)
    steps["postings_with_pii_masked"] = int((before != df["text"]).sum())

    keep = ["job_id", "source", "title", "company", "location", "country", "seniority",
            "seniority_raw", "work_type", "remote", "salary_yearly", "currency", "skills_listed",
            "text", "n_words", "length_flag", "en_ratio", "url", "date_posted", "valid_through"]
    keep += [c for c in ("experience_text", "gender_preference") if c in df.columns]
    out = df[keep]
    out.to_parquet(INTERIM / "jobs_clean.parquet", index=False)
    steps["final"] = len(out)
    steps["final_by_source"] = out["source"].value_counts().to_dict()
    steps["seniority"] = out["seniority"].value_counts().to_dict()
    steps["salary_available_share"] = round(float(out["salary_yearly"].notna().mean()), 3)
    save_stats("clean_jobs", steps)
    print(json.dumps(steps, indent=2))


if __name__ == "__main__":
    main()
