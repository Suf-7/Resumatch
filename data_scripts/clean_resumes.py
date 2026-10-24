"""Clean and anonymize resumes from all sources into one table.

Steps (each step's row count is written to report/stats/clean_resumes.json):
  1. load + harmonize categories      5. near-duplicates (TF-IDF cosine > 0.95)
  2. drop empty text                  6. length outliers (1st/99th pct, IQR cross-check)
  3. noise removal                    7. PII anonymization
  4. language filter + exact dups     8. save data/interim/resumes_clean.parquet
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (INTERIM, RAW, anonymize, clean_text, english_ratio, length_outliers,
                    near_duplicate_mask, save_stats, text_hash, truncate_words, word_count)

MIN_WORDS = 50  # below this a "resume" is junk (header only, parse failure)
EN_THRESHOLD = 0.3

# Fine-grained categories of the second Kaggle set -> the 24-category scheme of the main set.
CATEGORY_MAP = {
    "Java Developer": "INFORMATION-TECHNOLOGY", "Testing": "INFORMATION-TECHNOLOGY",
    "DevOps Engineer": "INFORMATION-TECHNOLOGY", "Python Developer": "INFORMATION-TECHNOLOGY",
    "Web Designing": "INFORMATION-TECHNOLOGY", "Hadoop": "INFORMATION-TECHNOLOGY",
    "Blockchain": "INFORMATION-TECHNOLOGY", "ETL Developer": "INFORMATION-TECHNOLOGY",
    "Data Science": "INFORMATION-TECHNOLOGY", "Database": "INFORMATION-TECHNOLOGY",
    "DotNet Developer": "INFORMATION-TECHNOLOGY", "Automation Testing": "INFORMATION-TECHNOLOGY",
    "Network Security Engineer": "INFORMATION-TECHNOLOGY", "SAP Developer": "INFORMATION-TECHNOLOGY",
    "HR": "HR", "Sales": "SALES", "Arts": "ARTS", "Advocate": "ADVOCATE",
    "Health and fitness": "FITNESS", "Mechanical Engineer": "ENGINEERING",
    "Electrical Engineering": "ENGINEERING", "Civil Engineer": "CONSTRUCTION",
    "Operations Manager": "BUSINESS-DEVELOPMENT", "PMO": "CONSULTANT",
    "Business Analyst": "CONSULTANT",
}


def load() -> pd.DataFrame:
    a = pd.read_csv(RAW / "resumes_kaggle_snehaanbhawal.csv")
    a = pd.DataFrame({
        "resume_id": "kr_" + a["ID"].astype(str),
        "source": "kaggle_resume_dataset",
        "category": a["Category"],
        "category_fine": a["Category"],
        "raw_text": a["Resume_str"],
        "synthetic": False,
    })
    b = pd.read_csv(RAW / "resumes_kaggle_gauravduttakiit.csv")
    b = pd.DataFrame({
        "resume_id": "ku_" + b.index.astype(str),
        "source": "kaggle_updated_resume_dataset",
        "category": b["Category"].map(CATEGORY_MAP),
        "category_fine": b["Category"],
        "raw_text": b["Resume"],
        "synthetic": False,
    })
    frames = [a, b]
    syn = RAW / "resumes_synthetic.jsonl"
    if syn.exists() and syn.stat().st_size > 0:
        s = pd.read_json(syn, lines=True)
        frames.append(pd.DataFrame({
            "resume_id": "sy_" + s.index.astype(str),
            "source": "synthetic_llm",
            "category": s["category"],
            "category_fine": s["role"],
            "raw_text": s["resume_text"],
            "synthetic": True,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    steps = {}
    df = load()
    steps["loaded"] = len(df)
    steps["loaded_by_source"] = df["source"].value_counts().to_dict()
    df["category"] = df["category"].fillna("unknown")

    df["raw_words"] = df["raw_text"].fillna("").map(word_count)
    df[["resume_id", "source", "raw_words"]].to_parquet(INTERIM / "resumes_raw_wordcount.parquet")

    df = df[df["raw_text"].fillna("").str.strip().astype(bool)]
    steps["after_drop_empty"] = len(df)

    df["text"] = df["raw_text"].map(clean_text)
    mojibake = df["raw_text"].str.contains("Ã|â€", regex=True).sum()
    steps["mojibake_repaired"] = int(mojibake)

    df["en_ratio"] = df["text"].map(english_ratio)
    df = df[df["en_ratio"] >= EN_THRESHOLD]
    steps["after_language_filter"] = len(df)

    df["hash"] = df["text"].map(text_hash)
    steps["exact_duplicates_by_source"] = df[df["hash"].duplicated()]["source"].value_counts().to_dict()
    df = df.drop_duplicates("hash").reset_index(drop=True)
    steps["after_exact_dedup"] = len(df)

    dup, pairs = near_duplicate_mask(df["text"].tolist(), threshold=0.95)
    steps["near_duplicates"] = int(dup.sum())
    with open(INTERIM / "resumes_near_dup_pairs.json", "w") as f:
        json.dump([(df.resume_id[i], df.resume_id[j], round(s, 4)) for i, j, s in pairs], f)
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
    steps["pii_tokens_inserted"] = int(sum(
        a.count(tok) for a in df["text"] for tok in ("[EMAIL]", "[PHONE]", "[URL]", "[ADDRESS]")))
    steps["resumes_with_pii_masked"] = int((before != df["text"]).sum())
    df["text_anon"] = df["text"]

    out = df[["resume_id", "source", "category", "category_fine", "synthetic", "text",
              "text_anon", "n_words", "length_flag", "en_ratio"]]
    out.to_parquet(INTERIM / "resumes_clean.parquet", index=False)
    steps["final"] = len(out)
    steps["final_by_source"] = out["source"].value_counts().to_dict()
    steps["categories"] = int(out["category"].nunique())
    save_stats("clean_resumes", steps)
    print(json.dumps(steps, indent=2))


if __name__ == "__main__":
    main()
