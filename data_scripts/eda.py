"""Exploratory data analysis on the cleaned data. Figures -> report/figures/*.pdf|png."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import FIGURES, INTERIM, PROCESSED, save_stats

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


ACRONYMS = {"INFORMATION-TECHNOLOGY": "IT", "HR": "HR", "BPO": "BPO", "PUBLIC-RELATIONS": "PR",
            "BUSINESS-DEVELOPMENT": "Business Dev."}


def short(cat: str) -> str:
    return ACRONYMS.get(cat, cat.replace("-", " ").title())


def fig_categories(res: pd.DataFrame, jobs: pd.DataFrame, stats: dict) -> None:
    rc = res["category"].value_counts()
    jc = jobs.loc[jobs.job_category != "unknown", "job_category"].value_counts().reindex(rc.index).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    y = np.arange(len(rc))
    axes[0].barh(y, rc.values, color=BLUE)
    axes[0].set_yticks(y, [short(c) for c in rc.index])
    axes[0].invert_yaxis()
    axes[0].set_title(f"Resumes by category (n={len(res):,})")
    axes[0].set_xlabel("resumes")
    axes[0].axvline(rc.mean(), color="k", ls="--", lw=0.8)
    axes[1].barh(y, jc.values, color=ORANGE)
    axes[1].set_title("Postings mapped to the same categories")
    axes[1].set_xlabel("postings (log scale)")
    axes[1].set_xscale("log")
    save(fig, "fig_category_distribution")
    stats["resume_imbalance_ratio"] = round(float(rc.max() / rc.min()), 2)
    stats["resume_category_min"] = {rc.idxmin(): int(rc.min())}
    stats["resume_category_max"] = {rc.idxmax(): int(rc.max())}


def fig_lengths(res, jobs, stats) -> None:
    raw_r = pd.read_parquet(INTERIM / "resumes_raw_wordcount.parquet")["raw_words"]
    raw_j = pd.read_parquet(INTERIM / "jobs_raw_wordcount.parquet")["raw_words"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, raw, clean, title in ((axes[0], raw_r, res.n_words, "Resumes"),
                                  (axes[1], raw_j, jobs.n_words, "Job postings")):
        hi = np.percentile(raw, 99.8)
        bins = np.linspace(0, hi, 60)
        ax.hist(raw.clip(upper=hi), bins=bins, color=GREY, alpha=0.7, label=f"raw (n={len(raw):,})")
        ax.hist(clean, bins=bins, color=BLUE if title == "Resumes" else ORANGE, alpha=0.7,
                label=f"cleaned (n={len(clean):,})")
        ax.axvline(np.median(clean), color="k", ls="--", lw=0.8)
        ax.set_title(f"{title}: words per document (median {int(np.median(clean))})")
        ax.set_xlabel("words")
        ax.legend()
    save(fig, "fig_length_distribution")
    stats["median_words_resume"] = int(res.n_words.median())
    stats["median_words_job"] = int(jobs.n_words.median())


def fig_skills(res, jobs, stats) -> None:
    def share(df):
        return df["skills"].explode().value_counts() / len(df)

    sj, sr = share(jobs), share(res)
    top = sj.head(20).index
    d = pd.DataFrame({"postings": sj.reindex(top), "resumes": sr.reindex(top).fillna(0)})
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d.postings * 100, height=0.4, color=ORANGE, label="postings")
    ax.barh(y + 0.2, d.resumes * 100, height=0.4, color=BLUE, label="resumes")
    ax.set_yticks(y, [s.replace(" (computer programming)", "") for s in d.index])
    ax.invert_yaxis()
    ax.set_xlabel("% of documents mentioning the skill")
    ax.set_title("Top 20 skills in postings vs. resumes (ESCO-normalized)")
    ax.legend()
    save(fig, "fig_top_skills")

    vj = set(jobs["skills"].explode().dropna())
    vr = set(res["skills"].explode().dropna())
    stats["skills_unique_jobs"] = len(vj)
    stats["skills_unique_resumes"] = len(vr)
    stats["skills_overlap"] = len(vj & vr)
    stats["skills_jaccard"] = round(len(vj & vr) / len(vj | vr), 3)
    top100 = set(sj.head(100).index)
    stats["top100_job_skills_missing_in_most_resumes"] = int(sum(sr.get(s, 0) < 0.01 for s in top100))


def fig_correlation(res, jobs, stats) -> None:
    r = res[["n_words", "n_skills", "years_exp"]].rename(
        columns={"n_words": "words", "n_skills": "skills", "years_exp": "years exp."})
    j = jobs[["n_words", "n_skills", "years_required", "salary_yearly"]].rename(
        columns={"n_words": "words", "n_skills": "skills", "years_required": "years req.",
                 "salary_yearly": "salary/yr"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, df, title in ((axes[0], r, "Resumes"), (axes[1], j, "Job postings")):
        c = df.corr(method="spearman")
        sns.heatmap(c, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="RdBu_r", ax=ax, square=True,
                    cbar=ax is axes[1])
        ax.set_title(f"{title} (Spearman)")
    save(fig, "fig_correlation")
    stats["corr_resume"] = r.corr(method="spearman").round(3).to_dict()
    stats["corr_jobs"] = j.corr(method="spearman").round(3).to_dict()


def fig_missing_seniority(jobs, stats) -> None:
    raw = pd.read_csv(
        __import__("common").RAW / "jobs_linkedin_arshkon.csv",
        usecols=["title", "description", "location", "formatted_experience_level", "skills_desc",
                 "min_salary", "max_salary", "med_salary", "remote_allowed", "applies", "views"])
    miss = raw.isna().mean().sort_values() * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(miss.index, miss.values, color=[BLUE if v < 30 else ORANGE for v in miss.values])
    axes[0].set_xlabel("% missing (raw LinkedIn postings)")
    axes[0].set_title("Missing values before cleaning")
    order = ["intern", "entry", "associate", "mid-senior", "director", "executive", "unknown"]
    sen = jobs["seniority"].value_counts().reindex(order).fillna(0)
    axes[1].bar(sen.index, sen.values, color=[ORANGE if s == "unknown" else BLUE for s in sen.index])
    axes[1].set_title("Seniority after cleaning ('unknown' = filled)")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "fig_missing_seniority")
    stats["missing_pct_raw_linkedin"] = miss.round(1).to_dict()


def fig_pairs(stats) -> None:
    from sklearn.metrics import roc_auc_score

    p = pd.read_parquet(PROCESSED / "pairs.parquet")
    cols = [("tfidf_cosine", "TF-IDF cosine")]
    if (PROCESSED / "emb_resumes.npy").exists():
        er, ej = np.load(PROCESSED / "emb_resumes.npy"), np.load(PROCESSED / "emb_jobs.npy")
        rid = pd.read_parquet(PROCESSED / "resumes_features.parquet", columns=["resume_id"])["resume_id"]
        jid = pd.read_parquet(PROCESSED / "jobs_features.parquet", columns=["job_id"])["job_id"]
        ri = pd.Series(range(len(rid)), index=rid)
        ji = pd.Series(range(len(jid)), index=jid)
        p["emb_cosine"] = np.einsum("ij,ij->i", er[ri[p.resume_id].values], ej[ji[p.job_id].values])
        cols.append(("emb_cosine", "MiniLM cosine (not fine-tuned)"))

    fig, axes = plt.subplots(1, len(cols), figsize=(5.4 * len(cols), 3.6), squeeze=False)
    hard = p[p.pair_type != "random_negative"]
    for ax, (col, label) in zip(axes[0], cols):
        for kind, color in (("positive", BLUE), ("hard_negative", ORANGE), ("random_negative", GREY)):
            ax.hist(p.loc[p.pair_type == kind, col], bins=40, alpha=0.65, color=color,
                    label=kind.replace("_", " "))
        auc = roc_auc_score(hard.weak_label, hard[col])
        stats[f"auc_pos_vs_hard_{col}"] = round(float(auc), 3)
        stats[f"mean_{col}_by_type"] = p.groupby("pair_type")[col].mean().round(3).to_dict()
        ax.set_xlabel(label)
        ax.set_title(f"{label.split(' (')[0]}: positive vs. hard negative AUC = {auc:.2f}")
        ax.legend()
    axes[0][0].set_ylabel("pairs")
    save(fig, "fig_pairs")


def main() -> None:
    res = pd.read_parquet(PROCESSED / "resumes_features.parquet")
    jobs = pd.read_parquet(PROCESSED / "jobs_features.parquet")
    stats: dict = {}
    fig_categories(res, jobs, stats)
    fig_lengths(res, jobs, stats)
    fig_skills(res, jobs, stats)
    fig_correlation(res, jobs, stats)
    fig_missing_seniority(jobs, stats)
    fig_pairs(stats)
    if "source" in jobs and (jobs.source == "rozee_pk_scraped").any():
        pk = jobs[jobs.source == "rozee_pk_scraped"]
        stats["rozee_top_skills"] = pk["skills"].explode().value_counts().head(15).to_dict()
        stats["rozee_cities"] = pk["location"].value_counts().head(8).to_dict()
        if "gender_preference" in pk:
            stats["rozee_gender_preference"] = pk["gender_preference"].value_counts().to_dict()
    save_stats("eda", stats)
    print(json.dumps(stats, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
