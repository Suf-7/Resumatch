"""Top-3 recommendation quality on the test split (the product flow).

For every test resume, rank all test-split postings with a known occupation and check the
top 3: precision@3 = share of the three that belong to the resume's occupation.
Rankers: TF-IDF cosine, pretrained MiniLM, fine-tuned bi-encoder, and fine-tuned bi-encoder
shortlist (top 20) re-ranked by MatchNet.

Usage: python src/retrieval_eval.py
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import BASE_ENCODER, INTERIM, MODELS, PROCESSED, save_json
from build_features import tfidf_normalize
from models import bi_encoder
from predict import MatchScorer

SHORTLIST = 20


def main() -> None:
    res = pd.read_parquet(PROCESSED / "resumes_features.parquet")
    res = res[res.split == "test"][["resume_id", "category"]].merge(
        pd.read_parquet(INTERIM / "resumes_clean.parquet")[["resume_id", "text_anon"]], on="resume_id")
    jobs = pd.read_parquet(PROCESSED / "jobs_features.parquet")
    jobs = jobs[(jobs.split == "test") & (jobs.job_category != "unknown")][["job_id", "title", "job_category"]]
    jobs = jobs.merge(pd.read_parquet(INTERIM / "jobs_clean.parquet")[["job_id", "text"]], on="job_id")
    r_text, j_text = res["text_anon"].tolist(), jobs["text"].tolist()
    r_cat, j_cat = res["category"].to_numpy(), jobs["job_category"].to_numpy()

    def p_at_3(sim: np.ndarray) -> np.ndarray:
        top = np.argsort(-sim, axis=1)[:, :3]
        return (j_cat[top] == r_cat[:, None]).mean(axis=1)

    out = {}
    vec = joblib.load(PROCESSED / "tfidf_vectorizer.joblib")
    R = vec.transform([tfidf_normalize(t) for t in r_text])
    J = vec.transform([tfidf_normalize(t) for t in j_text])
    out["TF-IDF cosine"] = p_at_3((R @ J.T).toarray())

    for name, path in (("MiniLM (pretrained)", BASE_ENCODER), ("Bi-encoder (fine-tuned)", MODELS / "bi_encoder")):
        enc = bi_encoder.load_encoder(path)
        sim = bi_encoder.encode(enc, r_text) @ bi_encoder.encode(enc, j_text).T
        out[name] = p_at_3(sim)
    ft_sim = sim

    scorer = MatchScorer("weak")
    short = np.argsort(-ft_sim, axis=1)[:, :SHORTLIST]
    flat_r = np.repeat(np.arange(len(r_text)), SHORTLIST)
    flat_j = short.ravel()
    s = scorer.score([r_text[i] for i in flat_r], [j_text[j] for j in flat_j],
                     jobs["title"].to_numpy()[flat_j]).reshape(len(r_text), SHORTLIST)
    rerank = np.take_along_axis(short, np.argsort(-s, axis=1), axis=1)[:, :3]
    out["Bi-encoder + MatchNet re-rank"] = (j_cat[rerank] == r_cat[:, None]).mean(axis=1)

    by_cat = pd.DataFrame({k: v for k, v in out.items()}).assign(category=r_cat)
    stats = {
        "test_resumes": len(r_text), "test_postings": len(j_text), "shortlist": SHORTLIST,
        "precision_at_3": {k: round(float(v.mean()), 4) for k, v in out.items()},
        "precision_at_3_by_category": by_cat.groupby("category").mean().round(3).to_dict(orient="index"),
        "random_baseline": round(float(np.mean([np.mean(j_cat == c) for c in r_cat])), 4),
    }
    save_json("retrieval", stats)
    print(stats["precision_at_3"], "random:", stats["random_baseline"])


if __name__ == "__main__":
    main()
