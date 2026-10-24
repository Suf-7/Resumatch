"""Human-labelled gold set: sample ~200 CV-job pairs and score annotator agreement.

The gold set is drawn only from the TEST split (resumes and postings that never
appear in training) and is never used for training.

    python data_scripts/sample_gold_set.py sample --n 200
        -> data/gold/gold_set_to_label.csv  (each annotator fills their own copy)
    python data_scripts/sample_gold_set.py score a1.csv a2.csv
        -> Cohen's kappa (unweighted + linear-weighted), confusion matrix,
           and data/gold/gold_set.csv with adjudication-needed flags
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import DATA, INTERIM, PROCESSED, SEED, save_stats

GOLD = DATA / "gold"
GOLD.mkdir(exist_ok=True)
LABELS = ["poor", "partial", "good"]
GUIDE = (
    "good = meets most required skills and the seniority level; "
    "partial = relevant background but clear skill/seniority gaps; "
    "poor = different role or most required skills missing. "
    "Ignore name, gender, age, university prestige."
)


def sample(n: int) -> None:
    pairs = pd.read_parquet(PROCESSED / "pairs.parquet")
    test = pairs[pairs["split"] == "test"]
    per_type = {"positive": n // 2, "hard_negative": n * 3 // 10, "random_negative": n // 5}
    picked = pd.concat([test[test.pair_type == t].sample(min(k, (test.pair_type == t).sum()),
                                                         random_state=SEED)
                        for t, k in per_type.items()])
    picked = picked.sample(frac=1, random_state=SEED).reset_index(drop=True)

    res = pd.read_parquet(INTERIM / "resumes_clean.parquet").set_index("resume_id")
    jobs = pd.read_parquet(INTERIM / "jobs_clean.parquet").set_index("job_id")
    out = pd.DataFrame({
        "pair_id": [f"g{i:03d}" for i in range(len(picked))],
        "resume_id": picked["resume_id"],
        "job_id": picked["job_id"],
        "job_title": picked["job_id"].map(jobs["title"]),
        "job_text": picked["job_id"].map(jobs["text"]).str[:4000],
        "resume_text": picked["resume_id"].map(res["text_anon"]).str[:4000],
        "label": "",
        "notes": "",
    })
    path = GOLD / "gold_set_to_label.csv"
    out.to_csv(path, index=False)
    # pair_type is hidden from annotators so it cannot bias their labels
    picked[["resume_id", "job_id", "pair_type"]].assign(pair_id=out["pair_id"]).to_csv(
        GOLD / "gold_set_key.csv", index=False)
    (GOLD / "GUIDELINES.txt").write_text(f"Allowed labels: {LABELS}\n{GUIDE}\n")
    print(f"wrote {len(out)} pairs to {path}")


def score(a1: str, a2: str) -> None:
    x = pd.read_csv(a1).set_index("pair_id")["label"].str.strip().str.lower()
    y = pd.read_csv(a2).set_index("pair_id")["label"].str.strip().str.lower()
    both = pd.concat([x, y], axis=1, keys=["a1", "a2"]).dropna()
    both = both[both.a1.isin(LABELS) & both.a2.isin(LABELS)]
    order = {l: i for i, l in enumerate(LABELS)}
    k = cohen_kappa_score(both.a1, both.a2)
    kw = cohen_kappa_score(both.a1.map(order), both.a2.map(order), weights="linear")
    cm = confusion_matrix(both.a1, both.a2, labels=LABELS)
    result = {"n": len(both), "cohen_kappa": round(k, 3), "weighted_kappa_linear": round(kw, 3),
              "raw_agreement": round(float((both.a1 == both.a2).mean()), 3),
              "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()}}
    both["needs_adjudication"] = both.a1 != both.a2
    both.to_csv(GOLD / "gold_set.csv")
    save_stats("gold_agreement", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=200)
    c = sub.add_parser("score")
    c.add_argument("a1")
    c.add_argument("a2")
    args = ap.parse_args()
    sample(args.n) if args.cmd == "sample" else score(args.a1, args.a2)


if __name__ == "__main__":
    main()
