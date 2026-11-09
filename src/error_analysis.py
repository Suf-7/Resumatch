"""Phase 4 error analysis -- owner: Muaaz Fahad (implement, run, and commit this module yourself).

Goal: find where MatchNet fails on the TEST split and why. Read the predictions written by
`python src/evaluate.py` and write report/stats/error_analysis.json plus
report/figures/fig_error_analysis.{pdf,png}. The report fills its "Error Analysis" section
from the JSON keys below, so keep the names exactly as written.

Inputs (one row per pair, all splits; filter split == "test"):
    data/processed/predictions_weak.parquet    Task A (our weakly labelled pairs)
    data/processed/predictions_human.parquet   Task B (human-labelled fit pairs)
  Useful columns: label, matchnet_pred, "score_MatchNet", pair_type (Task A),
  resume_category / job_category (Task A), fit3 (Task B: 0 No, 1 Potential, 2 Good Fit),
  and the features tfidf_cos, skill_cov, skill_jacc, years_fit, title_match, structure, ats_score.

To implement:
  1. per_category(): Task A precision, recall, F1 and support of MatchNet per resume_category
     (sklearn.metrics.precision_recall_fscore_support). List the 3 worst and 3 best categories by F1.
  2. error_rates(): Task A error rate per pair_type (positive / hard_negative / random_negative);
     Task B error rate per fit3 class. Counts of false positives and false negatives.
  3. feature_profile(): mean of each feature for true positives vs. false negatives, and for
     true negatives vs. false positives (Task A). Which features separate the errors?
  4. examples(): the 10 most confident mistakes in Task A (highest score among false positives,
     lowest among false negatives) with resume_id, job_id, categories, score and skill_cov.
  5. plot(): bar chart of F1 per category (Task A) next to error rate per fit3 class (Task B).

Output JSON schema (report/stats/error_analysis.json):
{
  "weak":  {"per_category": {"<CATEGORY>": {"precision": float, "recall": float, "f1": float, "support": int}},
            "worst_categories": ["<CAT>", ...3], "best_categories": ["<CAT>", ...3],
            "error_rate_by_pair_type": {"positive": float, "hard_negative": float, "random_negative": float},
            "false_positives": int, "false_negatives": int,
            "feature_means": {"TP": {...}, "FN": {...}, "TN": {...}, "FP": {...}},
            "examples": [{...}, ...]},
  "human": {"error_rate_by_fit": {"No Fit": float, "Potential Fit": float, "Good Fit": float}}
}

Run: cd src && ../.venv/bin/python error_analysis.py
"""
from __future__ import annotations

import pandas as pd

from config import FIGURES, PROCESSED, save_json


def load_test(task: str) -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / f"predictions_{task}.parquet")
    return p[p["split"] == "test"].reset_index(drop=True)


def per_category(weak: pd.DataFrame) -> dict:
    raise NotImplementedError("Muaaz: step 1")


def error_rates(weak: pd.DataFrame, human: pd.DataFrame) -> dict:
    raise NotImplementedError("Muaaz: step 2")


def feature_profile(weak: pd.DataFrame) -> dict:
    raise NotImplementedError("Muaaz: step 3")


def examples(weak: pd.DataFrame, n: int = 10) -> list:
    raise NotImplementedError("Muaaz: step 4")


def plot(result: dict) -> None:
    raise NotImplementedError("Muaaz: step 5 (save to FIGURES / 'fig_error_analysis.pdf' and .png)")


def main() -> None:
    weak, human = load_test("weak"), load_test("human")
    raise NotImplementedError("Muaaz: combine steps 1-5 into the schema above, save_json('error_analysis', ...)")


if __name__ == "__main__":
    main()
