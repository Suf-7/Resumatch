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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config import FIGURES, PROCESSED, save_json
from features import SCALAR_FEATURES

FIT = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}
SHORT = {"INFORMATION-TECHNOLOGY": "IT", "BUSINESS-DEVELOPMENT": "Business Dev.",
         "PUBLIC-RELATIONS": "PR", "DIGITAL-MEDIA": "Digital Media", "BPO": "BPO", "HR": "HR"}
BLUE, ORANGE, GREY = "#2B6CB0", "#DD6B20", "#718096"


def load_test(task: str) -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / f"predictions_{task}.parquet")
    return p[p["split"] == "test"].reset_index(drop=True)


def _cell(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f), 4), "support": int(len(y_true))}


def per_category(weak: pd.DataFrame) -> dict:
    rows = {}
    for cat, g in weak.groupby("resume_category"):
        rows[str(cat)] = _cell(g["label"], g["matchnet_pred"])
    ranked = sorted(rows, key=lambda c: (rows[c]["f1"], rows[c]["support"]))
    return {"per_category": rows, "worst_categories": ranked[:3], "best_categories": ranked[-3:][::-1]}


def error_rates(weak: pd.DataFrame, human: pd.DataFrame) -> dict:
    wrong = weak["label"] != weak["matchnet_pred"]
    by_type = {t: round(float(wrong[weak["pair_type"] == t].mean()), 4)
               for t in ("positive", "hard_negative", "random_negative")}
    fp = int(((weak["label"] == 0) & (weak["matchnet_pred"] == 1)).sum())
    fn = int(((weak["label"] == 1) & (weak["matchnet_pred"] == 0)).sum())
    h_wrong = human["label"] != human["matchnet_pred"]
    by_fit = {FIT[k]: round(float(h_wrong[human["fit3"] == k].mean()), 4) for k in (0, 1, 2)}
    return {"error_rate_by_pair_type": by_type, "false_positives": fp, "false_negatives": fn,
            "error_rate_by_fit": by_fit}


def feature_profile(weak: pd.DataFrame) -> dict:
    tp = (weak["label"] == 1) & (weak["matchnet_pred"] == 1)
    fn = (weak["label"] == 1) & (weak["matchnet_pred"] == 0)
    tn = (weak["label"] == 0) & (weak["matchnet_pred"] == 0)
    fp = (weak["label"] == 0) & (weak["matchnet_pred"] == 1)
    means = {}
    for name, mask in (("TP", tp), ("FN", fn), ("TN", tn), ("FP", fp)):
        means[name] = {f: round(float(weak.loc[mask, f].mean()), 4) for f in SCALAR_FEATURES}
    return means


def examples(weak: pd.DataFrame, n: int = 10) -> list:
    fp = weak[(weak["label"] == 0) & (weak["matchnet_pred"] == 1)].nlargest(n // 2, "score_MatchNet")
    fn = weak[(weak["label"] == 1) & (weak["matchnet_pred"] == 0)].nsmallest(n - n // 2, "score_MatchNet")
    rows = []
    for kind, block in (("FP", fp), ("FN", fn)):
        for _, r in block.iterrows():
            rows.append({"kind": kind, "resume_id": r["resume_id"], "job_id": r["job_id"],
                         "resume_category": r["resume_category"], "job_category": r["job_category"],
                         "score": round(float(r["score_MatchNet"]), 4),
                         "skill_cov": round(float(r["skill_cov"]), 4)})
    return rows


def plot(result: dict) -> None:
    cats = sorted(result["weak"]["per_category"].items(), key=lambda kv: kv[1]["f1"])
    names = [SHORT.get(c, c.replace("-", " ").title()) for c, _ in cats]
    f1 = [v["f1"] for _, v in cats]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2),
                               gridspec_kw={"width_ratios": [2.2, 1]})
    a.barh(names, f1, color=BLUE, height=0.7)
    a.axvline(sum(f1) / len(f1), color=GREY, ls=":", lw=1)
    a.set(xlabel="F1", title="Task A: MatchNet F1 by resume category", xlim=(0, 1))
    a.tick_params(axis="y", labelsize=7.5)

    fit = result["human"]["error_rate_by_fit"]
    labels = list(fit)
    b.bar(labels, [fit[k] for k in labels], color=[GREY, ORANGE, BLUE], width=0.6)
    b.set(ylabel="error rate", title="Task B: error rate by human grade", ylim=(0, 1))
    b.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_error_analysis.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    weak, human = load_test("weak"), load_test("human")
    cats = per_category(weak)
    rates = error_rates(weak, human)
    result = {
        "weak": {**cats, "error_rate_by_pair_type": rates["error_rate_by_pair_type"],
                 "false_positives": rates["false_positives"],
                 "false_negatives": rates["false_negatives"],
                 "feature_means": feature_profile(weak),
                 "examples": examples(weak)},
        "human": {"error_rate_by_fit": rates["error_rate_by_fit"]},
    }
    save_json("error_analysis", result)
    plot(result)
    print("worst", cats["worst_categories"], "best", cats["best_categories"])
    print("pair_type", rates["error_rate_by_pair_type"], "fit", rates["error_rate_by_fit"])
    print("FP/FN", rates["false_positives"], rates["false_negatives"])


if __name__ == "__main__":
    main()
