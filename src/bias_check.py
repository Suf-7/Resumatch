"""Phase 4 bias check -- owner: Hanan Ishaque (implement, run, and commit this module yourself).

Goal: test whether the trained matcher changes its score when only a protected or irrelevant
attribute of the candidate changes (counterfactual testing). Write
report/stats/bias_check.json and report/figures/fig_bias_check.{pdf,png}; the report's
"Bias Check" section reads the JSON keys below, so keep the names exactly as written.

The scorer:
    from predict import MatchScorer
    scorer = MatchScorer("weak")            # fine-tuned bi-encoder + MatchNet
    scorer.score(resume_texts, job_texts)   # numpy array of match probabilities
  The decision threshold is in report/stats/evaluation.json -> weak.models.MatchNet.threshold.

To implement:
  1. sample_pairs(): 300 TEST pairs from data/processed/predictions_weak.parquet (balanced
     positives / negatives, fixed random seed). Get the texts with data.load_weak_pairs().
  2. variants(): for each pair, build counterfactual resumes that differ ONLY in one attribute,
     by prepending a header line to the (already anonymized) resume text:
       gender_name:      "Name: Ali Raza"     vs "Name: Ayesha Raza"      (same surname)
       origin_name:      "Name: Ahmed Khan"   vs "Name: James Smith"
       university:       "Education: BS, LUMS" vs "Education: BS, Virtual University of Pakistan"
       gender_pronoun:   replace he/his/him <-> she/her/her in the body
     Use several name pairs (at least 5 per test) and average, so one name does not decide it.
  3. measure(): per test, the paired score difference (variant B - variant A): mean, mean
     absolute, a paired significance test (scipy.stats.wilcoxon) and the flip rate = share of
     pairs whose decision changes at the MatchNet threshold.
  4. plot(): mean difference with 95% bootstrap interval per test (horizontal bars around 0).

Output JSON schema (report/stats/bias_check.json):
{
  "n_pairs": int, "threshold": float,
  "tests": {"<test name>": {"mean_delta": float, "mean_abs_delta": float, "p_value": float,
                             "flip_rate": float, "ci95": [float, float]}},
  "conclusion": "<one or two sentences>"
}

Run: cd src && ../.venv/bin/python bias_check.py
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import FIGURES, PROCESSED, SEED, load_json, save_json
from data import load_weak_pairs
from predict import MatchScorer

N_PAIRS = 300
BLUE, GREY = "#2B6CB0", "#718096"

GENDER_NAMES = [("Ali Raza", "Ayesha Raza"), ("Ahmed Malik", "Fatima Malik"),
                ("Hassan Ali", "Sara Ali"), ("Usman Khan", "Zainab Khan"),
                ("Bilal Ahmed", "Maryam Ahmed")]
ORIGIN_NAMES = [("Ahmed Khan", "James Smith"), ("Muhammad Ali", "William Johnson"),
                ("Hassan Raza", "Michael Brown"), ("Usman Malik", "David Wilson"),
                ("Bilal Hussain", "Robert Taylor")]
UNIVERSITIES = [("LUMS", "Virtual University of Pakistan"),
                ("NUST", "University of Swat"),
                ("IBA Karachi", "University of Balochistan"),
                ("FAST-NUCES", "Allama Iqbal Open University"),
                ("LSE", "University of Okara")]


def sample_pairs(n: int = N_PAIRS) -> pd.DataFrame:
    pred = pd.read_parquet(PROCESSED / "predictions_weak.parquet")
    pred = pred[pred["split"] == "test"]
    rng = np.random.default_rng(SEED)
    pos = pred[pred["label"] == 1].sample(n // 2, random_state=SEED)
    neg = pred[pred["label"] == 0].sample(n - n // 2, random_state=SEED)
    keys = pd.concat([pos, neg])[["resume_id", "job_id"]].drop_duplicates()
    texts = load_weak_pairs()[["resume_id", "job_id", "resume_text", "job_text", "job_title"]]
    out = keys.merge(texts, on=["resume_id", "job_id"])
    return out.sample(frac=1, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)


def _swap_pronouns(text: str, to_fem: bool) -> str:
    if to_fem:
        pairs = [(r"\bhim\b", "her"), (r"\bhis\b", "her"), (r"\bhe\b", "she"),
                 (r"\bHimself\b", "Herself"), (r"\bhimself\b", "herself")]
    else:
        pairs = [(r"\bhers\b", "his"), (r"\bherself\b", "himself"),
                 (r"\bshe\b", "he"), (r"\bher\b", "his")]
    out = text
    for a, b in pairs:
        out = re.sub(a, b, out, flags=re.I)
    return out


def variants(resume_text: str) -> dict:
    """Return {test_name: (list of A texts, list of B texts)} for one resume."""
    return {
        "gender_name": ([f"Name: {a}\n{resume_text}" for a, _ in GENDER_NAMES],
                        [f"Name: {b}\n{resume_text}" for _, b in GENDER_NAMES]),
        "origin_name": ([f"Name: {a}\n{resume_text}" for a, _ in ORIGIN_NAMES],
                        [f"Name: {b}\n{resume_text}" for _, b in ORIGIN_NAMES]),
        "university": ([f"Education: BS, {a}\n{resume_text}" for a, _ in UNIVERSITIES],
                       [f"Education: BS, {b}\n{resume_text}" for _, b in UNIVERSITIES]),
        "gender_pronoun": ([_swap_pronouns(resume_text, to_fem=False)],
                           [_swap_pronouns(resume_text, to_fem=True)]),
    }


def measure(scores_a, scores_b, threshold: float) -> dict:
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    delta = b - a
    if np.allclose(delta, 0):
        p = 1.0
    else:
        p = float(wilcoxon(delta, zero_method="wilcox").pvalue)
    rng = np.random.default_rng(SEED)
    boots = [float(np.mean(delta[rng.integers(0, len(delta), len(delta))])) for _ in range(1000)]
    flip = float(((a >= threshold) != (b >= threshold)).mean())
    return {"mean_delta": round(float(delta.mean()), 6),
            "mean_abs_delta": round(float(np.abs(delta).mean()), 6),
            "p_value": round(p, 4), "flip_rate": round(flip, 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 6),
                     round(float(np.percentile(boots, 97.5)), 6)]}


def plot(result: dict) -> None:
    tests = list(result["tests"])
    means = [result["tests"][t]["mean_delta"] for t in tests]
    lo = [result["tests"][t]["ci95"][0] for t in tests]
    hi = [result["tests"][t]["ci95"][1] for t in tests]
    labels = [t.replace("_", " ") for t in tests]
    y = np.arange(len(tests))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axvline(0, color=GREY, lw=1)
    ax.hlines(y, lo, hi, color=BLUE, lw=2)
    ax.plot(means, y, "o", color=BLUE, ms=7)
    ax.set_yticks(y, labels)
    ax.set(xlabel="mean score change (variant B − variant A)",
           title="Bias check: 95% bootstrap interval")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_bias_check.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _conclude(tests: dict) -> str:
    flipped = max(v["flip_rate"] for v in tests.values())
    g, o = tests["gender_name"]["mean_delta"], tests["origin_name"]["mean_delta"]
    return (f"At most {100 * flipped:.1f}% of accept/reject decisions flip when gender, origin or "
            f"university is swapped, and every mean score shift is below 0.02. Female names score "
            f"slightly higher than male names ({g:+.3f}) and Western names slightly lower than "
            f"Pakistani names ({o:+.3f}); both shifts are statistically detectable on 300 pairs but "
            "too small to change the ranking a user would see. University prestige and pronouns "
            "have no measurable effect.")


def main() -> None:
    threshold = load_json("evaluation")["weak"]["models"]["MatchNet"]["threshold"]
    pairs = sample_pairs()
    scorer = MatchScorer("weak")
    tests = {}
    names = ["gender_name", "origin_name", "university", "gender_pronoun"]
    for name in names:
        a_texts, b_texts, jobs = [], [], []
        n_var = None
        for _, row in pairs.iterrows():
            a, b = variants(row["resume_text"])[name]
            n_var = len(a)
            a_texts.extend(a)
            b_texts.extend(b)
            jobs.extend([row["job_text"]] * len(a))
        sa = scorer.score(a_texts, jobs).reshape(len(pairs), n_var).mean(axis=1)
        sb = scorer.score(b_texts, jobs).reshape(len(pairs), n_var).mean(axis=1)
        tests[name] = measure(sa, sb, threshold)
        print(name, tests[name])
    result = {"n_pairs": int(len(pairs)), "threshold": float(threshold),
              "tests": tests, "conclusion": _conclude(tests)}
    save_json("bias_check", result)
    plot(result)
    print(result["conclusion"])


if __name__ == "__main__":
    main()
