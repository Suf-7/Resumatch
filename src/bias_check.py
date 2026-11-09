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

import pandas as pd

from config import FIGURES, PROCESSED, load_json, save_json


def sample_pairs(n: int = 300) -> pd.DataFrame:
    raise NotImplementedError("Hanan: step 1")


def variants(resume_text: str) -> dict:
    raise NotImplementedError("Hanan: step 2")


def measure(scores_a, scores_b, threshold: float) -> dict:
    raise NotImplementedError("Hanan: step 3")


def plot(result: dict) -> None:
    raise NotImplementedError("Hanan: step 4 (save to FIGURES / 'fig_bias_check.pdf' and .png)")


def main() -> None:
    threshold = load_json("evaluation")["weak"]["models"]["MatchNet"]["threshold"]
    raise NotImplementedError("Hanan: combine steps 1-4 into the schema above, save_json('bias_check', ...)")


if __name__ == "__main__":
    main()
