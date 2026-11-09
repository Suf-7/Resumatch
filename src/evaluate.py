"""Phase 4 evaluation: every model on the held-out test split of both tasks.

Models: rule-based ATS score, TF-IDF cosine, pretrained MiniLM cosine, fine-tuned bi-encoder
cosine, logistic regression and gradient boosting (scikit-learn) on the pair features, and
MatchNet. Decision thresholds are chosen on the validation split (max F1) and then applied
unchanged to the test split. 95% confidence intervals: 1,000 bootstrap resamples of the test set.

Outputs: report/stats/evaluation.json, data/processed/predictions_<task>.parquet
Usage: python src/evaluate.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS, PROCESSED, SEED, save_json
from features import SCALAR_FEATURES
from models.matchnet import MatchNet, predict
from pair_matrices import load

TAB = SCALAR_FEATURES + ["cos_finetuned"]


def best_threshold(y, s) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def metrics(y, s, thr) -> dict:
    yhat = (s >= thr).astype(int)
    return {"precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat), "accuracy": accuracy_score(y, yhat),
            "auc_roc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), "threshold": thr}


def bootstrap(y, s, thr, n=1000) -> dict:
    rng = np.random.default_rng(SEED)
    aucs, f1s = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
        f1s.append(f1_score(y[i], (s[i] >= thr).astype(int)))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"auc_ci": q(aucs), "f1_ci": q(f1s)}


def tune_sklearn(make, grid, X, y, tr, va):
    best = None
    for params in grid:
        m = make(**params).fit(X[tr], y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, params, m)
    return best


def matchnet_scores(task: str, d: dict) -> np.ndarray:
    ck = torch.load(MODELS / f"matchnet_{task}.pt", weights_only=False)
    net = MatchNet(ck["emb_dim"], ck["n_scalar"], **ck["config"])
    net.load_state_dict(ck["state_dict"])
    x = ((d["x"] - ck["scaler_mean"]) / ck["scaler_scale"]).astype(np.float32)
    return predict(net, {**d, "x": x}, np.arange(len(d["y"])))


def evaluate(task: str) -> dict:
    d, meta = load(task)
    y = meta["label"].to_numpy()
    sp = meta["split"].to_numpy()
    tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    X = meta[TAB].to_numpy()

    scores = {
        "ATS score (rules)": meta["ats_score"].to_numpy(),
        "TF-IDF cosine": meta["tfidf_cos"].to_numpy(),
        "MiniLM cosine (pretrained)": meta["cos_pretrained"].to_numpy(),
        "Bi-encoder cosine (fine-tuned)": meta["cos_finetuned"].to_numpy(),
    }
    lr_auc, lr_params, lr = tune_sklearn(
        lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)),
        [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)], X, y, tr, va)
    scores["Logistic regression"] = lr.predict_proba(X)[:, 1]
    gb_auc, gb_params, gb = tune_sklearn(
        lambda **p: HistGradientBoostingClassifier(random_state=SEED, **p),
        [{"max_depth": md, "learning_rate": lr_, "max_iter": 300}
         for md, lr_ in itertools.product((3, 6, None), (0.05, 0.1))], X, y, tr, va)
    scores["Gradient boosting"] = gb.predict_proba(X)[:, 1]
    scores["MatchNet"] = matchnet_scores(task, d)

    results = {}
    for name, s in scores.items():
        thr = best_threshold(y[va], s[va])
        results[name] = {k: round(float(v), 4) for k, v in metrics(y[te], s[te], thr).items()}
        results[name].update(bootstrap(y[te], s[te], thr))
        results[name]["val_auc_roc"] = round(float(roc_auc_score(y[va], s[va])), 4)

    mn = scores["MatchNet"]
    thr = results["MatchNet"]["threshold"]
    pred = meta.copy()
    for name, s in scores.items():
        pred[f"score_{name}"] = s
    pred["matchnet_pred"] = (mn >= thr).astype(int)
    pred.to_parquet(PROCESSED / f"predictions_{task}.parquet", index=False)

    out = {"models": results, "sklearn_params": {"logistic_regression": lr_params,
                                                 "gradient_boosting": gb_params},
           "test_size": int(len(te)), "test_positive_share": round(float(y[te].mean()), 3),
           "matchnet_confusion": confusion_matrix(y[te], (mn[te] >= thr).astype(int)).tolist()}
    tm = meta.iloc[te]
    if "pair_type" in meta and meta["pair_type"].notna().any():
        hit = (mn[te] >= thr).astype(int) == y[te]
        out["matchnet_accuracy_by_pair_type"] = (
            pd.Series(hit, index=tm.index).groupby(tm["pair_type"]).mean().round(3).to_dict())
    if "fit3" in meta and meta["fit3"].notna().any():
        out["matchnet_mean_score_by_fit"] = (
            pd.Series(mn[te], index=tm.index).groupby(tm["fit3"]).mean().round(3)
            .rename({0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}).to_dict())
        gv = tm["fit3"].isin([0, 2]).to_numpy()
        out["matchnet_auc_good_vs_no"] = round(float(roc_auc_score(tm["fit3"].to_numpy()[gv] == 2,
                                                                   mn[te][gv])), 4)
    return out


def main() -> None:
    res = {t: evaluate(t) for t in ("weak", "human")}
    save_json("evaluation", res)
    for t, r in res.items():
        print(f"\n== {t} (test n={r['test_size']}) ==")
        print(pd.DataFrame(r["models"]).T[["precision", "recall", "f1", "auc_roc", "pr_auc"]].round(3))


if __name__ == "__main__":
    main()
