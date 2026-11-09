"""Grid-search and train MatchNet for both tasks; select on validation AUC-ROC only.

Outputs: models/matchnet_<task>.pt (weights + config + feature scaler),
report/stats/matchnet_<task>.json (grid table, best config, curves, ablations).

Usage: python src/train_matchnet.py [--task weak|human]
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import MODELS, save_json, seed_everything
from models.matchnet import fit
from pair_matrices import load

GRID = {"lr": [1e-3, 3e-4, 1e-4], "batch_size": [64, 256], "dropout": [0.1, 0.3, 0.5],
        "hidden": [128, 256]}


def prepare(task: str):
    d, meta = load(task)
    idx = {s: np.where(meta["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}
    scaler = StandardScaler().fit(d["x"][idx["train"]])
    d["x"] = scaler.transform(d["x"]).astype(np.float32)
    return d, meta, idx, scaler


def run(task: str) -> None:
    seed_everything()
    d, meta, idx, scaler = prepare(task)
    rows, best = [], None
    for lr, bs, dp, h in itertools.product(*GRID.values()):
        t0 = time.time()
        model, hist = fit(d, idx["train"], idx["val"], lr=lr, batch_size=bs, dropout=dp, hidden=h)
        row = {"lr": lr, "batch_size": bs, "dropout": dp, "hidden": h,
               "best_epoch": hist["best_epoch"], "val_auc": round(hist["best_val_auc"], 4),
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(task, row)
        if best is None or row["val_auc"] > best[0]["val_auc"]:
            best = (row, model, hist)
    cfg, model, hist = best

    torch.save({"state_dict": model.state_dict(),
                "config": {k: cfg[k] for k in ("hidden", "dropout")},
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "emb_dim": d["u"].shape[1], "n_scalar": d["x"].shape[1]},
               MODELS / f"matchnet_{task}.pt")

    hp = {k: cfg[k] for k in ("lr", "batch_size", "dropout", "hidden")}
    ablations = {}
    for name, dd in {
        "pretrained_embeddings": {**d, "u": d["u0"], "v": d["v0"]},
        "no_ats_features": {**d, "x": np.zeros_like(d["x"])},
    }.items():
        _, h2 = fit(dd, idx["train"], idx["val"], **hp)
        ablations[name] = round(h2["best_val_auc"], 4)
    save_json(f"matchnet_{task}", {
        "grid": rows, "best": cfg, "history": {k: v for k, v in hist.items()},
        "ablations_val_auc": ablations, "n_params": sum(p.numel() for p in model.parameters()),
        "split_sizes": {k: int(len(v)) for k, v in idx.items()},
    })
    print(task, "best", cfg, "ablations", ablations)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["weak", "human"])
    args = ap.parse_args()
    for t in ([args.task] if args.task else ["weak", "human"]):
        run(t)
