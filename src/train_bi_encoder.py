"""Fine-tune the bi-encoder retriever on the training pairs (Task A) with a small grid search.

Model selection uses the validation split only: AUC of cosine similarity for positives vs.
hard negatives (the case Deliverable 1 showed the pretrained encoder cannot handle) and
precision@3 for retrieving same-occupation postings among all validation postings.

Usage: python src/train_bi_encoder.py [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import INTERIM, MODELS, PROCESSED, RUNS, save_json, seed_everything
from data import load_weak_pairs
from models import bi_encoder

GRID = [  # (learning rate, batch size, max sequence length)
    (1e-5, 32, 256), (3e-5, 32, 256), (1e-4, 32, 256), (3e-5, 16, 256), (3e-5, 32, 128),
]
EPOCHS = 3


def triplet_sampler_factory(pairs: pd.DataFrame):
    tr = pairs[pairs.split == "train"]
    pos = tr[tr.pair_type == "positive"].groupby("resume_id")
    hard = tr[tr.pair_type == "hard_negative"].groupby("resume_id")["job_text"].apply(list).to_dict()
    res_text = tr.drop_duplicates("resume_id").set_index("resume_id")["resume_text"].to_dict()
    pos_lists = pos["job_text"].apply(list).to_dict()
    ids = [r for r in pos_lists if r in hard]

    def sample(epoch: int):
        return [(res_text[r], pos_lists[r][epoch % len(pos_lists[r])],
                 hard[r][epoch % len(hard[r])]) for r in ids]
    return sample


class Validator:
    def __init__(self, pairs: pd.DataFrame):
        self.val = pairs[pairs.split == "val"].reset_index(drop=True)
        jobs = pd.read_parquet(PROCESSED / "jobs_features.parquet")[["job_id", "split", "job_category"]]
        jobs = jobs[(jobs.split == "val") & (jobs.job_category != "unknown")]
        text = pd.read_parquet(INTERIM / "jobs_clean.parquet").set_index("job_id")["text"]
        self.pool_text = text.loc[jobs.job_id].tolist()
        self.pool_cat = jobs.job_category.to_numpy()
        r = self.val.drop_duplicates("resume_id")
        self.res_text, self.res_cat = r.resume_text.tolist(), r.resume_category.to_numpy()

    def __call__(self, model, epoch=None) -> dict:
        texts = pd.unique(pd.concat([self.val.resume_text, self.val.job_text]))
        emb = dict(zip(texts, bi_encoder.encode(model, texts)))
        cos = np.array([emb[a] @ emb[b] for a, b in zip(self.val.resume_text, self.val.job_text)])
        y = self.val.label.to_numpy()
        hard = (self.val.pair_type != "random_negative").to_numpy()
        R = np.stack([emb[t] for t in self.res_text])
        P = bi_encoder.encode(model, self.pool_text)
        top3 = np.argsort(-(R @ P.T), axis=1)[:, :3]
        p_at_3 = float((self.pool_cat[top3] == self.res_cat[:, None]).mean())
        return {"auc": round(float(roc_auc_score(y, cos)), 4),
                "auc_hard": round(float(roc_auc_score(y[hard], cos[hard])), 4),
                "p_at_3": round(p_at_3, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one config, one epoch (smoke test)")
    args = ap.parse_args()
    seed_everything()
    pairs = load_weak_pairs()
    sampler = triplet_sampler_factory(pairs)
    validate = Validator(pairs)
    grid, epochs = (GRID[1:2], 1) if args.quick else (GRID, EPOCHS)

    base = validate(bi_encoder.load_encoder())
    print("pretrained (no fine-tuning):", base)
    results, best, best_hist = [], {"auc_hard": base["auc_hard"]}, None
    out_dir = MODELS / "bi_encoder"
    for lr, bs, msl in grid:
        seed_everything()
        name = f"lr{lr:g}_bs{bs}_len{msl}"
        print(f"[run] {name}")
        bi_encoder.free_device_memory()
        model = bi_encoder.load_encoder(max_seq_len=msl)
        run_best = {"auc_hard": -1}

        def on_epoch_end(m, ep):
            nonlocal best
            v = validate(m)
            if v["auc_hard"] > run_best["auc_hard"]:
                run_best.update(v, epoch=ep + 1)
            if v["auc_hard"] > best["auc_hard"]:
                best = {**v, "config": name, "epoch": ep + 1}
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                m.save(str(out_dir))
            return v

        t0 = time.time()
        hist = bi_encoder.train(model, sampler, lr=lr, batch_size=bs, epochs=epochs,
                                on_epoch_end=on_epoch_end)
        del model
        (RUNS / f"bi_encoder_{name}.json").write_text(json.dumps(hist))
        results.append({"lr": lr, "batch_size": bs, "max_seq_len": msl, "best_epoch": run_best["epoch"],
                        "val_auc": run_best["auc"], "val_auc_hard": run_best["auc_hard"],
                        "val_p_at_3": run_best["p_at_3"], "minutes": round((time.time() - t0) / 60, 1)})
        if best.get("config") == name:
            best_hist = hist
        print(f"[done] {name}: {results[-1]}")

    save_json("bi_encoder_tuning", {
        "pretrained_val": base, "grid": results, "best": best, "epochs": epochs,
        "best_history": best_hist, "loss": "multiple-negatives ranking (in-batch + 1 hard negative), scale 20",
        "optimizer": "AdamW, weight decay 0.01, 10% linear warm-up, linear decay, grad clip 1.0",
        "train_triplets_per_epoch": len(sampler(0)),
    })
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
