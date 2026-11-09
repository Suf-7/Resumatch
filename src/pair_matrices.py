"""Build and cache the model inputs for each task.

For every pair: resume and posting embeddings from the pretrained and the fine-tuned
bi-encoder, the hand-crafted features (features.py), the label and metadata.
Cached to data/processed/pairs_<task>.npz + pairs_<task>_meta.parquet.

Usage: python src/pair_matrices.py            # both tasks
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASE_ENCODER, MODELS, PROCESSED
from data import load_human_pairs, load_weak_pairs
from features import SCALAR_FEATURES, add_side_features, scalar_features
from models import bi_encoder

META = ["resume_id", "job_id", "split", "label", "pair_type", "resume_category", "job_category", "fit3"]


def embed_pairs(df: pd.DataFrame, encoder_path) -> tuple[np.ndarray, np.ndarray]:
    model = bi_encoder.load_encoder(encoder_path)
    texts = pd.unique(pd.concat([df["resume_text"], df["job_text"]]))
    emb = dict(zip(texts, bi_encoder.encode(model, texts)))
    return (np.stack(df["resume_text"].map(emb).to_list()).astype(np.float32),
            np.stack(df["job_text"].map(emb).to_list()).astype(np.float32))


def build(task: str) -> None:
    df = load_weak_pairs() if task == "weak" else load_human_pairs()
    df = add_side_features(df)
    feats = scalar_features(df)
    u0, v0 = embed_pairs(df, BASE_ENCODER)
    u, v = embed_pairs(df, MODELS / "bi_encoder")
    np.savez_compressed(PROCESSED / f"pairs_{task}.npz", u=u, v=v, u0=u0, v0=v0,
                        x=feats.to_numpy(), y=df["label"].to_numpy())
    meta = df[[c for c in META if c in df]].copy()
    meta["cos_pretrained"] = (u0 * v0).sum(1)
    meta["cos_finetuned"] = (u * v).sum(1)
    meta = pd.concat([meta, feats.reset_index(drop=True)], axis=1)
    meta.to_parquet(PROCESSED / f"pairs_{task}_meta.parquet", index=False)
    print(task, len(df), "features:", SCALAR_FEATURES)


def load(task: str) -> tuple[dict, pd.DataFrame]:
    z = np.load(PROCESSED / f"pairs_{task}.npz")
    return {k: z[k] for k in z.files}, pd.read_parquet(PROCESSED / f"pairs_{task}_meta.parquet")


if __name__ == "__main__":
    for t in ("weak", "human"):
        build(t)
