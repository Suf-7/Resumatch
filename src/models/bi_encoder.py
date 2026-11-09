"""Bi-encoder retriever: MiniLM-L6 (6 transformer layers, 384-d, mean pooling, L2 norm)
fine-tuned with a contrastive loss.

Loss (multiple-negatives ranking): for a batch of B (resume, matching posting, hard negative
posting) triplets, each resume must pick its own posting out of all 2B postings in the batch:
    L = CrossEntropy(scale * cos(resume_i, posting_j), target = i)
so the other B-1 positives act as in-batch negatives and the mined hard negatives as
difficult ones. Optimizer: AdamW with linear warm-up (10%) and linear decay.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from config import BASE_ENCODER, MAX_SEQ_LEN, device


def free_device_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_encoder(path: str = BASE_ENCODER, max_seq_len: int = MAX_SEQ_LEN) -> SentenceTransformer:
    model = SentenceTransformer(str(path), device=str(device()))
    model.max_seq_length = max_seq_len
    return model


def _embed(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    feats = model.tokenize(texts)
    feats = {k: v.to(model.device) for k, v in feats.items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)


@torch.no_grad()
def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    model.eval()
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def train(model: SentenceTransformer, triplet_sampler, *, lr: float, batch_size: int,
          epochs: int, scale: float = 20.0, weight_decay: float = 0.01, warmup: float = 0.1,
          on_epoch_end=None, log_every: int = 10) -> dict:
    """triplet_sampler(epoch) -> list of (resume_text, positive_text, hard_negative_text)."""
    n_per_epoch = len(triplet_sampler(0))
    steps_total = epochs * math.ceil(n_per_epoch / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = max(1, int(warmup * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps_total - s) / max(1, steps_total - warm))))
    history = {"step": [], "train_loss": [], "epoch": [], "val": []}
    model[0].auto_model.gradient_checkpointing_enable()
    rng = np.random.default_rng(0)
    step, running = 0, []
    for ep in range(epochs):
        model.train()
        triplets = triplet_sampler(ep)
        order = rng.permutation(len(triplets))
        t0 = time.time()
        for b in range(0, len(order), batch_size):
            batch = [triplets[i] for i in order[b: b + batch_size]]
            if len(batch) < 2:
                continue
            a = _embed(model, [t[0] for t in batch])
            c = _embed(model, [t[1] for t in batch] + [t[2] for t in batch])
            logits = scale * a @ c.T
            loss = F.cross_entropy(logits, torch.arange(len(batch), device=logits.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running.append(loss.item())
            step += 1
            if step % log_every == 0:
                history["step"].append(step)
                history["train_loss"].append(float(np.mean(running)))
                running = []
        free_device_memory()
        val = on_epoch_end(model, ep) if on_epoch_end else {}
        free_device_memory()
        history["epoch"].append(ep + 1)
        history["val"].append(val)
        print(f"  epoch {ep + 1}/{epochs} {time.time() - t0:.0f}s loss="
              f"{history['train_loss'][-1] if history['train_loss'] else float('nan'):.3f} val={val}")
    return history
