"""MatchNet: scores one (resume, posting) pair from fine-tuned embeddings + ATS features.

Architecture
    embedding branch : [u, v, |u - v|, u * v] (4 x 384 = 1536) -> Linear(1536, H) -> ReLU -> Dropout
    feature branch   : 10 standardized ATS / similarity features -> Linear(10, 32) -> ReLU
    head             : concat (H + 32) -> Linear(H + 32, H/4) -> ReLU -> Dropout -> Linear(H/4, 1)
Loss: binary cross-entropy on the logit, with pos_weight = #neg / #pos from the training split.
Optimizer: AdamW (weight decay 1e-4); early stopping on validation AUC-ROC.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class MatchNet(nn.Module):
    def __init__(self, emb_dim: int = 384, n_scalar: int = 10, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(4 * emb_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.feat = nn.Sequential(nn.Linear(n_scalar, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 32, hidden // 4), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden // 4, 1))

    def forward(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([u, v, (u - v).abs(), u * v], dim=-1)
        return self.head(torch.cat([self.emb(pair), self.feat(x)], dim=-1)).squeeze(-1)


def _tensors(d: dict, idx: np.ndarray):
    return (torch.from_numpy(d["u"][idx]), torch.from_numpy(d["v"][idx]),
            torch.from_numpy(d["x"][idx]), torch.from_numpy(d["y"][idx].astype(np.float32)))


@torch.no_grad()
def predict(model: MatchNet, d: dict, idx: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(idx), batch):
        u, v, x, _ = _tensors(d, idx[b: b + batch])
        out.append(torch.sigmoid(model(u, v, x)).numpy())
    return np.concatenate(out) if out else np.array([])


def fit(d: dict, train_idx: np.ndarray, val_idx: np.ndarray, *, lr: float, batch_size: int,
        dropout: float, hidden: int, max_epochs: int = 60, patience: int = 6, seed: int = 42):
    """d = {"u": resume emb, "v": posting emb, "x": scaled features, "y": labels} (numpy)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MatchNet(d["u"].shape[1], d["x"].shape[1], hidden, dropout)
    y_tr = d["y"][train_idx]
    pos_weight = torch.tensor((y_tr == 0).sum() / max(1, (y_tr == 1).sum()), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": [], "train_acc": [], "val_acc": []}
    best_auc, best_state, bad = -1.0, None, 0
    uv, vv, xv, yv = _tensors(d, val_idx)
    for ep in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        losses, correct = [], 0
        for b in range(0, len(order), batch_size):
            u, v, x, y = _tensors(d, order[b: b + batch_size])
            logit = model(u, v, x)
            loss = loss_fn(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item() * len(y))
            correct += int(((logit > 0).float() == y).sum())
        model.eval()
        with torch.no_grad():
            lv = model(uv, vv, xv)
            val_loss = float(loss_fn(lv, yv))
            pv = torch.sigmoid(lv).numpy()
        auc = float(roc_auc_score(yv.numpy(), pv))
        hist["epoch"].append(ep)
        hist["train_loss"].append(sum(losses) / len(order))
        hist["val_loss"].append(val_loss)
        hist["val_auc"].append(auc)
        hist["train_acc"].append(correct / len(order))
        hist["val_acc"].append(float(((pv > 0.5) == yv.numpy()).mean()))
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    hist["best_epoch"] = hist["epoch"][int(np.argmax(hist["val_auc"]))]
    hist["best_val_auc"] = best_auc
    return model, hist
