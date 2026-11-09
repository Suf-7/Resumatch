"""Figures for the Deliverable 2 report (PDF for LaTeX + PNG for the HTML build).

Usage: python src/plots.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from config import FIGURES, PROCESSED, RUNS, load_json  # noqa: E402

BLUE, ORANGE, GREY = "#2B6CB0", "#DD6B20", "#718096"
TASKS = {"weak": "Task A: weak labels (our pairs)", "human": "Task B: human labels (external)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def bi_encoder_curves() -> None:
    tun = load_json("bi_encoder_tuning")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for row in tun["grid"]:
        name = f"lr{row['lr']:g}_bs{row['batch_size']}_len{row['max_seq_len']}"
        hist = json.loads((RUNS / f"bi_encoder_{name}.json").read_text())
        best = name == tun["best"]["config"]
        a.plot(hist["step"], hist["train_loss"], color=BLUE if best else GREY,
               alpha=1 if best else 0.35, lw=1.6 if best else 1, label=name if best else None)
        ep = [0] + hist["epoch"]
        b.plot(ep, [tun["pretrained_val"]["auc_hard"]] + [v["auc_hard"] for v in hist["val"]],
               marker="o", ms=3, color=BLUE if best else GREY, alpha=1 if best else 0.35,
               lw=1.6 if best else 1)
    a.set(title="Training loss (best run in blue)", xlabel="step", ylabel="contrastive loss")
    a.legend(frameon=False)
    b.set(title="Validation AUC, positives vs. hard negatives", xlabel="epoch (0 = pretrained)",
          ylabel="AUC-ROC")
    save(fig, "fig_bi_encoder_curves")


def matchnet_curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for col, (task, title) in enumerate(TASKS.items()):
        h = load_json(f"matchnet_{task}")["history"]
        a, b = axes[0, col], axes[1, col]
        a.plot(h["epoch"], h["train_loss"], color=BLUE, label="train")
        a.plot(h["epoch"], h["val_loss"], color=ORANGE, label="validation")
        a.axvline(h["best_epoch"], color=GREY, ls=":", lw=1)
        a.set(title=f"{title}: loss", xlabel="epoch", ylabel="weighted BCE")
        a.legend(frameon=False)
        b.plot(h["epoch"], h["train_acc"], color=BLUE, label="train accuracy")
        b.plot(h["epoch"], h["val_acc"], color=ORANGE, label="validation accuracy")
        b.plot(h["epoch"], h["val_auc"], color=ORANGE, ls="--", label="validation AUC")
        b.axvline(h["best_epoch"], color=GREY, ls=":", lw=1)
        b.set(title=f"{title}: accuracy / AUC", xlabel="epoch", ylim=(0.4, 1.0))
        b.legend(frameon=False, fontsize=8)
    save(fig, "fig_matchnet_curves")


def roc_and_confusion() -> None:
    ev = load_json("evaluation")
    show = ["ATS score (rules)", "MiniLM cosine (pretrained)", "Bi-encoder cosine (fine-tuned)",
            "Gradient boosting", "MatchNet"]
    colors = [GREY, "#A0AEC0", "#63B3ED", "#68D391", BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (task, title) in zip(axes, TASKS.items()):
        p = pd.read_parquet(PROCESSED / f"predictions_{task}.parquet")
        p = p[p.split == "test"]
        for name, c in zip(show, colors):
            fpr, tpr, _ = roc_curve(p["label"], p[f"score_{name}"])
            auc = ev[task]["models"][name]["auc_roc"]
            ax.plot(fpr, tpr, color=c, lw=2 if name == "MatchNet" else 1.2, label=f"{name} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], color="#CBD5E0", ls=":")
        ax.set(title=f"{title}: test ROC", xlabel="false positive rate", ylabel="true positive rate")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig_roc")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, (task, title) in zip(axes, TASKS.items()):
        cm = np.array(ev[task]["matchnet_confusion"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no match", "match"],
               yticklabels=["no match", "match"], xlabel="predicted", ylabel="true",
               title=title.split(":")[0] + ": MatchNet (test)")
    save(fig, "fig_confusion")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, color=BLUE, fill="#EBF4FF", dashed=False, size=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    ec=color, fc=fill, lw=1.4, ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     color="#4A5568", lw=1.1))

    ax.text(1, 42, "Stage 1: retrieval (bi-encoder)", fontsize=9, weight="bold", color=BLUE)
    box(1, 31, 14, 7, "Resume text")
    box(1, 21, 14, 7, "Posting text")
    box(21, 21, 20, 17, "MiniLM-L6 encoder\n(shared weights)\n6 transformer layers\nmean pooling + L2 norm\n384-d")
    arrow(15.5, 34.5, 20.5, 32)
    arrow(15.5, 24.5, 20.5, 27)
    box(47, 31, 12, 7, "u (384)")
    box(47, 21, 12, 7, "v (384)")
    arrow(41.5, 32, 46.5, 34.5)
    arrow(41.5, 27, 46.5, 24.5)
    box(65, 26, 15, 7, "cosine(u, v)\ntop-20 shortlist", fill="#FFFAF0", color=ORANGE)
    arrow(59.5, 34.5, 64.5, 30.5)
    arrow(59.5, 24.5, 64.5, 28.5)
    ax.text(84, 31, "loss: in-batch\ncontrastive\n(+ hard negative)", fontsize=7.5, color=GREY, va="center")

    ax.text(1, 16.5, "Stage 2: scoring (MatchNet)", fontsize=9, weight="bold", color=BLUE)
    box(1, 3, 17, 9, "[u, v, |u-v|, u*v]\n1536", size=8)
    box(23, 3, 17, 9, "Linear 1536→H\nReLU, Dropout")
    box(23, -6, 17, 6, "10 ATS features\nLinear 10→32, ReLU", size=7.5)
    box(46, 0, 17, 9, "concat\nLinear (H+32)→H/4\nReLU, Dropout")
    box(69, 1.5, 11, 6, "Linear H/4→1\nsigmoid", size=7.5)
    box(85, 1.5, 13, 6, "match\nprobability", fill="#FFFAF0", color=ORANGE)
    arrow(18.5, 7.5, 22.5, 7.5)
    arrow(40.5, 7.5, 45.5, 5.5)
    arrow(40.5, -3, 45.5, 2.5)
    arrow(63.5, 4.5, 68.5, 4.5)
    arrow(80.5, 4.5, 84.5, 4.5)
    ax.text(46, 12, "loss: weighted binary cross-entropy;  optimizer: AdamW", fontsize=7.5, color=GREY)
    ax.set_ylim(-8, 44)
    save(fig, "fig_architecture")


if __name__ == "__main__":
    architecture()
    bi_encoder_curves()
    matchnet_curves()
    roc_and_confusion()
