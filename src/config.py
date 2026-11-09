"""Paths, seeds and device shared by the Phase 3/4 code."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_SCRIPTS = ROOT / "data_scripts"
if str(DATA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPTS))

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RUNS = MODELS / "runs"
STATS = ROOT / "report" / "stats"
FIGURES = ROOT / "report" / "figures"
for d in (MODELS, RUNS, STATS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
BASE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN = 256


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(name: str, obj) -> Path:
    path = STATS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path


def load_json(name: str) -> dict:
    path = STATS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}
