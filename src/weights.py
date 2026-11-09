"""Publish or fetch the trained weights (kept out of Git) on the Hugging Face Hub.

    python src/weights.py upload     # needs `hf auth login` with write access to REPO_ID
    python src/weights.py download   # restores models/bi_encoder/ and models/matchnet_*.pt

Uploaded: models/bi_encoder/ (fine-tuned MiniLM-L6, sentence-transformers format) and
models/matchnet_{weak,human}.pt (state dict + config + feature scaler).
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, snapshot_download

from config import MODELS

REPO_ID = os.environ.get("RESUMATCH_HF_REPO", "Suf-7/resumatch-models")
PATTERNS = ["bi_encoder/**", "matchnet_*.pt"]


def upload() -> None:
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO_ID, folder_path=str(MODELS), allow_patterns=PATTERNS,
                      commit_message="Resumatch Deliverable 2 weights")
    print(f"uploaded to https://huggingface.co/{REPO_ID}")


def download() -> None:
    snapshot_download(REPO_ID, local_dir=str(MODELS), allow_patterns=PATTERNS)
    print(f"weights in {MODELS}")


if __name__ == "__main__":
    {"upload": upload, "download": download}[sys.argv[1] if len(sys.argv) > 1 else "download"]()
