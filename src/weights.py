"""Publish or fetch the trained weights (kept out of Git).

    python src/weights.py upload          # Hugging Face Hub (needs `hf auth login`)
    python src/weights.py upload-github   # GitHub Release deliverable2-weights
    python src/weights.py download        # HF first, then the GitHub Release

Uploaded: models/bi_encoder/ (fine-tuned MiniLM-L6, sentence-transformers format) and
models/matchnet_{weak,human}.pt (state dict + config + feature scaler).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from config import MODELS, ROOT

REPO_ID = os.environ.get("RESUMATCH_HF_REPO", "Suf-7/resumatch-models")
GITHUB_TAG = "deliverable2-weights"
GITHUB_TARBALL = ("https://github.com/Suf-7/Resumatch/releases/download/"
                  f"{GITHUB_TAG}/resumatch-weights.tar.gz")
PATTERNS = ("bi_encoder", "matchnet_weak.pt", "matchnet_human.pt")


def _tarball(dest: Path) -> Path:
    with tarfile.open(dest, "w:gz") as tar:
        for name in PATTERNS:
            tar.add(MODELS / name, arcname=name)
    return dest


def upload() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO_ID, folder_path=str(MODELS),
                      allow_patterns=["bi_encoder/**", "matchnet_*.pt"],
                      commit_message="Resumatch Deliverable 2 weights")
    print(f"uploaded to https://huggingface.co/{REPO_ID}")


def upload_github() -> None:
    dest = Path(tempfile.gettempdir()) / "resumatch-weights.tar.gz"
    _tarball(dest)
    subprocess.run(["gh", "release", "delete", GITHUB_TAG, "--yes", "-R", "Suf-7/Resumatch"],
                   cwd=ROOT, check=False)
    subprocess.run(["gh", "release", "create", GITHUB_TAG, str(dest),
                    "-R", "Suf-7/Resumatch",
                    "-t", "Deliverable 2 model weights",
                    "-n", "Fine-tuned MiniLM-L6 bi-encoder (`bi_encoder/`) and MatchNet "
                          "(`matchnet_weak.pt`, `matchnet_human.pt`). Unpack into `models/`."],
                   cwd=ROOT, check=True)
    print(f"uploaded to {GITHUB_TARBALL}")


def download() -> None:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID, local_dir=str(MODELS),
                          allow_patterns=["bi_encoder/**", "matchnet_*.pt"])
        print(f"weights from Hugging Face in {MODELS}")
        return
    except Exception as exc:
        print(f"Hugging Face download failed ({exc}); trying GitHub Release")
    dest = Path(tempfile.gettempdir()) / "resumatch-weights.tar.gz"
    urlretrieve(GITHUB_TARBALL, dest)
    with tarfile.open(dest) as tar:
        tar.extractall(MODELS)
    print(f"weights from GitHub Release in {MODELS}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    {"upload": upload, "upload-github": upload_github, "download": download}[cmd]()
