"""Download the Kaggle resume and job-posting datasets.

Uses the official Kaggle API when credentials exist (~/.kaggle/kaggle.json or
KAGGLE_USERNAME/KAGGLE_KEY). Otherwise falls back to byte-identical public
mirrors on the Hugging Face Hub so the pipeline is reproducible without an account.

Usage:
    python data_scripts/download_kaggle.py            # all sources
    python data_scripts/download_kaggle.py --only resumes_main
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

from common import RAW, save_stats

SOURCES = {
    # Kaggle: snehaanbhawal/resume-dataset (2,484 resumes, 24 categories)
    "resumes_main": {
        "kaggle": ("snehaanbhawal/resume-dataset", "Resume/Resume.csv"),
        "hf": "https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv",
        "out": "resumes_kaggle_snehaanbhawal.csv",
    },
    # Kaggle: gauravduttakiit/resume-dataset (962 resumes, 25 categories)
    "resumes_updated": {
        "kaggle": ("gauravduttakiit/resume-dataset", "UpdatedResumeDataSet.csv"),
        "hf": "https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset/resolve/main/UpdatedResumeDataSet.csv",
        "out": "resumes_kaggle_gauravduttakiit.csv",
    },
    # Kaggle: arshkon/linkedin-job-postings (LinkedIn postings 2023-24)
    "linkedin_jobs": {
        "kaggle": ("arshkon/linkedin-job-postings", "job_postings.csv"),
        "hf": "https://huggingface.co/datasets/xanderios/linkedin-job-postings/resolve/main/job_postings.csv",
        "out": "jobs_linkedin_arshkon.csv",
    },
    # Human-labelled resume/JD fit pairs (No Fit / Potential Fit / Good Fit), used in Phase 3-4
    "fit_pairs_train": {
        "kaggle": None,
        "hf": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/train.csv",
        "out": "pairs_resume_jd_fit_train.csv",
    },
    "fit_pairs_test": {
        "kaggle": None,
        "hf": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/test.csv",
        "out": "pairs_resume_jd_fit_test.csv",
    },
}


def have_kaggle_credentials() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    )


def download_kaggle(dataset: str, member: str, out: Path) -> None:
    tmp = RAW / "_kaggle_tmp"
    tmp.mkdir(exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-f", member, "-p", str(tmp), "--unzip"],
        check=True,
    )
    found = next(tmp.rglob(Path(member).name))
    shutil.move(str(found), out)
    shutil.rmtree(tmp, ignore_errors=True)


def download_http(url: str, out: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(SOURCES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    use_kaggle = have_kaggle_credentials() and shutil.which("kaggle")
    manifest = {}
    for key, src in SOURCES.items():
        if args.only and key not in args.only:
            continue
        out = RAW / src["out"]
        if out.exists() and not args.force:
            print(f"[skip] {out.name} exists")
        else:
            t0 = time.time()
            if use_kaggle and src["kaggle"]:
                print(f"[kaggle] {src['kaggle'][0]} -> {out.name}")
                download_kaggle(*src["kaggle"], out)
                via = "kaggle"
            else:
                print(f"[hf-mirror] {src['hf']} -> {out.name}")
                download_http(src["hf"], out)
                via = "huggingface-mirror"
            print(f"  done in {time.time() - t0:.1f}s")
            manifest[key] = {"via": via}
        manifest.setdefault(key, {})
        manifest[key].update(
            {
                "file": out.name,
                "rows": int(len(pd.read_csv(out, usecols=[0]))),
                "bytes": out.stat().st_size,
                "kaggle_dataset": src["kaggle"][0] if src["kaggle"] else None,
                "mirror_url": src["hf"],
            }
        )
    save_stats("download_manifest", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
