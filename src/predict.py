"""Score arbitrary (resume, job) texts with the trained models, and recommend open jobs.

    from predict import MatchScorer
    scorer = MatchScorer()                      # fine-tuned bi-encoder + MatchNet (task "weak")
    scorer.score(["resume text ..."], ["job description ..."])   # -> match probabilities
    scorer.recommend("resume text ...", k=3)    # top-k open Rozee.pk postings + ATS breakdown
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from config import MODELS, PROCESSED
from features import add_side_features, scalar_features
from models import bi_encoder
from models.matchnet import MatchNet


class MatchScorer:
    def __init__(self, task: str = "weak", encoder_path=None):
        self.encoder = bi_encoder.load_encoder(encoder_path or MODELS / "bi_encoder")
        ck = torch.load(MODELS / f"matchnet_{task}.pt", weights_only=False)
        self.net = MatchNet(ck["emb_dim"], ck["n_scalar"], **ck["config"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.mean = ck["scaler_mean"].astype(np.float32)
        self.scale = ck["scaler_scale"].astype(np.float32)
        self._cache: dict[str, np.ndarray] = {}

    def _emb(self, texts) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if new:
            self._cache.update(zip(new, bi_encoder.encode(self.encoder, new)))
        return np.stack([self._cache[t] for t in texts]).astype(np.float32)

    def features(self, resumes, jobs, titles=None) -> pd.DataFrame:
        df = pd.DataFrame({"resume_text": list(resumes), "job_text": list(jobs)})
        df["job_title"] = list(titles) if titles is not None else df["job_text"].str.split().str[:8].str.join(" ")
        return scalar_features(add_side_features(df))

    @torch.no_grad()
    def score(self, resumes, jobs, titles=None) -> np.ndarray:
        x = (self.features(resumes, jobs, titles).to_numpy() - self.mean) / self.scale
        u, v = self._emb(list(resumes)), self._emb(list(jobs))
        logit = self.net(torch.from_numpy(u), torch.from_numpy(v), torch.from_numpy(x.astype(np.float32)))
        return torch.sigmoid(logit).numpy()

    def recommend(self, resume_text: str, k: int = 3, shortlist: int = 20) -> pd.DataFrame:
        """Retrieve `shortlist` open postings by cosine, rerank with MatchNet, return the top k."""
        live = pd.read_parquet(PROCESSED / "jobs_live.parquet")
        live = live[live["valid_through"] >= pd.Timestamp.now(tz="UTC")]
        text = pd.read_parquet(PROCESSED.parent / "interim" / "jobs_clean.parquet").set_index("job_id")["text"]
        live = live.assign(job_text=text.loc[live["job_id"]].to_numpy())
        r = self._emb([resume_text])[0]
        cand = live.assign(cos=self._emb(live["job_text"].tolist()) @ r).nlargest(shortlist, "cos")
        feats = self.features([resume_text] * len(cand), cand["job_text"], cand["title"])
        cand = cand.assign(match=self.score([resume_text] * len(cand), cand["job_text"], cand["title"]),
                           ats_score=(100 * feats["ats_score"]).round(1).to_numpy())
        return cand.nlargest(k, "match")[["title", "company", "location", "url", "valid_through",
                                          "match", "ats_score"]]
