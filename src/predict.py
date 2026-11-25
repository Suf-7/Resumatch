"""Score arbitrary (resume, job) texts with the trained models, and recommend open jobs.

    from predict import MatchScorer
    scorer = MatchScorer()                      # fine-tuned bi-encoder + MatchNet (task "weak")
    scorer.score(["resume text ..."], ["job description ..."])   # -> match probabilities
    scorer.recommend("resume text ...", k=3)    # top-k Rozee.pk postings + ATS breakdown
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from config import MODELS, PROCESSED  # adds data_scripts/ to sys.path
from ats_score import ats_score  # noqa: E402
from build_features import resume_years  # noqa: E402
from features import add_side_features, scalar_features
from models import bi_encoder
from models.matchnet import MatchNet
from skills import extract_skills, load_gazetteer  # noqa: E402


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
        self._gaz = None

    def _gazetteer(self):
        if self._gaz is None:
            self._gaz = load_gazetteer()
        return self._gaz

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

    def load_job_pool(self, allow_expired: bool = True) -> tuple[pd.DataFrame, bool]:
        """Return (pool, used_expired). Prefer currently open Rozee.pk postings."""
        live = pd.read_parquet(PROCESSED / "jobs_live.parquet")
        now = pd.Timestamp.now(tz="UTC")
        open_ = live[live["valid_through"] >= now]
        used_expired = False
        if len(open_) == 0:
            if not allow_expired:
                raise RuntimeError("No open postings in jobs_live.parquet. Run `make refresh`.")
            pool, used_expired = live.copy(), True
        else:
            pool = open_.copy()
        text = pd.read_parquet(PROCESSED.parent / "interim" / "jobs_clean.parquet").set_index("job_id")["text"]
        pool = pool.assign(job_text=text.reindex(pool["job_id"]).fillna("").to_numpy())
        pool = pool[pool["job_text"].str.len() > 40].reset_index(drop=True)
        return pool, used_expired

    def recommend(self, resume_text: str, k: int = 3, shortlist: int = 20,
                  allow_expired: bool = True) -> tuple[pd.DataFrame, dict]:
        """Retrieve `shortlist` postings by cosine, re-rank with MatchNet, return top k + meta.

        Each row includes match probability, ATS score (0-100), matched/missing skills and apply URL.
        """
        pool, used_expired = self.load_job_pool(allow_expired=allow_expired)
        if len(pool) == 0:
            raise RuntimeError("Job pool is empty.")
        r = self._emb([resume_text])[0]
        cand = pool.assign(cos=self._emb(pool["job_text"].tolist()) @ r).nlargest(
            min(shortlist, len(pool)), "cos").copy()
        feats = self.features([resume_text] * len(cand), cand["job_text"], cand["title"])
        matches = self.score([resume_text] * len(cand), cand["job_text"], cand["title"])
        years = resume_years(resume_text)
        res_skills = extract_skills(resume_text, self._gazetteer())
        rows = []
        for i, (_, job) in enumerate(cand.iterrows()):
            ats = ats_score(resume_text, res_skills, years, job["title"],
                            list(job["skills"]) if job["skills"] is not None else [],
                            job["years_required"] if pd.notna(job.get("years_required")) else np.nan)
            rows.append({
                "title": job["title"], "company": job["company"], "location": job["location"],
                "url": job["url"], "valid_through": job["valid_through"],
                "job_category": job.get("job_category", ""),
                "match": float(matches[i]),
                "ats_score": float(ats.score),
                "cosine": float(cand.iloc[i]["cos"]),
                "matched_skills": ats.matched_skills[:12],
                "missing_skills": ats.missing_skills[:12],
                "components": ats.components,
            })
        out = pd.DataFrame(rows).nlargest(k, "match").reset_index(drop=True)
        meta = {"n_pool": int(len(pool)), "used_expired": used_expired,
                "shortlist": int(len(cand)), "resume_years": years,
                "resume_skills": res_skills[:30]}
        return out, meta

    def rank_resumes(self, job_text: str, resumes: list[str], titles: list[str] | None = None,
                     k: int | None = None) -> pd.DataFrame:
        """Recruiter mode: rank many resumes against one job description."""
        if not resumes:
            return pd.DataFrame()
        title = (job_text.split()[:8] if titles is None else None)
        job_title = " ".join(title) if titles is None else None
        titles_list = titles if titles is not None else [job_title] * len(resumes)
        jobs = [job_text] * len(resumes)
        scores = self.score(resumes, jobs, titles_list)
        feats = self.features(resumes, jobs, titles_list)
        years = [resume_years(t) for t in resumes]
        gaz = self._gazetteer()
        job_skills = extract_skills(job_text, gaz)
        rows = []
        for i, text in enumerate(resumes):
            rs = extract_skills(text, gaz)
            ats = ats_score(text, rs, years[i], titles_list[i], job_skills, np.nan)
            rows.append({
                "resume_index": i,
                "preview": " ".join(text.split()[:28]) + ("…" if len(text.split()) > 28 else ""),
                "match": float(scores[i]),
                "ats_score": float(ats.score),
                "skill_cov": float(feats.iloc[i]["skill_cov"]),
                "matched_skills": ats.matched_skills[:10],
                "missing_skills": ats.missing_skills[:10],
                "n_words": len(text.split()),
            })
        out = pd.DataFrame(rows).sort_values("match", ascending=False).reset_index(drop=True)
        if k is not None:
            out = out.head(k)
        return out
