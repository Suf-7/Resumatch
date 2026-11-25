# Resumatch

**AI-powered resume ↔ job matching.** AI semester project, FAST-NUCES Lahore (Track A: Application Development).

A job seeker uploads a CV and gets the **top 3 open job postings** they can apply to — each with an apply link, a MatchNet score, an ATS score, and matched/missing skills. Recruiters paste a job description plus several CVs and get a ranked shortlist. Models are trained by us; LLM APIs are optional (labelling / synthetic data only).

## Deliverable 3 submission (due 17 November 2026)

| What the brief asks | Where it is |
|---|---|
| Production-ready GitHub (`requirements.txt`, README, source) | **https://github.com/Suf-7/Resumatch** |
| Final LaTeX PDF (Phases 1–6 merged) | **[`report/Resumatch_Deliverable3.pdf`](report/Resumatch_Deliverable3.pdf)** |
| Hosted / runnable app | Streamlit: `make app` → http://localhost:8501 (`app/streamlit_app.py`) |
| Model weights (not in Git) | **https://huggingface.co/Suf7/resumatch-models** |

| Deliverable | Scope | Status |
|---|---|---|
| 1 | Problem definition, data acquisition and preprocessing | Done: [`report/Resumatch_Deliverable1.pdf`](report/Resumatch_Deliverable1.pdf) |
| 2 | Model architecture, training, tuning, evaluation, error analysis, bias check | Done: [`report/Resumatch_Deliverable2.pdf`](report/Resumatch_Deliverable2.pdf) |
| 3 | Streamlit app + final merged technical report | Done: [`report/Resumatch_Deliverable3.pdf`](report/Resumatch_Deliverable3.pdf) |

## Quick start (app)

```bash
make setup                       # venv + requirements (includes streamlit)
make weights                     # download bi-encoder + MatchNet from Hugging Face
make app                         # opens http://localhost:8501
```

Candidate mode: upload a resume (PDF/TXT/DOCX) → top-3 Rozee.pk matches.  
Recruiter mode: paste a JD + upload several resumes → ranked shortlist.

If every posting in the local open-jobs pool has expired, the app uses the full scraped Rozee.pk pool as a **demo** and shows a warning. Refresh with `make refresh` when Rozee.pk allows crawling again.

## Reproduce training & reports

```bash
make data scrape clean features live ats eda   # Deliverable 1 pipeline
make train                                     # bi-encoder grid + MatchNet (~1.5 h on M1)
make evaluate error-analysis bias-check
make report3                                   # final PDF
```

Weights stay out of Git. `make weights` tries Hugging Face first, then the [GitHub Release](https://github.com/Suf-7/Resumatch/releases/tag/deliverable2-weights).

## Models (summary)

Two PyTorch stages in `src/models/`:

1. **Bi-encoder** — fine-tuned `all-MiniLM-L6-v2` (multiple-negatives ranking + hard negatives). Shortlists the 20 closest postings.
2. **MatchNet** — MLP over `[u, v, |u−v|, u*v]` plus ten ATS features. Re-ranks; top 3 returned with scores and skill gaps (`src/predict.py`).

| Model | Task A AUC | Task A F1 | Task B AUC | Task B F1 |
|---|---|---|---|---|
| ATS score (rules) | 0.588 | 0.665 | 0.536 | 0.678 |
| MiniLM (pretrained) | 0.661 | 0.686 | 0.579 | 0.666 |
| Bi-encoder (fine-tuned) | 0.790 | 0.739 | 0.633 | 0.695 |
| Gradient boosting | 0.832 | 0.764 | 0.695 | 0.708 |
| **MatchNet** | **0.862** | **0.784** | **0.768** | **0.737** |

Task A = our weakly labelled occupation pairs. Task B = external human-labelled fit pairs. Bias check: ≤1.7% of decisions flip when gender/origin/university is swapped.

## Repository layout

```
app/            Streamlit front end (Phase 5)
src/            models/, training, evaluation, error analysis, bias check, predict.py
data_scripts/   download, scrape, clean, features, EDA, live pool, ATS baseline
data/           raw/ interim/ processed/ gold/   (git-ignored)
models/         trained weights (git-ignored; Hugging Face)
report/         LaTeX (main.tex, deliverable2.tex, final.tex), figures, stats, PDFs
```

## Team
- Muhammad Sufyan (24L-2601, BSDS-5B)
- Muaaz Fahad (24L-2563, BSDS-5B)
- Hanan Ishaque (24L-2537, BSDS-5B)
