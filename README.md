# Resumatch

**AI-powered resume ↔ job matching.** AI semester project, FAST-NUCES Lahore (Track A: Application Development).

## Deliverable 2 submission (due 10 November 2026)

Hand in these three items from the brief:

| What the brief asks | Where it is |
|---|---|
| GitHub repository (`/src` or `/models`, plus `requirements.txt`) | **https://github.com/Suf-7/Resumatch** — code in [`src/`](src/), models in [`src/models/`](src/models/), [`requirements.txt`](requirements.txt) |
| LaTeX PDF (Phases 3–4: architecture, curves, contribution table) | **[`report/Resumatch_Deliverable2.pdf`](report/Resumatch_Deliverable2.pdf)** |
| Model weights (not in Git; Hugging Face or Drive link in this README) | **https://huggingface.co/Suf7/resumatch-models** (fallback: [GitHub Release](https://github.com/Suf-7/Resumatch/releases/tag/deliverable2-weights)) |

Reproduce locally: `make setup && make weights`. Train from scratch with `make train` (about 1.5 h on an M1).

A job seeker uploads a CV and gets the **top 3 open job postings** they can apply to. Each one comes with an apply link, an ATS-style score for that posting, matched and missing skills, and suggested bullet rewrites. Recruiters upload a job description plus a batch of CVs and get a ranked, explained shortlist. The retrieval, ranking, and skill-extraction models are our own trained or fine-tuned models. LLM APIs are used only for labelling, synthetic data, and explanations.

| Deliverable | Scope | Status |
|---|---|---|
| 1 | Problem definition, data acquisition and preprocessing | Done: [`report/Resumatch_Deliverable1.pdf`](report/Resumatch_Deliverable1.pdf) |
| 2 | Model architecture, training, tuning, evaluation, error analysis, bias check | Done: [`report/Resumatch_Deliverable2.pdf`](report/Resumatch_Deliverable2.pdf) |
| 3 | App (Streamlit/Gradio) | Not started |

## Current state (2026-10-24)

| Item | Count |
|---|---|
| Resumes (cleaned) | 2,632 of 3,446 |
| Job postings (cleaned) | 28,088: 27,740 LinkedIn (training only) + 348 Rozee.pk |
| Open-jobs pool | 348 Rozee.pk postings with apply links; 340 close within 7 days |
| Training pairs | 15,755 (7,859 positive, 5,264 hard negative, 2,632 random negative) |
| Baseline ATS score (AUC) | 0.73 vs. random negatives, 0.52 vs. hard negatives |

**To do**
- [ ] Retry `make refresh` (Rozee.pk rate-limited the scraper on 2026-10-24); the pool empties within a week without it.
- [ ] Label the 200 gold pairs in `data/gold/` (Muaaz and Hanan, independently), then run `sample_gold_set.py score`.
- [ ] Add `LLM_API_KEY` to `.env`, then `make synthetic label`.
- [x] Phase 3–4: fine-tune the bi-encoder, train MatchNet, evaluate, error analysis, bias check (see Deliverable 2 below).

Rebuilding the PDF (`make report`) is safe at any time. Don't run `make live` on its own before submitting: it drops postings that have closed since 2026-10-24, so the report numbers would shrink.

## Quick start

```bash
make setup                       # python3 -m venv .venv && pip install -r requirements.txt
cp .env.example .env             # optional: LLM_API_KEY for synthetic CVs / pair labelling
make data                        # Kaggle datasets (HF mirrors if no Kaggle creds) + ESCO
make scrape                      # Rozee.pk postings (robots.txt-aware, 2 s delay)
make clean features eda          # cleaning -> features/pairs/embeddings -> figures
make live ats                    # open-jobs pool + baseline ATS score check
make refresh                     # every few days: new Rozee.pk postings -> rebuilt pool
make gold                        # sample 200 test pairs for manual labelling
make report                      # make_numbers.py + build_pdf.py (headless Chrome) -> PDF
                                 # or compile report/main.tex with LaTeX / Overleaf
```

Run any step by hand from `data_scripts/`, for example `../.venv/bin/python clean_jobs.py`.

## Deliverable 2: models and evaluation

Two PyTorch models, both in `src/models/`:
- **Bi-encoder** (`bi_encoder.py`): `all-MiniLM-L6-v2`, fine-tuned with a multiple-negatives ranking loss (in-batch negatives plus one mined hard negative), AdamW with warm-up. It embeds resumes and postings in one space and shortlists the 20 closest open postings.
- **MatchNet** (`matchnet.py`): an MLP over `[u, v, |u-v|, u*v]` plus ten ATS features (skill coverage, years fit, title match, TF-IDF cosine and others). Trained with class-weighted BCE, AdamW and early stopping on validation AUC. It re-scores the shortlist; the top 3 are returned with their ATS score and apply link (`src/predict.py`, `MatchScorer.recommend`).

Two evaluation tasks: **A**, our weakly labelled pairs (positive = same occupation), and **B**, 8,000 external human-labelled pairs (`cnamuangtoun/resume-job-description-fit`; Potential or Good Fit = positive). Test results:

| Model | Task A AUC-ROC | Task A F1 | Task B AUC-ROC | Task B F1 |
|---|---|---|---|---|
| ATS score (rules) | 0.588 | 0.665 | 0.536 | 0.678 |
| MiniLM cosine (pretrained) | 0.661 | 0.686 | 0.579 | 0.666 |
| Bi-encoder cosine (fine-tuned) | 0.790 | 0.739 | 0.633 | 0.695 |
| Gradient boosting (best baseline) | 0.832 | 0.764 | 0.695 | 0.708 |
| **MatchNet** | **0.862** | **0.784** | **0.768** | **0.737** |

Top-3 recommendation (share of recommended postings in the resume's occupation): 0.417 pretrained, 0.634 fine-tuned, 0.641 with MatchNet re-ranking, 0.050 random. Full numbers, tuning tables and curves are in `report/stats/*.json` and the PDF.

### Running it

```bash
make weights        # download trained weights (or: make train, about 1.5 h on an M1)
make evaluate       # test metrics, top-3 retrieval, figures
make error-analysis
make bias-check
make report2        # contribution table + tables + report/Resumatch_Deliverable2.pdf
```

`make train` runs `train_bi_encoder.py` (5-run grid, best model saved to `models/bi_encoder/`), `pair_matrices.py` (embeddings and features for every pair) and `train_matchnet.py` (36-run grid per task, `models/matchnet_{weak,human}.pt`).

Error analysis (Task A): hard negatives are wrong 0.49 of the time vs. 0.12 for positives and 0.11 for random negatives. Worst categories by F1: Automobile, Banking, BPO; best: Advocate, HR, Business Development. On human labels the error rate is 0.43 for No Fit, 0.23 for Potential Fit, 0.13 for Good Fit.

Bias check (300 counterfactual test pairs): at most 1.7% of accept/reject decisions flip when gender, name origin or university is swapped. Mean score shifts stay below 0.02.

### Model weights
Weights are kept out of Git (`models/bi_encoder/` is 88 MB, plus `models/matchnet_*.pt`). Download them with `make weights`, which tries the Hugging Face Hub ([Suf7/resumatch-models](https://huggingface.co/Suf7/resumatch-models)) and falls back to the GitHub Release [deliverable2-weights](https://github.com/Suf-7/Resumatch/releases/tag/deliverable2-weights).

## Data pipeline

```
download_kaggle.py ─┐
download_esco.py   ─┤                                     ┌─> eda.py ─> report/figures
scrape_jobs.py     ─┼─> clean_resumes.py ─┬─> build_features.py ─┼─> sample_gold_set.py
generate_synthetic_cvs.py ─┘  clean_jobs.py ─┘                     ├─> label_pairs.py (LLM)
                                                                    ├─> live_pool.py (open jobs)
                                                                    └─> ats_score.py (baseline)
```

### Open-jobs pool (`live_pool.py`)
Users are only matched against postings they can still apply to: Rozee.pk postings with an apply URL and a closing date that hasn't passed. They're saved to `data/processed/jobs_live.parquet` with their skills, required years and embedding row. LinkedIn postings closed in 2023–24 and are used for training only. Rozee.pk postings close within about a week, so run `make refresh` every few days. On 2026-10-24, Rozee.pk answered our `robots.txt` request with HTTP 403, most likely a temporary rate limit after the first run. The scraper made no requests that day and will retry later.

### Per-job ATS score (`ats_score.py`)
The score runs from 0 to 100 and is a weighted mean of four components:
- skill coverage (0.50)
- experience fit (0.20)
- title match (0.15)
- resume structure (0.15)

Components a posting doesn't specify are skipped, and the remaining weights are rescaled. The function also returns matched and missing skills. `ats_score.py evaluate` checks the baseline on the test pairs; results are in `report/stats/ats_baseline.json`.

| Source | Script | Raw → clean |
|---|---|---|
| Kaggle *Resume Dataset* (snehaanbhawal) | `download_kaggle.py` | 2,484 → see `report/stats/clean_resumes.json` |
| Kaggle *Updated Resume Dataset* (gauravduttakiit) | `download_kaggle.py` | 962 (796 exact duplicates) |
| Kaggle *LinkedIn Job Postings* (arshkon) | `download_kaggle.py` | 33,246 → see `report/stats/clean_jobs.json` |
| Rozee.pk (Pakistan) | `scrape_jobs.py` | see `report/stats/scrape_rozee.json` |
| ESCO v1.1.2 skills | `download_esco.py` | 13,896 skills, 99k alt labels |
| Resume–JD fit pairs (HF) | `download_kaggle.py` | 6,241 + 1,759, held out for evaluation |

Every number in the report is regenerated from `report/stats/*.json`.

### Cleaning (`clean_resumes.py`, `clean_jobs.py`)
Each step drops or masks records, and its count is logged:
1. Drop records with empty text; fill missing categorical fields with `unknown`.
2. Repair mis-encoded text and strip HTML, URLs, and EEO boilerplate.
3. Filter out non-English text.
4. Remove exact duplicates (SHA-1 hash) and near-duplicates (TF-IDF cosine > 0.95).
5. Handle length outliers: flag below the 1st or above the 99th percentile (IQR as a cross-check), drop junk, and cap long texts.
6. Mask PII (email, phone, URL, address).

### Features (`build_features.py`, `skills.py`)
- ESCO skill gazetteer (variants → canonical skill).
- TF-IDF features (1–2-grams).
- Multi-hot skill vectors.
- Years of experience extracted by regex.
- Log-transformed, standardized numeric features.
- 384-d MiniLM embeddings (`--embeddings`).
- Train/val/test split **by resume and by posting**, with every fitted transformer fitted on train only.
- Pairs: same-category positives, hard negatives (similar text, wrong category), and random negatives.

### LLM usage and logging
`llm_client.py` works with any OpenAI-compatible endpoint (OpenAI, Groq, Gemini; see `.env.example`). Every call is appended to `logs/api_calls.jsonl` with prompt, model, response, usage, and timestamp, and responses are cached. Without a key, the scripts run in dry-run mode and log the prompts only.

### Scraping policy
Each URL is checked against `robots.txt`; if `robots.txt` itself can't be fetched, the whole site is treated as disallowed. Requests are spaced 2 s apart with a descriptive user agent, and HTTP 403 or 429 doubles the delay (up to 30 s). Only public posting text is stored, for academic use only. Requests are logged to `logs/scrape_rozee.log`.

### Gold set
`sample_gold_set.py sample` draws 200 pairs from the **test split only**. Two members label them independently, then `sample_gold_set.py score a1.csv a2.csv` reports Cohen's κ. This set is never used for training.

## Repository layout

```
data_scripts/   pipeline code (see above)
data/           raw/ interim/ processed/ gold/   (git-ignored, reproducible)
logs/           api_calls.jsonl, scrape log
src/            Deliverable 2: models/, training, evaluation, error analysis, bias check
models/         trained weights (git-ignored; Hugging Face)
notebooks/      exploration
report/         LaTeX sources (main.tex, deliverable2.tex), figures, stats, tables, PDF build
app/            Phase 5 front end
```

## Team
- Muhammad Sufyan (24L-2601, BSDS-5B)
- Muaaz Fahad (24L-2563, BSDS-5B)
- Hanan Ishaque (24L-2537, BSDS-5B)
