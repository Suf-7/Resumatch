# Resumatch

**AI-powered resume ↔ job matching.** AI semester project, FAST-NUCES Lahore (Track A: Application Development).

Job seekers upload a CV and get ranked job recommendations, an ATS-style score, a skill-gap report, and rewritten bullets. Recruiters upload a job description plus a batch of CVs and get a ranked, explained shortlist. The retrieval, ranking, and skill-extraction models are our own trained or fine-tuned models. LLM APIs are used only for labelling, synthetic data, and explanations.

| Deliverable | Scope | Status |
|---|---|---|
| 1 | Problem definition, data acquisition and preprocessing | Done: [`report/Resumatch_Deliverable1.pdf`](report/Resumatch_Deliverable1.pdf) |
| 2 | Model training (bi-encoder, cross-encoder, skill NER) | Not started |
| 3 | Evaluation and bias audit | Not started |
| 4 | App (Streamlit/Gradio) | Not started |

## Quick start

```bash
make setup                       # python3 -m venv .venv && pip install -r requirements.txt
cp .env.example .env             # optional: LLM_API_KEY for synthetic CVs / pair labelling
make data                        # Kaggle datasets (HF mirrors if no Kaggle creds) + ESCO
make scrape                      # Rozee.pk postings (robots.txt-aware, 2 s delay)
make clean features eda          # cleaning -> features/pairs/embeddings -> figures
make gold                        # sample 200 test pairs for manual labelling
make report                      # make_numbers.py + build_pdf.py (headless Chrome) -> PDF
                                 # or compile report/main.tex with LaTeX / Overleaf
```

Run any step by hand from `data_scripts/`, for example `../.venv/bin/python clean_jobs.py`.

## Data pipeline

```
download_kaggle.py ─┐
download_esco.py   ─┤                                     ┌─> eda.py ─> report/figures
scrape_jobs.py     ─┼─> clean_resumes.py ─┬─> build_features.py ─┼─> sample_gold_set.py
generate_synthetic_cvs.py ─┘  clean_jobs.py ─┘                     └─> label_pairs.py (LLM)
```

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
Each URL is checked against `robots.txt`, requests are spaced 2 s apart with a descriptive user agent, and HTTP 403 or 429 doubles the delay (up to 30 s). Only public posting text is stored, for academic use only. Requests are logged to `logs/scrape_rozee.log`.

### Gold set
`sample_gold_set.py sample` draws 200 pairs from the **test split only**. Two members label them independently, then `sample_gold_set.py score a1.csv a2.csv` reports Cohen's κ. This set is never used for training.

## Repository layout

```
data_scripts/   pipeline code (see above)
data/           raw/ interim/ processed/ gold/   (git-ignored, reproducible)
logs/           api_calls.jsonl, scrape log
notebooks/      exploration
report/         LaTeX source, figures, stats, make_numbers.py
app/            Phase 5 front end
```

## Team
- Muhammad Sufyan (24L-2601, BSDS-5B)
- [Member 2]
- [Member 3]
