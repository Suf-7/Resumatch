# Resumatch app (Phase 5)

Streamlit front end for real-time inference with the fine-tuned bi-encoder and MatchNet.

## Run

From the **repository root**:

```bash
make setup          # once
make weights        # download models into models/ (if missing)
.venv/bin/streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501.

## Modes

- **Candidate**: upload a resume (PDF / TXT / DOCX) → top-3 Rozee.pk postings with apply link, MatchNet match %, ATS score (0–100), matched and missing skills.
- **Recruiter**: paste a job description + upload several resumes → ranked shortlist with the same scores.

If every posting in `data/processed/jobs_live.parquet` has expired, the app falls back to the full scraped Rozee.pk pool and shows a demo warning. Refresh with `make refresh`.
