"""Resumatch — Streamlit front end (Deliverable 3 / Phase 5).

Run from the repo root:
    .venv/bin/streamlit run app/streamlit_app.py

Candidate mode: upload a resume → top-3 Rozee.pk postings with apply links, MatchNet
score, ATS score and skill gaps.
Recruiter mode: paste a job description + upload several resumes → ranked shortlist.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Resumatch", page_icon="R", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: "Source Sans 3", sans-serif; }
.block-container { padding-top: 1.4rem; max-width: 1100px; }
h1, h2, h3 { font-family: Fraunces, Georgia, serif !important; color: #1a365d !important; }
div[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% -10%, #e8f1fb 0%, transparent 55%),
              radial-gradient(900px 500px at 100% 0%, #f7fafc 0%, transparent 50%),
              #f8fafc;
}
.hero {
  padding: 1.2rem 0 0.4rem;
}
.hero h1 { font-size: 2.6rem; margin-bottom: 0.15rem; letter-spacing: -0.02em; }
.hero p { color: #4a5568; font-size: 1.05rem; margin-top: 0; }
.card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
  padding: 1rem 1.15rem; margin-bottom: 0.85rem;
}
.card h3 { margin: 0 0 0.35rem; font-size: 1.15rem; }
.meta { color: #718096; font-size: 0.92rem; margin-bottom: 0.55rem; }
.score { display: inline-block; background: #ebf4ff; color: #2b6cb0;
  font-weight: 600; padding: 0.15rem 0.55rem; border-radius: 6px; margin-right: 0.4rem; }
.score.ats { background: #fffaf0; color: #c05621; }
.skills span {
  display: inline-block; background: #edf2f7; color: #2d3748;
  padding: 0.12rem 0.45rem; border-radius: 4px; margin: 0.15rem 0.2rem 0 0; font-size: 0.85rem;
}
.skills.missing span { background: #fff5f5; color: #c53030; }
.warn { background: #fffbeb; border: 1px solid #f6e05e; color: #744210;
  padding: 0.65rem 0.9rem; border-radius: 8px; margin: 0.6rem 0 1rem; }
a.apply { color: #2b6cb0; font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def read_upload(file) -> str:
    """Extract plain text from an uploaded .pdf / .txt / .docx file."""
    name = (file.name or "").lower()
    raw = file.read()
    if name.endswith(".txt") or name.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            xml = zf.read("word/document.xml")
        root = ET.fromstring(xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return "\n".join(t.text for t in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                         if t.text)
    return raw.decode("utf-8", errors="ignore")


@st.cache_resource(show_spinner="Loading MatchNet and the fine-tuned bi-encoder…")
def get_scorer():
    from predict import MatchScorer
    weights = ROOT / "models" / "bi_encoder"
    if not weights.exists():
        st.error("Model weights missing. From the repo root run `make weights`, then reload.")
        st.stop()
    return MatchScorer("weak")


def render_job_card(rank: int, row) -> None:
    vt = row["valid_through"]
    vt_s = vt.strftime("%Y-%m-%d") if hasattr(vt, "strftime") else str(vt)[:10]
    matched = row.get("matched_skills") or []
    missing = row.get("missing_skills") or []
    matched_html = " ".join(f"<span>{s}</span>" for s in matched) or "<em>none detected</em>"
    missing_html = " ".join(f"<span>{s}</span>" for s in missing) or "<em>none</em>"
    st.markdown(f"""
    <div class="card">
      <h3>#{rank} · {row['title']}</h3>
      <div class="meta">{row['company']} · {row['location']} · closes {vt_s}</div>
      <div>
        <span class="score">Match {100 * row['match']:.0f}%</span>
        <span class="score ats">ATS {row['ats_score']:.0f}/100</span>
        <a class="apply" href="{row['url']}" target="_blank" rel="noopener">Apply on Rozee.pk →</a>
      </div>
      <p style="margin:0.7rem 0 0.2rem;font-size:0.9rem;color:#4a5568"><b>Matched skills</b></p>
      <div class="skills">{matched_html}</div>
      <p style="margin:0.55rem 0 0.2rem;font-size:0.9rem;color:#4a5568"><b>Missing skills</b></p>
      <div class="skills missing">{missing_html}</div>
    </div>
    """, unsafe_allow_html=True)


def candidate_mode(scorer) -> None:
    st.subheader("Candidate mode")
    st.caption("Upload a resume. We shortlist open (or demo) Rozee.pk postings with the bi-encoder, "
               "then re-rank the top 20 with MatchNet.")
    left, right = st.columns([1.15, 1])
    with left:
        up = st.file_uploader("Resume (.pdf, .txt, .docx)", type=["pdf", "txt", "docx", "md"])
        pasted = st.text_area("Or paste resume text", height=220,
                              placeholder="Paste the full resume here…")
        k = st.slider("How many matches to show", 1, 5, 3)
        go = st.button("Find matches", type="primary", use_container_width=True)
    text = ""
    if up is not None:
        try:
            text = read_upload(up).strip()
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
    if not text:
        text = pasted.strip()
    with right:
        if text:
            st.markdown("**Resume preview**")
            st.text(text[:1200] + ("…" if len(text) > 1200 else ""))
        else:
            st.info("Upload or paste a resume to start.")
    if not go:
        return
    if len(text.split()) < 40:
        st.warning("Resume looks too short. Paste at least a few paragraphs.")
        return
    with st.spinner("Embedding the resume and ranking postings…"):
        try:
            hits, meta = scorer.recommend(text, k=k, shortlist=20, allow_expired=True)
        except Exception as exc:
            st.error(str(exc))
            return
    if meta["used_expired"]:
        st.markdown(
            f'<div class="warn">No currently open Rozee.pk postings in the local pool '
            f"(n = {meta['n_pool']}). Showing the full scraped pool as a <b>demo</b> — "
            f"apply links may have expired. Run <code>make refresh</code> to update.</div>",
            unsafe_allow_html=True)
    else:
        st.success(f"Ranked against {meta['n_pool']} open postings.")
    for i, row in hits.iterrows():
        render_job_card(i + 1, row)


def recruiter_mode(scorer) -> None:
    st.subheader("Recruiter mode")
    st.caption("Paste a job description and upload several resumes. MatchNet ranks the batch.")
    jd = st.text_area("Job description", height=180,
                      placeholder="Paste the full job description…")
    files = st.file_uploader("Candidate resumes", type=["pdf", "txt", "docx", "md"],
                             accept_multiple_files=True)
    k = st.slider("Shortlist size", 1, 20, 5, key="rec_k")
    if not st.button("Rank candidates", type="primary"):
        return
    if len(jd.split()) < 20:
        st.warning("Job description looks too short.")
        return
    if not files:
        st.warning("Upload at least one resume.")
        return
    resumes, names = [], []
    for f in files:
        try:
            resumes.append(read_upload(f).strip())
            names.append(f.name)
        except Exception as exc:
            st.error(f"{f.name}: {exc}")
            return
    with st.spinner(f"Scoring {len(resumes)} resumes…"):
        ranked = scorer.rank_resumes(jd, resumes, k=k)
    ranked.insert(0, "file", [names[i] for i in ranked["resume_index"]])
    for i, row in ranked.iterrows():
        matched = " ".join(f"<span>{s}</span>" for s in (row["matched_skills"] or [])) or "—"
        missing = " ".join(f"<span>{s}</span>" for s in (row["missing_skills"] or [])) or "—"
        st.markdown(f"""
        <div class="card">
          <h3>#{i + 1} · {row['file']}</h3>
          <div class="meta">{row['preview']}</div>
          <div>
            <span class="score">Match {100 * row['match']:.0f}%</span>
            <span class="score ats">ATS {row['ats_score']:.0f}/100</span>
          </div>
          <p style="margin:0.7rem 0 0.2rem;font-size:0.9rem;color:#4a5568"><b>Matched</b></p>
          <div class="skills">{matched}</div>
          <p style="margin:0.55rem 0 0.2rem;font-size:0.9rem;color:#4a5568"><b>Missing</b></p>
          <div class="skills missing">{missing}</div>
        </div>
        """, unsafe_allow_html=True)


def main() -> None:
    st.markdown('<div class="hero"><h1>Resumatch</h1>'
                '<p>Upload a resume. Get the three best jobs you can still apply to — '
                'with a MatchNet score, an ATS score, and the skills you are missing.</p></div>',
                unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Mode")
        mode = st.radio("Choose a mode", ["Candidate", "Recruiter"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**Models**  \nFine-tuned MiniLM bi-encoder + MatchNet  \n"
                    "[Weights on Hugging Face](https://huggingface.co/Suf7/resumatch-models)")
        st.markdown("**Code**  \n[github.com/Suf-7/Resumatch](https://github.com/Suf-7/Resumatch)")
    scorer = get_scorer()
    if mode == "Candidate":
        candidate_mode(scorer)
    else:
        recruiter_mode(scorer)


if __name__ == "__main__":
    main()
