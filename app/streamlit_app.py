"""Resumatch — Streamlit front end (Deliverable 3 / Phase 5).

Run from the repo root:
    .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import io
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(
    page_title="Resumatch",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ink + chalk + signal amber. Avoid purple / cream-terracotta AI defaults.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');

:root {
  --ink: #0e2439;
  --ink-soft: #3d556b;
  --line: #d7e0ea;
  --paper: #f4f7fa;
  --card: #ffffff;
  --signal: #1f6f8b;
  --amber: #c47a1a;
  --danger: #b42318;
  --ok: #1a7a4c;
}

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; color: var(--ink); }
.stApp { background:
  radial-gradient(900px 420px at 8% -8%, #cfe3f0 0%, transparent 55%),
  radial-gradient(700px 380px at 100% 0%, #e8eef4 0%, transparent 45%),
  linear-gradient(180deg, #eef3f7 0%, var(--paper) 38%, #eef2f6 100%); }
.block-container { padding-top: 1.6rem !important; padding-bottom: 3rem !important; max-width: 1040px; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }

.brand {
  display: flex; align-items: baseline; gap: 0.85rem; margin-bottom: 0.35rem;
  animation: rise 0.55s ease both;
}
.brand-mark {
  font-family: Fraunces, Georgia, serif; font-weight: 700; font-size: 2.75rem;
  letter-spacing: -0.03em; color: var(--ink); line-height: 1;
}
.brand-mark span { color: var(--signal); }
.brand-tag {
  font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ink-soft); font-weight: 500;
}
.lede {
  max-width: 36rem; color: var(--ink-soft); font-size: 1.08rem; line-height: 1.45;
  margin: 0.35rem 0 1.35rem; animation: rise 0.65s ease 0.05s both;
}
@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.panel {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1.15rem 1.25rem 1.25rem; margin-bottom: 1rem;
  box-shadow: 0 1px 0 rgba(14,36,57,0.04);
  animation: rise 0.5s ease both;
}
.panel h2 {
  font-family: Fraunces, Georgia, serif !important; font-size: 1.25rem !important;
  color: var(--ink) !important; margin: 0 0 0.2rem !important; font-weight: 700 !important;
}
.panel .hint { color: var(--ink-soft); font-size: 0.92rem; margin-bottom: 0.9rem; }

.job {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1.1rem 1.2rem 1.15rem; margin-bottom: 0.85rem;
  transition: border-color 0.2s ease, transform 0.2s ease;
  animation: rise 0.45s ease both;
}
.job:hover { border-color: #a8c0d4; transform: translateY(-1px); }
.job-top { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; }
.rank {
  flex: 0 0 auto; width: 2rem; height: 2rem; border-radius: 8px;
  background: var(--ink); color: #fff; display: inline-flex; align-items: center;
  justify-content: center; font-weight: 600; font-size: 0.95rem; margin-right: 0.65rem;
}
.job h3 {
  font-family: Fraunces, Georgia, serif !important; font-size: 1.2rem !important;
  margin: 0 0 0.25rem !important; color: var(--ink) !important; font-weight: 700 !important;
  display: inline;
}
.meta { color: var(--ink-soft); font-size: 0.9rem; margin: 0.15rem 0 0.75rem 2.65rem; }
.scores { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.2rem 0 0.85rem 2.65rem; }
.meter { min-width: 9.5rem; flex: 1; }
.meter-label {
  display: flex; justify-content: space-between; font-size: 0.78rem;
  color: var(--ink-soft); margin-bottom: 0.22rem; font-weight: 500;
}
.meter-label b { color: var(--ink); font-size: 0.9rem; }
.bar {
  height: 7px; border-radius: 99px; background: #e6edf3; overflow: hidden;
}
.bar > i {
  display: block; height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, #1f6f8b, #2f8fad);
}
.bar.ats > i { background: linear-gradient(90deg, #b86a12, #d4923a); }
.apply {
  display: inline-flex; align-items: center; gap: 0.35rem;
  margin: 0.15rem 0 0.85rem 2.65rem; padding: 0.45rem 0.85rem;
  background: var(--ink); color: #fff !important; text-decoration: none !important;
  border-radius: 8px; font-weight: 600; font-size: 0.88rem;
}
.apply:hover { background: #16354f; }
.skill-block { margin: 0.35rem 0 0.15rem 2.65rem; }
.skill-block .lbl {
  font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--ink-soft); font-weight: 600; margin-bottom: 0.3rem;
}
.chips { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.chips span {
  background: #eaf3f7; color: #1a4d5c; border: 1px solid #cfe0e8;
  padding: 0.18rem 0.5rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500;
}
.chips.miss span {
  background: #faf0ee; color: var(--danger); border-color: #f0d4cf;
}
.warn {
  background: #fff8eb; border: 1px solid #ecd9a8; color: #6b4e12;
  padding: 0.7rem 0.95rem; border-radius: 10px; margin: 0.4rem 0 1rem; font-size: 0.92rem;
}
.preview-box {
  background: #f7fafc; border: 1px dashed var(--line); border-radius: 10px;
  padding: 0.85rem 1rem; max-height: 280px; overflow: auto;
  font-size: 0.84rem; line-height: 1.45; color: var(--ink-soft); white-space: pre-wrap;
}
.empty {
  border: 1px dashed var(--line); border-radius: 12px; padding: 1.4rem 1.2rem;
  text-align: center; color: var(--ink-soft); background: rgba(255,255,255,0.55);
}
button[data-baseweb="tab"] {
  font-weight: 600 !important; font-size: 0.95rem !important; color: var(--ink-soft) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--ink) !important;
}
.stButton > button[kind="primary"] {
  background: var(--ink) !important; border: none !important;
  border-radius: 10px !important; font-weight: 600 !important;
  padding: 0.55rem 1rem !important;
}
.stButton > button[kind="secondary"] {
  border-radius: 10px !important; border-color: var(--line) !important;
}
.stTextArea textarea, .stTextInput input {
  border-radius: 10px !important; border-color: var(--line) !important;
  background: #fff !important; color: var(--ink) !important;
  caret-color: var(--ink) !important;
}
[data-testid="stFileUploader"] section {
  border-radius: 10px !important; border-color: var(--line) !important;
  background: #fff !important;
}
.foot {
  margin-top: 1.8rem; padding-top: 0.9rem; border-top: 1px solid var(--line);
  color: var(--ink-soft); font-size: 0.82rem;
}
.foot a { color: var(--signal); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SAMPLE_RESUME = """INFORMATION TECHNOLOGY SPECIALIST

Summary
Results-driven IT professional with 5+ years of experience in systems administration,
networking, and cloud operations. Strong background in Linux, Python scripting, and
supporting cross-functional teams.

Skills
Python, Linux, Networking, MySQL, AWS, Docker, Git, REST APIs, Troubleshooting,
Documentation, Stakeholder communication

Experience
Systems Administrator — Contoso Labs (2021–Present)
- Managed Linux servers and internal tooling for a 40-person engineering team
- Automated routine maintenance with Python scripts, cutting weekly ops time by ~30%
- Supported AWS deployments and monitored uptime for customer-facing services

Junior Network Engineer — Northwind (2019–2021)
- Configured switches and firewalls for multi-site offices
- Documented network topology and incident response runbooks

Education
BS Computer Science — National University of Computer and Emerging Sciences
"""


def read_upload(file) -> str:
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
        return "\n".join(
            t.text for t in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
            if t.text)
    return raw.decode("utf-8", errors="ignore")


@st.cache_resource(show_spinner="Loading MatchNet and the fine-tuned bi-encoder…")
def get_scorer():
    from predict import MatchScorer
    if not (ROOT / "models" / "bi_encoder").exists():
        st.error("Model weights missing. From the repo root run `make weights`, then reload.")
        st.stop()
    return MatchScorer("weak")


def esc(s) -> str:
    return html.escape(str(s))


def meter(label: str, value: float, kind: str = "") -> str:
    pct = max(0, min(100, float(value)))
    return (
        f'<div class="meter"><div class="meter-label"><span>{esc(label)}</span>'
        f'<b>{pct:.0f}</b></div>'
        f'<div class="bar {kind}"><i style="width:{pct:.0f}%"></i></div></div>'
    )


def chips(items, miss: bool = False) -> str:
    if not items:
        return "<em style='color:#8aa0b4;font-size:0.85rem'>none detected</em>"
    cls = "chips miss" if miss else "chips"
    return f'<div class="{cls}">' + "".join(f"<span>{esc(s)}</span>" for s in items) + "</div>"


def render_job_card(rank: int, row, delay_ms: int = 0) -> None:
    vt = row["valid_through"]
    vt_s = vt.strftime("%Y-%m-%d") if hasattr(vt, "strftime") else str(vt)[:10]
    st.markdown(f"""
    <div class="job" style="animation-delay:{delay_ms}ms">
      <div class="job-top">
        <div>
          <span class="rank">{rank}</span>
          <h3>{esc(row['title'])}</h3>
        </div>
      </div>
      <div class="meta">{esc(row['company'])} · {esc(row['location'])} · closes {esc(vt_s)}</div>
      <div class="scores">
        {meter("MatchNet", 100 * row['match'])}
        {meter("ATS score", row['ats_score'], "ats")}
      </div>
      <a class="apply" href="{esc(row['url'])}" target="_blank" rel="noopener">Apply on Rozee.pk →</a>
      <div class="skill-block"><div class="lbl">Matched skills</div>{chips(row.get('matched_skills') or [])}</div>
      <div class="skill-block"><div class="lbl">Missing skills</div>{chips(row.get('missing_skills') or [], miss=True)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_resume_card(rank: int, row, delay_ms: int = 0) -> None:
    st.markdown(f"""
    <div class="job" style="animation-delay:{delay_ms}ms">
      <div class="job-top">
        <div>
          <span class="rank">{rank}</span>
          <h3>{esc(row['file'])}</h3>
        </div>
      </div>
      <div class="meta">{esc(row['preview'])}</div>
      <div class="scores" style="margin-left:2.65rem">
        {meter("MatchNet", 100 * row['match'])}
        {meter("ATS score", row['ats_score'], "ats")}
      </div>
      <div class="skill-block"><div class="lbl">Matched</div>{chips(row.get('matched_skills') or [])}</div>
      <div class="skill-block"><div class="lbl">Missing</div>{chips(row.get('missing_skills') or [], miss=True)}</div>
    </div>
    """, unsafe_allow_html=True)


def candidate_mode(scorer) -> None:
    st.markdown(
        '<div class="panel"><h2>Find jobs for a resume</h2>'
        '<div class="hint">We shortlist Rozee.pk postings with the bi-encoder, then re-rank '
        "the top 20 with MatchNet. You get apply links, scores, and skill gaps.</div></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.12, 1], gap="large")
    with left:
        up = st.file_uploader("Upload resume", type=["pdf", "txt", "docx", "md"],
                             help="PDF, TXT, or DOCX")
        if "paste" not in st.session_state:
            st.session_state.paste = ""
        pasted = st.text_area("Or paste resume text", height=210, key="paste",
                              placeholder="Paste the full resume here…")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.button("Use sample resume", use_container_width=True,
                      on_click=lambda: st.session_state.update(paste=SAMPLE_RESUME))
        with c2:
            k = st.selectbox("Matches to show", [3, 1, 2, 4, 5], index=0)
        go = st.button("Find matches", type="primary", use_container_width=True)

    text = ""
    if up is not None:
        try:
            text = read_upload(up).strip()
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
    if not text:
        text = (pasted or "").strip()

    with right:
        if text:
            st.markdown("**Resume preview**")
            st.markdown(
                f'<div class="preview-box">{esc(text[:1600])}{"…" if len(text) > 1600 else ""}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{len(text.split())} words")
        else:
            st.markdown(
                '<div class="empty">Upload or paste a resume to see a preview here.</div>',
                unsafe_allow_html=True,
            )

    if not go:
        return
    if len(text.split()) < 40:
        st.warning("Resume looks too short. Paste at least a few paragraphs.")
        return

    with st.spinner("Embedding the resume and ranking postings…"):
        try:
            hits, meta = scorer.recommend(text, k=int(k), shortlist=20, allow_expired=True)
        except Exception as exc:
            st.error(str(exc))
            return

    if meta["used_expired"]:
        st.markdown(
            f'<div class="warn">No currently open postings in the local pool '
            f"(n = {meta['n_pool']}). Showing the scraped Rozee.pk pool as a <b>demo</b> — "
            f"apply links may have expired. Run <code>make refresh</code> when you can.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success(f"Ranked against {meta['n_pool']} open postings.")

    st.markdown(f"### Top {len(hits)} matches")
    for i, row in hits.iterrows():
        render_job_card(i + 1, row, delay_ms=80 * i)


def recruiter_mode(scorer) -> None:
    st.markdown(
        '<div class="panel"><h2>Rank candidates for a job</h2>'
        '<div class="hint">Paste a job description and upload several resumes. '
        "MatchNet scores the batch with the same ATS breakdown.</div></div>",
        unsafe_allow_html=True,
    )
    jd = st.text_area("Job description", height=170,
                      placeholder="Paste the full job description…")
    files = st.file_uploader(
        "Candidate resumes", type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )
    k = st.slider("Shortlist size", 1, 20, 5, key="rec_k")
    if not st.button("Rank candidates", type="primary", use_container_width=True):
        return
    if len((jd or "").split()) < 20:
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
    st.markdown(f"### Shortlist ({len(ranked)})")
    for i, row in ranked.iterrows():
        render_resume_card(i + 1, row, delay_ms=70 * i)


def main() -> None:
    st.markdown(
        '<div class="brand"><div class="brand-mark">Resu<span>match</span></div>'
        '<div class="brand-tag">Resume ↔ job matching</div></div>'
        '<p class="lede">Upload a resume. Get the three best jobs you can apply to — '
        "with a MatchNet score, an ATS score, and the skills you are missing.</p>",
        unsafe_allow_html=True,
    )

    scorer = get_scorer()
    tab_cand, tab_rec = st.tabs(["Candidate", "Recruiter"])
    with tab_cand:
        candidate_mode(scorer)
    with tab_rec:
        recruiter_mode(scorer)

    st.markdown(
        '<div class="foot">Fine-tuned MiniLM bi-encoder + MatchNet · '
        '<a href="https://huggingface.co/Suf7/resumatch-models" target="_blank">Weights</a> · '
        '<a href="https://github.com/Suf-7/Resumatch" target="_blank">GitHub</a></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
