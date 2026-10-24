"""Shared paths and text utilities for the Resumatch data pipeline."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
LOGS = ROOT / "logs"
STATS = ROOT / "report" / "stats"
FIGURES = ROOT / "report" / "figures"

for _d in (RAW, INTERIM, PROCESSED, LOGS, STATS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42


def save_stats(name: str, stats: dict) -> None:
    with open(STATS / f"{name}.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Noise removal
# --------------------------------------------------------------------------- #
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(nbsp|amp|lt|gt|quot|#39|#\d+);")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,4}\)?[\s.-]?)\d{3}[\s.-]?\d{3,4}(?!\w)"
)
_STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|"
    r"Lane|Ln|Drive|Dr|Court|Ct|Way|Block|Sector)\b\.?"
)
_ZIP = re.compile(r"\b[A-Z]{2}\s\d{5}(?:-\d{4})?\b")
_NON_PRINTABLE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")
_MULTISPACE = re.compile(r"[ \t\u00A0]+")
_MULTINEWLINE = re.compile(r"\n{3,}")

BOILERPLATE = [
    r"(is an |are an |we are an )?equal (employment )?opportunity employer[^.\n]*[.\n]?",
    r"all qualified applicants will receive consideration[^.\n]*[.\n]?",
    r"without regard to (race|age|sex|religion)[^.\n]*[.\n]?",
    r"e-?verify[^.\n]*[.\n]?",
    r"reasonable accommodation[^.\n]*[.\n]?",
    r"click (here|apply)[^.\n]*[.\n]?",
]
_BOILERPLATE = re.compile("|".join(BOILERPLATE), re.I)


def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>|</p>|</li>|</div>", "\n", text, flags=re.I)
    text = _HTML_TAG.sub(" ", text)
    return _HTML_ENTITY.sub(" ", text)


_MOJIBAKE = re.compile("[\u00c2\u00c3\u00e2][\u0080-\u00bf\u20ac\u2122\u0153\u0161\u017e\u201a-\u201e\u2020-\u2026\u2030\u2039\u203a]")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1/CP1252 (e.g. 'NaÃ¯ve' -> 'Naïve')."""
    if not _MOJIBAKE.search(text):
        return text
    for enc in ("cp1252", "latin-1"):
        try:
            return text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def clean_text(text: str) -> str:
    """Noise removal that keeps casing and punctuation (transformer-friendly)."""
    if not isinstance(text, str):
        return ""
    text = fix_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = strip_html(text)
    text = _URL.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = _NON_PRINTABLE.sub(" ", text)
    text = text.replace("\r", "\n")
    text = _MULTISPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# PII anonymization (applied before any storage or API call)
# --------------------------------------------------------------------------- #
def anonymize(text: str, names: list[str] | None = None) -> str:
    if not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _STREET.sub("[ADDRESS]", text)
    text = _ZIP.sub("[ADDRESS]", text)
    for name in names or []:
        if name and len(name) > 2:
            text = re.sub(rf"\b{re.escape(name)}\b", "[NAME]", text, flags=re.I)
    return text


_NER = None


def anonymize_ner(text: str) -> str:
    """Optional spaCy PERSON/GPE masking. Falls back to regex-only if spaCy is absent."""
    global _NER
    if _NER is None:
        try:
            import spacy

            _NER = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            _NER = False
    text = anonymize(text)
    if not _NER:
        return text
    doc = _NER(text[:100_000])
    out, last = [], 0
    for ent in doc.ents:
        if ent.label_ in {"PERSON"}:
            out.append(text[last : ent.start_char])
            out.append("[NAME]")
            last = ent.end_char
    out.append(text[last:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Light-weight language filter (no external model)
# --------------------------------------------------------------------------- #
_EN_STOP = set(
    "the and to of a in for is on with as at by an be this that are or from you we "
    "our will your have has it not can all their they which who".split()
)


_FOREIGN_STOP = set(
    "de la que el los del las por con para una une les des et est pour dans sur avec "
    "der die und das mit für ist nicht ein eine não com uma são em os um".split()
)


def english_ratio(text: str) -> float:
    """~1 for English prose *and* English keyword lists; low for other languages."""
    tokens = re.findall(r"[a-zA-Z\u00C0-\uFFFF]+", text.lower())
    if not tokens:
        return 0.0
    ascii_share = sum(t.isascii() for t in tokens) / len(tokens)
    en = sum(t in _EN_STOP for t in tokens) / len(tokens)
    foreign = sum(t in _FOREIGN_STOP for t in tokens) / len(tokens)
    return ascii_share * (en + 0.01) / (en + foreign + 0.01)


def is_english(text: str, threshold: float = 0.3) -> bool:
    return english_ratio(text) >= threshold


# --------------------------------------------------------------------------- #
# Normalization for hashing / TF-IDF
# --------------------------------------------------------------------------- #
def normalize_for_hash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode()).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def near_duplicate_mask(texts: list[str], threshold: float = 0.95, chunk: int = 500):
    """Boolean mask marking rows whose TF-IDF cosine similarity to an *earlier* row
    exceeds `threshold`. Returns (mask, list of (kept_idx, dropped_idx, sim))."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, sublinear_tf=True,
                        dtype=np.float32).fit_transform(texts)
    n = X.shape[0]
    dup = np.zeros(n, dtype=bool)
    pairs = []
    XT = X.T.tocsc()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = (X[start:stop] @ XT).toarray()
        for local, row in enumerate(sims):
            i = start + local
            if dup[i]:
                continue
            row[: i + 1] = 0
            hits = np.where(row > threshold)[0]
            for j in hits:
                if not dup[j]:
                    dup[j] = True
                    pairs.append((i, int(j), float(row[j])))
    return dup, pairs


def length_outliers(word_counts, low_q: float = 0.01, high_q: float = 0.99) -> dict:
    """Percentile bounds (primary rule) with IQR fences as a cross-check."""
    import numpy as np

    wc = np.asarray(word_counts)
    q1, q3 = np.percentile(wc, [25, 75])
    iqr = q3 - q1
    return {
        "p_low": float(np.percentile(wc, low_q * 100)),
        "p_high": float(np.percentile(wc, high_q * 100)),
        "iqr_low": float(max(0, q1 - 1.5 * iqr)),
        "iqr_high": float(q3 + 1.5 * iqr),
    }


def truncate_words(text: str, max_words: int) -> str:
    parts = re.split(r"(\s+)", text)
    out, n = [], 0
    for p in parts:
        if p and not p.isspace():
            n += 1
            if n > max_words:
                break
        out.append(p)
    return "".join(out).rstrip()
