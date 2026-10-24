"""ESCO-based skill gazetteer: extraction + normalization of skill mentions.

This is the Phase-2 dictionary baseline used for feature engineering and EDA.
It maps surface variants ("JS", "Javascript", "Java script") to one canonical
ESCO preferred label ("JavaScript"). The learned skill-NER model in Phase 3
will be compared against this baseline.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

import pandas as pd

from common import RAW

ESCO_FILE = RAW / "esco_skills_v1.1.2.csv"
MAX_NGRAM = 4

# Variants frequently seen in CVs/postings that ESCO lacks or spells differently.
EXTRA_SYNONYMS = {
    "JavaScript": ["js", "es6", "vanilla js"],
    "TypeScript": ["typescript", "ts"],
    "Python (computer programming)": ["python3", "python 3"],
    "machine learning": ["ml", "machine-learning"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "deep learning": ["deep learning", "dl"],
    "computer vision": ["computer vision", "cv models"],
    "React": ["react", "reactjs", "react.js", "react js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Angular": ["angular", "angularjs", "angular.js"],
    "Vue.js": ["vue", "vuejs", "vue.js"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Spring Boot": ["spring boot", "springboot"],
    "Flutter": ["flutter"],
    "Kotlin": ["kotlin"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Amazon Web Services": ["aws", "amazon web services"],
    "Microsoft Azure": ["azure", "microsoft azure"],
    "Google Cloud Platform": ["gcp", "google cloud"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Microsoft Excel": ["excel", "ms excel", "microsoft excel"],
    "Git": ["git", "github", "gitlab"],
    "REST API": ["rest api", "restful api", "rest apis", "restful services", "rest services"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "MongoDB": ["mongodb", "mongo"],
    "PostgreSQL": ["postgresql", "postgres"],
    "Linux": ["linux", "unix"],
    "Jira": ["jira"],
    "Figma": ["figma"],
    "Adobe Photoshop": ["photoshop", "adobe photoshop"],
    "Adobe Illustrator": ["illustrator", "adobe illustrator"],
    "search engine optimisation": ["seo"],
    "QuickBooks": ["quickbooks"],
    "Salesforce": ["salesforce"],
    "AutoCAD": ["autocad"],
    "SolidWorks": ["solidworks"],
    "MATLAB": ["matlab"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "R": ["r programming", "rstudio"],
    ".NET": [".net", "dotnet", "asp.net"],
    "SQL": ["sql", "t-sql", "pl/sql"],
    "Hadoop": ["hadoop"],
    "Apache Spark": ["spark", "pyspark", "apache spark"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous delivery"],
    "Agile project management": ["agile", "scrum"],
    "SAP ERP": ["sap"],
}

# Single tokens that ESCO lists as labels but are too ambiguous in free text.
AMBIGUOUS = set(
    """
    work lead support service services plan planning manage management team office act go
    order care design draw drive build run test teach sell write read speak listen
    communication administration operations operation business development research
    analysis reporting marketing sales training learning safety quality customer
    customers clients client experience skills project projects production
    maintenance security finance accounting engineering technology computer internet
    english data science art arts music health education law help advise advice
    coordinate organise organize monitor review control process processes
    systems system policy policies documentation inventory logistics delivery
    leadership teamwork negotiation coaching mentoring supervision scheduling cooking
    cleaning nursing teaching selling purchasing budgeting writing editing
    editing presentation presentations networking hiring recruiting
    mail email phone internet office excel word access outlook windows mac
    model models core spring swing ruby pig hive storm shark rust crystal elixir
    dart oz pascal logo prolog scheme smalltalk ada apl awk bash basic
    less pregnancy childbirth history energy source philosophy ethics genetics politics
    medicine medicines terminology morality integrity current standard balance form
    space scale board light term language rights recovery hearing patterns collections
    measurement fuel transport brands confidence interaction correspondence merchandise
    assembly chef puppet salt
    """.split()
)
PHRASE_BLACKLIST = {"application process", "company policies", "work environment"}
LEADING_STOPWORDS = {"a", "an", "and", "be", "the", "of", "to", "in", "for", "with", "on", "or"}
ALLOW_SINGLE = {
    "python", "java", "javascript", "sql", "mysql", "nosql", "php", "perl", "scala",
    "haskell", "kotlin", "swift", "typescript", "matlab", "sas", "spss", "stata", "excel",
    "bash", "ruby", "rust", "dart", "tableau", "salesforce", "autocad", "solidworks",
    "photoshop", "illustrator", "figma", "jira", "docker", "kubernetes", "linux", "git",
    "html", "css", "seo", "sap", "hadoop", "spark", "aws", "azure", "gcp", "react",
    "angular", "django", "flask", "flutter", "tensorflow", "pytorch", "pandas", "numpy",
    "quickbooks", "c++", "c#", ".net", "mongodb", "postgresql", "excel", "ml", "ai", "nlp",
    "agile", "scrum", "selenium", "jenkins", "ansible", "terraform", "graphql", "redis",
    "oracle", "unix", "wordpress", "shopify", "xero", "peachtree", "revit", "sketchup",
    "keras", "opencv", "airflow", "kafka", "snowflake", "databricks", "looker", "postman",
}

TOKEN = re.compile(r"[a-z0-9.#+/][a-z0-9.#+/\-]*")


def tokenize(text: str) -> list[str]:
    toks = TOKEN.findall(text.lower())
    out = []
    for t in toks:
        if t.rstrip(".,") in {"c++", "c#", ".net", "asp.net", "node.js", "react.js", "vue.js"}:
            out.append(t.rstrip(".,"))
            continue
        t = t.strip(".-/")
        if t.endswith(("++", "#")):
            out.append(t)
        elif t:
            out.append(t.strip("+#"))
    return [t for t in out if t]


def _variants(label: str) -> list[str]:
    label = label.strip()
    out = {label}
    base = re.sub(r"\s*\(.*?\)\s*", " ", label).strip()
    if base:
        out.add(base)
    return [" ".join(tokenize(v)) for v in out if v]


@lru_cache(maxsize=1)
def load_gazetteer() -> dict[str, str]:
    """Return {normalized surface form -> canonical skill label}."""
    df = pd.read_csv(ESCO_FILE, usecols=["PREFERREDLABEL", "ALTLABELS", "SKILLTYPE"])
    df = df.dropna(subset=["PREFERREDLABEL"])[["PREFERREDLABEL", "ALTLABELS", "SKILLTYPE"]]
    ambiguous = AMBIGUOUS - ALLOW_SINGLE
    candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for pref, alts, stype in df.itertuples(index=False):
        labels = [(pref, True)] + [(a, False) for a in str(alts).split("\n") if a and a != "nan"]
        priority = 0 if stype == "knowledge" else 1
        for lab, is_pref in labels:
            for v in _variants(lab):
                toks = v.split()
                n = len(toks)
                if not v or n > MAX_NGRAM or len(v) < 2 or toks[0] in LEADING_STOPWORDS:
                    continue
                if n == 1 and v not in ALLOW_SINGLE and (
                    v in ambiguous or stype != "knowledge" or not is_pref or len(v) < 4
                ):
                    continue
                candidates[v].append((priority, pref))
    gaz = {v: sorted(c, key=lambda x: (x[0], len(x[1])))[0][1] for v, c in candidates.items()}
    for canon, syns in EXTRA_SYNONYMS.items():
        for s in syns + [canon]:
            v = " ".join(tokenize(s))
            if len(v) >= 2:
                gaz[v] = canon
    for amb in ambiguous | PHRASE_BLACKLIST:
        gaz.pop(amb, None)
    return gaz


def extract_skills(text: str, gaz: dict[str, str] | None = None) -> list[str]:
    """Greedy longest-match extraction of canonical skills (deduplicated, ordered)."""
    gaz = gaz or load_gazetteer()
    toks = tokenize(text or "")
    found: dict[str, None] = {}
    i = 0
    while i < len(toks):
        for n in range(MAX_NGRAM, 0, -1):
            if i + n > len(toks):
                continue
            cand = " ".join(toks[i : i + n])
            if cand in gaz:
                found.setdefault(gaz[cand], None)
                i += n
                break
        else:
            i += 1
    return list(found)


if __name__ == "__main__":
    g = load_gazetteer()
    print(len(g), "surface forms ->", len(set(g.values())), "canonical skills")
    demo = ("Built REST services in Node.js and Python; ML models with scikit-learn and "
            "TensorFlow; dashboards in Power BI; JS/React front-end; MS Excel; project management.")
    print(extract_skills(demo, g))
