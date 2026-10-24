"""Download the ESCO v1.1.2 skills taxonomy (English) as CSV.

The official portal (https://esco.ec.europa.eu/en/use-esco/download) requires an
interactive form, so we fetch the unmodified CSV export redistributed under the
EU open-data licence by the Tabiya taxonomy project.
"""
from __future__ import annotations

import pandas as pd
import requests

from common import RAW, save_stats

URL = (
    "https://raw.githubusercontent.com/tabiya-tech/taxonomy-model-application/main/"
    "data-sets/csv/esco-v1.1.2/skills.csv"
)
OUT = RAW / "esco_skills_v1.1.2.csv"


def main() -> None:
    if not OUT.exists():
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        OUT.write_bytes(r.content)
    df = pd.read_csv(OUT)
    alt = df["ALTLABELS"].fillna("").str.split("\n").map(lambda x: len([a for a in x if a]))
    stats = {
        "file": OUT.name,
        "source": URL,
        "skills": int(len(df)),
        "alt_labels": int(alt.sum()),
        "skill_types": df["SKILLTYPE"].value_counts().to_dict(),
        "reuse_levels": df["REUSELEVEL"].value_counts().to_dict(),
    }
    save_stats("esco", stats)
    print(stats)


if __name__ == "__main__":
    main()
