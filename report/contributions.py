"""Contribution table from the Git history: commit author -> commits, lines, modules touched.

Authors are matched to team members by e-mail or name; anything unmatched is listed under
its Git name so nothing is hidden. Writes report/stats/contributions.json.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEMBERS = {
    "Muhammad Sufyan": ("sufyansohail025@gmail.com", "suf-7", "suf"),
    "Muaaz Fahad": ("muaazfahad81@gmail.com", "muaaz"),
    "Hanan Ishaque": ("ishaqhanan525@gmail.com", "hanan"),
}
SKIP = re.compile(r"\.(pdf|png|html|json|tex)$|^report/(build|stats|figures)/")


def member(name: str, email: str) -> str:
    key = f"{name} {email}".lower()
    for m, hints in MEMBERS.items():
        if any(h in key for h in hints):
            return m
    return name


def module(path: str) -> str:
    if path.startswith(("src/", "data_scripts/")):
        return path
    if path.startswith("report/"):
        return "report (" + path.split("/")[-1] + ")"
    return path.split("/")[0] if "/" in path else path


def main() -> None:
    log = subprocess.run(["git", "log", "--no-merges", "--numstat", "--format=@@%an|%ae|%h|%s"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    stats = defaultdict(lambda: {"commits": 0, "lines_added": 0, "lines_removed": 0,
                                 "modules": defaultdict(int), "commit_list": []})
    who = None
    for line in log.splitlines():
        if line.startswith("@@"):
            name, email, sha, subject = line[2:].split("|", 3)
            who = member(name, email)
            stats[who]["commits"] += 1
            stats[who]["commit_list"].append(f"{sha} {subject}")
        elif line.strip() and who:
            add, rem, path = line.split("\t", 2)
            if SKIP.search(path) or add == "-":
                continue
            stats[who]["lines_added"] += int(add)
            stats[who]["lines_removed"] += int(rem)
            stats[who]["modules"][module(path)] += int(add) + int(rem)
    out = {}
    for m in list(MEMBERS) + [k for k in stats if k not in MEMBERS]:
        s = stats.get(m, {"commits": 0, "lines_added": 0, "lines_removed": 0, "modules": {}, "commit_list": []})
        mods = sorted(s["modules"].items(), key=lambda kv: -kv[1])
        out[m] = {"commits": s["commits"], "lines_added": s["lines_added"],
                  "lines_removed": s["lines_removed"], "modules": [k for k, _ in mods],
                  "commit_list": s["commit_list"]}
    (HERE / "stats" / "contributions.json").write_text(json.dumps(out, indent=2))
    for m, s in out.items():
        print(f"{m}: {s['commits']} commits, +{s['lines_added']}/-{s['lines_removed']}, {s['modules'][:6]}")


if __name__ == "__main__":
    main()
