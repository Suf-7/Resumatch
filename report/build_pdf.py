"""Build the report PDF from main.tex without a TeX installation.

Converts the (deliberately simple) LaTeX subset used in main.tex to HTML and prints it
with headless Chrome. `tectonic main.tex` produces the typeset LaTeX version of the same
source when a TeX engine is available; this is the fallback used for offline builds.

    python report/make_numbers.py && python report/build_pdf.py
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_HTML = HERE / "build" / "Resumatch_Deliverable1.html"
OUT_PDF = HERE / "Resumatch_Deliverable1.pdf"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

CSS = r"""
@page { size: A4; margin: 22mm 21mm 20mm 21mm;
  @top-left { content: "Resumatch — Deliverable 1"; font: 9pt Charter, Georgia, serif; color: #4a5568; }
  @top-right { content: "AI Semester Project, FAST-NUCES Lahore"; font: 9pt Charter, Georgia, serif; color: #4a5568; }
  @bottom-center { content: counter(page); font: 9.5pt Charter, Georgia, serif; } }
@page:first { @top-left { content: none; } @top-right { content: none; } @bottom-center { content: none; } }
html { font-family: Charter, "Bitstream Charter", Georgia, serif; font-size: 10.6pt; line-height: 1.42; color: #1a202c; }
body { margin: 0; }
p { margin: 0.35em 0 0.55em; text-align: justify; hyphens: auto; }
h2 { color: #2b6cb0; font-size: 13.5pt; margin: 1.15em 0 0.35em; break-after: avoid; }
h2 .num { display: inline-block; min-width: 1.6em; }
.phase { font-size: 16pt; font-weight: bold; color: #4a5568; border-bottom: 0.7pt solid #4a5568;
  padding-bottom: 2pt; margin: 1.3em 0 0.4em; break-after: avoid; }
.titlepage { height: 250mm; display: flex; flex-direction: column; align-items: center; text-align: center; break-after: page; }
.titlepage .uni { font-size: 13pt; margin-top: 4mm; }
.titlepage .course { font-size: 11pt; margin-top: 2mm; }
.titlepage .name { font-size: 42pt; font-weight: bold; color: #2b6cb0; margin-top: 30mm; }
.titlepage .sub { font-size: 16pt; margin-top: 3mm; }
.titlepage .deliv { font-size: 13.5pt; font-weight: bold; margin-top: 18mm; }
.titlepage .track { font-size: 11pt; margin-top: 2mm; }
.titlepage table.auto { margin: 22mm auto 0; font-size: 11pt; }
.titlepage .date { margin-top: auto; font-size: 13pt; }
.toc { break-after: page; }
.toc h2 { margin-top: 0; }
.toc ol { list-style: none; padding-left: 0; }
.toc li { margin: 0.25em 0; }
.toc li.phase-item { font-weight: bold; margin-top: 0.7em; color: #4a5568; }
.toc a { color: inherit; text-decoration: none; }
.toc .n { display: inline-block; width: 2em; }
table { border-collapse: collapse; width: 100%; font-size: 9.4pt; line-height: 1.3; }
table.auto { width: auto; margin: 0 auto; }
th, td { padding: 3pt 5pt; vertical-align: top; text-align: left; }
thead tr { border-top: 1.2pt solid #000; border-bottom: 0.7pt solid #000; }
tbody tr:last-child { border-bottom: 1.2pt solid #000; }
tr.midrule { border-top: 0.7pt solid #000; }
td.r, th.r { text-align: right; white-space: nowrap; }
td.c, th.c { text-align: center; }
figure, .tablewrap { margin: 0.8em 0 1em; break-inside: avoid; }
figure img { display: block; margin: 0 auto; }
figcaption, .caption { font-size: 9.3pt; margin-top: 0.4em; text-align: justify; }
figcaption b, .caption b { font-weight: bold; }
ul, ol { margin: 0.3em 0 0.6em; padding-left: 1.5em; }
li { margin: 0.15em 0; text-align: justify; }
code, .code { font-family: Menlo, "SF Mono", monospace; font-size: 8.6pt; }
pre { font-family: Menlo, "SF Mono", monospace; font-size: 8.2pt; line-height: 1.32; background: #f7fafc;
  border: 0.5pt solid #e2e8f0; padding: 8pt 10pt; break-inside: avoid; }
.pipeline { display: flex; align-items: center; justify-content: center; gap: 0; margin: 0.6em 0 0.2em; }
.pipeline .box { border: 1.6pt solid #2b6cb0; background: #ebf4ff; border-radius: 4pt; width: 27mm;
  min-height: 15mm; display: flex; flex-direction: column; justify-content: center; text-align: center;
  font-size: 9.2pt; padding: 3pt; }
.pipeline .box small { font-size: 7.4pt; color: #4a5568; }
.pipeline .box.todo { border-style: dashed; border-color: #4a5568; background: #f7fafc; }
.pipeline .arrow { width: 7mm; text-align: center; color: #4a5568; font-size: 13pt; }
.bib { font-size: 9.3pt; padding-left: 2.2em; }
.bib li { margin: 0.3em 0; text-align: left; }
a { color: #2b6cb0; text-decoration: none; }
.para-title { font-weight: bold; }
"""


# --------------------------------------------------------------------------- helpers
def find_arg(s: str, start: int) -> tuple[str, int]:
    """Return the balanced {...} argument starting at s[start] == '{' and the index after it."""
    assert s[start] == "{", s[start: start + 30]
    depth, i = 0, start
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1: i], i + 1
        i += 1
    raise ValueError("unbalanced braces")


def replace_cmd(s: str, name: str, fn, nargs: int = 1) -> str:
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])\s*")
    out, pos = [], 0
    while True:
        m = pat.search(s, pos)
        if not m:
            out.append(s[pos:])
            return "".join(out)
        j = m.end()
        if j >= len(s) or s[j] != "{":
            out.append(s[pos: j])
            pos = j
            continue
        args = []
        for _ in range(nargs):
            while j < len(s) and s[j] in " \n":
                j += 1
            a, j = find_arg(s, j)
            args.append(a)
        out.append(s[pos: m.start()])
        out.append(fn(*args))
        pos = j


def load_macros() -> dict[str, str]:
    macros = {}
    for line in (HERE / "numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}$", line)
        if m:
            macros[m.group(1)] = m.group(2)
    return macros


class Converter:
    def __init__(self, tex: str):
        self.macros = load_macros()
        self.tex = tex
        self.sections: list[tuple[str, str, str]] = []  # (kind, number, title)
        self.labels: dict[str, str] = {}
        self.cites: dict[str, int] = {}
        self.fig_n = 0
        self.tab_n = 0

    # ------------------------------------------------------------------ inline
    def inline(self, s: str) -> str:
        for k in sorted(self.macros, key=len, reverse=True):
            s = re.sub(r"\\" + k + r"(?![A-Za-z])", lambda _, v=self.macros[k]: v, s)
        s = s.replace("\\\\", "\x00BR")
        s = re.sub(r"\\(?=\s)", "", s)
        s = s.replace("\\&", "\x00AMP").replace("\\%", "%").replace("\\_", "_").replace("\\#", "#")
        s = s.replace('\\"{i}', "ï").replace("\\~{A}", "Ã").replace("\\textasciimacron{}", "¯")
        s = html.escape(s, quote=False).replace("\x00AMP", "&amp;")
        s = replace_cmd(s, "url", lambda u: f'<a href="{u}">{u}</a>')
        s = self.math(s)
        s = s.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”")
        s = s.replace("~", "&nbsp;").replace("\\,", "&thinsp;")
        s = replace_cmd(s, "textbf", lambda x: f"<b>{x}</b>")
        s = replace_cmd(s, "emph", lambda x: f"<em>{x}</em>")
        s = replace_cmd(s, "code", lambda x: f"<code>{x}</code>")
        s = replace_cmd(s, "texttt", lambda x: f"<code>{x}</code>")
        s = replace_cmd(s, "textcolor", lambda c, x: x, nargs=2)
        s = replace_cmd(s, "cite", lambda k: f"[{self.cite(k)}]")
        s = replace_cmd(s, "ref", lambda k: self.labels.get(k, "?"))
        s = replace_cmd(s, "label", lambda k: "")
        s = replace_cmd(s, "multicolumn", lambda n, a, x: f"\x00MC{n}|{a}|{x}\x00", nargs=3)
        s = re.sub(r"\\(small|centering|noindent|xspace|normalsize|footnotesize|scriptsize)\b\s*", "", s)
        s = re.sub(r"\\vspace\{[^}]*\}", "", s)
        s = s.replace("\x00BR", "<br>")
        s = re.sub(r"(?<!\\)\{([^{}]*)\}", r"\1", s)  # stray grouping braces
        return s.strip()

    @staticmethod
    def math(s: str) -> str:
        def conv(m):
            x = m.group(1)
            x = x.replace("\\kappa", "κ").replace("\\rho", "ρ").replace("\\rightarrow", "→")
            x = x.replace("&lt;", "<").replace("&gt;", ">")
            x = x.replace("-", "−")
            x = re.sub(r"\b([a-zA-Z])\b", r"<i>\1</i>", x)
            return html.escape(x, quote=False).replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
        return re.sub(r"\$([^$]+)\$", conv, s)

    def cite(self, keys: str) -> str:
        nums = []
        for k in keys.split(","):
            k = k.strip()
            if k not in self.cites:
                self.cites[k] = len(self.cites) + 1
            nums.append(str(self.cites[k]))
        return ", ".join(nums)

    # ------------------------------------------------------------------ blocks
    def table(self, body: str) -> str:
        self.tab_n += 1
        lab = re.search(r"\\label\{([^}]*)\}", body)
        if lab:
            self.labels[lab.group(1)] = str(self.tab_n)
        cap = ""
        m = re.search(r"\\caption\{", body)
        if m:
            cap, _ = find_arg(body, m.end() - 1)
        env_m = re.search(r"\\begin\{(tabularx|tabular)\}", body)
        env_name = env_m.group(1)
        j = env_m.end()
        if env_name == "tabularx":
            _, j = find_arg(body, j)
        spec, j = find_arg(body, j)
        rows_src = body[j: body.index(f"\\end{{{env_name}}}", j)]
        spec = re.sub(r"[@>]\{[^}]*\}|\|", "", spec)
        aligns = [c for c in spec if c in "lcrLXp"]
        rows_src = rows_src.replace("\\toprule", "").replace("\\bottomrule", "")
        parts = re.split(r"\\midrule", rows_src)
        header_src = parts[0] if len(parts) > 1 else ""
        body_parts = parts[1:] if len(parts) > 1 else parts

        def cells(row: str) -> list[str]:
            return [c.strip() for c in re.split(r"(?<!\\)&", row)]

        def render_row(row: str, tag: str, cls: str = "") -> str:
            out = []
            col = 0
            for c in cells(row):
                c = self.inline(c)
                mc = re.match(r"\x00MC(\d+)\|(\w)\|(.*)\x00$", c, re.S)
                if mc:
                    n, a, c = int(mc.group(1)), mc.group(2), mc.group(3)
                    out.append(f'<{tag} colspan="{n}" class="{a}">{c}</{tag}>')
                    col += n
                    continue
                a = aligns[col] if col < len(aligns) else "l"
                out.append(f'<{tag} class="{a}">{c}</{tag}>')
                col += 1
            return f'<tr class="{cls}">' + "".join(out) + "</tr>"

        def split_rows(src: str) -> list[str]:
            return [r for r in re.split(r"\\\\(?:\[[^\]]*\])?", src) if r.strip()]

        thead = "".join(render_row(r, "th") for r in split_rows(header_src))
        tbody = []
        for i, part in enumerate(body_parts):
            for j, r in enumerate(split_rows(part)):
                tbody.append(render_row(r, "td", "midrule" if i > 0 and j == 0 else ""))
        cls = "auto" if env_name == "tabular" else ""
        t = f'<table class="{cls}"><thead>{thead}</thead><tbody>{"".join(tbody)}</tbody></table>'
        if cap:
            t += f'<div class="caption"><b>Table {self.tab_n}:</b> {self.inline(cap)}</div>'
        return f'<div class="tablewrap">{t}</div>'

    def figure(self, body: str) -> str:
        self.fig_n += 1
        cap = ""
        m = re.search(r"\\caption\{", body)
        if m:
            cap, _ = find_arg(body, m.end() - 1)
        if "tikzpicture" in body:
            nodes = re.findall(r"\\node\[(\w+)[^\]]*\]\s*\(\w+\)\s*\{(.*?)\};", body)
            boxes = []
            for style, text in nodes:
                parts = []
                for piece in text.split("\\\\"):
                    small = piece.lstrip().startswith("\\scriptsize")
                    piece = self.inline(piece.replace("\\scriptsize", ""))
                    parts.append(f"<small>{piece}</small>" if small else piece)
                boxes.append(f'<div class="box {"todo" if style == "todo" else ""}">{"<br>".join(parts)}</div>')
            inner = '<div class="pipeline">' + '<div class="arrow">➜</div>'.join(boxes) + "</div>"
        else:
            src = re.search(r"\\includegraphics(?:\[width=([\d.]*)\\linewidth\])?\{([^}]*)\}", body)
            w = float(src.group(1) or 1.0) if src.group(1) != "" else 1.0
            png = src.group(2).replace(".pdf", ".png")
            inner = f'<img src="../{png}" style="width:{w * 100:.0f}%">'
        cap_html = f"<figcaption><b>Figure {self.fig_n}:</b> {self.inline(cap)}</figcaption>" if cap else ""
        return f"<figure>{inner}{cap_html}</figure>"

    def lists(self, s: str) -> str:
        def conv(m):
            tag = "ul" if m.group(1) == "itemize" else "ol"
            items = re.split(r"\\item\s*", m.group(2))[1:]
            return f"<{tag}>" + "".join(f"<li>{self.inline(i)}</li>" for i in items) + f"</{tag}>"
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\\begin\{(itemize|enumerate)\}(?:\[[^\]]*\])?((?:(?!\\begin\{(?:itemize|enumerate)\}).)*?)\\end\{\1\}",
                       conv, s, flags=re.S)
        return s

    # ------------------------------------------------------------------ document
    def number_sections(self, body: str) -> None:
        n = 0
        for m in re.finditer(r"\\(section|phase)\{(.*?)\}(\\label\{([^}]*)\})?", body):
            if m.group(1) == "section":
                n += 1
                if m.group(4):
                    self.labels[m.group(4)] = str(n)
                self.sections.append(("section", str(n), m.group(2)))
            else:
                self.sections.append(("phase", "", m.group(2)))

    def titlepage(self, tp: str) -> str:
        rows = re.search(r"\\midrule(.*?)\\bottomrule", tp, re.S).group(1)
        trs = "".join(
            "<tr>" + "".join(f"<td>{self.inline(c)}</td>" for c in r.split("&")) + "</tr>"
            for r in re.split(r"\\\\", rows) if r.strip())
        return f"""<div class="titlepage">
<div class="uni">FAST National University of Computer and Emerging Sciences, Lahore</div>
<div class="course">Artificial Intelligence — Semester Project</div>
<div class="name">Resumatch</div>
<div class="sub">AI-Powered Resume–Job Matching Platform</div>
<div class="deliv">Deliverable 1: Problem Definition &amp; Data Foundations</div>
<div class="track">Track A — Application Development</div>
<table class="auto"><thead><tr><th>Member</th><th>Roll No. / Section</th></tr></thead><tbody>{trs}</tbody></table>
<div class="date">October 2026</div></div>"""

    def toc(self) -> str:
        items = []
        for kind, num, title in self.sections:
            t = self.inline(title)
            if kind == "phase":
                items.append(f'<li class="phase-item">{t}</li>')
            else:
                items.append(f'<li><a href="#s{num}"><span class="n">{num}</span>{t}</a></li>')
        items.append('<li><a href="#refs"><span class="n"></span>References</a></li>')
        return f'<div class="toc"><h2>Contents</h2><ol>{"".join(items)}</ol></div>'

    def convert(self) -> str:
        body = self.tex.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
        body = re.sub(r"(?m)^\s*%.*$", "", body)
        self.number_sections(body)

        blocks: list[str] = []

        def stash(h: str) -> str:
            blocks.append(h)
            return f"\n\n\x00BLOCK{len(blocks) - 1}\x00\n\n"

        body = re.sub(r"\\begin\{titlepage\}(.*?)\\end\{titlepage\}", lambda m: stash(self.titlepage(m.group(1))), body, flags=re.S)
        body = body.replace("\\tableofcontents", stash(self.toc()))
        body = re.sub(r"\\begin\{verbatim\}\n?(.*?)\\end\{verbatim\}",
                      lambda m: stash(f"<pre>{html.escape(m.group(1))}</pre>"), body, flags=re.S)
        body = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?(.*?)\\end\{table\}", lambda m: stash(self.table(m.group(1))), body, flags=re.S)
        body = re.sub(r"\\begin\{figure\}(?:\[[^\]]*\])?(.*?)\\end\{figure\}", lambda m: stash(self.figure(m.group(1))), body, flags=re.S)

        def bib(m):
            items = re.split(r"\\bibitem\{([^}]*)\}", m.group(1))[1:]
            pairs = list(zip(items[::2], items[1::2]))
            for k, _ in pairs:
                self.cite(k)
            pairs.sort(key=lambda p: self.cites[p[0]])
            lis = "".join(f'<li value="{self.cites[k]}">{self.inline(t)}</li>' for k, t in pairs)
            return stash(f'<h2 id="refs">References</h2><ol class="bib">{lis}</ol>')
        body = re.sub(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", bib, body, flags=re.S)
        body = re.sub(r"\\begin\{(itemize|enumerate)\}.*?\\end\{\1\}",
                      lambda m: stash(self.lists(m.group(0))), body, flags=re.S)

        sec_iter = iter([s for s in self.sections])

        def heading(m):
            kind, num, title = next(sec_iter)
            if kind == "phase":
                return stash(f'<div class="phase">{self.inline(title)}</div>')
            return stash(f'<h2 id="s{num}"><span class="num">{num}</span>{self.inline(title)}</h2>')
        body = re.sub(r"\\(section|phase)\{.*?\}(\\label\{[^}]*\})?", heading, body)
        body = re.sub(r"\\(newpage|thispagestyle\{\w+\}|clearpage)", "", body)

        out = []
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para:
                continue
            m = re.fullmatch(r"\x00BLOCK(\d+)\x00", para)
            if m:
                out.append(blocks[int(m.group(1))])
                continue
            para = replace_cmd(para, "paragraph", lambda t: f"\x01{t}\x02")
            para = self.inline(para).replace("\x01", '<span class="para-title">').replace("\x02", "</span>&ensp;")
            out.append(f"<p>{para}</p>")
        html_body = "\n".join(out)
        html_body = re.sub(r"\x00BLOCK(\d+)\x00", lambda m: blocks[int(m.group(1))], html_body)
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f"<title>Resumatch — Deliverable 1</title><style>{CSS}</style></head>"
                f"<body>{html_body}</body></html>")


def main() -> None:
    conv = Converter((HERE / "main.tex").read_text())
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(conv.convert())
    chrome = next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        sys.exit(f"Chrome not found; open {OUT_HTML} in a browser and print to PDF.")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={OUT_PDF}",
                    OUT_HTML.as_uri()], check=True, capture_output=True)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
