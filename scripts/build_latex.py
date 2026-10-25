"""LaTeX version of the paper on the Apart Research hackathon template, with the vector figures.

    uv run --with pypandoc_binary python scripts/build_latex.py <paper.md>     # writes paper/
    tectonic paper/main.tex                                                   # -> paper/main.pdf

<paper.md> is the Markdown export of the paper doc. Pandoc converts the body; this script adds the template's look
(US Letter, 1 in margins, Old Standard, ruled title block, author grid "With Apart Research", centred abstract,
sprint footnote) and turns the doc's figure placeholders into LaTeX floats that include figures/*.pdf, numbered
1-4 in the main text and C1 in the appendices. Text in [brackets] that is not a number is for the authors and is
highlighted.
"""

import re
import shutil
import sys
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
FIGS = {"1": "fig0_concept", "2": "fig1_e1", "3": "fig2_mechanism", "4": "fig3_circuit", "5": "fig4_control",
        "C1": "figC1_sweep", "C2": "figC2_replication"}
AUTHORS = [("Peter Lu", "Independent")]
SPRINT = "https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25"

PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{OldStandard-Regular.otf}{%
  \setmainfont{OldStandard}[Extension=.otf, UprightFont=*-Regular, BoldFont=*-Bold, ItalicFont=*-Italic]}{%
  \setmainfont{lmroman10-regular.otf}}
\setmonofont{lmmono10-regular.otf}[Scale=0.92]
\usepackage{newunicodechar}
\newunicodechar{−}{–} % Old Standard's U+2212 is em-dash wide; its en dash matches the figure width
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\usepackage{graphicx,booktabs,longtable,array,calc,xcolor,enumitem,titlesec,caption,microtype}
\usepackage[hidelinks=false,colorlinks=true,linkcolor=apartblue,urlcolor=apartblue,citecolor=apartblue]{hyperref}
\definecolor{apartblue}{HTML}{1155CC}
\providecommand{\tightlist}{\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}}
\setlist{nosep,leftmargin=*,labelsep=0.5em,itemsep=1.5pt,topsep=2pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4.5pt plus 1pt}
\linespread{1.0}
\setlength{\emergencystretch}{2.5em}
\usepackage{xurl}
\titleformat{\section}{\fontsize{13.5}{16}\selectfont\bfseries}{\thesection.}{0.4em}{}
\titleformat{\subsection}{\fontsize{11.5}{14}\selectfont\bfseries}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 2pt}{4pt}
\titlespacing*{\subsection}{0pt}{9pt plus 2pt minus 1pt}{3pt}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false,skip=4pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.8}
\newcommand{\todo}[1]{{\color{red!75!black}\bfseries #1}} % author-facing placeholder; breakable, unlike a colorbox
\let\oldunderscore\_\renewcommand{\_}{\oldunderscore\allowbreak} % long file names in \texttt can wrap
\setlength{\LTpre}{4pt}\setlength{\LTpost}{4pt}
\makeatletter\@ifundefined{c@none}{\newcounter{none}}{}\makeatother % pandoc: uncaptioned longtables
\begin{document}
"""


def size_tables(md: str) -> str:
    """Pandoc sizes wide pipe-table columns by the dash counts in the separator row: make them proportional to each
    column's longest cell, so short columns (Stage, Layers) stop taking a quarter of the line each."""
    lines = md.split("\n")
    for i, line in enumerate(lines):
        if re.fullmatch(r"\|(\s*:?-+:?\s*\|)+\s*", line) and i > 0:
            j = i + 1
            rows = [lines[i - 1]]
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(lines[j])
                j += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            n = len(cells[0])
            col = lambda k: [r[k] for r in cells if k < len(r)]
            word = lambda k: max((len(w) for c in col(k) for w in c.split()), default=1)
            # longest cell, but never narrower than the longest unbreakable word plus padding
            width = [max(max(map(len, col(k))), 1.6 * word(k) + 4) for k in range(n)]
            total = sum(width) or 1
            lines[i] = "| " + " | ".join("-" * max(3, round(60 * w / total)) for w in width) + " |"
    return "\n".join(lines)


def to_latex(md: str) -> str:
    md = size_tables(md)
    return pypandoc.convert_text(md, "latex", format="markdown-auto_identifiers+pipe_tables",
                                 extra_args=["--wrap=preserve", "--shift-heading-level-by=-1"]).replace("\r\n", "\n")


def figures(tex: str) -> str:
    """\\[image: Figure X] + the italic caption paragraph after it -> a float with the vector PDF."""
    def repl(m):
        key, cap = m.group(1), m.group(3).strip()
        return (f"\\begin{{figure}}[tbp]\n\\centering\n\\includegraphics[width=\\linewidth]{{figures/{FIGS[key]}.pdf}}\n"
                f"\\caption{{{cap}}}\\label{{fig:{key}}}\n\\end{{figure}}")
    pat = r"(?:\{\[\}|\[)image: Figure (\w+)(?:\{\]\}|\])\s*\\emph\{Figure (\w+)\.\s*(.*?)\}\s*(?=\n\n|\Z)"
    return re.sub(pat, repl, tex, flags=re.S)


def unbreakable_short_tables(tex: str, max_rows: int = 8) -> str:
    """Pandoc emits every table as a longtable, which may split across pages and repeat its header on the next one.
    Short main-text tables become a plain tabular that moves as a block."""
    def repl(m):
        spec, body = m.group(1), m.group(2)
        head, _, rows = body.partition("\\endlastfoot\n")
        if rows.count("\\\\") > max_rows:
            return m.group(0)
        head = head.replace("\\endhead\n", "").replace("\\bottomrule\\noalign{}\n", "")
        return (f"\\begin{{center}}\\small\n\\begin{{tabular}}{{{spec}}}\n{head}{rows}\\bottomrule\\noalign{{}}\n"
                f"\\end{{tabular}}\n\\end{{center}}")
    return re.sub(r"\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\end\{longtable\}", repl, tex, flags=re.S)


def placeholders(tex: str) -> str:
    # pandoc writes "[" as {[} and "]" as {]}; author-facing placeholders are bracketed non-numeric text
    def repl(m):
        inner = m.group(1)
        if re.search(r"\d", inner):  # CIs, ranges and "95% CI" notes are not placeholders
            return m.group(0)
        return f"\\todo{{[{inner}]}}"
    return re.sub(r"\{\[\}([^{}\n]{1,80}?)\{\]\}", repl, tex)


def main():
    md = Path(sys.argv[1]).read_text(encoding="utf-8")
    title = md.splitlines()[0].lstrip("# ").strip()
    abstract = md[md.index("## Abstract") + len("## Abstract"):md.index("## 1 Introduction")].strip()
    body = md[md.index("## 1 Introduction"):]
    body = re.sub(r"(?m)^(#{2,3}) \d+(?:\.\d+)? ", r"\1 ", body)  # LaTeX numbers sections itself
    refs_at, app_at = body.index("## References"), body.index("## Appendix A")
    llm_at = body.index("## LLM Usage Statement") if "## LLM Usage Statement" in body else len(body)
    code_at = body.index("## Code and Data") if "## Code and Data" in body[:refs_at] else refs_at
    main_md, code_md, refs_md = body[:code_at], body[code_at:refs_at], body[refs_at:app_at]
    apps_md, llm_md = body[app_at:llm_at], body[llm_at:]
    apps_md = re.sub(r"(?m)^## Appendix [A-Z]\. ", "## ", apps_md)

    unnumbered = lambda t: re.sub(r"\\section\{", r"\\section*{", t)
    tex_main = unbreakable_short_tables(figures(to_latex(main_md)))
    tex_code = unnumbered(to_latex(code_md))
    tex_refs = unnumbered(to_latex(refs_md)).replace("\\begin{itemize}", "\\begin{itemize}\\small\\raggedright")
    tex_apps = figures(to_latex(apps_md))
    tex_llm = unnumbered(to_latex(llm_md))
    # appendix figures keep the doc's numbering (C1)
    tex_apps = re.sub(r"\\begin\{figure\}\[tbp\]\n\\centering\n\\includegraphics\[width=\\linewidth\]\{figures/(\w+)\.pdf\}",
                      lambda m: m.group(0) + "\n\\renewcommand{\\thefigure}{" +
                      {v: k for k, v in FIGS.items()}[m.group(1)] + "}", tex_apps)
    tex_abs = to_latex(abstract)

    tag = lambda s: re.sub(r"\[([^\]]+)\]", r"\\todo{[\1]}", s)  # a raw "[" after \\ would be read as a length
    authors = " & ".join(f"{tag(n)}\\\\{tag(a)}" for n, a in AUTHORS)
    head = rf"""\begin{{center}}
\rule{{\linewidth}}{{2.4pt}}\\[10pt]
{{\fontsize{{19}}{{23}}\selectfont\bfseries {title}\footnote{{Research conducted at the \href{{{SPRINT}}}{{AI Collusion Research Sprint}}, October 2026.}}\par}}
\vspace{{8pt}}\rule{{\linewidth}}{{0.6pt}}\\[8pt]
{{\small\begin{{tabular}}{{{'c' * len(AUTHORS)}}}{authors}\end{{tabular}}}}\\[6pt]
{{\small\textbf{{With}}\\Apart Research}}\\[10pt]
{{\fontsize{{13}}{{15}}\selectfont\bfseries Abstract}}
\end{{center}}
\vspace{{-4pt}}
\begin{{center}}\begin{{minipage}}{{5.4in}}\small\setlength{{\parskip}}{{4pt}}
{tex_abs}
\end{{minipage}}\end{{center}}
"""
    doc = (PREAMBLE + head + tex_main + tex_code + tex_refs + "\\clearpage\n\\appendix\n" + tex_apps + tex_llm +
           "\n\\end{document}\n")
    doc = placeholders(doc)
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for name in FIGS.values():
        shutil.copy(ROOT / "figures" / f"{name}.pdf", OUT / "figures" / f"{name}.pdf")
    (OUT / "main.tex").write_text(doc, encoding="utf-8")
    print(f"wrote paper/main.tex ({len(doc):,} chars) and paper/figures/*.pdf")
    left = re.findall(r"\{\[\}image", doc)
    print("unconverted figure placeholders:", len(left))


if __name__ == "__main__":
    main()
