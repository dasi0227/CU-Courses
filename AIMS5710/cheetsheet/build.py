#!/usr/bin/env python3
"""Build the complete source Markdown as a two-page, print-ready XeLaTeX PDF.

Run: python3 build.py [--font-size 7.7] [--second-page-font-size 7.2]
Requires XeLaTeX, Python pdfplumber/pypdf, and Poppler pdftoppm.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "cheetsheet.md"
OUTPUT = HERE / "output"
TEMP = HERE / "tmp"


def executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path("/Library/TeX/texbin") / name
    if fallback.exists():
        return str(fallback)
    raise RuntimeError(f"Required executable not found: {name}")


def text_tex(value: str) -> str:
    value = re.sub(r"[ \t]*→[ \t]*", " → ", value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "→": r"\(\to\)", "：": ": ", "、": ", ", "，": ", ",
        "；": "; ", "／": "/", "–": "--", "—": "---",
    }
    return "".join(replacements.get(c, c) for c in value)


def math_tex(value: str) -> str:
    # Display-style fractions stay readable even in a compact reference sheet.
    value = value.strip()
    value = value.replace(r"\]", "").replace(r"\[", "")
    return r"\(\displaystyle " + value + r"\)"


MATH = re.compile(r"(\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\))")
ENTRY_LABELS = {
    "VGG", "NiN", "GoogLeNet", "ResNet", "DenseNet", "ResNeXt", "MobileNet", "EfficientNet",
    "Forget Gate", "Input Gate", "Output Gate", "Candidate Cell State", "Cell State Update",
    "Hidden State", "Reset Gate", "Update Gate", "Candidate Hidden State", "Hidden State Update",
}


def inline(value: str) -> str:
    pieces = []
    for part in MATH.split(value):
        if not part:
            continue
        if part.startswith("$"):
            pieces.append(math_tex(part[1:-1]))
        elif part.startswith((r"\[", r"\(")):
            pieces.append(math_tex(part[2:-2]))
        else:
            pieces.append(text_tex(part))
    return "".join(pieces)


def source_blocks(source: str) -> list[tuple[str, list[tuple[int, str]]]]:
    # Join multiline dollar math without changing any source content.
    blocks = []
    pending = []
    dollars = 0
    for number, raw in enumerate(source.splitlines(), 1):
        if not raw.strip():
            if pending:
                raise ValueError(f"Unclosed math before source line {number}")
            continue
        pending.append((number, raw))
        dollars += raw.count("$")
        if dollars % 2 == 0:
            blocks.append((" ".join(v.strip() for _, v in pending), pending))
            pending, dollars = [], 0
    if pending:
        raise ValueError("Unclosed dollar math at end of source")
    return blocks


def render_content(source: str) -> tuple[str, list[dict]]:
    result, manifest = [], []
    module = ""
    section = ""
    in_table = False
    for content, lines in source_blocks(source):
        if in_table and not content.startswith("|"):
            result.append(r"\bottomrule\end{tabular}\par\endgroup")
            in_table = False
        record = {"lines": [n for n, _ in lines], "source": content}
        if content.startswith("##### "):
            title = content[6:]
            section = title
            result.append(r"\subsection*{" + text_tex(title) + "}")
            record["kind"] = "section"
        elif content.startswith("#### "):
            module = content[5:]
            if module == "CNN":
                result.append(r"\end{multicols}\newpage\pagetwostyle\begin{multicols}{2}")
            elif module in {"Optimization", "RNN"}:
                result.append(r"\columnbreak")
            result.append(r"\module{" + text_tex(module) + "}")
            record["kind"] = "module"
        elif content.startswith("|"):
            cells = [c.strip() for c in content.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", c) for c in cells):
                record["kind"] = "table_separator"
            elif cells[0] == "Activation":
                result.append(
                    r"\par\begingroup\fontsize{7}{8.5}\selectfont"
                    r"\setlength{\tabcolsep}{1.7pt}\renewcommand{\arraystretch}{1.55}"
                    r"\setlength{\activationwidth}{\dimexpr\linewidth-8\tabcolsep\relax}"
                    r"\noindent\begin{tabular}{@{}"
                    r">{\raggedright\arraybackslash}m{.12\activationwidth}"
                    r">{\centering\arraybackslash}m{.225\activationwidth}"
                    r">{\centering\arraybackslash}m{.225\activationwidth}"
                    r">{\raggedright\arraybackslash}m{.215\activationwidth}"
                    r">{\raggedright\arraybackslash}m{.215\activationwidth}@{}}\toprule"
                    + " & ".join(r"\textbf{" + text_tex(c) + "}" for c in cells)
                    + r"\\\midrule"
                )
                in_table = True
                record["kind"] = "table_header"
            elif section == "Pooling" and cells[0] == "Pooling":
                result.append(
                    r"\par\begingroup\fontsize{7}{8.5}\selectfont"
                    r"\setlength{\tabcolsep}{1.7pt}\renewcommand{\arraystretch}{1.55}"
                    r"\setlength{\activationwidth}{\dimexpr\linewidth-4\tabcolsep\relax}"
                    r"\noindent\begin{tabular}{@{}"
                    r">{\raggedright\arraybackslash}m{.16\activationwidth}"
                    r">{\centering\arraybackslash}m{.41\activationwidth}"
                    r">{\raggedright\arraybackslash}m{.43\activationwidth}@{}}\toprule"
                    + " & ".join(r"\textbf{" + text_tex(c) + "}" for c in cells)
                    + r"\\\midrule"
                )
                in_table = True
                record["kind"] = "table_header"
            elif section == "Modern Convolutions" and cells[0] == "Model":
                result.append(
                    r"\par\begingroup\fontsize{7}{8.5}\selectfont"
                    r"\setlength{\tabcolsep}{1.7pt}\renewcommand{\arraystretch}{1.3}"
                    r"\setlength{\activationwidth}{\dimexpr\linewidth-4\tabcolsep\relax}"
                    r"\noindent\begin{tabular}{@{}"
                    r">{\raggedright\arraybackslash}m{.17\activationwidth}"
                    r">{\raggedright\arraybackslash}m{.36\activationwidth}"
                    r">{\raggedright\arraybackslash}m{.47\activationwidth}@{}}\toprule"
                    + " & ".join(r"\textbf{" + text_tex(c) + "}" for c in cells)
                    + r"\\\midrule"
                )
                in_table = True
                record["kind"] = "table_header"
            else:
                expected_cells = 3 if section in {"Pooling", "Modern Convolutions"} else 5
                if len(cells) != expected_cells:
                    raise ValueError(f"Unexpected table row: {content}")
                result.append(
                    " & ".join(inline(c) for c in cells)
                    + (r"\\\addlinespace[1pt]" if section == "Modern Convolutions" else r"\\\addlinespace[2pt]")
                )
                record["kind"] = "table_row"
        elif section == "Multi-Layer Perceptron" and "；" in content:
            result.append(r"\pairednote{" + inline(content) + "}")
            record["kind"] = "body"
        elif section == "Weight Initialization" and content.startswith("Xavier / Glorot："):
            result.append(r"\fitline{" + inline(content) + "}")
            record["kind"] = "body"
        elif section == "Adaptive Moment Estimation" and content.startswith(("Adam：", "AdamW：")):
            result.append(r"\fitline{" + inline(content) + "}")
            record["kind"] = "body"
        elif section == "Resilient Propagation" and content.startswith("Rprop："):
            result.append(r"\fitline{" + inline(content) + "}")
            record["kind"] = "body"
        elif module in {"CNN", "RNN"} and content.split("：", 1)[0] in ENTRY_LABELS:
            label, detail = content.split("：", 1)
            result.append(r"\noindent \textbf{" + text_tex(label) + "}: " + inline(detail.strip()) + r"\par")
            record["kind"] = "body"
        else:
            result.append(r"\noindent " + inline(content) + r"\par")
            record["kind"] = "body"
        record["module"] = module
        manifest.append(record)
    if in_table:
        result.append(r"\bottomrule\end{tabular}\par\endgroup")
    all_lines = [n for item in manifest for n in item["lines"]]
    expected = [i for i, line in enumerate(source.splitlines(), 1) if line.strip()]
    if all_lines != expected:
        raise AssertionError("Source coverage failure: a line was dropped or duplicated")
    return "\n".join(result), manifest


def document(body: str, size: float, second_page_size: float = 7.2) -> str:
    preamble = r"""\documentclass[a4paper]{article}
\usepackage[hmargin=2mm,vmargin=8mm,noheadfoot]{geometry}
\usepackage{amsmath,amssymb,newtxmath}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold]
\usepackage{multicol,xcolor,titlesec,needspace,array,booktabs,adjustbox}
\usepackage[unicode,hidelinks,pdfencoding=auto]{hyperref}
\hypersetup{pdftitle={AIMS5710 Deep Learning Cheatsheet},pdfauthor={}}
\definecolor{accent}{HTML}{153C52}
\definecolor{rulecolor}{HTML}{BBC9D1}
\setlength{\columnsep}{5mm}
\setlength{\columnseprule}{0.25pt}
\def\columnseprulecolor{\color{rulecolor}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\setlength{\multicolsep}{0pt}
\setlength{\emergencystretch}{1em}
\newlength{\activationwidth}
\raggedright
\pagestyle{empty}
\titleformat{\subsection}{\sffamily\bfseries\color{accent}\fontsize{8.6}{10}\selectfont}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{5pt}{2pt}
\newcommand{\module}[1]{\par\needspace{45pt}\addvspace{4pt}\noindent\colorbox{accent}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\color{white}\sffamily\bfseries\fontsize{10}{12}\selectfont #1}}\par\vspace{1pt}}
\newcommand{\pairednote}[1]{\par\noindent\adjustbox{max width=\linewidth}{\fontsize{7.2}{8.3}\selectfont #1}\par}
\newcommand{\fitline}[1]{\par\noindent\adjustbox{max width=\linewidth}{#1}\par}
\newcommand{\pagetwostyle}{\fontsize{PAGE2SIZE}{PAGE2LEADING}\selectfont\raggedcolumns}
\newcommand{\bodyfont}{\fontsize{FONTSIZE}{LEADING}\selectfont}
\renewcommand{\small}{\fontsize{SMALLSIZE}{SMALLLEADING}\selectfont}
\renewcommand{\tiny}{\fontsize{7}{8}\selectfont}
\let\originalsum\sum
\let\originalprod\prod
\renewcommand{\sum}{\mathop{\textstyle\originalsum}\nolimits}
\renewcommand{\prod}{\mathop{\textstyle\originalprod}\nolimits}
\begin{document}\bodyfont\raggedcolumns
\begin{multicols}{2}
"""
    substitutions = {
        "FONTSIZE": size, "LEADING": 9,
        "SMALLSIZE": size - .3, "SMALLLEADING": size * 1.1,
        "PAGE2SIZE": second_page_size, "PAGE2LEADING": 9,
    }
    for key in sorted(substitutions, key=len, reverse=True):
        val = substitutions[key]
        preamble = preamble.replace(key, str(round(val, 3)))
    return preamble + body + "\n" + r"\end{multicols}\end{document}" + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-size", type=float, default=7.7)
    parser.add_argument("--second-page-font-size", type=float, default=7.2)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    source = SOURCE.read_text()
    body, manifest = render_content(source)
    source_formulas = [formula[1:-1] if formula.startswith("$") else formula[2:-2]
                       for block, _ in source_blocks(source) for formula in MATH.findall(block)]
    rendered_formulas = re.findall(r"\\\(\\displaystyle (.*?)\\\)", body)
    def normalized_formula(value: str) -> str:
        return re.sub(r"\s+", "", value)
    formula_sequence_matches = ([normalized_formula(f) for f in source_formulas]
                                == [normalized_formula(f) for f in rendered_formulas])
    if not formula_sequence_matches:
        raise AssertionError("Formula coverage failure: source and rendered math differ")
    tex = HERE / "cheetsheet.tex"
    tex.write_text(document(body, args.font_size, args.second_page_font_size))
    cmd = [executable("xelatex"), "-interaction=nonstopmode", "-halt-on-error",
           "-file-line-error", f"-output-directory={TEMP}", str(tex)]
    for _ in range(2):
        done = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(done.stdout[-6000:] + done.stderr[-1000:])
    shutil.copy2(TEMP / "cheetsheet.pdf", OUTPUT / "cheetsheet.pdf")
    from pypdf import PdfReader
    import pdfplumber
    pdf = OUTPUT / "cheetsheet.pdf"
    reader = PdfReader(pdf)
    extracted = "\n\n".join(p.extract_text() for p in reader.pages)
    prose = " ".join(MATH.sub("", block) for block, _ in source_blocks(source))
    def prose_words(value: str) -> Counter:
        return Counter(re.findall(r"[a-z]+", unicodedata.normalize("NFKC", value).lower()))
    missing_prose = dict(prose_words(prose) - prose_words(extracted))
    normalized_pdf_text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", extracted))
    missing_headings = [m["source"].lstrip("# ") for m in manifest
                        if m["kind"] in {"module", "section"}
                        and re.sub(r"\s+", "", m["source"].lstrip("# ")) not in normalized_pdf_text]
    log = (TEMP / "cheetsheet.log").read_text()
    warnings = [line for line in log.splitlines()
                if re.search(r"Overfull \\[hv]box|Missing character", line)]
    boundaries = []
    with pdfplumber.open(pdf) as doc:
        for i, page in enumerate(doc.pages, 1):
            for c in page.chars:
                if c["text"].strip() and (c["x0"] < 2 or c["x1"] > page.width - 2
                                          or c["top"] < 10 or c["bottom"] > page.height - 10):
                    boundaries.append({"page": i, "text": c["text"],
                                       "bbox": [c["x0"], c["top"], c["x1"], c["bottom"]]})
    report = {
        "source": str(SOURCE.relative_to(HERE.parent)),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "pages": len(reader.pages), "font_size_pt": args.font_size,
        "page_2_font_size_pt": args.second_page_font_size,
        "page_1_leading_pt": 9,
        "page_2_leading_pt": 9,
        "page_1_paragraph_spacing_pt": 3,
        "page_2_paragraph_spacing_pt": 3,
        "source_nonblank_lines": sum(bool(s.strip()) for s in source.splitlines()),
        "covered_nonblank_lines": sum(len(m["lines"]) for m in manifest),
        "module_count": sum(m["kind"] == "module" for m in manifest),
        "section_count": sum(m["kind"] == "section" for m in manifest),
        "formula_count": len(source_formulas),
        "rendered_formula_count": len(rendered_formulas),
        "formula_sequence_matches": formula_sequence_matches,
        "missing_prose_word_occurrences": missing_prose,
        "missing_section_headings": missing_headings,
        "latex_warnings": warnings, "out_of_bounds_chars": boundaries,
        "source_blocks": manifest,
    }
    (OUTPUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (TEMP / "extracted.txt").write_text(extracted)
    for stale_preview in OUTPUT.glob("page-*.png"):
        stale_preview.unlink()
    subprocess.run([executable("pdftoppm"), "-r", "150", "-png", str(pdf),
                    str(OUTPUT / "page")], check=True, capture_output=True)
    print(json.dumps({k: v for k, v in report.items() if k != "source_blocks"}, indent=2))
    if len(reader.pages) != 2 or warnings or boundaries or missing_prose or missing_headings:
        raise SystemExit("Layout validation failed; inspect PDF/log and revise layout.")
    shutil.copy2(pdf, HERE / "cheetsheet.pdf")
    shutil.rmtree(TEMP)


if __name__ == "__main__":
    main()
