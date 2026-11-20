"""Table 5 of the RESS manuscript (Section 5.3, Errors of the classifier and the LLM coders): why the agreement
rule works, for the eight cases (device / aircraft x 2025 / 2026 x Qwen3.8-27B / Gemma 4 31B).

Three blocks, the order of the 5.3 prose:
  1  rare answers of each LLM coder that pass the rule (the top rare code, within the classifier's top five, is the
     code it names), among its correct and among its false rare answers of the year
  2  the other side: shortlisted reports on which the LLM coder names the top rare code, when the report is filed
     under this code and when it is filed under another code
  3  likelihood ratio, (correct placements / rare reports) / (false placements / ordinary reports), of the classifier,
     each LLM coder alone and the rule (Ours, either LLM coder, top five)

Source: results/why_it_works/{maude,sdr}.json, years.<year>.<llm>.mechanism (written by scripts/why_it_works.py;
see notes/why-it-works-2026-11-16.md). Every value is recomputed here from the stored counts, not from the stored
rounded ratios. Estimated (dagger): the false rare answers of an LLM coder used alone over the year, taken from the
samples of the year as in Table 2 (results/paper_v4/<corpus>_numbers.json); every other count is exact.

Writes paper/RESS/figures/candidates/:
  tab_mechanism.tex             the table (cas-sc style of the manuscript tables), for \\input or pasting
  tab_mechanism_preview.pdf     the table compiled with the manuscript's class, cropped
  tab_mechanism_preview.png     the same at 150 dpi
  tab_mechanism_values.json     unrounded values, printed cells, 95% ranges of the estimated shares, and the check of
                                every number that Section 5.3 states about these quantities
Does not touch the manuscript, the supplement or any existing figure. CPU only.
Usage: python scripts/paper_v6_mechanism.py
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "why_it_works"
OUT = ROOT / "paper" / "RESS" / "figures" / "candidates"
TEX_BIN = Path("C:/texlive/2026/bin/windows")
NAME = "tab_mechanism"

STREAMS = (("maude", "Device"), ("sdr", "Aircraft"))  # device before aircraft
YEARS = ("2025", "2026")  # 2025 before 2026
LLMS = (("qwen38_27b", "Qwen3.8-27B"), ("gemma4_31b", "Gemma 4 31B"))  # Qwen before Gemma
CODERS = (("zs", "Zero-shot LLM"), ("rag", "Retrieval LLM"))  # zero-shot before retrieval


# ------------------------------------------------------------------ numbers
def load():
    J = {c: json.loads((SRC / f"{c}.json").read_text(encoding="utf-8")) for c, _ in STREAMS}
    cases = []
    for mk, llm in LLMS:
        for c, stream in STREAMS:
            for y in YEARS:
                m = J[c]["years"][y][mk]["mechanism"]
                n_rare, n_other = m["rare_reports"], m["other_reports"]
                v = {"stream": stream, "year": y, "llm": llm, "corpus": c, "llm_key": mk,
                     "rare_reports": n_rare, "ordinary_reports": n_other}
                for k, _ in CODERS:
                    z = m["confirmers"][k]
                    f_hat, (f_lo, f_hi) = z["llm_false_estimated"], z["llm_false_ci95"]
                    v[k] = {
                        # block 1: answers of the coder that pass the rule
                        "correct_answers": z["llm_right"], "correct_passing": z["right_passing"],
                        "false_answers_est": f_hat, "false_answers_ci95": [f_lo, f_hi],
                        "false_passing": z["false_passing"],
                        "pass_correct": 100 * z["right_passing"] / z["llm_right"],
                        "pass_false": 100 * z["false_passing"] / f_hat,
                        "pass_false_range95": [100 * z["false_passing"] / f_hi, 100 * z["false_passing"] / f_lo],
                        # block 2: shortlisted reports on which the coder names the top rare code
                        "short_filed_under_it": z["shortlisted_candidate_right"],
                        "short_filed_under_other": z["shortlisted_candidate_false"],
                        # numerators: the same reports as correct_passing / false_passing (checked below)
                        "names_if_filed": 100 * z["right_passing"] / z["shortlisted_candidate_right"],
                        "names_if_other": 100 * z["false_passing"] / z["shortlisted_candidate_false"],
                        # block 3: the coder alone
                        "lr_alone": (z["llm_right"] / n_rare) / (f_hat / n_other),
                    }
                    # the stored shares must agree with the recomputed ones (same reports)
                    assert abs(v[k]["names_if_filed"] - 100 * z["confirm_right"]) < 0.01, (c, y, mk, k)
                    assert abs(v[k]["names_if_other"] - 100 * z["confirm_false"]) < 0.001, (c, y, mk, k)
                    assert abs(v[k]["pass_correct"] - 100 * z["keep_right"]) < 0.01, (c, y, mk, k)
                    assert abs(v[k]["pass_false"] - 100 * z["keep_false"]) < 0.001, (c, y, mk, k)
                lr = m["likelihood_ratio"]
                for key, name in (("encoder", "classifier"), ("gate", "ours")):
                    r, f = lr[key]["right"], lr[key]["false"]
                    v["lr_" + name] = (r / n_rare) / (f / n_other)
                    v[name + "_counts"] = [r, f]
                for k, _ in CODERS:
                    assert abs(v[k]["lr_alone"] - lr[f"llm_{k}_alone"]["LR"]) < 0.06, (c, y, mk, k)
                cases.append(v)
    return cases


def half_up(x, nd):
    return Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def fmt(x):
    """Two significant digits below 10 (one decimal), whole numbers from 10, as the 5.3 prose prints them."""
    d = half_up(x, 1)
    return str(half_up(x, 0)) if d >= 10 else str(d)


# ------------------------------------------------------------------ table
CAPTION = ("Rare answers of the LLM coders that pass the agreement rule, shortlisted reports on which they name the top "
           "rare code, and likelihood ratios over each test year.")
NOTE = ("$^\\dagger$Estimated from samples of the year. The other values are exact. "
        "Bold: highest likelihood ratio in each column.")


def rows(cases):
    """(label, [8 cells]) with None for a block heading; cases are in column order."""
    out = [("Rare answers of the LLM coder that pass the rule (\\%)", None)]
    for k, coder in CODERS:
        out.append((f"{coder}, correct answers", [fmt(v[k]["pass_correct"]) for v in cases]))
        out.append((f"{coder}, false answers$^\\dagger$", [fmt(v[k]["pass_false"]) for v in cases]))
    out.append(("Shortlisted reports on which the LLM coder names the top rare code (\\%)", None))
    for k, coder in CODERS:
        out.append((f"{coder}, filed under this code", [fmt(v[k]["names_if_filed"]) for v in cases]))
        out.append((f"{coder}, filed under another code", [fmt(v[k]["names_if_other"]) for v in cases]))
    out.append(("Likelihood ratio", None))
    out.append(("Classifier", [fmt(v["lr_classifier"]) for v in cases]))
    for k, coder in CODERS:
        out.append((f"{coder} alone$^\\dagger$", [fmt(v[k]["lr_alone"]) for v in cases]))
    out.append(("Ours", ["\\textbf{" + fmt(v["lr_ours"]) + "}" for v in cases]))
    return out


def latex(cases):
    head = " & ".join(f"\\makecell{{{v['stream']}\\\\{v['year']}}}" for v in cases)
    L = ["% Candidate table for Section 5.3 (scripts/paper_v6_mechanism.py; source results/why_it_works/{maude,sdr}.json)",
         "\\begin{table}[pos=!htbp]",
         f"\\caption{{{CAPTION}}}",
         "\\label{tab:mechanism}",
         "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}cccccccc@{}}",
         "\\toprule",
         f" & \\multicolumn{{4}}{{c}}{{{LLMS[0][1]}}} & \\multicolumn{{4}}{{c}}{{{LLMS[1][1]}}} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
         f" & {head} \\\\",
         "\\midrule"]
    first = True
    for label, cells in rows(cases):
        if cells is None:
            if not first:
                L.append("\\addlinespace")
            L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
            first = False
        else:
            L.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule",
          "\\end{tabular*}",
          f"\\par\\smallskip{{\\footnotesize\\raggedright\\noindent {NOTE}\\par}}",
          "\\end{table}"]
    return "\n".join(L) + "\n"


PREVIEW = r"""\documentclass[a4paper,fleqn]{cas-sc}
\begin{document}
\pagestyle{empty}
\setcounter{table}{3}
\input{tab_mechanism.tex}
\end{document}
"""


def compile_preview(tex):
    """Compile with the manuscript's class (cas-sc), crop, and render a 150 dpi PNG with TeX Live's Ghostscript."""
    env = dict(os.environ, PATH=str(TEX_BIN) + os.pathsep + os.environ.get("PATH", ""))
    tmp = Path(tempfile.mkdtemp(prefix="mechanism_"))
    try:
        (tmp / f"{NAME}.tex").write_text(tex, encoding="utf-8")
        (tmp / "preview.tex").write_text(PREVIEW, encoding="utf-8")
        run = lambda *a: subprocess.run([str(x) for x in a], cwd=tmp, env=env, check=True,  # noqa: E731
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        run(TEX_BIN / "latexmk.exe", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "preview.tex")
        log = (tmp / "preview.log").read_text(encoding="latin-1")
        bad = [ln for ln in log.splitlines() if "Overfull" in ln or "Underfull \\hbox" in ln]
        run(TEX_BIN / "pdfcrop.exe", "--margins", "12", "preview.pdf", "cropped.pdf")
        run(TEX_BIN / "rungs.exe", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r150",
            "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-sOutputFile=cropped.png", "cropped.pdf")
        shutil.copyfile(tmp / "cropped.pdf", OUT / f"{NAME}_preview.pdf")
        shutil.copyfile(tmp / "cropped.png", OUT / f"{NAME}_preview.png")
        return bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ check against the 5.3 prose
def rng(vals, nd):
    return [float(half_up(min(vals), nd)), float(half_up(max(vals), nd))]


def check(cases):
    """Every number that Section 5.3 (manuscript of 2026-11-19, lines 378-382) states about these quantities.
    Numbers in the text that name no LLM are those with Qwen3.8-27B."""
    q = [v for v in cases if v["llm"] == "Qwen3.8-27B"]
    d26 = next(v for v in q if v["stream"] == "Device" and v["year"] == "2026")
    out = []

    def add(sentence, stated, found):
        out.append({"sentence": sentence, "stated": stated, "table": found, "ok": stated == found})

    add("zero-shot correct rare answers that pass: 38% to 55%", [38.0, 55.0],
        rng([v["zs"]["pass_correct"] for v in q], 0))
    add("zero-shot false rare answers that pass: an estimated 1.1% to 3.0%", [1.1, 3.0],
        rng([v["zs"]["pass_false"] for v in q], 1))
    add("retrieval false rare answers that pass: 2.9% to 9.3%", [2.9, 9.3],
        rng([v["rag"]["pass_false"] for v in q], 1))
    add("zero-shot names the top rare code when it is the filed code: 55% to 87%", [55.0, 87.0],
        rng([v["zs"]["names_if_filed"] for v in q], 0))
    add("zero-shot names it when it is not the filed code: 3.7% to 5.8%", [3.7, 5.8],
        rng([v["zs"]["names_if_other"] for v in q], 1))
    add("retrieval LLM shows the same difference (filed > other in every case, both LLMs)", True,
        all(v["rag"]["names_if_filed"] > v["rag"]["names_if_other"] for v in cases))
    add("device 2026 likelihood ratio: rule 512, classifier 175, zero-shot LLM alone 11", [512, 175, 11],
        [int(half_up(d26["lr_ours"], 0)), int(half_up(d26["lr_classifier"], 0)), int(half_up(d26["zs"]["lr_alone"], 0))])
    add("the rule has the highest ratio of the three in all eight cases", True,
        all(v["lr_ours"] > max(v["lr_classifier"], v["zs"]["lr_alone"]) for v in cases))
    add("(table) the rule's ratio is also above the retrieval LLM alone in all eight cases", True,
        all(v["lr_ours"] > v["rag"]["lr_alone"] for v in cases))
    add("(Discussion) a larger share of the retrieval LLM's false rare answers passes than of the zero-shot LLM's, "
        "all eight cases", True, all(v["rag"]["pass_false"] > v["zs"]["pass_false"] for v in cases))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load()
    tex = latex(cases)
    (OUT / f"{NAME}.tex").write_text(tex, encoding="utf-8")
    checks = check(cases)
    bad = compile_preview(tex)
    printed = {label: cells for label, cells in rows(cases) if cells is not None}
    (OUT / f"{NAME}_values.json").write_text(json.dumps({
        "script": "scripts/paper_v6_mechanism.py",
        "source": "results/why_it_works/{maude,sdr}.json, years.<year>.<llm>.mechanism",
        "columns": [f"{v['llm']} | {v['stream']} {v['year']}" for v in cases],
        "printed": printed, "cases": cases, "checks_5_3": checks,
        "caption_words": len(CAPTION.split()),
        "note_words": len(NOTE.replace("$^\\dagger$", "").split())}, indent=1), encoding="utf-8")
    print(tex)
    for c in checks:
        print(("OK   " if c["ok"] else "FAIL ") + c["sentence"], c["stated"], c["table"])
    print("caption words:", len(CAPTION.split()), " note words:", len(NOTE.replace("$^\\dagger$", "").split()))
    print("overfull/underfull boxes:", bad or "none")
    for ext in (".tex", "_preview.pdf", "_preview.png", "_values.json"):
        print("wrote", OUT / f"{NAME}{ext}")


if __name__ == "__main__":
    main()
