"""Method figure of the v4 paper (notes/paper-plan-v4-2026-11-16.md, figure 1), drawn as PowerPoint shapes with the
helpers and colours of paper_diagrams.py (the author's SMART/SARC figure language): rounded boxes, thin dark lines,
coder colours (encoder blue, LLM red, retrieval LLM orange, ours green), glyphs and one- or two-word labels only.

The figure shows only the core: a rare code is placed when an LLM names a code inside the encoder's top five.
  Coders -> Agreement: the encoder's ranking as a strip of code cells, its top five inside a dashed window (the rare
  candidate c in peach at rank 2); the LLM names c, which lies inside the window (check); the retrieval LLM names
  another rare code d that the encoder ranks far below (cross). c goes over the top to the code (green); without
  a match the selection (trees) picks the code.

The two 2026 reports in EXAMPLES are for the paper text (they are not drawn); python paper_v4_diagram.py
--check-example recomputes every value from the saved outputs and the final configuration of paper_v4_numbers.py.

Writes paper/figures/src/fig_method_v4.pptx and exports paper/figures/v4/fig_method_v4.{pdf,png} through PowerPoint.
Usage: python paper_v4_diagram.py [--no-export] [--check-example]
"""
import subprocess
import sys

import numpy as np
from pptx.enum.shapes import MSO_SHAPE

import paper_diagrams as D
from common import ROOT

OUT = ROOT / "paper" / "RESS" / "figures"
LLM_NAME = "qwen38_27b"

# Two 2026 MAUDE reports (scripts/paper_v4_diagram.py --check-example recomputes all of this).
EXAMPLES = [
    {"which": "2026", "key": 24721764, "quote": "“stem had subsided post-operatively.”",
     "top5": [("Migration", 5776), ("Implant subsidence", 4), ("Osseointegration Problem", 1066),
              ("Migration or Expulsion of Device", 1822), ("Unintended Movement", 2882)],
     "rare_rank": 1, "llm": ("Implant subsidence", 4), "passes": True, "final": "Implant subsidence",
     "filed": "Implant subsidence"},
    {"which": "2026nat", "key": 24955747, "quote": "“it is reported, clogged/defective filter. no patient harm.”",
     "top5": [("Complete Blockage", 1091), ("Blocked Connection", 54), ("Filtration Problem", 9),
              ("Partial Blockage", 801), ("Obstruction of Flow", 22892)],
     "rare_rank": None, "llm": ("Inadequate Filtration Process", 1), "passes": False, "final": "Complete Blockage",
     "filed": "Complete Blockage"},
]


def tree_icon(sl, x, y, w, h, color):
    """A small decision tree: marks a node as boosted trees."""
    r = 0.02
    root = (x + w / 2, y + r)
    mid = [(x + w * 0.22, y + h * 0.52), (x + w * 0.78, y + h * 0.52)]
    leaves = [(x + w * 0.06, y + h - r), (x + w * 0.38, y + h - r), (x + w * 0.62, y + h - r), (x + w * 0.94, y + h - r)]
    for m in mid:
        D.path(sl, [root, m], color=color, lw=0.8, end=None)
    for k, lf in enumerate(leaves):
        D.path(sl, [mid[k // 2], lf], color=color, lw=0.8, end=None)
    for cx, cy in [root] + mid:
        D.shape(sl, MSO_SHAPE.OVAL, cx - r, cy - r, 2 * r, 2 * r, fill=D.WHITE, edge=color, lw=0.8)
    for cx, cy in leaves:
        D.shape(sl, MSO_SHAPE.RECTANGLE, cx - r * 0.9, cy - r * 0.9, 1.8 * r, 1.8 * r, fill=color, edge=None)


def mark(sl, x, y, ok, size=13):
    """A check (green) or a cross (red) in the symbol font."""
    D.label(sl, x, y, 0.30, 0.30, {"t": "[[✓]]" if ok else "[[✗]]", "b": True}, size=size, color=D.WIN if ok else D.RED)


def cell(sl, x, y, w, h, text, fill, edge, lw=1.1):
    """A code: a small rounded cell, peach when the code is rare."""
    D.box(sl, x, y, w, h, [text], fill=fill, edge=edge, lw=lw, size=9.5, radius=0.05)


def method():
    W, H = 7.1, 2.42
    prs, sl = D.new_deck(W, H)
    INK = D.INKD
    (_, ENC_F, ENC_E), (_, LLM_F, LLM_E), (_, RAG_F, RAG_E) = D.CODERS
    r0, r1, r2 = rows = [0.82, 1.40, 1.98]
    TX, TW, TH = 0.60, 1.40, 0.42
    PX0, PX1 = 2.14, 5.08                       # gate panel
    SX, SW = 5.46, 0.56                         # selection
    CX, CW_, CH_ = 6.28, 0.68, 0.42             # code
    for x0, x1, t in ((TX, TX + TW, "Coders"), (PX0, PX1, "Agreement"), (SX - 0.10, SX + SW + 0.10, "Selection")):
        D.label(sl, x0, 0.04, x1 - x0, 0.26, {"t": t, "b": True}, size=10.5)
    D.shape(sl, MSO_SHAPE.RECTANGLE, 6.40, 2.20, 0.13, 0.11, fill=D.TAIL, edge=None)
    D.label(sl, 6.56, 2.155, 0.54, 0.20, "rare", size=8.5, align="l")
    # report
    D.shape(sl, MSO_SHAPE.FOLDED_CORNER, 0.08, r1 - 0.24, 0.38, 0.48, fill=D.WHITE, edge=INK, lw=1.0)
    for k, ln in enumerate((0.24, 0.24, 0.15)):
        yy = r1 - 0.13 + 0.10 * k
        D.path(sl, [(0.15, yy), (0.15 + ln, yy)], color=D.LGRAY, lw=1.3, end=None)
    D.label(sl, -0.02, r1 + 0.26, 0.60, 0.20, "Report *x*", size=9.5)
    D.path(sl, [(0.46, r1), (0.53, r1)], color=INK, end=None)
    D.path(sl, [(0.53, r0), (0.53, r2)], color=INK, end=None)
    for y in rows:
        D.path(sl, [(0.53, y), (TX, y)], color=INK)
    # coders
    for k, ((name, f, e), y) in enumerate(zip(D.CODERS, rows)):
        D.box(sl, TX, y - TH / 2, TW, TH, [""], fill=f, edge=e, lw=1.1, radius=0.08)
        if k == 0:
            D.net_icon(sl, TX + 0.09, y - 0.10, 0.24, 0.20, e)
        elif k == 1:
            D.bubble_icon(sl, TX + 0.07, y - 0.11, 0.28, 0.17, e)
        else:
            D.bubble_icon(sl, TX + 0.05, y - 0.13, 0.23, 0.15, e)
            D.docs_icon(sl, TX + 0.18, y - 0.02, 0.16, 0.14, e)
        D.label(sl, TX + 0.40, y - 0.13, TW - 0.42, 0.26, ("Classifier", "Zero-shot LLM", "Retrieval LLM")[k], size=9.5, align="l")
    # gate: the encoder's ranking, its top five in a window; the LLM answers checked against the window
    D.shape(sl, MSO_SHAPE.ROUNDED_RECTANGLE, PX0, 0.54, PX1 - PX0, 1.72, fill=D.PANEL, edge=None, radius=0.08)
    cw, chh, pitch = 0.30, 0.26, 0.35
    xs = [2.31 + k * pitch for k in range(5)]
    wx0, wx1 = xs[0] - 0.07, xs[-1] + cw + 0.07
    D.shape(sl, MSO_SHAPE.ROUNDED_RECTANGLE, wx0, r0 - 0.20, wx1 - wx0, 0.40, fill=D.WHITE, edge=D.WIN, lw=1.2,
            dash=True, radius=0.08)
    D.path(sl, [(TX + TW, r0), (wx0, r0)], color=INK)
    for k, x in enumerate(xs):
        cell(sl, x, r0 - chh / 2, cw, chh, "*c*" if k == 1 else "", D.TAIL if k == 1 else ENC_F, ENC_E)
    D.label(sl, wx0, r0 + 0.21, wx1 - wx0, 0.18, "top 5", size=8.5, color=D.WIN)
    dx = 4.40
    D.label(sl, wx1 + 0.01, r0 - 0.14, dx - wx1 - 0.02, 0.24, "…", size=11, color=D.GRAY)
    cell(sl, dx, r0 - chh / 2, cw, chh, "*d*", D.TAIL, ENC_E)
    D.label(sl, dx + cw + 0.01, r0 - 0.14, PX1 - dx - cw - 0.06, 0.24, "…", size=11, color=D.GRAY)
    # the LLM names c: inside the window
    lx = xs[1] + cw / 2
    D.path(sl, [(TX + TW, r1), (xs[1], r1)], color=INK)
    cell(sl, xs[1], r1 - chh / 2, cw, chh, "*c*", D.TAIL, LLM_E, lw=1.3)
    D.path(sl, [(lx, r1 - chh / 2), (lx, r0 + chh / 2)], color=D.WIN, lw=1.8, end=None)
    mark(sl, lx - 0.34, (r0 + r1) / 2 - 0.15, True)
    # the retrieval LLM names d: the encoder ranks it far below, outside the window
    rx = dx + cw / 2
    D.path(sl, [(TX + TW, r2), (dx, r2)], color=INK)
    cell(sl, dx, r2 - chh / 2, cw, chh, "*d*", D.TAIL, RAG_E, lw=1.3)
    D.path(sl, [(rx, r2 - chh / 2), (rx, r0 + chh / 2)], color=D.GRAY, lw=1.1, dash=True, end=None)
    mark(sl, rx - 0.36, (r0 + r1) / 2 - 0.15, False)
    # pass: c goes to the code
    yb = 0.40
    D.path(sl, [(lx, r0 - chh / 2), (lx, yb), (CX + CW_ / 2, yb), (CX + CW_ / 2, r1 - CH_ / 2)], color=D.WIN, lw=1.6)
    cell(sl, CX + CW_ / 2 - cw / 2, (yb + r1 - CH_ / 2) / 2 - chh / 2, cw, chh, "*c*", D.TAIL, D.WIN, lw=1.3)
    # no match: the selection picks the code
    D.path(sl, [(PX1, r1), (SX, r1)], color=INK)
    D.box(sl, SX, r1 - 0.20, SW, 0.40, [""], fill=D.SEL_F, edge=D.SEL_E, lw=1.1, radius=0.06, dash=True)
    tree_icon(sl, SX + 0.08, r1 - 0.12, SW - 0.16, 0.24, D.WIN)
    D.path(sl, [(SX + SW, r1), (CX, r1)], color=INK)
    D.box(sl, CX, r1 - CH_ / 2, CW_, CH_, ["*ŷ*"], fill=D.WHITE, edge=D.WIN, lw=1.6, size=11, radius=0.08)
    D.label(sl, CX, r1 + CH_ / 2 + 0.02, CW_, 0.18, "code", size=8.5, color=D.GRAY)
    D.label(sl, SX - 0.10, r1 + 0.22, SW + 0.20, 0.18, "optional", size=8.5, color=D.GRAY)
    return prs


def check_example():
    """Recompute every value of the two example reports from the saved outputs and the final configuration."""
    import candidate_ranker as cr
    import maude_selector as ms
    import paper_v4_numbers as pv
    import redesign_score as rs
    import selector_vs_rules as sv
    from maude_data import POOL_2026, class_info, load_items, load_split
    from rare_check import natural_weights
    info = class_info(load_split())
    leaf = lambda c: info[c]["path"].split(" > ")[-1]  # noqa: E731
    frozen, models = rs.check_frozen(LLM_NAME)
    D25 = cr.load("2025nat", LLM_NAME)
    R = cr.fit(D25, natural_weights({"D": D25}, "2025")[1], "wide_gbm_table")
    casc = cr.fit_cascade(D25, natural_weights({"D": D25}, "2025")[1])
    ok = True
    for ex in EXAMPLES:
        Dx = cr.load(ex["which"], LLM_NAME)
        i = Dx["keys"].index(ex["key"])
        codes, cnt = Dx["codes"], Dx["counts"]
        top5 = np.argsort(-Dx["probs"][i])[:5]
        got = {"top5": [(leaf(codes[c]), int(cnt[c])) for c in top5],
               "llm": (leaf(codes[Dx["zs"][i]]), int(cnt[Dx["zs"][i]])),
               "filed": leaf(codes[Dx["gold"][i]])}
        P = pv.predictions(cr, sv, rs, ms, Dx, LLM_NAME, frozen, models, R, casc)
        fire = pv.gate_parts(Dx)[0]
        got["passes"] = bool(fire[i])
        got["final"] = leaf(codes[P["ours_full"][i]])
        rare_ranks = [k for k, c in enumerate(top5) if 1 <= cnt[c] <= 5]
        got["rare_rank"] = rare_ranks[0] if rare_ranks else None
        text = load_items([ex["key"]], pool=POOL_2026)[ex["key"]][0]
        got["quote"] = "“" + text + "”"
        for k, v in got.items():
            same = v == ex[k]
            ok &= same
            print(f"{ex['key']} {k:9s} {'ok' if same else 'DIFFERS'}: {v}")
    print("all values match" if ok else "SOME VALUES DIFFER")
    return 0 if ok else 1


def main():
    if "--check-example" in sys.argv[1:]:
        return check_example()
    prs = method()
    D.SRC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = D.SRC / "fig_method_v4.pptx"
    prs.save(p)
    print("wrote", p.relative_to(ROOT))
    if "--no-export" not in sys.argv[1:]:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "export_pptx.ps1"),
                        "-Pptx", str(p), "-OutDir", str(OUT)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
