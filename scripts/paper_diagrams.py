"""Conceptual figures for the paper, drawn as PowerPoint shapes so they stay editable.

  fig_overview   (a) training reports per failure mode, MAUDE, from the split's counts;
                 (b) per-report selection on an actual 2025 test report (key 21616844),
                 with the frozen Opus selector's probabilities for it
  fig_framework  the selector in the manner of the SMART architecture and VRNet figures:
                 report -> three coders -> [Selector: phi(x) column with the report's own
                 feature values -> logistic -> p column] -> argmax -> code

Style follows the author's SMART (C&IE) and SARC (TSM) figures: thin black lines and
arrows, white or light-blue boxes with thin dark edges, the module of the paper inside a
large rounded frame titled above it, vectors drawn as tall rounded columns holding
their actual values with an ellipsis, the operation written beside the arrow, emphasis
by coloured text. Ours is green, as in SMART. Explanations go to the caption.

Writes paper/figures/src/<name>.pptx, then exports <name>.pdf and <name>.png into
paper/figures through PowerPoint (scripts/export_pptx.ps1). Re-running this script
overwrites hand edits to the .pptx; after editing by hand, run only the export.

Usage: python paper_diagrams.py [overview framework] [--no-export] [--check-example]
"""
import argparse
import re
import subprocess
import sys

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

from common import ROOT

SRC = ROOT / "paper" / "RESS" / "figures" / "src"
OUT = ROOT / "archive" / "paper_v3" / "figures"  # v3 diagrams; paper/ holds only RESS/ (2026-11-18)
FONT, SYM = "Times New Roman", "Segoe UI Symbol"


def rgb(h):
    return RGBColor.from_string(h)


BLACK, EDGE, GRAY, LGRAY = rgb("000000"), rgb("0D2233"), rgb("6E6E6E"), rgb("BFBFBF")
BOX, PALE, WHITE = rgb("DEEBF7"), rgb("F2F2F2"), rgb("FFFFFF")
COL_PHI, COL_P = rgb("FFF2CC"), rgb("DEEBF7")           # SMART's yellow and blue columns
OURS_FILL, OURS = rgb("E2F0D9"), rgb("2E8B57")          # ours: green
FRAME, ORANGE, RED = rgb("31708F"), rgb("C55A11"), rgb("C00000")
HEAD, MID, TAIL = rgb("9DC3E6"), rgb("DEEBF7"), rgb("F8CBAD")
LW = 1.0  # line width, pt
ENC_F, ENC_E = rgb("DAE8FC"), rgb("6C8EBF")
LLM_F, LLM_E = rgb("F8CECC"), rgb("B85450")
RAG_F, RAG_E = rgb("FFE6CC"), rgb("D79B00")
SEL_F, SEL_E = rgb("D5E8D4"), rgb("82B366")
WIN, BAR, INKD, PANEL = rgb("2E7D32"), rgb("D9D9D9"), rgb("333333"), rgb("F4F4F4")
CODERS = [("Encoder", ENC_F, ENC_E), ("LLM", LLM_F, LLM_E), ("LLM + retrieval", RAG_F, RAG_E)]


def In(x):
    return Emu(int(round(x * 914400)))


# ---------------------------------------------------------------- text
TOKEN = re.compile(r"(\*\*\*.+?\*\*\*|\*\*.+?\*\*|_\{.+?\}|\^\{.+?\}|\*.+?\*|\[\[.+?\]\])")


def runs_of(s):
    """Tiny markup: *italic*, **bold**, ***bold italic***, _{subscript} (may hold *italic*),
    ^{superscript}, [[symbol-font text]]."""
    out = []
    for tok in TOKEN.split(s):
        if not tok:
            continue
        if tok.startswith("***"):
            out.append((tok[3:-3], {"b": True, "i": True}))
        elif tok.startswith("**"):
            out.append((tok[2:-2], {"b": True}))
        elif tok.startswith("_{") or tok.startswith("^{"):
            key = "sub" if tok[0] == "_" else "sup"
            for t, st in runs_of(tok[2:-1]):
                out.append((t, {**st, key: True}))
        elif tok.startswith("[["):
            out.append((tok[2:-2], {"sym": True}))
        elif tok.startswith("*"):
            out.append((tok[1:-1], {"i": True}))
        else:
            out.append((tok, {}))
    return out


def add_par(tf, item, first, size, align, color):
    if isinstance(item, str):
        item = {"t": item}
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[item.get("align", align)]
    p.line_spacing = item.get("ls", 1.0)
    for text, st in runs_of(item["t"]):
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = SYM if st.get("sym") else FONT
        f.size = Pt(item.get("size", size))
        f.color.rgb = item.get("color", color)
        f.bold = bool(item.get("b") or st.get("b"))
        f.italic = bool(item.get("i") or st.get("i"))
        if st.get("sub"):
            r._r.get_or_add_rPr().set("baseline", "-25000")
        if st.get("sup"):
            r._r.get_or_add_rPr().set("baseline", "30000")


def fill_text(tf, items, size=11, align="c", color=BLACK, anchor="m", margin=0.02):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = In(margin)
    tf.margin_top = tf.margin_bottom = In(0)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    for k, it in enumerate(items):
        add_par(tf, it, k == 0, size, align, color)


# ---------------------------------------------------------------- shapes
def strip_style(sh):
    """Drop the theme style reference: no theme shadow, fill or font colour."""
    st = sh._element.find(qn("p:style"))
    if st is not None:
        sh._element.remove(st)


def shape(sl, kind, x, y, w, h, fill=None, edge=EDGE, lw=LW, dash=False, radius=0.08, text=None, **tk):
    sh = sl.shapes.add_shape(kind, In(x), In(y), In(w), In(h))
    strip_style(sh)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if edge is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = edge
        sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = min(0.5, radius / min(w, h))
    fill_text(sh.text_frame, text or [""], **tk)
    return sh


def box(sl, x, y, w, h, text, fill=BOX, rounded=True, **kw):
    tk = {k: kw.pop(k) for k in ("size", "align", "color", "anchor", "margin") if k in kw}
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    return shape(sl, kind, x, y, w, h, fill=fill, text=text, **kw, **tk)


def label(sl, x, y, w, h, items, size=11, align="c", color=BLACK, anchor="m", rot=0):
    tb = sl.shapes.add_textbox(In(x), In(y), In(w), In(h))
    fill_text(tb.text_frame, items if isinstance(items, list) else [items], size=size, align=align, color=color,
              anchor=anchor)
    if rot:
        tb.rotation = rot
    return tb


def path(sl, pts, color=BLACK, lw=LW, dash=False, end="triangle", closed=False, fill=None, start=None):
    fb = sl.shapes.build_freeform(In(pts[0][0]), In(pts[0][1]), scale=1.0)
    fb.add_line_segments([(In(x), In(y)) for x, y in pts[1:]], close=closed)
    sh = fb.convert_to_shape()
    strip_style(sh)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if color is None:
        sh.line.fill.background()
        return sh
    sh.line.color.rgb = color
    sh.line.width = Pt(lw)
    if dash:
        sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    for tag, kind in (("a:headEnd", start), ("a:tailEnd", end)):
        if kind:
            e = etree.SubElement(sh.line._get_or_add_ln(), qn(tag))
            e.set("type", kind)
            e.set("w", "med")
            e.set("len", "med")
    return sh


def column(sl, x, y, w, h, values, fill, colors=None, bold=None, size=11):
    """A vector in the manner of SMART's VRNet figure: a tall rounded column holding its values."""
    shape(sl, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, lw=1.25, radius=0.09)
    step = h / len(values)
    for k, v in enumerate(values):
        label(sl, x, y + k * step, w, step, {"t": v, "color": (colors[k] if colors else BLACK),
                                             "b": bool(bold and bold[k])}, size=size)
    return [y + (k + 0.5) * step for k in range(len(values))]


def card(sl, x, y, w, h):
    """A report: a white card with three text lines."""
    shape(sl, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=WHITE, lw=1.0)
    for k, frac in enumerate((0.72, 0.72, 0.45)):
        yy = y + h * (0.28 + 0.22 * k)
        path(sl, [(x + w * 0.14, yy), (x + w * (0.14 + frac), yy)], color=LGRAY, lw=1.5, end=None)


def trim_top(prs, sl, dy):
    """Move every shape up by dy inches and shorten the slide by as much."""
    for sh in sl.shapes:
        sh.top = sh.top - In(dy)
    prs.slide_height = prs.slide_height - In(dy)


def new_deck(w, h):
    prs = Presentation()
    prs.slide_width, prs.slide_height = In(w), In(h)
    return prs, prs.slides.add_slide(prs.slide_layouts[6])


# ---------------------------------------------------------------- the worked example
# 2025 (development-year) test report 21616844 (balanced test), the redesign's frozen Opus selector
# (splits/maude_selector_v1_cb-large-conv.*, family S2; notes/prereg-redesign-2026-11-06.md). The
# draft-v2 example, report 22083856, no longer shows a correct choice under the redesign's encoder
# (the selector keeps the encoder's answer there). --check-example recomputes every number below.
EXAMPLE = {
    "key": 21616844,
    "text": "“…device shows o2 not available… while respirating.” no patient health impact.",
    "coders": [("Encoder", "Gas Output Problem", 223, 0.20, False),
               ("LLM", "Medical Gas Supply Problem", 5, 0.28, True),
               ("LLM + retrieval", "Gas/Air Leak", 2848, 0.02, False)],
    # phi(x) in FEATURES order, shown to two decimals: log n(y1), msp, log n(y2), zs invalid,
    # log n(y3), rag invalid, neighbour vote share, ..., log report length
    "phi": [5.41, 0.46, 1.79, 0.00, 7.95, 0.00, 0.40],
    "phi_last": 5.57,
    # the twelve features as drawn in fig_framework: (group, FEATURES indices, shown values);
    # the two invalid-answer flags (indices 3 and 5, both 0 here) are left out of the drawing
    "groups": [("Confidence", [1], ["0.46"]),
               ("Neighbours", [6, 7, 8], ["0.40", "1", "0.31"]),
               ("Agreement", [9, 10, 11, 12], ["0", "0", "0", "0"]),
               ("Length", [13], ["5.57"]),
               ("Label frequency", [0, 2, 4], ["5.41", "1.79", "7.95"])],
}


def check_example():
    import json
    import os
    os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")  # the redesign's frozen selector
    import joblib
    import maude_selector as ms
    from maude_data import POOL_2025, class_info, load_items, load_split
    D = ms.load_coders("2025", "opus55")
    mpath, jpath = ms.frozen_paths("opus55")
    frozen = json.loads(mpath.read_text(encoding="utf-8"))
    models = joblib.load(jpath)
    X = ms.features(D)
    P = models[frozen["main_family"]].proba(X)
    pred, j = ms.predict_frozen(D, frozen, models)
    i = D["keys"].index(EXAMPLE["key"])
    info = class_info(load_split())
    leaf = lambda c: info[D["codes"][c]]["path"].split(" > ")[-1]
    got = [(leaf(D[a][i]), int(D["counts"][D[a][i]]), round(float(P[i][k]), 2)) for k, a in enumerate(("enc", "zs", "rag"))]
    want = [(c[1], c[2], c[3]) for c in EXAMPLE["coders"]]
    phi = [round(float(v), 2) for v in X[i][:7]]
    text = load_items([EXAMPLE["key"]], pool=POOL_2025)[EXAMPLE["key"]][0]
    print("text:", text)
    print("gold:", leaf(D["gold"][i]), "| chosen arm:", ["enc", "zs", "rag"][j[i]], "| pred correct:", pred[i] == D["gold"][i])
    print("coders:", got, "| phi[:7]:", phi, "| phi[-1]:", round(float(X[i][-1]), 2))
    assert got == want, (got, want)
    assert phi == EXAMPLE["phi"] and round(float(X[i][-1]), 2) == EXAMPLE["phi_last"], phi
    for name, idx, shown in EXAMPLE["groups"]:
        vals = [float(X[i][k]) for k in idx]
        want_v = [float(v) for v in shown]
        assert all(abs(a - b) < 0.006 for a, b in zip(vals, want_v)), (name, vals, shown)
    print("drawn feature groups match")
    print("example matches")


# ---------------------------------------------------------------- overview
def overview(counts):
    W, H = 7.0, 2.5
    prs, sl = new_deck(W, H)
    # (a) rank-frequency of training reports per mode, MAUDE, log scale
    L, R, T, B = 0.50, 2.95, 0.20, 1.80
    c = np.sort(np.asarray(counts))[::-1]
    n = len(c)
    top = np.log10(c[0]) + 0.3
    yv = lambda v: B - (B - T) * (np.log10(v) + 0.3) / top
    xv = lambda r: L + 0.04 + (R - L - 0.06) * r / n
    for (lo, hi), col in (((90, 10**9), HEAD), ((6, 89), MID), ((1, 5), TAIL)):
        idx = np.where((c >= lo) & (c <= hi))[0]
        pts = [(xv(idx[0]), B)]
        for r in idx:
            pts += [(xv(r), yv(c[r])), (xv(r + 1), yv(c[r]))]
        path(sl, pts + [(xv(idx[-1] + 1), B)], color=None, closed=True, fill=col, end=None)
    pts = [(xv(0), B)]
    for r in range(n):
        pts += [(xv(r), yv(c[r])), (xv(r + 1), yv(c[r]))]
    path(sl, pts + [(xv(n), B)], color=EDGE, lw=0.6, end=None)
    path(sl, [(L, T - 0.04), (L, B), (R, B)], color=BLACK, lw=0.75, end=None)
    for v, s in ((1, "1"), (10, "10"), (100, "100"), (1000, "1k"), (10000, "10k")):
        y = yv(v)
        path(sl, [(L - 0.04, y), (L, y)], color=BLACK, lw=0.75, end=None)
        label(sl, 0.12, y - 0.08, L - 0.17, 0.16, s, size=8.5, align="r")
    label(sl, -0.585, 0.92, 1.35, 0.18, "reports per mode", size=9.5, rot=270)
    label(sl, L, B + 0.03, R - L, 0.18, "failure modes", size=9.5)
    label(sl, xv(20), 1.30, 1.2, 0.22, "common modes", size=10, align="l")
    label(sl, 2.06, 0.70, 0.9, 0.22, "rare modes", size=10)
    path(sl, [(2.62, 0.93), (xv(n - 23), yv(4) - 0.05)], lw=0.9)
    label(sl, 0.0, 2.22, 3.0, 0.22, "(a) Training reports per mode", size=9.5)

    # (b) per-report selection on the worked example
    X0, X1 = 3.22, 6.95
    box(sl, X0, 0.08, X1 - X0, 0.40, [{"t": EXAMPLE["text"], "i": True}], fill=PALE, edge=None, size=9, margin=0.08)
    cw, gap = 1.17, 0.11
    for k, (coder, code, nn, p, right) in enumerate(EXAMPLE["coders"]):
        x = X0 + k * (cw + gap)
        mid = x + cw / 2
        path(sl, [(mid, 0.48), (mid, 0.64)])
        _, cf, ce = CODERS[k]
        box(sl, x, 0.64, cw, 0.32, [coder], fill=cf, edge=ce, lw=1.2, size=9.5)
        path(sl, [(mid, 0.96), (mid, 1.10)])
        mark = "[[✓]]" if right else "[[✗]]"
        label(sl, x - 0.03, 1.10, cw + 0.06, 0.32, f"{code} {mark}", size=8.8)
        label(sl, x, 1.42, cw, 0.16, f"*n* = {nn:,}   *p* = {p:.2f}", size=8.5, color=GRAY)
        if right:
            box(sl, x - 0.02, 1.07, cw + 0.04, 0.53, [""], fill=None, edge=OURS, lw=1.5)
            path(sl, [(mid, 1.60), (mid, 1.80)], color=OURS, lw=1.5)
    for shp in sl.shapes:  # colour the check and the crosses
        if shp.has_text_frame:
            for p in shp.text_frame.paragraphs:
                for r in p.runs:
                    if r.text == "✓":
                        r.font.color.rgb = OURS
                    elif r.text == "✗":
                        r.font.color.rgb = RED
    box(sl, X0, 1.80, X1 - X0, 0.32, ["Selector: max *p*"], fill=OURS_FILL, size=9.5)
    label(sl, X0, 2.22, X1 - X0, 0.22, "(b) Selection on one 2025 report", size=9.5)
    return prs


# ---------------------------------------------------------------- framework
def sigmoid_icon(sl, x, y, w, h, color=EDGE, lw=1.1):
    """A small logistic curve: marks a node as a logistic model."""
    xs = np.linspace(-4, 4, 17)
    ys = 1 / (1 + np.exp(-xs))
    pts = [(x + w * (k / (len(xs) - 1)), y + h * (1 - v)) for k, v in enumerate(ys)]
    path(sl, pts, color=color, lw=lw, end=None)


def gauge(sl, x, y, w, h, frac, fill):
    """A level bar for a probability or a score; no number printed."""
    shape(sl, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=WHITE, edge=GRAY, lw=0.6)
    if frac > 0:
        shape(sl, MSO_SHAPE.RECTANGLE, x, y, w * frac, h, fill=fill, edge=None)
        shape(sl, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=None, edge=GRAY, lw=0.6)


def net_icon(sl, x, y, w, h, color):
    """A fine-tuned network: three layers of nodes, fully connected."""
    cols, rows = [x, x + w / 2, x + w], [y, y + h / 2, y + h]
    for c0, c1 in zip(cols[:-1], cols[1:]):
        for y0 in rows:
            for y1 in rows:
                path(sl, [(c0, y0), (c1, y1)], color=color, lw=0.35, end=None)
    r = 0.024
    for cx in cols:
        for cy in rows:
            shape(sl, MSO_SHAPE.OVAL, cx - r, cy - r, 2 * r, 2 * r, fill=WHITE, edge=color, lw=0.75)


def bubble_icon(sl, x, y, w, h, color):
    """A prompted LLM: a speech bubble."""
    sh = shape(sl, MSO_SHAPE.ROUNDED_RECTANGULAR_CALLOUT, x, y, w, h, fill=WHITE, edge=color, lw=0.9)
    sh.adjustments[0], sh.adjustments[1] = -0.22, 0.78
    for k in range(3):
        cx, cy = x + w * (0.30 + 0.20 * k), y + h * 0.5
        shape(sl, MSO_SHAPE.OVAL, cx - 0.015, cy - 0.015, 0.03, 0.03, fill=color, edge=None)


def docs_icon(sl, x, y, w, h, color, fill=WHITE, text=None, size=8.5):
    """Coded reports: a stack of documents."""
    return shape(sl, MSO_SHAPE.FLOWCHART_MULTIDOCUMENT, x, y, w, h, fill=fill, edge=color, lw=0.9, text=text, size=size)


def tail_chart(sl, x, y, w, h, counts, n_answer, color, nbars=20):
    """Fig. 1a in miniature: training reports per code, most to least frequent, log scale;
    the bar of the answer's code in the coder's colour."""
    c = np.sort(np.asarray(counts))[::-1]
    n = len(c)
    lg = lambda v: np.log10(v + 1) / np.log10(c[0] + 1)
    bw = w / nbars
    hi = min(nbars - 1, int((c > n_answer).sum() / n * nbars))
    for k in range(nbars):
        v = n_answer if k == hi else c[int((k + 0.5) * n / nbars)]
        bh = max(0.014, h * lg(v))
        shape(sl, MSO_SHAPE.RECTANGLE, x + k * bw + bw * 0.12, y + h - bh, bw * 0.76, bh,
              fill=color if k == hi else BAR, edge=None)
    path(sl, [(x, y + h), (x + w, y + h)], color=GRAY, lw=0.6, end=None)


def vote_grid(sl, x, y, k_match, color, cell=0.056, gap=0.014):
    """The ten nearest coded reports, coloured where they carry the retrieval answer's code."""
    for i in range(10):
        r, cc = divmod(i, 5)
        shape(sl, MSO_SHAPE.RECTANGLE, x + cc * (cell + gap), y + r * (cell + gap), cell, cell,
              fill=color if i < k_match else BAR, edge=None)


def framework(counts):
    """The proposed coder. Three coders answer; the evidence about the three answers is mainly
    where each answer's code sits among the training-report counts (Fig. 1a in miniature),
    plus agreement between answers, the encoder's softmax score and the vote of the ten
    nearest coded reports; one logistic model per coder turns all of it into the chance that
    its answer is right, and the most likely answer is kept. Glyphs follow the Fig. 1b report:
    answer counts 223, 5, 2,848; softmax 0.33; 4 of 10 neighbours carry the retrieval answer;
    p = 0.21, 0.41, 0.09. No numbers are printed."""
    W, H = 7.1, 2.74
    prs, sl = new_deck(W, H)
    rows = [0.68, 1.33, 1.98]
    n_ans = [c[2] for c in EXAMPLE["coders"]]
    p_ans = [c[3] for c in EXAMPLE["coders"]]
    win = int(np.argmax(p_ans))
    for x0, x1, t in ((0.80, 2.34, "Coders"), (3.12, 4.66, "Evidence"), (4.82, 5.92, "Selector")):
        label(sl, x0, 0.08, x1 - x0, 0.26, {"t": t, "b": True}, size=10.5)
    # report
    shape(sl, MSO_SHAPE.FOLDED_CORNER, 0.10, 1.08, 0.40, 0.50, fill=WHITE, edge=INKD, lw=1.0)
    for k, ln in enumerate((0.26, 0.26, 0.17)):
        path(sl, [(0.17, 1.20 + 0.10 * k), (0.17 + ln, 1.20 + 0.10 * k)], color=LGRAY, lw=1.4, end=None)
    label(sl, -0.04, 1.62, 0.68, 0.22, "Report *x*", size=10)
    path(sl, [(0.50, rows[1]), (0.64, rows[1])], color=INKD, end=None)
    path(sl, [(0.64, rows[0]), (0.64, rows[2])], color=INKD, end=None)
    for y in rows:
        path(sl, [(0.64, y), (0.80, y)], color=INKD)
    # coders, each with its icon
    TX, TW, TH = 0.80, 1.54, 0.46
    for k, ((name, f, e), y) in enumerate(zip(CODERS, rows)):
        box(sl, TX, y - TH / 2, TW, TH, [""], fill=f, edge=e, lw=1.2, radius=0.10)
        if k == 0:
            net_icon(sl, TX + 0.10, y - 0.11, 0.26, 0.22, e)
        elif k == 1:
            bubble_icon(sl, TX + 0.08, y - 0.12, 0.30, 0.19, e)
        else:
            bubble_icon(sl, TX + 0.06, y - 0.15, 0.25, 0.16, e)
            docs_icon(sl, TX + 0.21, y - 0.03, 0.18, 0.15, e)
        label(sl, TX + 0.45, y - 0.13, TW - 0.47, 0.26, name, size=10.5, align="l")
    # coded reports: retrieval for the third coder, counts and neighbours for the evidence
    docs_icon(sl, 1.09, 2.32, 0.98, 0.36, INKD, fill=PALE, text=[{"t": "Coded reports"}], size=8.5)
    path(sl, [(1.57, 2.32), (1.57, rows[2] + TH / 2)], color=INKD)
    path(sl, [(2.07, 2.50), (4.36, 2.50), (4.36, 2.31)], color=INKD)
    # answers, with their (dis)agreement
    for k, ((name, f, e), y) in enumerate(zip(CODERS, rows)):
        path(sl, [(TX + TW, y), (2.54, y)], color=INKD)
        box(sl, 2.54, y - 0.15, 0.42, 0.30, [f"*\u0177*_{{{k + 1}}}"], fill=f, edge=e, lw=1.2, size=10.5, radius=0.08)
    for y in ((rows[0] + rows[1]) / 2, (rows[1] + rows[2]) / 2):
        label(sl, 2.54, y - 0.14, 0.42, 0.28, "\u2260", size=13, color=GRAY)
    # evidence
    shape(sl, MSO_SHAPE.ROUNDED_RECTANGLE, 3.12, 0.38, 1.54, 1.93, fill=PANEL, edge=None, radius=0.10)
    for k, ((name, f, e), y) in enumerate(zip(CODERS, rows)):
        path(sl, [(2.96, y), (3.20, y)], color=INKD)
        tail_chart(sl, 3.22, y - 0.19, 0.86, 0.38, counts, n_ans[k], e)
    label(sl, 3.18, rows[2] + 0.20, 0.94, 0.16, "label frequency", size=7.5, color=GRAY)
    gauge(sl, 4.18, rows[0] - 0.05, 0.38, 0.10, EXAMPLE["phi"][1], ENC_E)
    label(sl, 4.10, rows[0] + 0.07, 0.54, 0.16, "softmax", size=7.5, color=GRAY)
    vote_grid(sl, 4.19, rows[2] - 0.09, int(round(EXAMPLE["phi"][6] * 10)), RAG_E)
    label(sl, 4.10, rows[2] + 0.06, 0.54, 0.16, "10 nearest", size=7.5, color=GRAY)
    # everything to every head
    path(sl, [(4.66, rows[1]), (4.74, rows[1])], color=INKD, end=None)
    path(sl, [(4.74, rows[0]), (4.74, rows[2])], color=INKD, end=None)
    # selector: one logistic model per coder, its estimate, the most likely answer
    for k, y in enumerate(rows):
        path(sl, [(4.74, y), (4.84, y)], color=INKD)
        shape(sl, MSO_SHAPE.OVAL, 4.84, y - 0.18, 0.36, 0.36, fill=SEL_F, edge=SEL_E, lw=1.2)
        sigmoid_icon(sl, 4.91, y - 0.08, 0.22, 0.16, color=WIN, lw=1.2)
        path(sl, [(5.20, y), (5.28, y)], color=INKD, end=None)
        gauge(sl, 5.28, y - 0.06, 0.46, 0.12, p_ans[k], WIN if k == win else rgb("BDBDBD"))
    label(sl, 5.28, rows[0] - 0.32, 0.46, 0.22, "*p*", size=10)
    for k in (0, 2):
        path(sl, [(5.74, rows[k]), (5.88, rows[k])], color=INKD, end=None)
    path(sl, [(5.88, rows[0]), (5.88, rows[2])], color=INKD, end=None)
    label(sl, 5.91, rows[0] - 0.12, 0.40, 0.22, "max", size=9, align="l")
    path(sl, [(5.74, rows[win]), (6.14, rows[win])], color=WIN, lw=1.6)
    _, wf, _ = CODERS[win]
    box(sl, 6.14, rows[win] - 0.17, 0.80, 0.34, [f"*\u0177* = *\u0177*_{{{win + 1}}}"], fill=wf, edge=WIN, lw=1.6,
        size=11, radius=0.08)
    label(sl, 6.14, rows[win] + 0.19, 0.80, 0.20, "code", size=9, color=GRAY)
    return prs


# ---------------------------------------------------------------- main
def training_counts_maude():
    from maude_data import class_info, load_split
    from maude_router import training_counts
    split = load_split()
    info = class_info(split)
    allc = sorted(info)
    cnt = training_counts(split, allc)["full"]
    known = [cnt[i] for i, c in enumerate(allc) if not info[c]["hidden"]]
    return [int(v) for v in known if v > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["overview", "framework"])
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--check-example", action="store_true")
    a = ap.parse_args()
    if a.check_example:
        check_example()
        return
    SRC.mkdir(parents=True, exist_ok=True)
    built = []
    for name in a.figs:
        prs = {"overview": lambda: overview(training_counts_maude()), "framework": lambda: framework(training_counts_maude())}[name]()
        p = SRC / f"fig_{name}.pptx"
        prs.save(p)
        built.append(str(p))
        print("wrote", p.relative_to(ROOT))
    if not a.no_export:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(ROOT / "scripts" / "export_pptx.ps1"), "-Pptx", ";".join(built), "-OutDir", str(OUT)], check=True)


if __name__ == "__main__":
    sys.exit(main())
