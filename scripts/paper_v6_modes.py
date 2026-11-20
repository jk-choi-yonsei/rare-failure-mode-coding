"""Fig. 4 of the RESS manuscript: correct and false placements of each rare code over a test year, the classifier
against Ours, device reports above aircraft reports.

Rows: rare codes placed correctly at least once, or placed five or more times, by one of the methods drawn; sorted by
the larger correct count, then the summed correct count, then the larger false count. Each row joins the classifier
(blue circle) and Ours (green diamond; filled = Qwen3.8-27B, hollow = Gemma 4 31B, as in paper_v5_figures.py) with a
grey line, in a panel of correct and a panel of false placements. Left of the panels: reports filed under the code in
the year. Last row: all rare codes, i.e. the totals of Table 2 and the rare reports of its note.

Data: results/reliability/<corpus>.json (per rare code: reports filed, placed, placed correctly; written by
scripts/reliability_year.py). Checks before drawing (the script stops on a mismatch): the per-code counts sum to the
Classifier and Ours cells of Table 2 (parsed from paper/RESS/manuscript.tex) and to results/rare_year/*metrics.json;
the filed counts sum to the rare reports of Table 2's note; the codes with a correct placement and the codes placed
only falsely match Table 3; every filed code agrees with results/why_it_works/case_<corpus>.json.

Outputs in paper/RESS/figures/candidates/ (each figure as .pdf and a 150 dpi .png preview):
  fig_rare_codes_2026_qwen   main-text candidate: 2026, Ours with Qwen3.8-27B
  fig_rare_codes_2026_gemma  supplement: 2026, Gemma 4 31B
  fig_rare_codes_2025_qwen   supplement: 2025, Qwen3.8-27B
  fig_rare_codes_2025_gemma  supplement: 2025, Gemma 4 31B
  fig_rare_codes_2026_both   alternative main figure with both LLMs (the paper's main figures show both LLMs)
  fig_rare_codes_2025_both   its 2025 counterpart
  fig_rare_codes_rows.json   the numbers drawn, and how the counts move between codes
  fig_rare_codes_why_<corpus>.json  (--why) for the classifier's most frequent false rare answers: the filed codes of
                             those reports and whether either LLM coder names the code there
  fig_rare_codes_page_preview.pdf/.png  not written here: the 2026 figures (Qwen, both LLMs) set at 0.92 of the text
                             width in the manuscript's class (cas-sc) with the proposed captions, compiled by hand
CPU only. Usage: python paper_v6_modes.py
                 python paper_v6_modes.py --why ; MAINT_CORPUS=sdr python paper_v6_modes.py --why
"""
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

import paper_v5_figures as v5  # noqa: E402  (rcParams, colours and marker fill of the data figures)
from common import ROOT  # noqa: E402

OUT = ROOT / "paper" / "RESS" / "figures" / "candidates"
MANUSCRIPT = ROOT / "paper" / "RESS" / "manuscript.tex"
STREAMS = (("maude", "Device"), ("sdr", "Aircraft"))
YEARS = ("2025", "2026")
QWEN, GEMMA = v5.LLMS
CL, OU, INK, MUTED, GRID = v5.CL, v5.OU, v5.INK, v5.MUTED, v5.GRID
JOIN, RULE = "#D1D5DB", "#9CA3AF"

# layout in inches (the data figures are drawn 7.0 in wide and set at 0.92 of the text width)
FIG_W, PITCH, FS = 7.0, 0.125, 8.0  # figure width, row pitch, code-name font size
FILED_W, GAP, RIGHT = 0.30, 0.16, 0.10
HEAD, FOOT, BETWEEN, LEGEND = 0.40, 0.24, 0.18, 0.30


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


REL = {c: load(f"results/reliability/{c}.json") for c, _ in STREAMS}
CASE = {c: load(f"results/why_it_works/case_{c}.json") for c, _ in STREAMS}
RY = {"maude": load("results/rare_year/metrics.json"), "sdr": load("results/rare_year/sdr_metrics.json")}


def per_code(c, y):
    """{code: {"filed": n, "classifier": (correct, false), <llm>: (correct, false)}} for every rare code filed or placed
    in the year. A code missing from a method's list was filed nowhere and not placed by that method."""
    Y = REL[c]["years"][y]
    codes = {}
    for key, rows in [("classifier", Y["encoder"]["per_mode"])] + [(llm, Y["gate"][llm]["per_mode"]) for llm, _, _ in v5.LLMS]:
        for name, filed, placed, right in rows:
            d = codes.setdefault(name, {"filed": filed})
            assert d["filed"] == filed, (c, y, name)
            d[key] = (right, placed - right)
    for d in codes.values():
        for key in ["classifier"] + [llm for llm, _, _ in v5.LLMS]:
            d.setdefault(key, (0, 0))
    return codes


# ------------------------------------------------------------------ checks against the manuscript and the other files
def table(label):
    """Numeric rows of a manuscript table: {(group, row name): [numbers]}, group = the italic line above the row."""
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    start = tex.index(f"\\label{{{label}}}")
    body = tex[start:tex.index("\\end{table}", start)]
    group, rows = "", {}
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r"\\multicolumn\{\d+\}\{@\{\}l\}\{\\textit\{(.+?)\}\}", line)
        if m:
            group = m.group(1)
            continue
        if "&" not in line or not line.endswith("\\\\"):
            continue
        cells = [s.strip() for s in line[:-2].split("&")]
        nums = [re.fullmatch(r"([\d,]+)(\$\^\\dagger\$)?", s) for s in cells[1:]]
        if cells[0] and all(nums):
            rows[(group, cells[0].replace("\\quad", "").strip())] = [int(n.group(1).replace(",", "")) for n in nums]
    return rows, body


def check():
    t2, body2 = table("tab:rare")
    t3, _ = table("tab:modes")
    rare_reports = [int(v.replace(",", "")) for v in
                    re.search(r"Rare reports: ([\d,]+), ([\d,]+), ([\d,]+) and ([\d,]+)\.", body2).groups()]
    for col, (c, y) in enumerate((c, y) for c, _ in STREAMS for y in YEARS):  # columns of Tables 2 and 3
        codes = per_code(c, y)
        filed = sum(d["filed"] for d in codes.values())
        assert filed == rare_reports[col], (c, y, filed, rare_reports[col])
        R = RY[c]["years"][y]
        methods = [("classifier", ("", "Classifier"), "Classifier", R["encoder_alone"])]
        methods += [(llm, (lab, "Ours"), f"Ours, {lab}", R["llms"][llm]["rules"]["k5/either"]["confirmed_only"])
                    for llm, lab, _ in v5.LLMS]
        line = []
        for key, t2row, t3row, ry in methods:
            right = sum(d[key][0] for d in codes.values())
            false = sum(d[key][1] for d in codes.values())
            assert [right, false] == t2[t2row][2 * col:2 * col + 2], (c, y, key, right, false, t2[t2row])
            assert (right, false) == (ry["right"], ry["false"]), (c, y, key)
            found = sum(d[key][0] > 0 for d in codes.values())
            alarms = sum(d[key][0] == 0 and d[key][1] > 0 for d in codes.values())
            assert found == t3[("Rare modes found", t3row)][col], (c, y, key, found)
            assert alarms == t3[("False alarms", t3row)][col], (c, y, key, alarms)
            line.append(f"{key} {right}/{false} (found {found}, false alarms {alarms})")
        for p in CASE[c]["years"][y]["per_code"]:
            d = codes[p["code"].split(" > ")[-1]]
            assert d["filed"] == p["filed"] and list(d["classifier"]) == p["encoder"], (c, y, p["code"])
            for llm, _, _ in v5.LLMS:
                assert list(d[llm]) == p[f"gate_{llm}"], (c, y, llm, p["code"])
        print(f"check {c} {y}: {len(codes)} rare codes, {filed} rare reports; " + "; ".join(line)
              + f"; {len(CASE[c]['years'][y]['per_code'])} filed codes agree with case_{c}.json")
    print("check passed: per-code counts sum to Tables 2 and 3 and to rare_year metrics")


# ------------------------------------------------------------------ figure
def select(codes, keys):
    """Codes placed correctly at least once, or placed five or more times, by one of the methods `keys`."""
    rows = [n for n, d in codes.items() if any(d[k][0] > 0 or sum(d[k]) >= 5 for k in keys)]
    return sorted(rows, key=lambda n: (-max(codes[n][k][0] for k in keys), -sum(codes[n][k][0] for k in keys),
                                       -max(codes[n][k][1] for k in keys), n))


def text_width(strings, fontsize):
    fig = plt.figure()
    r = fig.canvas.get_renderer()
    w = max(fig.text(0, 0, s, fontsize=fontsize).get_window_extent(r).width for s in strings) / fig.dpi
    plt.close(fig)
    return w


def marker_style(key, filled=True):
    """The classifier's circle is drawn smaller and on top of Ours' diamond, so a tie shows both (a dot in a diamond)."""
    if key == "classifier":
        return {"marker": "o", "ms": 5.0, "color": CL, "mec": "white", "mew": 0.6, "ls": "none"}
    return {"marker": "D", "ms": 5.6, "color": OU, "ls": "none", **v5.mk(OU, filled)}


def totals(ax, y, items):
    """Marker and total of each method in the last row, left-aligned in the panel."""
    children = []
    for i, (style, value) in enumerate(items):
        da = DrawingArea(9, 8, 0, 0)
        da.add_artist(Line2D([4.5], [4], **style))
        children += [da, TextArea(f"{value:,}", textprops={"fontsize": FS + 0.5, "color": INK})]
        if i < len(items) - 1:
            children.append(TextArea("  ", textprops={"fontsize": FS}))
    ab = AnnotationBbox(HPacker(children=children, align="center", pad=0, sep=1.5), (0.0, y),
                        xycoords=blended_transform_factory(ax.transAxes, ax.transData), box_alignment=(0, 0.5),
                        frameon=False, pad=0, annotation_clip=False)
    ax.add_artist(ab)


def block(fig, top, left, pw, c, title, letter, year, codes, rows, ours, direct_labels):
    """One stream: code names, reports filed, correct placements, false placements. `top` in inches from the bottom."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    keys = ["classifier"] + [llm for llm, _, _ in ours]
    n = len(rows)
    y_top, y_foot, y_bot = -0.75 - (1.05 if direct_labels else 0.0), n + 0.5, n + 1.05
    h = (y_bot - y_top) * PITCH
    y0 = top - HEAD - h
    x_f = left
    x_c = x_f + FILED_W + GAP
    x_x = x_c + pw + GAP
    axf = fig.add_axes([x_f / W, y0 / H, FILED_W / W, h / H])
    axc = fig.add_axes([x_c / W, y0 / H, pw / W, h / H], sharey=axf)
    axx = fig.add_axes([x_x / W, y0 / H, pw / W, h / H], sharey=axf)
    fig.text(0.0, (top - 0.02) / H, f"({letter}) {title} reports, {year}", ha="left", va="top", fontsize=10, color=INK)

    # code names and reports filed
    axf.set_ylim(y_bot, y_top)
    axf.set_yticks(list(range(n)) + [y_foot])
    axf.set_yticklabels(rows + ["All rare codes"], fontsize=FS, color=INK)
    axf.tick_params(axis="y", length=0, pad=4)
    axf.tick_params(axis="x", bottom=False, labelbottom=False)
    axf.grid(False)
    for s in axf.spines.values():
        s.set_visible(False)
    axf.set_xlim(0, 1)
    for i, r in enumerate(rows):
        axf.text(0.92, i, f"{codes[r]['filed']:,}", ha="right", va="center", fontsize=FS, color=MUTED)
    axf.text(0.92, y_foot, f"{sum(d['filed'] for d in codes.values()):,}", ha="right", va="center", fontsize=FS + 0.5,
             color=INK)
    axf.set_title("Filed", fontsize=8, color=MUTED, loc="right", pad=3)

    for ax, j, head in ((axc, 0, "Correct placements"), (axx, 1, "False placements")):
        vmax = max(codes[r][k][j] for r in rows for k in keys)
        step = 5 if vmax <= 20 else (10 if vmax <= 60 else 20)
        for i, r in enumerate(rows):
            vals = [codes[r][k][j] for k in keys]
            if max(vals) > min(vals):
                ax.plot([min(vals), max(vals)], [i, i], color=JOIN, lw=2.2, solid_capstyle="round", zorder=2)
            cs = marker_style("classifier")
            if vals[0] in vals[1:]:
                cs.update(ms=3.0, mec=CL, mew=0)
            ax.plot(vals[0], i, zorder=6, **cs)
            for (llm, _, filled), v in zip(ours, vals[1:]):
                dy = 0.0 if len(ours) == 1 else (-0.2 if filled else 0.2)
                ax.plot(v, i + dy, zorder=4 if filled else 5, **marker_style(llm, filled))
        tot = [(marker_style("classifier"), sum(d["classifier"][j] for d in codes.values()))]
        tot += [(marker_style(llm, filled), sum(d[llm][j] for d in codes.values())) for llm, _, filled in ours]
        totals(ax, y_foot, tot)
        ax.set_xlim(-0.04 * vmax, vmax * 1.06 + 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        for i in range(n):
            ax.axhline(i, color=GRID, lw=0.6, zorder=0)
        ax.set_title(head, fontsize=8.5, color=INK, loc="left", pad=3)

    # rule above the last row, from the code names to the right end of the false panel
    y_rule = y0 + h * (y_bot - (n - 0.5 + 0.12)) / (y_bot - y_top)
    fig.add_artist(Line2D([0.0, (x_x + pw) / W], [y_rule / H] * 2, transform=fig.transFigure, color=RULE, lw=0.6))

    if direct_labels:  # name the markers once, above the first row of the correct panel
        r0 = codes[rows[0]]
        xc, xo = r0["classifier"][0], r0[ours[0][0]][0]
        span = axc.get_xlim()[1] - axc.get_xlim()[0]
        far = abs(xo - xc) > 0.3 * span
        yl = y_top + 0.62
        axc.text(xc, yl, "Classifier", ha="center" if far else "right", va="center", fontsize=8, color=CL)
        axc.text(xo, yl, "Ours", ha="center" if far else "left", va="center", fontsize=8.5, color=OU,
                 fontweight="bold")
    return y0 - FOOT


def figure(year, ours, name):
    keys = ["classifier"] + [llm for llm, _, _ in ours]
    data = []
    for (c, title), letter in zip(STREAMS, "ab"):
        codes = per_code(c, year)
        data.append((c, title, letter, codes, select(codes, keys)))
    left = text_width([r for *_, rows in data for r in rows] + ["All rare codes"], FS) + 0.10
    pw = (FIG_W - left - FILED_W - 2 * GAP - RIGHT) / 2
    single = len(ours) == 1
    heights = [HEAD + (len(rows) + 1.8 + (1.05 if single and k == 0 else 0)) * PITCH + FOOT
               for k, (*_, rows) in enumerate(data)]
    H = sum(heights) + BETWEEN + (0 if single else LEGEND)
    fig = plt.figure(figsize=(FIG_W, H))
    top = H
    for k, (c, title, letter, codes, rows) in enumerate(data):
        bottom = block(fig, top, left, pw, c, title, letter, year, codes, rows, ours, direct_labels=single and k == 0)
        top = bottom - BETWEEN
    if not single:
        handles = [Line2D([], [], **marker_style("classifier"), label="Classifier")]
        handles += [Line2D([], [], **marker_style(llm, filled), label=f"Ours, {lab}") for llm, lab, filled in ours]
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, (LEGEND - 0.02) / H), fontsize=8,
                   handletextpad=0.3, columnspacing=1.6, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png ({FIG_W:.1f} x {H:.2f} in; rows {', '.join(str(len(d[-1])) for d in data)})")
    return {"year": year, "ours": [lab for _, lab, _ in ours],
            "streams": {c: {"rows": [{"code": r, "filed": codes[r]["filed"], **{k: list(codes[r][k]) for k in keys}}
                                     for r in rows],
                            "all_rare_codes": {"filed": sum(d["filed"] for d in codes.values()),
                                               **{k: [sum(d[k][j] for d in codes.values()) for j in (0, 1)]
                                                  for k in keys}}}
                        for c, _, _, codes, rows in data}}


def code_changes(top=6):
    """How the counts move between codes from the classifier to Ours, per stream, year and LLM (for the text):
    codes with more / fewer correct placements, false placements removed / added summed over codes (net = Table 2),
    the share of each method's false placements that fall under codes it also places correctly at least once, and the
    codes whose false placements fall or rise most."""
    out = {}
    for c, _ in STREAMS:
        for y in YEARS:
            codes = per_code(c, y)
            for llm, _, _ in v5.LLMS:
                cl = {n: d["classifier"] for n, d in codes.items()}
                ou = {n: d[llm] for n, d in codes.items()}

                def share_found(m):
                    return round(sum(f for r, f in m.values() if r > 0) / max(sum(f for _, f in m.values()), 1), 3)

                def item(n):
                    return {"code": n, "filed": codes[n]["filed"], "classifier": list(cl[n]), "ours": list(ou[n])}
                out[f"{c} {y} {llm}"] = {
                    "codes_more_correct": sum(ou[n][0] > cl[n][0] for n in codes),
                    "codes_fewer_correct": sum(ou[n][0] < cl[n][0] for n in codes),
                    "false_removed": sum(max(0, cl[n][1] - ou[n][1]) for n in codes),
                    "false_added": sum(max(0, ou[n][1] - cl[n][1]) for n in codes),
                    "share_of_false_under_codes_also_placed_correctly": {"classifier": share_found(cl),
                                                                         "ours": share_found(ou)},
                    "largest_drops": [item(n) for n in sorted(codes, key=lambda n: (ou[n][1] - cl[n][1], n))[:top]],
                    "largest_rises": [item(n) for n in sorted(codes, key=lambda n: (cl[n][1] - ou[n][1], n))[:top]]}
    return out


def why(top=4):
    """Report level, for the codes the classifier places falsely most often in each test year: on every report where its
    first answer is the code, the codes the reports were filed under and whether either LLM coder names the code.
    Reads runs/<corpus>_encyear and the LLM answers as scripts/reliability_year.py does (CPU, about a minute).
    The stream is set by MAINT_CORPUS as in the other scripts. Writes fig_rare_codes_why_<corpus>.json."""
    import collections

    import numpy as np

    import maude_llm as ml
    import rare_year as ry
    from common import CORPUS
    from maude_data import load_split

    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {code: p for p, code in p2c.items()}
    rep = {"corpus": CORPUS, "encoder": ry.TAG, "years": {}}
    for year in ("2026", "2025"):
        z = np.load(ry.ENC / f"{year}.npz")
        cls = [str(c) for c in z["classes"]]
        names = [path_of.get(c, c).split(" > ")[-1] for c in cls]
        c2i = {c: i for i, c in enumerate(cls)}
        keys, gold, topk = z["keys"], z["gold"].astype(int), z["top_idx"].astype(int)
        enc, best, rank = topk[:, 0], z["rare_best"].astype(int), z["rare_rank"].astype(int)
        is_rare = np.zeros(len(cls), bool)
        is_rare[z["rare"].astype(int)] = True
        pos = {int(k): i for i, k in enumerate(keys)}
        false_enc = is_rare[enc] & (enc != gold)
        answers = {}
        for mk, _, _ in v5.LLMS:
            for kind_name, kind in ry.KINDS.items():
                a = np.full(len(keys), -2)
                for k, v in ry.llm_answers(year, mk, kind, c2i, legal, p2c).items():
                    if k in pos:
                        a[pos[k]] = v
                answers[mk, kind_name] = a
        rows = []
        for code, n_false in collections.Counter(enc[false_enc].tolist()).most_common(top):
            m = enc == code  # the classifier's first answer is this rare code: it is the top rare code, at rank 1
            assert (best[m] == code).all() and (rank[m] == 0).all()
            row = {"code": names[code], "classifier_answers": int(m.sum()), "correct": int((m & (gold == code)).sum()),
                   "false": n_false,
                   "false_filed_under": collections.Counter(names[g] for g in gold[m & (gold != code)]).most_common(3)}
            for mk, _, _ in v5.LLMS:
                zs, rag = answers[mk, "zs"], answers[mk, "rag"]
                named = m & ((zs == code) | (rag == code))
                row[mk] = {"named_by_an_llm_coder": int(named.sum()), "of_them_correct": int((named & (gold == code)).sum()),
                           "without_answer": int((m & ((zs == -2) | (rag == -2))).sum()),
                           "zero_shot_answers": collections.Counter(names[a] if a >= 0 else "(invalid)"
                                                                    for a in zs[m]).most_common(3),
                           "retrieval_answers": collections.Counter(names[a] if a >= 0 else "(invalid)"
                                                                    for a in rag[m]).most_common(3)}
            rows.append(row)
            print(f"[{CORPUS} {year}] {row['code']}: classifier answers it on {row['classifier_answers']} reports "
                  f"({row['correct']} correct), false ones filed under {row['false_filed_under']}; named by an LLM coder "
                  + ", ".join(f"{mk} {row[mk]['named_by_an_llm_coder']}" for mk, _, _ in v5.LLMS))
        rep["years"][year] = {"classifier_false": int(false_enc.sum()), "codes": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fig_rare_codes_why_{CORPUS}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", OUT / f"fig_rare_codes_why_{CORPUS}.json")


if __name__ == "__main__":
    if "--why" in sys.argv:
        why()
        sys.exit()
    check()
    drawn = {}
    for year in ("2026", "2025"):
        for ours, tag in (([QWEN], "qwen"), ([GEMMA], "gemma"), ([QWEN, GEMMA], "both")):
            name = f"fig_rare_codes_{year}_{tag}"
            drawn[name] = figure(year, ours, name)
    changes = code_changes()
    (OUT / "fig_rare_codes_rows.json").write_text(json.dumps({"figures": drawn, "changes_between_codes": changes},
                                                             indent=1), encoding="utf-8")
    print("wrote", OUT / "fig_rare_codes_rows.json")
    for k, v in changes.items():
        s = v["share_of_false_under_codes_also_placed_correctly"]
        drops = ", ".join(f"{d['code']} {d['classifier'][1]}->{d['ours'][1]}" for d in v["largest_drops"][:3])
        rises = ", ".join(f"{d['code']} {d['classifier'][1]}->{d['ours'][1]}" for d in v["largest_rises"][:3])
        print(f"{k}: codes with more / fewer correct {v['codes_more_correct']} / {v['codes_fewer_correct']}; false removed "
              f"{v['false_removed']}, added {v['false_added']}; false under codes also placed correctly: classifier "
              f"{s['classifier']:.0%}, Ours {s['ours']:.0%}; drops {drops}; rises {rises}")
