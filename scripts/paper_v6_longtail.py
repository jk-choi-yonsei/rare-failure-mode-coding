"""Supplementary Fig. S1 of the RESS manuscript: the long tail of the codes.

For every known code of each report stream, its training reports (x, log scale) against its reports filed in the 2026
test year (y, log scale). A code with no reports sits at 0, set off from 1 by a break in the axis. The band of one to
five training reports, the rare codes of Eq. (1), is shaded and labelled "rare". Many aircraft codes share the same
small counts (30 rare codes have one training report and none filed in 2026), so one circle stands for all codes at a
point and its area grows with their number.

Data, as in the paper:
  training reports  maude_router.training_counts(split, codes)["full"] over maude_data.known_codes(split), the reports
                    the classifier is trained on (MAINT_CORPUS=sdr selects the aircraft stream)
  filed in a year   counts per filed code among the known-code reports of data/<corpus>/pool_<year>.jsonl, checked
                    code by code against the filed codes (gold) of the classifier's year file
                    runs/<corpus>_encyear/<tag>/<year>.npz, which holds every known-code report of the year
Both are checked against Table 1 of the manuscript (known and rare codes, training reports, reports of each test year,
rare reports and the rare codes they cover); the script stops if one differs.

Style: paper_v5_figures.py (fonts and axes of paper_v4_figures.py, ink and muted grey, direct labels).

Usage: python paper_v6_longtail.py            counts of both streams (one process each, since common.CORPUS is read at
                                              import; skipped when the count files exist), then the figure
       python paper_v6_longtail.py --recount  the same, counting again
       python paper_v6_longtail.py --counts   counts of the stream that MAINT_CORPUS names, only
Writes paper/RESS/figures/candidates/: longtail_counts_<corpus>.json, fig_longtail.pdf, fig_longtail.png (150 dpi) and
fig_longtail.tex (the figure environment with the proposed caption, for Section 4.1 after Table 1)
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "RESS" / "figures" / "candidates"
STREAMS = (("maude", "Device"), ("sdr", "Aircraft"))
TAG = {"maude": "cb-large-conv_full_seed42", "sdr": "cb-conv_full_seed42"}  # the classifiers of the paper
YEARS = ("2025", "2026")
YEAR = "2026"  # the year drawn
# Table 1 of the manuscript: known codes, rare codes, training reports; for each test year: reports, rare reports and
# the rare codes they cover
TABLE1 = {"maude": {"known": 424, "rare": 46, "training": 656_460, "2025": [762_360, 597, 46], "2026": [661_160, 552, 43]},
          "sdr": {"known": 261, "rare": 130, "training": 86_106, "2025": [59_416, 436, 99], "2026": [42_717, 292, 85]}}


def counts_path(corpus):
    return OUT / f"longtail_counts_{corpus}.json"


# ------------------------------------------------------------------ counts (one stream per process)
def counts():
    from common import CORPUS, RUNS
    from maude_data import POOL_2025, POOL_2026, class_info, known_codes, load_split
    from maude_router import training_counts
    split = load_split()
    info = class_info(split)
    allc = sorted(info)
    n = dict(zip(allc, training_counts(split, allc)["full"].tolist()))
    known = known_codes(split)
    c2i = {c: i for i, c in enumerate(known)}
    filed = {}
    for y, pool in zip(YEARS, (POOL_2025, POOL_2026)):
        cnt = np.zeros(len(known), dtype=np.int64)
        with pool.open(encoding="utf-8") as f:
            for line in f:
                i = c2i.get(json.loads(line)["code"])
                if i is not None:
                    cnt[i] += 1
        z = np.load(RUNS / f"{CORPUS}_encyear" / TAG[CORPUS] / f"{y}.npz")
        if [str(c) for c in z["classes"]] != known:
            raise SystemExit(f"{y}: the year file's codes are not the known codes")
        gold = np.bincount(z["gold"].astype(np.int64), minlength=len(known))
        if not (gold == cnt).all():
            raise SystemExit(f"{y}: filed codes of {pool.name} and of the year file differ")
        filed[y] = cnt
    train = np.array([n[c] for c in known])
    # every cleaned report of the code in the training years, with those held out in the balanced training sample and
    # the validation reports (for rare codes about twice the training reports, Section 4.1)
    pool = np.array([info[c]["n_pool"] for c in known])
    if (pool < train).any():
        raise SystemExit("a code has more training reports than reports in the training years")
    rare = (train >= 1) & (train <= 5)
    got = {"known": len(known), "rare": int(rare.sum()), "training": int(train.sum())}
    for y in YEARS:
        got[y] = [int(filed[y].sum()), int(filed[y][rare].sum()), int((filed[y][rare] > 0).sum())]
    bad = {k: (got[k], v) for k, v in TABLE1[CORPUS].items() if got[k] != v}
    if bad:
        raise SystemExit(f"[{CORPUS}] differs from Table 1 (got, table): {bad}")
    summary = {"known_codes": len(known), "rare_codes": int(rare.sum()), "training_reports": int(train.sum()),
               "known_codes_without_training_reports": int((train == 0).sum()),
               "most_training_reports": int(train.max()), "years": {}}
    for y in YEARS:
        f = filed[y]
        summary["years"][y] = {
            "reports": int(f.sum()), "rare_reports": int(f[rare].sum()), "rare_codes_filed": int((f[rare] > 0).sum()),
            "rare_codes_none_filed": int((f[rare] == 0).sum()),
            "rare_codes_more_filed_than_training": int((f[rare] > train[rare]).sum()),
            "rare_codes_at_least_as_many_filed_as_training": int((f[rare] >= train[rare]).sum()),
            "rare_codes_more_filed_than_in_the_training_years": int((f[rare] > pool[rare]).sum()),
            "other_codes_more_filed_than_training": int((f[~rare] > train[~rare]).sum()),
            "known_codes_none_filed": int((f == 0).sum()), "most_filed": int(f.max())}
    rows = [{"code": c, "term": info[c]["term"], "path": info[c]["path"], "train": int(train[i]),
             "training_years": int(pool[i]), "rare": bool(rare[i]), "filed": {y: int(filed[y][i]) for y in YEARS}}
            for i, c in enumerate(known)]
    rep = {"corpus": CORPUS, "classifier": TAG[CORPUS], "table1_check": "known codes, rare codes, training reports, and "
           "for each test year the reports, rare reports and rare codes with reports equal Table 1",
           "sources": {"training": "maude_router.training_counts(split, codes)['full'] over maude_data.known_codes",
                       "training_years": "maude_data.class_info(split)[code]['n_pool'], every cleaned report of the code "
                                         "in the training years, held-out reports included",
                       "filed": [f"data/{CORPUS}/pool_{y}.jsonl, known-code reports; equal code by code to the gold of "
                                 f"runs/{CORPUS}_encyear/{TAG[CORPUS]}/{y}.npz" for y in YEARS]},
           "summary": summary, "codes": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    counts_path(CORPUS).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[{CORPUS}] Table 1 check passed; wrote {counts_path(CORPUS)}", flush=True)
    print(json.dumps(summary, indent=1), flush=True)


# ------------------------------------------------------------------ figure
FLOOR = 0.4             # where a count of 0 is drawn on the log axes
BREAK = np.sqrt(FLOOR)  # the axis break between 0 and 1
# room below the floor row for the largest circle (30 aircraft codes at 1 and 0), and above the head codes for the
# labels and "rare"
XLIM, YLIM = (FLOOR / 1.45, 8e4), (FLOOR / 2.2, 2e5)
BAND = (BREAK, np.sqrt(30))  # one to five training reports: from the break to halfway (in log) between 5 and 6
S1 = 9.0                # marker area (pt^2) of a point with one code
OTHER, BAND_FILL, LEADER = "#9CA3AF", "#ECEEF1", "#6B7280"
# labelled codes, at most three per panel: code -> (label, label position in data units, horizontal alignment, point
# of the label where the leader starts). The labels sit in the empty upper left, clear of every point; Migration
# (Section 3.3) lies inside the dense cloud and Computer System Security Problem (4, 67) under Implant subsidence
# (4, 63), so neither is labelled.
LABELS = {"maude": {"3198": ("Application Program Problem:\nMedication Error", (0.33, 30000), "left", (0.2, 0)),
                    "4078": ("Implant\nsubsidence", (3.0, 7000), "left", (0.2, 0))},
          "sdr": {"1001": ("CORRODED", (60000, 45000), "right", (0.75, 0))}}


CAPTION = ("Training reports and reports filed in 2026 for each known code. Shaded band: rare codes. Codes without "
           "reports are drawn at 0. Circle area: number of codes at a point.")
SNIPPET = r"""%% Candidate figure (scripts/paper_v6_longtail.py), proposed for Section 4.1 after Table 1, cited in its sentence
%% "... a test year holds every cleaned report filed under a known code (Table~\ref{tab:data} and Fig.~\ref{fig:longtail})."
\begin{figure}[pos=!htb]
\centering
\includegraphics[width=0.92\textwidth]{candidates/fig_longtail.pdf}
\caption{CAPTION}
\label{fig:longtail}
\end{figure}
"""


def at_floor(v):
    v = np.asarray(v, dtype=float)
    return np.where(v == 0, FLOOR, v)


def log_axis(axis, hi):
    """Ticks 0 (the floor), 1, 10, ... up to hi, with plain labels; minor ticks from 2 upward, none below 1."""
    from matplotlib.ticker import FixedFormatter, FixedLocator
    major = [FLOOR] + [10.0 ** e for e in range(0, 6) if 10.0 ** e <= hi]
    axis.set_major_locator(FixedLocator(major))
    axis.set_major_formatter(FixedFormatter(["0"] + [f"{v:,.0f}" for v in major[1:]]))
    axis.set_minor_locator(FixedLocator([m * 10.0 ** e for e in range(0, 6) for m in range(2, 10) if m * 10.0 ** e <= hi]))


def axis_breaks(ax, edge):
    """Two short slanted strokes across the left and bottom spines, between 0 and 1."""
    fx, fy = ((np.log(BREAK) - np.log(lo)) / (np.log(hi) - np.log(lo)) for lo, hi in (XLIM, YLIM))
    kw = {"transform": ax.transAxes, "clip_on": False, "zorder": 10}
    gap, d = 0.011, 0.012
    ax.plot([0, 0], [fy - gap, fy + gap], color="white", lw=2.2, solid_capstyle="butt", **kw)
    ax.plot([fx - gap, fx + gap], [0, 0], color="white", lw=2.2, solid_capstyle="butt", **kw)
    for o in (-gap, gap):
        ax.plot([-d, d], [fy + o - d, fy + o + d], color=edge, lw=0.8, **kw)
        ax.plot([fx + o - d, fx + o + d], [-d, d], color=edge, lw=0.8, **kw)


def panel(ax, D, labels, ink, label_grey):
    rows = D["codes"]
    ax.axvspan(*BAND, color=BAND_FILL, lw=0, zorder=0)
    ax.text(np.sqrt(BAND[0] * BAND[1]), YLIM[1] / 1.3, "rare", ha="center", va="top", fontsize=8.5, color=ink, zorder=1)
    cells = Counter((r["train"], r["filed"][YEAR], r["rare"]) for r in rows)
    for is_rare in (False, True):
        pts = sorted(((x, y, k) for (x, y, rr), k in cells.items() if rr == is_rare), key=lambda p: -p[2])
        x, y, k = (np.array(v) for v in zip(*pts))
        ax.scatter(at_floor(x), at_floor(y), s=S1 * k, color=ink if is_rare else OTHER, edgecolors="white",
                   linewidths=0.45, zorder=4 if is_rare else 3)
    by_code = {r["code"]: r for r in rows}
    for code, (text, xy, ha, rel) in labels.items():
        r = by_code[code]
        if text.replace("\n", " ") != r["term"]:
            raise SystemExit(f"label {text!r} does not name code {code} ({r['term']})")
        ax.annotate(text, (at_floor(r["train"]), at_floor(r["filed"][YEAR])), xytext=xy, textcoords="data", ha=ha,
                    va="center", fontsize=7.5, color=ink if r["rare"] else label_grey, zorder=6,
                    arrowprops={"arrowstyle": "-", "color": LEADER, "lw": 0.6, "shrinkA": 1, "shrinkB": 2.5,
                                "relpos": rel})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    log_axis(ax.xaxis, XLIM[1])
    log_axis(ax.yaxis, YLIM[1])
    ax.grid(axis="x", visible=False)
    axis_breaks(ax, ax.spines["left"].get_edgecolor())


def figure():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import paper_v5_figures as v5  # noqa: F401  (sets the fonts and axes of the paper's data figures)
    D = {c: json.loads(counts_path(c).read_text(encoding="utf-8")) for c, _ in STREAMS}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), gridspec_kw={"wspace": 0.2})
    for ax, (c, title), letter in zip(axes, STREAMS, "ab"):
        panel(ax, D[c], LABELS[c], v5.INK, v5.LABEL_GREY)
        ax.set_title(f"({letter}) {title} reports", fontsize=10)
        ax.set_xlabel("Training reports (log scale)")
    axes[0].set_ylabel(f"Reports filed in {YEAR} (log scale)")
    sizes = (1, 10, 30)
    handles = [Line2D([], [], ls="none", marker="o", ms=np.sqrt(S1 * k), color=v5.MUTED, mec="white", mew=0.45,
                      label=f"{k} code" + ("s" if k > 1 else "")) for k in sizes]
    fig.legend(handles=handles, loc="upper center", ncol=len(sizes), bbox_to_anchor=(0.5, 0.0), fontsize=8,
               handletextpad=0.3, columnspacing=1.6, title=None)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_longtail.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig_longtail.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    words = len(CAPTION.split())
    if words > 40:
        raise SystemExit(f"caption has {words} words, the ledger allows 40")
    (OUT / "fig_longtail.tex").write_text(SNIPPET.replace("CAPTION", CAPTION), encoding="utf-8")
    print(f"wrote {OUT / 'fig_longtail.pdf'}, .png (150 dpi) and .tex (caption of {words} words)")
    for c, title in STREAMS:
        s = D[c]["summary"]
        Y = s["years"][YEAR]
        print(f"{title}: {s['known_codes']} known codes, {s['rare_codes']} rare; in {YEAR} {Y['rare_codes_none_filed']} rare "
              f"codes with no filed report, {Y['rare_codes_more_filed_than_training']} with more filed reports than "
              f"training reports; points drawn: {len({(r['train'], r['filed'][YEAR]) for r in D[c]['codes']})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true", help="count the stream that MAINT_CORPUS names, then stop")
    ap.add_argument("--recount", action="store_true", help="count both streams again before drawing")
    a = ap.parse_args()
    if a.counts:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        counts()
        return
    for c, _ in STREAMS:
        if a.recount or not counts_path(c).exists():
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--counts"], check=True,
                           env={**os.environ, "MAINT_CORPUS": c})
    figure()


if __name__ == "__main__":
    main()
