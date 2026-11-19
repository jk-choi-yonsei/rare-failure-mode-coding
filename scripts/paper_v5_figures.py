"""Data figures of the RESS manuscript, second design (2026-11-18, after the user found the figures too alike).

Each figure takes the form of its message, the proposed method (Ours) is the one strong colour, the baselines are muted,
and the LLM is told apart by marker fill and line style (filled / solid: Qwen3.8-27B; hollow / dashed: Gemma 4 31B).
The categorical palette (classifier, zero-shot LLM, retrieval LLM, Ours) keeps the hues of the method figure and passes
the dataviz validator in light mode (scripts/validate_palette.js; CVD warning 7.9 for retrieval vs Ours, covered by
marker shape and direct labels).

  fig_rare_tradeoff       2026: one panel per stream with both LLMs; shaded area = more correct and fewer false than the
                          classifier
  fig_rare_tradeoff_2025  the same for 2025 (supplementary material)
  fig_rare_split          rare reports by the classifier's rank of the filed code and the LLM coders, stacked bars
  fig_rare_modes          (a, b) rare modes found and false alarms as dot plots, (c, d) modes found by month as steps

Reads the same result files as paper_v4_figures.py. Writes paper/RESS/figures/<name>.pdf/.png.
Usage: python paper_v5_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import paper_v4_figures as v4  # noqa: E402  (data and the save function)

CL, ZS, RG, OU = "#3A6EA5", "#B91C1C", "#D97706", "#059669"
INK, MUTED, GRID = "#1F2937", "#6B7280", "#EEEEEE"
LLMS = (("qwen38_27b", "Qwen3.8-27B", True), ("gemma4_31b", "Gemma 4 31B", False))
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#4B5563",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "axes.titleweight": "normal", "axes.titlelocation": "left"})


def mk(face, filled):
    return {"mfc": face if filled else "white", "mec": face, "mew": 1.2}


# ------------------------------------------------------------------ figure 2: trade-off
def tradeoff_panel(ax, c, y):
    R = v4.RY[c]["years"][y]
    e = R["encoder_alone"]
    eo = [p for p in v4.RC[c]["years"][y]["encoder_only_rule"]["prob_top5"] if p["false"] >= 20]
    xmin, xmax = 14, 4e4
    top = max(max(v4.J[c]["llms"][llm][y]["methods"]["zs"]["rare_right"] for llm, _, _ in LLMS),
              max(p["correct"] for p in eo)) * 1.12
    ax.fill_between([xmin, e["false"]], e["right"], top, color=OU, alpha=0.07, lw=0, zorder=0)
    ax.plot([p["false"] for p in eo], [p["correct"] for p in eo], color=CL, lw=1.0, ls="--", marker="o", ms=2.4, mfc="white",
            zorder=2)
    last = eo[-1]
    ax.annotate("Classifier,\nthreshold", (last["false"], last["correct"]), xytext=(4, -2), textcoords="offset points",
                fontsize=7.5, color=CL, va="top")
    ax.plot(e["false"], e["right"], ls="none", marker="o", ms=7.5, color=CL, zorder=5)
    ax.annotate("Classifier", (e["false"], e["right"]), xytext=(6, -9), textcoords="offset points", fontsize=7.5, color=CL)
    for llm, _, filled in LLMS:
        M = v4.J[c]["llms"][llm][y]["methods"]
        for k, col, marker in (("zs", ZS, "^"), ("rag", RG, "s")):
            m = M[k]
            lo, hi = m["rare_false_ci95"]
            ax.errorbar(m["rare_false"], m["rare_right"], xerr=[[m["rare_false"] - lo], [hi - m["rare_false"]]], fmt=marker,
                        ms=6, color=col, elinewidth=0.7, capsize=1.5, zorder=4, **mk(col, filled))
        pts = []
        for kk in (1, 3, 5, 10):
            r = R["llms"][llm]["rules"][f"k{kk}/either"]
            if r["complete"]:
                pts.append((r["confirmed_only"]["false"], r["confirmed_only"]["right"], kk))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=OU, lw=1.8, ls="-" if filled else (0, (4, 2)), zorder=6)
        for x, yv, kk in pts:
            ax.plot(x, yv, ls="none", marker="D", ms=7 if kk == 5 else 4.5, zorder=7, **mk(OU, filled))
        if filled and pts[-1][2] == 10:
            ax.annotate("$k$=10", pts[-1][:2], xytext=(5, 4), textcoords="offset points", fontsize=7, color=OU)
    z = min((v4.J[c]["llms"][llm][y]["methods"]["zs"] for llm, _, _ in LLMS), key=lambda m: m["rare_right"])
    r = min((v4.J[c]["llms"][llm][y]["methods"]["rag"] for llm, _, _ in LLMS), key=lambda m: m["rare_right"])
    ax.annotate("Zero-shot LLM", (z["rare_false"], z["rare_right"]), xytext=(0, -13), textcoords="offset points",
                fontsize=7.5, color=ZS, ha="center")
    ax.annotate("Retrieval LLM", (r["rare_false_ci95"][1], r["rare_right"]), xytext=(5, -1), textcoords="offset points",
                fontsize=7.5, color=RG, va="center")
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, top)
    ax.grid(axis="x", visible=False)


def fig_tradeoff(year, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.18})
    for ax, (c, title), letter in zip(axes, v4.CORPORA, "ab"):
        tradeoff_panel(ax, c, year)
        ax.set_title(f"({letter}) {title} reports", fontsize=10)
        ax.set_xlabel("False placements in the year")
    axes[0].set_ylabel("Correct placements in the year")
    handles = [Line2D([], [], color=OU, lw=1.8, marker="D", ms=5, label="Ours"),
               Line2D([], [], color=CL, ls="none", marker="o", ms=6, label="Classifier"),
               Line2D([], [], color=ZS, ls="none", marker="^", ms=6, label="Zero-shot LLM"),
               Line2D([], [], color=RG, ls="none", marker="s", ms=6, label="Retrieval LLM"),
               Line2D([], [], color=MUTED, lw=1.6, marker="o", ms=5, label="Qwen3.8-27B"),
               Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), marker="o", ms=5, mfc="white", label="Gemma 4 31B")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.0), fontsize=8, handletextpad=0.4,
               columnspacing=1.2)
    v4.save(fig, name)


# ------------------------------------------------------------------ figure 3: where the filed rare code ranks
SEGS = (("kept_top1", "Placed, classifier rank 1", "#065F46", "white"),
        ("recovered_rank2_5", "Placed, rank 2 to 5", OU, "white"),
        ("missed_in_top5", "In the top five, not placed", "#A7F3D0", INK),
        ("llm_only_outside_top5", "Below the top five, LLM right", "#FCD9B6", INK),
        ("neither", "Neither", "#E5E7EB", INK))


def fig_split(name):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    pos, labels, groups, y0 = [], [], [], 0.0
    for c, t in v4.CORPORA:
        for y in ("2025", "2026"):
            groups.append((y0 + 0.5, f"{t} {y}"))
            for j, (llm, lab, _) in enumerate(LLMS):
                s, left = v4.TS[c][f"{c} {llm} {y}"]["segments"], 0
                for key, _, col, tcol in SEGS:
                    v = s[key]
                    ax.barh(y0 + j, v, left=left, height=0.8, color=col, edgecolor="white", linewidth=1.2)
                    if key in ("kept_top1", "recovered_rank2_5", "llm_only_outside_top5") and v >= 16:
                        ax.text(left + v / 2, y0 + j, str(v), ha="center", va="center", fontsize=7.5, color=tcol)
                    left += v
                ax.text(left + 6, y0 + j, str(left), ha="left", va="center", fontsize=7.5, color=MUTED)
                pos.append(y0 + j)
                labels.append(lab)
            y0 += 2.6
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8, color=MUTED)
    for yy, g in groups:
        ax.text(-0.25, yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(0, 640)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rare reports in the year")
    fig.subplots_adjust(left=0.30, top=0.88)
    handles = [Patch(color=col, label=lab) for _, lab, col, _ in SEGS]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.6, 1.0), fontsize=7.8, handletextpad=0.4,
               columnspacing=0.9)
    v4.save(fig, name)


# ------------------------------------------------------------------ figure 4: rare modes
def dots(ax, key, title):
    rows = [(c, y, f"{t} {y}") for c, t in v4.CORPORA for y in ("2025", "2026")]
    for i, (c, y, _) in enumerate(rows):
        Y = v4.RL[c]["years"][y]
        vals = [Y["encoder"][key]] + [Y["gate"][llm][key] for llm, _, _ in LLMS]
        ax.plot([min(vals), max(vals)], [i, i], color="#D1D5DB", lw=2.2, zorder=1, solid_capstyle="round")
        ax.plot(vals[0], i, ls="none", marker="o", ms=7, color=CL, zorder=3)
        for (llm, _, filled), v in zip(LLMS, vals[1:]):
            ax.plot(v, i + (-0.12 if filled else 0.12), ls="none", marker="D", ms=6.5, zorder=4, **mk(OU, filled))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=10)


def steps(ax, c, t, letter):
    Y = v4.RL[c]["years"]["2026"]
    months = 10  # the 2026 files of both streams end in October
    xs = list(range(1, months + 1))
    filed = Y["gate"]["qwen38_27b"]["detection"]["cumulative_filed_by_month"][:months]
    ax.fill_between(xs, filed, step="post", color="#E5E7EB", lw=0, zorder=0)
    ax.step(xs, filed, where="post", color="#9CA3AF", lw=1.0, zorder=1)
    ax.step(xs, Y["encoder"]["detection"]["cumulative_seen_by_month"][:months], where="post", color=CL, lw=1.6, zorder=3)
    for llm, _, filled in LLMS:
        ax.step(xs, Y["gate"][llm]["detection"]["cumulative_seen_by_month"][:months], where="post", color=OU, lw=1.8,
                ls="-" if filled else (0, (4, 2)), zorder=4)
    ax.text(months + 0.05, filed[-1], "Filed", fontsize=7.5, color=MUTED, va="center")
    ax.text(months + 0.05, Y["gate"]["qwen38_27b"]["detection"]["cumulative_seen_by_month"][months - 1] + 1.5, "Ours",
            fontsize=7.5, color=OU, va="bottom")
    ax.text(months + 0.05, Y["encoder"]["detection"]["cumulative_seen_by_month"][months - 1] - 1.5, "Classifier",
            fontsize=7.5, color=CL, va="top")
    ax.set_xticks([1, 4, 7, 10])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
    ax.set_xlim(1, months + 1.9)
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"({letter}) {t} reports, 2026", fontsize=10)


def fig_modes(name):
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    dots(ax_a, "visible_modes", "(a) Rare modes found")
    dots(ax_b, "phantom_modes", "(b) False alarms")
    ax_b.set_yticklabels([])
    ax_a.set_xlabel("Rare modes")
    ax_b.set_xlabel("Rare codes placed only falsely")
    for k, (c, t) in enumerate(v4.CORPORA):
        ax = fig.add_subplot(gs[1, k])
        steps(ax, c, t, "cd"[k])
        if k == 0:
            ax.set_ylabel("Rare modes found so far")
    handles = [Line2D([], [], color=CL, ls="none", marker="o", ms=6, label="Classifier"),
               Line2D([], [], color=OU, lw=1.8, marker="D", ms=5, label="Ours, Qwen3.8-27B"),
               Line2D([], [], color=OU, lw=1.8, ls=(0, (4, 2)), marker="D", ms=5, mfc="white", label="Ours, Gemma 4 31B"),
               Patch(color="#E5E7EB", label="Rare modes filed")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=8, handletextpad=0.4,
               columnspacing=1.4)
    v4.save(fig, name)


# ================================================================== third design (2026-11-18 evening)
import json  # noqa: E402

RANK = {c: json.loads((v4.ROOT / "results" / "why_it_works" / f"rank_curve_{c}.json").read_text(encoding="utf-8"))
        for c, _ in v4.CORPORA}


def n_rare(c, y):
    return sum(v4.TS[c][f"{c} qwen38_27b {y}"]["segments"].values())


def pr(correct, false, n):
    return 100 * correct / n, 100 * correct / max(correct + false, 1)


def pr_panel(ax, c, y):
    R, n = v4.RY[c]["years"][y], n_rare(c, y)
    e = R["encoder_alone"]
    eo = [q for q in v4.RC[c]["years"][y]["encoder_only_rule"]["prob_top5"] if q["correct"] + q["false"] >= 10]
    xs, ys = zip(*[pr(q["correct"], q["false"], n) for q in eo])
    ax.plot(xs, ys, color=CL, lw=1.0, ls="--", marker="o", ms=2.4, mfc="white", zorder=2)
    ex, ey = pr(e["right"], e["false"], n)
    ax.plot(ex, ey, ls="none", marker="o", ms=7.5, color=CL, zorder=5)
    ax.annotate("Classifier", (ex, ey), xytext=(-8, 0), textcoords="offset points", fontsize=7.5, color=CL, ha="right",
                va="center")
    for llm, _, filled in LLMS:
        M = v4.J[c]["llms"][llm][y]["methods"]
        for k, col, marker in (("zs", ZS, "^"), ("rag", RG, "s")):
            m = M[k]
            lo, hi = m["rare_false_ci95"]
            x, yv = pr(m["rare_right"], m["rare_false"], n)
            y_lo, y_hi = pr(m["rare_right"], hi, n)[1], pr(m["rare_right"], lo, n)[1]
            ax.errorbar(x, yv, yerr=[[yv - y_lo], [y_hi - yv]], fmt=marker, ms=6, color=col, elinewidth=0.7, capsize=1.5,
                        zorder=4, **mk(col, filled))
        pts = []
        for kk in (1, 3, 5, 10):
            r = R["llms"][llm]["rules"][f"k{kk}/either"]
            if r["complete"]:
                pts.append((*pr(r["confirmed_only"]["right"], r["confirmed_only"]["false"], n), kk))
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=OU, lw=1.8, ls="-" if filled else (0, (4, 2)), zorder=6)
        for x, yv, kk in pts:
            ax.plot(x, yv, ls="none", marker="D", ms=7 if kk == 5 else 4.5, zorder=7, **mk(OU, filled))
        if filled:
            ax.annotate("Ours", (pts[2][0], pts[2][1]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=OU,
                        fontweight="bold")
    z = max((v4.J[c]["llms"][llm][y]["methods"]["zs"] for llm, _, _ in LLMS), key=lambda m: m["rare_right"])
    r = max((v4.J[c]["llms"][llm][y]["methods"]["rag"] for llm, _, _ in LLMS), key=lambda m: m["rare_right"])
    zx, zy = pr(z["rare_right"], z["rare_false"], n)
    rx, ry = pr(r["rare_right"], r["rare_false"], n)
    ax.annotate("better", xy=(0.97, 0.96), xytext=(0.80, 0.80), xycoords="axes fraction", textcoords="axes fraction",
                fontsize=8, color=MUTED, ha="center", va="center",
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.0})
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)


def fig_pr(year, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.2})
    for ax, (c, title), letter in zip(axes, v4.CORPORA, "ab"):
        pr_panel(ax, c, year)
        ax.set_title(f"({letter}) {title} reports", fontsize=10)
        ax.set_xlabel("Recall of rare reports (%)")
    axes[0].set_ylabel("Precision of rare placements (%)")
    handles = [Line2D([], [], color=OU, lw=1.8, marker="D", ms=5, label="Ours"),
               Line2D([], [], color=CL, ls="none", marker="o", ms=6, label="Classifier"),
               Line2D([], [], color=CL, lw=1.0, ls="--", marker="o", ms=3, mfc="white", label="Classifier, threshold"),
               Line2D([], [], color=ZS, ls="none", marker="^", ms=6, label="Zero-shot LLM"),
               Line2D([], [], color=RG, ls="none", marker="s", ms=6, label="Retrieval LLM"),
               Line2D([], [], color=MUTED, lw=1.6, marker="o", ms=5, label="Qwen3.8-27B"),
               Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), marker="o", ms=5, mfc="white", label="Gemma 4 31B")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=8, handletextpad=0.4,
               columnspacing=1.2)
    v4.save(fig, name)


def fig_rank(name):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"wspace": 0.18}, sharey=True)
    for ax, (c, title), letter in zip(axes, v4.CORPORA, "ab"):
        Rk = RANK[c]
        ks = Rk["k"]
        for y, ls in (("2025", "-"), ("2026", (0, (4, 2)))):
            Y = Rk["years"][y]
            ax.plot(ks, [100 * v for v in Y["other"]], color="#9CA3AF", lw=1.6, ls=ls)
            ax.plot(ks, [100 * v for v in Y["rare"]], color=CL, lw=1.9, ls=ls)
        ax.axvline(5, color=MUTED, lw=0.8, ls=":")
        ax.text(5.3, 3, "top five", fontsize=7.5, color=MUTED)
        hi_o = min(100 * Rk["years"][y]["other"][7] for y in ("2025", "2026")) - 10
        hi_r = max(100 * Rk["years"][y]["rare"][7] for y in ("2025", "2026"))
        ax.text(8, hi_o, "Ordinary reports", fontsize=7.5, color="#6B7280", ha="center", va="bottom")
        ax.text(8, hi_r + 3, "Rare reports", fontsize=7.5, color=CL, ha="center", va="bottom")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 100)
        ax.set_title(f"({letter}) {title} reports", fontsize=10)
        ax.set_xlabel("Rank $k$ in the classifier's ranking")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Filed code within the top $k$ (%)")
    handles = [Line2D([], [], color=MUTED, lw=1.6, label="2025"), Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), label="2026")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=8)
    v4.save(fig, name)


def fig_months(name):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"wspace": 0.22})
    months = 10  # the 2026 files of both streams end in October
    xs = list(range(1, months + 1))
    for ax, (c, title), letter in zip(axes, v4.CORPORA, "ab"):
        Y = v4.RL[c]["years"]["2026"]
        filed = Y["gate"]["qwen38_27b"]["detection"]["cumulative_filed_by_month"][:months]
        enc = Y["encoder"]["detection"]["cumulative_seen_by_month"][:months]
        ax.plot(xs, filed, color="#9CA3AF", lw=1.4, ls=(0, (4, 2)))
        ax.plot(xs, enc, color=CL, lw=1.6, marker="o", ms=3.5)
        for llm, _, filled in LLMS:
            ax.plot(xs, Y["gate"][llm]["detection"]["cumulative_seen_by_month"][:months], color=OU, lw=1.8,
                    ls="-" if filled else (0, (4, 2)), marker="D", ms=3.8, **mk(OU, filled))
        q = Y["gate"]["qwen38_27b"]["detection"]["cumulative_seen_by_month"][months - 1]
        ax.text(months + 0.2, filed[-1], "Filed", fontsize=7.5, color=MUTED, va="center")
        ax.text(months + 0.2, q + 1, "Ours", fontsize=7.5, color=OU, va="bottom", fontweight="bold")
        ax.text(months + 0.2, enc[-1] - 1, "Classifier", fontsize=7.5, color=CL, va="top")
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        ax.set_xlim(0.7, months + 2.2)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"({letter}) {title} reports, 2026", fontsize=10)
    axes[0].set_ylabel("Rare modes by month end")
    handles = [Line2D([], [], color=OU, lw=1.8, marker="D", ms=4, label="Ours, Qwen3.8-27B"),
               Line2D([], [], color=OU, lw=1.8, ls=(0, (4, 2)), marker="D", ms=4, mfc="white", label="Ours, Gemma 4 31B"),
               Line2D([], [], color=CL, lw=1.6, marker="o", ms=4, label="Classifier"),
               Line2D([], [], color="#9CA3AF", lw=1.4, ls=(0, (4, 2)), label="Rare modes filed")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=8, handletextpad=0.4,
               columnspacing=1.4)
    v4.save(fig, name)


# ================================================================== Fig. 2 as F1 by method (2026-11-18, candidate E)
SA = {c: json.loads((v4.ROOT / "results" / "why_it_works" / f"selection_ablation_{c}.json").read_text(encoding="utf-8"))
      for c, _ in v4.CORPORA}
BASE = (("Zero-shot LLM", "zs"), ("Retrieval LLM", "rag"), ("Confidence cascade", "cascade"), ("EnergyRoute", "energyroute"),
        ("DCS (balanced fit)", "selector_balanced"), ("DCS (prior-corrected)", "selector_prior"))


def f1(c_, f_, n):
    return 100 * 2 * c_ / (n + c_ + f_)


def f1_table():
    """F1 (%) of every method in every case: {method: {(corpus, year, llm): F1}}; the classifier rows do not depend on the LLM."""
    T = {}
    for c, _ in v4.CORPORA:
        for y in ("2025", "2026"):
            n, R = n_rare(c, y), v4.RY[c]["years"][y]
            thr = max(f1(q["correct"], q["false"], n) for q in v4.RC[c]["years"][y]["encoder_only_rule"]["prob_top5"])
            la = SA[c]["llms"]["qwen38_27b"][y]["methods"]["logit_adjusted_tau0.5"]
            for llm, _, _ in LLMS:
                key = (c, y, llm)
                r = R["llms"][llm]["rules"]["k5/either"]["confirmed_only"]
                T.setdefault("Ours", {})[key] = f1(r["right"], r["false"], n)
                T.setdefault("Classifier", {})[key] = f1(R["encoder_alone"]["right"], R["encoder_alone"]["false"], n)
                T.setdefault("Classifier, best threshold", {})[key] = thr
                T.setdefault("Classifier, logit-adjusted", {})[key] = f1(la["rare_correct"], la["rare_false_estimated"], n)
                M = v4.J[c]["llms"][llm][y]["methods"]
                for name, k in BASE:
                    T.setdefault(name, {})[key] = f1(M[k]["rare_right"], M[k]["rare_false"], n)
    return T


def fig_f1(name):
    T = f1_table()
    order = sorted(T, key=lambda m: sum(T[m].values()) / len(T[m]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (c, title), letter in zip(axes, v4.CORPORA, "ab"):
        for i, m in enumerate(order):
            col = OU if m == "Ours" else (CL if m.startswith("Classifier") else "#9CA3AF")
            per_llm = not m.startswith("Classifier")
            for y, marker in (("2025", "o"), ("2026", "s")):
                for llm, _, filled in (LLMS if per_llm else LLMS[:1]):
                    v = T[m][(c, y, llm)]
                    dy = (-0.15 if y == "2025" else 0.15) + ((-0.07 if filled else 0.07) if per_llm else 0)
                    ax.plot(v, i + dy, marker, ms=6.5 if m == "Ours" else 5.2, zorder=3, **mk(col, filled))
        ax.axhspan(len(order) - 1.5, len(order) - 0.5, color=OU, alpha=0.06, lw=0, zorder=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlim(0, None)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"({letter}) {title} reports", fontsize=10)
        ax.set_xlabel("F1 of rare-code placements (%)")
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == "Ours":
            lab.set_color(OU)
            lab.set_fontweight("bold")
    def h(marker, filled):
        return Line2D([], [], ls="none", marker=marker, ms=5.5, **mk(MUTED, filled))
    pairs = [((h("o", True), h("o", False)), "2025"), ((h("s", True), h("s", False)), "2026"),
             ((h("o", True), h("s", True)), "Qwen3.8-27B"), ((h("o", False), h("s", False)), "Gemma 4 31B")]
    fig.legend([a for a, _ in pairs], [b for _, b in pairs], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
               loc="upper center", ncol=4, bbox_to_anchor=(0.55, 0.02), fontsize=8, handlelength=2.4, handletextpad=0.4,
               columnspacing=1.6)
    v4.save(fig, name)
    best_other = {k: max(T[m][k] for m in T if m != "Ours") for k in T["Ours"]}
    gaps = {k: T["Ours"][k] - best_other[k] for k in T["Ours"]}
    print("F1 Ours:", {k: round(v, 1) for k, v in T["Ours"].items()})
    print("Ours minus best other:", {k: round(v, 1) for k, v in gaps.items()})
    print("best other method per case:", {k: max((m for m in T if m != "Ours"), key=lambda m: T[m][k]) for k in T["Ours"]})
    return T


# ================================================================== Fig. 2 as correct vs false placements (2026-11-18, v7)
# Chosen after seven variants were judged by three independent readers: LLM coders drawn in grey with direct labels,
# faint guide curves where 1 in N placements is correct, and one arrow from Ours to the classifier-threshold point
# that first reaches the same correct count (the numbers of the text of Section 5.1).
LLM_GREY, LABEL_GREY, GUIDE = "#6B7280", "#4B5563", "#D1D5DB"
GUIDES = (3, 10, 30, 100)


def _plain_log_ticks(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    lo, hi = ax.get_xlim()
    ticks = [10 ** e for e in range(0, 6) if lo <= 10 ** e <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def counts_panel(ax, c, y, first, arrow=True):
    R = v4.RY[c]["years"][y]
    e = R["encoder_alone"]
    thr = [p for p in v4.RC[c]["years"][y]["encoder_only_rule"]["prob_top5"] if p["false"] >= 5]
    llm_pts = {(k, llm): v4.J[c]["llms"][llm][y]["methods"][k] for llm, _, _ in LLMS for k in ("zs", "rag")}
    xmax = 2.0 * max(m["rare_false_ci95"][1] for m in llm_pts.values())
    xmin = 0.6 * min([p["false"] for p in thr] + [R["llms"][llm]["rules"]["k1/either"]["confirmed_only"]["false"]
                                                  for llm, _, _ in LLMS])
    top = 1.12 * max(m["rare_right"] for m in llm_pts.values())
    box = dict(fc="white", ec="none", pad=0.4)
    # guide curves: 1 in N placements correct, i.e. correct = false / (N - 1); drawn when they leave through the top
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 300)
    for n in GUIDES:
        xe = top * (n - 1)
        if xe > xmax:
            continue
        ax.plot(xs, xs / (n - 1), color=GUIDE, lw=0.8, zorder=1)
        ax.text(xe * 1.07, top * 0.99, f"1 in {n}", fontsize=7, color=MUTED, ha="left", va="top", zorder=1)
    # classifier: threshold sweep (thin solid, filled dots) and its default answer
    ax.plot([p["false"] for p in thr], [p["correct"] for p in thr], color=CL, lw=0.9, marker="o", ms=2.4, zorder=2)
    end = thr[-1]
    ax.annotate("Classifier,\nthreshold", (end["false"], end["correct"]), xytext=(5, 0), textcoords="offset points",
                fontsize=7.5, color=CL, ha="left", va="center", bbox=box, zorder=3)
    ax.plot(e["false"], e["right"], "o", ms=7.5, color=CL, zorder=5)
    ax.annotate("Classifier", (e["false"], e["right"]), xytext=(7, -6), textcoords="offset points", fontsize=7.5,
                color=INK, ha="left", va="top", bbox=box, zorder=3)
    # LLM coders alone, grey, with 95% intervals on their estimated false placements
    for llm, _, filled in LLMS:
        for k, marker in (("zs", "^"), ("rag", "s")):
            m = llm_pts[k, llm]
            lo, hi = m["rare_false_ci95"]
            ax.errorbar(m["rare_false"], m["rare_right"], xerr=[[m["rare_false"] - lo], [hi - m["rare_false"]]],
                        fmt=marker, ms=6.5, color=LLM_GREY, elinewidth=0.7, capsize=1.5, zorder=4, **mk(LLM_GREY, filled))
    for k, name in (("zs", "Zero-shot LLM"), ("rag", "Retrieval LLM")):
        pts = [llm_pts[k, llm] for llm, _, _ in LLMS]
        ym = np.mean([p["rare_right"] for p in pts])
        if k == "rag" and c == "maude":  # left of these squares lies Ours, so the label goes to their right
            x0 = max(p["rare_false_ci95"][1] for p in pts)
            ax.annotate(name, (x0, ym), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=LABEL_GREY,
                        ha="left", va="center", bbox=box, zorder=3)
            continue
        x0 = min(p["rare_false_ci95"][0] for p in pts)
        ax.annotate(name, (x0, ym), xytext=(-4, 0), textcoords="offset points", fontsize=7.5, color=LABEL_GREY,
                    ha="right", va="center", bbox=box, zorder=3)
    # Ours: k = 1, 3, 5, 10, large diamond at k = 5
    for llm, _, filled in LLMS:
        pts = [(kk, R["llms"][llm]["rules"][f"k{kk}/either"]["confirmed_only"]) for kk in (1, 3, 5, 10)
               if R["llms"][llm]["rules"][f"k{kk}/either"]["complete"]]
        ax.plot([p["false"] for _, p in pts], [p["right"] for _, p in pts], color=OU, lw=2 if filled else 1.6,
                ls="-" if filled else (0, (4, 2)), zorder=6)
        for kk, p in pts:
            ax.plot(p["false"], p["right"], "D", ms=7.5 if kk == 5 else 4.5, zorder=7, **mk(OU, filled))
        if filled and first:
            k1, k10 = pts[0][1], pts[-1][1]
            ax.annotate("$k$=1", (k1["false"], k1["right"]), xytext=(-7, 5), textcoords="offset points", fontsize=7,
                        color=MUTED, ha="right", va="bottom")
            ax.annotate(f"$k$={pts[-1][0]}", (k10["false"], k10["right"]), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=MUTED, ha="left", va="bottom")
    q5 = R["llms"]["qwen38_27b"]["rules"]["k5/either"]["confirmed_only"]
    ax.annotate("Ours", (q5["false"], q5["right"]), xytext=(-9, 9), textcoords="offset points", fontsize=9.5,
                color=OU, fontweight="bold", ha="right", va="bottom")
    if arrow:  # first threshold point with at least as many correct placements as Ours (Qwen3.8-27B, k = 5)
        t = next(p for p in thr if p["correct"] >= q5["right"])
        ax.annotate("", xy=(t["false"], t["correct"]), xytext=(q5["false"], q5["right"]), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=6, shrinkB=3, mutation_scale=9))
        below = c == "maude"
        ax.annotate(f"×{t['false'] / q5['false']:.1f} false", (np.sqrt(q5["false"] * t["false"]),
                    (q5["right"] + t["correct"]) / 2), xytext=(0, -6 if below else 7), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="center", va="top" if below else "bottom", zorder=8, bbox=box)
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, top)
    _plain_log_ticks(ax)
    ax.grid(axis="x", visible=False)


def fig_counts(year, name, arrow=True):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.2})
    for i, (ax, (c, title), letter) in enumerate(zip(axes, v4.CORPORA, "ab")):
        counts_panel(ax, c, year, first=(i == 0), arrow=arrow)
        ax.set_title(f"({letter}) {title} reports", fontsize=10)
        ax.set_xlabel("False placements (log scale)")
    axes[0].set_ylabel("Correct placements")
    handles = [Line2D([], [], color=MUTED, lw=1.6, marker="D", ms=5, label="Qwen3.8-27B (filled, solid)"),
               Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), marker="D", ms=5, mfc="white",
                      label="Gemma 4 31B (hollow, dashed)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=8, handletextpad=0.4,
               columnspacing=1.6)
    v4.save(fig, name)


if __name__ == "__main__":
    fig_counts("2026", "fig_tradeoff")
    fig_counts("2025", "fig_tradeoff_2025")
    fig_rank("fig_rank")
    fig_months("fig_rare_modes")
