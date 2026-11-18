"""Data figures of the v4 paper (notes/paper-plan-v4-2026-11-16.md), drawn from the frozen numbers only:
results/paper_v4/<corpus>_numbers.json, results/rare_year/<corpus>metrics.json, results/rare_year/tail_split_<corpus>.json.

  fig_rare_tradeoff   correct against false rare-code placements over a year: encoder, the zero-shot LLM, the
                      retrieval LLM, and ours with the encoder's top 1, 3, 5, 10 as candidates (MAUDE, SDR; Qwen,
                      2026); fig_rare_tradeoff_appendix has Qwen 2025 and Gemma 2026 and 2025
  fig_rare_split      the year's rare-code reports in five shares by where the encoder ranks the filed code and
                      whether an LLM names it
  fig_rare_modes      rare failure modes in a year's counts: modes found (at least one report coded right) and false
                      alarms (modes placed only wrongly), encoder against ours, both corpora and years; and the rare
                      modes seen by month in 2026 against those filed (results/reliability, scripts/reliability_year.py)

Style of the author's SARC (TSM) and SMART (C&IE) plots, as in paper_figures.py. Coder colours: encoder blue, LLM
red, retrieval LLM orange, ours green. Usage: python paper_v4_figures.py [qwen38_27b|gemma4_31b]  (default: both;
the Gemma files get the suffix _gemma)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import ROOT  # noqa: E402

OUT = ROOT / "paper" / "RESS" / "figures"  # 2026-11-18: the paper folder holds only RESS/
ENC, LLM, RAG, OURS = "#4C72B0", "#C44E52", "#DD8452", "#27AE60"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.9, "axes.edgecolor": "black",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#BDBDBD", "legend.fancybox": True,
    "lines.linewidth": 1.6, "lines.markersize": 5.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})
CORPORA = (("maude", "Device"), ("sdr", "Aircraft"))  # the text calls them device and aircraft reports


def jl(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


J = {c: jl(f"results/paper_v4/{c}_numbers.json") for c, _ in CORPORA}
RY = {"maude": jl("results/rare_year/metrics.json"), "sdr": jl("results/rare_year/sdr_metrics.json")}
TS = {c: jl(f"results/rare_year/tail_split_{c}.json") for c, _ in CORPORA}
RL = {c: jl(f"results/reliability/{c}.json") for c, _ in CORPORA}
RC = {c: jl(f"results/why_it_works/reviewer_checks_{c}.json") for c, _ in CORPORA}  # encoder-only threshold curves


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def draw_tradeoff(ax, c, llm, y):
    """One corpus, one LLM, one year: encoder, the two LLMs alone, and ours over k = 1, 3, 5, 10."""
    M, R = J[c]["llms"][llm][y]["methods"], RY[c]["years"][y]
    e = R["encoder_alone"]
    ax.plot(e["false"], e["right"], ls="none", marker="o", ms=6.5, color=ENC, zorder=4)
    for k, col, mk in (("zs", LLM, "^"), ("rag", RAG, "s")):
        m = M[k]
        lo, hi = m["rare_false_ci95"]
        ax.errorbar(m["rare_false"], m["rare_right"], xerr=[[m["rare_false"] - lo], [hi - m["rare_false"]]], fmt=mk, ms=6.5,
                    color=col, elinewidth=0.8, capsize=2, zorder=4)
    eo = [p for p in RC[c]["years"][y]["encoder_only_rule"]["prob_top5"] if p["false"] >= 20]
    ax.plot([p["false"] for p in eo], [p["correct"] for p in eo], color=ENC, lw=1.1, ls="--", marker="o", ms=2.6,
            mfc="white", zorder=2)
    pts = []
    for kk in (1, 3, 5, 10):
        r = R["llms"][llm]["rules"][f"k{kk}/either"]
        if r["complete"]:
            pts.append((r["confirmed_only"]["false"], r["confirmed_only"]["right"], kk))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=OURS, lw=1.4, zorder=3)
    for x, yv, kk in pts:
        ax.plot(x, yv, ls="none", marker="D", ms=7 if kk == 5 else 4.5, color=OURS, zorder=5)
        if kk in (1, 10):  # the curve's two ends; the large diamond is k = 5
            ax.annotate(f"$k$ = {kk}", (x, yv), xytext=(0, 9) if kk == 1 else (5, 7), textcoords="offset points",
                        ha="center" if kk == 1 else "left", va="center", fontsize=7.5, color=OURS)
    ax.set_xscale("log")
    ax.set_xlim(14, 4e4)
    ax.set_ylim(0, None)


def tradeoff_legend(fig, y):
    handles = [Line2D([], [], ls="none", marker="o", color=ENC, ms=6, label="Classifier"),
               Line2D([], [], ls="none", marker="^", color=LLM, ms=6, label="Zero-shot LLM"),
               Line2D([], [], ls="none", marker="s", color=RAG, ms=6, label="Retrieval LLM"),
               Line2D([], [], ls="--", marker="o", ms=3, mfc="white", color=ENC, lw=1.1, label="Classifier, threshold"),
               Line2D([], [], marker="D", color=OURS, ms=5, lw=1.4, label="Ours")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, y), handletextpad=0.4, columnspacing=1.5)


def fig_rare_tradeoff(llm, year, name):
    """Main-text figure: one LLM, one year (the test year 2026)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"wspace": 0.2})
    for ax, (c, title), letter in zip(axes, CORPORA, "ab"):
        draw_tradeoff(ax, c, llm, year)
        ax.set_title(f"({letter}) {title} reports")
        ax.set_xlabel("False placements in the year")
    axes[0].set_ylabel("Correct placements in the year")
    tradeoff_legend(fig, -0.02)
    save(fig, name)


def fig_rare_tradeoff_appendix(name):
    """Appendix: the other LLM-year combinations, one row each."""
    rows = (("qwen38_27b", "2025", "Qwen, 2025"), ("gemma4_31b", "2026", "Gemma, 2026"), ("gemma4_31b", "2025", "Gemma, 2025"))
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.0, 2.35 * len(rows)), gridspec_kw={"wspace": 0.2, "hspace": 0.55})
    letters = iter("abcdefgh")
    for r, (llm, year, lab) in enumerate(rows):
        for col, (c, title) in enumerate(CORPORA):
            ax = axes[r, col]
            draw_tradeoff(ax, c, llm, year)
            ax.set_title(f"({next(letters)}) {title} reports, {lab}")
            if r == len(rows) - 1:
                ax.set_xlabel("False placements in the year")
            if col == 0:
                ax.set_ylabel("Correct placements in the year")
    tradeoff_legend(fig, 0.06)
    save(fig, name)


def fig_rare_split(llm, name):
    segs = (("kept_top1", "Top 1, placed", ENC, "white"),
            ("recovered_rank2_5", "Top 2–5, placed", OURS, "white"),
            ("missed_in_top5", "Top 5, not placed", "#BFE3CC", "black"),
            ("llm_only_outside_top5", "Outside top 5, LLM right", "#F0B9BA", "black"),
            ("neither", "Neither", "#E5E5E5", "black"))
    rows = [(c, y, f"{t} {y}") for c, t in CORPORA for y in ("2025", "2026")]
    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    for i, (c, y, _) in enumerate(rows):
        s, left = TS[c][f"{c} {llm} {y}"]["segments"], 0
        for key, _, col, tcol in segs:
            v = s[key]
            ax.barh(i, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.6)
            if v >= 12:
                ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=8, color=tcol)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Rare reports in the year")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 620)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for _, lab, col, _ in segs]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.1), handletextpad=0.4, columnspacing=1.0,
               fontsize=8.5)
    save(fig, name)


def fig_rare_modes(llm, name):
    """Rare failure modes in a year's counts, encoder against ours: (a) modes found and false alarms for both corpora
    and years; (b, c) rare modes seen so far by month in 2026, with the modes filed so far."""
    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.34)
    ax = fig.add_subplot(gs[0])
    groups = [(c, y, y) for c, t in CORPORA for y in ("2025", "2026")]
    w = 0.36
    for i, (c, y, _) in enumerate(groups):
        Y = RL[c]["years"][y]
        for j, (m, col) in enumerate(((Y["encoder"], ENC), (Y["gate"][llm], OURS))):
            x = i + (j - 0.5) * w
            ax.bar(x, m["visible_modes"], w * 0.9, color=col)
            ax.bar(x, -m["phantom_modes"], w * 0.9, color="white", edgecolor=col, hatch="////", linewidth=0.8)
            ax.text(x, m["visible_modes"] + 0.8, str(m["visible_modes"]), ha="center", va="bottom", fontsize=7.5)
            ax.text(x, -m["phantom_modes"] - 0.8, str(m["phantom_modes"]), ha="center", va="top", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[2] for g in groups])
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for k, (_, title) in enumerate(CORPORA):
        ax.text(2 * k + 0.5, -0.115, title, transform=tr, ha="center", va="top", fontsize=9.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(-40, 40)
    ticks = [-30, -20, -10, 0, 10, 20, 30]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(abs(v)) for v in ticks])
    ax.set_ylabel("Rare failure modes")
    ax.text(0.03, 0.97, "found", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#555555")
    ax.text(0.03, 0.03, "false alarms", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color="#555555")
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Rare modes in a year's counts")
    for k, (c, t) in enumerate(CORPORA):
        axc = fig.add_subplot(gs[k + 1])
        Y = RL[c]["years"]["2026"]
        months = 10  # the 2026 files of both streams end in October (MAUDE 30 Oct, SDR 22 Oct)
        xs = list(range(1, months + 1))
        d_enc, d_ours = Y["encoder"]["detection"], Y["gate"][llm]["detection"]
        axc.plot(xs, d_ours["cumulative_filed_by_month"][:months], color="#9E9E9E", ls="--", lw=1.3)
        axc.plot(xs, d_enc["cumulative_seen_by_month"][:months], color=ENC, marker="o", ms=3.5)
        axc.plot(xs, d_ours["cumulative_seen_by_month"][:months], color=OURS, marker="D", ms=3.5)
        axc.set_xticks([1, 4, 7, 10])
        axc.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        axc.set_xlim(0.5, 10.5)
        axc.set_ylim(0, None)
        axc.set_title(f"({'bc'[k]}) {t}, 2026")
        if k == 0:
            axc.set_ylabel("Rare modes found so far")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENC, label="Classifier"), plt.Rectangle((0, 0), 1, 1, color=OURS, label="Ours"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555", hatch="////", label="False alarms"),
               Line2D([], [], color="#9E9E9E", ls="--", lw=1.3, label="Filed")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.05), handletextpad=0.4, columnspacing=1.5)
    save(fig, name)

LLMS = (("qwen38_27b", "Qwen3.8-27B"), ("gemma4_31b", "Gemma 4 31B"))
OURS2 = "#117A65"  # Ours with the second LLM


def fig_rare_tradeoff_both(year, name):
    """Both LLMs (rows) and both streams (columns) for one test year."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw={"wspace": 0.2, "hspace": 0.45})
    letters = iter("abcd")
    for r, (llm, lab) in enumerate(LLMS):
        for col, (c, title) in enumerate(CORPORA):
            ax = axes[r, col]
            draw_tradeoff(ax, c, llm, year)
            ax.set_title(f"({next(letters)}) {title} reports, {lab}")
            if r == 1:
                ax.set_xlabel("False placements in the year")
            if col == 0:
                ax.set_ylabel("Correct placements in the year")
    tradeoff_legend(fig, 0.005)
    save(fig, name)


def fig_rare_split_both(name):
    segs = (("kept_top1", "Top 1, placed", ENC, "white"),
            ("recovered_rank2_5", "Top 2\u20135, placed", OURS, "white"),
            ("missed_in_top5", "Top 5, not placed", "#BFE3CC", "black"),
            ("llm_only_outside_top5", "Outside top 5, LLM right", "#F0B9BA", "black"),
            ("neither", "Neither", "#E5E5E5", "black"))
    rows, pos, y0 = [], [], 0.0
    for c, t in CORPORA:
        for y in ("2025", "2026"):
            for j, (llm, lab) in enumerate(LLMS):
                rows.append((c, y, llm, f"{t} {y}, {lab}"))
                pos.append(y0 + j)
            y0 += 2.5
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for p_, (c, y, llm, _) in zip(pos, rows):
        s, left = TS[c][f"{c} {llm} {y}"]["segments"], 0
        for key, _, col, tcol in segs:
            v = s[key]
            ax.barh(p_, v, left=left, height=0.78, color=col, edgecolor="white", linewidth=0.6)
            if v >= 14:
                ax.text(left + v / 2, p_, str(v), ha="center", va="center", fontsize=7.5, color=tcol)
            left += v
    ax.set_yticks(pos)
    ax.set_yticklabels([r[3] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rare reports in the year")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 620)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for _, lab, col, _ in segs]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.0,
               fontsize=8.5)
    save(fig, name)


def fig_rare_modes_both(name):
    """Rare modes in the counts of a year, the classifier against Ours with each LLM: (a) modes found and false alarms;
    (b, c) rare modes found so far by month in 2026, with the modes filed so far."""
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.34)
    ax = fig.add_subplot(gs[0])
    groups = [(c, y) for c, _ in CORPORA for y in ("2025", "2026")]
    w = 0.27
    for i, (c, y) in enumerate(groups):
        Y = RL[c]["years"][y]
        bars = ((Y["encoder"], ENC), (Y["gate"]["qwen38_27b"], OURS), (Y["gate"]["gemma4_31b"], OURS2))
        for j, (m, col) in enumerate(bars):
            x = i + (j - 1) * w
            ax.bar(x, m["visible_modes"], w * 0.92, color=col)
            ax.bar(x, -m["phantom_modes"], w * 0.92, color="white", edgecolor=col, hatch="////", linewidth=0.8)
            ax.text(x, m["visible_modes"] + 0.8, str(m["visible_modes"]), ha="center", va="bottom", fontsize=6.5)
            ax.text(x, -m["phantom_modes"] - 0.8, str(m["phantom_modes"]), ha="center", va="top", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups])
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for k, (_, title) in enumerate(CORPORA):
        ax.text(2 * k + 0.5, -0.115, title, transform=tr, ha="center", va="top", fontsize=9.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(-40, 40)
    ticks = [-30, -20, -10, 0, 10, 20, 30]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(abs(v)) for v in ticks])
    ax.set_ylabel("Rare modes")
    ax.text(0.03, 0.97, "found", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#555555")
    ax.text(0.03, 0.03, "false alarms", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color="#555555")
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Rare modes in the counts of a year")
    for k, (c, t) in enumerate(CORPORA):
        axc = fig.add_subplot(gs[k + 1])
        Y = RL[c]["years"]["2026"]
        months = 10  # the 2026 files of both streams end in October
        xs = list(range(1, months + 1))
        axc.plot(xs, Y["gate"]["qwen38_27b"]["detection"]["cumulative_filed_by_month"][:months], color="#9E9E9E", ls="--", lw=1.3)
        axc.plot(xs, Y["encoder"]["detection"]["cumulative_seen_by_month"][:months], color=ENC, marker="o", ms=3.2)
        axc.plot(xs, Y["gate"]["qwen38_27b"]["detection"]["cumulative_seen_by_month"][:months], color=OURS, marker="D", ms=3.2)
        axc.plot(xs, Y["gate"]["gemma4_31b"]["detection"]["cumulative_seen_by_month"][:months], color=OURS2, marker="s", ms=3.0)
        axc.set_xticks([1, 4, 7, 10])
        axc.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        axc.set_xlim(0.5, 10.5)
        axc.set_ylim(0, None)
        axc.set_title(f"({'bc'[k]}) {t}, 2026")
        if k == 0:
            axc.set_ylabel("Rare modes found so far")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENC, label="Classifier"),
               plt.Rectangle((0, 0), 1, 1, color=OURS, label="Ours, Qwen3.8-27B"),
               plt.Rectangle((0, 0), 1, 1, color=OURS2, label="Ours, Gemma 4 31B"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555", hatch="////", label="False alarms"),
               Line2D([], [], color="#9E9E9E", ls="--", lw=1.3, label="Filed")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.05), handletextpad=0.4, columnspacing=1.1,
               fontsize=8.5)
    save(fig, name)


if __name__ == "__main__":
    # 2026-11-18: both LLMs in the main text; the 2025 trade-off goes to the supplementary material
    fig_rare_tradeoff_both("2026", "fig_rare_tradeoff")
    fig_rare_tradeoff_both("2025", "fig_rare_tradeoff_2025")
    fig_rare_split_both("fig_rare_split")
    fig_rare_modes_both("fig_rare_modes")
