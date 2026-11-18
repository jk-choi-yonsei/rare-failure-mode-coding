"""Data figures for the paper (plan: notes/paper-figures-plan-2026-11-05.md), built from
the saved predictions and metrics only. MAUDE bin scores are recomputed from the
prediction files through the project's loaders; SDR, selector and natural-sample numbers
come from the committed metrics JSON files. Conceptual figures are in paper_diagrams.py.

  fig_tail       per-code F1 of the selector's encoder (class-balanced RoBERTa-large, full
                 N) on the 2025 MAUDE test against the code's training examples
  fig_crossover  macro-F1 by training-count bin at full N: encoder against two prompted
                 LLMs, MAUDE and SDR
  fig_selector   the selector against its three coders per bin, MAUDE and SDR (Opus)
  fig_natural    natural-proportion 2025 sample: accuracy against rare-mode reports placed

Style follows the author's SARC (TSM) and SMART (C&IE) plots: Times New Roman, the axes
boxed on all four sides with a light grid, short panel titles above ("(a) MAUDE"), a
framed legend inside the axes, markers on lines, ours in green as in SMART; one
comparison per panel, nothing the caption can carry.

Usage: python paper_figures.py [--year 2025|2026] [fig_tail fig_crossover ...]   (default: all, 2026)
--year 2026 (notes/prereg-redesign-2026-11-06.md): the test year with the redesign's encoders and
frozen selectors (results/redesign/*.json, results/maude2026nat/); 2025 rebuilds the draft-v2 figures.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import ROOT, rdir  # noqa: E402
from maude_claude_score import per_code_f1  # noqa: E402
from maude_extra_score import read_enc_csv  # noqa: E402
from maude_router import load_eval  # noqa: E402

OUT = ROOT / "archive" / "paper_v3" / "figures"  # the v3 figures; paper/ holds only RESS/ (2026-11-18)
RES = ROOT / "results"
BINS = {"1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
BIN_NAMES = list(BINS)
BL = [b.replace("-", "–") for b in BIN_NAMES]

# SARC's blue and red, SMART's green for ours
ENC, LLM, RAG, SEL, RULE = "#4C72B0", "#C44E52", "#DD8452", "#27AE60", "#7F7F7F"
RARE_BG = "#FBE5D6"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.9, "axes.edgecolor": "black",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#BDBDBD", "legend.fancybox": True,
    "lines.linewidth": 1.6, "lines.markersize": 5.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def bin_scores(ev, preds_list):
    counts = ev["counts"]["full"]
    out = []
    for lo, hi in BINS.values():
        codes = np.array([c for c in ev["labels"] if lo <= counts[c] <= hi])
        out.append(float(per_code_f1(ev["gold"], preds_list, codes).mean()))
    return out


YEAR = "2026"


def encoder_large_full(ev):
    if YEAR == "2025":
        return read_enc_csv(ev, rdir("maude2025") / "encoder_cb-large_full_seed42.csv")[0]
    return read_enc_csv(ev, rdir("maude2026") / "encoder_cb-large-conv_full_seed42.csv")[0]


def redesign(corpus):
    return jload(RES / "redesign" / f"{corpus}_metrics.json")["years"][YEAR]["selectors"]


# ---------------------------------------------------------------- the tail
def fig_tail():
    ev = load_eval(YEAR)
    labels = ev["labels"]
    f1 = per_code_f1(ev["gold"], [encoder_large_full(ev)], labels)
    n = ev["counts"]["full"][labels]
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.axvspan(0.7, 5.5, color=RARE_BG, lw=0, zorder=0)
    ax.text(np.sqrt(0.7 * 5.5), 1.04, "rare", ha="center", va="top", fontsize=9, color="#7F7F7F")
    rng = np.random.default_rng(0)
    ax.scatter(n * np.exp(rng.normal(0, 0.06, len(n))), f1, s=12, color=ENC, alpha=0.5, lw=0, zorder=2)
    zero = {}
    for name, (lo, hi) in BINS.items():
        m = (n >= lo) & (n <= hi)
        zero[name] = round(float((f1[m] == 0).mean()), 3)
        ax.hlines(np.median(f1[m]), max(lo, 0.8) * 0.9, min(hi, n.max()) * 1.1, color="black", lw=2.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.7, n.max() * 1.6)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("Training reports per code")
    ax.set_ylabel("Per-code F1")
    print("  tail: codes at F1 = 0 by bin", zero)
    save(fig, "fig_tail")


# ---------------------------------------------------------------- crossover
def fig_crossover():
    ev = load_eval(YEAR)
    maude = {"Encoder": bin_scores(ev, [encoder_large_full(ev)]),
             "LLM (Opus 5.5)": bin_scores(ev, [ev["llm"]["opus55"]]),
             "LLM (Gemma 4 31B)": bin_scores(ev, [ev["llm"]["gemma4_31b"]])}
    if YEAR == "2025":
        sdr = jload(RES / "sdr2025" / "sdr_metrics.json")["arms_2025_macro_f1_bins_by_full_counts"]
        air = {"Encoder": [sdr["cb_large_long_full"][b] for b in BIN_NAMES],
               "LLM (Opus 5.5)": [sdr["zs_opus55"][b] for b in BIN_NAMES],
               "LLM (Gemma 4 31B)": [sdr["zs_gemma4_31b"][b] for b in BIN_NAMES]}
    else:  # the SDR selector encoder and the zero-shot LLMs, bins by full training counts
        s = redesign("sdr")
        air = {"Encoder": [s["opus55"]["macro_f1"]["enc"][b] for b in BIN_NAMES],
               "LLM (Opus 5.5)": [s["opus55"]["macro_f1"]["zs"][b] for b in BIN_NAMES],
               "LLM (Gemma 4 31B)": [s["gemma4_31b"]["macro_f1"]["zs"][b] for b in BIN_NAMES]}
    print("  crossover MAUDE", {k: [round(v, 3) for v in vs] for k, vs in maude.items()})
    print("  crossover SDR", air)
    style = {"Encoder": dict(color=ENC, marker="s", ls="-"),
             "LLM (Opus 5.5)": dict(color=LLM, marker="o", ls="-"),
             "LLM (Gemma 4 31B)": dict(color=LLM, marker="^", ls="--")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    x = np.arange(4)
    for ax, data, title in zip(axes, (maude, air), ("(a) MAUDE", "(b) SDR")):
        for k, ys in data.items():
            ax.plot(x, ys, label=k, **style[k])
        ax.set_xticks(x)
        ax.set_xticklabels(BL)
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(0, 0.7)
        ax.set_title(title)
        ax.set_xlabel("Training examples per mode")
    axes[0].set_ylabel("Macro-F1")
    axes[1].legend(loc="lower right")
    save(fig, "fig_crossover")


# ---------------------------------------------------------------- selector
def fig_selector():
    if YEAR == "2025":
        panels = [("(a) MAUDE", jload(RES / "maude2025" / "selector_metrics.json")["macro_f1_2025"]),
                  ("(b) SDR", jload(RES / "sdr2025" / "selector_metrics_cb-long.json")["macro_f1_2025"])]
    else:
        panels = [("(a) MAUDE", redesign("maude")["opus55"]["macro_f1"]),
                  ("(b) SDR", redesign("sdr")["opus55"]["macro_f1"])]
    arms = [("enc", "Encoder", ENC), ("zs", "LLM", LLM), ("rag", "LLM + retrieval", RAG), ("selector", "Ours", SEL)]
    cats = ["all"] + BIN_NAMES
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    w = 0.2
    for ax, (title, m) in zip(axes, panels):
        for j, (k, lab, col) in enumerate(arms):
            ax.bar(np.arange(len(cats)) + (j - 1.5) * w, [m[k][c] for c in cats], w, color=col, label=lab,
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(["All"] + BL)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 0.8)
        ax.set_title(title)
        ax.set_xlabel("Training examples per mode")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(loc="upper left", ncol=2)
    save(fig, "fig_selector")


# ---------------------------------------------------------------- natural proportions
def fig_natural():
    d = jload(RES / f"maude{YEAR}nat" / "natural_metrics.json")
    pts = [("Encoder", "enc", dict(marker="s", c=ENC, s=40, edgecolors="black", linewidths=0.5)),
           ("LLM", "zs", dict(marker="o", c=LLM, s=44, edgecolors="black", linewidths=0.5)),
           ("LLM + retrieval", "rag", dict(marker="^", c=RAG, s=50, edgecolors="black", linewidths=0.5)),
           ("Frequency rule", "llm_rare_k_star", dict(marker="D", c=RULE, s=34, edgecolors="black", linewidths=0.5)),
           ("Ours, cost-gated", "S_cost", dict(marker="*", facecolors="white", edgecolors=SEL, s=140, linewidths=1.3)),
           ("Ours", "selector", dict(marker="*", c=SEL, s=170, edgecolors="black", linewidths=0.5))]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for lab, k, kw in pts:
        ax.scatter(d["uniform"][k]["micro_f1"], d["rare"]["rare_1-5"]["surfaced_weighted_whole_year"][k], label=lab,
                   zorder=4 if k == "rag" else 3, **kw)  # retrieval drawn over the hollow cost-gate star
    ys = [d["rare"]["rare_1-5"]["surfaced_weighted_whole_year"][k] for _, k, _ in pts]
    xs = [d["uniform"][k]["micro_f1"] for _, k, _ in pts]
    ax.set_xlim(min(0.38, min(xs) - 0.03), max(0.80, max(xs) + 0.03))
    ax.set_ylim(0, max(125, 1.15 * max(ys)))
    ax.set_xlabel("Accuracy, natural proportions")
    ax.set_ylabel("Rare-mode reports placed")
    ax.legend(loc="lower left", scatterpoints=1)
    save(fig, "fig_natural")


FIGS = {f.__name__: f for f in (fig_tail, fig_crossover, fig_selector, fig_natural)}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--year" in args:
        i = args.index("--year")
        YEAR = args[i + 1]
        del args[i:i + 2]
    for k in args or list(FIGS):
        FIGS[k]()
