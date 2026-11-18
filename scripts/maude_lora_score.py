"""Score the LoRA arm on the 2025 test set (notes/prereg-maude-lora-2026-10-09.md).

Confirmatory, one-sided, 95% paired item bootstrap; every cell must hold, for
the zero-shot Qwen and Gemma both:
  H-F1  zero-shot LLM - LoRA > 0 in bin 1-5, N in {10k, 100k, full}
        (also reported per N as holds / reversed / indistinguishable, and as a
        partial verdict over 10k and 100k)
  H-F2  zero-shot LLM - LoRA < 0 in bin 21-89, N in {100k, full} (a manipulation
        check: it fails only if LoRA is broken)
  H-F3  retrieval few-shot LLM - LoRA > 0 in bin 1-5: 10k (Qwen, Gemma), 100k
        (Qwen), full (Qwen, Gemma)
A cell is "adequate" when LoRA's mean validation macro-F1 at that N reaches
CB-large's (LARGE_VAL); a held cell at an inadequate N is "held against an
under-trained LoRA". Each verdict is "holds", "fails" or "incomplete". Edge
cells (EDGE) are two-sided, each with the outcome expected in advance.
Everything is scored again without the near-duplicate items.

Descriptive: overall and per-bin LoRA against CB-base, CB-large (at 10k also
seeds 42 and 43 alone; seed 44 collapsed in practice), the zero-shot LLMs
(including gpt-oss and the size control Qwen3-4B) and the retrieval few-shot
LLMs; validation F1 and budget flags; LoRA on the 2024 test set; the frozen
llm_rare router with LoRA as the fallback.

Usage: python maude_lora_score.py  -> runs/maude2025/lora_metrics.json
"""
import json

import numpy as np

from maude_extra_score import OUT, load_arms, outcome, paired, read_enc_csv, resamples
from maude_router import FROZEN, apply, load_eval, macro_f1

MODELS = ("qwen38_27b", "gemma4_31b")
EXTRA_ZS = ("gptoss20b", "qwen3_4b")
SIZES = ("10k", "100k", "full")
BINS = {"0": (0, 0), "1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
TAG = "cb-lora"
LORA_SEEDS = (42, 43, 44, 45, 46)  # 45, 46 replace collapsed 10k seeds, in order
LORA_TARGET = {"10k": 3, "100k": 1, "full": 1}
LARGE_HEALTHY_10K = 2  # CB-large 10k seeds 42 and 43; seed 44 collapsed in practice (val 0.0003 but one epoch)
# CB-large validation macro-F1 (10k: seeds 42 and 43) -- the adequacy bar for LoRA.
LARGE_VAL = {"10k": 0.3105, "100k": 0.4654, "full": 0.5946}
# (contrast, N, bin, expected, models): "+" / "-" = CI above / below zero, "0" = CI
# covers zero; two characters allow either. "large" at 10k means seeds 42 and 43.
EDGE = [
    ("lora_minus_large", "10k", "all", "+0", (None,)), ("lora_minus_large", "100k", "all", "+", (None,)),
    ("lora_minus_large", "full", "all", "+", (None,)),
    *[("lora_minus_large", s, b, "+0", (None,)) for s in SIZES for b in ("1-5", "6-20")],
    ("rag_minus_lora", "10k", "inpool", "+", MODELS), ("rag_minus_lora", "100k", "all", "+0", MODELS),
    ("rag_minus_lora", "full", "all", "0-", MODELS), ("rag_minus_lora", "full", "21-89", "-", MODELS),
    ("zs_minus_lora", "10k", "6-20", "-", MODELS), ("zs_minus_lora", "100k", "6-20", "-", MODELS),
    ("zs_minus_lora", "full", "6-20", "0-", MODELS),
    ("zs_minus_lora", "10k", "1-5", "-", ("gptoss20b",)), ("zs_minus_lora", "full", "1-5", "0-", ("gptoss20b",)),
    *[("zs_minus_lora", s, "1-5", "-0", ("qwen3_4b",)) for s in SIZES],
]


def lora_runs(size):
    """[(seed, meta)] of the LoRA runs that count at N (meta written, not collapsed,
    the first LORA_TARGET in seed order) and the collapsed seeds."""
    keep, collapsed = [], []
    for s in LORA_SEEDS:
        m = OUT / f"encoder_{TAG}_{size}_seed{s}_meta.json"
        if m.exists():
            meta = json.loads(m.read_text())
            (collapsed if meta.get("collapsed") else keep).append((s, meta))
    return keep[:LORA_TARGET[size]], [s for s, _ in collapsed]


def load_lora(ev):
    out = {}
    for size in SIZES:
        runs, collapsed = lora_runs(size)
        preds = [read_enc_csv(ev, OUT / f"encoder_{TAG}_{size}_seed{s}.csv") for s, _ in runs]
        val = float(np.mean([m["val_macro_f1"] for _, m in runs])) if runs else None
        out[size] = {"seeds": [s for s, _ in runs], "collapsed": collapsed, "metas": [m for _, m in runs],
                     "preds": [p for p, _ in preds], "conf": [c for _, c in preds],
                     "complete": len(runs) == LORA_TARGET[size], "val": val,
                     "adequate": None if val is None else val >= LARGE_VAL[size]}
    return out


def evaluate(ev, A, L, keep=None):
    sel = np.arange(len(ev["gold"])) if keep is None else np.flatnonzero(keep)
    gold = ev["gold"][sel]
    n = len(gold)
    samples = resamples(n)
    whole = np.arange(n)
    present = np.array(sorted(set(gold.tolist())))

    def F(preds_list, labels):
        P = [p[sel] for p in preds_list]
        return lambda s: float(np.mean([macro_f1(gold, p, labels, s) for p in P]))

    def val(preds_list, labels):
        return round(F(preds_list, labels)(whole), 4)

    def pair(a, b, labels):
        return paired(F(a, labels), F(b, labels), n, samples)

    def labels_for(size, b):
        if b == "all":
            return present
        lo, hi = (1, 10**9) if b == "inpool" else BINS[b]
        return np.array([c for c in present if lo <= ev["counts"][size][c] <= hi])

    frozen = json.loads(FROZEN.read_text())["routers"]
    rep = {"n": n, "sizes": {}}
    for size in SIZES:
        lp = L[size]["preds"]
        if not lp:
            continue
        cb = [p for p, _ in ev["enc"][size]]
        large_all = A["large"][size]["preds"]
        large = large_all[:LARGE_HEALTHY_10K] if size == "10k" else large_all
        blk = {"seeds": L[size]["seeds"], "collapsed_seeds": L[size]["collapsed"], "complete": L[size]["complete"],
               "val": L[size]["val"], "adequate": L[size]["adequate"], "large_val_bar": LARGE_VAL[size],
               "per_seed": [val([p], present) for p in lp], "cells": {}}
        for b in ["all", "inpool"] + list(BINS):
            labels = labels_for(size, b)
            if len(labels) == 0:
                continue
            c = {"n_codes": len(labels), "lora": val(lp, labels), "cb": val(cb, labels),
                 "lora_minus_cb": pair(lp, cb, labels)}
            if large:
                c["cb_large"] = val(large, labels)
                c["lora_minus_large"] = pair(lp, large, labels)
                if size == "10k":  # as registered for H-L: all three seeds
                    c["cb_large_3seeds"] = val(large_all, labels)
                    c["lora_minus_large_3seeds"] = pair(lp, large_all, labels)
            for mk in MODELS + EXTRA_ZS:
                if mk in ev["llm"]:
                    c[f"zs_{mk}"] = val([ev["llm"][mk]], labels)
                    c[f"zs_minus_lora_{mk}"] = pair([ev["llm"][mk]], lp, labels)
                rag = A["rag"].get((size, mk))
                if rag is not None:
                    c[f"rag_{mk}"] = val([rag], labels)
                    c[f"rag_minus_lora_{mk}"] = pair([rag], lp, labels)
            blk["cells"][b] = c
        # Frozen llm_rare router (2024-chosen count threshold on the LLM's answer), LoRA as the fallback.
        routers = {}
        for mk in MODELS:
            spec = frozen.get(f"{size}/{mk}")
            if not spec:
                continue
            k = spec["families"]["llm_rare"]["param"]
            for kind, ans in (("zero_shot", ev["llm"].get(mk)), ("rag", A["rag"].get((size, mk)))):
                if ans is not None:
                    vals = [macro_f1(gold, apply("llm_rare", k, p[sel], c[sel], ans[sel], ev["counts"][size])[0], present)
                            for p, c in zip(lp, L[size]["conf"])]
                    routers[f"{mk}/{kind}"] = round(float(np.mean(vals)), 4)
        blk["llm_rare_router"] = routers
        rep["sizes"][size] = blk

    edge = []
    for contrast, size, b, expected, models in EDGE:
        cells = rep["sizes"].get(size, {}).get("cells", {}).get(b, {})
        for mk in models:
            if contrast == "rag_minus_lora" and (size, mk) not in A["rag"]:
                continue  # no rag run planned at this N for this model (100k: Qwen only)
            c = cells.get(contrast if mk is None else f"{contrast}_{mk}")
            o = outcome(c)
            edge.append({"contrast": contrast, "N": size, "bin": b, "model": mk, "expected": expected,
                         "cell": c, "outcome": o, "as_expected": None if o is None else o in expected})
    rep["edge"] = edge
    rep["verdicts"], rep["h_f1_by_n"] = verdicts(rep, L, A)
    return rep


def verdicts(rep, L, A):
    def cell(size, b, key):
        s = rep["sizes"].get(size)
        return s["cells"].get(b, {}).get(key) if s and L[size]["complete"] else None

    def judge(cells, sign):
        if any(c is None for c in cells):
            return "incomplete"
        return "holds" if all((c["ci95"][0] > 0) if sign > 0 else (c["ci95"][1] < 0) for c in cells) else "fails"

    f1 = {s: [cell(s, "1-5", f"zs_minus_lora_{m}") for m in MODELS] for s in SIZES}
    by_n = {}
    for s, cells in f1.items():
        outs = [outcome(c) for c in cells]
        reading = ("incomplete" if None in outs else "reversed" if "-" in outs
                   else "holds" if all(o == "+" for o in outs) else "indistinguishable")
        if reading == "holds" and not L[s]["adequate"]:
            reading = "holds against an under-trained LoRA"
        by_n[s] = {"cells": outs, "reading": reading, "adequate": L[s]["adequate"]}
    rag3 = [cell(s, "1-5", f"rag_minus_lora_{m}") for s in SIZES for m in MODELS if (s, m) in A["rag"]]
    return {
        "H_F1_llm_beats_lora_1_5": judge([c for s in SIZES for c in f1[s]], 1),
        "H_F1_partial_10k_100k": judge(f1["10k"] + f1["100k"], 1),
        "H_F2_lora_beats_llm_21_89": judge([cell(s, "21-89", f"zs_minus_lora_{m}") for s in ("100k", "full")
                                           for m in MODELS], -1),
        "H_F3_rag_beats_lora_1_5": judge(rag3, 1),
    }, by_n


def lora_2024():
    """LoRA on the 2024 test items next to the retrained CB-base (2024 -> 2025 drop)."""
    ev = load_eval("2024rerun")
    out = {}
    for size in SIZES:
        runs, _ = lora_runs(size)
        paths = [OUT / "rerun2024" / f"encoder_{TAG}_{size}_seed{s}.csv" for s, _ in runs]
        if runs and all(p.exists() for p in paths):
            out[size] = {"lora": round(float(np.mean([macro_f1(ev["gold"], read_enc_csv(ev, p)[0], ev["labels"])
                                                      for p in paths])), 4),
                         "cb_base": round(float(np.mean([macro_f1(ev["gold"], p, ev["labels"])
                                                         for p, _ in ev["enc"].get(size, [])])), 4)}
    return out


def run_info():
    out = {}
    for size in SIZES:
        runs, collapsed = lora_runs(size)
        out[size] = [{"seed": s, "lr": m["lr"], "lr_inherited_from": m.get("lr_inherited_from"),
                      "val_macro_f1": m["val_macro_f1"], "lr_at_top_of_grid": m.get("lr_at_top_of_grid"),
                      "best_at_last_epoch": m.get("best_at_last_epoch"),
                      "epochs": {lr: [h["val_macro_f1"] for h in r["epochs"]] for lr, r in m["lr_runs"].items()},
                      "partial": {lr: r.get("partial") for lr, r in m["lr_runs"].items() if r.get("partial")}}
                     for s, m in runs]
        if collapsed:
            out[size].append({"collapsed_seeds": collapsed})
    return out


def main():
    ev = load_eval("2025")
    A = load_arms(ev)
    L = load_lora(ev)
    report = {"runs": run_info(), "main": evaluate(ev, A, L), "lora_2024": lora_2024()}
    nd = OUT / "neardup.json"
    if nd.exists():
        cos = json.loads(nd.read_text())
        keep = np.array([cos[str(k)][0] < 0.9 for k in ev["keys"]])
        report["neardup_excluded"] = evaluate(ev, A, L, keep)
        report["verdicts_changed_without_neardup"] = {
            h: [v, report["neardup_excluded"]["verdicts"][h]]
            for h, v in report["main"]["verdicts"].items() if v != report["neardup_excluded"]["verdicts"][h]}
    (OUT / "lora_metrics.json").write_text(json.dumps(report, indent=1))
    show(report)


def fmt(c):
    return "-" if c is None else f"{c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{'*' if c['excludes_zero'] else ''}"


def show(r):
    print("===== runs:", json.dumps(r["runs"]))
    for size, b in r["main"]["sizes"].items():
        print(f"===== LoRA N={size} seeds {b['seeds']} (collapsed {b['collapsed_seeds']}, complete {b['complete']}) "
              f"val {b['val']} vs CB-large bar {b['large_val_bar']} -> adequate {b['adequate']}; per seed {b['per_seed']}; "
              f"llm_rare router {b['llm_rare_router']}")
        for name, c in b["cells"].items():
            parts = [f"LoRA {c['lora']:.3f}", f"CB {c['cb']:.3f}", f"large {c.get('cb_large', float('nan')):.3f}",
                     f"LoRA-large {fmt(c.get('lora_minus_large'))}"]
            for mk in MODELS + EXTRA_ZS:
                if f"zs_minus_lora_{mk}" in c:
                    parts.append(f"{mk[:5]} zs-LoRA {fmt(c[f'zs_minus_lora_{mk}'])}")
                if f"rag_minus_lora_{mk}" in c:
                    parts.append(f"{mk[:5]} rag-LoRA {fmt(c[f'rag_minus_lora_{mk}'])}")
            print(f"   {name:6} ({c['n_codes']} codes): " + "  ".join(parts))
    print("===== edge cells")
    for e in r["main"]["edge"]:
        print(f"   {e['contrast']:17} {e['N']:4} {e['bin']:6} {str(e['model'])[:8]:8} expected {e['expected']:2} -> "
              f"{e['outcome']} {fmt(e['cell'])}")
    print("2024 test, LoRA vs CB-base:", r["lora_2024"])
    print("H-F1 by N:", json.dumps(r["main"]["h_f1_by_n"]))
    print("verdicts:", json.dumps(r["main"]["verdicts"], indent=1))
    if "neardup_excluded" in r:
        print(f"without near-duplicates (n={r['neardup_excluded']['n']}): changed {r['verdicts_changed_without_neardup']}")


if __name__ == "__main__":
    main()
