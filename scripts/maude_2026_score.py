"""Score the 2026 replication once (notes/prereg-maude-2026-2026-11-05.md).

For Qwen3.8-27B and Gemma 4 31B separately, with everything frozen on 2024:
  T1 selector - best single arm chosen on 2024        (macro-F1, all codes)
  T2 selector - confidence cascade chosen on 2024     (arm and theta by 2024 macro-F1)
  T3 selector - frequency rule (llm_rare, frozen k*)
  T4 zero-shot LLM - CB-large encoder, 1-5 bin (full N)
Each holds when the item-level and the two-stage code-level 95% intervals both clear zero.

Usage: python maude_2026_score.py                    -> results/maude2026/metrics_2026.json (Qwen, Gemma)
       python maude_2026_score.py --models opus55    -> results/maude2026/metrics_2026_opus55.json (addendum)
"""
import argparse
import json

import numpy as np

from common import ROOT, rdir
from maude_extra_score import read_enc_csv
from maude_router import macro_f1
from maude_selector import (BINS, git_head, llm_rare, load_coders, load_frozen, predict_frozen, rule3, gate,
                            targets)
from maude_selector_score import contrast, positive, resamples, table

THETAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def cascade(D, arm, theta):
    return np.where((D["enc_p"] < theta) & (D[arm] >= 0), D[arm], D["enc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen38_27b,gemma4_31b",
                    help="comma-separated; opus55 is the Opus addendum (prereg section 6)")
    a = ap.parse_args()
    llms = a.models.split(",")
    rep = {"prereg": "notes/prereg-maude-2026-2026-11-05.md", "git_head_at_score": git_head(), "models": {}}
    for llm in llms:
        frozen, models = load_frozen(llm)
        D24, D = load_coders("2024", llm), load_coders("2026", llm)
        sel, j = predict_frozen(D, frozen, models)
        best = frozen["best_single_arm_2024"]
        rare = llm_rare(D, frozen["llm_rare_k_star"]["k"])
        grid = {(a, t): macro_f1(D24["gold"], cascade(D24, a, t), D24["labels"]) for a in ("zs", "rag") for t in THETAS}
        arm, th = max(grid, key=grid.get)
        casc = cascade(D, arm, th)
        samples = resamples(len(D["gold"]))
        bin15 = D["bins"]["1-5"]
        tests = {
            "T1_selector_minus_best_single": contrast(D, sel, D[best], D["labels"], samples),
            "T2_selector_minus_cascade": contrast(D, sel, casc, D["labels"], samples),
            "T3_selector_minus_llm_rare": contrast(D, sel, rare, D["labels"], samples),
            "T4_zeroshot_minus_encoder_1-5": contrast(D, D["zs"], D["enc"], bin15, samples),
        }
        arms = {"selector": sel, "encoder_cb_large": D["enc"], "zs": D["zs"], "rag": D["rag"], "llm_rare": rare,
                "cascade": casc, "S1": rule3(D, frozen["S1"]["t"], frozen["S1"]["k"]),
                "oracle": np.where(targets(D).any(axis=1), D["gold"], D["enc"])}
        if frozen.get("S_cost"):
            g = gate(D, frozen["S_cost"]["k"], frozen["S_cost"]["p"])
            arms["S_cost"] = np.where(g, sel, D["enc"])
        base = rdir("maude2026") / "encoder_cb_full_seed42.csv"
        if base.exists():
            arms["encoder_cb_base"] = read_enc_csv({"codes": D["codes"], "keys": D["keys"]}, base)[0]
        R = {"best_single_arm_2024": best, "cascade_chosen_2024": {"arm": arm, "theta": th, "macro_f1_2024": round(grid[(arm, th)], 4)},
             "n_2026": len(D["gold"]), "codes": int(len(D["labels"])),
             "macro_f1_2026": {k: table(D, p) for k, p in arms.items()},
             "tests": {k: {**v, "holds": positive(v)} for k, v in tests.items()},
             "choice_share_2026": {a: round(float((j == i).mean()), 4) for i, a in enumerate(("enc", "zs", "rag"))}}
        rep["models"][llm] = R
        print(f"\n{llm}: best single (2024) {best}; cascade (2024) {arm}@{th}")
        print("  macro-F1 2026:", {k: v["all"] for k, v in R["macro_f1_2026"].items()})
        for k, v in R["tests"].items():
            print(f"  {k}: {v['item']['point']:+.4f} item {v['item']['ci95']} code {v['code']['ci95']} -> {'HOLDS' if v['holds'] else 'fails'}")
    name = "metrics_2026.json" if llms == ["qwen38_27b", "gemma4_31b"] else f"metrics_2026_{'_'.join(llms)}.json"
    out = rdir("maude2026") / name
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    res = ROOT / "results" / "maude2026"
    res.mkdir(parents=True, exist_ok=True)
    (res / name).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    _ = BINS


if __name__ == "__main__":
    main()
