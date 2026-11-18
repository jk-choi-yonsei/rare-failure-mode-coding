"""Selector against a confidence cascade (notes/prereg-cascade-baseline-2026-11-05.md), run once per
encoder tag under the active corpus (MAINT_CORPUS=sdr; MAINT_SELECTOR_ENC=cb or cb-long).

Cascade: the LLM answer (Opus zero-shot or retrieval) replaces the encoder's when the encoder's
softmax score is below theta and the LLM answer is valid. The arm and theta are chosen together
on the 2024 known test items by macro-F1, frozen, and applied once to the 2025 test.
Contrast: selector - cascade, macro-F1 over every 2025 test code; item-level and two-stage
code-level 95% intervals (both must clear zero), per-bin values descriptive.

Usage: MAINT_CORPUS=sdr MAINT_SELECTOR_ENC=cb python cascade_baseline_score.py
"""
import json

import numpy as np

from common import CORPUS, ROOT, rdir
from maude_router import macro_f1
from maude_selector import BINS, SELECTOR_ENC, git_head, load_coders, load_frozen, predict_frozen
from maude_selector_score import contrast, positive, resamples, table

THETAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def cascade(D, arm, theta):
    return np.where((D["enc_p"] < theta) & (D[arm] >= 0), D[arm], D["enc"])


def main():
    frozen, models = load_frozen("opus55")
    if frozen.get("encoder_tag", "cb-large") != SELECTOR_ENC:
        raise SystemExit(f"frozen selector uses {frozen.get('encoder_tag')}, MAINT_SELECTOR_ENC is {SELECTOR_ENC}")
    D24, D = load_coders("2024", "opus55"), load_coders("2025", "opus55")
    grid = {(arm, th): round(macro_f1(D24["gold"], cascade(D24, arm, th), D24["labels"]), 4)
            for arm in ("zs", "rag") for th in THETAS}
    arm, th = max(grid, key=grid.get)
    casc = cascade(D, arm, th)
    sel, _ = predict_frozen(D, frozen, models)
    samples = resamples(len(D["gold"]))
    c = contrast(D, sel, casc, D["labels"], samples)
    rep = {"prereg": "notes/prereg-cascade-baseline-2026-11-05.md", "corpus": CORPUS, "encoder_tag": SELECTOR_ENC,
           "git_head_at_score": git_head(), "cascade_grid_2024": {f"{a}@{t}": v for (a, t), v in grid.items()},
           "cascade_chosen": {"arm": arm, "theta": th, "macro_f1_2024": grid[(arm, th)]},
           "macro_f1_2025": {"selector": table(D, sel), "cascade": table(D, casc), "encoder": table(D, D["enc"]),
                             "zs": table(D, D["zs"]), "rag": table(D, D["rag"])},
           "contrast_selector_minus_cascade": c, "holds": positive(c)}
    name = f"cascade_metrics_{SELECTOR_ENC}.json"
    out = rdir("maude2025") / name
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    res = ROOT / "results" / rdir("maude2025").name
    res.mkdir(parents=True, exist_ok=True)
    (res / name).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"{CORPUS} / {SELECTOR_ENC}: cascade chosen on 2024 = {arm} at theta {th} (2024 macro-F1 {grid[(arm, th)]})")
    print("  2025 macro-F1:", {k: v["all"] for k, v in rep["macro_f1_2025"].items()})
    print("  by bin selector:", rep["macro_f1_2025"]["selector"], "\n  by bin cascade: ", rep["macro_f1_2025"]["cascade"])
    print("  selector - cascade:", c, "->", "HOLDS" if rep["holds"] else "fails")
    _ = BINS


if __name__ == "__main__":
    main()
