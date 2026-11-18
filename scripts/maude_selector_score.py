"""Score the frozen starvation-aware selector on the 2025 test, once
(notes/prereg-maude-selector-2026-10-23.md).

Verdicts need both the item-level paired bootstrap (B=1000, seed 42) and the
two-stage code-level interval to exclude zero (or, for non-inferiority, to sit
above the margin):
  H-M1  selector - best single arm (chosen on 2024) > 0, overall macro-F1
  H-M2  selector - best single arm > 0 in bins 1-5 and 6-20, AND selector - encoder
        non-inferior in bin 90+ (5th percentile of the bootstrap difference > -0.01)
  H-M3  selector - llm_rare(k*) > 0, overall
  H-M4  S-cost - best single arm > 0, overall, with the LLM called on <= 50% of reports
Descriptive: every family, the oracle, choice shares by bin, the gate's call rate.

Usage: python maude_selector_score.py [--llm opus55]  -> runs/maude2025/selector_metrics.json
"""
import argparse
import json
import shutil

import numpy as np

from common import RUNS, SEED, rdir
from maude_claude_score import per_code_f1
from maude_router import macro_f1
from maude_selector import (ARMS, BINS, frozen_paths, gate, git_head, llm_rare, load_coders, load_frozen,
                            predict_frozen, rule3, sha256, features, targets)

B = 1000
MARGIN = 0.01


def out_paths(llm):
    from maude_selector import SELECTOR_ENC
    name = "selector_metrics" + ("" if llm == "opus55" else f"_{llm}") + ("" if SELECTOR_ENC == "cb-large" else f"_{SELECTOR_ENC}") + ".json"
    return rdir("maude2025") / name, RUNS.parent / "results" / rdir("maude2025").name / name


def resamples(n):
    rng = np.random.default_rng(SEED)
    return [rng.integers(0, n, n) for _ in range(B)]


def item_ci(gold, a, b, labels, samples):
    d = sorted(macro_f1(gold, a, labels, s) - macro_f1(gold, b, labels, s) for s in samples)
    return {"point": round(macro_f1(gold, a, labels) - macro_f1(gold, b, labels), 4),
            "ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)],
            "p05": round(d[round(0.05 * B) - 1], 4), "excludes_zero": d[round(0.025 * B) - 1] > 0 or d[round(0.975 * B) - 1] < 0}


def code_ci_p05(gold, a, b, labels, samples):
    """code_ci plus the 5th percentile, for the non-inferiority test."""
    rng = np.random.default_rng(SEED + 1)
    k = len(labels)
    d = sorted(float((per_code_f1(gold, [a], labels, s) - per_code_f1(gold, [b], labels, s))[rng.integers(0, k, k)].mean())
               for s in samples)
    return {"ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)],
            "p05": round(d[round(0.05 * B) - 1], 4), "excludes_zero": d[round(0.025 * B) - 1] > 0 or d[round(0.975 * B) - 1] < 0}


def contrast(D, a, b, labels, samples):
    return {"item": item_ci(D["gold"], a, b, labels, samples), "code": code_ci_p05(D["gold"], a, b, labels, samples)}


def positive(c):
    return c["item"]["ci95"][0] > 0 and c["code"]["ci95"][0] > 0


def non_inferior(c):
    return c["item"]["p05"] > -MARGIN and c["code"]["p05"] > -MARGIN


def table(D, pred):
    return {"all": round(macro_f1(D["gold"], pred, D["labels"]), 4),
            **{b: round(macro_f1(D["gold"], pred, D["bins"][b]), 4) for b in BINS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="opus55")
    a = ap.parse_args()
    frozen, models = load_frozen(a.llm)
    if frozen["llm"] != a.llm:
        raise SystemExit(f"the frozen selector was fitted for {frozen['llm']}, not {a.llm}")
    from maude_selector import SELECTOR_ENC
    if frozen.get("encoder_tag", "cb-large") != SELECTOR_ENC:
        raise SystemExit(f"the frozen selector uses encoder {frozen.get('encoder_tag', 'cb-large')}; set MAINT_SELECTOR_ENC to match")
    OUT, RESULTS = out_paths(a.llm)
    FROZEN_JSON = frozen_paths(a.llm)[0]
    D = load_coders("2025", a.llm)
    n = len(D["gold"])
    samples = resamples(n)
    best_name = frozen["best_single_arm_2024"]
    arms = {a_: D[a_] for a_ in ARMS}
    sel, j_main = predict_frozen(D, frozen, models)
    fam = {f: predict_frozen(D, frozen, models, family=f)[0] for f in ("S1", "S2", "S3")}
    if frozen.get("S_cost"):
        g = gate(D, frozen["S_cost"]["k"], frozen["S_cost"]["p"])
    else:  # no gate kept the 2024 call rate at or under 50% (a weak encoder is unconfident everywhere)
        g = np.ones(n, bool)
    scost = np.where(g, sel, D["enc"])
    rare = llm_rare(D, frozen["llm_rare_k_star"]["k"])
    Y = targets(D)
    oracle = np.where(Y.any(axis=1), D["gold"], D["enc"])

    rep = {"frozen": {"file": FROZEN_JSON.name, "sha256": sha256(FROZEN_JSON), "main_family": frozen["main_family"],
                      "git_head_at_fit": frozen["git_head_at_fit"], "git_head_at_score": git_head()},
           "llm": a.llm, "n_2025": n, "best_single_arm_2024": best_name,
           "macro_f1_2025": {"selector": table(D, sel), **{f"family_{f}": table(D, p) for f, p in fam.items()},
                             "S_cost": table(D, scost), "llm_rare_k_star": table(D, rare), "oracle": table(D, oracle),
                             **{a_: table(D, p) for a_, p in arms.items()}},
           "S_cost_call_rate_2025": round(float(g.mean()), 4),
           "choice_share_2025": {a_: round(float((j_main == i).mean()), 4) for i, a_ in enumerate(ARMS)},
           "choice_share_by_bin_2025": {}}
    cbin = D["counts"][D["gold"]]
    for b, (lo, hi) in BINS.items():
        m = (cbin >= lo) & (cbin <= hi)
        rep["choice_share_by_bin_2025"][b] = {a_: round(float((j_main[m] == i).mean()), 4) for i, a_ in enumerate(ARMS)} if m.any() else None

    best = arms[best_name]
    c = {"H-M1": contrast(D, sel, best, D["labels"], samples),
         "H-M2_1-5": contrast(D, sel, best, D["bins"]["1-5"], samples),
         "H-M2_6-20": contrast(D, sel, best, D["bins"]["6-20"], samples),
         "H-M2_90+_vs_enc": contrast(D, sel, D["enc"], D["bins"]["90+"], samples),
         "H-M3": contrast(D, sel, rare, D["labels"], samples),
         "H-M4": contrast(D, scost, best, D["labels"], samples)}
    rep["contrasts"] = c
    rep["verdicts"] = {
        "H-M1": "holds" if positive(c["H-M1"]) else "fails",
        "H-M2": "holds" if positive(c["H-M2_1-5"]) and positive(c["H-M2_6-20"]) and non_inferior(c["H-M2_90+_vs_enc"]) else "fails",
        "H-M3": "holds" if positive(c["H-M3"]) else "fails",
        "H-M4": ("not applicable (no gate under 50% on 2024)" if not frozen.get("S_cost")
                 else "holds" if positive(c["H-M4"]) and g.mean() <= 0.5 else "fails"),
    }
    # exploratory: the other families against the best arm, and feature importances where available
    rep["exploratory"] = {f"family_{f}_minus_best": item_ci(D["gold"], p, best, D["labels"], samples) for f, p in fam.items()}
    if frozen["main_family"] == "S3":
        try:
            from sklearn.inspection import permutation_importance
            X = features(D)
            imp = {}
            for i, a_ in enumerate(ARMS):
                m = models["S3"].models[i]
                if m is not None:
                    r = permutation_importance(m, X, Y[:, i], n_repeats=5, random_state=SEED)
                    imp[a_] = dict(zip(frozen["features"], np.round(r.importances_mean, 4).tolist()))
            rep["exploratory"]["permutation_importance_2025"] = imp
        except Exception as e:  # noqa: BLE001
            rep["exploratory"]["permutation_importance_2025"] = str(e)
    if frozen["main_family"] == "S2":
        rep["exploratory"]["S2_coefficients"] = frozen.get("S2_coefficients")

    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT, RESULTS)
    print(f"2025 n={n}; main family {frozen['main_family']}; best single arm (2024) {best_name}")
    print(f"{'arm':22s} " + " ".join(f"{k:>7s}" for k in ("all", *BINS)))
    for name, t in rep["macro_f1_2025"].items():
        print(f"{name:22s} " + " ".join(f"{t[k]:7.3f}" for k in ("all", *BINS)))
    for h, v in rep["verdicts"].items():
        print(f"{h}: {v}")
    for k, v in c.items():
        print(f"  {k}: item {v['item']['point']} {v['item']['ci95']} | code {v['code']['ci95']} | p05 item {v['item']['p05']} code {v['code']['p05']}")
    print(f"S-cost call rate 2025 {rep['S_cost_call_rate_2025']}; choice shares {rep['choice_share_2025']}")
    print(f"-> {OUT} (copied to results/)")


if __name__ == "__main__":
    main()
