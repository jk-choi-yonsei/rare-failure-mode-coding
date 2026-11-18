"""Score the natural-distribution 2025 sample (notes/prereg-maude-natural-2026-10-23.md), once.

Strata: uniform (4,000 reports drawn uniformly) and rare (a census of the reports under
codes with 1-5 training examples, a 1,500 sample of those with 6-20). The selector is
the frozen one (splits/maude_selector_v1*.json); nothing is refitted here.
  N1  uniform: selector - retrieval LLM > 0, macro-F1 over the codes present (item and
      code-level intervals both clear of zero)
  N2  uniform: selector - encoder >= -0.01, micro-F1 (5th percentile of the item bootstrap)
  N3  rare: selector - encoder > 0, recall, bins 1-5 and 6-20 separately (item-level
      interval and two-stage interval of the per-code recall mean, both clear of zero)
Descriptive: every arm per stratum, per-bin macro-F1 and recall, choice shares, S-cost
call rate, weighted whole-year counts of rare-mode reports surfaced, code-frequency
rank agreement, near-duplicate sensitivity (top-1 cosine >= 0.9).

Usage: python maude_natural_score.py [--llm opus55] [--year 2026]  -> runs/maude<year>nat/natural_metrics.json
With --year 2026 (notes/prereg-redesign-2026-11-06.md, F): the 2026 natural sample, and N1 compares
the selector with the single coder its frozen file names (best_single_arm_2024) instead of retrieval.
"""
import argparse
import json
import shutil

import numpy as np
from scipy.stats import spearmanr

from common import RUNS, SEED, SPLITS, rdir
from maude_router import macro_f1
from maude_selector import (ARMS, BINS, frozen_paths, gate, git_head, llm_rare, load_coders, load_frozen,
                            predict_frozen, rule3, sha256, targets)

B = 1000
MARGIN = 0.01


def resamples(n, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


def micro(gold, pred, sample=None):
    g, p = (gold, pred) if sample is None else (gold[sample], pred[sample])
    return float((g == p).mean())


def recall(gold, pred, sample=None):
    return micro(gold, pred, sample)  # every item in a rare stratum carries a gold code of that bin


def per_code_recall(gold, pred, codes):
    out = []
    for c in codes:
        m = gold == c
        out.append(float((pred[m] == c).mean()) if m.any() else 0.0)
    return np.array(out)


def ci_items(f, gold, a, b, samples):
    d = sorted(f(gold, a, s) - f(gold, b, s) for s in samples)
    return {"point": round(f(gold, a) - f(gold, b), 4), "ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)],
            "p05": round(d[round(0.05 * B) - 1], 4), "excludes_zero": d[round(0.025 * B) - 1] > 0 or d[round(0.975 * B) - 1] < 0}


def ci_codes_macro(gold, a, b, labels, samples):
    """Two-stage bootstrap of macro-F1(a) - macro-F1(b) over `labels` (items, then codes)."""
    rng = np.random.default_rng(SEED + 1)
    k = len(labels)

    def f1s(pred, s):
        g, p = gold[s], pred[s]
        n = int(max(g.max(), p.max(), labels.max())) + 1
        tc, pc = np.bincount(g, minlength=n), np.bincount(p[p >= 0], minlength=n)
        tp = np.bincount(g[p == g], minlength=n)
        den = (pc + tc)[labels]
        return np.where(den > 0, 2 * tp[labels] / np.maximum(den, 1), 0.0)
    d = sorted(float((f1s(a, s) - f1s(b, s))[rng.integers(0, k, k)].mean()) for s in samples)
    return {"ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)],
            "p05": round(d[round(0.05 * B) - 1], 4), "excludes_zero": d[round(0.025 * B) - 1] > 0 or d[round(0.975 * B) - 1] < 0}


def ci_codes_recall(gold, a, b, codes, samples):
    rng = np.random.default_rng(SEED + 2)
    k = len(codes)
    d = []
    for s in samples:
        diff = per_code_recall(gold[s], a[s], codes) - per_code_recall(gold[s], b[s], codes)
        d.append(float(diff[rng.integers(0, k, k)].mean()))
    d.sort()
    return {"ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)],
            "p05": round(d[round(0.05 * B) - 1], 4), "excludes_zero": d[round(0.025 * B) - 1] > 0 or d[round(0.975 * B) - 1] < 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="opus55")
    ap.add_argument("--year", default="2025", choices=["2025", "2026"])
    a = ap.parse_args()
    frozen, models = load_frozen(a.llm)
    from maude_selector import SELECTOR_ENC
    if frozen.get("encoder_tag", "cb-large") != SELECTOR_ENC:
        raise SystemExit(f"the frozen selector uses encoder {frozen.get('encoder_tag', 'cb-large')}; set MAINT_SELECTOR_ENC to match")
    D = load_coders(f"{a.year}nat", a.llm)
    from maude_data import SPLIT_NATURAL, SPLIT_NATURAL_2026
    nat = json.loads((SPLIT_NATURAL if a.year == "2025" else SPLIT_NATURAL_2026).read_text(encoding="utf-8"))
    n1_ref = "rag" if a.year == "2025" else frozen["best_single_arm_2024"]
    pos = {k: i for i, k in enumerate(D["keys"])}
    strata = {name: np.array([pos[k] for k in keys]) for name, keys in nat["strata"].items()}
    weights = nat["weights"]
    gold, counts = D["gold"], D["counts"]
    sel, j_main = predict_frozen(D, frozen, models)
    g = gate(D, frozen["S_cost"]["k"], frozen["S_cost"]["p"])
    arms = {"enc": D["enc"], "zs": D["zs"], "rag": D["rag"], "selector": sel,
            "S_cost": np.where(g, sel, D["enc"]), "llm_rare_k_star": llm_rare(D, frozen["llm_rare_k_star"]["k"]),
            "S1": rule3(D, frozen["S1"]["t"], frozen["S1"]["k"]),
            "oracle": np.where(targets(D).any(axis=1), gold, D["enc"])}
    base_csv = rdir(f"maude{a.year}nat") / "encoder_cb_full_seed42.csv"
    if base_csv.exists():
        from maude_extra_score import read_enc_csv
        arms["enc_base"] = read_enc_csv({"codes": D["codes"], "keys": D["keys"]}, base_csv)[0]

    rep = {"frozen": {"file": frozen_paths(a.llm)[0].name, "sha256": sha256(frozen_paths(a.llm)[0]),
                      "main_family": frozen["main_family"], "git_head_at_score": git_head()},
           "llm": a.llm, "n_union": len(gold), "strata_n": {k: int(len(v)) for k, v in strata.items()}, "weights": weights,
           "near_dup_share": {k: round(float((D["nb_cos"][v] >= 0.9).mean()), 4) for k, v in strata.items()}}

    # --- uniform stratum
    U = strata["uniform"]
    labels_u = np.array(sorted(set(gold[U].tolist())))
    bins_u = {b: np.array([c for c in labels_u if lo <= counts[c] <= hi]) for b, (lo, hi) in BINS.items()}
    rep["uniform"] = {name: {"micro_f1": round(micro(gold, p, U), 4), "macro_f1": round(macro_f1(gold, p, labels_u, U), 4),
                             **{f"macro_f1_{b}": (round(macro_f1(gold, p, bins_u[b], U), 4) if len(bins_u[b]) else None) for b in BINS}}
                      for name, p in arms.items()}
    rep["uniform"]["items_by_bin"] = {b: int(sum(1 for i in U if lo <= counts[gold[i]] <= hi)) for b, (lo, hi) in BINS.items()}
    rep["uniform"]["S_cost_call_rate"] = round(float(g[U].mean()), 4)
    rep["uniform"]["choice_share"] = {arm: round(float((j_main[U] == i).mean()), 4) for i, arm in enumerate(ARMS)}
    samples_u = resamples(len(U))
    sub = {k: v[U] for k, v in arms.items()}
    gu = gold[U]
    n1 = {"item": ci_items(lambda gg, p, s=None: macro_f1(gg, p, labels_u, s), gu, sub["selector"], sub[n1_ref], samples_u),
          "code": ci_codes_macro(gu, sub["selector"], sub[n1_ref], labels_u, samples_u)}
    n2 = {"item": ci_items(micro, gu, sub["selector"], sub["enc"], samples_u)}
    rep["contrasts"] = {f"N1_selector_minus_{n1_ref}_macro": n1, "N2_selector_minus_enc_micro": n2,
                        "desc_selector_minus_enc_macro": {"item": ci_items(lambda gg, p, s=None: macro_f1(gg, p, labels_u, s), gu, sub["selector"], sub["enc"], samples_u),
                                                          "code": ci_codes_macro(gu, sub["selector"], sub["enc"], labels_u, samples_u)},
                        "desc_selector_minus_rag_micro": {"item": ci_items(micro, gu, sub["selector"], sub["rag"], samples_u)}}
    # code-frequency agreement: predicted vs gold counts over the codes present
    rep["uniform"]["frequency_agreement"] = {}
    gold_counts = np.bincount(gu, minlength=len(D["codes"]))[labels_u]
    top20 = set(labels_u[np.argsort(-gold_counts)[:20]].tolist())
    for name, p in sub.items():
        pc = np.bincount(p[p >= 0], minlength=len(D["codes"]))[labels_u]
        rho = spearmanr(gold_counts, pc).correlation
        ptop = set(labels_u[np.argsort(-pc)[:20]].tolist())
        rep["uniform"]["frequency_agreement"][name] = {"spearman": round(float(rho), 4), "top20_overlap": len(top20 & ptop),
                                                      "mean_abs_count_error": round(float(np.abs(pc - gold_counts).mean()), 3)}

    # --- rare strata
    rep["rare"] = {}
    n3 = {}
    for name_s, bin_name in (("rare_1-5", "1-5"), ("rare_6-20", "6-20")):
        R = strata[name_s]
        gr = gold[R]
        codes_r = np.array(sorted(set(gr.tolist())))
        samples_r = resamples(len(R), SEED + 3)
        subr = {k: v[R] for k, v in arms.items()}
        w = weights[name_s]
        rep["rare"][name_s] = {"n": int(len(R)), "codes": int(len(codes_r)), "weight": w,
                               "recall": {k: round(recall(gr, p), 4) for k, p in subr.items()},
                               "surfaced_weighted_whole_year": {k: round(float((p == gr).sum() * w), 1) for k, p in subr.items()},
                               "population_whole_year": round(len(R) * w),
                               "choice_share": {arm: round(float((j_main[R] == i).mean()), 4) for i, arm in enumerate(ARMS)},
                               "S_cost_call_rate": round(float(g[R].mean()), 4)}
        n3[bin_name] = {"item": ci_items(recall, gr, subr["selector"], subr["enc"], samples_r),
                        "code": ci_codes_recall(gr, subr["selector"], subr["enc"], codes_r, samples_r)}
    rep["contrasts"]["N3_selector_minus_enc_recall"] = n3

    def pos_(c):
        return c["item"]["ci95"][0] > 0 and c["code"]["ci95"][0] > 0
    rep["verdicts"] = {"N1": "holds" if pos_(n1) else "fails",
                       "N2": "holds" if n2["item"]["p05"] > -MARGIN else "fails",
                       "N3": "holds" if pos_(n3["1-5"]) and pos_(n3["6-20"]) else "fails"}
    # near-duplicate sensitivity on the uniform stratum
    keep = U[D["nb_cos"][U] < 0.9]
    rep["uniform_without_near_dups"] = {"n": int(len(keep)), **{name: {"micro_f1": round(micro(gold, p, keep), 4),
                                        "macro_f1": round(macro_f1(gold, p, np.array(sorted(set(gold[keep].tolist()))), keep), 4)}
                                        for name, p in arms.items()}}

    out = rdir(f"maude{a.year}nat") / ("natural_metrics.json" if a.llm == "opus55" else f"natural_metrics_{a.llm}.json")
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    res = RUNS.parent / "results" / rdir(f"maude{a.year}nat").name / out.name
    res.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, res)
    print(f"natural sample ({a.llm}): union {len(gold)}; strata {rep['strata_n']}; near-dup {rep['near_dup_share']}")
    print(f"{'arm':16s} {'micro':>7s} {'macro':>7s} | rare recall 1-5 / 6-20 | surfaced (weighted, whole year) 1-5 / 6-20")
    for name in arms:
        u = rep["uniform"][name]
        r15, r620 = rep["rare"]["rare_1-5"], rep["rare"]["rare_6-20"]
        print(f"{name:16s} {u['micro_f1']:7.4f} {u['macro_f1']:7.4f} | {r15['recall'][name]:.3f} / {r620['recall'][name]:.3f} | "
              f"{r15['surfaced_weighted_whole_year'][name]:.0f} / {r620['surfaced_weighted_whole_year'][name]:.0f}")
    print(f"population (weighted) 1-5 {rep['rare']['rare_1-5']['population_whole_year']} / 6-20 {rep['rare']['rare_6-20']['population_whole_year']}")
    for k, v in rep["verdicts"].items():
        print(f"{k}: {v}")
    print("N1", n1, "\nN2", n2, "\nN3", n3)
    print(f"-> {out} (copied to results/)")


if __name__ == "__main__":
    main()
