"""Consequence for failure-mode counts, MAUDE 2025 (notes/prereg-maude-downstream-2026-11-05.md).

A reliability database counts one year's reports per failure mode. For each frozen coder this
estimates, for the 2025 population of 762,360 single-code reports under the known codes:
  D1  rare modes (1-5 training examples) with at least one report under the right code (exact:
      all 597 rare-mode reports have predictions, 273 in the balanced test + 324 census)
  D2  reports placed under a rare mode whose true mode differs (estimated, stratified bootstrap)
  D3  precision of the rare-mode placements, D4 per-rare-mode count ratio, D5 Spearman between
      assigned and true per-mode counts.
Weights: balanced test 1; natural census of 1-5 reports 1; 6-20 sample 2812/1500; uniform reports
whose gold code has 21+ examples, post-stratified, 755,207 / n_C. Nothing is refitted.

Usage: python maude_downstream.py [--year 2026]  -> runs/ and results/maude<year>nat/downstream_metrics.json (once)
With --year 2026 (notes/prereg-redesign-2026-11-06.md, F): the same definitions on the 2026 population
(balanced test + the 2026 natural sample); population sizes come from the split files.
"""
import json
import shutil
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from common import ROOT, SEED, rdir
from maude_data import POOL_2025, POOL_2026, SPLIT_NATURAL, SPLIT_NATURAL_2026
from maude_selector import (frozen_paths, gate, git_head, llm_rare, load_coders, load_frozen, predict_frozen,
                            rule3, sha256)

B = 1000
BINS = {"1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
N_CANDIDATES = 758_343


def arms_for(D, frozen, models, full=True):
    sel, _ = predict_frozen(D, frozen, models)
    if not full:
        return {"selector": sel}
    g = gate(D, frozen["S_cost"]["k"], frozen["S_cost"]["p"])
    return {"enc": D["enc"], "zs": D["zs"], "rag": D["rag"], "selector": sel,
            "S_cost": np.where(g, sel, D["enc"]), "llm_rare": llm_rare(D, frozen["llm_rare_k_star"]["k"]),
            "S1": rule3(D, frozen["S1"]["t"], frozen["S1"]["k"])}


def bin_of(counts):
    out = np.full(len(counts), "", dtype=object)
    for name, (lo, hi) in BINS.items():
        out[(counts >= lo) & (counts <= hi)] = name
    return out


def add(v, pred, w):
    p = pred[pred >= 0]
    np.add.at(v, p, w)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2025", choices=["2025", "2026"])
    year = ap.parse_args().year
    pool = POOL_2025 if year == "2025" else POOL_2026
    nat = json.loads((SPLIT_NATURAL if year == "2025" else SPLIT_NATURAL_2026).read_text(encoding="utf-8"))
    n_candidates = N_CANDIDATES if year == "2025" else nat["counts"]["candidates"]
    frozen, models = load_frozen("opus55")
    Db, Dn = load_coders(year, "opus55"), load_coders(f"{year}nat", "opus55")
    assert Db["codes"] == Dn["codes"], "code order differs between the balanced and natural sets"
    codes, counts = Dn["codes"], Dn["counts"]
    nc = len(codes)
    cbin = bin_of(counts)

    # true 2025 counts per code (population: known codes with training examples)
    idx = {c: i for i, c in enumerate(codes)}
    N = np.zeros(nc)
    with open(pool, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)["code"]
            i = idx.get(c)
            if i is not None and counts[i] > 0:
                N[i] += 1
    assert int(N.sum()) == nat["counts"]["known_code_reports"], int(N.sum())

    pos = {k: i for i, k in enumerate(Dn["keys"])}
    A = np.array([pos[k] for k in nat["strata"]["rare_1-5"]])
    Bs = np.array([pos[k] for k in nat["strata"]["rare_6-20"]])
    U = np.array([pos[k] for k in nat["strata"]["uniform"]])
    gb_n = cbin[Dn["gold"]]
    C = U[np.isin(gb_n[U], ["21-89", "90+"])]
    wB = nat["weights"]["rare_6-20"]
    N_C = n_candidates - len(A) - nat["counts"]["rare_6-20_population"]
    wC = N_C / len(C)
    n_rare_total = int(N[cbin == "1-5"].sum())
    n_rare_codes = int(((cbin == "1-5") & (N > 0)).sum())
    assert len(A) == nat["counts"]["rare_1-5"]
    assert n_rare_total == len(A) + int(np.isin(cbin[Db["gold"]], ["1-5"]).sum()), "a rare-mode report lacks a prediction"

    # arms: Opus (all), Qwen and Gemma selectors (descriptive)
    arms_b, arms_n = arms_for(Db, frozen, models), arms_for(Dn, frozen, models)
    for llm in ("qwen38_27b", "gemma4_31b") if year == "2025" else ():  # 2026: Opus only (prereg section 5)
        fz, md = load_frozen(llm)
        arms_b[f"selector_{llm}"] = arms_for(load_coders(year, llm), fz, md, full=False)["selector"]
        arms_n[f"selector_{llm}"] = arms_for(load_coders(f"{year}nat", llm), fz, md, full=False)["selector"]

    rare = np.where(cbin == "1-5")[0]
    mid = np.where(cbin == "6-20")[0]
    # per-mode ratios and rank correlations need a true count: modes with reports in the year
    # (every rare mode had reports in 2025; 43 of 46 have in 2026). False placements still count
    # every rare code, present or not.
    rare_p, mid_p = rare[N[rare] > 0], mid[N[mid] > 0]
    present = np.where(N > 0)[0]
    gold_b, gold_n = Db["gold"], Dn["gold"]
    rare_b = np.isin(gold_b, rare)
    rare_A = A  # every census item has a 1-5 gold code

    def assigned(pb, pn, iB, iC):
        v = np.zeros(nc)
        add(v, pb, 1.0)
        add(v, pn[A], 1.0)
        add(v, pn[iB], wB)
        add(v, pn[iC], wC)
        return v

    def stats(name, iB, iC):
        pb, pn = arms_b[name], arms_n[name]
        v = assigned(pb, pn, iB, iC)
        tp_items = np.concatenate([gold_b[rare_b][pb[rare_b] == gold_b[rare_b]], gold_n[rare_A][pn[rare_A] == gold_n[rare_A]]])
        tp = len(tp_items)
        fp = float(v[rare].sum() - tp)
        return v, tp, fp, len(set(tp_items.tolist()))

    rng = np.random.default_rng(SEED)
    draws = [(Bs[rng.integers(0, len(Bs), len(Bs))], C[rng.integers(0, len(C), len(C))]) for _ in range(B)]

    def ci(xs, nd=1):
        xs = sorted(xs)
        return [round(xs[round(0.025 * B) - 1], nd), round(xs[round(0.975 * B) - 1], nd)]

    rep = {"prereg": "notes/prereg-maude-downstream-2026-11-05.md", "git_head_at_score": git_head(),
           "frozen": {"file": frozen_paths("opus55")[0].name, "sha256": sha256(frozen_paths("opus55")[0])},
           "population": int(N.sum()), "true_reports_by_bin": {b: int(N[cbin == b].sum()) for b in BINS},
           "codes_present_by_bin": {b: int(((cbin == b) & (N > 0)).sum()) for b in BINS},
           "weights": {"balanced": 1.0, "rare_1-5_census": 1.0, "rare_6-20": round(wB, 4), "C": round(wC, 3)},
           "n_C": int(len(C)), "N_C": int(N_C), "arms": {}}
    boot = {}
    for name in arms_b:
        v, tp, fp, vis = stats(name, Bs, C)
        fps, precs = [], []
        for iB, iC in draws:
            _, _, f_, _ = stats(name, iB, iC)
            fps.append(f_)
            precs.append(tp / (tp + f_) if tp + f_ > 0 else 0.0)
        boot[name] = np.array(fps)
        ratio = v[rare_p] / N[rare_p]
        # 6-20, descriptive: TP from the balanced test (exact) and the 6-20 sample (weighted)
        mid_b = np.isin(gold_b, mid)
        mid_s = Bs
        pb, pn = arms_b[name], arms_n[name]
        tp_mid = float((pb[mid_b] == gold_b[mid_b]).sum() + wB * (pn[mid_s] == gold_n[mid_s]).sum())
        vis_mid = len(set(gold_b[mid_b][pb[mid_b] == gold_b[mid_b]].tolist()) | set(gold_n[mid_s][pn[mid_s] == gold_n[mid_s]].tolist()))
        rep["arms"][name] = {
            "D1_visible_rare_modes": vis, "rare_modes_present": n_rare_codes, "rare_reports_right": tp,
            "rare_reports_total": n_rare_total,
            "D2_false_rare_placements": round(fp, 1), "D2_ci95": ci(fps),
            "D3_precision_rare": round(tp / (tp + fp), 4) if tp + fp > 0 else 0.0, "D3_ci95": ci(precs, 4),
            "rare_placements_total": round(tp + fp, 1),
            "D4_ratio_median": round(float(np.median(ratio)), 3),
            "D4_share_within_2x": round(float(((ratio >= 0.5) & (ratio <= 2)).mean()), 3),
            "D4_share_zero": round(float((v[rare_p] == 0).mean()), 3),
            "D5_spearman_all": round(float(spearmanr(v[present], N[present]).statistic), 4),
            "D5_spearman_1-5": round(float(spearmanr(v[rare_p], N[rare_p]).statistic), 4),
            "D5_spearman_6-20": round(float(spearmanr(v[mid_p], N[mid_p]).statistic), 4),
            "mid_6-20": {"visible_modes_lower_bound": vis_mid, "reports_right_est": round(tp_mid, 1),
                         "reports_total": int(N[mid].sum()),
                         "false_placements_est": round(float(v[mid].sum() - tp_mid), 1)},
        }
    d = boot["selector"] - boot["enc"]
    rep["P2_selector_minus_encoder_false_rare"] = {"point": rep["arms"]["selector"]["D2_false_rare_placements"]
                                                   - rep["arms"]["enc"]["D2_false_rare_placements"],
                                                   "ci95": ci(d.tolist()), "excludes_zero": bool(np.sort(d)[24] > 0 or np.sort(d)[974] < 0)}
    out = rdir(f"maude{year}nat") / "downstream_metrics.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    res = ROOT / "results" / f"maude{year}nat"
    res.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, res / "downstream_metrics.json")
    print(f"population {rep['population']}; n_C {rep['n_C']}, wC {wC:.1f}, wB {wB:.4f}")
    for name, r in rep["arms"].items():
        print(f"{name:22s} visible {r['D1_visible_rare_modes']:2d}/{n_rare_codes}  right {r['rare_reports_right']:3d}/{n_rare_total}  "
              f"false {r['D2_false_rare_placements']:8.1f} {r['D2_ci95']}  prec {r['D3_precision_rare']:.3f} {r['D3_ci95']}  "
              f"ratio med {r['D4_ratio_median']:.2f} within2x {r['D4_share_within_2x']:.2f} zero {r['D4_share_zero']:.2f}  "
              f"rho all {r['D5_spearman_all']:.3f} 1-5 {r['D5_spearman_1-5']:.3f} 6-20 {r['D5_spearman_6-20']:.3f}")
    print("P2 selector - encoder false rare placements:", rep["P2_selector_minus_encoder_false_rare"])


if __name__ == "__main__":
    main()
