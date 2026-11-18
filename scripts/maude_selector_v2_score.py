"""NOT RUN (2026-11-05). Written for a possible v2 = frozen v1 selector + example-grounded verification
(notes/selector-fix-and-baselines-2026-11-05.md); the 2024 development showed a small effect, so it was not
pre-registered or scored. Use only after pre-registering, preferably on a fresh year.

Score the revised selector once on 2025 (pre-registration would be notes/prereg-maude-selector-v2-<date>.md).

v2 = the frozen v1 selector of a pipeline + the example-grounded verification filter: when the
selector picks an LLM answer that names a code with <= 20 training examples and differs from the
encoder's answer, the answer stands only if the verifier prefers it over the encoder's code;
otherwise (including an unparseable reply) the encoder's answer is used. Nothing is refitted.

Pipelines: (answers, verifier) = (qwen38_27b, qwen38_27b), (gemma4_31b, gemma4_31b),
(opus55, qwen38_27b).
Scores, for v2, v1, the encoder and the best single arm chosen on 2024:
  balanced 2025 test   macro-F1 overall and per bin; item bootstrap (B=1000, seed 42)
  natural sample       uniform-stratum accuracy (micro-F1); item bootstrap
  failure-mode counts  visible rare modes and rare reports right (exact, as maude_downstream.py),
                       false rare placements and precision (stratified bootstrap), and population
                       macro-F1: every code weighted equally at the 2025 volume, with per-code
                       placements estimated from all strata and per-code hits from the census
                       (rare codes) or N_c times the code's recall on its balanced-test items

Usage: python maude_selector_v2_score.py   -> runs/ and results/maude2025nat/selector_v2_metrics.json
"""
import json
import shutil

import numpy as np

from common import ROOT, RUNS, SEED, rdir
from maude_data import POOL_2025, SPLIT_NATURAL
from maude_router import macro_f1
from maude_selector import BINS, git_head, load_coders, load_frozen, predict_frozen
from maude_verify import RARE_MAX

B = 1000
N_CANDIDATES = 758_343
PIPELINES = [("qwen38_27b", "qwen38_27b"), ("gemma4_31b", "gemma4_31b"), ("opus55", "qwen38_27b")]


def verdicts(which, answers, verifier):
    tag = f"{which}_{answers}" if answers == verifier else f"{which}_{answers}_by_{verifier}"
    path = RUNS / "maude_verify" / f"{tag}_overrides.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ok = r["type"] == "succeeded" and r.get("choice") in (1, 2) and r["choice"] == r["rare_position"]
        out[(int(r["key"]), r["rare_code"])] = ok
    return out


def apply_filter(D, pick, V):
    out, checked, missing = pick.copy(), 0, 0
    for i in range(len(pick)):
        p = pick[i]
        if p >= 0 and p != D["enc"][i] and D["counts"][p] <= RARE_MAX:
            checked += 1
            v = V.get((D["keys"][i], D["codes"][p]))
            missing += v is None
            if not v:
                out[i] = D["enc"][i]
    return out, checked, missing


def boot_items(n, rng):
    return [rng.integers(0, n, n) for _ in range(B)]


def ci(xs, nd=4):
    xs = sorted(xs)
    return [round(xs[round(0.025 * B) - 1], nd), round(xs[round(0.975 * B) - 1], nd)]


def main():
    nat = json.loads(SPLIT_NATURAL.read_text(encoding="utf-8"))
    rep = {"prereg": "notes/prereg-maude-selector-v2-2026-11-05.md", "git_head_at_score": git_head(), "pipelines": {}}
    for answers, verifier in PIPELINES:
        frozen, models = load_frozen(answers)
        Db, Dn = load_coders("2025", answers), load_coders("2025nat", answers)
        codes, counts = Dn["codes"], Dn["counts"]
        nc = len(codes)
        assert Db["codes"] == codes
        pb1, _ = predict_frozen(Db, frozen, models)
        pn1, _ = predict_frozen(Dn, frozen, models)
        pb2, chk_b, mis_b = apply_filter(Db, pb1, verdicts("2025", answers, verifier))
        pn2, chk_n, mis_n = apply_filter(Dn, pn1, verdicts("2025nat", answers, verifier))
        best = frozen["best_single_arm_2024"]
        arms_b = {"encoder": Db["enc"], "best_single": Db[best], "v1": pb1, "v2": pb2}
        arms_n = {"encoder": Dn["enc"], "best_single": Dn[best], "v1": pn1, "v2": pn2}
        R = {"answers": answers, "verifier": verifier, "best_single_arm_2024": best,
             "overrides_checked": {"balanced": chk_b, "natural": chk_n}, "verdicts_missing": {"balanced": mis_b, "natural": mis_n}}

        # balanced 2025: macro-F1 over the test labels, per bin
        labels = Db["labels"]
        bins = {b: np.array([c for c in labels if lo <= counts[c] <= hi]) for b, (lo, hi) in BINS.items()}
        rng = np.random.default_rng(SEED)
        draws = boot_items(len(Db["gold"]), rng)
        bal = {}
        for k, p in arms_b.items():
            bal[k] = {"all": round(macro_f1(Db["gold"], p, labels), 4),
                      **{b: round(macro_f1(Db["gold"], p, bins[b]), 4) for b in BINS}}
        for a, b_ in (("v2", "best_single"), ("v2", "v1"), ("v2", "encoder")):
            d = [macro_f1(Db["gold"], arms_b[a], labels, s) - macro_f1(Db["gold"], arms_b[b_], labels, s) for s in draws]
            bal[f"{a}_minus_{b_}"] = {"point": round(bal[a]["all"] - bal[b_]["all"], 4), "ci95": ci(d)}
        R["balanced_macro_f1"] = bal

        # natural sample, uniform stratum: accuracy
        pos = {k: i for i, k in enumerate(Dn["keys"])}
        U = np.array([pos[k] for k in nat["strata"]["uniform"]])
        gU = Dn["gold"][U]
        acc = {k: round(float((p[U] == gU).mean()), 4) for k, p in arms_n.items()}
        dU = boot_items(len(U), np.random.default_rng(SEED))
        R["natural_accuracy"] = acc
        for b_ in ("encoder", "v1"):
            d = sorted(float((pn2[U][s] == gU[s]).mean() - (arms_n[b_][U][s] == gU[s]).mean()) for s in dU)
            R["natural_accuracy"][f"v2_minus_{b_}"] = {"point": round(acc["v2"] - acc[b_], 4), "ci95": ci(d),
                                                       "p05": round(d[round(0.05 * B) - 1], 4)}

        # failure-mode counts at the 2025 volume (as maude_downstream.py)
        idx = {c: i for i, c in enumerate(codes)}
        N = np.zeros(nc)
        with open(POOL_2025, encoding="utf-8") as f:
            for line in f:
                i = idx.get(json.loads(line)["code"])
                if i is not None and counts[i] > 0:
                    N[i] += 1
        A = np.array([pos[k] for k in nat["strata"]["rare_1-5"]])
        Bs = np.array([pos[k] for k in nat["strata"]["rare_6-20"]])
        gn = Dn["gold"]
        C = U[counts[gn[U]] >= 21]
        wB = nat["weights"]["rare_6-20"]
        wC = (N_CANDIDATES - len(A) - nat["counts"]["rare_6-20_population"]) / len(C)
        rare = np.where((counts >= 1) & (counts <= 5))[0]
        present = np.where(N > 0)[0]
        gb = Db["gold"]

        rare_b = np.isin(gb, rare)

        def counts_for(p_b, p_n, iB, iC):
            """assigned: estimated 2025 reports placed under each code (all strata, weighted).
            tp: rare codes exact (balanced items + the census); every other code N_c times its
            recall on its own balanced-test items (a random sample of that code's 2025 reports)."""
            assigned = np.zeros(nc)
            for pred, ix, w in ((p_b, np.arange(len(gb)), 1.0), (p_n, A, 1.0), (p_n, iB, wB), (p_n, iC, wC)):
                pr = pred[ix]
                np.add.at(assigned, pr[pr >= 0], w)
            tp = np.zeros(nc)
            hit_b = p_b == gb
            n_b = np.bincount(gb, minlength=nc).astype(float)
            r_b = np.bincount(gb[hit_b], minlength=nc) / np.maximum(n_b, 1)
            tp[:] = N * r_b
            tp[rare] = 0.0
            np.add.at(tp, gb[rare_b & hit_b], 1.0)
            hitA = p_n[A] == gn[A]
            np.add.at(tp, gn[A][hitA], 1.0)
            return assigned, tp

        def summary(assigned, tp):
            fp = np.maximum(assigned - tp, 0.0)
            fn = np.maximum(N - tp, 0.0)
            f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9), 0.0)
            tp_r, fp_r = float(tp[rare].sum()), float(fp[rare].sum())
            return {"pop_macro_f1": float(f1[present].mean()), "rare_right": tp_r, "rare_false": fp_r,
                    "rare_precision": tp_r / (tp_r + fp_r) if tp_r + fp_r > 0 else 0.0,
                    "visible_rare": int((tp[rare] > 0).sum())}

        rng = np.random.default_rng(SEED)
        cdraws = [(Bs[rng.integers(0, len(Bs), len(Bs))], C[rng.integers(0, len(C), len(C))]) for _ in range(B)]
        pts = {k: summary(*counts_for(arms_b[k], arms_n[k], Bs, C)) for k in arms_b}
        boots = {k: [summary(*counts_for(arms_b[k], arms_n[k], iB, iC)) for iB, iC in cdraws] for k in ("encoder", "v1", "v2")}
        cnt = {}
        for k, s in pts.items():
            cnt[k] = {"visible_rare_of_46": s["visible_rare"], "rare_right_of_597": round(s["rare_right"]),
                      "rare_false": round(s["rare_false"], 1), "rare_precision": round(s["rare_precision"], 4),
                      "pop_macro_f1": round(s["pop_macro_f1"], 4)}
            if k in boots:
                cnt[k]["rare_false_ci95"] = ci([b["rare_false"] for b in boots[k]], 1)
                cnt[k]["rare_precision_ci95"] = ci([b["rare_precision"] for b in boots[k]])
                cnt[k]["pop_macro_f1_ci95"] = ci([b["pop_macro_f1"] for b in boots[k]])
        for a, b_ in (("v1", "v2"),):
            d = [x["rare_false"] - y["rare_false"] for x, y in zip(boots[a], boots[b_])]
            cnt["v1_minus_v2_rare_false"] = {"point": round(pts[a]["rare_false"] - pts[b_]["rare_false"], 1), "ci95": ci(d, 1)}
            d = [y["rare_precision"] - x["rare_precision"] for x, y in zip(boots[a], boots[b_])]
            cnt["v2_minus_v1_rare_precision"] = {"point": round(pts[b_]["rare_precision"] - pts[a]["rare_precision"], 4),
                                                 "ci95": ci(d)}
        for b_ in ("encoder", "v1"):
            d = [x["pop_macro_f1"] - y["pop_macro_f1"] for x, y in zip(boots["v2"], boots[b_])]
            cnt[f"v2_minus_{b_}_pop_macro_f1"] = {"point": round(pts["v2"]["pop_macro_f1"] - pts[b_]["pop_macro_f1"], 4),
                                                  "ci95": ci(d)}
        R["failure_mode_counts"] = cnt
        rep["pipelines"][f"{answers}_by_{verifier}"] = R
        print(f"\n{answers} answers, {verifier} verifier: overrides checked {chk_b} balanced / {chk_n} natural; "
              f"missing verdicts {mis_b}/{mis_n}")
        print("  balanced macro-F1:", {k: v["all"] for k, v in bal.items() if isinstance(v, dict) and "all" in v},
              "| v2-best", bal["v2_minus_best_single"])
        print("  natural accuracy:", {k: v for k, v in acc.items()}, "| v2-enc", R["natural_accuracy"]["v2_minus_encoder"])
        for k in ("encoder", "v1", "v2"):
            print(f"  {k:8s}", cnt[k])
    out = rdir("maude2025nat") / "selector_v2_metrics.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    res = ROOT / "results" / "maude2025nat"
    res.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, res / "selector_v2_metrics.json")


if __name__ == "__main__":
    main()
