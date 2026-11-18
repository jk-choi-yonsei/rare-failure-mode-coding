"""Where the rare-code answers of each method fall in 2026, and how often a rare-code answer is right
in the two sets the selector is fitted on (notes/natural-selector-2026-11-09.md, 2026-11-11 addendum).

Part 1 splits the whole-year counts of natural_selector.py by sample stratum. The balanced test and the
census of rare (1-5) reports count once each; the 6-20 stratum and the uniform draw carry their sampling
weights. It shows how many sampled ordinary reports stand behind each whole-year count of false rare
placements. Part 2 counts, for each coder, its rare-code answers and how many are right in the balanced
practice set and in the 10,000 training-year validation reports (natural proportions).
Usage: python natural_breakdown.py   -> results/natural_selector/breakdown_2026.json
"""
import json
import os

import joblib
import numpy as np

os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")
import maude_selector as ms  # noqa: E402
import natural_selector as ns  # noqa: E402
from common import ROOT  # noqa: E402
from prior_correct import corrected, priors  # noqa: E402
from rare_check import RARE_MAX, natural_weights, selector_outputs  # noqa: E402


def rare(D, f):
    c = D["counts"]
    g = np.maximum(f, 0)
    return (f >= 0) & (c[g] >= 1) & (c[g] <= RARE_MAX)


def by_stratum(Sb, Sn, wn, fb, fn):
    D = Sb["D"]
    p = rare(D, fb)
    r = p & (fb == D["gold"])
    out = {"balanced_test": {"reports": int(len(fb)), "weight": 1.0, "right": int(r.sum()),
                             "false": int((p & ~r).sum())}}
    D = Sn["D"]
    p = rare(D, fn)
    r = p & (fn == D["gold"])
    top = wn.max()
    tp, fp = float(out["balanced_test"]["right"]), float(out["balanced_test"]["false"])
    for name, m in (("rare_1-5_census", wn == 1.0), ("rare_6-20_stratum", (wn > 1.0) & (wn < top)),
                    ("uniform_draw", wn == top)):
        w = float(wn[m][0])
        out[name] = {"reports": int(m.sum()), "weight": round(w, 2), "right": int((r & m).sum()),
                     "false": int(((p & ~r) & m).sum())}
        tp += w * out[name]["right"]
        fp += w * out[name]["false"]
    out["year"] = {"right": round(tp, 1), "false": round(fp, 1)}
    return out


def fitting_set(D):
    g = D["counts"][D["gold"]]
    out = {"reports": int(len(D["gold"])), "rare_reports": int(((g >= 1) & (g <= RARE_MAX)).sum())}
    for a in ms.ARMS:
        f = np.asarray(D[a])
        p = rare(D, f)
        out[a] = {"rare_answers": int(p.sum()), "right": int((p & (f == D["gold"])).sum())}
    return out


def main():
    res = {"note": "notes/natural-selector-2026-11-09.md (2026-11-11 addendum)", "git_head": ms.git_head()}
    for llm in ("qwen38_27b", "gemma4_31b", "opus55"):
        frozen, fmodels = ms.load_frozen(llm)
        pi_bal, pi_nat = priors(llm)
        fb = ns.paths(llm)[1]
        models = joblib.load(fb)["models"] if fb.exists() else None
        S = {}
        for which in ("2026", "2026nat"):
            D = ms.load_coders(which, llm)
            P, Pm, arms, pick, j = selector_outputs(D, frozen, fmodels)
            S[which] = {"D": D, "sel": pick, "cor": corrected(P, arms, pi_bal, pi_nat)}
            if models is not None:
                S[which]["nat"] = ns.natural_pick(D, models)
        Sb, Sn = S["2026"], S["2026nat"]
        wn, U = natural_weights(Sn, "2026")
        D = Sn["D"]
        rows = {"encoder": (Sb["D"]["enc"], D["enc"]), "selector": (Sb["sel"], Sn["sel"]),
                "selector_prior_corrected": (Sb["cor"], Sn["cor"])}
        if models is not None:
            rows["natural_selector"] = (Sb["nat"], Sn["nat"])
        cb, cn = Sb["D"]["counts"][Sb["D"]["gold"]], D["counts"][D["gold"]]
        out = {"year_reports": round(float(len(cb) + wn.sum())),
               "year_rare_reports": round(float(((cb >= 1) & (cb <= RARE_MAX)).sum()
                                                + (wn * ((cn >= 1) & (cn <= RARE_MAX))).sum())),
               "2026_by_stratum": {k: by_stratum(Sb, Sn, wn, b, n) for k, (b, n) in rows.items()}}
        Pr = ms.load_coders("2024", llm)
        out["practice_set"] = fitting_set(Pr)
        try:
            out["validation_reports"] = fitting_set(ms.load_coders("2024val", llm))
        except (SystemExit, FileNotFoundError) as e:  # no outputs of this LLM on the validation reports
            out["validation_reports"] = {"not_run": str(e)}
        if "training" not in res:
            n = np.asarray(Pr["counts"], dtype=float)
            r = (n >= 1) & (n <= RARE_MAX)
            res["training"] = {"codes": int(len(n)), "rare_codes": int(r.sum()), "reports": int(n.sum()),
                               "rare_reports": int(n[r].sum())}
        res[llm] = out
        print(llm, "year", out["year_reports"], "rare", out["year_rare_reports"])
        for k, v in out["2026_by_stratum"].items():
            print(f"  {k:26s}", {s: (t["right"], t["false"]) for s, t in v.items()})
        print("  practice  ", out["practice_set"])
        print("  validation", out["validation_reports"])
    print("training", res["training"])
    out_dir = ROOT / "results" / "natural_selector"
    (out_dir / "breakdown_2026.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
