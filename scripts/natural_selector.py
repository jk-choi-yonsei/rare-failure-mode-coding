"""Natural-proportion selector (option B, notes/natural-selector-2026-11-09.md).

The balanced practice set (training-year balanced test, up to ten reports per code) teaches the
selector that a rare-code answer is usually right. This version learns the same per-coder logistic
models (the 14 inputs of maude_selector.py) on the practice set together with the 10,000
training-year validation reports, which come in natural proportions, and weights the practice
reports by natural prior / practice prior of their gold code (importance weighting to natural
proportions, as in black-box shift estimation). The validation reports supply what the practice set
lacks: many ordinary reports, including those where an LLM proposes a rare code wrongly. Everything
is fitted on the training year; 2025 and 2026 are tests.

Usage: python natural_selector.py fit <llm>      -> splits/natural_selector_v1_<llm>.joblib/.json
       python natural_selector.py score <llm>    -> results/natural_selector/metrics_<llm>.json
"""
import argparse
import hashlib
import json
import os

import joblib
import numpy as np

os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")
import maude_selector as ms  # noqa: E402
from common import ROOT, SPLITS  # noqa: E402
from maude_router import macro_f1  # noqa: E402
from prior_correct import corrected, priors, rare_detail  # noqa: E402
from rare_check import RARE_MAX, natural_weights, selector_outputs, year_counts  # noqa: E402


def paths(llm):
    return SPLITS / f"natural_selector_v1_{llm}.json", SPLITS / f"natural_selector_v1_{llm}.joblib"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def fit(llm):
    P = ms.load_coders("2024", llm)
    V = ms.load_coders("2024val", llm)
    pi_bal, pi_nat = priors(llm)
    X = np.vstack([ms.features(P), ms.features(V)])
    Y = np.vstack([ms.targets(P), ms.targets(V)])
    w = np.concatenate([pi_nat[P["gold"]] / pi_bal[P["gold"]], np.ones(len(V["gold"]))])
    models, constant = [], []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        m = make()
        m.fit(X, y, logisticregression__sample_weight=w)
        models.append(m)
    singles = {a: round(ms.f1_of(V, V[a]), 4) for a in ms.ARMS}
    fj, fb = paths(llm)
    joblib.dump({"models": models, "features": ms.FEATURES}, fb)
    meta = {"note": "notes/natural-selector-2026-11-09.md", "llm": llm, "encoder": ms.SELECTOR_ENC,
            "rows": {"practice": int(len(P["gold"])), "validation": int(len(V["gold"]))},
            "weights": "practice: natural prior / practice prior of the gold code; validation: 1",
            "validation_single_coder_accuracy": {a: round(float((V[a] == V["gold"]).mean()), 4) for a in ms.ARMS},
            "validation_single_coder_macro_f1": singles,
            "coef": {a: dict(zip(ms.FEATURES, np.round(m.named_steps["logisticregression"].coef_[0], 4).tolist()))
                     for a, m in zip(ms.ARMS, models)},
            "joblib_sha256": sha(fb), "git_head": ms.git_head()}
    fj.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("rows", "validation_single_coder_accuracy")}))
    print(f"frozen -> {fj.name}")


def natural_pick(D, models):
    X = ms.features(D)
    P = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    arms = ms.arms_matrix(D)
    Pm = np.where(arms >= 0, P, -1.0)
    j = Pm.argmax(axis=1)
    return arms[np.arange(len(j)), j]


def score(llm):
    fj, fb = paths(llm)
    meta = json.loads(fj.read_text(encoding="utf-8"))
    if sha(fb) != meta["joblib_sha256"]:
        raise SystemExit(f"{fb.name} does not match its manifest")
    models = joblib.load(fb)["models"]
    frozen, fmodels = ms.load_frozen(llm)
    pi_bal, pi_nat = priors(llm)
    res = {"note": meta["note"], "llm": llm, "git_head": ms.git_head()}
    for year in ("2025", "2026"):
        S = {}
        try:
            for which in (year, year + "nat"):
                D = ms.load_coders(which, llm)
                P, Pm, arms, pick, j = selector_outputs(D, frozen, fmodels)
                S[which] = {"D": D, "sel": pick, "cor": corrected(P, arms, pi_bal, pi_nat), "nat": natural_pick(D, models)}
        except SystemExit as e:  # an LLM's outputs on this year's natural sample do not exist yet
            res[year] = {"not_run": str(e)}
            print(year, "not run:", e)
            continue
        Sb, Sn = S[year], S[year + "nat"]
        wn, U = natural_weights(Sn, year)
        D = Sn["D"]
        labels_u = np.array(sorted(set(D["gold"][U].tolist())))
        out = {}
        for name, key in (("encoder", None), ("selector", "sel"), ("selector_prior_corrected", "cor"),
                          ("natural_selector", "nat")):
            b = Sb["D"]["enc"] if key is None else Sb[key]
            n = D["enc"] if key is None else Sn[key]
            tp, fp, vis = year_counts(Sb, Sn, wn, b, n)
            out[name] = {"right_rare": round(tp, 1), "false_rare": round(fp, 1),
                         "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                         "false_next_to_gold": round(float(rare_detail(Sb, Sn, wn, b, n)), 3), "modes_visible": vis,
                         "accuracy_uniform": round(float((n[U] == D["gold"][U]).mean()), 4),
                         "macro_f1_uniform": round(macro_f1(D["gold"], n, labels_u, U), 4),
                         "macro_f1_balanced": round(macro_f1(Sb["D"]["gold"], b, Sb["D"]["labels"]), 4)}
            print(year, llm, f"{name:26s}", out[name])
        res[year] = out
    out_dir = ROOT / "results" / "natural_selector"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{llm}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "score"])
    ap.add_argument("llm", choices=["qwen38_27b", "gemma4_31b", "opus55"])
    a = ap.parse_args()
    import natural_selector as _m  # pickled models reference this module
    {"fit": _m.fit, "score": _m.score}[a.cmd](a.llm)
