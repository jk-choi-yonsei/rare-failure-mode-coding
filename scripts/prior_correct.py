"""Prior correction of the frozen selector for natural proportions (option A, 2026-11-09).

The selector's per-coder models were fitted on the training-year balanced practice set (up to ten
reports per code), so an answer naming a rare code looks far more likely to be right than it is in
a year of reports. The standard correction for a change of class priors (Saerens, Latinne and
Decaestecker, 2002) rescales each answer's odds of being right by the ratio of prior odds of its
code: natural prior (the code's share of the training reports) against practice-set prior (its
share of the practice set), both with add-half smoothing. The selector then keeps the answer with
the largest corrected probability. Nothing is fitted or tuned, so 2025 and 2026 are both tests.

Reported for each year, with the natural sample weighted to the whole year as in maude_downstream.py:
right and false rare placements, precision, rare modes visible, and the share of false placements
whose code is a sibling, parent or child of the gold code; accuracy and macro-F1 on the uniform draw.
Usage: python prior_correct.py   -> results/prior_correct/metrics.json
"""
import json
import os

import numpy as np

os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")
import maude_selector as ms  # noqa: E402
from common import ROOT  # noqa: E402
from maude_data import class_info, load_split  # noqa: E402
from maude_router import macro_f1  # noqa: E402
from rare_check import RARE_MAX, natural_weights, selector_outputs, year_counts  # noqa: E402

INFO = class_info(load_split())


def priors(llm):
    D = ms.load_coders("2024", llm)
    K = len(D["codes"])
    m = np.bincount(D["gold"], minlength=K).astype(float)
    n = np.asarray(D["counts"], dtype=float)
    return (m + 0.5) / (m.sum() + 0.5 * K), (n + 0.5) / (n.sum() + 0.5 * K)


def corrected(P, arms, pi_bal, pi_nat):
    c = np.maximum(arms, 0)
    odds = P / np.clip(1.0 - P, 1e-9, None)
    ratio = (pi_nat[c] / (1 - pi_nat[c])) / (pi_bal[c] / (1 - pi_bal[c]))
    Pc = odds * ratio / (1.0 + odds * ratio)
    Pc = np.where(arms >= 0, Pc, -1.0)
    j = Pc.argmax(axis=1)
    return arms[np.arange(len(j)), j]


def near(D, a, g):
    pa, pg = INFO[D["codes"][a]]["path"], INFO[D["codes"][g]]["path"]
    if pa.startswith(pg + " >") or pg.startswith(pa + " >"):
        return True
    return " > " in pa and " > " in pg and pa.rsplit(" > ", 1)[0] == pg.rsplit(" > ", 1)[0]


def rare_detail(Sb, Sn, wn, fb, fn):
    """False rare placements that sit next to the gold code in the code tree (weighted)."""
    tot = close = 0.0
    for S, f, w in ((Sb, fb, np.ones(len(fb))), (Sn, fn, wn)):
        D = S["D"]
        c = D["counts"]
        for i in np.where((f >= 0) & (w > 0))[0]:
            if 1 <= c[f[i]] <= RARE_MAX and f[i] != D["gold"][i]:
                tot += w[i]
                close += w[i] * near(D, f[i], D["gold"][i])
    return close / tot if tot else 0.0


def evaluate(year, llm):
    frozen, models = ms.load_frozen(llm)
    pi_bal, pi_nat = priors(llm)
    out = {}
    S = {}
    for which in (year, year + "nat"):
        D = ms.load_coders(which, llm)
        P, Pm, arms, pick, j = selector_outputs(D, frozen, models)
        S[which] = {"D": D, "sel": pick, "cor": corrected(P, arms, pi_bal, pi_nat)}
    Sb, Sn = S[year], S[year + "nat"]
    wn, U = natural_weights(Sn, year)
    D = Sn["D"]
    labels_u = np.array(sorted(set(D["gold"][U].tolist())))
    for name, (b, n) in {"encoder": (Sb["D"]["enc"], D["enc"]), "selector": (Sb["sel"], Sn["sel"]),
                         "selector_prior_corrected": (Sb["cor"], Sn["cor"])}.items():
        tp, fp, vis = year_counts(Sb, Sn, wn, b, n)
        out[name] = {"right_rare": round(tp, 1), "false_rare": round(fp, 1),
                     "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                     "false_next_to_gold": round(rare_detail(Sb, Sn, wn, b, n), 3), "modes_visible": vis,
                     "accuracy_uniform": round(float((n[U] == D["gold"][U]).mean()), 4),
                     "macro_f1_uniform": round(macro_f1(D["gold"], n, labels_u, U), 4),
                     "macro_f1_balanced": round(macro_f1(Sb["D"]["gold"], b, Sb["D"]["labels"]), 4),
                     "rare_answer_share_uniform": round(float(np.mean([1 <= D["counts"][x] <= RARE_MAX
                                                                        for x in n[U] if x >= 0])), 4)}
    return out


def main():
    res = {"note": "notes/rare-check-2026-11-09.md, option A", "git_head": ms.git_head(),
           "method": "Saerens et al. 2002 prior correction of each answer's odds; natural prior = training share, "
                     "practice prior = share of the training-year balanced practice set; add-half smoothing"}
    runs = [("2025", "opus55"), ("2026", "opus55"), ("2025", "qwen38_27b"), ("2025", "gemma4_31b")]
    for year, llm in runs:
        r = evaluate(year, llm)
        res[f"{year}/{llm}"] = r
        print(f"\n{year} {llm}")
        for k, v in r.items():
            print(f"  {k:26s} {v}")
    out = ROOT / "results" / "prior_correct"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
