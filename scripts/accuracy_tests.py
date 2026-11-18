"""McNemar's exact test for the accuracy differences of the paper (2026-11-18: the user asked for significance marks in
the accuracy table in place of bootstrap intervals).

On the random sample of each test year (the uniform draw of the natural sample, 4,000 reports), every method of the
accuracy table is compared with the classifier and with Ours (rule + selection): a two-sided exact binomial test on the
reports that exactly one of the two methods codes correctly (McNemar's test). The methods and their fitting are those
of paper_v4_numbers.py (selection model, cascade) and selection_ablation.py (selection model without LLM inputs,
logit-adjusted classifier at tau = 0.5), each fitted on the other test year.

Usage: [MAINT_CORPUS=sdr] python accuracy_tests.py  -> results/paper_v4/<corpus>_accuracy_tests.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

import paper_v4_numbers as p4  # noqa: E402

LLMS = ("qwen38_27b", "gemma4_31b")


def mcnemar(a, b):
    """a, b: per-report correctness of two methods on the same reports."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = float(binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    return {"diff_points": round(100 * float(a.mean() - b.mean()), 2), "only_first": only_a, "only_second": only_b,
            "p": round(p, 6)}


def main():
    import candidate_ranker as cr  # sets two threads and idle priority (selector_vs_rules)
    import maude_selector as ms
    import redesign_score as rs
    import selection_ablation as sa
    import selector_vs_rules as sv
    from common import CORPUS, ROOT
    from rare_check import natural_weights
    t0 = time.time()
    ref = json.loads((ROOT / "results" / "paper_v4" / f"{CORPUS}_numbers.json").read_text(encoding="utf-8"))
    rep = {"note": "McNemar exact tests on the random sample of each test year", "corpus": CORPUS, "llms": {}}
    for llm in LLMS:
        frozen, models = rs.check_frozen(llm)
        D = {w: cr.load(w, llm) for w in ("2025nat", "2026nat")}
        W = {y: natural_weights({"D": D[y + "nat"]}, y) for y in ("2025", "2026")}
        L = {}
        for tr, te in (("2025", "2026"), ("2026", "2025")):
            Dt, rows = D[tr + "nat"], W[tr][1]
            R = cr.fit(Dt, rows, "wide_gbm_table")
            casc = cr.fit_cascade(Dt, rows)
            R_nollm = sa.fit_nollm(Dt, rows)
            wn, U = W[te]
            Dn = D[te + "nat"]
            P = p4.predictions(cr, sv, rs, ms, Dn, llm, frozen, models, R, casc, wn)
            fire, best, rare, enc_nonrare = p4.gate_parts(Dn)
            pn, pn_nr = sa.outputs_nollm(R_nollm, Dn, rare, enc_nonrare)
            P["ours_nollm"] = p4.gated(pn, pn_nr, fire, best, rare)
            P["logit_adjusted"] = sa.logit_adjusted(Dn, 0.5)
            g = Dn["gold"][U]
            ok = {k: P[k][U] == g for k in P}
            S = {}
            for k in ok:
                S[k] = {"accuracy": round(100 * float(ok[k].mean()), 2), "vs_classifier": mcnemar(ok[k], ok["enc"]),
                        "ours_full_vs": mcnemar(ok["ours_full"], ok[k])}
                if k in ref["llms"][llm][te]["methods"]:  # the accuracy must equal the paper's (rounded to 0.01%)
                    assert abs(100 * float(ok[k].mean()) - 100 * ref["llms"][llm][te]["methods"][k]["accuracy"]) <= 0.0051, \
                        (llm, te, k, S[k]["accuracy"], ref["llms"][llm][te]["methods"][k]["accuracy"])
            L[te] = S
            print(f"[{CORPUS} {llm} {te}] " + ", ".join(f"{k} {S[k]['accuracy']:.1f} (p {S[k]['vs_classifier']['p']:.3g})"
                                                       for k in S) + f" ({time.time() - t0:.0f}s)", flush=True)
        rep["llms"][llm] = L
        del D
    out = ROOT / "results" / "paper_v4" / f"{CORPUS}_accuracy_tests.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
