"""Where the selection model's accuracy comes from, and the encoder corrected for label frequency (review of manuscript v4).

Per corpus and LLM, each test year scored with models fitted on the other year's uniform draw (2025 -> 2026, 2026 ->
2025), exactly as paper_v4_numbers.py does for "Ours (gate + selection)". Variants of the selection model:
  full          wide candidates, coder tables, gradient-boosted trees (the paper's model; a check that numbers match)
  no_table      the same without the per-coder tables
  no_llm        candidates and inputs from the encoder and the ten nearest training reports only (no LLM answer, no
                LLM-based input, the encoder's table only); the agreement rule still places rare codes, so the LLMs
                are needed only on the shortlist
  no_llm_alone  no_llm without the agreement rule: rare picks are replaced by the best non-rare pick (no LLM at all)
Encoder corrected for label frequency (post-hoc logit adjustment, Menon et al. 2021): argmax of log p(c) - tau log pi(c),
pi(c) the code's share of the training reports (add-half smoothing); tau = 0.5 and 1.0, and tau chosen for accuracy on
the other year's uniform draw (grid 0 to 1.5).
Reported: accuracy on the uniform draw of 4,000 reports with paired-bootstrap 95% intervals against the encoder, and
rare correct placements (exact) and false placements (estimated from the weighted natural sample) as in
paper_v4_numbers.py.

Usage: python selection_ablation.py   (MAINT_CORPUS=sdr for SDR) -> results/why_it_works/selection_ablation_<corpus>.json
"""
import json
import time

import numpy as np

import candidate_ranker as cr  # sets two threads and idle priority
import maude_selector as ms
import paper_v4_numbers as p4
import selector_vs_rules as sv
from common import CORPUS, ROOT, SEED
from maude_selector_score import resamples
from rare_check import natural_weights, year_counts
from sklearn.model_selection import KFold

LLMS = ("qwen38_27b", "gemma4_31b")
TOPK = 5
TAU_GRID = [round(x, 2) for x in np.arange(0.0, 1.51, 0.1)]


def pairs_nollm(D):
    """(report index, candidate, inputs, is-gold): candidates and inputs from the encoder and the neighbours only."""
    if "nollm" in D["_pairs"]:
        return D["_pairs"]["nollm"]
    N, C = D["probs"].shape
    P = D["probs"]
    top = np.argsort(-P, axis=1)[:, :TOPK]
    I, Cn = [], []
    for i in range(N):
        cs = {int(D["enc"][i])} | set(top[i].tolist()) | set(np.nonzero(D["knn"][i])[0].tolist())
        cs = sorted(cs)
        I += [i] * len(cs)
        Cn += cs
    I, Cn = np.array(I), np.array(Cn)
    logp = np.log(np.clip(P, 1e-9, 1.0))
    order = np.argsort(-P, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(N)[:, None], order] = np.arange(C)[None, :]
    ent = -(P * logp).sum(1)
    cnt, enc = D["counts"], D["enc"]
    X = np.column_stack([
        logp[I, Cn], np.log1p(rank[I, Cn]), enc[I] == Cn, D["knn"][I, Cn], D["nn1"][I] == Cn, np.log1p(cnt[Cn]),
        cr.near_matrix(D)[Cn, enc[I]], D["enc_p"][I], D["energy"][I], ent[I], D["nb_cos"][I], np.log1p(D["text_len"][I]),
        np.log1p(cnt[enc[I]])]).astype(np.float32)
    D["_pairs"]["nollm"] = (I, Cn, X, (Cn == D["gold"][I]).astype(int))
    return D["_pairs"]["nollm"]


def enc_table(D, rows):
    C = len(D["codes"])
    M = np.zeros((C, C))
    np.add.at(M, (D["enc"][rows], D["gold"][rows]), 1.0)
    return M, np.bincount(D["gold"][rows], minlength=C) / len(rows)


def enc_table_input(D, I, Cn, M, prior, m=5.0):
    z = D["enc"][I]
    return ((M[z, Cn] + m * prior[Cn]) / (M[z].sum(1) + m)).astype(np.float32)


def fit_nollm(D, rows):
    rows = np.asarray(rows)
    I, Cn, X, y = pairs_nollm(D)
    keep = np.isin(I, rows)
    I, Cn, X, y = I[keep], Cn[keep], X[keep], y[keep]
    T = np.zeros(len(I), np.float32)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(rows):
        M, pr = enc_table(D, rows[tr])
        s = np.isin(I, rows[te])
        T[s] = enc_table_input(D, I[s], Cn[s], M, pr)
    M, pr = enc_table(D, rows)
    return {"model": cr.model("gbm").fit(np.column_stack([X, T]), y), "M": M, "prior": pr}


def outputs_nollm(R, D, rare, default):
    I, Cn, X, _ = pairs_nollm(D)
    p = R["model"].predict_proba(np.column_stack([X, enc_table_input(D, I, Cn, R["M"], R["prior"])]))[:, 1]
    N = len(D["gold"])
    pick, s_pick = D["enc"].copy(), np.full(N, -1.0)
    nonrare, s_nonrare = default.copy(), np.full(N, -1.0)
    for i, c, s in zip(I, Cn, p):
        if s > s_pick[i]:
            s_pick[i], pick[i] = s, c
        if not rare[c] and s > s_nonrare[i]:
            s_nonrare[i], nonrare[i] = s, c
    return pick, nonrare


def logit_adjusted(D, tau):
    pi = (D["counts"] + 0.5) / (D["counts"] + 0.5).sum()
    return (np.log(np.clip(D["probs"], 1e-12, 1.0)) - tau * np.log(pi)[None, :]).argmax(1)


def predictions(D, R_full, R_notab, R_nollm, tau_star):
    fire, best, rare, enc_nonrare = p4.gate_parts(D)
    pf, pf_nr = p4.ranker_outputs(cr, R_full, D, rare, enc_nonrare)
    pt, pt_nr = p4.ranker_outputs(cr, R_notab, D, rare, enc_nonrare)
    pn, pn_nr = outputs_nollm(R_nollm, D, rare, enc_nonrare)
    out = {"enc": D["enc"], "full": p4.gated(pf, pf_nr, fire, best, rare), "no_table": p4.gated(pt, pt_nr, fire, best, rare),
           "no_llm": p4.gated(pn, pn_nr, fire, best, rare), "no_llm_alone": np.where(rare[pn], pn_nr, pn)}
    for tau in (0.5, 1.0):
        out[f"logit_adjusted_tau{tau}"] = logit_adjusted(D, tau)
    out["logit_adjusted_tau_chosen"] = logit_adjusted(D, tau_star)
    return out


def main():
    t0 = time.time()
    rep = {"note": "review of manuscript v4", "corpus": CORPUS, "llms": {}}
    for llm in LLMS:
        D = {w: cr.load(w, llm) for w in ("2025", "2026", "2025nat", "2026nat")}
        W = {y: natural_weights({"D": D[y + "nat"]}, y) for y in ("2025", "2026")}
        L = {}
        for tr, te in (("2025", "2026"), ("2026", "2025")):
            Dt, rows = D[tr + "nat"], W[tr][1]
            R_full = cr.fit(Dt, rows, "wide_gbm_table")
            R_notab = cr.fit(Dt, rows, "wide_gbm")
            R_nollm = fit_nollm(Dt, rows)
            tau_star = max(TAU_GRID, key=lambda t: float((logit_adjusted(Dt, t)[rows] == Dt["gold"][rows]).mean()))
            wn, U = W[te]
            Pb = predictions(D[te], R_full, R_notab, R_nollm, tau_star)
            Pn = predictions(D[te + "nat"], R_full, R_notab, R_nollm, tau_star)
            gold_u = D[te + "nat"]["gold"][U]
            samples = resamples(len(U))
            S = {}
            for k in Pn:
                tp, fp, vis = year_counts({"D": D[te]}, {"D": D[te + "nat"]}, wn, Pb[k], Pn[k])
                S[k] = {"accuracy": round(float((Pn[k][U] == gold_u).mean()), 4),
                        "minus_encoder": sv.acc_ci(gold_u, Pn[k][U], Pn["enc"][U], samples),
                        "minus_full": sv.acc_ci(gold_u, Pn[k][U], Pn["full"][U], samples),
                        "rare_correct": round(tp, 1), "rare_false_estimated": round(fp, 1),
                        "rare_false_ci95": p4.false_interval(D[te], Pb[k], D[te + "nat"], Pn[k], wn)}
            L[te] = {"fitted_on": tr, "tau_chosen": tau_star, "methods": S}
            print(f"[{CORPUS} {llm} {te}] " + ", ".join(f"{k} {v['accuracy']}" for k, v in S.items())
                  + f" tau* {tau_star} ({time.time() - t0:.0f}s)", flush=True)
        rep["llms"][llm] = L
        del D
    p = ROOT / "results" / "why_it_works" / f"selection_ablation_{CORPUS}.json"
    p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
