"""All numbers of the v4 paper from one fixed configuration (notes/paper-plan-v4-2026-11-16.md).

Ours = three coders (encoder, zero-shot LLM, retrieval LLM) and two parts:
  gate       a rare code (1-5 training reports) is placed only when it is the encoder's best-ranked rare code, it
             is inside the encoder's top K, and the zero-shot or the retrieval LLM names exactly that code. Any other
             rare answer is replaced by the best non-rare answer.
  selection  for every report, the wide candidate ranker of candidate_ranker.py (candidates: the three answers, the
             encoder's top 5, the ten neighbours' codes; boosting with the coder tables), fitted on the uniform draw of
             the other year's natural sample. Its rare picks go through the gate.
Two versions: "ours_rule" (encoder + gate; the LLMs are needed only for the reports with a rare code in the encoder's
top K) and "ours_full" (selection + gate; LLM answers needed for every report).

Per corpus and LLM, each test year scored with the configuration fitted on the other year (2025 -> 2026, 2026 -> 2025):
accuracy on the natural sample's uniform draw (paired bootstrap intervals against the encoder and against Ours),
macro-F1 on the same draw, right and false rare placements over the year (rare_check.year_counts: every rare report of
the year is counted; false placements on ordinary reports are estimated from the weighted uniform draw), and macro-F1 on
the balanced test. Baselines: the encoder, each LLM alone, the confidence cascade fitted on the other year's uniform
draw, EnergyRoute (selector_vs_rules.py setting), the selector fitted on the balanced practice set (frozen), and the
same selector with prior correction (option A). Exact whole-year rare counts of the gate come from rare_year.py.

Usage: python paper_v4_numbers.py ; MAINT_CORPUS=sdr python paper_v4_numbers.py  -> results/paper_v4/<corpus>_numbers.json
       python paper_v4_numbers.py --summary   -> results/paper_v4/tables.md (both corpora, rare_year and tail_split added)
"""
import json
import sys
import time

import numpy as np

K = 5
LLMS_MAIN = ("qwen38_27b", "gemma4_31b")
METHODS = ("enc", "zs", "rag", "cascade", "energyroute", "selector_balanced", "selector_prior", "ours_rule", "ours_full")
NAMES = {"enc": "Encoder", "zs": "Zero-shot LLM", "rag": "Retrieval LLM", "cascade": "Confidence cascade",
         "energyroute": "EnergyRoute", "selector_balanced": "Selector (balanced fit)", "selector_prior": "Selector (prior-corrected)",
         "ours_rule": "Ours (gate only)", "ours_full": "Ours (gate + selection)"}


def gate_parts(D, k=K):
    """(fire, best rare code, rare mask, encoder's best non-rare code) for every report of D."""
    P = D["probs"]
    N, C = P.shape
    rare = (D["counts"] >= 1) & (D["counts"] <= 5)
    order = np.argsort(-P, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(N)[:, None], order] = np.arange(C)[None, :]
    rr = np.where(rare[None, :], rank, 10**6)
    best, brank = rr.argmin(1), rr.min(1)
    fire = (brank < k) & ((D["zs"] == best) | (D["rag"] == best))
    enc_nonrare = np.where(rare[None, :], -1.0, P).argmax(1)
    return fire, best, rare, enc_nonrare


def gated(base, fallback, fire, best, rare):
    out = np.where((base >= 0) & rare[np.maximum(base, 0)], fallback, base)
    return np.where(fire, best, out)


def ranker_outputs(cr, R, D, rare, default):
    """The ranker's pick and its best non-rare pick for every report."""
    mode, use_table, _ = cr.VARIANTS[R["variant"]]
    I, Cn, X, _ = cr.pairs(D, mode)
    if use_table:
        X = np.column_stack([X, cr.table_inputs(D, I, Cn, R["tabs"], R["prior"])])
    p = R["model"].predict_proba(X)[:, 1]
    N = len(D["gold"])
    pick, s_pick = D["enc"].copy(), np.full(N, -1.0)
    nonrare, s_nonrare = default.copy(), np.full(N, -1.0)
    for i, c, s in zip(I, Cn, p):
        if s > s_pick[i]:
            s_pick[i], pick[i] = s, c
        if not rare[c] and s > s_nonrare[i]:
            s_nonrare[i], nonrare[i] = s, c
    return pick, nonrare


# ---------------------------------------------------------------- one corpus
def predictions(cr, sv, rs, ms, D, llm, frozen, models, R, casc, w=None):
    fire, best, rare, enc_nonrare = gate_parts(D)
    pick, pick_nonrare = ranker_outputs(cr, R, D, rare, enc_nonrare)
    return {"enc": D["enc"], "zs": D["zs"], "rag": D["rag"], "cascade": rs.cascade(D, *casc),
            "energyroute": sv.energyroute(D, cr.er_setting(llm), w),
            "selector_balanced": ms.predict_frozen(D, frozen, models)[0],
            "selector_prior": sv.natural_setting(D, llm, frozen, models),
            "ours_rule": gated(D["enc"], enc_nonrare, fire, best, rare),
            "ours_full": gated(pick, pick_nonrare, fire, best, rare)}


def false_interval(Db, pb, Dn, pn, wn):
    """95% interval of the year's false rare placements: the balanced test and every stratum drawn in full count
    exactly; each sampled stratum (weight above 1) gets a Garwood Poisson interval on its count, times its weight."""
    from scipy.stats import chi2
    c = Db["counts"]

    def wrong(D, p):
        q = np.maximum(p, 0)
        return (p >= 0) & (c[q] >= 1) & (c[q] <= 5) & (p != D["gold"])
    exact, lo, hi = float(wrong(Db, pb).sum()), 0.0, 0.0
    fw = wrong(Dn, pn)
    for w in np.unique(wn[wn > 0]):
        h = int((fw & (wn == w)).sum())
        if abs(w - 1.0) < 1e-9:
            exact += h
            continue
        lo += w * (chi2.ppf(0.025, 2 * h) / 2 if h else 0.0)
        hi += w * chi2.ppf(0.975, 2 * h + 2) / 2
    return [round(exact + lo, 1), round(exact + hi, 1)]


def score(sv, rs, Db, Pb, Dn, Pn, wn, U):
    from maude_selector_score import resamples
    from rare_check import year_counts
    gold_u = Dn["gold"][U]
    labels_u = np.array(sorted(set(gold_u.tolist())))
    samples = resamples(len(U))
    out = {}
    for k in METHODS:
        tp, fp, vis = year_counts({"D": Db}, {"D": Dn}, wn, Pb[k], Pn[k])
        out[k] = {"accuracy": round(float((Pn[k][U] == gold_u).mean()), 4),
                  "macro_f1_natural": rs.mf(gold_u, Pn[k][U], labels_u),
                  "rare_right": round(tp, 1), "rare_false": round(fp, 1), "rare_modes": vis,
                  "rare_false_ci95": false_interval(Db, Pb[k], Dn, Pn[k], wn),
                  "macro_f1_balanced": rs.mf(Db["gold"], Pb[k], Db["labels"]),
                  "acc_minus_enc": sv.acc_ci(gold_u, Pn[k][U], Pn["enc"][U], samples),
                  "ours_full_minus": sv.acc_ci(gold_u, Pn["ours_full"][U], Pn[k][U], samples)}
    return out


def data_stats():
    from maude_data import (class_info, known_codes, load_natural_2025, load_natural_2026, load_split, load_test_2025,
                            load_test_2026, train_keys)
    from maude_router import training_counts
    split = load_split()
    codes = sorted(class_info(split))
    counts = dict(zip(codes, training_counts(split, codes)["full"].tolist()))
    known = known_codes(split)
    return {"training_reports": len(train_keys(split, "full")), "known_codes": len(known),
            "rare_codes": sum(1 <= counts[c] <= 5 for c in known),
            "balanced_test": {"2025": len(load_test_2025()), "2026": len(load_test_2026())},
            "natural_sample": {"2025": len(load_natural_2025()), "2026": len(load_natural_2026())}}


def run_corpus():
    import candidate_ranker as cr  # sets two threads and idle priority (selector_vs_rules)
    import maude_selector as ms
    import redesign_score as rs
    import selector_vs_rules as sv
    from common import CORPUS, ROOT
    from rare_check import natural_weights
    t0 = time.time()
    rep = {"note": "notes/paper-plan-v4-2026-11-16.md", "corpus": CORPUS, "git_head": ms.git_head(),
           "config": {"gate_top_k": K, "gate_confirmer": "zero-shot or retrieval LLM", "unconfirmed_rare": "best non-rare answer",
                      "selection": "candidate_ranker wide_gbm_table, fitted on the other year's uniform draw"},
           "data": data_stats(), "llms": {}}
    print(f"[{CORPUS}] data {rep['data']} ({time.time() - t0:.0f}s)", flush=True)
    for llm in rs.LLMS:
        frozen, models = rs.check_frozen(llm)
        try:
            D = {w: cr.load(w, llm) for w in ("2025", "2026", "2025nat", "2026nat")}
        except SystemExit as e:
            rep["llms"][llm] = {"not_run": str(e)}
            print(f"[{CORPUS} {llm}] not run: {e}", flush=True)
            continue
        W = {y: natural_weights({"D": D[y + "nat"]}, y) for y in ("2025", "2026")}
        L = {}
        for tr, te in (("2025", "2026"), ("2026", "2025")):
            R = cr.fit(D[tr + "nat"], W[tr][1], "wide_gbm_table")
            casc = cr.fit_cascade(D[tr + "nat"], W[tr][1])
            wn, U = W[te]
            Pb = predictions(cr, sv, rs, ms, D[te], llm, frozen, models, R, casc)
            Pn = predictions(cr, sv, rs, ms, D[te + "nat"], llm, frozen, models, R, casc, wn)
            S = score(sv, rs, D[te], Pb, D[te + "nat"], Pn, wn, U)
            L[te] = {"fitted_on": tr, "cascade": {"arm": casc[0], "theta": casc[1]}, "n_uniform": int(len(U)), "methods": S}
            print(f"[{CORPUS} {llm} {te}] accuracy " + ", ".join(f"{k} {S[k]['accuracy']}" for k in METHODS)
                  + f" | rare right/false ours {S['ours_full']['rare_right']:.0f}/{S['ours_full']['rare_false']:.0f}"
                  + f" enc {S['enc']['rare_right']:.0f}/{S['enc']['rare_false']:.0f} ({time.time() - t0:.0f}s)", flush=True)
        rep["llms"][llm] = L
        del D
    out = ROOT / "results" / "paper_v4"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{CORPUS}_numbers.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"wrote {out / (CORPUS + '_numbers.json')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- tables for the paper
def summary():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    rd = lambda p: json.loads((root / p).read_text(encoding="utf-8"))  # noqa: E731
    J = {c: rd(f"results/paper_v4/{c}_numbers.json") for c in ("maude", "sdr")}
    RY = {"maude": rd("results/rare_year/metrics.json"), "sdr": rd("results/rare_year/sdr_metrics.json")}
    TS = {c: rd(f"results/rare_year/tail_split_{c}.json") for c in ("maude", "sdr")}
    cols = [(c, y) for c in ("maude", "sdr") for y in ("2025", "2026")]
    ci = lambda x: f"{100 * x['point']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"  # noqa: E731
    L = ["# Paper v4 numbers", "", "Generated by `scripts/paper_v4_numbers.py --summary` from results/paper_v4, results/rare_year. "
         "Accuracy in percent on the natural sample's uniform draw (4,000 reports a year); differences in points with paired-bootstrap 95% intervals.", ""]
    L += ["## Table 1. Data", "", "| Corpus | Codes | Rare codes | Training reports | Year | Reports | Rare-code reports | Rare share | Reports with a rare code in the encoder's top 5 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, y in cols:
        d, Y = J[c]["data"], RY[c]["years"][y]
        top5 = Y["best_rare_code_inside_top_k"]["5"]
        L.append(f"| {c.upper()} | {d['known_codes']} | {d['rare_codes']} | {d['training_reports']:,} | {y} | {Y['reports']:,} | {Y['rare_reports']} "
                 f"({Y['rare_modes_with_reports']} codes) | 1 in {round(Y['reports'] / Y['rare_reports']):,} | {top5:,} ({100 * top5 / Y['reports']:.2f}%) |")
    for llm in LLMS_MAIN:
        L += ["", f"## Table 2 ({llm}). Accuracy at natural proportions", "", "| Method | " + " | ".join(f"{c.upper()} {y}" for c, y in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for k in METHODS:
            cells = []
            for c, y in cols:
                m = J[c]["llms"][llm][y]["methods"][k]
                cells.append(f"{100 * m['accuracy']:.1f}" + ("" if k == "enc" else f" ({ci(m['acc_minus_enc'])})"))
            L.append(f"| {NAMES[k]} | " + " | ".join(cells) + " |")
        L += ["", f"Ours (gate + selection) minus each method, accuracy points [95%]:", "", "| Method | " + " | ".join(f"{c.upper()} {y}" for c, y in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for k in METHODS[:-1]:
            L.append(f"| {NAMES[k]} | " + " | ".join(ci(J[c]["llms"][llm][y]["methods"][k]["ours_full_minus"]) for c, y in cols) + " |")
    for llm in LLMS_MAIN:
        L += ["", f"## Figure 3 data ({llm}). Rare codes over a year: right / false", "",
              "Right placements are exact for every method (all rare reports of the year carry every coder's answer). False placements are "
              "exact for the encoder and Ours (whole-year encoder pass + LLM on the shortlist) and estimated from the weighted uniform draw "
              "for the other methods (the 'estimated' column is the same estimator for every method).", "",
              "| Method | " + " | ".join(f"{c.upper()} {y} right / false (estimated)" for c, y in cols) + " |", "|---|" + "---|" * len(cols)]
        for k in METHODS:
            L.append(f"| {NAMES[k]} | " + " | ".join(
                f"{J[c]['llms'][llm][y]['methods'][k]['rare_right']:.0f} / {J[c]['llms'][llm][y]['methods'][k]['rare_false']:.0f}" for c, y in cols) + " |")
        L.append("| Encoder, exact | " + " | ".join(f"{RY[c]['years'][y]['encoder_alone']['right']} / {RY[c]['years'][y]['encoder_alone']['false']}" for c, y in cols) + " |")
        for kk in ("k1", "k3", "k5", "k10"):
            cells = []
            for c, y in cols:
                r = RY[c]["years"][y]["llms"][llm]["rules"][f"{kk}/either"]
                v = r["confirmed_only"]
                cells.append(f"{v['right']} / {v['false']}" if r["complete"] else f"incomplete ({r['coverage']:.2f})")
            L.append(f"| Ours gate, exact, top {kk[1:]} | " + " | ".join(cells) + " |")
    L += ["", "## Appendix A. Macro-F1 on the balanced test (up to 10 reports per code)", ""]
    for llm in LLMS_MAIN:
        L += [f"{llm}:", "", "| Method | " + " | ".join(f"{c.upper()} {y}" for c, y in cols) + " |", "|---|" + "---|" * len(cols)]
        for k in METHODS:
            L.append(f"| {NAMES[k]} | " + " | ".join(f"{100 * J[c]['llms'][llm][y]['methods'][k]['macro_f1_balanced']:.1f}" for c, y in cols) + " |")
        L.append("")
    L += ["## Analysis. Where the LLM's correct rare answers sit in the encoder's ranking", "",
          "| Corpus | Year | LLM | Rare reports | Encoder top 1 | Filed code in encoder top 5 | LLM alone right | of which in top 5 | outside top 5 | Gate right |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ("maude", "sdr"):
        for key, r in TS[c].items():
            _, llm, y = key.split()
            L.append(f"| {c.upper()} | {y} | {llm} | {r['rare_reports']} | {r['encoder_top1']} | {r['gold_in_encoder_top5']} | {r['llm_either_right']} | "
                     f"{r['llm_either_right_in_top5']} | {r['llm_either_right_outside_top5']} | {r['rule_right']} |")
    out = root / "results" / "paper_v4" / "tables.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--summary" in sys.argv[1:]:
        summary()
    else:
        run_corpus()
