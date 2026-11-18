"""Candidate ranker (exploration; notes/selector-vs-rules-2026-11-11.md, section "후보 재순위").

Can a different model, richer inputs or a wider set of candidate answers do better than the
selector, on the balanced test and at natural proportions? Saved outputs only (MAUDE; no GPU,
no API). Not frozen in advance: every number here is exploratory.

For each report the candidates are either the three coders' answers ("three") or those plus the
encoder's five most probable codes and the codes of the ten nearest training reports ("wide").
One model scores every (report, candidate) pair for "is this the filed code" and the best-scored
candidate is kept. Inputs per pair: the encoder's log-probability and rank of the candidate,
which coders named it, its share of the neighbours, whether the nearest neighbour carries it,
its training count, whether it sits next to the encoder's answer in the code tree, and the
report-level inputs of the selector (confidence, energy, entropy, agreement, counts of the three
answers, the encoder's log-probability of each LLM answer). "+table" adds, for each coder, how
often the filed code was the candidate when that coder gave its present answer (training rows
only, out-of-fold inside the training set).

Settings: "balanced" is fitted on the 2024 practice set; "natural" on the uniform draw of one
year's natural sample and scored on the other year (2025 -> 2026 and 2026 -> 2025). In the natural
setting "sel14" is the selector's own form (one logistic model per coder on its 14 inputs) fitted
on the same uniform draw, to separate the effect of the training data from that of the inputs,
and "cascade_nat" is a confidence cascade (encoder confidence below a threshold -> one LLM's
answer) whose LLM and threshold are chosen for accuracy on that draw. "+zs5" adds the rule of
rare_confirm.py: the encoder's best-ranked rare code replaces the answer when it is inside the
encoder's top five and the zero-shot LLM names the same code.

Usage: python candidate_ranker.py                    -> results/candidate_ranker/metrics.json
       MAINT_CORPUS=sdr python candidate_ranker.py   -> results/candidate_ranker/sdr_metrics.json (2026-11-16)
"""
import json
import time

import numpy as np

import selector_vs_rules as sv  # sets two threads and idle priority; loads the frozen choices
import redesign_score as rs  # noqa: E402
import maude_selector as ms  # noqa: E402
from common import ROOT, SEED  # noqa: E402
from maude_selector_score import resamples  # noqa: E402
from rare_check import natural_weights, year_counts  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

LLMS = rs.LLMS
TOPK = 5
VARIANTS = {"three_gbm": ("three", False, "gbm"), "wide_gbm": ("wide", False, "gbm"),
            "wide_gbm_table": ("wide", True, "gbm"), "wide_logit_table": ("wide", True, "logit")}
ER = {"q1": 0.4, "q2": 0.8, "lam": 0.9}  # the literal EnergyRoute of selector_vs_rules.py (lam from 2024)


def er_setting(llm):
    """The literal EnergyRoute with the fusion weight selector_vs_rules.py chose on 2024 for this corpus."""
    from common import CORPUS
    p = ROOT / "results" / "selector_vs_rules" / f"{CORPUS}_metrics.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))["llms"].get(llm, {}).get("settings", {}).get("energyroute")
        if s:
            return {"q1": s["q1"], "q2": s["q2"], "lam": s["lam"]}
    return ER
_NEAR = {}


def near_matrix(D):
    """[a, g]: code a is a parent, child or sibling of code g in the code tree."""
    key = len(D["codes"])
    if key not in _NEAR:
        from maude_data import class_info, load_split
        info = class_info(load_split())
        paths = [info[c]["path"] for c in D["codes"]]
        par = [p.rsplit(" > ", 1)[0] if " > " in p else None for p in paths]
        M = np.zeros((key, key), bool)
        for a in range(key):
            for g in range(key):
                if a != g and (paths[a].startswith(paths[g] + " >") or paths[g].startswith(paths[a] + " >")
                               or (par[a] is not None and par[a] == par[g])):
                    M[a, g] = True
        _NEAR[key] = M
    return _NEAR[key]


def load(which, llm):
    D = sv.load(which, llm)
    nb = json.loads(sv.nb_file(which).read_text())
    idx = {c: i for i, c in enumerate(D["codes"])}
    D["nn1"] = np.array([idx[nb[str(k)][0][1]] if nb[str(k)] else -1 for k in D["keys"]])
    D["which"], D["_pairs"] = which, {}
    return D


def pairs(D, mode):
    """(report index, candidate code, inputs, is-gold) for every candidate of every report."""
    if mode in D["_pairs"]:
        return D["_pairs"][mode]
    N, C = D["probs"].shape
    P = D["probs"]
    top = np.argsort(-P, axis=1)[:, :TOPK]
    I, Cn = [], []
    for i in range(N):
        cs = {int(D["enc"][i])} | {int(x) for x in (D["zs"][i], D["rag"][i]) if x >= 0}
        if mode == "wide":
            cs |= set(top[i].tolist()) | set(np.nonzero(D["knn"][i])[0].tolist())
        cs = sorted(cs)
        I += [i] * len(cs)
        Cn += cs
    I, Cn = np.array(I), np.array(Cn)
    logp = np.log(np.clip(P, 1e-9, 1.0))
    order = np.argsort(-P, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(N)[:, None], order] = np.arange(C)[None, :]
    ent = -(P * logp).sum(1)
    cnt, enc, zs, rag = D["counts"], D["enc"], D["zs"], D["rag"]

    def lc(pred):
        return np.where(pred >= 0, np.log1p(cnt[np.maximum(pred, 0)]), -1.0)

    def lp(pred):
        return np.where(pred >= 0, logp[np.arange(N), np.maximum(pred, 0)], -25.0)
    X = np.column_stack([
        logp[I, Cn], np.log1p(rank[I, Cn]), enc[I] == Cn, zs[I] == Cn, rag[I] == Cn, D["knn"][I, Cn], D["nn1"][I] == Cn,
        np.log1p(cnt[Cn]), near_matrix(D)[Cn, enc[I]],
        D["enc_p"][I], D["energy"][I], ent[I], D["nb_cos"][I], np.log1p(D["text_len"][I]), zs[I] < 0, rag[I] < 0,
        enc[I] == zs[I], enc[I] == rag[I], (zs[I] == rag[I]) & (zs[I] >= 0), lc(enc)[I], lc(zs)[I], lc(rag)[I],
        lp(zs)[I], lp(rag)[I]]).astype(np.float32)
    D["_pairs"][mode] = (I, Cn, X, (Cn == D["gold"][I]).astype(int))
    return D["_pairs"][mode]


# ---------------------------------------------------------------- the ranker
def tables(D, rows):
    """Per coder: counts of (the coder's answer, filed code) over the reports `rows`; last row = invalid answer."""
    C = len(D["codes"])
    out = {}
    for a in ms.ARMS:
        M = np.zeros((C + 1, C))
        np.add.at(M, (np.where(D[a][rows] >= 0, D[a][rows], C), D["gold"][rows]), 1.0)
        out[a] = M
    return out, np.bincount(D["gold"][rows], minlength=C) / len(rows)


def table_inputs(D, I, Cn, tabs, prior, m=5.0):
    cols = []
    for a in ms.ARMS:
        z = np.where(D[a][I] >= 0, D[a][I], tabs[a].shape[0] - 1)
        cols.append((tabs[a][z, Cn] + m * prior[Cn]) / (tabs[a][z].sum(1) + m))
    return np.column_stack(cols).astype(np.float32)


def model(kind):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                                              l2_regularization=1.0, early_stopping=False, random_state=SEED)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def fit(D, rows, variant):
    mode, use_table, kind = VARIANTS[variant]
    rows = np.asarray(rows)
    I, Cn, X, y = pairs(D, mode)
    keep = np.isin(I, rows)
    I, Cn, X, y = I[keep], Cn[keep], X[keep], y[keep]
    tabs = prior = None
    if use_table:
        T = np.zeros((len(I), len(ms.ARMS)), np.float32)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(rows):
            tb, pr = tables(D, rows[tr])
            s = np.isin(I, rows[te])
            T[s] = table_inputs(D, I[s], Cn[s], tb, pr)
        X = np.column_stack([X, T])
        tabs, prior = tables(D, rows)
    return {"variant": variant, "model": model(kind).fit(X, y), "tabs": tabs, "prior": prior, "n_reports": int(len(rows)),
            "n_pairs": int(len(y))}


def predict(R, D):
    mode, use_table, _ = VARIANTS[R["variant"]]
    I, Cn, X, y = pairs(D, mode)
    if use_table:
        X = np.column_stack([X, table_inputs(D, I, Cn, R["tabs"], R["prior"])])
    p = R["model"].predict_proba(X)[:, 1]
    best, pred = np.full(len(D["gold"]), -1.0), D["enc"].copy()
    for i, c, s in zip(I, Cn, p):
        if s > best[i]:
            best[i], pred[i] = s, c
    return pred


def fit_sel14(D, rows):
    X, Y = ms.features(D)[rows], ms.targets(D)[rows]
    return ms.Learned("S2").fit(X, Y)


def fit_cascade(D, rows):
    """(LLM arm, threshold) of the confidence cascade with the best accuracy on the reports `rows`."""
    grid = [(a, round(float(th), 2)) for a in ("zs", "rag") for th in np.arange(0.05, 1.001, 0.05)]
    return max(grid, key=lambda g: float((rs.cascade(D, *g)[rows] == D["gold"][rows]).mean()))


def confirm(D, pred, k=5):
    """The rule of rare_confirm.py on top of any answer vector (confirmer: the zero-shot LLM)."""
    P = D["probs"]
    N, C = P.shape
    rare = (D["counts"] >= 1) & (D["counts"] <= 5)
    order = np.argsort(-P, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(N)[:, None], order] = np.arange(C)[None, :]
    rr = np.where(rare[None, :], rank, 10**6)
    best = rr.argmin(1)
    return np.where((rr.min(1) < k) & (D["zs"] == best), best, pred)


BASE = ("enc", "selector", "selector_natural", "energyroute", "oracle_three", "oracle_wide")


def covered(D, mode):
    I, _, _, y = pairs(D, mode)
    has = np.zeros(len(D["gold"]), bool)
    np.logical_or.at(has, I, y.astype(bool))
    return has


# ---------------------------------------------------------------- scoring
def baselines(D, llm, frozen, models, w=None):
    return {"enc": D["enc"], "selector": ms.predict_frozen(D, frozen, models)[0],
            "selector_natural": sv.natural_setting(D, llm, frozen, models), "energyroute": sv.energyroute(D, er_setting(llm), w),
            "oracle_three": np.where(ms.targets(D).any(axis=1), D["gold"], D["enc"]),
            "oracle_wide": np.where(covered(D, "wide"), D["gold"], D["enc"])}


def balanced_scores(D, P, ref=("selector", "enc")):
    gold, labels, counts = D["gold"], D["labels"], D["counts"]
    rare = np.array([c for c in labels if 1 <= counts[c] <= 5])
    samples = resamples(len(gold))
    out = {k: {"all": rs.mf(gold, p, labels), "1-5": rs.mf(gold, p, rare)} for k, p in P.items()}
    con = {k: {r: rs.contrast(gold, P[k], P[r], labels, samples) for r in ref} for k in P if k not in BASE}
    return {"macro_f1": out, "minus": con}


def natural_scores(Db, Pb, Dn, Pn, wn, U):
    ref = [r for r in ("enc", "selector_natural", "energyroute", "cascade_nat") if r in Pn]
    gold_u = Dn["gold"][U]
    labels_u = np.array(sorted(set(gold_u.tolist())))
    samples = resamples(len(U))
    out = {}
    for k in Pn:
        tp, fp, vis = year_counts({"D": Db}, {"D": Dn}, wn, Pb[k], Pn[k])
        out[k] = {"accuracy": round(float((Pn[k][U] == gold_u).mean()), 4), "macro_f1_uniform": rs.mf(gold_u, Pn[k][U], labels_u),
                  "right_rare": round(tp, 1), "false_rare": round(fp, 1), "modes_visible": vis}
    con = {k: {r: {"accuracy": sv.acc_ci(gold_u, Pn[k][U], Pn[r][U], samples),
                   "macro_f1": rs.contrast(gold_u, Pn[k][U], Pn[r][U], labels_u, samples)} for r in ref}
           for k in Pn if k not in BASE}
    return {"scores": out, "minus": con}


def show(tag, S, keys):
    print(f"   {tag}: " + ", ".join(f"{k} {S[k]}" for k in keys), flush=True)


def main():
    t0 = time.time()
    rep = {"note": "notes/selector-vs-rules-2026-11-11.md", "git_head": ms.git_head(), "variants": {k: list(v) for k, v in VARIANTS.items()},
           "top_k": TOPK, "llms": {}}
    base = ["enc", "selector", "selector_natural", "energyroute"]
    for llm in LLMS:
        frozen, models = rs.check_frozen(llm)
        try:
            D = {w: load(w, llm) for w in ("2024", "2025", "2026", "2025nat", "2026nat")}
        except SystemExit as e:  # e.g. Opus has no answers on SDR's natural samples
            rep["llms"][llm] = {"not_run": str(e)}
            print(f"[{llm}] not run: {e}", flush=True)
            continue
        W = {y: natural_weights({"D": D[y + "nat"]}, y) for y in ("2025", "2026")}
        L = {"gold_among_candidates": {}, "balanced_setting": {}, "natural_setting": {}}
        for w in D:
            idx = W[w[:4]][1] if w.endswith("nat") else np.arange(len(D[w]["gold"]))
            L["gold_among_candidates"][w] = {m: round(float(covered(D[w], m)[idx].mean()), 4) for m in ("three", "wide")}
        print(f"[{llm}] gold among candidates {L['gold_among_candidates']} ({time.time() - t0:.0f}s)", flush=True)

        def run(rankers, year, sel14=None, casc=None):
            Db, Dn = D[year], D[year + "nat"]
            wn, U = W[year]
            Pb, Pn = baselines(Db, llm, frozen, models), baselines(Dn, llm, frozen, models, wn)
            if sel14 is not None:
                Pb["sel14"], Pn["sel14"] = ms.choose(sel14.proba(ms.features(Db)), Db)[0], ms.choose(sel14.proba(ms.features(Dn)), Dn)[0]
            for v, R in rankers.items():
                Pb[v], Pn[v] = predict(R, Db), predict(R, Dn)
            if casc is not None:
                Pb["cascade_nat"], Pn["cascade_nat"] = rs.cascade(Db, *casc), rs.cascade(Dn, *casc)
                for k in ("enc", "wide_gbm_table"):
                    Pb[k + "+zs5"], Pn[k + "+zs5"] = confirm(Db, Pb[k]), confirm(Dn, Pn[k])
            return {"balanced_test": balanced_scores(Db, Pb), "natural": natural_scores(Db, Pb, Dn, Pn, wn, U)}

        bal = {v: fit(D["2024"], np.arange(len(D["2024"]["gold"])), v) for v in VARIANTS}
        for year in ("2026", "2025"):
            S = L["balanced_setting"][year] = run(bal, year)
            print(f"[{llm}] balanced setting (fit 2024) -> {year} ({time.time() - t0:.0f}s)", flush=True)
            show("balanced macro-F1", {k: v["all"] for k, v in S["balanced_test"]["macro_f1"].items()}, base + list(VARIANTS))
            show("natural accuracy ", {k: v["accuracy"] for k, v in S["natural"]["scores"].items()}, base + list(VARIANTS))
            for v in VARIANTS:
                c = S["balanced_test"]["minus"][v]["selector"]
                print(f"      {v:18s} - selector (balanced macro-F1) {c['item']['point']:+.4f} item {c['item']['ci95']} "
                      f"code {c['code']['ci95']}", flush=True)
        for tr, te in (("2025", "2026"), ("2026", "2025")):
            nat = {v: fit(D[tr + "nat"], W[tr][1], v) for v in VARIANTS}
            casc = fit_cascade(D[tr + "nat"], W[tr][1])
            S = L["natural_setting"][f"{tr}->{te}"] = run(nat, te, fit_sel14(D[tr + "nat"], W[tr][1]), casc)
            S["cascade_nat"] = {"arm": casc[0], "theta": casc[1]}
            print(f"[{llm}] natural setting (fit {tr} uniform draw) -> {te} ({time.time() - t0:.0f}s)", flush=True)
            show("natural accuracy ", {k: v["accuracy"] for k, v in S["natural"]["scores"].items()},
                 base + ["cascade_nat", "sel14"] + list(VARIANTS) + ["enc+zs5", "wide_gbm_table+zs5", "oracle_three", "oracle_wide"])
            show("balanced macro-F1", {k: v["all"] for k, v in S["balanced_test"]["macro_f1"].items()},
                 base + list(VARIANTS) + ["enc+zs5", "wide_gbm_table+zs5"])
            show("rare right/false ", {k: f"{v['right_rare']:.0f}/{v['false_rare']:.0f}" for k, v in S["natural"]["scores"].items()},
                 base + ["cascade_nat"] + list(VARIANTS) + ["enc+zs5", "wide_gbm_table+zs5"])
            for v in ["sel14"] + list(VARIANTS) + ["wide_gbm_table+zs5"]:
                for r in ("enc", "selector_natural", "energyroute", "cascade_nat"):
                    c = S["natural"]["minus"][v][r]["accuracy"]
                    print(f"      {v:18s} - {r:16s} (natural accuracy) {c['point']:+.4f} {c['ci95']}", flush=True)
        rep["llms"][llm] = L
        del D
    out = ROOT / "results" / "candidate_ranker"
    out.mkdir(parents=True, exist_ok=True)
    from common import CORPUS
    name = "metrics.json" if CORPUS == "maude" else f"{CORPUS}_metrics.json"
    (out / name).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"wrote {out / name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
