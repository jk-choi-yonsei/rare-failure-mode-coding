"""Selector against simple rules (notes/selector-vs-rules-2026-11-11.md).

Does the per-report selector beat simple rules that need no learned model? Every rule is
fitted on the training-year practice set, the data the frozen selector was fitted on
(maude_selector.load_coders("2024")); the development year (2025) and the test year (2026)
are scored against the redesign's frozen selector (splits/redesign_v1_<corpus>.json).
Saved outputs only: no GPU, no API. Runs at idle priority with two threads.

Rules (settings by 5-fold cross-validated macro-F1 on 2024, the selector's own folds)
  energy_gate        encoder energy above a threshold -> the LLM's answer (arm, threshold)
  energyroute        EnergyRoute (Jang et al., Sci Rep 2026): energy percentiles 40/80 of the
                     scored batch split it into encoder / encoder-neighbour fusion / retrieval
                     LLM. The middle tier fuses the encoder's probabilities with the 10 neighbours'
                     label shares (weight from 2024). Before SDR's probabilities were saved
                     (2026-11-16) SDR used the neighbours' vote there.
  energyroute_tuned  the same with both percentiles chosen on 2024
  code_table         each coder's precision for each answered code on 2024, shrunk to the
                     coder's overall precision; the coder with the highest value is kept
                     (local class accuracy)
  code_table_bin     the same, shrunk to the coder's precision in the answered code's
                     training-count bin (0, 1-5, 6-20, 21-89, 90+)
  majority_table     two coders agree -> their answer; otherwise code_table_bin
  pattern_table      agreement pattern (5 kinds) -> one coder (behaviour knowledge space)
  count_conf_gate    encoder's code has <= k training reports or its confidence < p -> the
                     LLM's answer (the Li et al. rule, sent to the LLM instead of a person)
Selector variants (the frozen selector's family, refit on 2024)
  refit              all 14 inputs (a check: must equal the frozen selector)
  no_counts          without the three training-count inputs
  counts_only        the three training-count inputs and the two invalid-answer flags
  selector_natural   the frozen selector moved to natural proportions by the prior correction
                     of prior_correct.py (option A: nothing fitted; the selector's setting for
                     the real mix of reports)

Usage: python selector_vs_rules.py                    -> results/selector_vs_rules/maude_metrics.json
       MAINT_CORPUS=sdr python selector_vs_rules.py   -> results/selector_vs_rules/sdr_metrics.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
try:  # idle priority: the machine may be in use
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:  # noqa: BLE001
    pass

import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import redesign_score as rs  # noqa: E402  (names the selector's encoder before maude_selector loads)
import maude_selector as ms  # noqa: E402
from common import CORPUS, ROOT, SEED, rdir  # noqa: E402
from maude_router import K_GRID, P_GRID  # noqa: E402
from maude_selector_score import resamples  # noqa: E402

LLMS = rs.LLMS
YEARS = ("2026", "2025")
PROBS = ROOT / "runs" / f"{CORPUS}_encprobs" / f"{ms.SELECTOR_ENC}_full_seed42"  # SDR's added 2026-11-16
LAM_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
M_GRID = [0.25, 0.5, 1, 2, 5, 10, 20, 50]
Q_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
CODE_BINS = [(0, 0), (1, 5), (6, 20), (21, 89), (90, 10**9)]
VARIANTS = ("refit", "no_counts", "counts_only")

_TEXT, _load_items = {}, ms.load_items


def _cached_items(keys, pool=None):  # the same reports are read once for all three LLMs
    k = (str(pool), len(keys), keys[0], keys[-1])
    if k not in _TEXT:
        _TEXT[k] = _load_items(keys, pool=pool)
    return _TEXT[k]


ms.load_items = _cached_items


# ---------------------------------------------------------------- loading
def run_dir(which):
    return rdir("maude2025") / "rerun2024" if which == "2024" else rdir("maude" + which)


def nb_file(which):
    return (rdir("maude") if which == "2024" else rdir("maude" + which)) / "neighbors_full.json"


def load(which, llm):
    """maude_selector.load_coders plus the encoder's energy, the neighbours' label shares
    and (MAUDE) the encoder's full probabilities."""
    D = ms.load_coders(which, llm)
    with (run_dir(which) / ms.ENC_CSV).open(encoding="utf-8") as fh:
        e = {int(r["key"]): float(r["energy_novelty"]) for r in csv.DictReader(fh)}
    D["energy"] = np.array([e[k] for k in D["keys"]])
    nb = json.loads(nb_file(which).read_text())
    idx = {c: i for i, c in enumerate(D["codes"])}
    K = np.zeros((len(D["keys"]), len(D["codes"])), np.float32)
    for i, k in enumerate(D["keys"]):
        lst = nb[str(k)]
        for _, c, _ in lst:
            K[i, idx[c]] += 1.0 / len(lst)
    D["knn"], D["probs"], D["probs_check"] = K, None, None
    p = PROBS / f"{which}.npz"
    if p.exists():
        z = np.load(p)  # the encoder's trained classes, a subset of the evaluation set's codes
        cols = [idx.get(str(c)) for c in z["classes"]]
        if None in cols:
            raise SystemExit(f"{p.name}: a class the evaluation set does not know")
        pos = {int(k): i for i, k in enumerate(z["keys"])}
        P = np.zeros((len(D["keys"]), len(D["codes"])), np.float32)
        P[:, np.array(cols)] = np.exp(z["logp"][np.array([pos[k] for k in D["keys"]])].astype(np.float32))
        D["probs"] = P
        D["probs_check"] = round(float((D["probs"].argmax(1) == D["enc"]).mean()), 4)
    return D


# ---------------------------------------------------------------- rules
ITEM = ("gold", "enc", "enc_p", "zs", "rag", "nb_share", "nb_vote", "nb_cos", "text_len", "energy", "knn", "probs")


def sub(D, idx):
    S = dict(D)
    for k in ITEM:
        if S.get(k) is not None:
            S[k] = D[k][idx]
    S["keys"] = [D["keys"][i] for i in idx]
    if "mids" in D:
        S["mids"] = {lam: m[idx] for lam, m in D["mids"].items()}
    return S


def wquantile(x, qs, w=None):
    if w is None:
        return np.quantile(x, qs)
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return np.array([x[o][min(int(np.searchsorted(cw, q)), len(x) - 1)] for q in qs])


def to_llm(D, use, arm):
    return np.where(use & (D[arm] >= 0), D[arm], D["enc"])


def mid_tier(D, lam):
    if lam is None or D["probs"] is None:
        return np.where(D["nb_vote"] >= 0, D["nb_vote"], D["enc"])
    if "mids" in D and lam in D["mids"]:
        return D["mids"][lam]
    return (lam * D["probs"] + (1.0 - lam) * D["knn"]).argmax(1)


def er_thresholds(D, h, w=None):
    return wquantile(D["energy"], [h["q1"], h["q2"]], w)


def energyroute(D, h, w=None):
    t1, t2 = er_thresholds(D, h, w)
    mid = mid_tier(D, h["lam"])
    top = np.where(D["rag"] >= 0, D["rag"], mid)
    e = D["energy"]
    return np.where(e <= t1, D["enc"], np.where(e <= t2, mid, top))


def table(D, h):
    """Per coder and answered code: (right + m * backoff) / (answered + m) on D."""
    C, gold = len(D["codes"]), D["gold"]
    cb = np.zeros(C, int)
    for i, (lo, hi) in enumerate(CODE_BINS):
        cb[(D["counts"] >= lo) & (D["counts"] <= hi)] = i
    T = {}
    for a in ms.ARMS:
        p = D[a]
        v = p >= 0
        n = np.bincount(p[v], minlength=C).astype(float)
        k = np.bincount(p[v & (p == gold)], minlength=C).astype(float)
        base = k.sum() / max(n.sum(), 1.0)
        if h["backoff"] == "bin":
            nb_, kb = np.bincount(cb, n, len(CODE_BINS)), np.bincount(cb, k, len(CODE_BINS))
            b = np.where(nb_ > 0, kb / np.maximum(nb_, 1.0), base)[cb]
        else:
            b = np.full(C, base)
        T[a] = (k + h["m"] * b) / (n + h["m"])
    return T


def by_table(D, T):
    arms = ms.arms_matrix(D)
    S = np.column_stack([np.where(arms[:, i] >= 0, T[a][np.maximum(arms[:, i], 0)], -1.0) for i, a in enumerate(ms.ARMS)])
    j = S.argmax(1)  # ties go to the encoder, as in the selector
    return arms[np.arange(len(j)), j]


def majority_then(D, T):
    enc, zs, rag = D["enc"], D["zs"], D["rag"]
    return np.where((enc == zs) | (enc == rag), enc, np.where((zs == rag) & (zs >= 0), zs, by_table(D, T)))


def pattern(D):
    enc, zs, rag = D["enc"], D["zs"], D["rag"]
    ez, er, zr = enc == zs, enc == rag, (zs == rag) & (zs >= 0)
    return np.select([ez & er, ez, er, zr], [0, 1, 2, 3], 4)


def by_pattern(D, assign):
    j = np.array(assign)[pattern(D)]
    out = ms.arms_matrix(D)[np.arange(len(j)), j]
    return np.where(out >= 0, out, D["enc"])


def same(D, h):
    return h


def rules(D24):
    lam = LAM_GRID if D24["probs"] is not None else [None]
    taus = [float(t) for t in np.quantile(D24["energy"], np.arange(0.05, 0.96, 0.05))]
    return {
        "energy_gate": ([{"arm": a, "tau": t} for a in ("zs", "rag") for t in taus], same,
                        lambda D, s, w=None: to_llm(D, D["energy"] > s["tau"], s["arm"])),
        "energyroute": ([{"q1": 0.4, "q2": 0.8, "lam": x} for x in lam], same, energyroute),
        "energyroute_tuned": ([{"q1": a, "q2": b, "lam": x} for a in Q_GRID for b in Q_GRID if b >= a for x in lam],
                              same, energyroute),
        "code_table": ([{"m": m, "backoff": "arm"} for m in M_GRID], table, lambda D, T, w=None: by_table(D, T)),
        "code_table_bin": ([{"m": m, "backoff": "bin"} for m in M_GRID], table, lambda D, T, w=None: by_table(D, T)),
        "majority_table": ([{"m": m, "backoff": "bin"} for m in M_GRID], table, lambda D, T, w=None: majority_then(D, T)),
        "pattern_table": ([{"assign": list(a)} for a in itertools.product(range(3), repeat=5)], same,
                          lambda D, s, w=None: by_pattern(D, s["assign"])),
        "count_conf_gate": ([{"arm": a, "k": k, "p": p} for a in ("zs", "rag") for k in [-1] + K_GRID for p in [0.0] + P_GRID],
                            same, lambda D, s, w=None: to_llm(D, (D["counts"][D["enc"]] <= s["k"]) | (D["enc_p"] < s["p"]), s["arm"])),
    }


def variants(family):
    out = {}
    for name, cols in (("refit", ms.FEATURES), ("no_counts", [f for f in ms.FEATURES if not f.startswith("log_count")]),
                       ("counts_only", ["log_count_enc", "log_count_zs", "log_count_rag", "zs_invalid", "rag_invalid"])):
        ci = [ms.FEATURES.index(c) for c in cols]
        out[name] = ([{"inputs": cols}],
                     lambda D, h, ci=ci: ms.Learned(family).fit(ms.features(D)[:, ci], ms.targets(D)),
                     lambda D, m, w=None, ci=ci: ms.choose(m.proba(ms.features(D)[:, ci]), D)[0])
    return out


def cross_validate(D, folds, rule):
    """Pooled out-of-fold macro-F1 of each setting; the best setting and its score."""
    grid, fit, apply = rule
    best, best_f = None, -1.0
    for h in grid:
        oof = np.full(len(D["gold"]), -1)
        for (tr, te), (Dtr, Dte) in folds:
            oof[te] = apply(Dte, fit(Dtr, h))
        f = ms.f1_of(D, oof)
        if f > best_f:
            best, best_f = h, f
    return best, round(best_f, 4)


# ---------------------------------------------------------------- scoring
CONTRASTS = ("energyroute", "energyroute_tuned", "energy_gate", "code_table", "code_table_bin", "majority_table",
             "pattern_table", "count_conf_gate", "no_counts", "counts_only", "cascade_msp", "best_single",
             "selector_natural")
_PRIORS = {}


def natural_setting(D, llm, frozen, models):
    """The frozen selector with each answer's odds rescaled to the code's natural prior (option A)."""
    from prior_correct import corrected, priors
    if llm not in _PRIORS:
        _PRIORS[llm] = priors(llm)
    return corrected(models[frozen["main_family"]].proba(ms.features(D)), ms.arms_matrix(D), *_PRIORS[llm])


def acc_ci(gold, a, b, samples):
    """Accuracy(a) - accuracy(b) with a paired item bootstrap."""
    d = (a == gold).astype(float) - (b == gold).astype(float)
    bs = sorted(float(d[s].mean()) for s in samples)
    return {"point": round(float(d.mean()), 4),
            "ci95": [round(bs[round(0.025 * len(bs)) - 1], 4), round(bs[round(0.975 * len(bs)) - 1], 4)]}
GATES = ("energy_gate", "energyroute", "energyroute_tuned", "count_conf_gate")


def llm_share(name, D, s, w=None):
    """Share of reports (weighted) for which a gate rule asks an LLM at all."""
    if name == "energy_gate":
        g = D["energy"] > s["tau"]
    elif name == "count_conf_gate":
        g = (D["counts"][D["enc"]] <= s["k"]) | (D["enc_p"] < s["p"])
    else:
        g = D["energy"] > er_thresholds(D, s, w)[1]
    w = np.ones(len(g)) if w is None else w
    return round(float((w * g).sum() / w.sum()), 4)


def predictions(D, llm, frozen, models, R, states, w=None):
    cz = rs.FRZ["cascade"][llm]
    P = {"selector": ms.predict_frozen(D, frozen, models)[0], "best_single": D[frozen["best_single_arm_2024"]],
         "cascade_msp": rs.cascade(D, cz["arm"], cz["theta"]), "enc": D["enc"], "zs": D["zs"], "rag": D["rag"],
         "oracle": np.where(ms.targets(D).any(axis=1), D["gold"], D["enc"]),
         "selector_natural": natural_setting(D, llm, frozen, models)}
    for name, (_, _, apply) in R.items():
        P[name] = apply(D, states[name], w)
    return P


def score_year(D, P):
    gold, labels, counts = D["gold"], D["labels"], D["counts"]
    bins = {b: np.array([c for c in labels if lo <= counts[c] <= hi]) for b, (lo, hi) in ms.BINS.items()}
    tab = {k: {"all": rs.mf(gold, p, labels), **{b: rs.mf(gold, p, bins[b]) for b in ms.BINS}} for k, p in P.items()}
    samples = resamples(len(gold))
    con = {k: rs.contrast(gold, P["selector"], P[k], labels, samples) for k in CONTRASTS}
    return {"n": int(len(gold)), "codes": int(len(labels)), "macro_f1": tab, "selector_minus": con,
            "refit_equals_frozen": round(float((P["refit"] == P["selector"]).mean()), 4), "probs_check": D["probs_check"]}


def natural_metrics(Db, Pb, Dn, Pn, wn, U):
    from rare_check import year_counts
    gold_u = Dn["gold"][U]
    labels_u = np.array(sorted(set(gold_u.tolist())))  # the codes present in the year's uniform sample
    out = {}
    for k in Pn:
        tp, fp, vis = year_counts({"D": Db}, {"D": Dn}, wn, Pb[k], Pn[k])
        out[k] = {"right_rare": round(tp, 1), "false_rare": round(fp, 1),
                  "precision": round(tp / (tp + fp), 4) if tp + fp else None, "modes_visible": vis,
                  "accuracy_uniform": round(float((Pn[k][U] == gold_u).mean()), 4),
                  "macro_f1_uniform": rs.mf(gold_u, Pn[k][U], labels_u)}
    samples = resamples(len(U))
    con = {k: rs.contrast(gold_u, Pn["selector"][U], Pn[k][U], labels_u, samples) for k in CONTRASTS + ("enc",)}
    others = ("enc", "selector", "energyroute", "energyroute_tuned", "code_table", "code_table_bin", "count_conf_gate")
    nat = {"accuracy": {k: acc_ci(gold_u, Pn["selector_natural"][U], Pn[k][U], samples) for k in others},
           "macro_f1": {k: rs.contrast(gold_u, Pn["selector_natural"][U], Pn[k][U], labels_u, samples) for k in others},
           "energyroute_minus_enc_accuracy": acc_ci(gold_u, Pn["energyroute"][U], Pn["enc"][U], samples)}
    return out, con, {"n": int(len(U)), "codes": int(len(labels_u))}, nat


def main():
    t0 = time.time()
    rep = {"note": "notes/selector-vs-rules-2026-11-11.md", "corpus": CORPUS, "selector_encoder": ms.SELECTOR_ENC,
           "git_head": ms.git_head(), "freeze": f"splits/redesign_v1_{CORPUS}.json", "llms": {}}
    for llm in LLMS:
        frozen, models = rs.check_frozen(llm)
        family = frozen["main_family"]
        D24 = load("2024", llm)
        if D24["probs"] is not None:
            D24["mids"] = {x: (x * D24["probs"] + (1.0 - x) * D24["knn"]).argmax(1) for x in LAM_GRID}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        split = list(skf.split(np.zeros((len(D24["gold"]), 1)), ms.bin_index(D24)))
        folds = [((tr, te), (sub(D24, tr), sub(D24, te))) for tr, te in split]
        R = {**rules(D24), **variants(family)}
        settings, cv, states = {}, {}, {}
        for name, rule in R.items():
            settings[name], cv[name] = cross_validate(D24, folds, rule)
            states[name] = rule[1](D24, settings[name])
            print(f"[{CORPUS} {llm}] 2024 CV {name:18s} {cv[name]:.4f}  {settings[name]}"[:200], flush=True)
        best = max((n for n in R if n not in VARIANTS), key=cv.get)
        L = {"family": family, "selector_cv_2024": frozen["cv_macro_f1"][family], "cv_2024": cv, "settings": settings,
             "best_rule_2024": best, "probs_check_2024": D24["probs_check"], "years": {}}
        print(f"[{CORPUS} {llm}] frozen selector CV {L['selector_cv_2024']}; best simple rule on 2024: {best} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del D24, folds
        for year in YEARS:
            D = load(year, llm)
            P = predictions(D, llm, frozen, models, R, states)
            Y = score_year(D, P)
            Y["llm_share"] = {g: llm_share(g, D, states[g]) for g in GATES}
            show = dict.fromkeys(["selector", best, "energyroute", "code_table", "code_table_bin", "no_counts", "best_single"])
            print(f"[{CORPUS} {llm} {year}] " + ", ".join(f"{k} {Y['macro_f1'][k]['all']}" for k in show)
                  + f"; refit==frozen {Y['refit_equals_frozen']}", flush=True)
            for k in (best, "energyroute", "code_table_bin", "no_counts"):
                c = Y["selector_minus"][k]
                print(f"   selector - {k:18s} {c['item']['point']:+.4f} item {c['item']['ci95']} code {c['code']['ci95']}",
                      flush=True)
            from maude_data import SPLIT_NATURAL, SPLIT_NATURAL_2026
            if (SPLIT_NATURAL_2026 if year == "2026" else SPLIT_NATURAL).exists():  # SDR's natural samples: 2026-11-14
                from rare_check import natural_weights
                try:
                    Dn = load(year + "nat", llm)
                except SystemExit as e:
                    Y["natural"] = {"not_run": str(e)}
                else:
                    wn, U = natural_weights({"D": Dn}, year)
                    Pn = predictions(Dn, llm, frozen, models, R, states, wn)
                    (Y["natural"], Y["natural_selector_minus"], Y["natural_uniform"],
                     Y["natural_setting_minus"]) = natural_metrics(D, P, Dn, Pn, wn, U)
                    Y["natural_llm_share"] = {g: llm_share(g, Dn, states[g], wn) for g in GATES}
                    nat = Y["natural"]
                    print("   natural: " + ", ".join(f"{k} acc {nat[k]['accuracy_uniform']} rare right/false "
                                                    f"{nat[k]['right_rare']:.0f}/{nat[k]['false_rare']:.0f}" for k in show), flush=True)
                    print(f"   natural macro-F1 ({Y['natural_uniform']}): "
                          + ", ".join(f"{k} {nat[k]['macro_f1_uniform']}" for k in list(show) + ["enc"]), flush=True)
                    for k in ("enc", best, "energyroute"):
                        c = Y["natural_selector_minus"][k]
                        print(f"   natural selector - {k:18s} {c['item']['point']:+.4f} item {c['item']['ci95']} "
                              f"code {c['code']['ci95']}", flush=True)
                    a = nat["selector_natural"]
                    print(f"   natural setting: acc {a['accuracy_uniform']} macro-F1 {a['macro_f1_uniform']} rare right/false "
                          f"{a['right_rare']:.0f}/{a['false_rare']:.0f}; balanced macro-F1 {Y['macro_f1']['selector_natural']['all']}",
                          flush=True)
                    for k in ("enc", "energyroute", "selector"):
                        c, m = Y["natural_setting_minus"]["accuracy"][k], Y["natural_setting_minus"]["macro_f1"][k]
                        print(f"   natural setting - {k:12s} accuracy {c['point']:+.4f} {c['ci95']}; macro-F1 "
                              f"{m['item']['point']:+.4f} item {m['item']['ci95']} code {m['code']['ci95']}", flush=True)
                    del Dn, Pn
            L["years"][year] = Y
            print(f"   ({time.time() - t0:.0f}s)", flush=True)
        rep["llms"][llm] = L
    out = ROOT / "results" / "selector_vs_rules"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{CORPUS}_metrics.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"wrote {out / f'{CORPUS}_metrics.json'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
