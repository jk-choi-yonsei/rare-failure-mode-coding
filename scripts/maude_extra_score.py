"""Score the follow-up experiments on the 2025 test set
(notes/prereg-maude-rag-large-2026-10-09.md).

Confirmatory, one-sided, 95% paired item bootstrap; every cell must hold, for
Qwen and Gemma both:
  H-R1  rag - zero-shot > 0, overall macro-F1, N in {10k, full}
  H-R2  rag - 1-NN > 0 and rag - kNN vote > 0, macro-F1 over the codes with at
        least one training example at N, N in {10k, full}
  H-R3  rag - CB(N) > 0 in bins 1-5 and 6-20 at full N
  H-L1  zero-shot LLM - CB-large > 0 in bin 1-5, N in {10k, 100k, full}
  H-L2  zero-shot LLM - CB-large < 0 in bin 21-89, N in {100k, full}
  H-S1, H-S2  as H-L1, H-L2 against the best supervised arm at N: the highest
        mean 2024 validation macro-F1 among CB-base, CB-large and the linear model
Each verdict is "holds", "fails" or "incomplete" (an input missing, errored or
short of its planned seeds). Edge cells (EDGE) are two-sided, each with the
outcome expected in advance. Everything is scored again without the near-
duplicate items (TF-IDF cosine >= 0.9 to a 2024 training report).

Descriptive: bins including "0" (codes with no example at N); retrieval
coverage@k; copy rates; validity and cost per LLM file; frozen routers fed the
rag answers; Qwen at 100k, gpt-oss at full and the labels-only control; CB-large
per seed with validation F1 and budget flags, and its 2024 -> 2025 drop; the
linear model.

Usage: python maude_extra_score.py  -> runs/maude2025/extra_metrics.json
"""
import csv
import json
import random

import numpy as np

import maude_llm as ml
from common import RUNS, SEED, rdir
from maude_data import load_split
from maude_router import FROZEN, SEEDS, apply, load_eval, macro_f1

B = 1000
OUT = rdir("maude2025")
MODELS = ("qwen38_27b", "gemma4_31b")  # confirmatory
ALL_MODELS = MODELS + ("gptoss20b",) + tuple(ml.CLAUDE_MODELS)  # descriptive columns
SIZES = ("10k", "100k", "full")
BINS = {"0": (0, 0), "1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
RAG_RUNS = ([(s, m) for s in ("10k", "full") for m in MODELS] + [("100k", "qwen38_27b"), ("full", "gptoss20b")]
            + [(s, m) for s in ("10k", "full") for m in ml.CLAUDE_MODELS])  # Claude: descriptive, if run
LABEL_RUNS = [("full", "qwen38_27b")]
LARGE_SEEDS = (42, 43, 44, 45, 46)  # 45, 46 replace collapsed 10k seeds, in order
LARGE_TARGET = {"10k": 3, "100k": 1, "full": 1}
COVERAGE_K = (1, 3, 5, 10)
# (contrast, N, bin, expected outcome): "+" / "-" = CI above / below zero, "0" = CI
# covers zero; two characters allow either. Scored for Qwen and Gemma.
EDGE = [
    ("rag_minus_cb", "10k", "21-89", "+"), ("rag_minus_cb", "10k", "90+", "+0"),
    ("rag_minus_cb", "full", "21-89", "-"), ("rag_minus_cb", "full", "90+", "-"),
    ("rag_minus_zero_shot", "10k", "0", "0-"), ("rag_minus_zero_shot", "10k", "1-5", "+0"),
    ("rag_minus_zero_shot", "full", "1-5", "0-"),
    ("llm_minus_large", "10k", "21-89", "0-"), ("llm_minus_large", "100k", "6-20", "-"),
    ("llm_minus_large", "full", "6-20", "+0"),
]


# ---------------------------------------------------------------- loading
def load_llm(ev, name):
    """(predictions or None, status) for runs/maude2025/{name}.jsonl; the last row per
    item counts, and predictions are None unless every item has a successful row."""
    p = OUT / f"{name}.jsonl"
    if not p.exists():
        return None, {"present": False}
    split = load_split()
    legal = sorted(ml.known_labels(split))
    path_to_code = ml.known_labels(split)
    idx = {c: i for i, c in enumerate(ev["codes"])}
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[int(r["custom_id"][1:])] = r
    pred, ok = [], []
    for k in ev["keys"]:
        r = rows.get(k)
        code = path_to_code.get(ml.normalize(r.get("text", ""), legal)) if r and r["type"] == "succeeded" else None
        pred.append(idx[code] if code else -1)
        if r and r["type"] == "succeeded":
            ok.append(r)
    missing = sum(k not in rows for k in ev["keys"])
    errored = len(ev["keys"]) - missing - len(ok)
    pred = np.array(pred)

    def q(vals, f):
        return round(float(f(vals)), 2) if vals else None
    tin = [r["usage"]["in"] for r in ok if "usage" in r]
    tout = [r["usage"]["out"] for r in ok if "usage" in r]
    sec = [r["usage"]["seconds"] for r in ok if "usage" in r and "seconds" in r["usage"]]
    status = {"present": True, "missing": missing, "errored": errored,
              "invalid": int(sum(1 for k, x in zip(ev["keys"], pred) if x < 0 and k in rows
                                 and rows[k]["type"] == "succeeded")),
              "truncated": sum(r.get("finish_reason") == "length" for r in ok),
              "tokens_in_median": q(tin, np.median), "tokens_in_p95": q(tin, lambda v: np.percentile(v, 95)),
              "tokens_out_median": q(tout, np.median), "seconds_median": q(sec, np.median),
              "seconds_p95": q(sec, lambda v: np.percentile(v, 95)), "hours": q(sec, lambda v: sum(v) / 3600)}
    status["complete"] = missing == 0 and errored == 0
    return (pred if status["complete"] else None), status


def load_neighbours(ev, size):
    """(kNN vote, 1-NN, retrieved label lists) from neighbors_{size}.json, or None."""
    p = OUT / f"neighbors_{size}.json"
    if not p.exists():
        return None
    nb = json.loads(p.read_text())
    idx = {c: i for i, c in enumerate(ev["codes"])}
    vote, nn1, lists = [], [], []
    for k in ev["keys"]:
        votes = {}
        for rank, (_, code, cos) in enumerate(nb[str(k)]):
            v = votes.setdefault(code, [0.0, -rank])  # similarity-weighted; ties go to the nearest
            v[0] += cos
        vote.append(idx[max(votes, key=lambda c: tuple(votes[c]))] if votes else -1)
        lists.append([idx[c] for _, c, _ in nb[str(k)]])
        nn1.append(lists[-1][0] if lists[-1] else -1)
    return {"knn_vote": np.array(vote), "nn1": np.array(nn1), "lists": lists}


def read_enc_csv(ev, path):
    idx = {c: i for i, c in enumerate(ev["codes"])}
    with path.open(encoding="utf-8") as fh:
        rows = {int(r["key"]): r for r in csv.DictReader(fh)}
    return (np.array([idx[rows[k]["pred"]] for k in ev["keys"]]),
            np.array([1.0 - float(rows[k]["msp_novelty"]) for k in ev["keys"]]))


def large_runs(size):
    """[(seed, meta)] of the CB-large runs that count at N (meta written, not collapsed,
    the first LARGE_TARGET in seed order) and the collapsed seeds."""
    keep, collapsed = [], []
    for s in LARGE_SEEDS:
        m = OUT / f"encoder_cb-large_{size}_seed{s}_meta.json"
        if m.exists():
            meta = json.loads(m.read_text())
            (collapsed if meta.get("collapsed") else keep).append((s, meta))
    return keep[:LARGE_TARGET[size]], [s for s, _ in collapsed]


def base_val(size):
    vals = []
    for s in SEEDS:
        m = OUT / f"encoder_cb_{size}_seed{s}_meta.json"
        if m.exists():
            vals.append(json.loads(m.read_text())["val_macro_f1"])
    return vals


def load_arms(ev):
    A = {"rag": {}, "labels": {}, "files": {}, "nb": {}, "large": {}, "linear": {}, "best": {}}
    for size, mk in RAG_RUNS:
        A["rag"][size, mk], A["files"][f"llm_rag{size}_{mk}"] = load_llm(ev, f"llm_rag{size}_{mk}")
    for size, mk in LABEL_RUNS:
        A["labels"][size, mk], A["files"][f"llm_labels{size}_{mk}"] = load_llm(ev, f"llm_labels{size}_{mk}")
    for mk in ALL_MODELS:
        A["files"][f"llm_cls_{mk}"] = load_llm(ev, f"llm_cls_{mk}")[1]
    idx = {c: i for i, c in enumerate(ev["codes"])}
    for size in SIZES:
        A["nb"][size] = load_neighbours(ev, size)
        runs, collapsed = large_runs(size)
        A["large"][size] = {"seeds": [s for s, _ in runs], "collapsed": collapsed, "metas": [m for _, m in runs],
                            "preds": [read_enc_csv(ev, OUT / f"encoder_cb-large_{size}_seed{s}.csv")[0]
                                      for s, _ in runs],
                            "complete": len(runs) == LARGE_TARGET[size]}
        lin = OUT / f"linear_{size}.csv"
        if lin.exists() and (OUT / f"linear_{size}_meta.json").exists():
            with lin.open(encoding="utf-8") as fh:
                rows = {int(r["key"]): r["pred"] for r in csv.DictReader(fh)}
            A["linear"][size] = {"pred": np.array([idx[rows[k]] for k in ev["keys"]]),
                                 "val": json.loads((OUT / f"linear_{size}_meta.json").read_text())["val_macro_f1"]}
        # Best supervised arm at N, by mean 2024 validation macro-F1; needs all three.
        val = {"cb_base": float(np.mean(base_val(size))) if base_val(size) else None,
               "cb_large": (float(np.mean([m["val_macro_f1"] for m in A["large"][size]["metas"]]))
                            if A["large"][size]["complete"] else None),
               "linear": A["linear"][size]["val"] if size in A["linear"] else None}
        best = max(val, key=lambda a: val[a]) if all(v is not None for v in val.values()) else None
        A["best"][size] = {"val": {a: round(v, 4) if v is not None else None for a, v in val.items()},
                           "best": best}
    return A


# ---------------------------------------------------------------- scoring
def resamples(n):
    rng = random.Random(SEED)
    return [np.array([rng.randrange(n) for _ in range(n)]) for _ in range(B)]


def paired(f_a, f_b, n, samples):
    d = sorted(f_a(s) - f_b(s) for s in samples)
    lo, hi = d[round(0.025 * B) - 1], d[round(0.975 * B) - 1]
    return {"point": round(f_a(np.arange(n)) - f_b(np.arange(n)), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": lo > 0 or hi < 0}


def outcome(c):
    return None if c is None else "+" if c["ci95"][0] > 0 else "-" if c["ci95"][1] < 0 else "0"


def evaluate(ev, A, keep=None):
    sel = np.arange(len(ev["gold"])) if keep is None else np.flatnonzero(keep)
    gold = ev["gold"][sel]
    n = len(gold)
    samples = resamples(n)
    whole = np.arange(n)
    present = np.array(sorted(set(gold.tolist())))

    def F(preds_list, labels):
        P = [p[sel] for p in preds_list]
        return lambda s: float(np.mean([macro_f1(gold, p, labels, s) for p in P]))

    def val(preds_list, labels):
        return round(F(preds_list, labels)(whole), 4)

    def pair(a, b, labels):
        return paired(F(a, labels), F(b, labels), n, samples)

    def bin_labels(size, lo, hi):
        cnt = ev["counts"][size]
        return np.array([c for c in present if lo <= cnt[c] <= hi])

    rep = {"n": n, "rag": {}, "large": {}, "linear": {}, "best_supervised": {}}
    for size in SIZES:
        cb = [p for p, _ in ev["enc"][size]]
        nb = A["nb"][size]
        lin = A["linear"].get(size)
        inpool = bin_labels(size, 1, 10**9)
        blk = {"overall": {"cb": val(cb, present)}, "inpool": {"n_codes": len(inpool)}, "bins": {}, "models": {}}
        if nb:
            for a in ("knn_vote", "nn1"):
                blk["overall"][a] = val([nb[a]], present)
                blk["inpool"][a] = val([nb[a]], inpool)
        if lin:
            blk["overall"]["linear"] = val([lin["pred"]], present)
        for b, (lo, hi) in BINS.items():
            labels = bin_labels(size, lo, hi)
            if len(labels) == 0:
                continue
            in_bin = np.isin(gold, labels)
            cell = {"n_codes": len(labels), "n_items": int(in_bin.sum()), "cb": val(cb, labels)}
            for mk in ALL_MODELS:
                if mk in ev["llm"]:
                    cell[f"zero_shot_{mk}"] = val([ev["llm"][mk]], labels)
            if nb:
                cell["knn_vote"] = val([nb["knn_vote"]], labels)
                cell["nn1"] = val([nb["nn1"]], labels)
                lists = [nb["lists"][i] for i in sel]
                cell["coverage"] = {k: round(float(np.mean([g in l[:k] for g, l, m in zip(gold, lists, in_bin) if m])), 4)
                                    for k in COVERAGE_K} if in_bin.any() else None
            if lin:
                cell["linear"] = val([lin["pred"]], labels)
            blk["bins"][b] = cell
        if nb:
            lists = [nb["lists"][i] for i in sel]
            blk["overall"]["coverage"] = {k: round(float(np.mean([g in l[:k] for g, l in zip(gold, lists)])), 4)
                                          for k in COVERAGE_K}
        for (s, mk), rag in A["rag"].items():
            if s != size or rag is None or mk not in ev["llm"]:
                continue
            zs = ev["llm"][mk]
            m = {"zero_shot": val([zs], present), "rag": val([rag], present),
                 "rag_minus_zero_shot": pair([rag], [zs], present), "rag_minus_cb": pair([rag], cb, present),
                 "inpool": {"rag": val([rag], inpool)}, "bins": {}}
            if nb:
                m["inpool"]["rag_minus_nn1"] = pair([rag], [nb["nn1"]], inpool)
                m["inpool"]["rag_minus_knn_vote"] = pair([rag], [nb["knn_vote"]], inpool)
            for b, (lo, hi) in BINS.items():
                labels = bin_labels(size, lo, hi)
                if len(labels):
                    m["bins"][b] = {"zero_shot": val([zs], labels), "rag": val([rag], labels),
                                    "rag_minus_zero_shot": pair([rag], [zs], labels),
                                    "rag_minus_cb": pair([rag], cb, labels),
                                    "zero_shot_minus_cb": pair([zs], cb, labels)}
            if nb:  # does the answer copy the shown labels?
                r, g = rag[sel], gold
                lists = [nb["lists"][i] for i in sel]
                inside = np.array([x in l for x, l in zip(r, lists)])
                outside = (r >= 0) & ~inside
                m["copy"] = {"same_as_nn1": round(float((r == nb["nn1"][sel]).mean()), 4),
                             "in_retrieved": round(float(inside.mean()), 4),
                             "outside_share": round(float(outside.mean()), 4),
                             "outside_correct": round(float((r[outside] == g[outside]).mean()), 4) if outside.any() else None}
            lab = A["labels"].get((size, mk))
            if lab is not None:
                m["labels_only"] = {"overall": val([lab], present),
                                    "rag_minus_labels_only": pair([rag], [lab], present),
                                    "labels_only_minus_zero_shot": pair([lab], [zs], present),
                                    "bins": {b: val([lab], bin_labels(size, lo, hi))
                                             for b, (lo, hi) in BINS.items() if len(bin_labels(size, lo, hi))}}
            spec = json.loads(FROZEN.read_text())["routers"].get(f"{size}/{mk}")
            if spec:  # routers chosen on 2024 with zero-shot answers, fed the rag answers
                m["routers"] = {}
                for fam in dict.fromkeys(("conf", spec["selected"])):
                    p = spec["families"][fam]["param"]
                    p = tuple(p) if isinstance(p, list) else p
                    outs = [apply(fam, p, e[sel], c[sel], rag[sel], ev["counts"][size]) for e, c in ev["enc"][size]]
                    m["routers"][fam] = {"macro_f1": round(float(np.mean([macro_f1(gold, o, present) for o, _ in outs])), 4),
                                         "llm_share": round(float(np.mean([c for _, c in outs])), 3)}
            blk["models"][mk] = m
        rep["rag"][size] = blk

        L = A["large"][size]
        if L["preds"]:
            lp = L["preds"]
            lb = {"seeds": L["seeds"], "collapsed_seeds": L["collapsed"], "complete": L["complete"],
                  "val_large": [m["val_macro_f1"] for m in L["metas"]], "val_base": base_val(size),
                  "budget": [{"seed": s, "lr": m["lr"], "lr_at_top_of_grid": m.get("lr_at_top_of_grid"),
                              "best_at_last_epoch": m.get("best_at_last_epoch")} for s, m in zip(L["seeds"], L["metas"])],
                  "overall": {"cb_large": val(lp, present), "cb_base": val(cb, present),
                              "per_seed_large": [val([p], present) for p in lp],
                              "per_seed_base": [val([p], present) for p in cb],
                              "large_minus_base": pair(lp, cb, present),
                              "large_minus_base_seed42": pair(lp[:1], cb[:1], present) if L["seeds"][0] == 42 else None},
                  "bins": {}}
            for b, (lo, hi) in BINS.items():
                labels = bin_labels(size, lo, hi)
                if len(labels) == 0:
                    continue
                cell = {"cb_large": val(lp, labels), "cb_base": val(cb, labels),
                        "large_minus_base": pair(lp, cb, labels)}
                for mk in ALL_MODELS:
                    if mk in ev["llm"]:
                        cell[f"{mk}_minus_large"] = pair([ev["llm"][mk]], lp, labels)
                lb["bins"][b] = cell
            rep["large"][size] = lb

        if lin:
            cell = {"val": lin["val"], "overall": val([lin["pred"]], present), "bins": {}}
            for b, (lo, hi) in BINS.items():
                labels = bin_labels(size, lo, hi)
                if len(labels):
                    cell["bins"][b] = {"linear": val([lin["pred"]], labels),
                                       **{f"{mk}_minus_linear": pair([ev["llm"][mk]], [lin["pred"]], labels)
                                          for mk in ALL_MODELS if mk in ev["llm"]}}
            rep["linear"][size] = cell

        best = A["best"][size]
        bs = {"val": best["val"], "best": best["best"], "bins": {}}
        if best["best"]:
            arm = {"cb_base": cb, "cb_large": L["preds"], "linear": [lin["pred"]] if lin else None}[best["best"]]
            for b in ("1-5", "6-20", "21-89"):
                labels = bin_labels(size, *BINS[b])
                if len(labels):
                    bs["bins"][b] = {f"{mk}_minus_best": pair([ev["llm"][mk]], arm, labels)
                                     for mk in ALL_MODELS if mk in ev["llm"]}
        rep["best_supervised"][size] = bs

    edge = []
    for contrast, size, b, expected in EDGE:
        for mk in MODELS:
            if contrast == "llm_minus_large":
                c = rep["large"].get(size, {}).get("bins", {}).get(b, {}).get(f"{mk}_minus_large")
            else:
                c = rep["rag"][size]["models"].get(mk, {}).get("bins", {}).get(b, {}).get(contrast)
            o = outcome(c)
            edge.append({"contrast": contrast, "N": size, "bin": b, "model": mk, "expected": expected,
                         "cell": c, "outcome": o, "as_expected": None if o is None else o in expected})
    rep["edge"] = edge
    rep["verdicts"] = verdicts(rep, A)
    return rep


def verdicts(rep, A):
    def judge(cells, sign):
        if any(c is None for c in cells):
            return "incomplete"
        return "holds" if all((c["ci95"][0] > 0) if sign > 0 else (c["ci95"][1] < 0) for c in cells) else "fails"

    def rag_cell(size, mk, *path):
        x = rep["rag"].get(size, {}).get("models", {}).get(mk)
        for p in path:
            x = x.get(p) if x else None
        return x

    def large_cell(size, b, mk):
        L = rep["large"].get(size)
        return L["bins"].get(b, {}).get(f"{mk}_minus_large") if L and L["complete"] else None

    def best_cell(size, b, mk):
        return rep["best_supervised"].get(size, {}).get("bins", {}).get(b, {}).get(f"{mk}_minus_best")

    R = ("10k", "full")
    return {
        "H_R1_rag_beats_zero_shot": judge([rag_cell(s, m, "rag_minus_zero_shot") for s in R for m in MODELS], 1),
        "H_R2_rag_beats_copying": judge([rag_cell(s, m, "inpool", c) for s in R for m in MODELS
                                         for c in ("rag_minus_nn1", "rag_minus_knn_vote")], 1),
        "H_R3_rag_beats_cb_1_20_at_full": judge([rag_cell("full", m, "bins", b, "rag_minus_cb")
                                                 for b in ("1-5", "6-20") for m in MODELS], 1),
        "H_L1_llm_beats_large_1_5": judge([large_cell(s, "1-5", m) for s in SIZES for m in MODELS], 1),
        "H_L2_large_beats_llm_21_89": judge([large_cell(s, "21-89", m) for s in ("100k", "full") for m in MODELS], -1),
        "H_S1_llm_beats_best_supervised_1_5": judge([best_cell(s, "1-5", m) for s in SIZES for m in MODELS], 1),
        "H_S2_best_supervised_beats_llm_21_89": judge([best_cell(s, "21-89", m) for s in ("100k", "full")
                                                       for m in MODELS], -1),
    }


def large_2024():
    """CB-large vs the retrained CB-base on the 2024 test items (2024 -> 2025 drop)."""
    ev = load_eval("2024rerun")
    out = {}
    for size in SIZES:
        runs, _ = large_runs(size)
        paths = [OUT / "rerun2024" / f"encoder_cb-large_{size}_seed{s}.csv" for s, _ in runs]
        if runs and all(p.exists() for p in paths):
            out[size] = {"cb_large": round(float(np.mean([macro_f1(ev["gold"], read_enc_csv(ev, p)[0], ev["labels"])
                                                          for p in paths])), 4),
                         "cb_base": round(float(np.mean([macro_f1(ev["gold"], p, ev["labels"])
                                                         for p, _ in ev["enc"].get(size, [])])), 4)}
    return out


def main():
    ev = load_eval("2025")
    A = load_arms(ev)
    report = {"files": A["files"], "main": evaluate(ev, A), "large_2024": large_2024()}
    nd = OUT / "neardup.json"
    if nd.exists():
        cos = json.loads(nd.read_text())
        keep = np.array([cos[str(k)][0] < 0.9 for k in ev["keys"]])
        report["neardup_excluded"] = evaluate(ev, A, keep)
        report["verdicts_changed_without_neardup"] = {
            h: [v, report["neardup_excluded"]["verdicts"][h]]
            for h, v in report["main"]["verdicts"].items() if v != report["neardup_excluded"]["verdicts"][h]}
    (OUT / "extra_metrics.json").write_text(json.dumps(report, indent=1))
    show(report)


# ---------------------------------------------------------------- printing
def fmt(c):
    return "-" if c is None else f"{c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{'*' if c['excludes_zero'] else ''}"


def show(r):
    print("===== LLM files")
    for name, st in r["files"].items():
        if st.get("present"):
            print(f"   {name:24} complete={st['complete']} errored={st['errored']} missing={st['missing']} "
                  f"invalid={st['invalid']} truncated={st['truncated']} in~{st['tokens_in_median']} "
                  f"s/item~{st['seconds_median']} ({st['hours']} h)")
    m = r["main"]
    for size, b in m["rag"].items():
        o = b["overall"]
        print(f"===== N={size}: CB {o['cb']}  kNN vote {o.get('knn_vote')}  1-NN {o.get('nn1')}  linear {o.get('linear')}"
              f"  coverage {o.get('coverage')}")
        for mk, x in b["models"].items():
            print(f"   {mk:11} zero-shot {x['zero_shot']:.3f} rag {x['rag']:.3f} | rag-zs {fmt(x['rag_minus_zero_shot'])} "
                  f"rag-CB {fmt(x['rag_minus_cb'])} | in-pool rag-1NN {fmt(x['inpool'].get('rag_minus_nn1'))} "
                  f"rag-vote {fmt(x['inpool'].get('rag_minus_knn_vote'))}")
            if "copy" in x:
                print(f"      copy: {x['copy']}  routers: {x.get('routers')}")
            if "labels_only" in x:
                lo = x["labels_only"]
                print(f"      labels-only {lo['overall']:.3f}: rag-labels {fmt(lo['rag_minus_labels_only'])} "
                      f"labels-zs {fmt(lo['labels_only_minus_zero_shot'])}")
        for name, cell in b["bins"].items():
            parts = [f"CB {cell['cb']:.3f}", f"1NN {cell.get('nn1', float('nan')):.3f}",
                     f"lin {cell.get('linear', float('nan')):.3f}", f"cov@10 {(cell.get('coverage') or {}).get(10)}"]
            for mk, x in b["models"].items():
                if name in x["bins"]:
                    y = x["bins"][name]
                    parts.append(f"{mk[:4]} zs {y['zero_shot']:.3f} rag {y['rag']:.3f} (rag-zs {fmt(y['rag_minus_zero_shot'])}, "
                                 f"rag-CB {fmt(y['rag_minus_cb'])})")
            print(f"   bin {name:5} ({cell['n_codes']} codes, {cell['n_items']} items): " + "  ".join(parts))
    for size, b in m["large"].items():
        o = b["overall"]
        print(f"===== CB-large N={size} seeds {b['seeds']} (collapsed {b['collapsed_seeds']}, complete {b['complete']}): "
              f"large {o['cb_large']:.3f} vs base {o['cb_base']:.3f} -> {fmt(o['large_minus_base'])}; seed 42 "
              f"{fmt(o['large_minus_base_seed42'])}; per seed large {o['per_seed_large']} base {o['per_seed_base']}; "
              f"val large {b['val_large']} base {b['val_base']}; budget {b['budget']}")
        for name, cell in b["bins"].items():
            print(f"   bin {name:5}: large {cell['cb_large']:.3f} base {cell['cb_base']:.3f} ({fmt(cell['large_minus_base'])})  "
                  + "  ".join(f"{mk[:4]}-large {fmt(cell.get(f'{mk}_minus_large'))}" for mk in ALL_MODELS))
    for size, b in m["linear"].items():
        print(f"===== linear N={size}: val {b['val']} test {b['overall']:.3f}  " + "  ".join(
            f"{name}: {c['linear']:.3f}" for name, c in b["bins"].items()))
    for size, b in m["best_supervised"].items():
        print(f"===== best supervised N={size}: {b['best']} (val {b['val']})  " + "  ".join(
            f"{name}: " + " ".join(f"{mk[:4]} {fmt(c)}" for mk, c in cells.items()) for name, cells in b["bins"].items()))
    print("===== edge cells")
    for e in m["edge"]:
        print(f"   {e['contrast']:20} {e['N']:4} {e['bin']:5} {e['model'][:4]} expected {e['expected']:2} -> "
              f"{e['outcome']} {fmt(e['cell'])}")
    print("2024 test, CB-large vs base:", r["large_2024"])
    print("verdicts:", json.dumps(m["verdicts"], indent=1))
    if "neardup_excluded" in r:
        print(f"without near-duplicates (n={r['neardup_excluded']['n']}): changed {r['verdicts_changed_without_neardup']}")


if __name__ == "__main__":
    main()
