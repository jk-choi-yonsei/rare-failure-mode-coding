"""Encoder -> LLM routers for MAUDE classification, and the shared loaders the
out-of-time scorer uses.

Pre-registered in notes/prereg-maude-2025-router-2026-10-08.md. Each router
answers every item with the class-balanced encoder CB(N) unless its rule sends
the item to an LLM; an invalid LLM answer falls back to the encoder.

  count(k)          send it when the encoder predicts a code with <= k
                    training examples at N
  conf(p)           send it when the encoder's top softmax probability < p
  count_or_conf     either condition
  llm_rare(k)       ask the LLM on every item; keep its answer when it names a
                    code with <= k training examples (cost: every item)

`choose` picks, for each N and LLM, the best parameter of each family and the
best router overall by macro-F1 on the 2024 test set (original CB models), and
freezes them in splits/maude_router_v1.json before any 2025 output exists.

Usage: python maude_router.py choose
"""
import csv
import json
import sys
from collections import Counter

import numpy as np

import maude_llm as ml
from common import RUNS, SPLITS, rdir
from maude_data import POOL, load_split, load_test_2025, load_test_2026
from maude_llm import llm_keys

SIZES = ["1k", "10k", "100k", "full"]
N_OF = {"1k": 1_000, "10k": 10_000, "100k": 100_000}
SEEDS = [42, 43, 44]
K_GRID = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
P_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
FROZEN = SPLITS / "maude_router_v1.json"


def training_counts(split, codes):
    """{size: array of training examples per code index} for the 2024 training sets."""
    info = {c["code"]: c for c in split["classes"]}
    code_of = {}
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        code_of[int(r["report_key"])] = r["code"]
    held = {t["key"] for t in split["test"]} | set(split["val"])
    raw = {s: Counter(code_of[k] for k in split["train_order_prefix"][:n]) for s, n in N_OF.items()}
    raw["full"] = Counter(c for k, c in code_of.items() if k not in held and not info[c]["hidden"])
    return {s: np.array([raw[s][c] for c in codes]) for s in SIZES}


def load_eval(which):
    """Arrays for one evaluation set, in a fixed item order.

    which: "2024"      known 2024 test items, original CB models, 2024 LLM outputs
           "2025"      2025 test items, retrained CB models, 2025 LLM outputs
           "2024rerun" known 2024 test items, retrained CB models (reproducibility)
    """
    split = load_split()
    info = {c["code"]: c for c in split["classes"]}
    codes = sorted(info)
    idx = {c: i for i, c in enumerate(codes)}
    if which == "2025":
        test, enc_dir, llm_dir = load_test_2025(), rdir("maude2025"), rdir("maude2025")
    elif which == "2026":  # notes/prereg-maude-2026-2026-11-05.md
        test, enc_dir, llm_dir = load_test_2026(), rdir("maude2026"), rdir("maude2026")
    else:
        test = [t for t in split["test"] if not info[t["code"]]["hidden"]]
        enc_dir = rdir("maude2025") / "rerun2024" if which == "2024rerun" else rdir("maude")
        llm_dir = rdir("maude")
    keys = [t["key"] for t in test]
    gold = np.array([idx[t["code"]] for t in test])

    enc = {}
    for size in SIZES:
        runs = []
        for s in SEEDS:
            p = enc_dir / f"encoder_cb_{size}_seed{s}.csv"
            if p.exists():
                with p.open(encoding="utf-8") as fh:
                    rows = {int(r["key"]): r for r in csv.DictReader(fh)}
                runs.append((np.array([idx[rows[k]["pred"]] for k in keys]),
                             np.array([1.0 - float(rows[k]["msp_novelty"]) for k in keys])))
        if runs:
            enc[size] = runs

    legal = sorted(ml.known_labels(split))
    path_to_code = ml.known_labels(split)
    llm = {}
    for mk in llm_keys():
        p = llm_dir / f"llm_cls_{mk}.jsonl"
        if not p.exists():
            continue
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            code = path_to_code.get(ml.normalize(r.get("text", ""), legal)) if r["type"] == "succeeded" else None
            out[int(r["custom_id"][1:])] = idx[code] if code else -1
        if all(k in out for k in keys):
            llm[mk] = np.array([out[k] for k in keys])
    labels = np.array(sorted(set(gold.tolist())))
    return {"keys": keys, "gold": gold, "codes": codes, "info": info, "labels": labels,
            "enc": enc, "llm": llm, "counts": training_counts(split, codes)}


def macro_f1(gold, pred, labels, sample=None):
    """sklearn macro-F1 restricted to `labels` (zero_division=0), vectorized."""
    if sample is not None:
        gold, pred = gold[sample], pred[sample]
    n = int(max(gold.max(), pred.max(), labels.max())) + 1
    tc = np.bincount(gold, minlength=n)
    tp = np.bincount(gold[pred == gold], minlength=n)
    pc = np.bincount(pred[pred >= 0], minlength=n)
    denom = (pc + tc)[labels]
    return float(np.where(denom > 0, 2 * tp[labels] / np.maximum(denom, 1), 0.0).mean())


def apply(rule, param, enc_pred, enc_p, llm_pred, counts):
    """(predictions, share of items the LLM is asked about)."""
    if rule == "enc":
        return enc_pred, 0.0
    if rule == "llm":
        return llm_pred, 1.0
    if rule == "llm_rare":
        use = (llm_pred >= 0) & (counts[np.maximum(llm_pred, 0)] <= param)
        return np.where(use, llm_pred, enc_pred), 1.0
    if rule == "count":
        route = counts[enc_pred] <= param
    elif rule == "conf":
        route = enc_p < param
    elif rule == "count_or_conf":
        k, p = param
        route = (counts[enc_pred] <= k) | (enc_p < p)
    else:
        raise ValueError(rule)
    use = route & (llm_pred >= 0)  # an invalid LLM answer falls back to the encoder
    return np.where(use, llm_pred, enc_pred), float(route.mean())


def candidates():
    yield "count", K_GRID
    yield "conf", P_GRID
    yield "count_or_conf", [(k, p) for k in K_GRID for p in P_GRID]
    yield "llm_rare", K_GRID


def score(ev, size, mk, rule, param, sample=None):
    """Seed-averaged (macro-F1, cost) of one router on one evaluation set."""
    f1s, costs = [], []
    for enc_pred, enc_p in ev["enc"][size]:
        pred, cost = apply(rule, param, enc_pred, enc_p, ev["llm"][mk], ev["counts"][size])
        f1s.append(macro_f1(ev["gold"], pred, ev["labels"], sample))
        costs.append(cost)
    return float(np.mean(f1s)), float(np.mean(costs))


def choose():
    ev = load_eval("2024")
    frozen = {"selected_on": "2024 known test items, original CB models (runs/maude)",
              "objective": "macro-F1 over known codes, CB seeds averaged",
              "k_grid": K_GRID, "p_grid": P_GRID, "routers": {}}
    for size in SIZES:
        for mk in ev["llm"]:
            fam = {}
            for rule, grid in candidates():
                best = max(grid, key=lambda p: score(ev, size, mk, rule, p)[0])
                f1, cost = score(ev, size, mk, rule, best)
                fam[rule] = {"param": best, "dev_macro_f1": round(f1, 4), "dev_cost": round(cost, 3)}
            sel = max(fam, key=lambda r: fam[r]["dev_macro_f1"])
            frozen["routers"][f"{size}/{mk}"] = {
                "families": fam, "selected": sel,
                "dev_enc_only": round(score(ev, size, mk, "enc", None)[0], 4),
                "dev_llm_only": round(score(ev, size, mk, "llm", None)[0], 4)}
            r = frozen["routers"][f"{size}/{mk}"]
            print(f"N={size:4} {mk:11} enc {r['dev_enc_only']:.3f}  llm {r['dev_llm_only']:.3f}  "
                  + "  ".join(f"{k} {v['param']} -> {v['dev_macro_f1']:.3f} (cost {v['dev_cost']:.2f})"
                              for k, v in fam.items()) + f"  | selected {sel}")
    FROZEN.write_text(json.dumps(frozen, indent=1))
    print(f"wrote {FROZEN}")


if __name__ == "__main__":
    if sys.argv[1:] == ["choose"]:
        choose()
    else:
        sys.exit(__doc__)
