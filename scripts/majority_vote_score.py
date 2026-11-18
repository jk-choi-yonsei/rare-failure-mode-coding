"""Selector against a majority vote (notes/prereg-majority-vote-2026-11-06.md), run once per group.

Majority vote: if two or more valid answers among the encoder, the zero-shot LLM and the
retrieval LLM name the same code, take it; otherwise take the fallback.
  MV-enc  (the test)       fallback = the encoder's answer
  MV-best (descriptive)    fallback = the single coder chosen on the fitting year
                           (frozen best_single_arm_2024), or the encoder if that answer is invalid
HV, per cell: selector - MV-enc > 0, macro-F1 over every test code; item-level paired bootstrap
(B = 1000, seed 42) and the two-stage code-level interval must both clear zero.

Usage: python majority_vote_score.py maude
           -> MAUDE 2025 and 2026 cells (Opus, Qwen, Gemma) and the 2025 natural sample (descriptive)
       MAINT_CORPUS=sdr MAINT_SELECTOR_ENC=cb python majority_vote_score.py sdr
       MAINT_CORPUS=sdr MAINT_SELECTOR_ENC=cb-long python majority_vote_score.py sdr
"""
import json
import sys

import numpy as np

from common import CORPUS, ROOT, rdir
from maude_selector import SELECTOR_ENC, git_head, load_coders, load_frozen, predict_frozen
from maude_selector_score import contrast, positive, resamples, table

LLMS = ("opus55", "qwen38_27b", "gemma4_31b")


def agreed(D):
    """The code at least two valid answers share, or -1 where no two agree."""
    enc, zs, rag = D["enc"], D["zs"], D["rag"]
    zr = (zs >= 0) & (zs == rag)
    ez = (zs >= 0) & (enc == zs)
    er = (rag >= 0) & (enc == rag)
    return np.where(zr, zs, np.where(ez | er, enc, -1))


def majority(D, fallback):
    a = agreed(D)
    return np.where(a >= 0, a, fallback)


def best_fallback(D, frozen):
    best = D[frozen["best_single_arm_2024"]]
    return np.where(best >= 0, best, D["enc"])


def cell(D, frozen, models):
    sel, _ = predict_frozen(D, frozen, models)
    mv_enc = majority(D, D["enc"])
    mv_best = majority(D, best_fallback(D, frozen))
    samples = resamples(len(D["gold"]))
    hv = contrast(D, sel, mv_enc, D["labels"], samples)
    return {"n": len(D["gold"]), "codes": int(len(D["labels"])), "best_single_arm_2024": frozen["best_single_arm_2024"],
            "fallback_share": round(float((agreed(D) < 0).mean()), 4),
            "mv_enc_differs_from_encoder": round(float((mv_enc != D["enc"]).mean()), 4),
            "macro_f1": {"selector": table(D, sel), "mv_enc": table(D, mv_enc), "mv_best": table(D, mv_best),
                         "encoder": table(D, D["enc"]), "zs": table(D, D["zs"]), "rag": table(D, D["rag"])},
            "HV_selector_minus_mv_enc": {**hv, "holds": positive(hv)},
            "selector_minus_mv_best": contrast(D, sel, mv_best, D["labels"], samples)}


def show(tag, R):
    c = R["HV_selector_minus_mv_enc"]
    m = R["macro_f1"]
    print(f"{tag}: selector {m['selector']['all']} | MV-enc {m['mv_enc']['all']} | MV-best {m['mv_best']['all']} "
          f"| encoder {m['encoder']['all']} | fallback share {R['fallback_share']}")
    print(f"   HV selector - MV-enc {c['item']['point']:+.4f} item {c['item']['ci95']} code {c['code']['ci95']} "
          f"-> {'HOLDS' if c['holds'] else 'fails'}")
    d = R["selector_minus_mv_best"]
    print(f"   (descr.) selector - MV-best {d['item']['point']:+.4f} item {d['item']['ci95']} code {d['code']['ci95']}")


def natural(llm):
    """Descriptive: the 2025 natural sample, uniform-stratum accuracy and 1-5 rare-stratum recall."""
    from maude_data import SPLIT_NATURAL
    frozen, models = load_frozen(llm)
    D = load_coders("2025nat", llm)
    nat = json.loads(SPLIT_NATURAL.read_text(encoding="utf-8"))
    pos = {k: i for i, k in enumerate(D["keys"])}
    U = np.array([pos[k] for k in nat["strata"]["uniform"]])
    R = np.array([pos[k] for k in nat["strata"]["rare_1-5"]])
    sel, _ = predict_frozen(D, frozen, models)
    arms = {"selector": sel, "mv_enc": majority(D, D["enc"]), "mv_best": majority(D, best_fallback(D, frozen)),
            "encoder": D["enc"]}
    gold = D["gold"]
    return {"uniform_n": int(len(U)), "rare_1-5_n": int(len(R)),
            "uniform_accuracy": {k: round(float((p[U] == gold[U]).mean()), 4) for k, p in arms.items()},
            "rare_1-5_recall": {k: round(float((p[R] == gold[R]).mean()), 4) for k, p in arms.items()},
            "rare_1-5_correct": {k: int((p[R] == gold[R]).sum()) for k, p in arms.items()}}


def write(rep, run_name, file_name):
    out = rdir(run_name) / file_name
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    res = ROOT / "results" / rdir(run_name).name
    res.mkdir(parents=True, exist_ok=True)
    (res / file_name).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("   ->", res / file_name)


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "maude"
    head = git_head()
    if group == "maude":
        if CORPUS != "maude":
            raise SystemExit("group maude needs MAINT_CORPUS unset or maude")
        for year in ("2025", "2026"):
            rep = {"prereg": "notes/prereg-majority-vote-2026-11-06.md", "git_head_at_score": head, "year": year, "cells": {}}
            for llm in LLMS:
                frozen, models = load_frozen(llm)
                R = cell(load_coders(year, llm), frozen, models)
                rep["cells"][llm] = R
                show(f"MAUDE {year} / {llm}", R)
            write(rep, "maude" + year, f"majority_vote_{year}.json")
        rep = {"prereg": "notes/prereg-majority-vote-2026-11-06.md", "git_head_at_score": head,
               "natural_2025": {llm: natural(llm) for llm in LLMS}}
        for llm, v in rep["natural_2025"].items():
            print(f"natural 2025 / {llm}: accuracy {v['uniform_accuracy']} | 1-5 correct {v['rare_1-5_correct']}")
        write(rep, "maude2025nat", "majority_vote_natural.json")
    elif group == "sdr":
        if CORPUS != "sdr":
            raise SystemExit("group sdr needs MAINT_CORPUS=sdr")
        frozen, models = load_frozen("opus55")
        if frozen.get("encoder_tag", "cb-large") != SELECTOR_ENC:
            raise SystemExit(f"frozen selector uses {frozen.get('encoder_tag')}, MAINT_SELECTOR_ENC is {SELECTOR_ENC}")
        R = cell(load_coders("2025", "opus55"), frozen, models)
        show(f"SDR 2025 / {SELECTOR_ENC}", R)
        rep = {"prereg": "notes/prereg-majority-vote-2026-11-06.md", "git_head_at_score": head, "corpus": CORPUS,
               "encoder_tag": SELECTOR_ENC, "cell": R}
        write(rep, "maude2025", f"majority_vote_{SELECTOR_ENC}.json")
    else:
        raise SystemExit(f"unknown group {group}")


if __name__ == "__main__":
    main()
