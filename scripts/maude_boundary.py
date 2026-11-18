"""CB learning curve and the per-mode training-count boundary test.

Pre-registered in notes/prereg-maude-cb-curve-2026-10-07.md, before the CB
curve runs. At each training size N every evaluated known code is binned by
its number of training examples AT THAT N (0 / 1-5 / 6-20 / 21-89 / 90+).
Macro-F1 per bin is scored over all known test items with the bin's codes as
the label set, as maude_score scores a tier, so a false positive from any
other item counts against the code.

Confirmatory test at N = 10k and 100k, for Qwen and Gemma: LLM minus CB(N)
> 0 in bin 1-5 and < 0 in bin 21-89, each 95% interval clear of zero. Bin 6-20
is reported without a predicted direction.

Usage: python maude_boundary.py  -> runs/maude/boundary.json
"""
import json
import random
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

import maude_llm as ml
from common import SEED
from maude_data import POOL, TIERS, class_info, load_split
from maude_score import B, OUT, load_encoder, load_llm
from maude_llm import llm_keys

OPEN_MODELS = llm_keys()  # open-weight and Claude arms

SIZES = ["1k", "10k", "100k", "full"]
N_OF = {"1k": 1_000, "10k": 10_000, "100k": 100_000}
BINS = [(0, 0), (1, 5), (6, 20), (21, 89), (90, 10**9)]
TEST_SIZES = ("10k", "100k")
TEST_MODELS = ("qwen38_27b", "gemma4_31b")


def bin_name(lo, hi):
    return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"


def main():
    split = load_split()
    info = class_info(split)
    legal = sorted(ml.known_labels(split))
    path_to_code = ml.known_labels(split)
    test = split["test"]
    known = [t["key"] for t in test if not info[t["code"]]["hidden"]]
    codes = sorted({c["code"] for c in split["classes"]})
    idx = {c: i for i, c in enumerate(codes)}
    y_true = np.array([idx[t["code"]] for t in test if not info[t["code"]]["hidden"]])
    evaluated = sorted({t["code"] for t in test if not info[t["code"]]["hidden"]})

    # Training-example counts per code at each N.
    code_of = {}
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        code_of[int(r["report_key"])] = r["code"]
    held = {t["key"] for t in test} | set(split["val"])
    counts = {s: Counter(code_of[k] for k in split["train_order_prefix"][:n]) for s, n in N_OF.items()}
    counts["full"] = Counter(c for k, c in code_of.items() if k not in held and not info[c]["hidden"])

    def as_pred(pred_codes):
        return np.array([idx.get(c, -1) if c else -1 for c in pred_codes])

    llm = {}
    for mk in OPEN_MODELS:
        preds = load_llm("cls", mk, legal, len(known))
        if preds is not None:
            llm[mk] = [as_pred([path_to_code.get(preds[k]) for k in known])]

    def macro_f1(preds_list, labels, sample):
        """Mean over seeds of macro-F1 restricted to `labels`, sklearn semantics
        (zero_division=0), vectorized with bincount."""
        yt = y_true[sample]
        tc = np.bincount(yt, minlength=len(codes))
        out = []
        for yp_all in preds_list:
            yp = yp_all[sample]
            ok = yp == yt
            tp = np.bincount(yt[ok], minlength=len(codes))
            pc = np.bincount(yp[yp >= 0], minlength=len(codes))
            denom = (pc + tc)[labels]
            f1 = np.where(denom > 0, 2 * tp[labels] / np.maximum(denom, 1), 0.0)
            out.append(f1.mean())
        return float(np.mean(out))

    rng = random.Random(SEED)
    n = len(known)
    resamples = [np.array([rng.randrange(n) for _ in range(n)]) for _ in range(B)]
    full_sample = np.arange(n)

    def paired(a, b, labels):
        d = sorted(macro_f1(a, labels, s) - macro_f1(b, labels, s) for s in resamples)
        lo, hi = d[round(0.025 * B) - 1], d[round(0.975 * B) - 1]
        return {"point": round(macro_f1(a, labels, full_sample) - macro_f1(b, labels, full_sample), 4),
                "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": lo > 0 or hi < 0}

    report = {"bins": {}, "cb_curve": {}, "verdict": {}}
    enc = {}
    for size in SIZES:
        for variant in ("cb", "ce"):
            runs = load_encoder(variant, size)
            if runs:
                enc[(variant, size)] = [as_pred([r[k]["pred"] for k in known]) for r in runs]

    # Cross-check the vectorized F1 against sklearn once.
    if ("cb", "full") in enc:
        labels = [idx[c] for c in evaluated if info[c]["tier"] == "tail"]
        mine = macro_f1(enc[("cb", "full")], np.array(labels), full_sample)
        ref = f1_score(y_true, enc[("cb", "full")][0], labels=labels, average="macro", zero_division=0)
        assert abs(mine - ref) < 1e-9, (mine, ref)

    for size in SIZES:
        if ("cb", size) not in enc:
            continue
        block = {}
        for lo, hi in BINS:
            bin_codes = [c for c in evaluated if lo <= counts[size][c] <= hi]
            if not bin_codes:
                continue
            labels = np.array([idx[c] for c in bin_codes])
            entry = {"n_codes": len(bin_codes),
                     "n_items": int(np.isin(y_true, labels).sum()),
                     "tiers": dict(Counter(info[c]["tier"] for c in bin_codes)),
                     "macro_f1": {}, "llm_minus_cb": {}}
            for variant in ("cb", "ce"):
                if (variant, size) in enc:
                    entry["macro_f1"][f"enc_{variant}"] = round(macro_f1(enc[(variant, size)], labels, full_sample), 4)
            for mk, p in llm.items():
                entry["macro_f1"][mk] = round(macro_f1(p, labels, full_sample), 4)
                if lo > 0:
                    entry["llm_minus_cb"][mk] = paired(p, enc[("cb", size)], labels)
            block[bin_name(lo, hi)] = entry
        report["bins"][size] = block

        tiers = {}
        for t in TIERS:
            labels = np.array([idx[c] for c in evaluated if info[c]["tier"] == t])
            tiers[t] = {"enc_cb": round(macro_f1(enc[("cb", size)], labels, full_sample), 4),
                        **{mk: round(macro_f1(p, labels, full_sample), 4) for mk, p in llm.items()}}
        report["cb_curve"][size] = tiers

    for size in TEST_SIZES:
        b = report["bins"].get(size)
        if not b or not all(m in llm for m in TEST_MODELS):
            continue
        low = all(b["1-5"]["llm_minus_cb"][m]["ci95"][0] > 0 for m in TEST_MODELS)
        high = all(b["21-89"]["llm_minus_cb"][m]["ci95"][1] < 0 for m in TEST_MODELS)
        report["verdict"][size] = {"llm_wins_1_5": low, "cb_wins_21_89": high, "boundary_holds": low and high}

    (OUT / "boundary.json").write_text(json.dumps(report, indent=1))
    for size, block in report["bins"].items():
        print(f"== N={size}")
        for name, e in block.items():
            f = "  ".join(f"{a} {v:.3f}" for a, v in e["macro_f1"].items())
            print(f"   {name:6} {e['n_codes']:3} codes {e['n_items']:4} items {str(e['tiers']):28} {f}")
            for mk, c in e["llm_minus_cb"].items():
                print(f"          {mk:11} - CB {c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
                      f"{'  *' if c['excludes_zero'] else ''}")
    print("== CB curve by tier:", json.dumps(report["cb_curve"]))
    print("== verdict:", json.dumps(report["verdict"]))


if __name__ == "__main__":
    main()
