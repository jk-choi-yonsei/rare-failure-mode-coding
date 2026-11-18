"""2024 development of the verification step (reads no 2025 file).

For each open-weight model: (1) how good the verifier is on its own: among candidate pairs
(an LLM answer naming a code with <= 20 training examples that differs from the encoder's),
the share kept when the rare code is right and when it is wrong; (2) what it does to the
selector: the model's frozen selector family refitted in the same 5 folds as v1
(out-of-fold), with and without the filter "if the selector picks such a rare answer and the
verifier prefers the encoder's code, fall back to the encoder". Scores: balanced macro-F1 and
the natural-proportion estimates of maude_selector_v2_dev.

Usage: python maude_verify_dev.py
"""
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import RUNS, SEED
from maude_selector import Learned, bin_index, choose, f1_of, features, load_coders, load_frozen, targets
from maude_selector_v2_dev import natural_scores, natural_weights
from maude_verify import RARE_MAX


def verdicts(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["type"] == "succeeded" and r["choice"] in (1, 2):
            out[(int(r["key"]), r["rare_code"])] = r["choice"] == r["rare_position"]
    return out


def main():
    for mk in ("qwen38_27b", "gemma4_31b"):
        path = RUNS / "maude_verify" / f"2024_{mk}.jsonl"
        if not path.exists():
            print(f"{mk}: no verification file yet")
            continue
        D = load_coders("2024", mk)
        V = verdicts(path)
        codes, gold, counts = D["codes"], D["gold"], D["counts"]
        # (1) the verifier alone
        keep_right, keep_wrong = [], []
        for (key, rare), keep in V.items():
            i = D["keys"].index(key)
            (keep_right if codes[gold[i]] == rare else keep_wrong).append(keep)
        print(f"\n{mk}: {len(V)} verified pairs; rare code right in {len(keep_right)}, wrong in {len(keep_wrong)}")
        print(f"  kept when the rare code is right: {np.mean(keep_right):.3f}   kept when it is wrong: {np.mean(keep_wrong):.3f}")
        # (2) inside the selector, out of fold
        frozen, _ = load_frozen(mk)
        fam = frozen["main_family"]
        X, Y = features(D), targets(D)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, bin_index(D)))
        arms = {"enc": D["enc"], "zs": D["zs"], "rag": D["rag"]}
        oof = np.full(len(gold), -1)
        for tr, te in folds:
            m = Learned(fam).fit(X[tr], Y[tr])
            oof[te], _ = choose(m.proba(X[te]), {k: v[te] for k, v in arms.items()})
        filt = oof.copy()
        missing = 0
        for i in range(len(gold)):
            p = oof[i]
            if p >= 0 and p != D["enc"][i] and counts[p] <= RARE_MAX:
                v = V.get((D["keys"][i], codes[p]))
                if v is None:
                    missing += 1
                elif not v:
                    filt[i] = D["enc"][i]
        w = natural_weights(D)
        rare_codes = np.where((counts >= 1) & (counts <= 5))[0]
        print(f"  selector family {fam}; rare overrides without a verdict: {missing}")
        print(f"  {'design':22s} {'bal mF1':>8s} {'nat acc':>8s} {'pop mF1':>8s} {'rare prec':>9s} {'FP/100k':>8s} {'rare rec':>8s} {'visible':>8s}")
        for name, pred in (("encoder", D["enc"]), ("selector v1 (oof)", oof), ("v1 + verification", filt)):
            r = natural_scores(D, pred, w, rare_codes)
            print(f"  {name:22s} {f1_of(D, pred):8.4f} {r['nat_acc']:8.4f} {r['pop_macro_f1']:8.4f} {r['rare_prec']:9.3f} "
                  f"{r['rare_fp_per_100k']:8.1f} {r['rare_recall']:8.3f} {r['rare_visible']:4d}/{r['rare_codes']}")


if __name__ == "__main__":
    main()
