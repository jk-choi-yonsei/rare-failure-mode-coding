"""MAUDE exploratory analyses -- NOT pre-registered; added after the
pre-registered results were in (notes/maude-results-2026-10-07.md).

1. Tail by training count. The pre-registered tail (< 100 pool items) spans
   classes with 1 to 89 training examples. MaintIE's tail had <= 4, where
   class-balancing failed; here it won. Splitting the tail by training count
   asks where along that axis the LLMs stop beating the class-balanced encoder.
2. Novelty from the class-balanced encoder. The pre-registration fixed CE as
   the novelty source; CB is the better classifier on rare classes, so its
   confidence on rare known items may separate them from novel ones better.

Usage: python maude_explore.py   -> runs/maude/explore.json
"""
import json
import statistics
import warnings
from collections import Counter

from sklearn.metrics import f1_score, roc_auc_score

# Small tail bins trip sklearn's "looks like regression" heuristic on string
# labels; the warning is spurious here and buried the output.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import maude_llm as ml
from common import RUNS
from maude_data import POOL, class_info, load_split
from maude_score import OUT, load_encoder, load_llm, paired_ci
from maude_llm import llm_keys

OPEN_MODELS = llm_keys()  # open-weight and Claude arms

BINS = [(1, 5), (6, 20), (21, 89)]


def main():
    split = load_split()
    info = class_info(split)
    legal = sorted(ml.known_labels(split))
    path_to_code = ml.known_labels(split)
    test = split["test"]
    gold = {t["key"]: t["code"] for t in test}
    known = [t["key"] for t in test if not info[t["code"]]["hidden"]]
    novel = {t["key"]: int(info[t["code"]]["hidden"]) for t in test}

    # Exact training count per code in the full-N training set.
    held = {t["key"] for t in test} | set(split["val"])
    n_train = Counter()
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        if int(r["report_key"]) not in held and not info[r["code"]]["hidden"]:
            n_train[r["code"]] += 1

    arms = {"enc_ce_full": [{k: r[k]["pred"] for k in known} for r in load_encoder("ce", "full")],
            "enc_cb_full": [{k: r[k]["pred"] for k in known} for r in load_encoder("cb", "full")]}
    for mk in OPEN_MODELS:
        preds = load_llm("cls", mk, legal, len(known))
        if preds is not None:
            arms[mk] = [{k: path_to_code.get(preds[k]) for k in known}]
    llms = [a for a in arms if a in OPEN_MODELS]

    out = {"tail_by_training_count": {}, "cb_novelty": {}}
    tail_codes = {gold[k] for k in known if info[gold[k]]["tier"] == "tail"}
    for lo, hi in BINS:
        codes = sorted(c for c in tail_codes if lo <= n_train[c] <= hi)
        items = [k for k in known if gold[k] in codes]

        # Scored over every known test item with the bin's codes as the label
        # set, as maude_score scores a tier: an item from another bin predicted
        # as one of these codes is a false positive for it. (The first version
        # scored only the bin's own items, which dropped those false positives;
        # corrected 2026-10-07 before the boundary pre-registration.)
        def f1(sample, arm, codes=codes):
            yt = [gold[k] for k in sample]
            return statistics.mean(f1_score(yt, [p.get(k) or "__invalid__" for k in sample],
                                            labels=codes, average="macro", zero_division=0)
                                   for p in arms[arm])

        block = {"n_codes": len(codes), "n_items": len(items),
                 "macro_f1": {a: round(f1(known, a), 4) for a in arms}, "contrasts": {}}
        for o in llms:
            block["contrasts"][f"{o}_minus_enc_cb_full"] = paired_ci(
                lambda s, o=o: f1(s, o) - f1(s, "enc_cb_full"), known)
        out["tail_by_training_count"][f"{lo}-{hi}"] = block

    cb = load_encoder("cb", "full")
    keys = [t["key"] for t in test]
    scores = {"cb_msp": {k: statistics.mean(float(r[k]["msp_novelty"]) for r in cb) for k in keys},
              "cb_energy": {k: statistics.mean(float(r[k]["energy_novelty"]) for r in cb) for k in keys}}
    for mk in OPEN_MODELS:
        o = load_llm("novel", mk, legal, len(keys))
        if o is not None:
            scores[mk] = {k: (o[k][1] if o[k][1] is not None else 50.0) for k in keys}

    def auroc(sample, arm):
        return roc_auc_score([novel[k] for k in sample], [scores[arm][k] for k in sample])

    for t in ("pooled", "head", "body", "tail"):
        items = keys if t == "pooled" else [k for k in keys if novel[k] or info[gold[k]]["tier"] == t]
        block = {"auroc": {a: round(auroc(items, a), 4) for a in scores}, "contrasts": {}}
        for mk in (m for m in OPEN_MODELS if m in scores):
            block["contrasts"][f"{mk}_minus_cb_msp"] = paired_ci(
                lambda s, mk=mk: auroc(s, mk) - auroc(s, "cb_msp"), items)
        out["cb_novelty"][t] = block

    (OUT / "explore.json").write_text(json.dumps(out, indent=1))
    for b, block in out["tail_by_training_count"].items():
        print(f"== tail codes with {b} training examples ({block['n_codes']} codes, {block['n_items']} items): "
              + "  ".join(f"{a} {v:.3f}" for a, v in block["macro_f1"].items()))
        for k, c in block["contrasts"].items():
            print(f"   {k:30} {c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
                  f"{'  *' if c['excludes_zero'] else ''}")
    for t, block in out["cb_novelty"].items():
        print(f"== CB-encoder novelty, {t}: " + "  ".join(f"{a} {v:.3f}" for a, v in block["auroc"].items()))
        for k, c in block["contrasts"].items():
            print(f"   {k:30} {c['point']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                  f"{'  *' if c['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
