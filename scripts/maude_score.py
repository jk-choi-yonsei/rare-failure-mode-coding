"""MAUDE scoring: classification by tier, novelty AUROC, the training-size curve,
and the pre-registered contrasts (notes/prereg-maude-2026-10-07.md).

Item bootstrap, 1000 resamples, seed 42, paired (both arms read the same
resample; differences sorted afterwards), encoder seeds averaged inside each
resample -- the estimator of the earlier stages after the 736f4a7 fix.

Usage: python maude_score.py              full report -> runs/maude/metrics.json
       python maude_score.py ce_seed_sd   SD of tail macro-F1 over the CE full-N seeds
"""
import csv
import json
import random
import statistics
import sys

from sklearn.metrics import f1_score, roc_auc_score

import maude_llm as ml
from common import RUNS, SEED
from maude_data import TIERS, class_info, load_split
from maude_llm import llm_keys

OPEN_MODELS = llm_keys()  # open-weight and Claude arms

B = 1000
SIZES = ["1k", "10k", "100k", "full"]
SEEDS = [42, 43, 44]
NEARDUP_COS = 0.9
OUT = RUNS / "maude"


def load_encoder(variant, size):
    """[{key: row}] per available seed."""
    runs = []
    for s in SEEDS:
        p = OUT / f"encoder_{variant}_{size}_seed{s}.csv"
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                runs.append({int(r["key"]): r for r in csv.DictReader(fh)})
    return runs


def load_llm(kind, mk, legal, n_expected):
    p = OUT / f"llm_{kind}_{mk}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    if len(rows) < n_expected:
        print(f"   (skipping {mk}/{kind}: {len(rows)}/{n_expected} rows, run incomplete)")
        return None
    out = {}
    for r in rows:
        key = int(r["custom_id"][1:])
        text = r.get("text", "") if r["type"] == "succeeded" else ""
        if kind == "cls":
            out[key] = ml.normalize(text, legal)
        else:
            out[key] = ml.parse_novel(text, legal)
    return out


def paired_ci(diff_fn, items):
    rng = random.Random(SEED)
    res = [[items[rng.randrange(len(items))] for _ in items] for _ in range(B)]
    d = sorted(diff_fn(s) for s in res)
    lo, hi = d[round(0.025 * B) - 1], d[round(0.975 * B) - 1]  # d[24], d[974] at B = 1000
    point = diff_fn(items)
    return {"point": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": lo > 0 or hi < 0}


def main():
    split = load_split()
    info = class_info(split)
    legal = sorted(ml.known_labels(split))
    path_to_code = ml.known_labels(split)
    test = split["test"]
    known = [t["key"] for t in test if not info[t["code"]]["hidden"]]
    gold = {t["key"]: t["code"] for t in test}
    tier_of = {t["key"]: info[t["code"]]["tier"] for t in test}
    novel = {t["key"]: int(info[t["code"]]["hidden"]) for t in test}
    tested = {gold[k] for k in known}
    tier_cls = {t: sorted(c for c in tested if info[c]["tier"] == t) for t in TIERS}
    level1 = lambda code: info[code]["imdrf"][:3] if code in info else None  # noqa: E731

    # ---- classification arms: {name: [ {key: predicted code or None} per seed ]}
    arms = {}
    for variant in ("ce", "cb"):
        for size in SIZES:
            runs = load_encoder(variant, size)
            if runs:
                arms[f"enc_{variant}_{size}"] = [{k: r[k]["pred"] for k in known} for r in runs]
    llm_names, validity = [], {}
    for mk in OPEN_MODELS:
        preds = load_llm("cls", mk, legal, len(known))
        if preds is None:
            continue
        arms[mk] = [{k: path_to_code.get(preds[k]) for k in known}]
        validity[mk] = round(sum(preds[k] is not None for k in known) / len(known), 4)
        llm_names.append(mk)

    def tier_f1(sample, arm, tier):
        yt = [gold[k] for k in sample]
        vals = [f1_score(yt, [p.get(k) or "__invalid__" for k in sample], labels=tier_cls[tier],
                         average="macro", zero_division=0) for p in arms[arm]]
        return sum(vals) / len(vals)

    report = {"llm_validity_cls": validity, "arms": {}, "contrasts": {}, "curve": {},
              "novelty": {}, "seed_sd": {}}
    for name, runs in arms.items():
        accs = [sum(p.get(k) == gold[k] for k in known) / len(known) for p in runs]
        l1 = [sum(level1(p.get(k) or "") == level1(gold[k]) for k in known) / len(known) for p in runs]
        report["arms"][name] = {"n_seeds": len(runs), "acc": round(statistics.mean(accs), 4),
                                "level1_acc": round(statistics.mean(l1), 4),
                                **{f"{t}_macro_f1": round(tier_f1(known, name, t), 4) for t in TIERS}}
        if len(runs) > 1:
            per_seed = [f1_score([gold[k] for k in known], [p.get(k) or "__invalid__" for k in known],
                                 labels=tier_cls["tail"], average="macro", zero_division=0) for p in runs]
            report["seed_sd"][name] = {"tail_macro_f1_sd": round(statistics.stdev(per_seed), 4)}

    for o in llm_names:
        pairs = [("tail", o, "enc_cb_full"), ("tail", o, "enc_ce_full"), ("head", "enc_ce_full", o),
                 ("body", o, "enc_ce_full")]
        for tier, a, b in pairs:
            if a in arms and b in arms:
                report["contrasts"][f"{a}_minus_{b}_{tier}"] = paired_ci(
                    lambda s, a=a, b=b, tier=tier: tier_f1(s, a, tier) - tier_f1(s, b, tier), known)

    # ---- training-size curve and crossover brackets
    for tier in TIERS:
        enc = {s: report["arms"][f"enc_ce_{s}"][f"{tier}_macro_f1"]
               for s in SIZES if f"enc_ce_{s}" in report["arms"]}
        report["curve"][tier] = {"encoder_ce": enc}
        for o in llm_names:
            llm = report["arms"][o][f"{tier}_macro_f1"]
            first = next((s for s in SIZES if s in enc and enc[s] >= llm), None)
            report["curve"][tier][o] = {"llm": llm, "encoder_catches_up_by": first or "not by full"}

    # ---- novelty: CE full-N encoder (seeds averaged) vs each LLM's novelty score
    ce_full = load_encoder("ce", "full")
    if ce_full:
        keys = [t["key"] for t in test]
        scores = {"msp": {k: statistics.mean(float(r[k]["msp_novelty"]) for r in ce_full) for k in keys},
                  "energy": {k: statistics.mean(float(r[k]["energy_novelty"]) for r in ce_full) for k in keys}}
        diag = {}
        for mk in OPEN_MODELS:
            out = load_llm("novel", mk, legal, len(keys))
            if out is None:
                continue
            scores[mk] = {k: (out[k][1] if out[k][1] is not None else 50.0) for k in keys}
            valid = [k for k in keys if out[k][1] is not None]
            diag[mk] = {"validity": round(len(valid) / len(keys), 4),
                        **{f"none_rate_{g}": round(sum(out[k][0] == "NONE_OF_THESE" for k in valid
                                                       if novel[k] == v) /
                                                   max(1, sum(novel[k] == v for k in valid)), 4)
                           for g, v in (("known", 0), ("novel", 1))},
                        "known_acc_under_reject_prompt": round(
                            sum(path_to_code.get(out[k][0]) == gold[k] for k in known) / len(known), 4)}

        def auroc(sample, arm):
            y = [novel[k] for k in sample]
            return roc_auc_score(y, [scores[arm][k] for k in sample]) if len(set(y)) == 2 else 0.5

        subsets = {"pooled": keys, **{f"vs_known_{t}": [k for k in keys if novel[k] or tier_of[k] == t]
                                      for t in TIERS}}
        for name, items in subsets.items():
            block = {"n": len(items), "n_novel": sum(novel[k] for k in items),
                     "auroc": {a: round(auroc(items, a), 4) for a in scores}, "contrasts": {}}
            for o in (m for m in OPEN_MODELS if m in scores):
                for b in ("msp", "energy"):
                    block["contrasts"][f"{o}_minus_{b}"] = paired_ci(
                        lambda s, o=o, b=b: auroc(s, o) - auroc(s, b), items)
            report["novelty"][name] = block
        report["novelty_diagnostic"] = diag

    # ---- sensitivity: drop test items with a near-duplicate in the full training set
    nd = OUT / "neardup.json"
    if nd.exists():
        cos = {int(k): v[0] for k, v in json.loads(nd.read_text()).items()}
        clean_known = [k for k in known if cos[k] < NEARDUP_COS]
        sens = {"n_known_kept": len(clean_known), "n_known_dropped": len(known) - len(clean_known)}
        for o in llm_names:
            for b in ("enc_cb_full", "enc_ce_full"):
                if b in arms:
                    sens[f"{o}_minus_{b}_tail"] = paired_ci(
                        lambda s, o=o, b=b: tier_f1(s, o, "tail") - tier_f1(s, b, "tail"), clean_known)
        if report["novelty"]:
            items = [t["key"] for t in test if (novel[t["key"]] or tier_of[t["key"]] == "tail")
                     and cos[t["key"]] < NEARDUP_COS]
            for o in (m for m in OPEN_MODELS if m in scores):
                sens[f"{o}_minus_msp_vs_known_tail"] = paired_ci(
                    lambda s, o=o: auroc(s, o) - auroc(s, "msp"), items)
        report["sensitivity_neardup"] = sens

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1))
    show(report)


def show(r):
    print("== classification (known test items)")
    for n, m in r["arms"].items():
        print(f"   {n:14} acc {m['acc']:.3f}  L1 {m['level1_acc']:.3f}  head {m['head_macro_f1']:.3f}  "
              f"body {m['body_macro_f1']:.3f}  tail {m['tail_macro_f1']:.3f}  (seeds {m['n_seeds']})")
    for k, c in r["contrasts"].items():
        print(f"   {k:36} {c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
              f"{'  *' if c['excludes_zero'] else ''}")
    print("== curve (encoder CE macro-F1 by N; crossover vs each LLM)")
    for t, c in r["curve"].items():
        print(f"   {t}: {c}")
    for name, block in r["novelty"].items():
        print(f"== novelty {name} (n={block['n']}, novel {block['n_novel']}): "
              + "  ".join(f"{a} {v:.3f}" for a, v in block["auroc"].items()))
        for k, c in block["contrasts"].items():
            print(f"   {k:28} {c['point']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                  f"{'  *' if c['excludes_zero'] else ''}")
    if r.get("novelty_diagnostic"):
        print("== novelty diagnostic", json.dumps(r["novelty_diagnostic"]))
    if r.get("sensitivity_neardup"):
        print("== near-duplicate sensitivity", json.dumps(r["sensitivity_neardup"]))


def ce_seed_sd():
    split = load_split()
    info = class_info(split)
    known = [t["key"] for t in split["test"] if not info[t["code"]]["hidden"]]
    gold = {t["key"]: t["code"] for t in split["test"]}
    tail = sorted({gold[k] for k in known if info[gold[k]]["tier"] == "tail"})
    runs = load_encoder("ce", "full")
    vals = [f1_score([gold[k] for k in known], [r[k]["pred"] for k in known], labels=tail,
                     average="macro", zero_division=0) for r in runs]
    print(f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "nan")


if __name__ == "__main__":
    ce_seed_sd() if sys.argv[1:] == ["ce_seed_sd"] else main()
