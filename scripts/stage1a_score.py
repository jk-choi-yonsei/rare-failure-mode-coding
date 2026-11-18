"""Stage 1a scoring: encoder variants (mean±sd over seeds) vs Stage 0 arms.

Usage: python stage1a_score.py <fmc|maintie_m1>
"""
import csv
import json
import statistics
import sys

from sklearn.metrics import f1_score

from common import RUNS, SPLITS, load_fmc, load_maintie_gold
from stage0_score import normalize

SEEDS = [42, 43, 44]


def tiers_for(ds):
    if ds == "fmc":
        t = json.loads((SPLITS / "fmc_tiers.json").read_text(encoding="utf-8"))
        gold = {r["gid"]: r["label"] for r in load_fmc()}
        return gold, t["class_tiers"], sorted(t["class_counts"])
    t = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
    gold = {r["idx"]: r["label"] for r in t["records"]}
    return gold, t["class_tiers"], sorted(t["class_counts"])


def metrics(gold, preds, class_tiers, legal):
    gids = sorted(gold)
    y_true = [gold[g] for g in gids]
    y_pred = [preds.get(g, "__missing__") for g in gids]
    out = {
        "overall_acc": sum(a == b for a, b in zip(y_true, y_pred)) / len(gids),
        "overall_macro_f1": f1_score(y_true, y_pred, labels=legal,
                                     average="macro", zero_division=0),
    }
    for t in ("head", "body", "tail"):
        cls = [c for c, tt in class_tiers.items() if tt == t]
        out[f"{t}_macro_f1"] = f1_score(y_true, y_pred, labels=cls,
                                        average="macro", zero_division=0) if cls else None
    return out


def load_csv_preds(path, cast_int=True):
    with path.open(encoding="utf-8") as fh:
        return {int(r["id"]) if cast_int else r["id"]: r["pred"]
                for r in csv.DictReader(fh)}


def main():
    ds = sys.argv[1]
    gold, class_tiers, legal = tiers_for(ds)
    report = {}

    if ds == "fmc":  # Stage 0 arms on the same split
        s0 = RUNS / "stage0"
        with (s0 / "svm_fmc_preds.csv").open(encoding="utf-8") as fh:
            report["svm"] = metrics(gold, {int(r["gid"]): r["pred"]
                                           for r in csv.DictReader(fh)}, class_tiers, legal)
        for mk in ("haiku45", "sonnet5"):
            preds = {}
            for line in (s0 / f"{mk}_preds.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                lab = normalize(row.get("text", ""), legal) if row["type"] == "succeeded" else None
                preds[int(row["custom_id"][3:])] = lab or "__invalid__"
            report[f"{mk}_zeroshot"] = metrics(gold, preds, class_tiers, legal)

    for variant in ("ce", "cb"):
        per_seed = []
        for s in SEEDS:
            p = RUNS / "stage1" / f"encoder_{ds}_{variant}_seed{s}.csv"
            if p.exists():
                per_seed.append(metrics(gold, load_csv_preds(p), class_tiers, legal))
        if not per_seed:
            continue
        agg = {}
        for k in per_seed[0]:
            vals = [m[k] for m in per_seed if m[k] is not None]
            agg[k] = round(statistics.mean(vals), 4)
            agg[k + "_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        agg["n_seeds"] = len(per_seed)
        report[f"encoder_{variant}"] = agg

    out = RUNS / "stage1" / f"metrics_{ds}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
