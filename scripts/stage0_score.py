"""Stage 0 scoring and gate evaluation.

Metrics (pre-registered):
- validity: fraction of LLM outputs that normalize to a legal label
- per-class F1 on pooled predictions (632 items), aggregated as macro-F1
  overall and per tier (head/body/tail per splits/fmc_tiers.json)
- gates:
    G0-integrity: validity >= 0.95 for every LLM arm
    G0-signal:    max over LLM arms of tail macro-F1 > SVM tail macro-F1
- bootstrap (1000 resamples over items, seed 42): 95% CI for
  (best-LLM tail macro-F1 - SVM tail macro-F1), reported not gated
"""
import csv
import json
import random

from sklearn.metrics import f1_score

from common import RUNS, SEED, SPLITS, load_fmc

OUT = RUNS / "stage0"


def normalize(text, legal):
    t = " ".join(text.strip().split()).lower().rstrip(".")
    by_norm = {" ".join(l.lower().split()): l for l in legal}
    if t in by_norm:
        return by_norm[t]
    hits = [l for n, l in by_norm.items() if n in t or t in n]
    return hits[0] if len(hits) == 1 else None


def tier_macro_f1(y_true, y_pred, classes):
    if not classes:
        return None
    return f1_score(y_true, y_pred, labels=classes, average="macro",
                    zero_division=0)


def main():
    fmc = load_fmc()
    gold = {r["gid"]: r["label"] for r in fmc}
    tiers = json.loads((SPLITS / "fmc_tiers.json").read_text(encoding="utf-8"))
    legal = list(tiers["class_counts"])
    tier_classes = {t: [c for c, tt in tiers["class_tiers"].items() if tt == t]
                    for t in ("head", "body", "tail")}

    arms = {}
    with (OUT / "svm_fmc_preds.csv").open(encoding="utf-8") as fh:
        arms["svm"] = {int(r["gid"]): r["pred"] for r in csv.DictReader(fh)}
    validity = {}
    for mk in ("haiku45", "sonnet5"):
        p = OUT / f"{mk}_preds.jsonl"
        if not p.exists():
            continue
        preds, ok = {}, 0
        for line in p.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            gid = int(row["custom_id"][3:])
            lab = normalize(row.get("text", ""), legal) if row["type"] == "succeeded" else None
            if lab:
                ok += 1
            preds[gid] = lab or "__invalid__"
        arms[mk] = preds
        validity[mk] = ok / len(preds)

    gids = sorted(gold)
    y_true = [gold[g] for g in gids]
    report = {"validity": validity, "arms": {}}
    for name, preds in arms.items():
        y_pred = [preds.get(g, "__missing__") for g in gids]
        report["arms"][name] = {
            "overall_acc": sum(a == b for a, b in zip(y_true, y_pred)) / len(gids),
            "overall_macro_f1": f1_score(y_true, y_pred, labels=legal,
                                         average="macro", zero_division=0),
            **{f"{t}_macro_f1": tier_macro_f1(y_true, y_pred, tier_classes[t])
               for t in ("head", "body", "tail")},
        }

    llm_arms = [a for a in report["arms"] if a != "svm"]
    if llm_arms:
        best = max(llm_arms, key=lambda a: report["arms"][a]["tail_macro_f1"])
        svm_tail = report["arms"]["svm"]["tail_macro_f1"]
        report["gates"] = {
            "G0_integrity": all(v >= 0.95 for v in validity.values()),
            "G0_signal": report["arms"][best]["tail_macro_f1"] > svm_tail,
            "best_llm": best,
        }
        rng = random.Random(SEED)
        diffs = []
        for _ in range(1000):
            sample = [gids[rng.randrange(len(gids))] for _ in gids]
            yt = [gold[g] for g in sample]
            d = (tier_macro_f1(yt, [arms[best].get(g, "__m__") for g in sample], tier_classes["tail"])
                 - tier_macro_f1(yt, [arms["svm"].get(g, "__m__") for g in sample], tier_classes["tail"]))
            diffs.append(d)
        diffs.sort()
        report["gates"]["tail_diff_ci95"] = [diffs[24], diffs[974]]

    (OUT / "metrics.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
