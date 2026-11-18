"""Stage 1b scoring: MaintIE M1 LLM arms vs Stage 1a encoder arms."""
import csv
import json
import random

from sklearn.metrics import f1_score

from common import RUNS, SEED, SPLITS

SEEDS = [42, 43, 44]
B = 1000


def normalize_maintie(text, legal):
    """Pre-registered normalization: full-path exact -> leaf exact ->
    unique substring -> invalid."""
    t = " ".join(text.strip().split()).rstrip(".")
    by_full = {l.lower(): l for l in legal}
    if t.lower() in by_full:
        return by_full[t.lower()]
    by_leaf = {l.split("/")[-1].lower(): l for l in legal}
    if t.lower() in by_leaf:
        return by_leaf[t.lower()]
    hits = [l for l in legal if l.lower() in t.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:  # prefer the most specific (longest) path
        longest = max(hits, key=len)
        if sum(1 for h in hits if len(h) == len(longest)) == 1:
            return longest
    return None


def main():
    task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
    gold = {r["idx"]: r["label"] for r in task["records"]}
    legal = sorted(task["class_counts"])
    tier_cls = {t: [c for c, tt in task["class_tiers"].items() if tt == t]
                for t in ("head", "body", "tail")}

    arms, validity = {}, {}
    for v in ("ce", "cb"):
        seeds = []
        for s in SEEDS:
            p = RUNS / "stage1" / f"encoder_maintie_m1_{v}_seed{s}.csv"
            with p.open(encoding="utf-8") as fh:
                seeds.append({int(r["id"]): r["pred"] for r in csv.DictReader(fh)})
        arms[f"encoder_{v}"] = seeds
    for mk in ("haiku45", "sonnet5"):
        p = RUNS / "stage1b" / f"{mk}_preds.jsonl"
        if not p.exists():
            continue
        preds, ok = {}, 0
        for line in p.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            idx = int(row["custom_id"][3:])
            lab = normalize_maintie(row.get("text", ""), legal) if row["type"] == "succeeded" else None
            if lab:
                ok += 1
            preds[idx] = lab or "__invalid__"
        arms[mk] = [preds]
        validity[mk] = round(ok / len(preds), 4)

    ids = sorted(gold)

    def tier_f1(sample, arm, tier):
        yt = [gold[g] for g in sample]
        vals = [f1_score(yt, [pr.get(g, "__m__") for g in sample],
                         labels=tier_cls[tier], average="macro", zero_division=0)
                for pr in arms[arm]]
        return sum(vals) / len(vals)

    report = {"validity": validity, "arms": {}}
    yt_all = [gold[g] for g in ids]
    for name, seeds in arms.items():
        accs = [sum(gold[g] == pr.get(g) for g in ids) / len(ids) for pr in seeds]
        report["arms"][name] = {
            "overall_acc": round(sum(accs) / len(accs), 4),
            "overall_macro_f1": round(sum(
                f1_score(yt_all, [pr.get(g, "__m__") for g in ids], labels=legal,
                         average="macro", zero_division=0) for pr in seeds) / len(seeds), 4),
            **{f"{t}_macro_f1": round(tier_f1(ids, name, t), 4)
               for t in ("head", "body", "tail")},
        }

    llms = [a for a in arms if a in ("haiku45", "sonnet5")]
    if llms:
        best = max(llms, key=lambda a: report["arms"][a]["tail_macro_f1"])
        rng = random.Random(SEED)
        resamples = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(B)]
        contrasts = [
            (best, "encoder_ce", "tail", "best LLM vs fine-tuned encoder, tail"),
            (best, "encoder_cb", "tail", "best LLM vs long-tail-optimised encoder, tail"),
            ("encoder_ce", best, "head", "encoder advantage on the head"),
        ]
        if len(llms) == 2:
            contrasts.append(("haiku45", "sonnet5", "tail", "cheap vs strong LLM, tail"))
        out = {}
        for a, b, tier, desc in contrasts:
            point = tier_f1(ids, a, tier) - tier_f1(ids, b, tier)
            diffs = sorted(tier_f1(s, a, tier) - tier_f1(s, b, tier) for s in resamples)
            out[f"{a}_minus_{b}_{tier}"] = {
                "description": desc, "point": round(point, 4),
                "ci95": [round(diffs[24], 4), round(diffs[974], 4)],
                "excludes_zero": diffs[24] > 0 or diffs[974] < 0,
            }
            print(f"{desc}: {point:+.3f} 95% CI [{diffs[24]:+.3f}, {diffs[974]:+.3f}]"
                  f"{'  *' if (diffs[24] > 0 or diffs[974] < 0) else '   (spans 0)'}")
        report["contrasts"] = out
        report["best_llm"] = best

    (RUNS / "stage1b" / "metrics.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["arms"], indent=1))
    print("validity:", validity)


if __name__ == "__main__":
    main()
