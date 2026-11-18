"""Item-bootstrap CIs for the key Stage 1a contrasts on FMC.

Pre-registered reporting (notes/prereg-stage1a-encoder-2026-08-18.md):
seed variance is reported alongside an item bootstrap. Encoder arms are
averaged over seeds inside each bootstrap resample, so the interval
carries item sampling error with seed averaging held as the estimator.
"""
import csv
import json
import random

from sklearn.metrics import f1_score

from common import RUNS, SEED, SPLITS, load_fmc
from stage0_score import normalize

SEEDS = [42, 43, 44]
B = 1000


def main():
    gold = {r["gid"]: r["label"] for r in load_fmc()}
    tiers = json.loads((SPLITS / "fmc_tiers.json").read_text(encoding="utf-8"))
    legal = sorted(tiers["class_counts"])
    tier_cls = {t: [c for c, tt in tiers["class_tiers"].items() if tt == t]
                for t in ("head", "body", "tail")}

    arms = {}
    with (RUNS / "stage0" / "svm_fmc_preds.csv").open(encoding="utf-8") as fh:
        arms["svm"] = [{int(r["gid"]): r["pred"] for r in csv.DictReader(fh)}]
    for mk in ("haiku45", "sonnet5"):
        preds = {}
        for line in (RUNS / "stage0" / f"{mk}_preds.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            lab = normalize(row.get("text", ""), legal) if row["type"] == "succeeded" else None
            preds[int(row["custom_id"][3:])] = lab or "__invalid__"
        arms[mk] = [preds]
    for v in ("ce", "cb"):
        seeds = []
        for s in SEEDS:
            p = RUNS / "stage1" / f"encoder_fmc_{v}_seed{s}.csv"
            with p.open(encoding="utf-8") as fh:
                seeds.append({int(r["id"]): r["pred"] for r in csv.DictReader(fh)})
        arms[f"encoder_{v}"] = seeds

    gids = sorted(gold)

    def tier_f1(sample, arm, tier):
        yt = [gold[g] for g in sample]
        vals = [f1_score(yt, [pr.get(g, "__m__") for g in sample],
                         labels=tier_cls[tier], average="macro", zero_division=0)
                for pr in arms[arm]]
        return sum(vals) / len(vals)

    contrasts = [
        ("tail", "encoder_ce", "svm", "fine-tuning gain on the tail (vs untuned SVM)"),
        ("head", "encoder_ce", "svm", "fine-tuning gain on the head (vs untuned SVM)"),
        ("tail", "encoder_cb", "encoder_ce", "class-balancing gain on the tail"),
        ("tail", "sonnet5", "encoder_cb", "best LLM vs long-tail-optimised encoder, tail"),
        ("tail", "haiku45", "encoder_cb", "cheapest LLM vs long-tail-optimised encoder, tail"),
        ("head", "encoder_cb", "sonnet5", "encoder advantage on the head"),
    ]
    rng = random.Random(SEED)
    resamples = [[gids[rng.randrange(len(gids))] for _ in gids] for _ in range(B)]

    out = {}
    for tier, a, b, desc in contrasts:
        point = tier_f1(gids, a, tier) - tier_f1(gids, b, tier)
        diffs = sorted(tier_f1(s, a, tier) - tier_f1(s, b, tier) for s in resamples)
        out[f"{a}_minus_{b}_{tier}"] = {
            "description": desc,
            "point": round(point, 4),
            "ci95": [round(diffs[24], 4), round(diffs[974], 4)],
            "excludes_zero": diffs[24] > 0 or diffs[974] < 0,
        }
        print(f"{desc}\n   {a} - {b} [{tier}] = {point:+.3f} "
              f"95% CI [{diffs[24]:+.3f}, {diffs[974]:+.3f}]"
              f"{'  *' if (diffs[24] > 0 or diffs[974] < 0) else '   (spans 0)'}")

    (RUNS / "stage1" / "contrasts_fmc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
