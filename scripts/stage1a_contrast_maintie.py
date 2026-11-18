"""Item-bootstrap CIs for the Stage 1a encoder contrasts on MaintIE M1."""
import csv
import json
import random

from sklearn.metrics import f1_score

from common import RUNS, SEED, SPLITS

SEEDS = [42, 43, 44]
B = 1000


def main():
    task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
    gold = {r["idx"]: r["label"] for r in task["records"]}
    tier_cls = {t: [c for c, tt in task["class_tiers"].items() if tt == t]
                for t in ("head", "body", "tail")}

    arms = {}
    for v in ("ce", "cb"):
        seeds = []
        for s in SEEDS:
            p = RUNS / "stage1" / f"encoder_maintie_m1_{v}_seed{s}.csv"
            with p.open(encoding="utf-8") as fh:
                seeds.append({int(r["id"]): r["pred"] for r in csv.DictReader(fh)})
        arms[v] = seeds

    ids = sorted(gold)

    def tier_f1(sample, arm, tier):
        yt = [gold[g] for g in sample]
        vals = [f1_score(yt, [pr.get(g, "__m__") for g in sample],
                         labels=tier_cls[tier], average="macro", zero_division=0)
                for pr in arms[arm]]
        return sum(vals) / len(vals)

    rng = random.Random(SEED)
    resamples = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(B)]

    out = {}
    for tier, desc in (("tail", "class-balancing gain on the tail"),
                       ("head", "class-balancing cost on the head")):
        point = tier_f1(ids, "cb", tier) - tier_f1(ids, "ce", tier)
        diffs = sorted(tier_f1(s, "cb", tier) - tier_f1(s, "ce", tier) for s in resamples)
        out[f"cb_minus_ce_{tier}"] = {
            "description": desc, "point": round(point, 4),
            "ci95": [round(diffs[24], 4), round(diffs[974], 4)],
            "excludes_zero": diffs[24] > 0 or diffs[974] < 0,
        }
        print(f"{desc}: {point:+.3f} 95% CI [{diffs[24]:+.3f}, {diffs[974]:+.3f}]"
              f"{'  *' if (diffs[24] > 0 or diffs[974] < 0) else '   (spans 0)'}")

    # the asymmetry itself, within the CE arm
    h = tier_f1(ids, "ce", "head")
    t = tier_f1(ids, "ce", "tail")
    diffs = sorted(tier_f1(s, "ce", "head") - tier_f1(s, "ce", "tail") for s in resamples)
    out["ce_head_minus_tail"] = {
        "description": "head-tail gap within the fine-tuned encoder",
        "point": round(h - t, 4),
        "ci95": [round(diffs[24], 4), round(diffs[974], 4)],
        "excludes_zero": diffs[24] > 0 or diffs[974] < 0,
    }
    print(f"head-tail gap within CE: {h - t:+.3f} 95% CI [{diffs[24]:+.3f}, {diffs[974]:+.3f}]")

    (RUNS / "stage1" / "contrasts_maintie_m1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
