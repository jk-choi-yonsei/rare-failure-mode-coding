"""Generate pre-registered split files under splits/.

Deterministic (SEED=42). Run once before Stage 0; outputs are committed.
"""
import collections
import json

from sklearn.model_selection import StratifiedKFold

from common import (FMC_HEAD_MIN, FMC_TAIL_MAX, MAINTIE_HEAD_MIN,
                    MAINTIE_TAIL_MAX, SEED, SPLITS, load_fmc,
                    load_maintie_gold, maintie_label_m1, maintie_label_m2,
                    novel_rotations, tier_of)


def write(name, obj):
    SPLITS.mkdir(exist_ok=True)
    p = SPLITS / name
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p.name}")


def stratified_folds(labels, n=5):
    """StratifiedKFold with singleton-class fallback: classes with < n
    members are assigned round-robin by sorted index (StratifiedKFold
    cannot guarantee them)."""
    counts = collections.Counter(labels)
    big_idx = [i for i, l in enumerate(labels) if counts[l] >= n]
    small_idx = [i for i, l in enumerate(labels) if counts[l] < n]
    fold_of = {}
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)
    big_labels = [labels[i] for i in big_idx]
    for f, (_, test) in enumerate(skf.split(big_idx, big_labels)):
        for j in test:
            fold_of[big_idx[j]] = f
    for r, i in enumerate(sorted(small_idx)):
        fold_of[i] = r % n
    return [fold_of[i] for i in range(len(labels))]


def main():
    # ---------------- FMC
    fmc = load_fmc()
    assert len(fmc) == 632, f"expected 632 pooled FMC rows, got {len(fmc)}"
    labels = [r["label"] for r in fmc]
    counts = collections.Counter(labels)
    tiers = {c: tier_of(n, FMC_HEAD_MIN, FMC_TAIL_MAX) for c, n in counts.items()}
    folds = stratified_folds(labels)
    write("fmc_task.json", {
        "task": "FMC failure-mode classification (ISO 14224, 22 classes)",
        "n": len(fmc),
        "records": [{"gid": r["gid"], "origin": r["origin"], "sha1": r["sha1"],
                     "label": r["label"], "fold": folds[i]}
                    for i, r in enumerate(fmc)],
    })
    write("fmc_tiers.json", {
        "thresholds": {"head_min": FMC_HEAD_MIN, "tail_max": FMC_TAIL_MAX},
        "class_counts": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
        "class_tiers": tiers,
        "tier_sizes": dict(collections.Counter(tiers.values())),
    })
    write("fmc_novel_rotations.json", {
        "rule": "ranks [3+r, ceil(.3C)+r, ceil(.6C)+r, ceil(.85C)+r], r in 0..2",
        "rotations": novel_rotations(counts),
    })
    print("FMC tiers:", {t: sum(1 for c in tiers.values() if c == t) for t in ("head", "body", "tail")},
          "| items/tier:", {t: sum(n for c, n in counts.items() if tiers[c] == t) for t in ("head", "body", "tail")})

    # ---------------- MaintIE M1 / M2
    gold = load_maintie_gold()
    for task, fn, head_min, tail_max in (
            ("m1_first_object", maintie_label_m1, MAINTIE_HEAD_MIN, MAINTIE_TAIL_MAX),
            ("m2_failing_component", maintie_label_m2, MAINTIE_HEAD_MIN, MAINTIE_TAIL_MAX)):
        recs = [(i, fn(r)) for i, r in enumerate(gold)]
        recs = [(i, l) for i, l in recs if l]
        lab = [l for _, l in recs]
        cnt = collections.Counter(lab)
        trs = {c: tier_of(n, head_min, tail_max) for c, n in cnt.items()}
        fds = stratified_folds(lab)
        write(f"maintie_{task}.json", {
            "task": task, "n": len(recs), "classes": len(cnt),
            "thresholds": {"head_min": head_min, "tail_max": tail_max},
            "class_counts": dict(sorted(cnt.items(), key=lambda x: (-x[1], x[0]))),
            "class_tiers": trs,
            "records": [{"idx": i, "label": l, "fold": fds[j]}
                        for j, (i, l) in enumerate(recs)],
        })
        print(f"MaintIE {task}: n={len(recs)} classes={len(cnt)}",
              {t: sum(1 for c in trs.values() if c == t) for t in ("head", "body", "tail")})
    write("maintie_m1_novel_rotations.json", {
        "rule": "ranks [3+r, ceil(.3C)+r, ceil(.6C)+r, ceil(.85C)+r], r in 0..2",
        "rotations": novel_rotations(collections.Counter(
            l for l in (maintie_label_m1(r) for r in gold) if l)),
    })


if __name__ == "__main__":
    main()
