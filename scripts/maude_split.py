"""Fix the MAUDE splits: balanced test set, hidden classes, validation set,
nested training subsets. Pre-registered in notes/prereg-maude-2026-10-07.md.

Rules (all deterministic from data/maude/pool.jsonl):
- codes ranked by pool count (desc), ties by numeric code; tiers by pool
  count: head >= 1000, tail < 100, body otherwise
- hidden (novel) classes: every 10th rank (10, 20, ...); all their items
  leave training, and their test items are the novel items
- test: per code, min(10, n // 2) items drawn at random (long-tailed training,
  class-balanced evaluation); a code with n = 1 is trained on, never tested
- validation: 10,000 items drawn uniformly from known non-test items
- training: the remaining known items, shuffled once; N = 1k / 10k / 100k
  are prefixes of that order, so the subsets are nested

Writes splits/maude_v1.json. Usage: python maude_split.py
"""
import hashlib
import json
import random
from collections import Counter, defaultdict

from common import CORPUS, DATA
from maude_data import POOL, SPLIT  # data/<corpus>/pool.jsonl, splits/<corpus>_v1.json

SEED = 42
K_TEST = 10
HIDE_EVERY = 10
N_VAL = 10_000
HEAD_MIN, TAIL_BELOW = 1000, 100
SUBSETS = (1_000, 10_000, 100_000)


def tier(n):
    return "head" if n >= HEAD_MIN else "tail" if n < TAIL_BELOW else "body"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    by_code = defaultdict(list)
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        by_code[r["code"]].append(int(r["report_key"]))
    codes_meta = json.loads((DATA / CORPUS / "pool_codes.json").read_text(encoding="utf-8"))

    ranked = sorted(by_code, key=lambda c: (-len(by_code[c]), int(c)))
    hidden = {c for i, c in enumerate(ranked, 1) if i % HIDE_EVERY == 0}

    test, discarded = [], 0
    rest_known = []
    for c in ranked:
        keys = sorted(by_code[c])
        t = min(K_TEST, len(keys) // 2)
        picked = set(random.Random(f"{SEED}-test-{c}").sample(keys, t))
        test += [{"key": k, "code": c} for k in sorted(picked)]
        if c in hidden:
            discarded += len(keys) - t
        else:
            rest_known += [k for k in keys if k not in picked]

    rest_known.sort()
    val = sorted(random.Random(f"{SEED}-val").sample(rest_known, N_VAL))
    val_set = set(val)
    train = [k for k in rest_known if k not in val_set]
    random.Random(f"{SEED}-train-order").shuffle(train)

    classes = []
    for i, c in enumerate(ranked, 1):
        m = codes_meta[c]
        classes.append({"code": c, "rank": i, "n_pool": len(by_code[c]),
                        "tier": tier(len(by_code[c])), "hidden": c in hidden,
                        "term": m["term"], "imdrf": m["imdrf"], "path": m["path"],
                        "depth": m["depth"]})

    code_of = {k: c for c, ks in by_code.items() for k in ks}
    n_test = Counter((tier(len(by_code[t["code"]])), t["code"] in hidden) for t in test)
    train_by_code = Counter(code_of[k] for k in train)
    split = {
        "version": 1,
        "pool_sha256": sha256_file(POOL),
        "rules": {"seed": SEED, "k_test": K_TEST, "test_per_code": "min(k_test, n // 2)",
                  "hide_every": HIDE_EVERY, "n_val": N_VAL, "head_min": HEAD_MIN,
                  "tail_below": TAIL_BELOW, "subsets": list(SUBSETS)},
        "counts": {
            "pool": sum(len(v) for v in by_code.values()), "codes": len(ranked),
            "hidden_codes": len(hidden), "known_codes": len(ranked) - len(hidden),
            "test_items": len(test), "novel_items": sum(1 for t in test if t["code"] in hidden),
            "test_known_by_tier": {t: n for (t, h), n in n_test.items() if not h},
            "test_novel_by_tier": {t: n for (t, h), n in n_test.items() if h},
            "hidden_codes_by_tier": dict(Counter(tier(len(by_code[c])) for c in hidden)),
            "val": len(val), "train_full": len(train), "hidden_discarded": discarded,
            "known_codes_untested": sum(1 for c in ranked if c not in hidden
                                        and len(by_code[c]) // 2 == 0),
            "known_codes_absent_from_train": sum(1 for c in ranked if c not in hidden
                                                 and train_by_code[c] == 0),
        },
        "train_full_sha256": hashlib.sha256(
            "\n".join(map(str, sorted(train))).encode()).hexdigest(),
        "classes": classes,
        "test": test,
        "val": val,
        "train_order_prefix": train[:max(SUBSETS)],
    }
    out = SPLIT
    out.write_text(json.dumps(split))
    print(json.dumps(split["counts"], indent=1))
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
