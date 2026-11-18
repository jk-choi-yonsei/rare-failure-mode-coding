"""Fix the out-of-time test set: 2025 reports for the 424 known codes.

Pre-registered in notes/prereg-maude-2025-router-2026-10-08.md. Nothing from
2025 is trained on, so each known code gets up to 10 items -- min(10, n), no
half kept back -- drawn at random from data/maude/pool_2025.jsonl, which already
excludes every narrative that appears in the 2024 pool. Tiers and training
counts stay the 2024 ones: they describe what the models were trained on.

Writes splits/maude_2025_test.json. Usage: python maude_split_2025.py
"""
import json
import random
import sys
from collections import Counter, defaultdict

from common import CORPUS, DATA, SPLITS
from maude_data import class_info, load_split
from maude_split import K_TEST, SEED, sha256_file


def main():
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    pool_path = DATA / CORPUS / f"pool_{year}.jsonl"
    split = load_split()
    info = class_info(split)
    known = {c["code"] for c in split["classes"] if not c["hidden"]}
    by_code = defaultdict(list)
    for line in pool_path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["code"] in known:
            by_code[r["code"]].append(int(r["report_key"]))
    test = []
    for c in sorted(known, key=int):
        keys = sorted(by_code.get(c, []))
        picked = random.Random(f"{SEED}-test{year}-{c}").sample(keys, min(K_TEST, len(keys)))
        test += [{"key": k, "code": c} for k in sorted(picked)]
    per_code = Counter(t["code"] for t in test)
    out = {
        "version": 1,
        f"pool_{year}_sha256": sha256_file(pool_path),
        "rules": {"seed": SEED, "k_test": K_TEST, "test_per_code": "min(k_test, n_2025)",
                  "codes": "the 424 known codes of splits/maude_v1.json"},
        "counts": {
            "test_items": len(test),
            "codes_with_items": len(per_code),
            f"codes_absent_in_{year}": len(known) - len(per_code),
            "items_by_tier": dict(Counter(info[t["code"]]["tier"] for t in test)),
            "codes_by_tier": dict(Counter(info[c]["tier"] for c in per_code)),
            "codes_with_fewer_than_10": sum(1 for n in per_code.values() if n < K_TEST),
        },
        "test": test,
    }
    path = SPLITS / f"{CORPUS}_{year}_test.json"  # splits/<corpus>_<year>_test.json
    path.write_text(json.dumps(out))
    print(json.dumps(out["counts"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
