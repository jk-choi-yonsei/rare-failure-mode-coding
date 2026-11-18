"""Fix the 2026 never-seen-mode set (notes/prereg-redesign-2026-11-06.md, test E).

The hidden codes of splits/maude_v1.json were never trained on. Their 2026 reports in
data/maude/pool_2026.jsonl are drawn at up to K_TEST per code with the split seed, the
rule maude_split_2025.py applies to the known codes. Test E scores these reports against
the 2026 balanced test of known codes (splits/maude_2026_test.json).

Writes splits/maude_2026_novel.json. Usage: python maude_split_novel_2026.py
"""
import json
import random
from collections import Counter, defaultdict

from common import SPLITS
from maude_data import POOL_2026, class_info, load_split
from maude_split import K_TEST, SEED, sha256_file


def main():
    split = load_split()
    info = class_info(split)
    hidden = {c["code"] for c in split["classes"] if c["hidden"]}
    by_code = defaultdict(list)
    for line in POOL_2026.open(encoding="utf-8"):
        r = json.loads(line)
        if r["code"] in hidden:
            by_code[r["code"]].append(int(r["report_key"]))
    test = []
    for c in sorted(hidden, key=int):
        keys = sorted(by_code.get(c, []))
        picked = random.Random(f"{SEED}-novel2026-{c}").sample(keys, min(K_TEST, len(keys)))
        test += [{"key": k, "code": c} for k in sorted(picked)]
    per_code = Counter(t["code"] for t in test)
    out = {
        "version": 1,
        "pool_2026_sha256": sha256_file(POOL_2026),
        "rules": {"seed": f"{SEED}-novel2026-<code>", "k_test": K_TEST, "per_code": "min(k_test, n_2026)",
                  "codes": "the hidden codes of splits/maude_v1.json"},
        "counts": {"items": len(test), "hidden_codes": len(hidden), "codes_with_items": len(per_code),
                   "pool_2026_reports_of_hidden_codes": sum(len(v) for v in by_code.values()),
                   "items_by_tier": dict(Counter(info[t["code"]]["tier"] for t in test)),
                   "codes_with_fewer_than_10": sum(1 for n in per_code.values() if n < K_TEST)},
        "test": test,
    }
    path = SPLITS / "maude_2026_novel.json"
    path.write_text(json.dumps(out))
    print(json.dumps(out["counts"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
