"""Fix the natural-distribution 2025 sample (notes/prereg-maude-natural-2026-10-23.md).

The balanced 2025 test (at most 10 reports per code) weights a rare code like a
common one. Two strata restore the real 2025 picture:
- uniform: N_UNIFORM reports drawn uniformly at random from data/maude/pool_2025.jsonl
  among reports filed under the 424 known codes, excluding the balanced test items.
  In a uniform draw the rare codes all but vanish (a 6,000 draw held 1 report of a
  code with 1-5 training examples), which is the point of the second stratum.
- rare: every 2025 report (outside the balanced test) whose code has 1-5 training
  examples in 2024, plus N_RARE_6_20 drawn at random from those with 6-20. Each rare
  report carries a weight (population / sampled) for whole-year counts.
Training counts and bins stay the 2024 ones.

Writes splits/maude_2025_natural.json. Usage: python maude_split_natural.py [2026]
With 2026 (notes/prereg-redesign-2026-11-06.md): the same rules on data/maude/pool_2026.jsonl,
excluding splits/maude_2026_test.json, seeds "natural2026-*", written to splits/maude_2026_natural.json.
MAINT_CORPUS=sdr (2026-11-14, notes/selector-vs-rules-2026-11-11.md): the same rules on the SDR pools,
written to splits/sdr_<year>_natural.json.
"""
import json
import random
import sys
from collections import Counter

from common import CORPUS, SPLITS
from maude_data import POOL_2025, POOL_2026, class_info, load_split, load_test_2025, load_test_2026
from maude_router import training_counts
from maude_split import SEED, sha256_file

N_UNIFORM = 4000
N_RARE_6_20 = 1500
BINS = {"0": (0, 0), "1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}


def bin_of(n):
    return next(b for b, (lo, hi) in BINS.items() if lo <= n <= hi)


def main():
    year = "2026" if "2026" in sys.argv[1:] else "2025"
    pool = POOL_2026 if year == "2026" else POOL_2025
    split = load_split()
    info = class_info(split)
    codes = sorted(info)
    counts = dict(zip(codes, training_counts(split, codes)["full"].tolist()))
    known = {c["code"] for c in split["classes"] if not c["hidden"]}
    held = {t["key"] for t in (load_test_2026() if year == "2026" else load_test_2025())}
    cands, n_pool, n_known = [], 0, 0
    for line in pool.open(encoding="utf-8"):
        r = json.loads(line)
        n_pool += 1
        if r["code"] in known:
            n_known += 1
            k = int(r["report_key"])
            if k not in held:
                cands.append((k, r["code"]))
    cands.sort()
    uniform = sorted(random.Random(f"{SEED}-natural{year}-uniform").sample(cands, N_UNIFORM))
    rare_1_5 = sorted((k, c) for k, c in cands if bin_of(counts[c]) == "1-5")
    pop_6_20 = sorted((k, c) for k, c in cands if bin_of(counts[c]) == "6-20")
    rare_6_20 = sorted(random.Random(f"{SEED}-natural{year}-rare620").sample(pop_6_20, min(N_RARE_6_20, len(pop_6_20))))
    weights = {"uniform": len(cands) / N_UNIFORM, "rare_1-5": 1.0, "rare_6-20": len(pop_6_20) / len(rare_6_20)}
    sample = ([{"key": k, "code": c, "stratum": "uniform"} for k, c in uniform]
              + [{"key": k, "code": c, "stratum": "rare_1-5"} for k, c in rare_1_5]
              + [{"key": k, "code": c, "stratum": "rare_6-20"} for k, c in rare_6_20])
    # a report can sit in both the uniform draw and the rare census; the union is what the models see
    seen, union = set(), []
    for t in sample:
        if t["key"] not in seen:
            seen.add(t["key"])
            union.append(t)
    overlap = len(sample) - len(union)
    by_bin = Counter(bin_of(counts[t["code"]]) for t in uniform_items(uniform))
    out = {
        "version": 2,
        f"pool_{year}_sha256": sha256_file(pool),
        "rules": {"seed": f"{SEED}-natural{year}-*", "n_uniform": N_UNIFORM, "n_rare_6_20": N_RARE_6_20,
                  "rare_1-5": "census", "population": f"pool_{year} reports under the {len(known)} known codes, "
                  f"excluding splits/{CORPUS}_{year}_test.json", "sampling": "uniform without replacement per stratum"},
        "weights": weights,
        "counts": {f"pool_{year}": n_pool, "known_code_reports": n_known, "candidates": len(cands),
                   "uniform": len(uniform), "rare_1-5": len(rare_1_5), "rare_6-20_population": len(pop_6_20),
                   "rare_6-20": len(rare_6_20), "union": len(union), "overlap_uniform_rare": overlap,
                   "uniform_items_by_2024_training_bin": dict(by_bin),
                   "uniform_codes_present": len({c for _, c in uniform}),
                   "rare_codes": {"1-5": len({c for _, c in rare_1_5}), "6-20": len({c for _, c in rare_6_20})},
                   "top5_uniform_codes": [(c, n, info[c]["path"]) for c, n in Counter(c for _, c in uniform).most_common(5)]},
        "strata": {"uniform": [k for k, _ in uniform], "rare_1-5": [k for k, _ in rare_1_5],
                   "rare_6-20": [k for k, _ in rare_6_20]},
        "sample": union,
    }
    path = SPLITS / f"{CORPUS}_{year}_natural.json"
    path.write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out["counts"].items() if k != "top5_uniform_codes"}, indent=1))
    print("weights", weights)
    print(f"wrote {path}")


def uniform_items(pairs):
    return [{"key": k, "code": c} for k, c in pairs]


if __name__ == "__main__":
    main()
