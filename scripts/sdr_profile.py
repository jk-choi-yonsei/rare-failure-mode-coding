"""Profile the FAA Service Difficulty Report files in data/sdr (candidate second corpus,
notes/corpus-search-2026-10-23.md). Read-only, standard library only: schema per file,
rows per difficulty year, label fill and distinct counts (PartCondition, JASCCode),
training-count-style bins, narrative length, exact-duplicate share and keyword leakage
(the PartCondition term's stem appearing in its own narrative).
Usage: python sdr_profile.py [years...]
"""
import csv
import re
import statistics
import sys
from collections import Counter

from common import DATA

SDR = DATA / "sdr"
BINS = {"1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
csv.field_size_limit(10_000_000)


def bins_of(counter):
    out = {b: 0 for b in BINS}
    for n in counter.values():
        for b, (lo, hi) in BINS.items():
            if lo <= n <= hi:
                out[b] += 1
    return out


def read(year):
    with (SDR / f"SDR-{year}.csv").open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
    return rd.fieldnames, rows


def norm(s):
    return re.sub(r"\s+", " ", s.strip().upper())


def main():
    years = [int(y) for y in sys.argv[1:]] or [2023, 2024, 2025, 2026]
    data = {}
    for y in years:
        cols, rows = read(y)
        data[y] = rows
        print(f"\n=== SDR-{y}.csv: {len(rows):,} rows x {len(cols)} cols")
        yrs = Counter((r["DifficultyDate"].strip()[-4:] or "?") for r in rows)
        print("  difficulty year:", dict(sorted(yrs.items())))
        print("  SDRType:", Counter(r["SDRType"] for r in rows).most_common(6))
        for col in ("PartCondition", "JASCCode", "Discrepancy", "PartName", "NatureOfConditionA"):
            vals = [r[col].strip() for r in rows]
            filled = sum(1 for v in vals if v) / len(vals)
            print(f"  {col:20s} filled {filled:6.1%}  distinct {len(set(v.upper() for v in vals if v)):,}")
        pc = [r["PartCondition"].strip().upper() for r in rows]
        pcc = Counter(v for v in pc if v)
        print(f"  PartCondition terms {len(pcc)}; bins {bins_of(pcc)}; top5 {pcc.most_common(5)}")
        jc = [r["JASCCode"].strip() for r in rows]
        jcc = Counter(v for v in jc if v)
        print(f"  JASC codes {len(jcc)}; bins {bins_of(jcc)}; top3 {jcc.most_common(3)}")
        txt = [r["Discrepancy"] for r in rows]
        nt = [norm(t) for t in txt]
        words = [len(t.split()) for t in txt]
        print(f"  narrative: median {int(statistics.median(words))} words, mean {statistics.mean(words):.0f}, "
              f"empty {sum(1 for t in nt if not t) / len(nt):.2%}, exact duplicates {1 - len(set(nt)) / len(nt):.1%}")
        leak = sum(1 for t, n in zip(pc, nt) if t and re.search(r"\b" + re.escape(t[:5]), n))
        print(f"  PartCondition stem (first 5 chars) appears in its own narrative: {leak / len(rows):.1%}")
        ops = Counter(r["OperatorDesignator"].strip() for r in rows)
        print(f"  top-3 operators share {sum(n for _, n in ops.most_common(3)) / len(rows):.1%}; distinct operators {len(ops):,}")
    if all(y in data for y in (2023, 2024, 2025)):
        train = Counter(r["PartCondition"].strip().upper() for y in (2023, 2024) for r in data[y])
        test = Counter(r["PartCondition"].strip().upper() for r in data[2025])
        train.pop("", None)
        test.pop("", None)
        unseen = [t for t in test if t not in train]
        print(f"\n2025 PartCondition terms unseen in 2023+2024: {len(unseen)} terms, {sum(test[t] for t in unseen)} reports; "
              f"pooled 2023+2024 rows {len(data[2023]) + len(data[2024]):,}; pooled terms {len(train)} bins {bins_of(train)}")
        jtrain = Counter(r["JASCCode"].strip() for y in (2023, 2024) for r in data[y])
        jtrain.pop("", None)
        jtest = Counter(r["JASCCode"].strip() for r in data[2025])
        jtest.pop("", None)
        junseen = [c for c in jtest if c not in jtrain]
        print(f"2023+2024 JASC codes {len(jtrain)} bins {bins_of(jtrain)}; 2025 codes unseen in 2023+2024: {len(junseen)} "
              f"({sum(jtest[c] for c in junseen)} reports)")
        # bins of the 2025 test codes by their 2024 training count (the paper's convention)
        c24 = Counter(r["PartCondition"].strip().upper() for r in data[2024])
        c24.pop("", None)
        by24 = Counter()
        for t, n in test.items():
            k = c24.get(t, 0)
            b = "0" if k == 0 else next(b for b, (lo, hi) in BINS.items() if lo <= k <= hi)
            by24[b] += 1
        print(f"2025 PartCondition terms by 2024 training count: {dict(by24)}")


if __name__ == "__main__":
    main()
