"""Build the FAA Service Difficulty Report corpus in the MAUDE pool schema
(notes/prereg-sdr-2026-10-24.md; corpus search notes/corpus-search-2026-10-23.md).

Input: data/sdr/SDR-{2023,2024,2025,2026}.csv (76 columns; downloaded 2026-10-23 from
https://external.apic4e.faa.gov/sdrs/retrieve/). Label = PartCondition (the failure
condition of the part, e.g. CORRODED, CRACKED, INOPERATIVE), taken as filed, upper-cased;
JASC (the ATA component code) is carried along for later use. Rules, fixed before any
model run:
- drop rows whose PartCondition is one of DROP_TERMS (not a failure condition: UNKNOWN,
  NONE, NO TEST, NO DISCREPANCY, NOT REPORTED, NOT ENTERED, NOT REQUIRED) or empty, or
  whose Discrepancy narrative is empty
- exact-duplicate narratives (whitespace-collapsed, upper-cased) are kept once, the
  earliest DifficultyDate first (ties: OperatorControlNumber order), across all years
- no synonym merging (FAILED / FAULTED / MALFUNCTIONED / INOPERATIVE stay distinct)
- codes: every term in the 2023-2024 pool gets a numeric id by frequency rank (1001, ...);
  2025/2026 rows under terms absent from that pool are written with id 0 and are
  excluded from the known-code test by the split builder (they are the 'unseen' terms)
Pools: pool.jsonl = 2023 + 2024 (training years; the 2024 practice change that removed
UNKNOWN is handled by the drop rule), pool_2025.jsonl = 2025, pool_2026.jsonl = 2026 to date.
Each row: report_key (int, sequential over all years), text (Discrepancy), code, plus
year, ocn, jasc, operator, date. Also pool_codes.json (code -> term meta) and keys.json.

Usage: python sdr_data.py
"""
import csv
import json
import re
from collections import Counter, OrderedDict
from datetime import datetime

from common import DATA

SDR = DATA / "sdr"
YEARS_TRAIN = (2023, 2024)
YEARS = (2023, 2024, 2025, 2026)
DROP_TERMS = {"UNKNOWN", "NONE", "NO TEST", "NO DISCREPANCY", "NOT REPORTED", "NOT ENTERED", "NOT REQUIRED", ""}
csv.field_size_limit(10_000_000)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().upper())


def read_year(year):
    rows = []
    with (SDR / f"SDR-{year}.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            term = norm(r["PartCondition"])
            text = r["Discrepancy"].strip()
            try:
                date = datetime.strptime(r["DifficultyDate"].strip(), "%m/%d/%Y")
            except ValueError:
                date = None
            rows.append({"year": year, "ocn": r["OperatorControlNumber"].strip(), "term": term, "text": text,
                         "jasc": r["JASCCode"].strip(), "operator": r["OperatorDesignator"].strip(),
                         "date": date.strftime("%Y-%m-%d") if date else None, "_norm": norm(text)})
    return rows


def main():
    stats = OrderedDict()
    rows = []
    for y in YEARS:
        ry = read_year(y)
        stats[f"rows_{y}"] = len(ry)
        rows += ry
    # drop rules
    kept = [r for r in rows if r["term"] not in DROP_TERMS and r["text"]]
    stats["dropped_term_or_empty"] = len(rows) - len(kept)
    stats["dropped_by_term"] = dict(Counter(r["term"] for r in rows if r["term"] in DROP_TERMS).most_common())
    # exact duplicates across all years: keep the earliest
    kept.sort(key=lambda r: (r["date"] or "9999-99-99", r["ocn"]))
    seen, dedup = set(), []
    for r in kept:
        if r["_norm"] in seen:
            continue
        seen.add(r["_norm"])
        dedup.append(r)
    stats["dropped_exact_duplicates"] = len(kept) - len(dedup)
    # codes from the training years
    train_terms = Counter(r["term"] for r in dedup if r["year"] in YEARS_TRAIN)
    ranked = sorted(train_terms, key=lambda t: (-train_terms[t], t))
    code_of = {t: str(1001 + i) for i, t in enumerate(ranked)}
    codes_meta = {code_of[t]: {"term": t, "imdrf": "", "path": t, "depth": 1, "n_train_years": train_terms[t]} for t in ranked}
    # keys in a stable order: (year, date, ocn)
    dedup.sort(key=lambda r: (r["year"], r["date"] or "9999-99-99", r["ocn"]))
    keys = {}
    pools = {y: [] for y in YEARS}
    unseen = Counter()
    for i, r in enumerate(dedup, 1):
        code = code_of.get(r["term"], "0")
        if code == "0":
            unseen[r["year"]] += 1
        keys[i] = r["ocn"]
        pools[r["year"]].append({"report_key": i, "text": r["text"], "code": code, "year": r["year"], "ocn": r["ocn"],
                                 "term": r["term"], "jasc": r["jasc"], "operator": r["operator"], "date": r["date"]})
    with (SDR / "pool.jsonl").open("w", encoding="utf-8") as fh:
        for y in YEARS_TRAIN:
            for row in pools[y]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for y in (2025, 2026):
        with (SDR / f"pool_{y}.jsonl").open("w", encoding="utf-8") as fh:
            for row in pools[y]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (SDR / "pool_codes.json").write_text(json.dumps(codes_meta, indent=1, ensure_ascii=False), encoding="utf-8")
    (SDR / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    stats.update({"pool_train_rows": sum(len(pools[y]) for y in YEARS_TRAIN), "pool_2025_rows": len(pools[2025]),
                  "pool_2026_rows": len(pools[2026]), "codes": len(ranked),
                  "rows_under_unseen_terms": dict(unseen),
                  "train_terms_by_count": dict(Counter("1-5" if n <= 5 else "6-20" if n <= 20 else "21-89" if n <= 89 else "90+"
                                                       for n in train_terms.values())),
                  "top5_terms": train_terms.most_common(5)})
    (SDR / "build_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
