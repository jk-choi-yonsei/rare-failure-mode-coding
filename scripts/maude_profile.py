"""Profile the 2024 MAUDE corpus before its pre-registration fixes anything.

One streaming pass over each zip (nothing is extracted), measuring what the
design choices hinge on: how many reports carry a Description-of-event text
(type D), how their device problem codes distribute once missing-value codes
are dropped, how much of the head is bulk filing by a few manufacturers, how
often narratives are exact duplicates, and what the tail looks like.

Writes runs/maude/profile.json. Usage: python maude_profile.py
"""
import csv
import io
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict

from common import DATA, RUNS

MAUDE = DATA / "maude"
OUT = RUNS / "maude"

# IMDRF Annex A A24-A27 plus the legacy FDA catch-alls: these record that no
# device problem was identified, not which one occurred. "-" is a blank code
# field (one report in the 2024 pool).
NON_INFORMATIVE = {"2993", "3189", "3190", "3191", "2203", "2204", "1", "-"}
OSSEO = {"3003", "1863", "2408"}  # osseointegration family (bulk dental filings)
DIGITS = re.compile(r"\d+")


def lines(zip_name, encoding):
    with zipfile.ZipFile(MAUDE / zip_name) as zf:
        with zf.open(zf.namelist()[0]) as raw:
            yield from io.TextIOWrapper(raw, encoding=encoding, errors="replace", newline="")


def load_codes():
    with zipfile.ZipFile(MAUDE / "deviceproblemcodes.zip") as zf:
        rows = csv.DictReader(io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8-sig"))
        return {r["FDA_CODE"]: {"term": r["TERM"], "imdrf": r["IMDRF_CODE"]} for r in rows}


def norm(text):
    return " ".join(text.lower().split())


def mask_digits(text):
    """Template filings vary only in dates, lot and serial numbers."""
    return DIGITS.sub("0", text)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    codes = load_codes()

    # Pass 1: narratives. Keep, per report, the count of D texts and the first one's
    # length, normalized-text hash and digit-masked hash.
    d_info, keys_any, malformed, type_counts = {}, set(), 0, Counter()
    it = lines("foitext2024.zip", "cp1252")
    next(it)  # header
    for line in it:
        parts = line.rstrip("\r\n").split("|", 5)
        if len(parts) != 6:
            malformed += 1
            continue
        key, _, ttype, _, _, text = parts
        keys_any.add(key)
        type_counts[ttype] += 1
        if ttype != "D":
            continue
        if key in d_info:
            n, *rest = d_info[key]
            d_info[key] = (n + 1, *rest)
        else:
            t = norm(text)
            d_info[key] = (1, len(t), hash(t), hash(mask_digits(t)))
    print(f"pass 1: {len(keys_any):,} reports with any text, {len(d_info):,} with a D text", flush=True)

    # Pass 2: device problem codes for those reports.
    report_codes = defaultdict(set)
    for line in lines("foidevproblem.zip", "utf-8"):
        key, _, code = line.rstrip("\r\n").partition("|")
        if key in d_info:
            report_codes[key].add(code.strip())
    print(f"pass 2: {len(report_codes):,} D-text reports carry at least one code", flush=True)

    # Pass 3: manufacturer and product code (first device record per report).
    maker, product = {}, {}
    it = lines("device2024.zip", "cp1252")
    header = next(it).rstrip("\r\n").split("|")
    i_key, i_mfr = header.index("MDR_REPORT_KEY"), header.index("MANUFACTURER_D_NAME")
    i_prod = header.index("DEVICE_REPORT_PRODUCT_CODE")
    for line in it:
        parts = line.rstrip("\r\n").split("|")
        if len(parts) <= max(i_key, i_mfr, i_prod):
            continue
        key = parts[i_key]
        if key in report_codes and key not in maker:
            maker[key] = parts[i_mfr].strip().upper()
            product[key] = parts[i_prod].strip().upper()
    print(f"pass 3: device record found for {len(maker):,} coded reports", flush=True)

    # Label policies.
    raw_single = {k: next(iter(c)) for k, c in report_codes.items() if len(c) == 1}
    informative = {k: c - NON_INFORMATIVE for k, c in report_codes.items()}
    p_strict = {k: c for k, c in raw_single.items() if c not in NON_INFORMATIVE}
    p_drop = {k: next(iter(c)) for k, c in informative.items() if len(c) == 1}
    n_coded = len(report_codes)

    def dist(policy):
        cnt = Counter(policy.values())
        bins = {">=10000": 0, "1000-9999": 0, "100-999": 0, "10-99": 0, "1-9": 0}
        for v in cnt.values():
            bins[">=10000" if v >= 10000 else "1000-9999" if v >= 1000 else
                 "100-999" if v >= 100 else "10-99" if v >= 10 else "1-9"] += 1
        return cnt, bins

    cnt_drop, bins_drop = dist(p_drop)
    no_osseo = {k: c for k, c in p_drop.items() if c not in OSSEO}
    cnt_no_osseo, bins_no_osseo = dist(no_osseo)

    # Bulk filing: how concentrated each code is in one manufacturer.
    by_code_mfr = defaultdict(Counter)
    for k, c in p_drop.items():
        by_code_mfr[c][maker.get(k, "?")] += 1
    top_codes = []
    for c, n in cnt_drop.most_common(40):
        mfr, m = by_code_mfr[c].most_common(1)[0]
        top_codes.append({"code": c, "term": codes.get(c, {}).get("term"),
                          "imdrf": codes.get(c, {}).get("imdrf"), "n": n,
                          "top_manufacturer": mfr, "top_mfr_share": round(m / n, 3)})

    # Exact-duplicate narratives among single-label reports.
    hashes = Counter(d_info[k][2] for k in p_drop)
    dup_reports = sum(n for n in hashes.values() if n > 1)
    lengths = [d_info[k][1] for k in p_drop]

    # Deduplicated pool: one item per digit-masked text; a text filed under
    # more than one code is a label conflict and is dropped.
    groups = defaultdict(Counter)
    for k, c in p_drop.items():
        groups[d_info[k][3]][c] += 1
    conflicted = [g for g in groups.values() if len(g) > 1]
    dedup = Counter(next(iter(g)) for g in groups.values() if len(g) == 1)
    _, bins_dedup = dist({i: c for i, c in enumerate(dedup.elements())})
    dedup_no_osseo = sum(n for c, n in dedup.items() if c not in OSSEO)

    legacy = sum(1 for c in p_drop.values() if not codes.get(c, {}).get("imdrf"))
    report = {
        "foitext_type_counts": dict(type_counts), "foitext_malformed_lines": malformed,
        "reports_any_text": len(keys_any), "reports_with_D": len(d_info),
        "reports_with_D_and_code": n_coded,
        "D_texts_per_report": dict(Counter(min(v[0], 5) for v in d_info.values())),
        "codes_per_report": dict(Counter(min(len(c), 5) for c in report_codes.values())),
        "policy_strict_single": {"n": len(p_strict), "share": round(len(p_strict) / n_coded, 4)},
        "policy_drop_noninformative_single": {
            "n": len(p_drop), "share": round(len(p_drop) / n_coded, 4),
            "n_classes": len(cnt_drop), "count_bins": bins_drop},
        "only_noninformative_codes": sum(1 for c in informative.values() if not c),
        "osseo_share_of_policy_drop": round(sum(1 for c in p_drop.values() if c in OSSEO) / len(p_drop), 4),
        "policy_drop_without_osseo": {"n": len(no_osseo), "n_classes": len(cnt_no_osseo),
                                      "count_bins": bins_no_osseo},
        "legacy_non_imdrf_share": round(legacy / len(p_drop), 4),
        "D_length_chars": {"mean": round(statistics.mean(lengths), 1),
                           "median": statistics.median(lengths),
                           "p05": pct(lengths, 0.05), "p95": pct(lengths, 0.95)},
        "exact_duplicate_text_share": round(dup_reports / len(p_drop), 4),
        "largest_duplicate_groups": sorted(hashes.values(), reverse=True)[:10],
        "dedup": {
            "unique_exact_texts": len(hashes),
            "unique_masked_texts": len(groups),
            "conflicting_texts": len(conflicted),
            "reports_in_conflicting_texts": sum(sum(g.values()) for g in conflicted),
            "items": sum(dedup.values()), "n_classes": len(dedup),
            "items_without_osseo": dedup_no_osseo,
            "osseo_share": round(1 - dedup_no_osseo / sum(dedup.values()), 4),
            "count_bins": bins_dedup,
            "top_codes": [{"code": c, "term": codes.get(c, {}).get("term"), "n": n,
                           "n_before_dedup": cnt_drop[c]} for c, n in dedup.most_common(30)],
            "rank_counts": [n for _, n in dedup.most_common()],
        },
        "distinct_manufacturers": len(set(maker[k] for k in p_drop if k in maker)),
        "distinct_product_codes": len(set(product[k] for k in p_drop if k in product)),
        "top_codes": top_codes,
        "tail_codes_sample": [{"code": c, "term": codes.get(c, {}).get("term"), "n": n}
                              for c, n in cnt_drop.most_common()[-15:]],
        "rank_counts": [n for _, n in cnt_drop.most_common()],
    }
    (OUT / "profile.json").write_text(json.dumps(report, indent=1))
    shown = {k: v for k, v in report.items() if k != "rank_counts"}
    shown["dedup"] = {k: v for k, v in report["dedup"].items() if k != "rank_counts"}
    print(json.dumps(shown, indent=1))


if __name__ == "__main__":
    main()
