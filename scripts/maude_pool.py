"""Build the deduplicated MAUDE 2024 item pool that every later split draws from.

Rules, as measured in notes/maude-profile-2026-10-06.md:
- text: the first Description-of-event (type D) narrative of each report
- label: the report's device problem codes minus the missing-value markers
  (maude_profile.NON_INFORMATIVE); kept only if exactly one code remains
- dedup: one item per case/whitespace-normalized, digit-masked narrative,
  represented by its smallest MDR report key; a narrative filed under more
  than one code is a label conflict and is dropped

Writes data/maude/pool.jsonl (one item per line) and data/maude/pool_codes.json
(per-code count, term, IMDRF code and path).

For a later year (the out-of-time test, notes/prereg-maude-2025-router-2026-10-08.md)
the same rules run on that year's narratives, and one more applies: a narrative
whose digit-masked form appears anywhere in the 2024 pool is dropped, so no
2025 item repeats a text the 2024 models could have trained on. The device file
is optional there; its fields are descriptive only.

Usage: python maude_pool.py          2024 -> pool.jsonl, pool_codes.json
       python maude_pool.py 2025     2025 -> pool_2025.jsonl, pool_codes_2025.json
       python maude_pool.py 2026     2026 to date (foitext2026.zip = FDA's current-year file) -> pool_2026.jsonl
"""
import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict

from maude_profile import MAUDE, NON_INFORMATIVE, lines, mask_digits, norm


def imdrf_paths():
    """FDA code -> (term, IMDRF code, 'Level1 > Level2 > Term' path)."""
    with zipfile.ZipFile(MAUDE / "deviceproblemcodes.zip") as zf:
        rows = list(csv.DictReader(io.TextIOWrapper(zf.open(zf.namelist()[0]),
                                                    encoding="utf-8-sig")))
    by_imdrf = {r["IMDRF_CODE"]: r["TERM"] for r in rows if r["IMDRF_CODE"]}
    out = {}
    for r in rows:
        im = r["IMDRF_CODE"]
        chain = [by_imdrf[im[:n]] for n in range(3, len(im) + 1, 2) if im[:n] in by_imdrf]
        out[r["FDA_CODE"]] = (r["TERM"], im, " > ".join(chain) if chain else r["TERM"])
    return out


# Device-problem file per year: the 2024/2025 pools used the 2026-10-06 copy; the 2026 pool
# needs the 2026-11-04 copy, which carries the codes of reports filed since (MANIFEST_2026.json).
PROBLEM_FILE = {"2026": "foidevproblem_20261104.zip"}


def main():
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    suffix = "" if year == "2024" else f"_{year}"
    paths = imdrf_paths()
    seen_2024 = set()
    if year != "2024":  # same process, so the same (salted) string hash
        for line in (MAUDE / "pool.jsonl").open(encoding="utf-8"):
            seen_2024.add(hash(mask_digits(norm(json.loads(line)["text"]))))

    # Pass 1: first D narrative per report -> masked-text key.
    first_d = {}
    it = lines(f"foitext{year}.zip", "cp1252")
    next(it)
    for line in it:
        key, _, ttype, _, _, text = line.rstrip("\r\n").split("|", 5)
        if ttype == "D" and key not in first_d:
            first_d[key] = hash(mask_digits(norm(text)))

    codes = defaultdict(set)
    for line in lines(PROBLEM_FILE.get(year, "foidevproblem.zip"), "utf-8"):
        key, _, code = line.rstrip("\r\n").partition("|")
        if key in first_d:
            codes[key].add(code.strip())
    single = {}
    for key, cs in codes.items():
        cs = cs - NON_INFORMATIVE
        if len(cs) == 1:
            single[key] = next(iter(cs))

    groups = defaultdict(lambda: {"codes": Counter(), "rep": None})
    for key, code in single.items():
        g = groups[first_d[key]]
        g["codes"][code] += 1
        if g["rep"] is None or int(key) < int(g["rep"]):
            g["rep"] = key
    in_2024 = sum(1 for h in groups if h in seen_2024)
    reps = {g["rep"]: (next(iter(g["codes"])), sum(g["codes"].values()))
            for h, g in groups.items() if len(g["codes"]) == 1 and h not in seen_2024}
    print(f"{len(single):,} single-label reports -> {len(groups):,} narratives -> "
          f"{len(reps):,} items after dropping conflicts"
          + (f" and {in_2024:,} narratives already in the 2024 pool" if seen_2024 else ""),
          flush=True)

    # Pass 2: the representatives' own text and device fields.
    texts = {}
    it = lines(f"foitext{year}.zip", "cp1252")
    next(it)
    for line in it:
        key, _, ttype, _, _, text = line.rstrip("\r\n").split("|", 5)
        if ttype == "D" and key in reps and key not in texts:
            texts[key] = " ".join(text.split())
    device = {}
    if (MAUDE / f"device{year}.zip").exists():
        it = lines(f"device{year}.zip", "cp1252")
        header = next(it).rstrip("\r\n").split("|")
        i_key, i_mfr = header.index("MDR_REPORT_KEY"), header.index("MANUFACTURER_D_NAME")
        i_prod, i_gen = header.index("DEVICE_REPORT_PRODUCT_CODE"), header.index("GENERIC_NAME")
        for line in it:
            parts = line.rstrip("\r\n").split("|")
            if len(parts) > max(i_key, i_mfr, i_prod, i_gen) and parts[i_key] in reps \
                    and parts[i_key] not in device:
                device[parts[i_key]] = (parts[i_mfr].strip().upper(), parts[i_prod].strip().upper(),
                                        parts[i_gen].strip())

    counts = Counter()
    with (MAUDE / f"pool{suffix}.jsonl").open("w", encoding="utf-8") as fh:
        for key in sorted(reps, key=int):
            code, n_reports = reps[key]
            term, im, path = paths.get(code, (code, "", code))
            mfr, prod, generic = device.get(key, ("", "", ""))
            counts[code] += 1
            fh.write(json.dumps({"report_key": key, "text": texts[key], "code": code,
                                 "term": term, "imdrf": im, "path": path,
                                 "manufacturer": mfr, "product_code": prod,
                                 "generic_name": generic, "n_reports": n_reports},
                                ensure_ascii=False) + "\n")
    meta = {}
    for c, n in counts.most_common():
        term, im, path = paths.get(c, (c, "", c))
        meta[c] = {"n": n, "term": term, "imdrf": im, "path": path,
                   "depth": (len(im) - 1) // 2 if im else None}
    (MAUDE / f"pool_codes{suffix}.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote pool{suffix}.jsonl: {sum(counts.values()):,} items, {len(counts)} codes; "
          f"depth counts {dict(Counter(m['depth'] for m in meta.values()))}", flush=True)


if __name__ == "__main__":
    main()
