"""Receipt dates of the MAUDE test-year reports, for the first-detection analysis of reliability_year.py
(notes/reliability-2026-11-16.md). The narrative file (foitext) leaves DATE_REPORT empty, so the date comes from
FDA's device file: DATE_RECEIVED of the report's device rows (the earliest when a report lists several devices).

Inputs (downloaded 2026-11-16 with the user's permission into data/maude/device_files/):
  device2025.zip  https://www.accessdata.fda.gov/MAUDE/ftparea/device2025.zip
  device.zip      https://www.accessdata.fda.gov/MAUDE/ftparea/device.zip  (current year, 2026 to date)
Output: data/maude/dates_<year>.json {report key: "YYYY-MM-DD"} for the keys of runs/maude_encyear/<tag>/<year>.npz.

Usage: python maude_dates.py
"""
import collections
import io
import json
import sys
import zipfile

import numpy as np

from common import ROOT

SRC = ROOT / "data" / "maude" / "device_files"
ENC = ROOT / "runs" / "maude_encyear" / "cb-large-conv_full_seed42"


def read_dates(zpath, wanted):
    """{key: earliest DATE_RECEIVED as YYYY-MM-DD} for the keys in `wanted`."""
    out = {}
    with zipfile.ZipFile(zpath) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
        with zf.open(name) as raw:
            fh = io.TextIOWrapper(raw, encoding="latin-1", newline="")
            head = fh.readline().rstrip("\r\n").split("|")
            ik, idate = head.index("MDR_REPORT_KEY"), head.index("DATE_RECEIVED")
            for line in fh:
                p = line.split("|")
                if len(p) <= idate:
                    continue
                try:
                    k = int(p[ik])
                except ValueError:
                    continue
                if k not in wanted:
                    continue
                d = p[idate].strip()
                if len(d) == 10 and d[4] == "/" and d[7] == "/":
                    d = d.replace("/", "-")
                    if k not in out or d < out[k]:
                        out[k] = d
    print(f"{zpath.name}: {name}, dates for {len(out):,} of {len(wanted):,} wanted keys", flush=True)
    return out


def main():
    years = {y: {int(k) for k in np.load(ENC / f"{y}.npz")["keys"]} for y in ("2025", "2026")}
    wanted = years["2025"] | years["2026"]
    dates = {}
    for f in ("device2025.zip", "device.zip"):
        for k, d in read_dates(SRC / f, wanted).items():
            if k not in dates or d < dates[k]:
                dates[k] = d
    for y, keys in years.items():
        got = {k: dates[k] for k in keys if k in dates}
        months = collections.Counter(d[:7] for d in got.values())
        out = ROOT / "data" / "maude" / f"dates_{y}.json"
        out.write_text(json.dumps({str(k): v for k, v in sorted(got.items())}), encoding="utf-8")
        print(f"{y}: {len(got):,} of {len(keys):,} reports dated ({len(got) / len(keys):.1%}); "
              f"range {min(got.values())} .. {max(got.values())}; by month {sorted(months.items())}", flush=True)
        print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
