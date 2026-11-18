"""Failure-mode statistics over a whole year: what the gate changes in the year's counts of rare failure modes
(notes/paper-outline-v4-2026-11-16.md; user 2026-11-16: "신뢰성, 그런 분석들을 좀 해야겠는데").

For every known-code report of a test year, two codings are compared with the filed codes:
  encoder  the encoder's top answer
  gate     the paper's final configuration for rare codes (paper_v4_numbers.py): the encoder's best-ranked rare
           code inside its top five is placed when the zero-shot or the retrieval LLM names it; any other rare
           answer is replaced by the encoder's best non-rare code. (The selection never places a rare code the
           gate did not confirm, so the rare-code statistics of gate + selection are the same.)

Per rare code (1-5 training reports): reports filed, placed, placed right. From these:
  visible modes       rare codes filed in the year with at least one right placement
  phantom modes       rare codes placed only falsely (no right placement)
  count error         sum over rare codes of |placed - filed|; "reviewed" counts keep only right placements
  review load         placed reports per right placement (a reader confirms each rare placement)
  first detection     (reports with dates: SDR; MAUDE once device-file dates are on disk) days from a rare
                      code's first filed report in the year to its first right placement, and the cumulative
                      number of rare codes seen by month

CPU only; reads runs/<corpus>_encyear and the LLM answers that rare_year.py uses.
Usage: python reliability_year.py            -> results/reliability/maude.json
       MAINT_CORPUS=sdr python reliability_year.py -> results/reliability/sdr.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import datetime as dt  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import maude_llm as ml  # noqa: E402
import rare_year as ry  # noqa: E402
from common import CORPUS, ROOT  # noqa: E402
from maude_data import POOL_2025, POOL_2026, load_split  # noqa: E402

K = 5
LLMS = ("qwen38_27b", "gemma4_31b")
OUT = ROOT / "results" / "reliability"


def report_days(year, keys):
    """Day of the year of each report (0 = 1 January), or None when the corpus has no dates on disk."""
    dates = {}
    if CORPUS == "sdr":
        with open(POOL_2026 if year == "2026" else POOL_2025, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("date"):
                    dates[int(r["report_key"])] = r["date"]
    else:
        p = ROOT / "data" / "maude" / f"dates_{year}.json"  # written by a later step from FDA's device file
        if p.exists():
            dates = {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    if not dates:
        return None
    start = dt.date(int(year), 1, 1)
    day = np.array([(dt.date.fromisoformat(dates[int(k)]) - start).days if int(k) in dates else -1 for k in keys])
    return day


def mode_stats(final, gold, is_rare, names, day):
    C = len(is_rare)
    filed = np.bincount(gold, minlength=C)
    placed = np.bincount(final, minlength=C)
    right = np.bincount(final[final == gold], minlength=C)
    rows = [[names[c], int(filed[c]), int(placed[c]), int(right[c])] for c in np.nonzero(is_rare)[0] if filed[c] or placed[c]]
    out = {"right": int(sum(r[3] for r in rows)), "placed": int(sum(r[2] for r in rows)),
           "modes_filed": sum(r[1] > 0 for r in rows), "visible_modes": sum(r[3] > 0 for r in rows),
           "phantom_modes": sum(r[2] > 0 and r[3] == 0 for r in rows),
           "phantom_modes_not_filed": sum(r[2] > 0 and r[1] == 0 for r in rows),
           "count_error": int(sum(abs(r[2] - r[1]) for r in rows)),
           "count_error_reviewed": int(sum(abs(r[3] - r[1]) for r in rows)),
           "reports_filed": int(sum(r[1] for r in rows))}
    out["false"] = out["placed"] - out["right"]
    out["review_load"] = round(out["placed"] / out["right"], 2) if out["right"] else None
    out["per_mode"] = sorted(rows, key=lambda r: (-r[1], r[0]))
    if day is not None:
        delays, monthly = [], np.zeros(12, int)
        first_filed_month = np.zeros(12, int)
        for c in np.nonzero(is_rare & (filed > 0))[0]:
            m_filed = (gold == c) & (day >= 0)
            if not m_filed.any():
                continue
            t0 = int(day[m_filed].min())
            first_filed_month[min(11, dt.date.fromordinal(dt.date(2001, 1, 1).toordinal() + t0).month - 1)] += 1
            hit = m_filed & (final == c)
            if hit.any():
                t1 = int(day[hit].min())
                delays.append(t1 - t0)
                monthly[min(11, dt.date.fromordinal(dt.date(2001, 1, 1).toordinal() + t1).month - 1)] += 1
            else:
                delays.append(None)
        got = [d for d in delays if d is not None]
        out["detection"] = {"modes_with_dates": len(delays), "detected": len(got),
                            "median_delay_days": float(np.median(got)) if got else None,
                            "within_30_days": sum(d <= 30 for d in got), "within_90_days": sum(d <= 90 for d in got),
                            "same_report_day": sum(d == 0 for d in got),
                            "cumulative_seen_by_month": np.cumsum(monthly).tolist(),
                            "cumulative_filed_by_month": np.cumsum(first_filed_month).tolist()}
    return out


def main():
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    rep = {"note": "notes/paper-outline-v4-2026-11-16.md", "corpus": CORPUS, "encoder": ry.TAG, "k": K, "years": {}}
    for year in ("2026", "2025"):
        z = np.load(ry.ENC / f"{year}.npz")
        cls = [str(c) for c in z["classes"]]
        names = [path_of.get(c, c).split(" > ")[-1] for c in cls]
        c2i = {c: i for i, c in enumerate(cls)}
        keys, gold, top = z["keys"], z["gold"].astype(int), z["top_idx"].astype(int)
        enc, best, rank = top[:, 0], z["rare_best"].astype(int), z["rare_rank"].astype(int)
        N = len(keys)
        is_rare = np.zeros(len(cls), bool)
        is_rare[z["rare"].astype(int)] = True
        nonrare = top[np.arange(N), (~is_rare[top]).argmax(1)]
        pos = {int(k): i for i, k in enumerate(keys)}
        day = report_days(year, keys)
        Y = {"reports": N, "dated_reports": None if day is None else int((day >= 0).sum()),
             "encoder": mode_stats(enc, gold, is_rare, names, day), "gate": {}}
        for mk in LLMS:
            A = {}
            for name, kind in ry.KINDS.items():
                a = np.full(N, -2)
                for k, v in ry.llm_answers(year, mk, kind, c2i, legal, p2c).items():
                    if k in pos:
                        a[pos[k]] = v
                A[name] = a
            short = rank < K
            cover = min(float((A[n][short] > -2).mean()) for n in A)
            fire = short & ((A["zs"] == best) | (A["rag"] == best))
            gate = np.where(fire, best, np.where(is_rare[enc], nonrare, enc))
            Y["gate"][mk] = mode_stats(gate, gold, is_rare, names, day)
            Y["gate"][mk]["llm_coverage_of_shortlist"] = round(cover, 4)
        rep["years"][year] = Y
        e = Y["encoder"]
        print(f"[{CORPUS} {year}] {N:,} reports, rare modes filed {e['modes_filed']}; encoder: right {e['right']}, "
              f"false {e['false']}, visible {e['visible_modes']}, phantom {e['phantom_modes']}, count error {e['count_error']}",
              flush=True)
        for mk, g in Y["gate"].items():
            print(f"   gate {mk:11s}: right {g['right']}, false {g['false']}, visible {g['visible_modes']}, "
                  f"phantom {g['phantom_modes']}, count error {g['count_error']} (reviewed {g['count_error_reviewed']}), "
                  f"review load {g['review_load']}, coverage {g['llm_coverage_of_shortlist']}", flush=True)
            if "detection" in g:
                d, de = g["detection"], e["detection"]
                print(f"      detection: gate {d['detected']}/{d['modes_with_dates']} (median {d['median_delay_days']} d, "
                      f"<=30 d {d['within_30_days']}, <=90 d {d['within_90_days']}); encoder {de['detected']} "
                      f"(median {de['median_delay_days']} d, <=30 d {de['within_30_days']}, <=90 d {de['within_90_days']})",
                      flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{CORPUS}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", OUT / f"{CORPUS}.json", flush=True)


if __name__ == "__main__":
    main()
