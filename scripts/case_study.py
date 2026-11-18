"""One rare failure mode followed through a year, and how the gate's gain is spread over the rare codes
(notes/why-it-works-2026-11-16.md, section "사례"; user 2026-11-17: "사례 하나는 돌려볼까 일단?").

  per code   every rare code with reports in the year: filed, right and false placements of the encoder and of the
             gate, correct answers of each LLM alone, first filed date and first right placement
  spread     share of the gate's gain in right placements that comes from its best code, and the totals without it
  case       one code (default MAUDE "Implant subsidence", keyword stem "subsid"): filed and found by month; where the
             encoder ranks the code on its filed reports; what the encoder answers instead; the filed codes of the
             gate's false placements; and a text check that needs no rater: of all the year's reports whose narrative
             contains the keyword, how they were filed and how many each coding places under the code

Whole-year exact counts (runs/<corpus>_encyear + the LLM answers of rare_year.py). CPU only.
Usage: python case_study.py ["code leaf name" keyword]   (MAINT_CORPUS=sdr for SDR)
Output: results/why_it_works/case_<corpus>.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import collections  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

import maude_llm as ml  # noqa: E402
import why_it_works as w  # noqa: E402
from common import CORPUS, ROOT  # noqa: E402
from maude_data import POOL_2025, POOL_2026, load_split  # noqa: E402

LLMS = ("qwen38_27b", "gemma4_31b")
DEFAULT = {"maude": ("Implant subsidence", "subsid"), "sdr": (None, None)}[CORPUS]


def iso(year, d):
    return (dt.date(int(year), 1, 1) + dt.timedelta(int(d))).isoformat()


def per_code(Y, finals):
    gold, enc, day = Y["gold"], Y["enc"], Y["day"]
    rows = []
    for c in np.nonzero(Y["is_rare"])[0]:
        m = gold == c
        if not m.any():
            continue
        r = {"code": Y["names"][c], "filed": int(m.sum()), "first_filed": iso(Y["year"], day[m].min()) if day is not None else None,
             "encoder": [int((m & (enc == c)).sum()), int(((enc == c) & ~m).sum())]}
        if day is not None and (m & (enc == c)).any():
            r["encoder_first"] = iso(Y["year"], day[m & (enc == c)].min())
        for mk, g in finals.items():
            r[f"gate_{mk}"] = [int((m & (g == c)).sum()), int(((g == c) & ~m).sum())]
            if day is not None and (m & (g == c)).any():
                r[f"gate_{mk}_first"] = iso(Y["year"], day[m & (g == c)].min())
            for kind in ("zs", "rag"):
                a = Y["A"][mk][kind]
                r[f"{kind}_{mk}_right"] = int((m & (a == c)).sum())
                r[f"{kind}_{mk}_answered"] = int((m & (a > -2)).sum())
        rows.append(r)
    return sorted(rows, key=lambda r: -r["filed"])


def spread(rows, mk):
    gain = sorted(((r[f"gate_{mk}"][0] - r["encoder"][0], r["code"]) for r in rows), reverse=True)
    pos = [g for g in gain if g[0] > 0]
    tot_enc, tot_gate = sum(r["encoder"][0] for r in rows), sum(r[f"gate_{mk}"][0] for r in rows)
    top = pos[0] if pos else (0, None)
    best = next(r for r in rows if r["code"] == top[1]) if top[1] else None
    out = {"encoder_right": tot_enc, "gate_right": tot_gate, "codes_gaining": len(pos), "codes_losing": sum(g[0] < 0 for g in gain),
           "top_code": top[1], "top_code_gain": top[0],
           "top_code_share_of_positive_gain": round(top[0] / sum(g[0] for g in pos), 3) if pos else None}
    if best:
        out["without_top_code"] = {"encoder_right": tot_enc - best["encoder"][0], "gate_right": tot_gate - best[f"gate_{mk}"][0],
                                   "encoder_false": sum(r["encoder"][1] for r in rows) - best["encoder"][1],
                                   "gate_false": sum(r[f"gate_{mk}"][1] for r in rows) - best[f"gate_{mk}"][1]}
    return out


def keyword_flags(year, keys, stem):
    """True where the report's narrative contains the stem (case-insensitive)."""
    pos = {int(k): i for i, k in enumerate(keys)}
    has = np.zeros(len(keys), bool)
    with open(POOL_2026 if year == "2026" else POOL_2025, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            i = pos.get(int(r["report_key"]))
            if i is not None and stem in r["text"].lower():
                has[i] = True
    return has


def case(Y, finals, name, stem, groups):
    c = Y["names"].index(name)
    gold, enc, day, top = Y["gold"], Y["enc"], Y["day"], Y["top"]
    m = gold == c
    month = np.array([(dt.date(int(Y["year"]), 1, 1) + dt.timedelta(int(d))).month for d in day])
    out = {"code": Y["paths"][c], "filed": int(m.sum()),
           "by_month": [{"month": mo, "filed": int((m & (month == mo)).sum()), "encoder": int((m & (month == mo) & (enc == c)).sum()),
                         **{f"gate_{mk}": int((m & (month == mo) & (g == c)).sum()) for mk, g in finals.items()}}
                        for mo in range(1, 13) if (month == mo).any()]}
    rank = np.where((top[m] == c).any(1), (top[m] == c).argmax(1), 99)
    out["encoder_rank_on_filed"] = {"top1": int((rank == 0).sum()), "rank2_5": int(((rank >= 1) & (rank < 5)).sum()),
                                    "rank6_20": int(((rank >= 5) & (rank < 20)).sum()), "beyond20": int((rank == 99).sum())}
    wrong = collections.Counter(Y["names"][x] for x in enc[m & (enc != c)])
    out["encoder_answers_instead"] = wrong.most_common(5)
    has = keyword_flags(Y["year"], Y["keys"], stem)
    out["keyword"] = {"stem": stem, "reports_with_keyword": int(has.sum()),
                      "filed_as": collections.Counter(Y["names"][x] for x in gold[has]).most_common(6),
                      "filed_reports_with_keyword": int((m & has).sum()),
                      "encoder_places_code_on_keyword_reports": int((has & (enc == c)).sum())}
    for mk, g in finals.items():
        f = (g == c) & ~m
        out[f"gate_{mk}"] = {"right": int((m & (g == c)).sum()), "false": int(f.sum()),
                             "false_filed_as": collections.Counter(Y["names"][x] for x in gold[f]).most_common(6),
                             "false_with_keyword": int((f & has).sum()),
                             "places_code_on_keyword_reports": int((has & (g == c)).sum())}
        for kind in ("zs", "rag"):
            out[f"gate_{mk}"][f"{kind}_alone_right"] = int((m & (Y["A"][mk][kind] == c)).sum())
    fe = (enc == c) & ~m
    out["encoder"] = {"right": int((m & (enc == c)).sum()), "false": int(fe.sum()), "false_with_keyword": int((fe & has).sum())}
    if groups is not None:
        out["device_groups_of_filed"] = collections.Counter(groups[m].tolist()).most_common(6)
        out["device_groups_count"] = len(set(groups[m].tolist()))
    return out


def main():
    name, stem = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else DEFAULT
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    rep = {"note": "notes/why-it-works-2026-11-16.md", "corpus": CORPUS, "years": {}}
    for year in ("2026", "2025"):
        Y = w.load_year(year, split, legal, p2c, path_of)
        finals = {mk: w.gate(Y, mk) for mk in LLMS}
        rows = per_code(Y, finals)
        R = {"per_code": rows, "spread": {mk: spread(rows, mk) for mk in LLMS}}
        if name:
            groups = w.product_codes(year, Y["keys"]) if CORPUS == "maude" else w.ata_chapters(year, Y["keys"])
            R["case"] = case(Y, finals, name, stem, groups)
        rep["years"][year] = R
        for mk in LLMS:
            s = R["spread"][mk]
            print(f"[{CORPUS} {year} {mk}] right {s['encoder_right']} -> {s['gate_right']}; gaining codes {s['codes_gaining']}, losing "
                  f"{s['codes_losing']}; top code {s['top_code']} +{s['top_code_gain']} ({s['top_code_share_of_positive_gain']}); "
                  f"without it {s.get('without_top_code')}", flush=True)
    out = ROOT / "results" / "why_it_works" / f"case_{CORPUS}.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
