"""A check of rare-code placements that needs no rater: does the narrative contain the placed code's own word?
(notes/why-it-works-2026-11-16.md, section "글에 그 말이 있는가"; user 2026-11-17: "ㅇㅇ 돌려봐").

A placement counts as false when the filed code differs, yet filers often choose a broader code for the same event
(the subsidence case of case_study.py). For every rare code one term is fixed by a rule that uses no labels:
  words of the code's leaf name, without generic taxonomy words (problem, device, failure, ...);
  a word of six or more letters is reduced to its first six letters and matches any narrative word starting with them,
  a shorter word must match a whole narrative word;
  the code's term is the one found in the fewest reports of the year (but in at least one).
Then, over every known-code report of the year, for the encoder and for the gate (paper configuration):
  share of right placements, and of false placements, whose narrative contains the placed code's term;
  the share expected if the same placements fell on random reports (the terms' frequencies in the year);
  the same restricted to specific terms (found in under 1% of the year's reports).

CPU only. Usage: python term_check.py   (MAINT_CORPUS=sdr for SDR)  ->  results/why_it_works/terms_<corpus>.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

import maude_llm as ml  # noqa: E402
import why_it_works as w  # noqa: E402
from common import CORPUS, ROOT  # noqa: E402
from maude_data import POOL_2025, POOL_2026, load_split  # noqa: E402

LLMS = ("qwen38_27b", "gemma4_31b")
STOP = {"problem", "problems", "device", "devices", "failure", "failed", "fail", "issue", "during", "related", "with",
        "without", "from", "into", "and", "the", "for", "not", "due", "use", "other", "inadequate", "improper"}
WORD = re.compile(r"[a-z]+")
SPECIFIC = 0.01


def candidates(name):
    """[(stem, prefix?)] for the leaf name's content words."""
    out = []
    for wd in WORD.findall(name.lower()):
        if wd in STOP or len(wd) < 3:
            continue
        out.append((wd[:6], True) if len(wd) >= 6 else (wd, False))
    return out


def scan(year, keys, pref, whole):
    """{stem: bool array over the year's reports} for prefix stems and for whole-word stems."""
    pos = {int(k): i for i, k in enumerate(keys)}
    N = len(keys)
    P = {s: np.zeros(N, bool) for s in pref}
    Wd = {s: np.zeros(N, bool) for s in whole}
    with open(POOL_2026 if year == "2026" else POOL_2025, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            i = pos.get(int(r["report_key"]))
            if i is None:
                continue
            words = set(WORD.findall(r["text"].lower()))
            for s in pref & {x[:6] for x in words if len(x) >= 6}:
                P[s][i] = True
            for s in whole & words:
                Wd[s][i] = True
    return P, Wd


def main():
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    rep = {"note": "notes/why-it-works-2026-11-16.md", "corpus": CORPUS, "specific_below": SPECIFIC, "years": {}}
    for year in ("2026", "2025"):
        Y = w.load_year(year, split, legal, p2c, path_of)
        N, gold = Y["N"], Y["gold"]
        rare = np.nonzero(Y["is_rare"])[0]
        cand = {int(c): candidates(Y["names"][c]) for c in rare}
        pref = {s for v in cand.values() for s, p in v if p}
        whole = {s for v in cand.values() for s, p in v if not p}
        P, Wd = scan(year, Y["keys"], pref, whole)
        term, flag, df = {}, {}, {}
        for c, v in cand.items():
            best = None
            for s, p in v:
                f = (P if p else Wd)[s]
                n = int(f.sum())
                if n > 0 and (best is None or n < best[0]):
                    best = (n, s, p, f)
            if best:
                df[c], term[c], flag[c] = best[0] / N, best[1] + ("*" if best[2] else ""), best[3]
        R = {"codes_with_term": len(term), "rare_codes": int(len(rare)), "methods": {}, "per_code": {}}
        finals = {"encoder": Y["enc"]} | {f"gate_{mk}": w.gate(Y, mk) for mk in LLMS}
        for nm, final in finals.items():
            tot = {k: [0, 0] for k in ("right", "false", "right_specific", "false_specific")}
            exp = {"false": 0.0, "false_specific": 0.0}
            rows = []
            for c in term:
                placed = final == c
                right, false = placed & (gold == c), placed & (gold != c)
                r, f = int(right.sum()), int(false.sum())
                rt, ft = int((right & flag[c]).sum()), int((false & flag[c]).sum())
                for key, n, k in (("right", r, rt), ("false", f, ft)):
                    tot[key][0] += k
                    tot[key][1] += n
                    if df[c] < SPECIFIC:
                        tot[key + "_specific"][0] += k
                        tot[key + "_specific"][1] += n
                exp["false"] += f * df[c]
                if df[c] < SPECIFIC:
                    exp["false_specific"] += f * df[c]
                if r + f:
                    rows.append([Y["names"][c], term[c], round(100 * df[c], 3), rt, r, ft, f,
                                 int(((gold == c) & flag[c]).sum()), int((gold == c).sum())])
            R["methods"][nm] = {k: {"with_term": v[0], "n": v[1], "share": round(v[0] / v[1], 3) if v[1] else None} for k, v in tot.items()}
            for k in exp:
                n = tot[k][1]
                R["methods"][nm][k]["expected_share_at_random"] = round(exp[k] / n, 4) if n else None
            R["per_code"][nm] = sorted(rows, key=lambda x: -x[6])
        rep["years"][year] = R
        for nm, M in R["methods"].items():
            print(f"[{CORPUS} {year} {nm}] right with term {M['right']['with_term']}/{M['right']['n']}; false with term "
                  f"{M['false']['with_term']}/{M['false']['n']} (random {M['false']['expected_share_at_random']}); specific terms: "
                  f"right {M['right_specific']['with_term']}/{M['right_specific']['n']}, false "
                  f"{M['false_specific']['with_term']}/{M['false_specific']['n']} (random {M['false_specific']['expected_share_at_random']})",
                  flush=True)
    out = ROOT / "results" / "why_it_works" / f"terms_{CORPUS}.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
