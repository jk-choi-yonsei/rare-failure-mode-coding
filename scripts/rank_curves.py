"""Where the classifier ranks the filed code (2026-11-18, for the redesigned Fig. 3 of the paper).

For every known-code report of each test year: the rank of the filed code in the classifier's ranking (from the
whole-year pass, top 20 kept), and the share of reports whose filed code is within the top k, k = 1..20, separately for
rare reports (filed under a code with one to five training reports) and for all other reports.

Usage: [MAINT_CORPUS=sdr] python rank_curves.py  -> results/why_it_works/rank_curve_<corpus>.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402

import numpy as np  # noqa: E402

import maude_llm as ml  # noqa: E402
import why_it_works as w  # noqa: E402
from common import CORPUS, ROOT  # noqa: E402
from maude_data import load_split  # noqa: E402


def main():
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    out = {"corpus": CORPUS, "k": list(range(1, 21)), "years": {}}
    for year in ("2025", "2026"):
        Y = w.load_year(year, split, legal, p2c, path_of)
        gold, rr = Y["gold"], Y["is_rare"][Y["gold"]]
        hit = Y["top"] == gold[:, None]
        rank = np.where(hit.any(1), hit.argmax(1), Y["top"].shape[1])
        out["years"][year] = {"rare_reports": int(rr.sum()), "other_reports": int((~rr).sum()),
                              "rare": [round(float((rank[rr] < k).mean()), 4) for k in out["k"]],
                              "other": [round(float((rank[~rr] < k).mean()), 4) for k in out["k"]]}
        Ry = out["years"][year]
        print(f"[{CORPUS} {year}] rare {Ry['rare_reports']}: top1 {Ry['rare'][0]:.3f} top5 {Ry['rare'][4]:.3f}; "
              f"other: top1 {Ry['other'][0]:.3f} top5 {Ry['other'][4]:.3f}", flush=True)
    p = ROOT / "results" / "why_it_works" / f"rank_curve_{CORPUS}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
