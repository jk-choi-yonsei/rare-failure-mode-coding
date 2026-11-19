"""Exact whole-year rare placements of the logit-adjusted classifier (tau = 0.5), from the classifier's year files
(runs/<corpus>_encyear/<tag>/<year>.npz: top 20 log-probabilities and the log-probability of every rare code).
2026-11-19, after the manuscript review: the method uses no LLM, so its false placements can be counted over every report
instead of being estimated from samples (selection_ablation.logit_adjusted gives the same rule on the samples).

A report gets a rare code when the best adjusted score among the rare codes beats every non-rare code. Non-rare codes
outside the top 20 are bounded by the 20th log-probability plus the largest adjustment of a non-rare code; reports where
that bound leaves the outcome open are counted as undecided and reported.

Usage: [MAINT_CORPUS=sdr] python logit_adjusted_year.py  -> results/paper_v4/<corpus>_logit_adjusted_year.json
"""
import json

import numpy as np

from common import CORPUS, ROOT, RUNS
from maude_data import class_info, known_codes, load_split
from maude_router import training_counts

TAU = 0.5
TAG = {"maude": "cb-large-conv_full_seed42", "sdr": "cb-conv_full_seed42"}[CORPUS]


def main():
    split = load_split()
    classes = known_codes(split)
    codes = sorted(class_info(split))
    n = dict(zip(codes, training_counts(split, codes)["full"].tolist()))
    cnt = np.array([n[c] for c in classes], dtype=float)
    logpi = np.log((cnt + 0.5) / (cnt + 0.5).sum())
    is_rare = (cnt >= 1) & (cnt <= 5)
    adj_nonrare_max = float((-TAU * logpi[~is_rare]).max())
    out = {"note": "logit_adjusted_year.py", "corpus": CORPUS, "tau": TAU, "years": {}}
    for year in ("2025", "2026"):
        z = np.load(RUNS / f"{CORPUS}_encyear" / TAG / f"{year}.npz")
        assert [str(c) for c in z["classes"]] == classes
        rare = z["rare"].astype(int)
        assert set(rare.tolist()) == set(np.nonzero(is_rare)[0].tolist())
        gold = z["gold"].astype(int)
        top = z["top_idx"].astype(int)
        tl = z["top_logp"].astype(np.float64)
        rl = z["rare_logp"].astype(np.float64)
        s_top = tl - TAU * logpi[top]
        s_top_nr = np.where(is_rare[top], -np.inf, s_top).max(1)
        s_r = rl - TAU * logpi[rare][None, :]
        r_best = rare[s_r.argmax(1)]
        s_r_max = s_r.max(1)
        bound = tl[:, -1] + adj_nonrare_max
        placed = s_r_max > np.maximum(s_top_nr, bound)
        open_ = (s_r_max >= s_top_nr) & ~placed
        right = placed & (r_best == gold)
        out["years"][year] = {"reports": int(len(gold)), "rare_right": int(right.sum()), "rare_false": int((placed & ~right).sum()),
                              "undecided": int(open_.sum()), "undecided_rare_reports": int((open_ & is_rare[gold]).sum())}
        print(CORPUS, year, out["years"][year], flush=True)
    p = ROOT / "results" / "paper_v4" / f"{CORPUS}_logit_adjusted_year.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
