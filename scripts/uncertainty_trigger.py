"""Shortlist of the agreement rule against an uncertainty trigger, over whole years (paper v4, Section 5.3).

The agreement rule asks an LLM only about the reports whose five most likely codes under the encoder include a rare
code (1-5 training reports): the shortlist. Combinations such as cascades instead pass the reports on which the
encoder is least confident (largest probability lowest). For each corpus and test year, over every known-code report:
  shortlist            reports whose best-ranked rare code is within the encoder's top five (rare_rank < 5, the rule's
                       own test, ties going to the rare code), and the rare reports in it
  same budget          the same number of reports taken in order of the encoder's confidence, lowest first, and the
                       rare reports in it
  budget to match      reports needed in that order to hold as many rare reports as the shortlist does
  missed rare reports  the encoder's confidence on rare reports whose filed code is not its first answer (median and
                       share at or above 0.5), and its median confidence on all reports

Reads the encoder's whole-year outputs (rare_year.ENC: top-20 codes with log-probabilities). CPU only.
Usage: python uncertainty_trigger.py   (MAINT_CORPUS=sdr for SDR) -> results/why_it_works/uncertainty_<corpus>.json
"""
import json

import numpy as np

import rare_year as ry
from common import CORPUS, ROOT


def year_stats(year):
    z = np.load(ry.ENC / f"{year}.npz")
    gold, top = z["gold"].astype(int), z["top_idx"].astype(int)
    keys = list(z.keys())
    lp = z["top_logp"] if "top_logp" in keys else z["top_lp"]
    conf = np.exp(lp[:, 0])
    ncls = len(z["classes"])
    is_rare = np.zeros(ncls, bool)
    is_rare[z["rare"].astype(int)] = True
    known = gold >= 0
    rare_rep = np.zeros(len(gold), bool)
    rare_rep[known] = is_rare[gold[known]]
    short = z["rare_rank"].astype(int) < 5
    n_short = int(short.sum())
    order = np.argsort(conf, kind="stable")  # least confident first
    cum_rare = np.cumsum(rare_rep[order])
    target = int((short & rare_rep).sum())
    need = int(np.searchsorted(cum_rare, target) + 1) if target else 0
    missed = rare_rep & (top[:, 0] != gold)
    return {"reports": int(len(gold)),
            "shortlist": {"reports": n_short, "share": round(n_short / len(gold), 4),
                          "rare_reports": target},
            "same_budget_uncertainty": {"rare_reports": int(cum_rare[n_short - 1])},
            "uncertainty_budget_to_match": {"reports": need, "share": round(need / len(gold), 4),
                                            "times_shortlist": round(need / n_short, 1)},
            "encoder_confidence": {"missed_rare_median": round(float(np.median(conf[missed])), 3),
                                   "missed_rare_share_ge_0.5": round(float((conf[missed] >= 0.5).mean()), 3),
                                   "all_reports_median": round(float(np.median(conf)), 3)}}


def main():
    out = {"note": "paper v4, Section 5.3", "corpus": CORPUS, "years": {y: year_stats(y) for y in ("2026", "2025")}}
    p = ROOT / "results" / "why_it_works" / f"uncertainty_{CORPUS}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
