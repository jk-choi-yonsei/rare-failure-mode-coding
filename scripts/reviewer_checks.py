"""Whole-year checks asked for by the review of the v4 manuscript (CPU only, exact counts over every known-code report).

  1 encoder_only   An encoder-only rule with no LLM: the encoder's best-ranked rare code c inside its top k (k = 5 and
                   k = 20) is placed when p(c) / p(first code) >= r; the encoder's own rare first answers always pass
                   (ratio 1); any other report keeps the encoder's best non-rare code. Swept over r, giving a curve of
                   correct against false placements; also the encoder-only point with as many false placements as
                   the agreement rule, for comparison at equal cost.
  2 paired         Encoder against the agreement rule on the same reports: reports placed correctly by one and not
                   the other (discordant), and the same for false placements, with a two-sided exact binomial
                   (sign) test on the discordant counts.
  3 multi_rare     Reports with two or more rare codes in the encoder's top five; shortlisted reports on which an LLM
                   names a rare code in the top five that is not the encoder's best-ranked one (the rule ignores it),
                   and how often that answer is the filed code.
  4 delays         Per rare code filed in the year: days from its first filed report to its first correct placement,
                   for the encoder and the rule; paired over the codes both find, with the codes first filed in the
                   last 90 days of the data counted separately.
  5 count_error    Sum over rare codes of |placed - filed| (from results/reliability).

Usage: python reviewer_checks.py   (MAINT_CORPUS=sdr for SDR) -> results/why_it_works/reviewer_checks_<corpus>.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

import maude_llm as ml  # noqa: E402
import rare_year as ry  # noqa: E402
import why_it_works as w  # noqa: E402
from common import CORPUS, ROOT  # noqa: E402
from maude_data import load_split  # noqa: E402

LLMS = ("qwen38_27b", "gemma4_31b")
R_GRID = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01,
          0.0075, 0.005, 0.004, 0.003, 0.002, 0.0015, 0.001, 0.0005, 0.0002, 0.0001, 0.0]


def counts(Y, final):
    gold, rare = Y["gold"], Y["is_rare"]
    placed = rare[final]
    return int((placed & (final == gold)).sum()), int((placed & (final != gold)).sum())


P_GRID = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02,
          0.015, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0]


def encoder_only(Y, z):
    """Two encoder-only families. ratio: p(c)/p(first) >= r, the encoder's own rare first answers always pass.
    prob: p(c) >= t, so a rare first answer with low probability is also replaced (the encoder vetoes itself)."""
    best_lp = z["rare_logp"].astype(np.float32).max(1)
    top1_lp = z["top_logp"][:, 0].astype(np.float32)
    ratio = np.exp(best_lp - top1_lp)
    p_best = np.exp(best_lp)
    other = np.where(Y["is_rare"][Y["enc"]], Y["nonrare"], Y["enc"])
    out = {}
    for k in (5, 20):
        inside = Y["rank"] < k
        curve = []
        for r in R_GRID:
            fire = inside & ((ratio >= r) | (Y["rank"] == 0))
            c, f = counts(Y, np.where(fire, Y["best"], other))
            curve.append({"r": r, "correct": c, "false": f})
        out[f"top{k}"] = curve
        curve = []
        for t in P_GRID:
            fire = inside & (p_best >= t)
            c, f = counts(Y, np.where(fire, Y["best"], other))
            curve.append({"t": t, "correct": c, "false": f})
        out[f"prob_top{k}"] = curve
    return out, ratio


def at_equal_false(curve, false_target):
    """Encoder-only point with the most correct placements among those with no more false placements than the target."""
    ok = [p for p in curve if p["false"] <= false_target]
    return max(ok, key=lambda p: (p["correct"], -p["false"])) if ok else None


def paired(Y, enc_final, rule_final):
    gold, rare = Y["gold"], Y["is_rare"]
    ce, cr = rare[enc_final] & (enc_final == gold), rare[rule_final] & (rule_final == gold)
    fe, fr = rare[enc_final] & (enc_final != gold), rare[rule_final] & (rule_final != gold)
    b, c = int((cr & ~ce).sum()), int((ce & ~cr).sum())
    bf, cf = int((fe & ~fr).sum()), int((fr & ~fe).sum())
    return {"correct_rule_only": b, "correct_encoder_only": c,
            "correct_sign_test_p": round(binomtest(b, b + c).pvalue, 6) if b + c else None,
            "false_encoder_only": bf, "false_rule_only": cf,
            "false_sign_test_p": round(binomtest(bf, bf + cf).pvalue, 6) if bf + cf else None}


def multi_rare(Y, mk):
    top5 = Y["top"][:, :5]
    n_rare5 = Y["is_rare"][top5].sum(1)
    short = Y["rank"] < 5
    out = {"reports_two_plus_rare_in_top5": int((n_rare5 >= 2).sum()),
           "shortlist": int(short.sum()),
           "shortlist_two_plus_rare": int((short & (n_rare5 >= 2)).sum())}
    for arm in ("zs", "rag"):
        a = Y["A"][mk][arm]
        valid = a >= 0
        in5 = (top5 == np.where(valid, a, -9)[:, None]).any(1)
        other_rare = short & valid & Y["is_rare"][np.maximum(a, 0)] & in5 & (a != Y["best"])
        out[f"{arm}_names_other_rare_in_top5"] = int(other_rare.sum())
        out[f"{arm}_names_other_rare_in_top5_correct"] = int((other_rare & (a == Y["gold"])).sum())
    return out


def delays(Y, finals, last_day):
    day, gold, rare = Y["day"], Y["gold"], Y["is_rare"]
    if day is None:
        return None
    res = {}
    codes = sorted({int(g) for g in np.unique(gold[(gold >= 0)]) if rare[int(g)]})
    first = {}
    for g in codes:
        m = (gold == g) & (day >= 0)
        if m.any():
            first[g] = int(day[m].min())
    det = {}
    for name, f in finals.items():
        det[name] = {}
        for g in first:
            m = (gold == g) & (f == g) & (day >= 0)
            if m.any():
                det[name][g] = int(day[m].min()) - first[g]
    names = list(finals)
    a, b = names
    both = sorted(set(det[a]) & set(det[b]))
    early = [g for g in both if first[g] <= last_day - 90]
    da = np.array([det[a][g] for g in both]) if both else np.array([])
    db = np.array([det[b][g] for g in both]) if both else np.array([])
    res = {"codes_with_dates": len(first), f"found_{a}": len(det[a]), f"found_{b}": len(det[b]), "found_both": len(both),
           f"median_delay_{a}_both": float(np.median(da)) if len(da) else None,
           f"median_delay_{b}_both": float(np.median(db)) if len(db) else None,
           f"{b}_earlier": int((db < da).sum()), f"{b}_later": int((db > da).sum()), "same_day": int((db == da).sum()),
           "found_both_first_filed_90_days_before_end": len(early),
           f"only_{a}": sorted(set(det[a]) - set(det[b])).__len__(), f"only_{b}": sorted(set(det[b]) - set(det[a])).__len__()}
    for name in names:
        res[f"within_90_days_{name}"] = int(sum(1 for g, d in det[name].items() if d <= 90))
    return res


def main():
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    rel = json.loads((ROOT / "results" / "reliability" / f"{CORPUS}.json").read_text(encoding="utf-8"))
    out = {"note": "review of manuscript v4 (paper/v4)", "corpus": CORPUS, "years": {}}
    for year in ("2026", "2025"):
        Y = w.load_year(year, split, legal, p2c, path_of)
        z = np.load(ry.ENC / f"{year}.npz")
        enc_final = Y["enc"]
        eo, _ratio = encoder_only(Y, z)
        last_day = int(Y["day"].max()) if Y["day"] is not None else None
        R = {"encoder": dict(zip(("correct", "false"), counts(Y, enc_final))), "encoder_only_rule": eo, "llms": {}}
        for mk in LLMS:
            rule = w.gate(Y, mk)
            rc, rf = counts(Y, rule)
            R["llms"][mk] = {"rule": {"correct": rc, "false": rf},
                             "encoder_only_at_equal_false": {name: at_equal_false(eo[name], rf) for name in eo},
                             "encoder_only_at_equal_correct": {name: min((p for p in eo[name] if p["correct"] >= rc),
                                                                         key=lambda p: p["false"], default=None)
                                                               for name in eo},
                             "paired_vs_encoder": paired(Y, enc_final, rule),
                             "multi_rare": multi_rare(Y, mk),
                             "delays": delays(Y, {"encoder": enc_final, "rule": rule}, last_day) if last_day else None,
                             "count_error": {"encoder": rel["years"][year]["encoder"]["count_error"],
                                             "rule": rel["years"][year]["gate"][mk]["count_error"]}}
            print(f"[{CORPUS} {year} {mk}] rule {rc}/{rf}; encoder-only at equal false: "
                  f"{R['llms'][mk]['encoder_only_at_equal_false']}; paired {R['llms'][mk]['paired_vs_encoder']}", flush=True)
        out["years"][year] = R
    p = ROOT / "results" / "why_it_works" / f"reviewer_checks_{CORPUS}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
