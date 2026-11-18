"""The agreement rule with another encoder (a check asked for after the review of manuscript v4, 2026-11-18).

Does the paper's premise (the filed rare code is often inside the encoder's top five without being first) and the
rule's gain hold when the encoder is replaced? The other encoder is trained with the same recipe (maude_encoder.py with
MAINT_ENC_MODEL / MAINT_ENC_NAME, or another seed), passed over every known-code report of both test years
(maude_encoder_year.py --tag), and its shortlist is answered by the same LLM coders (rare_year.py and
openweight_llm.py with MAINT_ENC_TAG and MAINT_SHORT_SUFFIX). This script reads those files; it runs on the CPU.

Per test year: reports, encoder accuracy over the whole year, rare reports with the filed code at rank 1 and within
the top five, the shortlist (reports with a rare code in the encoder's top five) and the share of it that holds LLM
answers, and correct / false rare placements (exact, whole year) of the encoder alone and of the rule (top five,
unconfirmed rare answers dropped) with either LLM coder, the zero-shot LLM only, the retrieval LLM only and both;
rare modes found and false alarms; a two-sided exact sign test of the rule against the encoder on discordant reports.

Usage (environment set by run_alt_encoder.ps1): python alt_encoder_check.py
Output: results/alt_encoder/<corpus><MAINT_SHORT_SUFFIX>.json
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


def sign_test(a, b):
    return round(float(binomtest(a, a + b).pvalue), 6) if a + b else None


def main():
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    out = {"note": "another encoder under the agreement rule (2026-11-18)", "corpus": CORPUS, "encoder": ry.TAG,
           "shortlist_suffix": ry.ALT, "years": {}}
    for year in ry.YEARS:
        if not (ry.ENC / f"{year}.npz").exists():
            out["years"][year] = {"not_run": "no whole-year pass of this encoder"}
            continue
        Y = w.load_year(year, split, legal, p2c, path_of)
        gold, enc, rare = Y["gold"], Y["enc"], Y["is_rare"]
        rr = rare[gold]
        hit = Y["top"] == gold[:, None]
        rank = np.where(hit.any(1), hit.argmax(1), Y["top"].shape[1])
        short = Y["rank"] < 5
        R = {"reports": int(Y["N"]), "encoder_accuracy": round(float((enc == gold).mean()), 4),
             "rare_reports": int(rr.sum()),
             "filed_rare_code_rank1": int((rank[rr] < 1).sum()), "filed_rare_code_top5": int((rank[rr] < 5).sum()),
             "other_reports_rank1_share": round(float((rank[~rr] < 1).mean()), 4),
             "other_reports_top5_share": round(float((rank[~rr] < 5).mean()), 4),
             "shortlist": int(short.sum()), "shortlist_share": round(float(short.mean()), 4),
             "rare_reports_on_shortlist": int((short & rr).sum()),
             "encoder_alone": w.counts(Y, enc), "llms": {}}
        ce = rare[enc] & (enc == gold)
        fe = rare[enc] & (enc != gold)
        for mk in LLMS:
            A = Y["A"][mk]
            cover = {n: round(float((A[n][short] > -2).mean()), 4) if short.any() else 1.0 for n in ("zs", "rag")}
            M = {"shortlist_answered": cover, "rules": {}}
            for conf in ("either", "zs", "rag", "both"):
                g = w.gate(Y, mk, k=5, conf=conf)
                M["rules"][conf] = w.counts(Y, g)
            g = w.gate(Y, mk)
            cr, fr = rare[g] & (g == gold), rare[g] & (g != gold)
            b, c = int((cr & ~ce).sum()), int((ce & ~cr).sum())
            bf, cf = int((fe & ~fr).sum()), int((fr & ~fe).sum())
            M["paired_vs_encoder"] = {"correct_rule_only": b, "correct_encoder_only": c, "correct_p": sign_test(b, c),
                                      "false_encoder_only": bf, "false_rule_only": cf, "false_p": sign_test(bf, cf)}
            R["llms"][mk] = M
            e, r = R["encoder_alone"], M["rules"]["either"]
            print(f"[{CORPUS} {year} {mk}] answered {cover}; encoder {e['right']}/{e['false']} -> rule {r['right']}/{r['false']} "
                  f"(modes {e['visible_modes']}/{e['phantom_modes']} -> {r['visible_modes']}/{r['phantom_modes']}); "
                  f"paired {M['paired_vs_encoder']}", flush=True)
        print(f"[{CORPUS} {year}] {R['reports']:,} reports, accuracy {R['encoder_accuracy']}, rare {R['rare_reports']}: "
              f"rank 1 {R['filed_rare_code_rank1']}, top five {R['filed_rare_code_top5']}; shortlist {R['shortlist']} "
              f"({R['shortlist_share']:.2%})", flush=True)
        out["years"][year] = R
    d = ROOT / "results" / "alt_encoder"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{CORPUS}{ry.ALT or '_paper'}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
