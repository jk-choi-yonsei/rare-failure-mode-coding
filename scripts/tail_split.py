"""Where do the LLM's correct rare answers sit in the encoder's ranking? (notes/paper-narrative-v4-2026-11-16.md, 11-16)

For every rare-code report of a year (balanced test + natural sample, which together hold all of them): the encoder's rank of the
filed code, whether the zero-shot or retrieval LLM names it, and whether the agreement rule (k = 5, either LLM) places it.
Usage: python tail_split.py ; MAINT_CORPUS=sdr python tail_split.py  -> results/rare_year/tail_split_<corpus>.json
"""
import json
import os
import sys

import numpy as np

import candidate_ranker as cr  # noqa: E402
from common import CORPUS  # noqa: E402

out = {}
for llm in ("qwen38_27b", "gemma4_31b"):
    for year in ("2026", "2025"):
        rows = []
        for which in (year, year + "nat"):
            D = cr.load(which, llm)
            P = D["probs"]
            N, C = P.shape
            order = np.argsort(-P, axis=1)
            rank = np.empty_like(order)
            rank[np.arange(N)[:, None], order] = np.arange(C)[None, :]
            rare = (D["counts"] >= 1) & (D["counts"] <= 5)
            rr = np.where(rare[None, :], rank, 10**6)
            best = rr.argmin(1)
            for i in np.where(rare[D["gold"]])[0]:
                g = D["gold"][i]
                rows.append((D["keys"][i], int(rank[i, g]), int(best[i] == g and rr[i].min() < 5),
                             int(D["zs"][i] == g), int(D["rag"][i] == g)))
        seen, uniq = set(), []
        for r in rows:  # a report sits in one set only, but keep the check
            if r[0] not in seen:
                seen.add(r[0])
                uniq.append(r)
        a = np.array([r[1:] for r in uniq])
        rk, cand, zs, rag = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        either = (zs | rag).astype(bool)
        top5 = rk < 5
        res = {"rare_reports": int(len(a)), "encoder_top1": int((rk == 0).sum()), "gold_in_encoder_top5": int(top5.sum()),
               "llm_either_right": int(either.sum()), "llm_either_right_in_top5": int((either & top5).sum()),
               "llm_either_right_outside_top5": int((either & ~top5).sum()), "rule_right": int((cand.astype(bool) & either).sum()),
               "llm_zs_right": int(zs.sum()),
               # five shares of the year's rare reports (figure 3 of the v4 paper)
               "segments": {"kept_top1": int(((rk == 0) & cand.astype(bool) & either).sum()),
                            "recovered_rank2_5": int(((rk >= 1) & (rk < 5) & cand.astype(bool) & either).sum()),
                            "missed_in_top5": int((top5 & ~(cand.astype(bool) & either)).sum()),
                            "llm_only_outside_top5": int((~top5 & either).sum()),
                            "neither": int((~top5 & ~either).sum())}}
        out[f"{CORPUS} {llm} {year}"] = res
        print(CORPUS, llm, year, res, flush=True)
from common import ROOT  # noqa: E402
json.dump(out, open(ROOT / "results" / "rare_year" / f"tail_split_{CORPUS}.json", "w"), indent=1)
