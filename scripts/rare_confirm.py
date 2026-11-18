"""Rare codes at natural proportions: the encoder shortlists, a second source confirms
(exploration; notes/selector-vs-rules-2026-11-11.md, section "드문 코드: 인코더 후보를 LLM이 확인").

Rule: take the rare code (1-5 training reports) that the encoder ranks best for a report. If it
is inside the encoder's top k and a confirmer names the same code, the report gets that code;
otherwise the encoder's answer stays. Confirmers: the zero-shot LLM, the retrieval LLM, either,
both, any of the ten nearest training reports, the nearest training report, or none.

Counted over the whole year as in rare_check.year_counts: right and false rare placements (the
balanced test and the rare strata count one report each, the uniform draw is weighted), and
"hits" = the number of ordinary reports (codes with 21+ training reports) of the uniform draw
that receive a rare code falsely; one hit stands for about 164 reports a year. Also: where the
encoder ranks the filed code of the year's rare reports, and how many ordinary reports have a
rare code inside the encoder's top k. Saved outputs only; exploratory.

Usage: python rare_confirm.py   -> results/candidate_ranker/rare_confirm.json
"""
import json

import numpy as np

import candidate_ranker as cr
from common import ROOT
from rare_check import natural_weights, year_counts

KS = (1, 3, 5, 10, 20, 50)


def main():
    out = {}
    for llm in cr.LLMS:
        for year in ("2026", "2025"):
            Db, Dn = cr.load(year, llm), cr.load(year + "nat", llm)
            wn, U = natural_weights({"D": Dn}, year)
            cnt = Db["counts"]
            rare = (cnt >= 1) & (cnt <= 5)
            ordn = U[cnt[Dn["gold"][U]] >= 21]
            S, gold_rank = {}, []
            for name, D, w in (("b", Db, np.ones(len(Db["gold"]))), ("n", Dn, wn)):
                P = D["probs"]
                N, C = P.shape
                order = np.argsort(-P, axis=1)
                rank = np.empty_like(order)
                rank[np.arange(N)[:, None], order] = np.arange(C)[None, :]
                rr = np.where(rare[None, :], rank, 10**6)
                best = rr.argmin(1)
                S[name] = {"best": best, "rank": rr.min(1), "D": D}
                m = rare[D["gold"]] & (w > 0)
                gold_rank += rank[np.where(m)[0], D["gold"][m]].tolist()
            gold_rank = np.array(gold_rank)
            res = {"rare_reports": int(len(gold_rank)), "ordinary_in_uniform_draw": int(len(ordn)),
                   "filed_code_in_encoder_top": {f"top{k}": int((gold_rank < k).sum()) for k in KS},
                   "ordinary_with_rare_code_in_encoder_top": {f"top{k}": round(float((S["n"]["rank"][ordn] < k).mean()), 4) for k in KS}}

            def score(rule):
                fin = {}
                for name in ("b", "n"):
                    s = S[name]
                    f = s["D"]["enc"].copy()
                    use = rule(s, s["D"])
                    f[use] = s["best"][use]
                    fin[name] = f
                tp, fp, vis = year_counts({"D": Db}, {"D": Dn}, wn, fin["b"], fin["n"])
                f = fin["n"][ordn]
                return {"right": round(tp), "false": round(fp), "hits": int((rare[f] & (f != Dn["gold"][ordn])).sum()), "modes": vis,
                        "accuracy": round(float((fin["n"][U] == Dn["gold"][U]).mean()), 4)}
            res["encoder_alone"] = score(lambda s, D: np.zeros(len(D["gold"]), bool))
            for k in (3, 5, 10):
                def top(s):
                    return s["rank"] < k
                res[f"k{k}"] = {
                    "zs": score(lambda s, D: top(s) & (D["zs"] == s["best"])),
                    "rag": score(lambda s, D: top(s) & (D["rag"] == s["best"])),
                    "either": score(lambda s, D: top(s) & ((D["zs"] == s["best"]) | (D["rag"] == s["best"]))),
                    "both": score(lambda s, D: top(s) & (D["zs"] == s["best"]) & (D["rag"] == s["best"])),
                    "neighbours_any": score(lambda s, D: top(s) & (D["knn"][np.arange(len(D["gold"])), s["best"]] > 0)),
                    "nearest_neighbour": score(lambda s, D: top(s) & (D["nn1"] == s["best"])),
                    "none": score(lambda s, D: top(s))}
            out[f"{llm} {year}"] = res
            print(llm, year, "encoder", res["encoder_alone"], flush=True)
            for k in (3, 5, 10):
                print("  ", f"k{k}", {c: f"{v['right']}/{v['false']}({v['hits']})" for c, v in res[f"k{k}"].items()}, flush=True)
    d = ROOT / "results" / "candidate_ranker"
    d.mkdir(parents=True, exist_ok=True)
    (d / "rare_confirm.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", d / "rare_confirm.json")


if __name__ == "__main__":
    main()
