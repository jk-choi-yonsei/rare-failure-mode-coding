"""What limits rare-code coding at natural proportions (notes/natural-selector-2026-11-09.md, 2026-11-11).

Counts are for a whole year: the balanced test and the census of rare (1-5) reports count once, the 6-20
stratum and the uniform draw carry their sampling weights (rare_check.natural_weights). For each LLM:
  methods    encoder, selector, prior-corrected selector; uniform-sample accuracy minus the encoder
             (paired bootstrap); the codes behind the sampled false rare answers
  ceiling    rare-code reports where some coder gives the filed code; right and false rare answers of
             each coder and of coders that agree; what ranking by the selector probability reaches
  codes      right and false rare answers of each coder, code by code
  hindsight  rare answers kept only for (coder, code) pairs chosen with the 2026 answers: an upper bound
  rules      encoder or prior-corrected selector as the base; an LLM rare answer replaces it when the
             code passes a screen (the coder did not name it falsely in earlier natural-proportion
             data), when both LLM coders agree, and when the encoder confidence is low
ceiling, codes, hindsight and the 2026 rules read the 2026 answers: they describe limits, not a test.
Usage: python rare_limits.py   -> results/natural_selector/limits_2026.json
"""
import collections
import json
import os

import numpy as np

os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")
import maude_selector as ms  # noqa: E402
from common import ROOT  # noqa: E402
from maude_router import macro_f1  # noqa: E402
from prior_correct import INFO, corrected, priors  # noqa: E402
from rare_check import RARE_MAX, natural_weights, selector_outputs  # noqa: E402

LLMS = ("qwen38_27b", "gemma4_31b", "opus55")
TAUS = (None, 0.9, 0.7, 0.5)
MODES = ("zs", "zs_or_rag", "zs_and_rag")


def israre(c, f):
    g = np.maximum(f, 0)
    return (f >= 0) & (c[g] >= 1) & (c[g] <= RARE_MAX)


def load_year(llm, yr):
    """Balanced test and natural sample of a year, with the selector picks and the corrected picks."""
    frozen, fmodels = ms.load_frozen(llm)
    pi_bal, pi_nat = priors(llm)
    parts = []
    for which in (yr, yr + "nat"):
        D = ms.load_coders(which, llm)
        P, Pm, arms, pick, j = selector_outputs(D, frozen, fmodels)
        if which == yr:
            w, U = np.ones(len(pick)), np.arange(0)
            top = np.zeros(len(pick), dtype=bool)
        else:
            w, U = natural_weights({"D": D}, yr)
            top = w == w.max()
        parts.append({"D": D, "c": np.asarray(D["counts"]), "gold": np.asarray(D["gold"]), "arms": arms,
                      "sel": np.asarray(pick), "cor": np.asarray(corrected(P, arms, pi_bal, pi_nat)),
                      "Pm": Pm, "w": w, "top": top, "U": U, "encp": np.asarray(D["enc_p"], dtype=float)})
    return parts


def term(D, code):
    return INFO[D["codes"][code]]["term"]


def score(parts, fb, fn):
    """Right and false rare answers a year, sampled false ones, visible modes, accuracy, macro-F1."""
    Pb, Pn = parts
    tp = fp = 0.0
    vis = set()
    for p, f in ((Pb, fb), (Pn, fn)):
        placed = israre(p["c"], f)
        right = placed & (f == p["gold"])
        tp += float((p["w"] * right).sum())
        fp += float((p["w"] * (placed & ~right)).sum())
        vis |= set(p["gold"][right & (p["w"] > 0)].tolist())
    U = Pn["U"]
    return {"right_rare": round(tp, 1), "false_rare": round(fp, 1),
            "false_rare_uniform_sample": int((israre(Pn["c"], fn) & (fn != Pn["gold"]) & Pn["top"]).sum()),
            "modes_visible": len(vis), "accuracy_uniform": round(float((fn[U] == Pn["gold"][U]).mean()), 4),
            "macro_f1_balanced": round(macro_f1(Pb["gold"], fb, Pb["D"]["labels"]), 4)}


def accuracy_gain(Pn, f, B=2000, seed=0):
    """Uniform-sample accuracy minus the encoder accuracy, with a paired bootstrap 95% interval."""
    U = Pn["U"]
    g = Pn["gold"][U]
    d = (f[U] == g).astype(float) - (Pn["arms"][U, 0] == g).astype(float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def false_names(llm, source, K):
    """How often each coder named each code falsely in earlier natural-proportion data."""
    if source == "val2024":
        D = ms.load_coders("2024val", llm)
        keep = np.ones(len(D["gold"]), dtype=bool)
    else:
        D = ms.load_coders("2025nat", llm)
        w, U = natural_weights({"D": D}, "2025")
        keep = w == w.max()
    arms, g = ms.arms_matrix(D), np.asarray(D["gold"])
    out = np.zeros((3, K))
    for k in range(3):
        m = keep & (arms[:, k] >= 0) & (arms[:, k] != g)
        np.add.at(out[k], arms[m, k], 1.0)
    return out, int(keep.sum())


def accept(p, base, allowed, mode, tau):
    """base, with an LLM rare answer put in where the rule accepts it."""
    zs, rag = p["arms"][:, 1], p["arms"][:, 2]
    ok = np.ones(len(zs), dtype=bool) if tau is None else p["encp"] < tau
    zr = israre(p["c"], zs) & allowed[1][np.maximum(zs, 0)] & ok
    rr = israre(p["c"], rag) & allowed[2][np.maximum(rag, 0)] & ok
    final = base.copy()
    if mode == "zs_or_rag":
        final[rr] = rag[rr]
    if mode in ("zs", "zs_or_rag"):
        final[zr] = zs[zr]
    if mode == "zs_and_rag":
        m = zr & (zs == rag)
        final[m] = zs[m]
    return final


def frontier(s, right, w):
    """Rare answers taken in order of s: the most right ones while false stays within 1x, 3x, 9x of right."""
    o = np.argsort(-s)
    r, f = np.cumsum(w[o] * right[o]), np.cumsum(w[o] * (~right[o]))
    out = {}
    for label, ratio in (("false<=right", 1.0), ("false<=3xright", 3.0), ("false<=9xright", 9.0)):
        ok = np.where((f <= ratio * r) & (r > 0))[0]
        i = ok[np.argmax(r[ok])] if len(ok) else None
        out[label] = [0, 0] if i is None else [round(float(r[i]), 1), round(float(f[i]), 1)]
    return out


def ceiling(parts):
    D, c = parts[1]["D"], parts[1]["c"]
    gold, w, top, encp, pick = (np.concatenate([p[k] for p in parts]) for k in ("gold", "w", "top", "encp", "sel"))
    arms, Pm = np.vstack([p["arms"] for p in parts]), np.vstack([p["Pm"] for p in parts])
    gr = israre(c, gold)
    ar = np.column_stack([israre(c, arms[:, k]) for k in range(3)])
    hit = arms == gold[:, None]
    cand = ar.any(1)
    wsum = lambda m: round(float((w * m).sum()), 1)  # noqa: E731
    out = {"year_reports": round(float(w.sum())), "rare_reports": wsum(gr),
           "rare_reports_some_coder_right": wsum(gr & hit.any(1)),
           "rare_reports_all_three_right": wsum(gr & hit.all(1)),
           "per_coder_right_false": {a: [wsum(ar[:, k] & hit[:, k]), wsum(ar[:, k] & ~hit[:, k])]
                                     for k, a in enumerate(ms.ARMS)},
           "reports_where_some_coder_says_rare": wsum(cand),
           "uniform_sample": {"reports": int(top.sum()), "some_coder_says_rare": int((cand & top).sum()),
                              "per_coder": {a: int((ar[:, k] & top).sum()) for k, a in enumerate(ms.ARMS)}}}
    e, z, r = arms[:, 0], arms[:, 1], arms[:, 2]
    rules = {"zs_and_rag": ar[:, 1] & (z == r), "enc_and_zs": ar[:, 0] & (e == z), "enc_and_rag": ar[:, 0] & (e == r),
             "all_three": ar[:, 0] & (e == z) & (e == r)}
    rules["any_two"] = rules["zs_and_rag"] | rules["enc_and_zs"] | rules["enc_and_rag"]
    code = np.where(ar[:, 0] & ((e == z) | (e == r)), e, z)
    out["agreement"] = {name: {"right": wsum(m & (code == gold)), "false": wsum(m & (code != gold)),
                               "false_uniform_sample": int((m & (code != gold) & top).sum())}
                        for name, m in rules.items()}
    idx, ci = np.where(israre(c, pick))[0], np.where(cand)[0]
    s = np.where(ar, Pm, -1.0)
    k = s.argmax(1)
    out["ranked_by_selector_probability"] = {
        "selector_rare_picks": frontier(Pm.max(1)[idx], (pick == gold)[idx], w[idx]),
        "any_rare_answer": frontier(s.max(1)[ci], hit[np.arange(len(k)), k][ci], w[ci])}
    rare_codes = np.where((c >= 1) & (c <= RARE_MAX))[0]
    right_kc, false_kc = np.zeros((3, len(c))), np.zeros((3, len(c)))
    rows = []
    for q in rare_codes:
        row = {"term": term(D, q), "n_train": int(c[q]), "gold_year": wsum(gold == q)}
        for k_, a in enumerate(ms.ARMS):
            m = arms[:, k_] == q
            f = m & ~hit[:, k_] & top
            right_kc[k_, q], false_kc[k_, q] = (w * (m & hit[:, k_])).sum(), (w * (m & ~hit[:, k_])).sum()
            row[a] = [round(float(right_kc[k_, q]), 1), round(float(false_kc[k_, q]), 1), int(f.sum())]
            if f.sum() >= 5:
                row[a + "_false_sample"] = {
                    "filed_as": collections.Counter(term(D, g) for g in gold[f]).most_common(5),
                    "encoder_right": int((e[f] == gold[f]).sum()), "retrieval_right": int((r[f] == gold[f]).sum()),
                    "mean_encoder_confidence": round(float(encp[f].mean()), 2)}
        rows.append(row)
    rows.sort(key=lambda x: -(x["zs"][1] + x["rag"][1]))
    hind = {}
    for label, ratio in (("false<=right", 1.0), ("false<=3xright", 3.0)):
        allowed = (right_kc > 0) & (false_kc <= ratio * right_kc)
        ok = np.column_stack([ar[:, k_] & allowed[k_][np.maximum(arms[:, k_], 0)] for k_ in range(3)])
        anyright = (ok & hit).any(1)
        hind[label] = {"right": wsum(anyright), "false": wsum(ok.any(1) & ~anyright),
                       "pairs_allowed": {a: int(allowed[k_][rare_codes].sum()) for k_, a in enumerate(ms.ARMS)}}
    return out, rows, hind


def main():
    res = {"note": "notes/natural-selector-2026-11-09.md", "git_head": ms.git_head()}
    for llm in LLMS:
        out = {"methods": {}, "rules": {}}
        for yr in ("2026", "2025"):
            parts = load_year(llm, yr)
            Pb, Pn = parts
            K = len(Pn["c"])
            base = {"encoder": (Pb["arms"][:, 0], Pn["arms"][:, 0]), "selector": (Pb["sel"], Pn["sel"]),
                    "selector_prior_corrected": (Pb["cor"], Pn["cor"])}
            meth = {}
            for name, (fb, fn) in base.items():
                meth[name] = score(parts, fb, fn)
                if name != "encoder":
                    meth[name]["accuracy_minus_encoder"] = accuracy_gain(Pn, fn)
                    m = israre(Pn["c"], fn) & (fn != Pn["gold"]) & Pn["top"]
                    meth[name]["false_rare_sample_codes"] = collections.Counter(term(Pn["D"], x) for x in fn[m]).most_common(8)
                print(yr, llm, f"{name:26s}", {k: v for k, v in meth[name].items() if k != "false_rare_sample_codes"})
            out["methods"][yr] = meth
            if yr == "2026":
                out["ceiling"], out["codes"], out["hindsight"] = ceiling(parts)
                print(llm, json.dumps(out["ceiling"]), json.dumps(out["hindsight"]))
            screens = {"none": (np.zeros((3, K)), 0)}
            wanted = {"nat2025": ("nat2025",), "val2024+nat2025": ("val2024", "nat2025")} if yr == "2026" else {"val2024": ("val2024",)}
            for name, srcs in wanted.items():
                try:
                    got = [false_names(llm, s, K) for s in srcs]
                except (SystemExit, FileNotFoundError):  # no outputs of this LLM on those reports
                    continue
                screens[name] = (sum(g[0] for g in got), sum(g[1] for g in got))
            rules = {}
            for sname, (sf, n) in screens.items():
                for t in ((0,) if sname == "none" else (0, 1)):
                    allowed = np.ones((3, K), dtype=bool) if sname == "none" else sf <= t
                    for mode in MODES:
                        for bname in ("encoder", "selector_prior_corrected"):
                            for tau in TAUS:
                                fb = accept(Pb, base[bname][0], allowed, mode, tau)
                                fn = accept(Pn, base[bname][1], allowed, mode, tau)
                                name = f"base={bname}|screen={sname}" + ("" if sname == "none" else f"<={t}") + f"|{mode}|enc_conf<{tau}"
                                rules[name] = score(parts, fb, fn)
                                rules[name]["accuracy_minus_encoder"] = accuracy_gain(Pn, fn)
            out["rules"][yr] = {"screen_reports": {s: v[1] for s, v in screens.items()}, "rules": rules}
            for sfmax in (0, 1, 2, 3, 5):
                ok = [kv for kv in rules.items() if kv[1]["false_rare_uniform_sample"] <= sfmax]
                if ok:
                    k_, v_ = max(ok, key=lambda kv: kv[1]["right_rare"])
                    print("   most right with <=%d sampled false: %-72s right %5.1f false %6.1f (%d) acc %.4f" % (
                        sfmax, k_, v_["right_rare"], v_["false_rare"], v_["false_rare_uniform_sample"], v_["accuracy_uniform"]))
        res[llm] = out
    out_dir = ROOT / "results" / "natural_selector"
    (out_dir / "limits_2026.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
