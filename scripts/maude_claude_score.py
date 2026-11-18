"""Score the Claude arms on MAUDE (notes/prereg-maude-claude-2026-10-22.md).

Confirmatory, 95% paired item bootstrap (B = 1000, seed 42, the inference unit
of every earlier MAUDE test); a two-stage code-cluster interval (items within
codes, then codes) is reported next to each confirmatory cell but does not
decide the verdict. 2025 out-of-time set unless stated. "CB-large" at 10k means
the healthy seeds 42 and 43 (seed 44 collapsed), as in the LoRA test.

  H-C1  per Claude model m in CLAUDE: m - CB-large > 0 in the 1-5 bin at 10k
        and full N (both cells) -- which models pass the <=5 rule
  H-C2  Opus 5.5 - Qwen3.8-27B in the 21-89 bin at 100k and full N: CI above 0
        and point >= +0.05 (both cells) -- a frontier model moves the upper edge
  H-C3  Opus 5.5 with full-data retrieval - CB-large > 0 overall at full N
  H-C4  Opus 5.5 at effort high (reasoning) - effort low > 0 in the 1-5 bin at
        10k and full N (the arm was picked by a pilot rule written in the note)
  H-C5  per m in CLAUDE: novel-vs-known-tail AUROC, m - CB-MSP > 0 on the 2024
        set (the MAUDE P1 contrast), read from runs/maude/metrics.json
Each verdict is holds / fails / incomplete. Everything is scored again without
the near-duplicate items. Descriptive: every Claude arm (Fable 5.1 included)
against CB-base, CB-large, Qwen and the LLM's own retrieval version per bin;
P3 (pooled AUROC); agreement and per-bin differences between repeated runs
and between the thinking on/off pair.

Usage: python maude_claude_score.py  -> runs/maude2025/claude_metrics.json
"""
import json

import numpy as np

import maude_llm as ml
from common import RUNS, SEED
from maude_extra_score import OUT, load_arms, outcome, paired, resamples
from maude_router import load_eval, macro_f1

CLAUDE = ("haiku45", "haiku55", "sonnet5", "opus55")
EXTRA = ("fable51", "opus55high", "sonnet5think", "sonnet5max")
FRONTIER, OPEN_REF, THINK = "opus55", "qwen38_27b", ("opus55high", "opus55")  # reasoning arm, its thinking-off twin
SIZES = ("10k", "100k", "full")
BINS = {"0": (0, 0), "1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
LARGE_HEALTHY_10K = 2  # seeds 42 and 43
MARGIN_C2 = 0.05
REPEAT_DIR = RUNS / "claude" / "repeat"
B = 1000


def per_code_f1(gold, preds_list, labels, sample=None):
    """Seed-averaged per-code F1 over `labels`, scored on every item (false positives count)."""
    g = gold if sample is None else gold[sample]
    out = []
    for p in preds_list:
        pp = p if sample is None else p[sample]
        n = int(max(g.max(), pp.max(), labels.max())) + 1
        tc, pc = np.bincount(g, minlength=n), np.bincount(pp[pp >= 0], minlength=n)
        tp = np.bincount(g[pp == g], minlength=n)
        denom = (pc + tc)[labels]
        out.append(np.where(denom > 0, 2 * tp[labels] / np.maximum(denom, 1), 0.0))
    return np.mean(out, axis=0)


def code_ci(gold, a, b, labels, samples):
    """Two-stage bootstrap of macro-F1(a) - macro-F1(b): items resampled, then the bin's codes."""
    rng = np.random.default_rng(SEED + 1)
    k = len(labels)
    d = sorted(float((per_code_f1(gold, a, labels, s) - per_code_f1(gold, b, labels, s))[rng.integers(0, k, k)].mean())
               for s in samples)
    lo, hi = d[round(0.025 * B) - 1], d[round(0.975 * B) - 1]
    return {"ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": lo > 0 or hi < 0}


def load_repeats(ev):
    out = {}
    if REPEAT_DIR.exists():
        for p in REPEAT_DIR.glob("maude2025_cls_*.jsonl"):
            mk = p.stem.removeprefix("maude2025_cls_")
            pred, status = _load_path(ev, p)
            if pred is not None:
                out[mk] = pred
    return out


def _load_path(ev, path):
    """load_llm on an arbitrary path (it reads runs/maude2025/{name}.jsonl by name)."""
    import maude_extra_score as mx
    saved = mx.OUT
    try:
        mx.OUT = path.parent
        return mx.load_llm(ev, path.stem)
    finally:
        mx.OUT = saved


def evaluate(ev, A, keep=None):
    sel = np.arange(len(ev["gold"])) if keep is None else np.flatnonzero(keep)
    gold = ev["gold"][sel]
    n = len(gold)
    samples = resamples(n)
    whole = np.arange(n)
    present = np.array(sorted(set(gold.tolist())))

    def F(preds_list, labels):
        P = [p[sel] for p in preds_list]
        return lambda s: float(np.mean([macro_f1(gold, p, labels, s) for p in P]))

    def val(preds_list, labels):
        return round(F(preds_list, labels)(whole), 4)

    def pair(a, b, labels, cluster=True):
        c = paired(F(a, labels), F(b, labels), n, samples)
        if cluster:
            c["code"] = code_ci(gold, [p[sel] for p in a], [p[sel] for p in b], labels, samples)
        return c

    def labels_for(size, b):
        if b == "all":
            return present
        lo, hi = BINS[b]
        return np.array([c for c in present if lo <= ev["counts"][size][c] <= hi])

    llm = ev["llm"]
    rep = {"n": n, "models": {}, "pairs": {}}
    for mk in CLAUDE + EXTRA:
        if mk not in llm:
            continue
        blk = {"sizes": {}}
        for size in SIZES:
            cb = [p for p, _ in ev["enc"][size]]
            large_all = A["large"][size]["preds"]
            large = large_all[:LARGE_HEALTHY_10K] if size == "10k" else large_all
            cells = {}
            for b in ["all"] + list(BINS):
                labels = labels_for(size, b)
                if len(labels) == 0:
                    continue
                c = {"n_codes": len(labels), "llm": val([llm[mk]], labels), "cb": val(cb, labels),
                     "llm_minus_cb": pair([llm[mk]], cb, labels, cluster=(b in ("1-5", "21-89")))}
                if large:
                    c["cb_large"] = val(large, labels)
                    c["llm_minus_large"] = pair([llm[mk]], large, labels, cluster=(b in ("1-5", "21-89")))
                if OPEN_REF in llm:
                    c["qwen"] = val([llm[OPEN_REF]], labels)
                    c["llm_minus_qwen"] = pair([llm[mk]], [llm[OPEN_REF]], labels, cluster=(b in ("1-5", "21-89")))
                rag = A["rag"].get((size, mk))
                if rag is not None:
                    c["rag"] = val([rag], labels)
                    c["rag_minus_llm"] = pair([rag], [llm[mk]], labels, cluster=False)
                    if large:
                        c["rag_minus_large"] = pair([rag], large, labels, cluster=(b in ("all", "21-89")))
                cells[b] = c
            blk["sizes"][size] = cells
        rep["models"][mk] = blk

    # Thinking on vs off (same Sonnet 5) and repeats: agreement and per-bin differences.
    def compare(a, b, name):
        if a not in llm or b not in llm:
            return
        pa, pb = llm[a][sel], llm[b][sel]
        out = {"agreement": round(float((pa == pb).mean()), 4), "sizes": {}}
        for size in SIZES:
            out["sizes"][size] = {bn: pair([llm[a]], [llm[b]], labels_for(size, bn), cluster=(bn == "1-5"))
                                  for bn in ("all", "1-5", "6-20", "21-89")
                                  if len(labels_for(size, bn))}
        rep["pairs"][name] = out
    compare(THINK[0], THINK[1], "reasoning_minus_plain")
    for mk, pred in A.get("repeats", {}).items():
        if mk in llm:
            pb = pred[sel]
            out = {"agreement": round(float((llm[mk][sel] == pb).mean()), 4), "sizes": {}}
            for size in SIZES:
                out["sizes"][size] = {bn: paired(F([pred], labels_for(size, bn)), F([llm[mk]], labels_for(size, bn)), n, samples)
                                      for bn in ("all", "1-5", "21-89") if len(labels_for(size, bn))}
            rep["pairs"][f"repeat_{mk}_minus_first"] = out

    rep["verdicts"] = verdicts(rep)
    return rep


def novelty():
    """P1 (novel vs known-tail AUROC, m - MSP) and P3 (pooled) for every Claude arm, from the 2024 scorer."""
    p = RUNS / "maude" / "metrics.json"
    if not p.exists():
        return {}
    nov = json.load(open(p, encoding="utf-8")).get("novelty", {})
    out = {}
    for mk in CLAUDE + EXTRA:
        p1 = nov.get("vs_known_tail", {}).get("contrasts", {}).get(f"{mk}_minus_msp")
        p3 = nov.get("pooled", {}).get("contrasts", {}).get(f"{mk}_minus_msp")
        if p1 or p3:
            out[mk] = {"P1_tail_minus_msp": p1, "P3_pooled_minus_msp": p3,
                       "auroc": {k: v["auroc"].get(mk) for k, v in nov.items() if mk in v.get("auroc", {})}}
    return out


def judge(cells, sign, margin=0.0):
    if any(c is None for c in cells):
        return "incomplete"
    ok = all(((c["ci95"][0] > 0 and c["point"] >= margin) if sign > 0 else (c["ci95"][1] < 0 and c["point"] <= -margin))
             for c in cells)
    return "holds" if ok else "fails"


def verdicts(rep):
    def cell(mk, size, b, key):
        return rep["models"].get(mk, {}).get("sizes", {}).get(size, {}).get(b, {}).get(key)
    v = {}
    for mk in CLAUDE:
        v[f"H_C1_{mk}_beats_large_1_5"] = judge([cell(mk, s, "1-5", "llm_minus_large") for s in ("10k", "full")], 1)
    v["H_C2_opus_moves_upper_edge"] = judge([cell(FRONTIER, s, "21-89", "llm_minus_qwen") for s in ("100k", "full")], 1, MARGIN_C2)
    v["H_C3_opus_rag_beats_large_overall"] = judge([cell(FRONTIER, "full", "all", "rag_minus_large")], 1)
    t = rep["pairs"].get("reasoning_minus_plain", {}).get("sizes", {})
    v["H_C4_reasoning_helps_1_5"] = judge([t.get(s, {}).get("1-5") for s in ("10k", "full")], 1)
    return v


def main():
    ev = load_eval("2025")
    A = load_arms(ev)
    A["repeats"] = load_repeats(ev)
    report = {"files": {k: s for k, s in A["files"].items() if any(k.endswith(m) for m in CLAUDE + EXTRA)},
              "main": evaluate(ev, A), "novelty_2024": novelty()}
    nov = report["novelty_2024"]
    for mk in CLAUDE:
        c = nov.get(mk, {}).get("P1_tail_minus_msp")
        report["main"]["verdicts"][f"H_C5_{mk}_novelty_P1"] = judge([c], 1) if c else "incomplete"
    nd = OUT / "neardup.json"
    if nd.exists():
        cos = json.loads(nd.read_text())
        keep = np.array([cos[str(k)][0] < 0.9 for k in ev["keys"]])
        report["neardup_excluded"] = evaluate(ev, A, keep)
        report["verdicts_changed_without_neardup"] = {
            h: [v, report["neardup_excluded"]["verdicts"][h]] for h, v in report["main"]["verdicts"].items()
            if h in report["neardup_excluded"]["verdicts"] and v != report["neardup_excluded"]["verdicts"][h]}
    (OUT / "claude_metrics.json").write_text(json.dumps(report, indent=1))
    show(report)


def fmt(c):
    if c is None:
        return "-"
    s = f"{c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{'*' if c['excludes_zero'] else ''}"
    if "code" in c:
        s += f" code[{c['code']['ci95'][0]:+.3f}, {c['code']['ci95'][1]:+.3f}]{'*' if c['code']['excludes_zero'] else ''}"
    return s


def show(r):
    print("===== files:", {k: (v.get("complete"), v.get("invalid"), v.get("truncated")) for k, v in r["files"].items() if v.get("present")})
    m = r["main"]
    for mk, blk in m["models"].items():
        for size, cells in blk["sizes"].items():
            for b, c in cells.items():
                if b in ("0", "6-20", "90+") and mk in EXTRA:
                    continue
                print(f"{mk:12} {size:4} {b:5} llm {c['llm']:.3f} cb {c['cb']:.3f} large {c.get('cb_large', float('nan')):.3f} "
                      f"qwen {c.get('qwen', float('nan')):.3f} | -large {fmt(c.get('llm_minus_large'))} | -qwen {fmt(c.get('llm_minus_qwen'))}"
                      + (f" | rag {c['rag']:.3f} rag-large {fmt(c.get('rag_minus_large'))}" if "rag" in c else ""))
    for name, p in m["pairs"].items():
        print(f"===== {name}: agreement {p['agreement']}; " + "; ".join(
            f"{s} {b} {fmt(c)}" for s, bb in p["sizes"].items() for b, c in bb.items()))
    print("===== novelty 2024:", json.dumps(r["novelty_2024"])[:1500])
    print("verdicts:", json.dumps(m["verdicts"], indent=1))
    if "neardup_excluded" in r:
        print(f"without near-duplicates (n={r['neardup_excluded']['n']}): changed {r['verdicts_changed_without_neardup']}")


if __name__ == "__main__":
    main()
