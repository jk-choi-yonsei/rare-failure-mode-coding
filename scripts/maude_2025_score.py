"""Out-of-time scoring on MAUDE 2025 (notes/prereg-maude-2025-router-2026-10-08.md).

1. Reproducibility: the retrained CB models against the originals on the 2024 test.
2. Boundary hypotheses, with codes binned by their 2024 training count at N:
   H1   1-5 bin, all N in {1k, 10k, 100k, full}: LLM - CB(N) > 0
   H2a  21-89 bin, N in {100k, full}:           LLM - CB(N) < 0
   H2b  21-89 bin, N = 1k:                      LLM - CB(N) > 0
   each for Qwen and Gemma, 95% interval clear of zero.
3. Routers frozen on 2024 (splits/maude_router_v1.json), N in {10k, 100k, full}:
   R1  selected router - CB(N) only > 0      R2  selected router - LLM only > 0
   secondary: the confidence router (partial cost) against the same two.
4. Near-duplicate sensitivity: 1-3 again without 2025 items whose nearest 2024
   training item has TF-IDF cosine >= 0.9 (runs/maude2025/neardup.json).

Macro-F1 over the known codes present in the evaluation set (bins: the bin's
codes), every item in the sample; paired item bootstrap, 1000 resamples,
seed 42, CB seeds averaged inside each resample.

Usage: python maude_2025_score.py   -> runs/maude2025/metrics.json
"""
import json
import random

import numpy as np

from common import RUNS, SEED
from maude_router import FROZEN, SIZES, apply, load_eval, macro_f1

B = 1000
BINS = {"1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89)}
TEST_MODELS = ("qwen38_27b", "gemma4_31b")
ROUTER_SIZES = ("10k", "100k", "full")
OUT = RUNS / "maude2025"


def resamples(n):
    rng = random.Random(SEED)
    return [np.array([rng.randrange(n) for _ in range(n)]) for _ in range(B)]


def seed_mean(f, preds, sample):
    return float(np.mean([f(p, sample) for p in preds]))


def paired(f_a, f_b, n, samples):
    full = np.arange(n)
    d = sorted(f_a(s) - f_b(s) for s in samples)
    lo, hi = d[round(0.025 * B) - 1], d[round(0.975 * B) - 1]
    return {"point": round(f_a(full) - f_b(full), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": lo > 0 or hi < 0}


def reproducibility():
    orig, rerun = load_eval("2024"), load_eval("2024rerun")
    out = {}
    for size in SIZES:
        if size not in rerun["enc"]:
            continue
        rows = []
        for (p0, _), (p1, _) in zip(orig["enc"][size], rerun["enc"][size]):
            rows.append({"agreement": round(float((p0 == p1).mean()), 4),
                         "macro_f1_original": round(macro_f1(orig["gold"], p0, orig["labels"]), 4),
                         "macro_f1_rerun": round(macro_f1(orig["gold"], p1, orig["labels"]), 4)})
        out[size] = rows
    return out


def evaluate(ev, keep=None):
    """Boundary cells and router contrasts on the 2025 items (optionally a subset)."""
    gold = ev["gold"] if keep is None else ev["gold"][keep]
    n = len(gold)
    samples = resamples(n)
    sub = (lambda a: a) if keep is None else (lambda a: a[keep])
    enc = {s: [(sub(p), sub(c)) for p, c in runs] for s, runs in ev["enc"].items()}
    llm = {m: sub(p) for m, p in ev["llm"].items()}
    present = np.array(sorted(set(gold.tolist())))
    report = {"n": n, "bins": {}, "routers": {}}

    for size in SIZES:
        if size not in enc:
            continue
        cnt = ev["counts"][size]
        for name, (lo, hi) in BINS.items():
            labels = np.array([c for c in present if lo <= cnt[c] <= hi])
            if len(labels) == 0:
                continue

            def cb(s, labels=labels, size=size):
                return seed_mean(lambda p, s2: macro_f1(gold, p, labels, s2), [p for p, _ in enc[size]], s)

            cell = {"n_codes": len(labels), "n_items": int(np.isin(gold, labels).sum()),
                    "cb": round(cb(np.arange(n)), 4), "llm": {}}
            for mk, lp in llm.items():
                f = lambda s, lp=lp, labels=labels: macro_f1(gold, lp, labels, s)  # noqa: E731
                cell["llm"][mk] = {"macro_f1": round(f(np.arange(n)), 4), "minus_cb": paired(f, cb, n, samples)}
            report["bins"][f"{size}/{name}"] = cell

    frozen = json.loads(FROZEN.read_text())["routers"]
    for size in ROUTER_SIZES:
        if size not in enc:
            continue
        for mk in llm:
            spec = frozen.get(f"{size}/{mk}")
            if spec is None:
                continue
            ev_sub = {"gold": gold, "labels": present, "enc": {size: enc[size]}, "llm": {mk: llm[mk]},
                      "counts": ev["counts"]}

            def router(rule, param):
                def f(s):
                    vals = []
                    for p, c in enc[size]:
                        pred, _ = apply(rule, param, p, c, llm[mk], ev["counts"][size])
                        vals.append(macro_f1(gold, pred, present, s))
                    return float(np.mean(vals))
                return f

            sel_rule = spec["selected"]
            sel = router(sel_rule, _param(spec["families"][sel_rule]["param"]))
            conf = router("conf", spec["families"]["conf"]["param"])
            enc_only, llm_only = router("enc", None), router("llm", None)
            costs = {}
            for name, (rule, param) in {"selected": (sel_rule, _param(spec["families"][sel_rule]["param"])),
                                        "conf": ("conf", spec["families"]["conf"]["param"])}.items():
                costs[name] = round(float(np.mean([apply(rule, param, p, c, llm[mk], ev_sub["counts"][size])[1]
                                                   for p, c in enc[size]])), 3)
            full = np.arange(n)
            report["routers"][f"{size}/{mk}"] = {
                "selected_rule": sel_rule, "selected_param": spec["families"][sel_rule]["param"],
                "macro_f1": {"enc_only": round(enc_only(full), 4), "llm_only": round(llm_only(full), 4),
                             "selected": round(sel(full), 4), "conf": round(conf(full), 4)},
                "cost": costs,
                "selected_minus_enc": paired(sel, enc_only, n, samples),
                "selected_minus_llm": paired(sel, llm_only, n, samples),
                "conf_minus_enc": paired(conf, enc_only, n, samples),
                "conf_minus_llm": paired(conf, llm_only, n, samples)}
    return report


def _param(p):
    return tuple(p) if isinstance(p, list) else p


def verdicts(r):
    def cell(size, name, mk):
        return r["bins"].get(f"{size}/{name}", {}).get("llm", {}).get(mk, {}).get("minus_cb")

    def holds(c, sign):
        return c is not None and (c["ci95"][0] > 0 if sign > 0 else c["ci95"][1] < 0)

    h1 = all(holds(cell(s, "1-5", m), +1) for s in SIZES for m in TEST_MODELS)
    h2a = all(holds(cell(s, "21-89", m), -1) for s in ("100k", "full") for m in TEST_MODELS)
    h2b = all(holds(cell("1k", "21-89", m), +1) for m in TEST_MODELS)
    routers = {}
    for size in ROUTER_SIZES:
        rows = [r["routers"].get(f"{size}/{m}") for m in TEST_MODELS]
        routers[size] = all(x is not None and x["selected_minus_enc"]["ci95"][0] > 0
                            and x["selected_minus_llm"]["ci95"][0] > 0 for x in rows)
    return {"H1_llm_wins_1_5_all_N": h1, "H2a_cb_wins_21_89_from_100k": h2a,
            "H2b_llm_wins_21_89_at_1k": h2b, "router_beats_both": routers}


def show(r, title):
    print(f"===== {title} (n={r['n']})")
    for k, c in r["bins"].items():
        cells = "  ".join(f"{mk} {v['macro_f1']:.3f} ({v['minus_cb']['point']:+.3f} "
                          f"[{v['minus_cb']['ci95'][0]:+.3f}, {v['minus_cb']['ci95'][1]:+.3f}]"
                          f"{'*' if v['minus_cb']['excludes_zero'] else ''})" for mk, v in c["llm"].items())
        print(f"   {k:11} {c['n_codes']:3} codes {c['n_items']:4} items  CB {c['cb']:.3f}  {cells}")
    for k, v in r["routers"].items():
        m = v["macro_f1"]
        print(f"   router {k:16} enc {m['enc_only']:.3f} llm {m['llm_only']:.3f} "
              f"{v['selected_rule']}{v['selected_param']} {m['selected']:.3f} (cost {v['cost']['selected']}) "
              f"conf {m['conf']:.3f} (cost {v['cost']['conf']})")
        for c in ("selected_minus_enc", "selected_minus_llm", "conf_minus_enc", "conf_minus_llm"):
            x = v[c]
            print(f"      {c:20} {x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
                  f"{'  *' if x['excludes_zero'] else ''}")
    print("   verdicts:", json.dumps(r["verdicts"]))


def main():
    report = {"reproducibility": reproducibility()}
    ev = load_eval("2025")
    report["main"] = evaluate(ev)
    report["main"]["verdicts"] = verdicts(report["main"])
    nd = OUT / "neardup.json"
    if nd.exists():
        cos = json.loads(nd.read_text())
        keep = np.array([cos[str(k)][0] < 0.9 for k in ev["keys"]])
        report["neardup_excluded"] = evaluate(ev, keep)
        report["neardup_excluded"]["verdicts"] = verdicts(report["neardup_excluded"])
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1))
    print("===== reproducibility (retrained CB vs original, 2024 test):")
    for size, rows in report["reproducibility"].items():
        print(f"   {size:5} " + "  ".join(f"agree {x['agreement']:.3f} F1 {x['macro_f1_original']:.3f}->"
                                        f"{x['macro_f1_rerun']:.3f}" for x in rows))
    show(report["main"], "2025 out-of-time")
    if "neardup_excluded" in report:
        show(report["neardup_excluded"], "2025 without near-duplicates")


if __name__ == "__main__":
    main()
