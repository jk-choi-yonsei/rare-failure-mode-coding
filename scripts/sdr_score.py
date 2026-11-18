"""Score the SDR replication on the 2025 test (notes/prereg-sdr-2026-10-24.md), once.

Verdicts need the item-level paired bootstrap (B=1000, seed 42) and the two-stage
code-level interval both clear of zero; a hypothesis holds only if every one of its
cells holds.
  S1  zero-shot LLM - CB-base > 0 in bin 1-5 at N in {1k, 10k, full}, Qwen and Gemma
  S2  CB-base - zero-shot LLM > 0 in bins 21-89 and 90+ at full N, Qwen and Gemma
  S3  Opus - CB-base > 0 in bin 1-5 at full N; Opus - Qwen > 0 overall at full N
  S4  retrieval - zero-shot > 0 overall at full N, Qwen, Gemma and Opus
  S6b keyword rule - best model < 0 overall (the best model by 2025 overall macro-F1 is
      named in the report; the contrast is against every arm, the verdict on the best)
  S6a on the leakage-free subset (gold term's first five characters absent from the
      narrative): S1 at full N (Qwen, Gemma) keeps its sign with intervals clear of zero
S5 (the selector) is scored by maude_selector_score.py under MAINT_CORPUS=sdr; S6a's
selector part reads that selector's predictions if its frozen file exists.
CB-base at a size is the seed mean of the available seeds (1k, 10k: 3; full: 1).

Usage: MAINT_CORPUS=sdr python sdr_score.py  -> runs/sdr2025/sdr_metrics.json
"""
import csv
import json
import shutil

import numpy as np

from common import CORPUS, RUNS, SEED, rdir
from maude_claude_score import code_ci, per_code_f1
from maude_router import load_eval, macro_f1

if CORPUS != "sdr":
    raise SystemExit("set MAINT_CORPUS=sdr")

B = 1000
BINS = {"1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
OPEN = ("qwen38_27b", "gemma4_31b")
OUT = rdir("maude2025") / "sdr_metrics.json"


def resamples(n):
    rng = np.random.default_rng(SEED)
    return [rng.integers(0, n, n) for _ in range(B)]


def item_ci(gold, a_list, b_list, labels, samples):
    """Seed-averaged per-code F1 means (macro-F1) difference, item bootstrap."""
    def m(preds, s=None):
        return float(per_code_f1(gold, preds, labels, s).mean())
    d = sorted(m(a_list, s) - m(b_list, s) for s in samples)
    return {"point": round(m(a_list) - m(b_list), 4), "ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)],
            "excludes_zero": d[round(0.025 * B) - 1] > 0 or d[round(0.975 * B) - 1] < 0}


def contrast(gold, a_list, b_list, labels, samples):
    return {"item": item_ci(gold, a_list, b_list, labels, samples), "code": code_ci(gold, a_list, b_list, labels, samples)}


def sign(c):
    lo_i, hi_i = c["item"]["ci95"]
    lo_c, hi_c = c["code"]["ci95"]
    return "+" if lo_i > 0 and lo_c > 0 else "-" if hi_i < 0 and hi_c < 0 else "0"


def load_llm_file(ev, path):
    import maude_llm as ml
    from maude_data import load_split
    split = load_split()
    legal = sorted(ml.known_labels(split))
    path_to_code = ml.known_labels(split)
    idx = {c: i for i, c in enumerate(ev["codes"])}
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[int(r["custom_id"][1:])] = r
    if any(k not in rows for k in ev["keys"]):
        return None
    pred = []
    for k in ev["keys"]:
        r = rows[k]
        code = path_to_code.get(ml.normalize(r.get("text", ""), legal)) if r["type"] == "succeeded" else None
        pred.append(idx[code] if code else -1)
    return np.array(pred)


def main():
    ev = load_eval("2025")
    gold, labels, n = ev["gold"], ev["labels"], len(ev["gold"])
    samples = resamples(n)
    counts = ev["counts"]
    bins = {size: {b: np.array([c for c in labels if lo <= counts[size][c] <= hi]) for b, (lo, hi) in BINS.items()}
            for size in ("1k", "10k", "full")}
    enc = {size: [p for p, _ in ev["enc"].get(size, [])] for size in ("1k", "10k", "full")}
    zs = {mk: ev["llm"].get(mk) for mk in OPEN + ("opus55",)}
    rag = {mk: load_llm_file(ev, rdir("maude2025") / f"llm_ragfull_{mk}.jsonl") for mk in OPEN + ("opus55",)}
    idx = {c: i for i, c in enumerate(ev["codes"])}
    kw_rows = {}
    with (rdir("maude2025") / "keyword_2025.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            kw_rows[int(r["key"])] = r
    keyword = np.array([idx.get(kw_rows[k]["pred"], -1) if kw_rows[k]["pred"] != "-1" else -1 for k in ev["keys"]])
    leaked = np.array([int(kw_rows[k]["leaked"]) for k in ev["keys"]])
    clean = np.where(leaked == 0)[0]

    def table(pred_list):
        return {"all": round(float(per_code_f1(gold, pred_list, labels).mean()), 4),
                **{b: round(float(per_code_f1(gold, pred_list, bins["full"][b]).mean()), 4) for b in BINS}}
    rep = {"n_2025": n, "codes_present": int(len(labels)), "leak_free_items": int(len(clean)),
           "arms_2025_macro_f1_bins_by_full_counts": {}, "contrasts": {}, "verdicts": {}, "missing": []}
    for size in ("1k", "10k", "full"):
        if enc[size]:
            rep["arms_2025_macro_f1_bins_by_full_counts"][f"cb_base_{size}"] = table(enc[size])
        else:
            rep["missing"].append(f"cb_base_{size}")
    for mk in OPEN + ("opus55",):
        for name, arr in ((f"zs_{mk}", zs[mk]), (f"rag_{mk}", rag[mk])):
            if arr is None:
                rep["missing"].append(name)
            else:
                rep["arms_2025_macro_f1_bins_by_full_counts"][name] = table([arr])
    rep["arms_2025_macro_f1_bins_by_full_counts"]["keyword"] = table([keyword])
    large = [p for p, _ in ev["enc"].get("full", [])]  # placeholder, CB-large read below if present
    for size in ("10k", "full"):
        p = rdir("maude2025") / f"encoder_cb-large_{size}_seed42.csv"
        if p.exists():
            from maude_extra_score import read_enc_csv
            rep["arms_2025_macro_f1_bins_by_full_counts"][f"cb_large_{size}"] = table([read_enc_csv(ev, p)[0]])

    # S1 / S2
    s1, s2 = {}, {}
    for mk in OPEN:
        if zs[mk] is None:
            continue
        for size in ("1k", "10k", "full"):
            if enc[size]:
                s1[f"{mk}_{size}"] = contrast(gold, [zs[mk]], enc[size], bins[size]["1-5"], samples)
        if enc["full"]:
            for b in ("21-89", "90+"):
                s2[f"{mk}_{b}"] = contrast(gold, enc["full"], [zs[mk]], bins["full"][b], samples)
    rep["contrasts"]["S1"], rep["contrasts"]["S2"] = s1, s2
    rep["verdicts"]["S1"] = ("holds" if s1 and len(s1) == 6 and all(sign(c) == "+" for c in s1.values())
                             else "incomplete" if len(s1) < 6 else "fails")
    rep["verdicts"]["S2"] = ("holds" if s2 and len(s2) == 4 and all(sign(c) == "+" for c in s2.values())
                             else "incomplete" if len(s2) < 4 else "fails")
    # S3
    s3 = {}
    if zs["opus55"] is not None and enc["full"]:
        s3["opus_minus_cb_1-5"] = contrast(gold, [zs["opus55"]], enc["full"], bins["full"]["1-5"], samples)
    if zs["opus55"] is not None and zs["qwen38_27b"] is not None:
        s3["opus_minus_qwen_all"] = contrast(gold, [zs["opus55"]], [zs["qwen38_27b"]], labels, samples)
    rep["contrasts"]["S3"] = s3
    rep["verdicts"]["S3"] = "holds" if len(s3) == 2 and all(sign(c) == "+" for c in s3.values()) else "incomplete" if len(s3) < 2 else "fails"
    # S4
    s4 = {mk: contrast(gold, [rag[mk]], [zs[mk]], labels, samples) for mk in OPEN + ("opus55",) if rag[mk] is not None and zs[mk] is not None}
    rep["contrasts"]["S4"] = s4
    rep["verdicts"]["S4"] = "holds" if len(s4) == 3 and all(sign(c) == "+" for c in s4.values()) else "incomplete" if len(s4) < 3 else "fails"
    if rag["opus55"] is not None and enc["full"]:
        rep["contrasts"]["desc_opus_rag_minus_cb_all"] = contrast(gold, [rag["opus55"]], enc["full"], labels, samples)
    # S6b: keyword vs best model overall
    scores = {k: v["all"] for k, v in rep["arms_2025_macro_f1_bins_by_full_counts"].items() if k != "keyword"}
    best = max(scores, key=scores.get) if scores else None
    arm_pred = {}
    for size in ("1k", "10k", "full"):
        if enc[size]:
            arm_pred[f"cb_base_{size}"] = enc[size]
    for mk in OPEN + ("opus55",):
        if zs[mk] is not None:
            arm_pred[f"zs_{mk}"] = [zs[mk]]
        if rag[mk] is not None:
            arm_pred[f"rag_{mk}"] = [rag[mk]]
    if best in arm_pred:
        rep["contrasts"]["S6b_keyword_minus_best"] = {"best": best, **contrast(gold, [keyword], arm_pred[best], labels, samples)}
        rep["verdicts"]["S6b"] = "holds" if sign(rep["contrasts"]["S6b_keyword_minus_best"]) == "-" else "fails"
    # S6a: leakage-free subset, S1 at full N
    s6a = {}
    if len(clean):
        sub_samples = [clean[np.random.default_rng(SEED + 7 + i).integers(0, len(clean), len(clean))] for i in range(B)]
        gc = gold
        lab_clean = np.array([c for c in bins["full"]["1-5"] if c in set(gold[clean].tolist())])
        for mk in OPEN:
            if zs[mk] is not None and enc["full"] and len(lab_clean):
                def mf(preds, s):
                    return float(per_code_f1(gc, preds, lab_clean, s).mean())
                d = sorted(mf([zs[mk]], s) - mf(enc["full"], s) for s in sub_samples)
                s6a[f"S1_{mk}_full_leakfree"] = {"point": round(mf([zs[mk]], clean) - mf(enc["full"], clean), 4),
                                                 "ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)],
                                                 "n_items": int(len(clean)), "n_codes_1-5_present": int(len(lab_clean))}
    rep["contrasts"]["S6a"] = s6a
    rep["verdicts"]["S6a"] = ("holds" if len(s6a) == 2 and all(v["ci95"][0] > 0 for v in s6a.values())
                              else "incomplete" if len(s6a) < 2 else "fails")
    rep["leaked_share_2025"] = round(float(leaked.mean()), 4)
    # Amendment (notes/prereg-sdr-amendment-2026-10-27.md): the full-N encoders retrained with
    # the 10k epoch budget, tagged "-long". Reported beside the pre-registered verdicts.
    from maude_extra_score import read_enc_csv
    amended = {}
    for tag in ("cb-long", "cb-large-long"):
        p = rdir("maude2025") / f"encoder_{tag}_full_seed42.csv"
        if p.exists():
            amended[tag] = read_enc_csv(ev, p)[0]
            rep["arms_2025_macro_f1_bins_by_full_counts"][tag.replace("-", "_") + "_full"] = table([amended[tag]])
    if "cb-long" in amended:
        enc_long = [amended["cb-long"]]
        s2a = {f"{mk}_{b}": contrast(gold, enc_long, [zs[mk]], bins["full"][b], samples)
               for mk in OPEN if zs[mk] is not None for b in ("21-89", "90+")}
        s3a = {"opus_minus_cb_long_1-5": contrast(gold, [zs["opus55"]], enc_long, bins["full"]["1-5"], samples)} if zs["opus55"] is not None else {}
        rep["contrasts"]["S2_amended"], rep["contrasts"]["S3_amended"] = s2a, s3a
        rep["verdicts"]["S2_amended"] = "holds" if len(s2a) == 4 and all(sign(c) == "+" for c in s2a.values()) else "incomplete" if len(s2a) < 4 else "fails"
        rep["verdicts"]["S3_amended"] = "holds" if s3a and sign(next(iter(s3a.values()))) == "+" else "incomplete" if not s3a else "fails"
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    res = RUNS.parent / "results" / rdir("maude2025").name / OUT.name
    res.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT, res)
    print(f"SDR 2025 test n={n}, codes {len(labels)}, leak-free items {len(clean)}; missing arms: {rep['missing']}")
    print(f"{'arm':20s} " + " ".join(f"{k:>7s}" for k in ("all", *BINS)))
    for name, t in rep["arms_2025_macro_f1_bins_by_full_counts"].items():
        print(f"{name:20s} " + " ".join(f"{t[k]:7.3f}" for k in ("all", *BINS)))
    for h, v in rep["verdicts"].items():
        print(f"{h}: {v}")
    for h in ("S1", "S2", "S3", "S4"):
        for k, c in rep["contrasts"][h].items():
            print(f"  {h} {k}: {c['item']['point']} item {c['item']['ci95']} code {c['code']['ci95']}")
    if "S6b_keyword_minus_best" in rep["contrasts"]:
        c = rep["contrasts"]["S6b_keyword_minus_best"]
        print(f"  S6b keyword - {c['best']}: {c['item']['point']} item {c['item']['ci95']} code {c['code']['ci95']}")
    for k, v in s6a.items():
        print(f"  S6a {k}: {v}")
    for h in ("S2_amended", "S3_amended"):
        for k, c in rep["contrasts"].get(h, {}).items():
            print(f"  {h} {k}: {c['item']['point']} item {c['item']['ci95']} code {c['code']['ci95']}")
    print(f"-> {OUT} (copied to results/)")


if __name__ == "__main__":
    main()
