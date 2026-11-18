"""Score the redesign once (notes/prereg-redesign-2026-11-06.md).

Tests D1-D8 on the test year (2026) and, with the same code, on the development year (2025);
on MAUDE also test E (never-seen modes, 2026; the 2024 hidden reports as the practice check).
One success rule for every cell: the item-level paired bootstrap (B = 1000, seed 42) and the
two-stage code-level interval (items, then the cell's codes; seed 43) must both clear zero in
the predicted direction; a hypothesis holds only if all its cells hold. Everything frozen is
read from splits/redesign_v1_<corpus>.json and the selector files it names (hash-checked).
F (natural 2026) is scored by maude_natural_score.py --year 2026 and maude_downstream.py --year 2026.

Usage: python redesign_score.py                    -> results/redesign/maude_metrics.json
       MAINT_CORPUS=sdr python redesign_score.py   -> results/redesign/sdr_metrics.json
"""
import csv
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import CORPUS, ROOT, SEED, SPLITS, rdir

FRZ = json.loads((SPLITS / f"redesign_v1_{CORPUS}.json").read_text(encoding="utf-8"))
os.environ["MAINT_SELECTOR_ENC"] = FRZ["selector_encoder"]  # before maude_selector is imported

import maude_llm as ml  # noqa: E402
from majority_vote_score import majority  # noqa: E402
from maude_claude_score import code_ci, per_code_f1  # noqa: E402
from maude_data import class_info, load_split  # noqa: E402
from maude_extra_score import read_enc_csv  # noqa: E402
from maude_router import load_eval, macro_f1  # noqa: E402
from maude_selector import (BINS, SELECTOR_ENC, gate, git_head, llm_rare, load_coders, load_frozen,  # noqa: E402
                            load_llm_file, predict_frozen, rule3, sha256, frozen_paths, targets)
from maude_selector_score import resamples  # noqa: E402

B = 1000
LLMS = ("opus55", "qwen38_27b", "gemma4_31b")
SIZES = ("1k", "10k", "100k", "full") if CORPUS == "maude" else ("1k", "10k", "full")
D6_SIZES = ("100k", "full") if CORPUS == "maude" else ("full",)
BASE_TAGS = {"1k": [f"cb_1k_seed{s}" for s in (42, 43, 44)], "10k": [f"cb_10k_seed{s}" for s in (42, 43, 44)],
             "100k": [f"cb-conv_100k_seed{s}" for s in (42, 43, 44)], "full": ["cb-conv_full_seed42"]}
YEARS = ("2026", "2025")
D6_BINS = {"21-89": (21, 89), "90+": (90, 10**9)}


# ---------------------------------------------------------------- the success rule
def item_ci(gold, A, Bl, labels, samples):
    """Seed-averaged macro-F1 difference over `labels`, paired item bootstrap."""
    def m(preds, s=None):
        return float(per_code_f1(gold, preds, labels, s).mean())
    d = sorted(m(A, s) - m(Bl, s) for s in samples)
    return {"point": round(m(A) - m(Bl), 4), "ci95": [round(d[round(0.025 * B) - 1], 4), round(d[round(0.975 * B) - 1], 4)]}


def contrast(gold, A, Bl, labels, samples):
    A = A if isinstance(A, list) else [A]
    Bl = Bl if isinstance(Bl, list) else [Bl]
    c = {"item": item_ci(gold, A, Bl, labels, samples), "code": code_ci(gold, A, Bl, labels, samples),
         "n_items": int(len(gold)), "n_codes": int(len(labels))}
    c["holds"] = bool(c["item"]["ci95"][0] > 0 and c["code"]["ci95"][0] > 0)
    return c


def verdict(cells):
    ran = [c for c in cells if c is not None]
    if not ran:
        return "not run"
    if len(ran) < len(cells):
        return (f"incomplete ({len(ran)}/{len(cells)} cells run; "
                + ("every run cell holds)" if all(c["holds"] for c in ran) else "a run cell fails)"))
    return "holds" if all(c["holds"] for c in ran) else "fails"


def mf(gold, pred, labels):
    return round(macro_f1(gold, pred, labels), 4) if len(labels) else None


# ---------------------------------------------------------------- D1-D4, D7 (and D8's D1)
def cascade(D, arm, theta):
    return np.where((D["enc_p"] < theta) & (D[arm] >= 0), D[arm], D["enc"])


def check_frozen(llm):
    fj, fb = frozen_paths(llm)
    want = FRZ["selectors"][llm]
    if sha256(fj) != want["sha256"] or sha256(fb) != want["joblib_sha256"]:
        raise SystemExit(f"{fj.name}: the frozen selector differs from splits/redesign_v1_{CORPUS}.json")
    frozen, models = load_frozen(llm)
    if frozen.get("encoder_tag", "cb-large") != SELECTOR_ENC:
        raise SystemExit(f"{fj.name} uses encoder {frozen.get('encoder_tag')}, not {SELECTOR_ENC}")
    return frozen, models


def selector_arms(year, llm):
    frozen, models = check_frozen(llm)
    try:
        D = load_coders(year, llm)
    except (SystemExit, FileNotFoundError) as e:
        return None, None, str(e)
    sel, j = predict_frozen(D, frozen, models)
    cz = FRZ["cascade"][llm]
    g = gate(D, frozen["S_cost"]["k"], frozen["S_cost"]["p"]) if frozen.get("S_cost") else np.zeros(len(sel), bool)
    arms = {"selector": sel, "best_single": D[frozen["best_single_arm_2024"]], "cascade": cascade(D, cz["arm"], cz["theta"]),
            "llm_rare": llm_rare(D, frozen["llm_rare_k_star"]["k"]), "mv_enc": majority(D, D["enc"]),
            "enc": D["enc"], "zs": D["zs"], "rag": D["rag"], "S_cost": np.where(g, sel, D["enc"]),
            "S1": rule3(D, frozen["S1"]["t"], frozen["S1"]["k"]),
            "oracle": np.where(targets(D).any(axis=1), D["gold"], D["enc"])}
    info = {"best_single_arm_2024": frozen["best_single_arm_2024"], "cascade": cz, "main_family": frozen["main_family"],
            "llm_rare_k": frozen["llm_rare_k_star"]["k"],
            "choice_share": {a: round(float((j == i).mean()), 4) for i, a in enumerate(("enc", "zs", "rag"))}}
    return D, arms, info


def selector_tests(D, arms, keep=None):
    idx = np.arange(len(D["gold"])) if keep is None else keep
    gold = D["gold"][idx]
    labels = np.array(sorted(set(gold.tolist())))
    counts = D["counts"]
    samples = resamples(len(idx))
    sub = {k: v[idx] for k, v in arms.items()}
    bins = {b: np.array([c for c in labels if lo <= counts[c] <= hi]) for b, (lo, hi) in BINS.items()}
    table = {k: {"all": mf(gold, p, labels), **{b: mf(gold, p, bins[b]) for b in BINS}} for k, p in sub.items()}
    tests = {"D1_selector_minus_best_single": contrast(gold, sub["selector"], sub["best_single"], labels, samples)}
    if keep is None:
        tests.update({"D2_selector_minus_cascade": contrast(gold, sub["selector"], sub["cascade"], labels, samples),
                      "D3_selector_minus_frequency_rule": contrast(gold, sub["selector"], sub["llm_rare"], labels, samples),
                      "D4_selector_minus_majority_vote": contrast(gold, sub["selector"], sub["mv_enc"], labels, samples),
                      "D7_retrieval_minus_zeroshot": contrast(gold, sub["rag"], sub["zs"], labels, samples)})
    return {"n": int(len(idx)), "codes": int(len(labels)), "macro_f1": table, "tests": tests}


# ---------------------------------------------------------------- D5, D6 (and D8's D5)
def boundary(year, keep=None):
    ev = load_eval(year)
    d = rdir("maude" + year)
    encs, missing = {}, []
    for n in SIZES:
        runs = []
        for t in BASE_TAGS[n]:
            p = d / f"encoder_{t}.csv"
            if p.exists():
                runs.append(read_enc_csv(ev, p)[0])
            else:
                missing.append(p.name)
        if len(runs) == len(BASE_TAGS[n]):
            encs[n] = runs
    zs = {}
    for llm in LLMS:
        pred, st = load_llm_file(ev, d / f"llm_cls_{llm}.jsonl")
        if pred is not None and st.get("complete"):
            zs[llm] = pred
        else:
            missing.append(f"llm_cls_{llm}.jsonl")
    idx = np.arange(len(ev["gold"])) if keep is None else keep
    gold = ev["gold"][idx]
    present = np.array(sorted(set(gold.tolist())))
    samples = resamples(len(idx))
    out = {"missing": missing, "D5": {}, "D6": {}, "macro_f1": {}}
    for n in SIZES:
        cnt = ev["counts"][n]
        for bname, (lo, hi) in {"1-5": (1, 5), "6-20": (6, 20), **D6_BINS}.items():
            labels = np.array([c for c in present if lo <= cnt[c] <= hi])
            if not len(labels):
                continue
            cell = {"n_codes": int(len(labels)), "n_items": int(np.isin(gold, labels).sum())}
            if n in encs:
                cell["encoder_base"] = round(float(np.mean([macro_f1(gold, p[idx], labels) for p in encs[n]])), 4)
            for llm, p in zs.items():
                cell[llm] = mf(gold, p[idx], labels)
            out["macro_f1"][f"{n}/{bname}"] = cell
            if n not in encs:
                continue
            for llm, p in zs.items():
                if bname == "1-5":
                    out["D5"][f"{n}/{llm}"] = contrast(gold, [p[idx]], [e[idx] for e in encs[n]], labels, samples)
                if bname in D6_BINS and n in D6_SIZES and keep is None:
                    out["D6"][f"{n}/{bname}/{llm}"] = contrast(gold, [e[idx] for e in encs[n]], [p[idx]], labels, samples)
    out["D5_cells_expected"] = [f"{n}/{llm}" for n in (("full",) if keep is not None else SIZES) for llm in LLMS]
    out["D6_cells_expected"] = [f"{n}/{b}/{llm}" for n in D6_SIZES for b in D6_BINS for llm in LLMS] if keep is None else []
    if keep is not None:  # D8 uses D5 at full N only
        out["D5"] = {k: v for k, v in out["D5"].items() if k.startswith("full/")}
    return out


def leak_free(year, keys):
    path = rdir("maude" + year) / f"keyword_{year}.csv"
    with path.open(encoding="utf-8") as fh:
        leaked = {int(r["key"]): int(r["leaked"]) for r in csv.DictReader(fh)}
    return np.array([i for i, k in enumerate(keys) if leaked[k] == 0])


# ---------------------------------------------------------------- E: never-seen modes (MAUDE)
def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(pos, neg, pos_codes, neg_codes):
    rng = np.random.default_rng(SEED)
    item = sorted(auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(B))
    rng2 = np.random.default_rng(SEED + 1)

    gp = [pos[pos_codes == c] for c in np.unique(pos_codes)]
    gn = [neg[neg_codes == c] for c in np.unique(neg_codes)]

    def two_stage(groups):
        pick = rng2.integers(0, len(groups), len(groups))
        return np.concatenate([groups[i][rng2.integers(0, len(groups[i]), len(groups[i]))] for i in pick])
    code = sorted(auroc(two_stage(gp), two_stage(gn)) for _ in range(B))
    lo, hi = round(0.025 * B) - 1, round(0.975 * B) - 1
    c = {"auroc": round(auroc(pos, neg), 4), "item_ci95": [round(item[lo], 4), round(item[hi], 4)],
         "code_ci95": [round(code[lo], 4), round(code[hi], 4)], "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    c["better_than_chance"] = bool(c["item_ci95"][0] > 0.5 and c["code_ci95"][0] > 0.5)
    return c


def enc_scores(path):
    with path.open(encoding="utf-8") as fh:
        return {int(r["key"]): (float(r["msp_novelty"]), float(r["energy_novelty"])) for r in csv.DictReader(fh)}


def llm_novelty(path, legal):
    if not path.exists():
        return None
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["type"] != "succeeded":
            continue
        _, nov = ml.parse_novel(r.get("text", ""), legal)
        out[int(r["custom_id"][1:])] = nov if nov is not None else 50.0
    return out


def novelty(practice=False):
    from maude_data import load_novel_2026, load_test_2026
    from maude_router import training_counts
    split = load_split()
    info = class_info(split)
    codes = sorted(info)
    counts = dict(zip(codes, training_counts(split, codes)["full"].tolist()))
    legal = sorted(ml.known_labels(split))
    if practice:
        known = [t for t in split["test"] if not info[t["code"]]["hidden"]]
        novel = [t for t in split["test"] if info[t["code"]]["hidden"]]
        enc = enc_scores(rdir("maude2025") / "rerun2024" / f"encoder_{SELECTOR_ENC}_full_seed42.csv")
        enc_n = enc
        llm_dir = rdir("maude")
    else:
        known, novel = load_test_2026(), load_novel_2026()
        enc = enc_scores(rdir("maude2026") / f"encoder_{SELECTOR_ENC}_full_seed42.csv")
        enc_n = enc_scores(rdir("maude2026") / f"encoder_novel_{SELECTOR_ENC}_full_seed42.csv")
        llm_dir = rdir("maude2026")
    scores = {"encoder_softmax": ({t["key"]: enc[t["key"]][0] for t in known}, {t["key"]: enc_n[t["key"]][0] for t in novel}),
              "encoder_energy": ({t["key"]: enc[t["key"]][1] for t in known}, {t["key"]: enc_n[t["key"]][1] for t in novel})}
    missing = []
    for llm in LLMS:
        nv = llm_novelty(llm_dir / f"llm_novel_{llm}.jsonl", legal)
        if nv is None or any(t["key"] not in nv for t in known + novel):
            missing.append(llm)
            continue
        scores[f"{llm}_novelty"] = ({t["key"]: nv[t["key"]] for t in known}, {t["key"]: nv[t["key"]] for t in novel})
    groups = {"all_known": known, "known_1-5": [t for t in known if 1 <= counts[t["code"]] <= 5],
              "known_6-89": [t for t in known if 6 <= counts[t["code"]] <= 89]}
    out = {"n_novel": len(novel), "novel_codes": len({t["code"] for t in novel}), "missing": missing,
           "groups": {g: {"n": len(v), "codes": len({t["code"] for t in v})} for g, v in groups.items()}, "auroc": {}}
    pos_codes = np.array([t["code"] for t in novel])
    for name, (sk, sn) in scores.items():
        pos = np.array([sn[t["key"]] for t in novel])
        for g, items in groups.items():
            neg = np.array([sk[t["key"]] for t in items])
            if practice:
                out["auroc"][f"{name}/{g}"] = {"auroc": round(auroc(pos, neg), 4)}
            else:
                out["auroc"][f"{name}/{g}"] = auroc_ci(pos, neg, pos_codes, np.array([t["code"] for t in items]))
    return out


# ---------------------------------------------------------------- main
def show(tag, c):
    print(f"  {tag:55s} {c['item']['point']:+.4f} item {c['item']['ci95']} code {c['code']['ci95']} "
          f"-> {'HOLDS' if c['holds'] else 'fails'}", flush=True)


def main():
    rep = {"prereg": "notes/prereg-redesign-2026-11-06.md", "corpus": CORPUS, "git_head_at_score": git_head(),
           "freeze": f"splits/redesign_v1_{CORPUS}.json", "selector_encoder": SELECTOR_ENC, "years": {}}
    for year in YEARS:
        print(f"\n===== {CORPUS} {year} ({'test year' if year == '2026' else 'development year'})", flush=True)
        Y = {"selectors": {}, "verdicts": {}}
        for llm in LLMS:
            D, arms, info = selector_arms(year, llm)
            if D is None:
                Y["selectors"][llm] = {"not_run": info}
                print(f"  {llm}: not run ({info})", flush=True)
                continue
            R = selector_tests(D, arms)
            R["frozen"] = info
            if CORPUS == "sdr":
                keep = leak_free(year, D["keys"])
                R["D8_leak_free"] = selector_tests(D, arms, keep)
            Y["selectors"][llm] = R
            print(f"  {llm}: macro-F1 " + ", ".join(f"{k} {v['all']}" for k, v in R["macro_f1"].items()), flush=True)
            for k, c in R["tests"].items():
                show(f"{llm} {k}", c)
            if CORPUS == "sdr":
                show(f"{llm} D8 D1 (leak-free)", R["D8_leak_free"]["tests"]["D1_selector_minus_best_single"])
        Y["boundary"] = boundary(year)
        for k, c in {**Y["boundary"]["D5"], **Y["boundary"]["D6"]}.items():
            show(("D5 " if k in Y["boundary"]["D5"] else "D6 ") + k, c)
        if CORPUS == "sdr":
            ev = load_eval(year)
            Y["boundary_leak_free"] = boundary(year, leak_free(year, ev["keys"]))
            for k, c in Y["boundary_leak_free"]["D5"].items():
                show("D8 D5 (leak-free) " + k, c)

        def cells(test):
            return [Y["selectors"][l]["tests"].get(test) if "tests" in Y["selectors"][l] else None for l in LLMS]
        for t in ("D1_selector_minus_best_single", "D2_selector_minus_cascade", "D3_selector_minus_frequency_rule",
                  "D4_selector_minus_majority_vote", "D7_retrieval_minus_zeroshot"):
            Y["verdicts"][t.split("_")[0]] = verdict(cells(t))
            Y["verdicts"][t.split("_")[0] + "_by_llm"] = {l: (c["holds"] if c else None) for l, c in zip(LLMS, cells(t))}
        b = Y["boundary"]
        Y["verdicts"]["D5"] = verdict([b["D5"].get(k) for k in b["D5_cells_expected"]])
        Y["verdicts"]["D6"] = verdict([b["D6"].get(k) for k in b["D6_cells_expected"]])
        for name, keys in (("D5", b["D5_cells_expected"]), ("D6", b["D6_cells_expected"])):
            Y["verdicts"][name + "_by_llm"] = {l: verdict([b[name].get(k) for k in keys if k.endswith("/" + l)]) for l in LLMS}
        if CORPUS == "sdr":
            d8 = [Y["selectors"][l]["D8_leak_free"]["tests"]["D1_selector_minus_best_single"] if "D8_leak_free" in Y["selectors"][l]
                  else None for l in LLMS]
            bl = Y["boundary_leak_free"]
            d8 += [bl["D5"].get(k) for k in bl["D5_cells_expected"]]
            Y["verdicts"]["D8"] = verdict(d8)
        print("  verdicts:", {k: v for k, v in Y["verdicts"].items() if not k.endswith("_by_llm")}, flush=True)
        rep["years"][year] = Y
    if CORPUS == "maude":
        print("\n===== E: never-seen modes", flush=True)
        rep["E_2026"] = novelty()
        rep["E_2024_practice"] = novelty(practice=True)
        for k, v in rep["E_2026"]["auroc"].items():
            print(f"  {k:40s} AUROC {v['auroc']:.3f} item {v['item_ci95']} code {v['code_ci95']} "
                  f"{'better than chance' if v['better_than_chance'] else 'not better than chance'}", flush=True)
        print("  2024 practice:", {k: v["auroc"] for k, v in rep["E_2024_practice"]["auroc"].items()}, flush=True)
    out = ROOT / "results" / "redesign"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{CORPUS}_metrics.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\nwrote {out / f'{CORPUS}_metrics.json'}")


if __name__ == "__main__":
    main()
