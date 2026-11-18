"""Why the gate works: the analyses of notes/why-it-works-plan-2026-11-16.md (CPU; experiment 3's GPU runs come from
run_cand5.ps1 and are scored here once their files exist).

  1 mechanism  The gate keeps an LLM's rare answer only when the encoder ranks the same code best among the rare codes
               of its top five. Exact whole-year counts give, per confirmer (zero-shot, retrieval):
                 keep_right  share of the LLM's correct rare answers that pass the encoder's check
                 keep_false  share of the LLM's false rare answers that pass (denominator estimated from the natural
                             sample, as in paper_v4_numbers.py; numerator exact)
               and from the other side, on the reports the encoder shortlists (rare candidate c in its top five):
                 confirm_right / confirm_false  share of shortlisted reports on which the LLM names c, when c is the
                             filed code / when it is not
               Base rate: a placement's precision equals prior odds x likelihood ratio, LR = (right / rare reports) /
               (false / other reports); the LR of the encoder, the LLMs alone and the gate are reported.
  2 ablation   k = 1, 3, 5, 10; confirmer zero-shot / retrieval / either / both; veto of unconfirmed rare answers on or
               off: right, false, precision, rare modes found and false alarms (whole year, exact; cells whose LLM
               answers do not cover the shortlist are marked incomplete)
  3 shown      the LLM shown the encoder's top five (+ NONE_OF_THESE) picks one; a rare code is placed when it picks
               the encoder's best rare candidate; compared with the gate on the same reports
  4 stability  right and false placements by month (report dates) and by group (MAUDE device product code from FDA's
               device file, SDR ATA chapter = first two digits of the JASC code)
  5 errors     the gate's false placements: MAUDE, where the placed code sits relative to the filed code in the IMDRF
               tree; SDR, whether the placed condition term is written in the narrative; plus a seeded sample of
               false placements with their narratives for reading

Usage: python why_it_works.py [parts, default 1245 + 3 when files exist]  (MAINT_CORPUS=sdr for SDR)
Output: results/why_it_works/<corpus>.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

import maude_llm as ml  # noqa: E402
import rare_year as ry  # noqa: E402
import reliability_year as rl  # noqa: E402
from common import CORPUS, ROOT, rdir  # noqa: E402
from maude_data import POOL_2025, POOL_2026, load_items, load_split  # noqa: E402

LLMS = ("qwen38_27b", "gemma4_31b")
YEARS = ("2026", "2025")
OUT = ROOT / "results" / "why_it_works"


# ---------------------------------------------------------------- whole-year data
def load_year(year, split, legal, p2c, path_of):
    z = np.load(ry.ENC / f"{year}.npz")
    cls = [str(c) for c in z["classes"]]
    c2i = {c: i for i, c in enumerate(cls)}
    keys, gold, top = z["keys"], z["gold"].astype(int), z["top_idx"].astype(int)
    N = len(keys)
    is_rare = np.zeros(len(cls), bool)
    is_rare[z["rare"].astype(int)] = True
    Y = {"year": year, "cls": cls, "c2i": c2i, "paths": [path_of.get(c, c) for c in cls], "keys": keys, "gold": gold,
         "top": top, "enc": top[:, 0], "best": z["rare_best"].astype(int), "rank": z["rare_rank"].astype(int),
         "is_rare": is_rare, "nonrare": top[np.arange(N), (~is_rare[top]).argmax(1)], "N": N, "A": {}}
    Y["names"] = [p.split(" > ")[-1] for p in Y["paths"]]
    pos = {int(k): i for i, k in enumerate(keys)}
    Y["pos"] = pos
    for mk in LLMS:
        Y["A"][mk] = {}
        for name, kind in ry.KINDS.items():
            a = np.full(N, -2)
            for k, v in ry.llm_answers(year, mk, kind, c2i, legal, p2c).items():
                if k in pos:
                    a[pos[k]] = v
            Y["A"][mk][name] = a
    Y["day"] = rl.report_days(year, keys)
    return Y


def gate(Y, mk, k=5, conf="either", veto=True, pick=None):
    """Final codes of the gate variant; pick (optional) replaces the confirmers with a shown-candidates pick."""
    best, enc = Y["best"], Y["enc"]
    short = Y["rank"] < k
    if pick is not None:
        c = pick == best
    else:
        zs, rag = Y["A"][mk]["zs"], Y["A"][mk]["rag"]
        c = {"zs": zs == best, "rag": rag == best, "either": (zs == best) | (rag == best),
             "both": (zs == best) & (rag == best)}[conf]
    fire = short & c
    other = np.where(Y["is_rare"][enc], Y["nonrare"], enc) if veto else enc
    return np.where(fire, best, other)


def coverage(Y, mk, k, conf):
    short = Y["rank"] < k
    names = {"zs": ["zs"], "rag": ["rag"]}.get(conf, ["zs", "rag"])
    return min(float((Y["A"][mk][n][short] > -2).mean()) for n in names) if short.any() else 1.0


def counts(Y, final):
    m = rl.mode_stats(final, Y["gold"], Y["is_rare"], Y["names"], None)
    return {k: m[k] for k in ("right", "false", "visible_modes", "phantom_modes")} | {
        "precision": round(m["right"] / (m["right"] + m["false"]), 4) if m["right"] + m["false"] else None}


# ---------------------------------------------------------------- 1 mechanism
def natural_false(year, mk, kind):
    """Estimated whole-year false rare placements of an LLM used alone (paper_v4_numbers.py's frozen estimate)."""
    J = json.loads((ROOT / "results" / "paper_v4" / f"{CORPUS}_numbers.json").read_text(encoding="utf-8"))
    m = J["llms"][mk][year]["methods"][kind]
    return float(m["rare_false"]), m["rare_false_ci95"], float(m["rare_right"])


def mechanism(Y, mk):
    gold, best, rank, is_rare = Y["gold"], Y["best"], Y["rank"], Y["is_rare"]
    n_rare = int(is_rare[gold].sum())
    n_other = Y["N"] - n_rare
    short = rank < 5
    out = {"rare_reports": n_rare, "other_reports": n_other, "prior_odds": n_rare / n_other, "confirmers": {}}
    for name in ("zs", "rag"):
        a = Y["A"][mk][name]
        has = a > -2
        rare_ans = has & (a >= 0) & is_rare[np.maximum(a, 0)]
        right = rare_ans & (a == gold)
        false = rare_ans & (a != gold)
        passes = short & (a == best)
        f_hat, f_ci, r_nat = natural_false(Y["year"], mk, name)
        kr = float((right & passes).sum()) / max(1, int(right.sum()))
        kf = float((false & passes).sum()) / f_hat if f_hat else None
        sc_right = short & (best == gold)
        sc_false = short & (best != gold)
        out["confirmers"][name] = {
            "llm_right": int(right.sum()), "llm_right_check_natural": r_nat,
            "llm_false_estimated": round(f_hat, 1), "llm_false_ci95": f_ci,
            "right_passing": int((right & passes).sum()), "false_passing": int((false & passes).sum()),
            "keep_right": round(kr, 4), "keep_false": round(kf, 5) if kf is not None else None,
            "filter_factor": round(kr / kf, 1) if kf else None,
            "shortlisted_candidate_right": int(sc_right.sum()), "shortlisted_candidate_false": int(sc_false.sum()),
            "confirm_right": round(float((a[sc_right] == best[sc_right]).mean()), 4) if sc_right.any() else None,
            "confirm_false": round(float((a[sc_false] == best[sc_false]).mean()), 5) if sc_false.any() else None}
        c = out["confirmers"][name]
        if c["confirm_false"]:
            c["confirm_ratio"] = round(c["confirm_right"] / c["confirm_false"], 1)
    lr = {}
    for nm, final in (("encoder", Y["enc"]), ("gate", gate(Y, mk))):
        m = counts(Y, final)
        lr[nm] = {"right": m["right"], "false": m["false"],
                  "LR": round((m["right"] / n_rare) / (m["false"] / n_other), 1) if m["false"] else None}
    for name in ("zs", "rag"):
        c = out["confirmers"][name]
        lr[f"llm_{name}_alone"] = {"right": c["llm_right"], "false": c["llm_false_estimated"],
                                   "LR": round((c["llm_right"] / n_rare) / (c["llm_false_estimated"] / n_other), 1)}
    out["likelihood_ratio"] = lr
    return out


# ---------------------------------------------------------------- 2 ablation
def ablation(Y, mk):
    rows = []
    for k in (1, 3, 5, 10):
        for conf in ("zs", "rag", "either", "both"):
            cov = coverage(Y, mk, k, conf)
            for veto in (True, False):
                r = {"k": k, "confirmer": conf, "veto": veto, "coverage": round(cov, 4), "complete": cov == 1.0}
                if r["complete"]:
                    r |= counts(Y, gate(Y, mk, k, conf, veto))
                rows.append(r)
    return {"encoder": counts(Y, Y["enc"]), "rows": rows}


# ---------------------------------------------------------------- 3 shown candidates
def shown(Y, mk):
    p = rdir(f"maude{Y['year']}short") / f"llm_cand5_{mk}.jsonl"
    if not p.exists():
        return None
    pick = np.full(Y["N"], -2)
    path_to_i = {pth: i for i, pth in enumerate(Y["paths"])}
    n_none = n_bad = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["type"] != "succeeded":
            continue
        i = Y["pos"].get(int(r["custom_id"][1:]))
        if i is None:
            continue
        cands = [Y["paths"][c] for c in Y["top"][i, :5]]
        text = r.get("text", "")
        if "NONE_OF_THESE" in text.upper():
            pick[i] = -1
            n_none += 1
            continue
        hit = ml.normalize(text, cands)
        if hit is None:
            n_bad += 1
        pick[i] = path_to_i[hit] if hit else -1
    short = Y["rank"] < 5
    cov = float((pick[short] > -2).mean())
    out = {"answers": int((pick > -2).sum()), "shortlisted": int(short.sum()), "coverage": round(cov, 4),
           "none_of_these": n_none, "unparsed": n_bad,
           "picks_rare_candidate": round(float((pick[short] == Y["best"][short]).mean()), 4) if cov == 1.0 else None}
    if cov == 1.0:
        sel = short
        out["shown"] = counts(Y, gate(Y, mk, pick=pick))
        out["gate_zs"] = counts(Y, gate(Y, mk, conf="zs"))
        out["gate_either"] = counts(Y, gate(Y, mk))
        zs = Y["A"][mk]["zs"]
        out["free_answer_equals_rare_candidate"] = round(float((zs[sel] == Y["best"][sel]).mean()), 4)
    return out


# ---------------------------------------------------------------- 4 stability
def product_codes(year, keys):
    """MAUDE: DEVICE_REPORT_PRODUCT_CODE of each report (first device row), cached in data/maude/product_<year>.json."""
    cache = ROOT / "data" / "maude" / f"product_{year}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return np.array([d.get(str(int(k)), "") for k in keys])
    wanted = {int(k) for k in keys}
    got = {}
    for f in ("device2025.zip", "device.zip"):
        with zipfile.ZipFile(ROOT / "data" / "maude" / "device_files" / f) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
            with zf.open(name) as raw:
                fh = io.TextIOWrapper(raw, encoding="latin-1", newline="")
                head = fh.readline().rstrip("\r\n").split("|")
                ik, ip = head.index("MDR_REPORT_KEY"), head.index("DEVICE_REPORT_PRODUCT_CODE")
                for line in fh:
                    q = line.split("|")
                    if len(q) <= ip:
                        continue
                    try:
                        k = int(q[ik])
                    except ValueError:
                        continue
                    if k in wanted and k not in got and q[ip].strip():
                        got[k] = q[ip].strip()
    cache.write_text(json.dumps({str(k): v for k, v in got.items()}), encoding="utf-8")
    return np.array([got.get(int(k), "") for k in keys])


def ata_chapters(year, keys):
    d = {}
    with open(POOL_2026 if year == "2026" else POOL_2025, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            d[int(r["report_key"])] = str(r.get("jasc") or "")[:2]
    return np.array([d.get(int(k), "") for k in keys])


def stability(Y, mk, groups):
    gold, rare = Y["gold"], Y["is_rare"]
    g = gate(Y, mk)

    def rf(final, m):
        placed = rare[final] & m
        right = placed & (final == gold)
        return int(right.sum()), int((placed & ~right).sum())
    out = {"months": [], "groups": {}}
    if Y["day"] is not None:
        month = np.array([(np.datetime64(f"{Y['year']}-01-01") + np.timedelta64(int(d), "D")).astype(object).month
                          if d >= 0 else 0 for d in Y["day"]])
        for m in range(1, 13):
            sel = month == m
            if not sel.any():
                continue
            (er, ef), (gr, gf) = rf(Y["enc"], sel), rf(g, sel)
            out["months"].append({"month": m, "reports": int(sel.sum()), "encoder": [er, ef], "gate": [gr, gf]})
        M = out["months"]
        out["month_summary"] = {"months": len(M), "gate_right_ge": sum(r["gate"][0] >= r["encoder"][0] for r in M),
                                "gate_false_le": sum(r["gate"][1] <= r["encoder"][1] for r in M),
                                "both": sum(r["gate"][0] >= r["encoder"][0] and r["gate"][1] <= r["encoder"][1] for r in M)}
    rows = []
    for grp in sorted(set(groups.tolist())):
        sel = groups == grp
        (er, ef), (gr, gf) = rf(Y["enc"], sel), rf(g, sel)
        if er or ef or gr or gf:
            rows.append([grp, int(sel.sum()), er, ef, gr, gf])
    gain = [r[4] - r[2] for r in rows]
    tot = sum(gain)
    pos = sorted([x for x in gain if x > 0], reverse=True)
    out["groups"] = {"with_placements": len(rows), "right_gain_total": tot, "groups_gaining": len(pos),
                     "groups_losing": sum(x < 0 for x in gain),
                     "top3_share_of_positive_gain": round(sum(pos[:3]) / sum(pos), 3) if pos else None,
                     "groups_false_down": sum(r[5] < r[3] for r in rows), "groups_false_up": sum(r[5] > r[3] for r in rows),
                     "rows": sorted(rows, key=lambda r: -(r[4] - r[2]))}
    return out


# ---------------------------------------------------------------- 5 errors
def relation(pa, pb):
    a, b = pa.split(" > "), pb.split(" > ")
    if a[:-1] == b[:-1]:
        return "sibling"
    if a[:len(b)] == b or b[:len(a)] == a:
        return "ancestor_or_descendant"
    if a[0] == b[0]:
        return "same_top_level"
    return "unrelated"


def errors(Y, mk, texts):
    gold, rare = Y["gold"], Y["is_rare"]
    out = {}
    for nm, final in (("encoder", Y["enc"]), ("gate", gate(Y, mk))):
        f = np.nonzero(rare[final] & (final != gold))[0]
        if CORPUS == "maude":
            rel = [relation(Y["paths"][final[i]], Y["paths"][gold[i]]) for i in f]
            out[nm] = {r: rel.count(r) for r in ("sibling", "ancestor_or_descendant", "same_top_level", "unrelated")}
        else:
            said = [Y["names"][final[i]].lower()[:5] in texts[int(Y["keys"][i])].lower() for i in f]
            out[nm] = {"false": len(f), "placed_term_in_narrative": int(sum(said))}
    f = np.nonzero(rare[gate(Y, mk)] & (gate(Y, mk) != gold))[0]
    rng = np.random.default_rng(42)
    pick = rng.choice(f, size=min(20, len(f)), replace=False) if len(f) else []
    g = gate(Y, mk)
    out["sample"] = [{"key": int(Y["keys"][i]), "placed": Y["paths"][g[i]], "filed": Y["paths"][gold[i]],
                      "encoder_top1": Y["paths"][Y["enc"][i]], "text": texts[int(Y["keys"][i])][:600]} for i in pick]
    return out


def main():
    parts = sys.argv[1] if len(sys.argv) > 1 else "12345"
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{CORPUS}.json"
    rep = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "note": "notes/why-it-works-plan-2026-11-16.md", "corpus": CORPUS, "years": {}}
    for year in YEARS:
        Y = load_year(year, split, legal, p2c, path_of)
        R = rep["years"].setdefault(year, {})
        groups = (product_codes(year, Y["keys"]) if CORPUS == "maude" else ata_chapters(year, Y["keys"])) if "4" in parts else None
        texts = None
        if "5" in parts:
            need = set()
            for mk in LLMS:
                g = gate(Y, mk)
                need |= {int(Y["keys"][i]) for i in np.nonzero(Y["is_rare"][g] & (g != Y["gold"]))[0]}
            need |= {int(Y["keys"][i]) for i in np.nonzero(Y["is_rare"][Y["enc"]] & (Y["enc"] != Y["gold"]))[0]}
            texts = {k: v[0] for k, v in load_items(need, pool=POOL_2026 if year == "2026" else POOL_2025).items()}
        for mk in LLMS:
            M = R.setdefault(mk, {})
            if "1" in parts:
                M["mechanism"] = mechanism(Y, mk)
            if "2" in parts:
                M["ablation"] = ablation(Y, mk)
            if "3" in parts:
                M["shown"] = shown(Y, mk)
            if "4" in parts:
                M["stability"] = stability(Y, mk, groups)
            if "5" in parts:
                M["errors"] = errors(Y, mk, texts)
            print(f"[{CORPUS} {year} {mk}] done parts {parts}", flush=True)
        out_path.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
