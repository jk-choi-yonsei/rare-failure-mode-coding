"""Rare codes over a whole year: the encoder shortlists, an LLM confirms
(notes/selector-vs-rules-2026-11-11.md, section "한 해 전체로 재는 방법").

Rule: take the rare code (1-5 training reports) the encoder ranks best for a report. If it is inside
the encoder's top k and the confirmer names the same code, the report gets that code; otherwise the
encoder's answer stays ("keep_encoder") or, in the "confirmed_only" variant, an unconfirmed rare
answer of the encoder is replaced by its best non-rare code. Everything is counted over every
known-code report of the year (maude_encoder_year.py), so no sampling weight is involved.

  shortlist    -> runs/maude<year>short/shortlist.json: the reports whose best rare code is inside the
                  encoder's top K_MAX and that have no LLM answer yet (the balanced test and the natural
                  sample have), sorted by that rank
  neighbours   -> runs/maude<year>short/neighbors_full.json: the ten nearest training reports of the
                  shortlisted reports with rank < RAG_K, computed as in maude_retrieval.py (CPU)
  score        -> results/rare_year/metrics.json

Usage: python rare_year.py shortlist | neighbours [--pilot] [--rag-k 10] | score [--suffix _pilot]
       MAINT_CORPUS=sdr python rare_year.py ...   (SDR, 2026-11-14: encoder cb-conv, runs/sdr_encyear, runs/sdr<year>short,
                                                    results/rare_year/sdr_metrics.json)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import maude_llm as ml  # noqa: E402
from common import CORPUS, ROOT, RUNS, rdir  # noqa: E402
from maude_data import (POOL_2025, POOL_2026, load_items, load_natural_2025, load_natural_2026, load_split,  # noqa: E402
                        load_test_2025, load_test_2026, train_keys)

# MAINT_ENC_TAG and MAINT_SHORT_SUFFIX (2026-11-18): the same steps for another encoder, in its own directories
TAG = os.environ.get("MAINT_ENC_TAG") or {"maude": "cb-large-conv_full_seed42", "sdr": "cb-conv_full_seed42"}[CORPUS]
ALT = os.environ.get("MAINT_SHORT_SUFFIX", "")
ENC = RUNS / f"{CORPUS}_encyear" / TAG
K_MAX, RAG_K, KS = 10, 5, (1, 3, 5, 10)
YEARS = ("2026", "2025")
LLMS = ("qwen38_27b", "gemma4_31b", "opus55")
KINDS = {"zs": "cls", "rag": "ragfull"}


def pool_of(year):
    return POOL_2026 if year == "2026" else POOL_2025


def have_llm(year):
    """Reports that already have LLM answers: the balanced test, and the natural sample when its zero-shot and
    retrieval answers of Qwen and Gemma are on file (SDR had no natural sample before 2026-11-14)."""
    from maude_data import SPLIT_NATURAL, SPLIT_NATURAL_2026
    test = load_test_2026() if year == "2026" else load_test_2025()
    keys = {t["key"] for t in test}
    nat_dir = rdir(f"maude{year}nat")
    if (SPLIT_NATURAL_2026 if year == "2026" else SPLIT_NATURAL).exists() and all(
            (nat_dir / f"llm_{kind}_{mk}.jsonl").exists() for mk in LLMS[:2] for kind in KINDS.values()):
        keys |= {t["key"] for t in (load_natural_2026() if year == "2026" else load_natural_2025())}
    if ALT:  # reports of the paper's shortlist that already hold both LLMs' zero-shot and retrieval answers
        sets = []
        for mk in LLMS[:2]:
            for kind in KINDS.values():
                p = rdir(f"maude{year}short") / f"llm_{kind}_{mk}.jsonl"
                sets.append({int(json.loads(l)["custom_id"][1:]) for l in p.read_text(encoding="utf-8").splitlines()
                             if '"succeeded"' in l} if p.exists() else set())
        keys |= set.intersection(*sets)
    return keys


def shortlist(suffix="", include_existing=False):
    for year in YEARS:
        p = ENC / f"{year}{suffix}.npz"
        if not p.exists():
            print(f"[{year}] {p.name} missing, skipped", flush=True)
            continue
        z = np.load(p)
        cls = [str(c) for c in z["classes"]]
        # each array is read once: an NpzFile decompresses the whole array at every access (the first run of
        # 2026-11-12 indexed z[...] per report and took about 20 minutes for the two years)
        keys, gold, rank, best = z["keys"], z["gold"], z["rare_rank"], z["rare_best"]
        have = set() if include_existing else have_llm(year)
        idx = np.where(rank < K_MAX)[0]
        items = sorted(({"key": int(keys[i]), "code": cls[gold[i]], "rank": int(rank[i]), "rare": cls[best[i]]}
                        for i in idx if int(keys[i]) not in have), key=lambda t: (t["rank"], t["key"]))
        counts = {"year_reports": int(len(keys)),
                  "rank_below_k": {str(k): int((rank < k).sum()) for k in KS},
                  "to_run_rank_below_k": {str(k): sum(t["rank"] < k for t in items) for k in KS}}
        d = ml.short_dir(year)
        d.mkdir(parents=True, exist_ok=True)
        (d / "shortlist.json").write_text(json.dumps({"note": "notes/selector-vs-rules-2026-11-11.md", "encoder": TAG, "source": p.name,
                                                      "k_max": K_MAX, "counts": counts, "items": items}), encoding="utf-8")
        print(f"[{year}] shortlist: {counts}", flush=True)


def neighbours(pilot=False, rag_k=RAG_K):
    from sklearn.feature_extraction.text import TfidfVectorizer
    split = load_split()
    full = train_keys(split, "full")
    items = load_items(full)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english", dtype=np.float32)
    XT = vec.fit_transform([items[k][0] for k in full]).T.tocsr()
    print(f"index: {len(full):,} training reports, {XT.shape[0]:,} terms", flush=True)
    for year in YEARS:
        d = ml.short_dir(year)
        if pilot:  # the first 50 reports of the natural sample, compared with the neighbours already on file
            test = [t["key"] for t in (load_natural_2026() if year == "2026" else load_natural_2025())[:50]]
        else:
            if not (d / "shortlist.json").exists():
                print(f"[{year}] no shortlist, skipped", flush=True)
                continue
            test = [t["key"] for t in json.loads((d / "shortlist.json").read_text(encoding="utf-8"))["items"] if t["rank"] < rag_k]
        texts = load_items(test, pool=pool_of(year))
        Q = vec.transform([texts[k][0] for k in test])
        out = {}
        for s in range(0, Q.shape[0], 64):
            sims = (Q[s:s + 64] @ XT).tocsr()
            for r in range(sims.shape[0]):
                row = sims.getrow(r)
                top = np.argsort(-row.data)[:10]
                out[test[s + r]] = [[full[row.indices[j]], items[full[row.indices[j]]][1], round(float(row.data[j]), 4)] for j in top]
        if pilot:
            ref = json.loads((rdir(f"maude{year}nat") / "neighbors_full.json").read_text())
            same = sum([n[0] for n in out[k]] == [n[0] for n in ref[str(k)]] for k in test)
            same_set = sum({n[0] for n in out[k]} == {n[0] for n in ref[str(k)]} for k in test)
            print(f"[{year}] pilot: {same}/{len(test)} reports with the same ten neighbours in the same order, {same_set} as a set", flush=True)
        else:
            (d / "neighbors_full.json").write_text(json.dumps(out))
            print(f"[{year}] neighbours of {len(out):,} shortlisted reports (rank < {rag_k}) -> {d / 'neighbors_full.json'}", flush=True)


# ---------------------------------------------------------------- scoring
def llm_answers(year, mk, kind, c2i, legal, p2c):
    """{report key: class index, or -1 for an invalid answer} from every file that holds this LLM's answers of the year."""
    out = {}
    for d in dict.fromkeys((rdir(f"maude{year}"), rdir(f"maude{year}nat"), rdir(f"maude{year}short"), ml.short_dir(year))):
        p = d / f"llm_{kind}_{mk}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["type"] == "succeeded":
                code = p2c.get(ml.normalize(r.get("text", ""), legal))
                out[int(r["custom_id"][1:])] = c2i[code] if code else -1
    return out


def csv_agreement(year, keys, enc, cls):
    """Share of the balanced-test and natural-sample reports where this pass and the saved encoder CSV agree."""
    pos = {int(k): i for i, k in enumerate(keys)}
    out = {}
    for name in (f"maude{year}", f"maude{year}nat"):
        p = rdir(name) / f"encoder_{TAG}.csv"
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                rows = [(pos.get(int(r["key"])), r["pred"]) for r in csv.DictReader(fh)]
            rows = [(i, c) for i, c in rows if i is not None]
            out[name] = {"n": len(rows), "same": round(float(np.mean([cls[enc[i]] == c for i, c in rows])), 4) if rows else None}
    return out


def score(suffix=""):
    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    path_of = {c: p for p, c in p2c.items()}
    rep = {"note": "notes/selector-vs-rules-2026-11-11.md", "encoder": TAG, "years": {}}
    for year in YEARS:
        p = ENC / f"{year}{suffix}.npz"
        if not p.exists():
            rep["years"][year] = {"not_run": f"{p.name} missing"}
            continue
        z = np.load(p)
        cls = [str(c) for c in z["classes"]]
        c2i = {c: i for i, c in enumerate(cls)}
        keys, gold, top = z["keys"], z["gold"].astype(int), z["top_idx"].astype(int)
        enc, best, rank = top[:, 0], z["rare_best"].astype(int), z["rare_rank"].astype(int)
        N = len(keys)
        is_rare = np.zeros(len(cls), bool)
        is_rare[z["rare"].astype(int)] = True
        nonrare = top[np.arange(N), (~is_rare[top]).argmax(1)]  # the encoder's best non-rare code

        def count(final):
            placed = is_rare[final]
            right = placed & (final == gold)
            r, f = int(right.sum()), int((placed & ~right).sum())
            return {"right": r, "false": f, "precision": round(r / (r + f), 4) if r + f else None,
                    "modes": len(set(gold[right].tolist())), "accuracy": round(float((final == gold).mean()), 5)}

        Y = {"reports": N, "rare_reports": int(is_rare[gold].sum()), "rare_modes_with_reports": len(set(gold[is_rare[gold]].tolist())),
             "encoder_alone": count(enc), "best_rare_code_inside_top_k": {str(k): int((rank < k).sum()) for k in KS},
             "agreement_with_saved_encoder_csv": csv_agreement(year, keys, enc, cls), "llms": {}}
        pos = {int(k): i for i, k in enumerate(keys)}
        for mk in LLMS:
            A = {}
            for name, kind in KINDS.items():
                a = np.full(N, -2)  # -2: no answer on file
                for k, v in llm_answers(year, mk, kind, c2i, legal, p2c).items():
                    if k in pos:
                        a[pos[k]] = v
                if (a > -2).any():
                    A[name] = a
            if "zs" not in A:
                continue
            R = {"answers_on_file": {n: int((a > -2).sum()) for n, a in A.items()}, "rules": {}}
            for k in KS:
                short = rank < k
                conf = {n: A[n] == best for n in A}
                cover = {n: float((A[n][short] > -2).mean()) if short.any() else 1.0 for n in A}
                if "rag" in A:
                    conf["either"], conf["both"] = conf["zs"] | conf["rag"], conf["zs"] & conf["rag"]
                    cover["either"] = cover["both"] = min(cover["zs"], cover["rag"])
                for n, c in conf.items():
                    fire = short & c
                    keep = np.where(fire, best, enc)
                    only = np.where(fire, best, np.where(is_rare[enc], nonrare, enc))
                    cell = {"coverage": round(cover[n], 4), "complete": bool(cover[n] == 1.0), "keep_encoder": count(keep),
                            "confirmed_only": count(only)}
                    if k == 5 and n == "zs":  # where the false placements of the main rule fall
                        f = keep[is_rare[keep] & (keep != gold)]
                        cell["false_by_code"] = [[path_of[cls[c]], int(v)] for c, v in
                                                 sorted(zip(*np.unique(f, return_counts=True)), key=lambda t: -t[1])[:10]]
                    R["rules"][f"k{k}/{n}"] = cell
            Y["llms"][mk] = R
        rep["years"][year] = Y
        e = Y["encoder_alone"]
        print(f"[{year}] {N:,} reports, {Y['rare_reports']} rare; encoder alone {e['right']} right / {e['false']} false; "
              f"accuracy {e['accuracy']}; csv agreement {Y['agreement_with_saved_encoder_csv']}", flush=True)
        for mk, R in Y["llms"].items():
            for name, cell in R["rules"].items():
                a, b = cell["keep_encoder"], cell["confirmed_only"]
                print(f"   {mk:11s} {name:10s} coverage {cell['coverage']:.3f}  keep {a['right']}/{a['false']}  "
                      f"confirmed only {b['right']}/{b['false']}", flush=True)
    out = ROOT / "results" / "rare_year"
    out.mkdir(parents=True, exist_ok=True)
    name = f"metrics{ALT}{suffix}.json" if CORPUS == "maude" else f"{CORPUS}_metrics{ALT}{suffix}.json"
    (out / name).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"wrote {out / name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shortlist", "neighbours", "score"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--include-existing", action="store_true")
    ap.add_argument("--rag-k", type=int, default=RAG_K)
    a = ap.parse_args()
    if a.cmd == "shortlist":
        shortlist(a.suffix, a.include_existing)
    elif a.cmd == "neighbours":
        neighbours(a.pilot, a.rag_k)
    else:
        score(a.suffix)
