"""Reports filed under the withheld codes, both test years: how many rare codes the classifier and the agreement rule
place on them (notes/mock-review-2026-11-18.md, E3; asked for by the user on 2026-11-18 to decide whether the paper says
what the withheld codes are for, or counts their reports as reports whose failure mode has no code). The withheld codes
are the split's hidden codes, kept out of training and out of the LLMs' code list, so every rare code placed on one of
these reports is a false placement, and every method answers them wrongly.

  year   GPU: the paper's classifier over every report of each test year filed under a withheld code (pool_<year>), with
         the summary of maude_encoder_year.py -> runs/<corpus>_encyear/<tag>/<year>_withheld.npz. The reports whose best
         rare code is inside the top five go to the shortlist of ml.short_dir(year) (runs/<corpus><year>short_withheld),
         at most CAP of them, drawn at random with seed 0 when there are more.
  score  CPU: rare placements of the classifier (exact) and of the rule with each LLM, either LLM coder and the zero-shot
         LLM only (exact when every shortlisted report was answered, otherwise scaled from the drawn reports with a
         Clopper-Pearson 95% interval) -> results/withheld/<corpus>.json
Between them: rare_year.py neighbours, then openweight_llm.py <llm> maude<year>short5_cls maude<year>short5_ragfull.
Every step runs with MAINT_SHORT_SUFFIX=_withheld (run_withheld.ps1 sets it).

Usage: MAINT_SHORT_SUFFIX=_withheld python withheld_check.py year      (pilot on the CPU: MAINT_SHORT_SUFFIX=_withheld_pilot
       ... withheld_check.py year --pilot 40)   |   MAINT_SHORT_SUFFIX=_withheld python withheld_check.py score
"""
import json
import os
import sys
import time

import numpy as np

YEARS, K, CAP = ("2026", "2025"), 5, 5000
SUFFIX = "_withheld"


def hidden_codes(split):
    return {c["code"] for c in split["classes"] if c["hidden"]}


def is_pilot():
    return os.environ.get("MAINT_SHORT_SUFFIX", "").endswith("_pilot")


def year(pilot=0):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    import maude_encoder_year as mey
    import maude_llm as ml
    from encoder_finetune import MODEL_NAME as BASE_NAME
    from maude_data import POOL_2025, POOL_2026, canonical, known_codes, load_split
    from maude_encoder import MAX_LEN, WEIGHTS

    hf_logging.set_verbosity_error()
    split = load_split()
    classes = known_codes(split)
    rare = mey.rare_classes(split, classes)
    hid = hidden_codes(split)
    tag = mey.DEFAULT_TAG
    wdir = WEIGHTS / tag
    tok = AutoTokenizer.from_pretrained(wdir if (wdir / "tokenizer_config.json").exists()
                                        else ("roberta-large" if "-large" in tag else BASE_NAME))
    dev = "cpu" if pilot else "cuda"
    model = AutoModelForSequenceClassification.from_pretrained(WEIGHTS / tag).to(dev).eval()
    assert model.config.num_labels == len(classes), (model.config.num_labels, len(classes))
    for yr in YEARS:
        t0 = time.time()
        rows = []
        for line in (POOL_2026 if yr == "2026" else POOL_2025).open(encoding="utf-8"):
            r = json.loads(line)
            if r["code"] in hid:
                rows.append((int(r["report_key"]), r["code"], canonical(r["text"])))
        if pilot:
            rows = rows[:pilot]
        print(f"[{yr}] {len(rows):,} reports under {len(hid)} withheld codes, {len(rare)} rare codes, device {dev}", flush=True)
        parts = {}
        for s in range(0, len(rows), mey.PART):
            part = rows[s:s + mey.PART]
            ids = tok([t for _, _, t in part], truncation=True, max_length=MAX_LEN)["input_ids"]
            logits = (mey.cpu_logits if pilot else mey.gpu_logits)(model, ids, tok.pad_token_id)
            for k, v in mey.summarize(logits, rare).items():
                parts.setdefault(k, []).append(v)
        M = {k: np.concatenate(v) for k, v in parts.items()}
        keys = np.array([k for k, _, _ in rows], dtype=np.int64)
        codes = np.array([c for _, c, _ in rows])
        name = f"{yr}{SUFFIX}{'_pilot' if pilot else ''}.npz"
        (mey.OUT / tag).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mey.OUT / tag / name, classes=np.array(classes), rare=rare.astype(np.int16), keys=keys,
                            codes=codes, **M)
        rank, best = M["rare_rank"].astype(int), M["rare_best"].astype(int)
        idx = np.where(rank < K)[0]
        sel = np.sort(np.random.default_rng(0).choice(idx, CAP, replace=False)) if len(idx) > CAP else idx
        items = [{"key": int(keys[i]), "code": str(codes[i]), "rank": int(rank[i]), "rare": classes[best[i]]} for i in sel]
        d = ml.short_dir(yr)
        d.mkdir(parents=True, exist_ok=True)
        counts = {"year_reports": int(len(keys)), "rank_below_5": int(len(idx)), "drawn": int(len(sel)),
                  "encoder_first_answer_rare": int(np.isin(M["top_idx"][:, 0].astype(int), rare).sum())}
        (d / "shortlist.json").write_text(json.dumps({"note": "withheld_check.py", "encoder": tag, "source": name,
                                                      "k_max": K, "counts": counts, "items": items}), encoding="utf-8")
        print(f"[{yr}] wrote {name} and {d / 'shortlist.json'}: {counts}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def clopper(x, n, N):
    from scipy.stats import beta
    lo = 0.0 if x == 0 else beta.ppf(0.025, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(0.975, x + 1, n - x)
    return [round(lo * N, 1), round(hi * N, 1)]


def score():
    import maude_llm as ml
    import rare_year as ry
    from common import CORPUS, ROOT
    from maude_data import load_split

    split = load_split()
    legal, p2c = sorted(ml.known_labels(split)), ml.known_labels(split)
    rep = {"note": "withheld_check.py: every rare code placed on these reports is false", "corpus": CORPUS,
           "withheld_codes": len(hidden_codes(split)), "years": {}}
    for yr in YEARS:
        p = ry.ENC / f"{yr}{SUFFIX}{'_pilot' if is_pilot() else ''}.npz"
        if not p.exists():
            rep["years"][yr] = {"not_run": p.name}
            continue
        z = np.load(p)
        cls = [str(c) for c in z["classes"]]
        c2i = {c: i for i, c in enumerate(cls)}
        keys, top = z["keys"], z["top_idx"].astype(int)
        best = z["rare_best"].astype(int)
        is_rare = np.zeros(len(cls), bool)
        is_rare[z["rare"].astype(int)] = True
        short = json.loads((ml.short_dir(yr) / "shortlist.json").read_text(encoding="utf-8"))
        n_short = short["counts"]["rank_below_5"]
        pos = {int(k): i for i, k in enumerate(keys)}
        Y = {"reports": int(len(keys)), "classifier_rare_placements": int(is_rare[top[:, 0]].sum()),
             "best_rare_code_inside_top5": int(n_short), "drawn": len(short["items"]), "llms": {}}
        for mk in ("qwen38_27b", "gemma4_31b"):
            A = {name: ry.llm_answers(yr, mk, kind, c2i, legal, p2c) for name, kind in ry.KINDS.items()}
            have = [t["key"] for t in short["items"] if t["key"] in A["zs"] and t["key"] in A["rag"]]
            out = {"answered": len(have)}
            for rule, hit in (("either", lambda k, b: A["zs"][k] == b or A["rag"][k] == b),
                              ("zero_shot_only", lambda k, b: A["zs"][k] == b)):
                x = sum(1 for k in have if hit(k, best[pos[k]]))
                if len(have) == n_short:
                    out[rule] = {"placements": x, "exact": True}
                elif have:
                    out[rule] = {"placements_in_drawn": x, "estimate": round(x / len(have) * n_short, 1),
                                 "ci95": clopper(x, len(have), n_short), "exact": False}
                else:
                    out[rule] = {"placements": None}
            Y["llms"][mk] = out
        rep["years"][yr] = Y
    od = ROOT / "results" / "withheld"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{CORPUS}{'_pilot' if is_pilot() else ''}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    if os.environ.get("MAINT_SHORT_SUFFIX", "") not in (SUFFIX, SUFFIX + "_pilot"):
        raise SystemExit("set MAINT_SHORT_SUFFIX=_withheld (or _withheld_pilot for a check)")
    if sys.argv[1] == "year":
        year(int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--pilot" else 0)
    elif sys.argv[1] == "score":
        score()
    else:
        raise SystemExit(__doc__)
