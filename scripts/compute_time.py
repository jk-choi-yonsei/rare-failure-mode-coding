"""Time per report of each part of the method on this machine (2026-11-18: a reviewer will ask how much the method
adds to the classifier alone). Each phase runs in its own process, so that only one model holds the GPU at a time.

  classifier    the paper's classifier with the batches of maude_encoder_year.py (bf16 autocast, longest first) over a
                fixed random set of reports of the 2026 test year: tokenizing, one warm-up pass, three timed passes
  llm <mk>      the zero-shot and the retrieval LLM coder through llama.cpp with the server flags and request settings of
                openweight_llm.py (one slot, greedy decoding, prompt cache) on a fixed random set of reports of the 2026
                shortlist, after warm-up requests: all zero-shot requests first, then all retrieval requests
  cpu           the retrieval of the ten most similar training reports (TF-IDF, as in rare_year.neighbours) for the same
                reports as the classifier phase, and the selection model of the paper (candidate_ranker, variant
                wide_gbm_table) fitted on the 2025 sample and applied to the 2026 sample, with Qwen3.8-27B answers
Each phase writes results/compute_time/<corpus>_<phase>[_<mk>].json with seconds per report, the GPU state before and
after (nvidia-smi), and the sizes of both test years and their shortlists, so that per-year totals can be computed.

Usage: [MAINT_CORPUS=sdr] python compute_time.py classifier [--n 20000] [--device cpu]
       [MAINT_CORPUS=sdr] python compute_time.py llm <qwen38_27b|gemma4_31b> [--n 200] [--dry]
       [MAINT_CORPUS=sdr] python compute_time.py cpu [--n 20000]
"""
import argparse
import json
import platform
import random
import statistics as st
import subprocess
import sys
import tempfile
import time

import numpy as np

from common import CORPUS, ROOT, RUNS

OUT = ROOT / "results" / "compute_time"
TAG = {"maude": "cb-large-conv_full_seed42", "sdr": "cb-conv_full_seed42"}[CORPUS]  # the paper's classifiers
SEED = 20261118


def gpu_state():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,clocks.sm",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": q}


def year_sizes():
    """Reports of each test year (known codes) and its shortlist (top rare code within the top five), from the
    paper's whole-year pass of the classifier."""
    out = {}
    for y in ("2025", "2026"):
        z = np.load(RUNS / f"{CORPUS}_encyear" / TAG / f"{y}.npz")
        out[y] = {"reports": int(len(z["keys"])), "shortlist": int((z["rare_rank"] < 5).sum())}
    return out


def summary(xs):
    xs = sorted(xs)
    q = st.quantiles(xs, n=10) if len(xs) >= 10 else [xs[0]] * 9
    return {"n": len(xs), "median": round(st.median(xs), 4), "mean": round(st.mean(xs), 4), "p10": round(q[0], 4),
            "p90": round(q[-1], 4), "total": round(sum(xs), 2)}


def sample_reports(n, year="2026"):
    """A fixed random set of n reports of the test year filed under known codes (reservoir sample of the pool)."""
    from maude_data import POOL_2025, POOL_2026, canonical, known_codes, load_split
    c2i = {c: i for i, c in enumerate(known_codes(load_split()))}
    rng, keep, total = random.Random(SEED), [], 0
    for line in (POOL_2026 if year == "2026" else POOL_2025).open(encoding="utf-8"):
        r = json.loads(line)
        if r["code"] not in c2i:
            continue
        total += 1
        item = (int(r["report_key"]), canonical(r["text"]))
        if len(keep) < n:
            keep.append(item)
        else:
            j = rng.randrange(total)
            if j < n:
                keep[j] = item
    return keep, total


def write(name, rep):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{CORPUS}_{name}.json"
    p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", p, flush=True)


# ---------------------------------------------------------------- classifier (GPU)
def classifier(n, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    from encoder_finetune import MODEL_NAME as BASE_NAME
    from maude_encoder import MAX_LEN, WEIGHTS
    from maude_encoder_year import cpu_logits, gpu_logits
    hf_logging.set_verbosity_error()
    rep = {"phase": "classifier", "corpus": CORPUS, "tag": TAG, "device": device, "gpu_before": gpu_state(),
           "years": year_sizes(), "torch": torch.__version__, "cpu": platform.processor()}
    items, total = sample_reports(n)
    texts = [t for _, t in items]
    print(f"[{CORPUS}] {len(texts):,} of {total:,} reports of 2026", flush=True)
    wdir = WEIGHTS / TAG
    tok = AutoTokenizer.from_pretrained(wdir if (wdir / "tokenizer_config.json").exists()
                                        else ("roberta-large" if "-large" in TAG else BASE_NAME))
    t = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(wdir).to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    rep["load_seconds"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    ids = tok(texts, truncation=True, max_length=MAX_LEN)["input_ids"]
    rep["tokenize_seconds_per_report"] = (time.perf_counter() - t) / len(texts)
    rep["mean_tokens"] = round(float(np.mean([len(x) for x in ids])), 1)
    run = gpu_logits if device == "cuda" else cpu_logits
    run(model, ids[:min(1000, len(ids))], tok.pad_token_id)  # warm-up
    passes = []
    for k in range(3 if device == "cuda" else 1):
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        run(model, ids, tok.pad_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        passes.append(time.perf_counter() - t)
        print(f"[{CORPUS}] pass {k + 1}: {len(ids) / passes[-1]:.0f} reports/s", flush=True)
    rep["reports_timed"] = len(ids)
    rep["pass_seconds"] = [round(x, 3) for x in passes]
    rep["model_seconds_per_report"] = st.median(passes) / len(ids)
    rep["seconds_per_report"] = rep["model_seconds_per_report"] + rep["tokenize_seconds_per_report"]
    if device == "cuda":
        rep["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    rep["gpu_after"] = gpu_state()
    write("classifier" + ("" if device == "cuda" else "_cpu_smoke"), rep)


# ---------------------------------------------------------------- the two LLM coders (GPU, llama.cpp)
def llm(mk, n, dry):
    import maude_llm
    import openweight_llm as ow
    # keyed by report (the custom ids of the two prompts differ in their first letter)
    zs = {r["custom_id"][1:]: r for r in maude_llm.build_requests("cls", "open", "open", test_set="2026short5")}
    rag = {r["custom_id"][1:]: r for r in maude_llm.build_requests("cls", "open", "open", test_set="2026short5", rag="full")}
    ids = sorted(set(zs) & set(rag))
    pick = random.Random(SEED).sample(ids, min(len(ids), n + 6))
    warm, timed = pick[:6], pick[6:]
    rep = {"phase": "llm", "corpus": CORPUS, "llm": mk, "model_file": ow.model_path(mk).name, "llama_build": ow.LLAMA_BUILD,
           "ctx": ow.CTX_MAUDE, "shortlist_2026_requests": len(ids), "timed_reports": len(timed), "years": year_sizes(),
           "gpu_before": gpu_state()}
    print(f"[{CORPUS} {mk}] {len(ids):,} shortlisted reports of 2026; {len(timed)} timed, {len(warm)} warm-up", flush=True)
    if dry:
        rep["dry"] = True
        write(f"llm_{mk}_dry", rep)
        return
    log_path = __import__("pathlib").Path(tempfile.gettempdir()) / f"maintlog_time_{mk}.log"
    t = time.perf_counter()
    proc = ow.start_server(mk, log_path, ow.CTX_MAUDE)
    rep["server_start_seconds"] = round(time.perf_counter() - t, 1)
    try:
        for kind, reqs in (("zero_shot", zs), ("retrieval", rag)):
            for cid in warm[:3] if kind == "zero_shot" else warm[3:]:
                ow.run_one(mk, reqs[cid])
            secs, tin, cached, tout, errs = [], [], [], [], 0
            for i, cid in enumerate(timed, 1):
                t = time.perf_counter()
                row = ow.run_one(mk, reqs[cid])
                dt = time.perf_counter() - t
                if row["type"] != "succeeded":
                    errs += 1
                    continue
                secs.append(dt)
                u = row["usage"]
                tin.append(u["in"])
                cached.append(u.get("cached") or 0)
                tout.append(u["out"])
                if i % 50 == 0:
                    print(f"  {kind} {i}/{len(timed)}  median {st.median(secs):.2f} s", flush=True)
            rep[kind] = {"seconds": summary(secs), "errors": errs, "mean_prompt_tokens": round(st.mean(tin), 1),
                         "mean_cached_tokens": round(st.mean(cached), 1), "mean_output_tokens": round(st.mean(tout), 1)}
            print(f"[{CORPUS} {mk}] {kind}: {rep[kind]}", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=120)
    rep["gpu_after"] = gpu_state()
    write(f"llm_{mk}", rep)
    if any(rep[k]["errors"] for k in ("zero_shot", "retrieval")):
        raise SystemExit("some requests errored")


# ---------------------------------------------------------------- retrieval and the selection model (CPU)
def cpu(n):
    from sklearn.feature_extraction.text import TfidfVectorizer

    from maude_data import load_items, load_split, train_keys
    rep = {"phase": "cpu", "corpus": CORPUS, "years": year_sizes(), "cpu": platform.processor()}
    split = load_split()
    full = train_keys(split, "full")
    items = load_items(full)
    t = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english", dtype=np.float32)
    XT = vec.fit_transform([items[k][0] for k in full]).T.tocsr()
    rep["retrieval_index_seconds"] = round(time.perf_counter() - t, 2)
    keys = [k for k, _ in sample_reports(n)[0]]
    from maude_data import POOL_2026
    texts = load_items(keys, pool=POOL_2026)
    t = time.perf_counter()
    Q = vec.transform([texts[k][0] for k in keys])
    for s in range(0, Q.shape[0], 64):
        sims = (Q[s:s + 64] @ XT).tocsr()
        for r in range(sims.shape[0]):
            row = sims.getrow(r)
            np.argsort(-row.data)[:10]
    rep["retrieval_seconds_per_report"] = (time.perf_counter() - t) / len(keys)
    rep["retrieval_reports_timed"] = len(keys)
    print(f"[{CORPUS}] retrieval: {rep['retrieval_seconds_per_report'] * 1e3:.2f} ms per report", flush=True)

    import candidate_ranker as cr
    from rare_check import natural_weights
    D = {w: cr.load(w, "qwen38_27b") for w in ("2025nat", "2026nat")}
    rows = natural_weights({"D": D["2025nat"]}, "2025")[1]
    t = time.perf_counter()
    R = cr.fit(D["2025nat"], rows, "wide_gbm_table")
    rep["selection_fit_seconds"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    cr.predict(R, D["2026nat"])
    m = len(D["2026nat"]["gold"])
    rep["selection_seconds_per_report"] = (time.perf_counter() - t) / m
    rep["selection_reports_timed"] = m
    print(f"[{CORPUS}] selection model: {rep['selection_seconds_per_report'] * 1e3:.2f} ms per report "
          f"(fit {rep['selection_fit_seconds']} s)", flush=True)
    write("cpu", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["classifier", "llm", "cpu"])
    ap.add_argument("model_key", nargs="?", choices=["qwen38_27b", "gemma4_31b"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.phase == "classifier":
        classifier(a.n or 20000, a.device)
    elif a.phase == "llm":
        if not a.model_key:
            sys.exit("llm needs a model key")
        llm(a.model_key, a.n or 200, a.dry)
    else:
        cpu(a.n or 20000)


if __name__ == "__main__":
    main()
