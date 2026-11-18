"""Example-grounded verification of rare overrides, for a revised selector.

When an LLM answer names a code with few training examples (<= 20) and differs from the
encoder's answer, the same open-weight LLM is shown the report next to coded training reports
of both codes: the rare code's own examples (all of them, at most 5, nearest first) and the 5
nearest training reports of the encoder's code. It is asked which code fits. The rare code's
handful of examples is too few for the encoder to learn from but enough to compare against.
The order of the two codes is randomised per pair (seeded) to cancel position bias. Examples
come from the 2024 training reports only.

Candidates (a superset of the overrides any selector can make): every item where the
zero-shot or the retrieval answer names a code with <= 20 training examples and differs from
the encoder's answer; one request per (item, proposed code).

Sets: 2024 (development; the known 2024 test items), 2025 and 2025nat (test sets, run only
after the revised selector is pre-registered).

Usage: python maude_verify.py <verifier_key> <2024|2025|2025nat> [--answers opus55] [--only-overrides]
                              [--limit N] [--pilot]
       -> runs/maude_verify/<set>_<mk>.jsonl (resumable; --pilot writes runs/maude_verify/pilot/)
"""
import argparse
import json
import random
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import openweight_llm as ow
from common import RUNS
from maude_data import POOL, POOL_2025, class_info, load_items, load_split, train_keys
from maude_llm import RAG_EXAMPLE_CHARS
from maude_selector import load_coders

RARE_MAX = 20
N_EX = 5
OUT = RUNS / "maude_verify"
SYSTEM = ("You check which of two failure-mode codes fits a medical device adverse event report. "
          "Each code is shown with reports that were filed under it.")


def candidates(D):
    out = []
    for i in range(len(D["keys"])):
        seen = set()
        for arm in ("zs", "rag"):
            c = int(D[arm][i])
            if c >= 0 and c != D["enc"][i] and D["counts"][c] <= RARE_MAX and c not in seen:
                seen.add(c)
                out.append((i, c))
    return out


def clip(t):
    return t if len(t) <= RAG_EXAMPLE_CHARS else t[:RAG_EXAMPLE_CHARS].rstrip() + " ..."


def build_requests(which, mk, only_overrides=False):
    """mk: whose answers are checked (the verifier is chosen at run time). only_overrides: just
    the rare overrides the frozen selector of mk makes (the pairs a deployed filter would see)."""
    D = load_coders(which, mk)
    split = load_split()
    info = class_info(split)
    full = train_keys(split, "full")
    items = load_items(full)
    by_code = {}
    for k in full:
        by_code.setdefault(items[k][1], []).append(k)
    cand = candidates(D)
    if only_overrides:
        from maude_selector import load_frozen, predict_frozen
        frozen, models = load_frozen(mk)
        pick, _ = predict_frozen(D, frozen, models)
        cand = [(i, int(pick[i])) for i in range(len(pick))
                if pick[i] >= 0 and pick[i] != D["enc"][i] and D["counts"][pick[i]] <= RARE_MAX]
    pool = POOL if which == "2024" else POOL_2025
    q_items = load_items(sorted({D["keys"][i] for i, _ in cand}), pool=pool)
    need = {D["codes"][c] for _, c in cand} | {D["codes"][D["enc"][i]] for i, _ in cand}
    docs = sorted({k for code in need for k in by_code.get(code, [])})
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english",
                          dtype=np.float32)
    vec.fit([items[k][0] for k in full])
    X = vec.transform([items[k][0] for k in docs])
    row = {k: r for r, k in enumerate(docs)}

    def nearest(qv, code):
        keys = by_code.get(code, [])
        if not keys:
            return []
        sims = (X[[row[k] for k in keys]] @ qv.T).toarray().ravel()
        return [keys[j] for j in np.argsort(-sims)[:N_EX]]

    reqs = []
    for i, c in cand:
        key = D["keys"][i]
        text = q_items[key][0]
        qv = vec.transform([text])
        rare, enc = D["codes"][c], D["codes"][D["enc"][i]]
        blocks = {rare: nearest(qv, rare), enc: nearest(qv, enc)}
        order = [rare, enc]
        random.Random(f"verify-{key}-{rare}").shuffle(order)
        parts = []
        for n, code in enumerate(order, 1):
            parts.append(f"Code {n}: {info[code]['path']}")
            parts.append(f"Reports filed under code {n}:")
            parts += [f"- {clip(items[k][0])}" for k in blocks[code]]
            parts.append("")
        parts += ["Report to code:", text, "",
                  "Which code fits this report better? Answer with exactly one digit: 1 or 2."]
        reqs.append({"custom_id": f"{key}-{rare}", "key": key, "rare_code": rare, "enc_code": enc,
                     "rare_position": order.index(rare) + 1,
                     "params": {"system": [{"type": "text", "text": SYSTEM}], "max_tokens": 16,
                                "messages": [{"role": "user", "content": "\n".join(parts)}]}})
    return reqs


def parse(text):
    """The last 1 or 2 in the reply; None if there is none."""
    digits = [ch for ch in text if ch in "12"]
    return int(digits[-1]) if digits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key", choices=["qwen38_27b", "gemma4_31b"])
    ap.add_argument("which", choices=["2024", "2025", "2025nat"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--answers", choices=["opus55", "qwen38_27b", "gemma4_31b"],
                    help="whose answers to check (default: the verifier's own)")
    ap.add_argument("--only-overrides", action="store_true")
    a = ap.parse_args()
    mk = a.model_key
    src = a.answers or mk
    reqs = build_requests(a.which, src, a.only_overrides)
    if a.limit:
        reqs = reqs[:a.limit]
    tag = f"{a.which}_{mk}" if src == mk else f"{a.which}_{src}_by_{mk}"
    tag += "_overrides" if a.only_overrides else ""
    out = (OUT / "pilot" if a.pilot else OUT) / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["custom_id"] for l in out.read_text(encoding="utf-8").splitlines()
                if json.loads(l)["type"] == "succeeded"}
    todo = [r for r in reqs if r["custom_id"] not in done]
    print(f"{mk}/{a.which}: {len(reqs)} verification requests, {len(done)} done, {len(todo)} to run -> {out}", flush=True)
    if not todo:
        return
    proc = ow.start_server(mk, RUNS / "openweight" / f"server_{mk}.log", ow.CTX)
    try:
        t0 = time.time()
        with out.open("a", encoding="utf-8") as fh:
            for n, req in enumerate(todo, 1):
                row = ow.run_one(mk, req)
                row.update({"key": req["key"], "rare_code": req["rare_code"], "enc_code": req["enc_code"],
                            "rare_position": req["rare_position"],
                            "choice": parse(row.get("text", "")) if row["type"] == "succeeded" else None})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)}  {n / el:.2f}/s  eta {(len(todo) - n) / (n / el) / 60:.1f} min", flush=True)
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
