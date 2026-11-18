"""Opus check of the selector's rare answers (notes/rare-check-2026-11-09.md, step 2).

For every report whose selector answer names a rare code (training support 1-5), Opus 5.5 sees the
report next to coded training reports of two codes: the rare code (its own training reports, at
most five, nearest first) and the selector's best answer naming a non-rare code (the five nearest
training reports filed under it), in a seeded random order, and says which code fits. The prompt is
that of maude_verify.py, which ran this check with Qwen and Gemma on 2024. One Message Batches
request per report, no new coder outputs.

Usage: python rare_verify.py build             -> runs/rare_verify/requests.jsonl (all four sets)
       python rare_verify.py submit            -> one batch (recorded in runs/rare_verify/batch.json)
       python rare_verify.py poll              -> runs/rare_verify/<set>_opus55.jsonl once the batch has ended
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")
import maude_selector as ms  # noqa: E402
from common import RUNS  # noqa: E402
from maude_data import POOL_2025, POOL_2026, class_info, load_items, load_split, train_keys  # noqa: E402
from maude_llm import CLAUDE_MODELS, RAG_EXAMPLE_CHARS  # noqa: E402
from rare_check import LLM, RARE_MAX, fallback, selector_outputs  # noqa: E402

OUT = RUNS / "rare_verify"
SETS = ("2025", "2025nat", "2026", "2026nat")
N_EX = 5
SYSTEM = ("You check which of two failure-mode codes fits a medical device adverse event report. "
          "Each code is shown with reports that were filed under it.")
PRICE = (4.0, 20.0)  # Opus 5.5, $ per million input / output tokens; batch halves it


def clip(t):
    return t if len(t) <= RAG_EXAMPLE_CHARS else t[:RAG_EXAMPLE_CHARS].rstrip() + " ..."


def build():
    from sklearn.feature_extraction.text import TfidfVectorizer
    frozen, models = ms.load_frozen(LLM)
    split = load_split()
    info = class_info(split)
    full = train_keys(split, "full")
    items = load_items(full)
    by_code = {}
    for k in full:
        by_code.setdefault(items[k][1], []).append(k)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english",
                          dtype=np.float32)
    vec.fit([items[k][0] for k in full])
    per_set = {}
    need = set()
    for which in SETS:
        D = ms.load_coders(which, LLM)
        P, Pm, arms, pick, j = selector_outputs(D, frozen, models)
        alt = fallback(D, Pm, arms)
        c = D["counts"]
        rows = [(i, int(pick[i]), int(alt[i])) for i in range(len(pick))
                if pick[i] >= 0 and 1 <= c[pick[i]] <= RARE_MAX and alt[i] >= 0 and alt[i] != pick[i]]
        per_set[which] = (D, rows)
        need |= {D["codes"][r] for _, r, _ in rows} | {D["codes"][a] for _, _, a in rows}
        print(f"{which}: {len(rows)} reports with a rare selector answer", flush=True)
    docs = sorted({k for code in need for k in by_code.get(code, [])})
    X = vec.transform([items[k][0] for k in docs])
    row = {k: r for r, k in enumerate(docs)}

    def nearest(qv, code):
        keys = by_code.get(code, [])
        sims = (X[[row[k] for k in keys]] @ qv.T).toarray().ravel()
        return [keys[j] for j in np.argsort(-sims)[:N_EX]]

    model_id, extras, max_tokens = CLAUDE_MODELS[LLM]
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with (OUT / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for which, (D, rows) in per_set.items():
            pool = POOL_2025 if which.startswith("2025") else POOL_2026
            q = load_items([D["keys"][i] for i, _, _ in rows], pool=pool)
            for i, rc, ac in rows:
                key = D["keys"][i]
                text = q[key][0]
                qv = vec.transform([text])
                rare, other = D["codes"][rc], D["codes"][ac]
                blocks = {rare: nearest(qv, rare), other: nearest(qv, other)}
                order = [rare, other]
                random.Random(f"rareverify-{key}-{rare}").shuffle(order)
                parts = []
                for m, code in enumerate(order, 1):
                    parts.append(f"Code {m}: {info[code]['path']}")
                    parts.append(f"Reports filed under code {m}:")
                    parts += [f"- {clip(items[k][0])}" for k in blocks[code]]
                    parts.append("")
                parts += ["Report to code:", text, "",
                          "Which code fits this report better? Answer with exactly one digit: 1 or 2."]
                params = {"model": model_id, "max_tokens": max_tokens, **extras,
                          "system": [{"type": "text", "text": SYSTEM}],
                          "messages": [{"role": "user", "content": "\n".join(parts)}]}
                fh.write(json.dumps({"custom_id": f"{which}-{key}", "set": which, "key": key, "rare_code": rare,
                                     "other_code": other, "rare_position": order.index(rare) + 1,
                                     "params": params}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} requests -> {OUT / 'requests.jsonl'}")


def submit():
    from stage0_batch import get_client
    book = OUT / "batch.json"
    if book.exists() and not json.loads(book.read_text()).get("collected"):
        sys.exit("a batch is already submitted and not collected; run poll")
    reqs = [json.loads(l) for l in (OUT / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    client = get_client()
    entry = {"n": len(reqs), "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "batch_id": None, "collected": False}
    book.write_text(json.dumps(entry, indent=1))
    batch = client.with_options(max_retries=0, timeout=1800).messages.batches.create(
        requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in reqs])
    entry["batch_id"] = batch.id
    book.write_text(json.dumps(entry, indent=1))
    print(f"submitted {batch.id} ({len(reqs)} requests)")


def parse(text):
    digits = [ch for ch in text if ch in "12"]
    return int(digits[-1]) if digits else None


def poll():
    from stage0_batch import get_client
    book = OUT / "batch.json"
    entry = json.loads(book.read_text())
    client = get_client()
    r = client.messages.batches.retrieve(entry["batch_id"])
    if r.processing_status != "ended":
        print(f"{r.processing_status} {dict(r.request_counts)}")
        print("STILL_RUNNING")
        return
    meta = {json.loads(l)["custom_id"]: json.loads(l) for l in (OUT / "requests.jsonl").read_text(encoding="utf-8").splitlines()}
    out = {w: [] for w in SETS}
    cost, bad = 0.0, 0
    for res in client.messages.batches.results(entry["batch_id"]):
        m = meta[res.custom_id]
        row = {"custom_id": res.custom_id, "key": m["key"], "rare_code": m["rare_code"], "other_code": m["other_code"],
               "type": res.result.type}
        if res.result.type == "succeeded":
            msg = res.result.message
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            d = parse(text)
            row.update(text=text, choice=d, verdict=None if d is None else ("rare" if d == m["rare_position"] else "other"))
            cost += 0.5 * (msg.usage.input_tokens * PRICE[0] + msg.usage.output_tokens * PRICE[1]) / 1e6
            bad += d is None
        out[m["set"]].append(row)
    for w, rows in out.items():
        (OUT / f"{w}_opus55.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                                               encoding="utf-8")
        v = [x.get("verdict") for x in rows]
        print(f"{w}: {len(rows)} rows, rare chosen {v.count('rare')}, other {v.count('other')}, unparsed {v.count(None)}")
    entry.update(collected=True, cost_usd=round(cost, 3), unparsed=bad)
    book.write_text(json.dumps(entry, indent=1))
    print(f"cost ${cost:.2f}")
    print("ALL_ENDED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "submit", "poll"])
    {"build": build, "submit": submit, "poll": poll}[ap.parse_args().cmd]()
