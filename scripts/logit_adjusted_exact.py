"""Exact whole-year rare placements of the logit-adjusted classifier (tau = 0.5), from one GPU pass of the classifier
over every known-code report of a year. 2026-11-19, after the manuscript review: the year files of
maude_encoder_year.py keep float16 top-20 log-probabilities, which leave a few hundred device reports undecided
(logit_adjusted_year.py), so the pass is repeated here and the adjusted scores are compared in float32.

The pass is the one of maude_encoder_year.py (same tokenizer, batches, longest first, bf16 autocast). For each report
it keeps the classifier's answer, the logit-adjusted answer and the adjusted margin of the best rare code over the
best code that is not rare. Parts are written as they finish (runs/<corpus>_encyear/<tag>/<year>_la_parts), so a
rerun resumes. Checks printed at the end: agreement of the classifier's answer with the year file, and the
classifier's own rare placements (the exact counts of the paper's Table 2).

Usage: [MAINT_CORPUS=sdr] python logit_adjusted_exact.py <2025|2026>   -> results/paper_v4/<corpus>_logit_adjusted_exact.json
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from common import CORPUS, ROOT, RUNS
from maude_data import POOL_2025, POOL_2026, canonical, class_info, known_codes, load_split
from maude_encoder import MAX_LEN, WEIGHTS
from maude_encoder_year import DEFAULT_TAG, PART, gpu_logits
from encoder_finetune import MODEL_NAME as BASE_NAME
from maude_router import training_counts

TAU = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", choices=["2025", "2026"])
    a = ap.parse_args()
    hf_logging.set_verbosity_error()
    split = load_split()
    classes = known_codes(split)
    c2i = {c: i for i, c in enumerate(classes)}
    codes = sorted(class_info(split))
    n = dict(zip(codes, training_counts(split, codes)["full"].tolist()))
    cnt = np.array([n[c] for c in classes], dtype=np.float64)
    is_rare = (cnt >= 1) & (cnt <= 5)
    adj = torch.as_tensor(-TAU * np.log((cnt + 0.5) / (cnt + 0.5).sum()), dtype=torch.float32)
    rare_t = torch.as_tensor(is_rare)
    pool = POOL_2026 if a.year == "2026" else POOL_2025
    total = sum(json.loads(line)["code"] in c2i for line in pool.open(encoding="utf-8"))
    out_dir = RUNS / f"{CORPUS}_encyear" / DEFAULT_TAG
    parts = out_dir / f"{a.year}_la_parts"
    parts.mkdir(parents=True, exist_ok=True)
    wdir = WEIGHTS / DEFAULT_TAG
    tok = AutoTokenizer.from_pretrained(wdir if (wdir / "tokenizer_config.json").exists()
                                        else ("roberta-large" if "-large" in DEFAULT_TAG else BASE_NAME))
    model = AutoModelForSequenceClassification.from_pretrained(WEIGHTS / DEFAULT_TAG).to("cuda").eval()
    assert model.config.num_labels == len(classes)
    print(f"[{CORPUS} {a.year}] {total:,} reports, {int(is_rare.sum())} rare codes, tau {TAU}", flush=True)
    t0, state = time.time(), {"done": 0, "new": 0}

    def flush(buf, no):
        path = parts / f"part_{no:04d}.npz"
        if path.exists():
            z = np.load(path)
            if len(z["keys"]) == len(buf) and int(z["keys"][0]) == buf[0][0] and int(z["keys"][-1]) == buf[-1][0]:
                state["done"] += len(buf)
                return
        ids = tok([t for _, _, t in buf], truncation=True, max_length=MAX_LEN)["input_ids"]
        logits = gpu_logits(model, ids, tok.pad_token_id)
        logp = torch.log_softmax(logits, dim=-1)
        s = logp + adj[None, :]
        best_rare = s.masked_fill(~rare_t[None, :], -float("inf")).max(dim=-1).values
        best_other = s.masked_fill(rare_t[None, :], -float("inf")).max(dim=-1).values
        tmp = parts / f"part_{no:04d}.tmp.npz"
        np.savez(tmp, keys=np.array([k for k, _, _ in buf], dtype=np.int64), gold=np.array([g for _, g, _ in buf], dtype=np.int16),
                 enc=logp.argmax(dim=-1).numpy().astype(np.int16), la=s.argmax(dim=-1).numpy().astype(np.int16),
                 margin=(best_rare - best_other).numpy().astype(np.float32))
        tmp.replace(path)
        state["done"] += len(buf)
        state["new"] += len(buf)
        rate = state["new"] / max(time.time() - t0, 1e-6)
        print(f"[{CORPUS} {a.year}] part {no}: {state['done']:,}/{total:,}  {rate:.0f}/s  "
              f"eta {(total - state['done']) / rate / 60:.1f} min", flush=True)

    buf, no = [], 0
    for line in pool.open(encoding="utf-8"):
        r = json.loads(line)
        if r["code"] not in c2i:
            continue
        buf.append((int(r["report_key"]), c2i[r["code"]], canonical(r["text"])))
        if len(buf) == PART:
            flush(buf, no)
            buf, no = [], no + 1
    if buf:
        flush(buf, no)
        no += 1
    Z = [np.load(parts / f"part_{i:04d}.npz") for i in range(no)]
    M = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}
    assert len(M["keys"]) == total and len(set(M["keys"].tolist())) == total
    gold, enc, la = M["gold"].astype(int), M["enc"].astype(int), M["la"].astype(int)
    y = np.load(out_dir / f"{a.year}.npz")
    assert (y["keys"] == M["keys"]).all() and (y["gold"] == M["gold"]).all()
    placed_la, placed_enc = is_rare[la], is_rare[enc]
    res = {"reports": int(total), "tau": TAU,
           "classifier_answer_same_as_year_file": int((y["top_idx"][:, 0].astype(int) == enc).sum()),
           "classifier_rare_right": int((placed_enc & (enc == gold)).sum()),
           "classifier_rare_false": int((placed_enc & (enc != gold)).sum()),
           "logit_adjusted_rare_right": int((placed_la & (la == gold)).sum()),
           "logit_adjusted_rare_false": int((placed_la & (la != gold)).sum()),
           "logit_adjusted_rare_within_0.01": int((np.abs(M["margin"]) < 0.01).sum()),
           "minutes": round((time.time() - t0) / 60, 1)}
    print(f"[{CORPUS} {a.year}] {res}", flush=True)
    p = ROOT / "results" / "paper_v4" / f"{CORPUS}_logit_adjusted_exact.json"
    rep = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"note": "logit_adjusted_exact.py", "corpus": CORPUS,
                                                                         "years": {}}
    rep["years"][a.year] = res
    p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
