"""The selector's encoder over every known-code report of a year (notes/selector-vs-rules-2026-11-11.md,
section "한 해 전체로 재는 방법"). GPU: lent by the user for 2026-11-12 20:00 to 2026-11-13 09:00.

For each report of data/maude/pool_<year>.jsonl filed under one of the 424 known codes it keeps the
encoder's 20 most probable codes with their log-probabilities, the log-probabilities of the rare codes
(1-5 training reports), the best-ranked rare code and its rank (0 = the encoder's answer), and the
logsumexp of the logits. Parts of PART reports are written as they finish
(runs/maude_encyear/<tag>/<year>_parts), so a rerun resumes; the merged file is
runs/maude_encyear/<tag>/<year>.npz. The GPU pass makes the batches of maude_encoder.logits_for (the
pass that produced the encoder CSVs) and takes them longest first: taken shortest first on 50,000
reports, the allocator's cache outgrew the card and spilled into system RAM (24 GB dedicated plus
25 GB shared, about 150 reports/s on 2026-11-12).

Usage: python maude_encoder_year.py <2025|2026> [--tag cb-large-conv_full_seed42]
       MAINT_CORPUS=sdr python maude_encoder_year.py <2025|2026>   (SDR: the redesign's cb-conv encoder,
       files under runs/sdr_encyear; 2026-11-14)
       python maude_encoder_year.py 2026 --pilot 300 [--device cuda]    check on the first 300 reports of
                                                         the year's natural sample -> <year>_pilot.npz (CPU by default)
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from common import CORPUS, RUNS
from encoder_finetune import MODEL_NAME as BASE_NAME
from maude_data import POOL_2025, POOL_2026, canonical, class_info, known_codes, load_split, load_test_2025, load_test_2026
from maude_encoder import EVAL_BATCH, MAX_LEN, WEIGHTS, collate
from maude_router import training_counts

TOP = 20
PART = 50_000
OUT = RUNS / f"{CORPUS}_encyear"
DEFAULT_TAG = {"maude": "cb-large-conv_full_seed42", "sdr": "cb-conv_full_seed42"}[CORPUS]  # the redesign's selector encoders


def rare_classes(split, classes):
    """Indices (in the encoder's label order) of the codes with 1-5 training reports."""
    codes = sorted(class_info(split))
    counts = dict(zip(codes, training_counts(split, codes)["full"].tolist()))
    return np.array([i for i, c in enumerate(classes) if 1 <= counts[c] <= 5])


def gpu_logits(model, ids, pad):
    """The batches of maude_encoder.logits_for, longest first, so the first batch sets the memory the rest reuse."""
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.empty((len(ids), model.config.num_labels), dtype=torch.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in reversed(range(0, len(order), EVAL_BATCH)):
            idx = order[s:s + EVAL_BATCH]
            x, m = collate(ids, idx, pad)
            out[idx] = model(input_ids=x.to("cuda"), attention_mask=m.to("cuda")).logits.float().cpu()
    torch.cuda.empty_cache()
    return out


def cpu_logits(model, ids, pad):
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.empty((len(ids), model.config.num_labels), dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, len(order), 16):
            idx = order[s:s + 16]
            x, m = collate(ids, idx, pad)
            out[idx] = model(input_ids=x, attention_mask=m).logits.float()
    return out


def summarize(logits, rare):
    logp = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logp, TOP, dim=-1)
    rl = logp[:, torch.as_tensor(rare)]
    best = rl.max(dim=-1)
    rank = (logp > best.values[:, None]).sum(dim=-1)
    return {"top_idx": top.indices.numpy().astype(np.int16), "top_logp": top.values.numpy().astype(np.float16),
            "rare_logp": rl.numpy().astype(np.float16), "rare_best": rare[best.indices.numpy()].astype(np.int16),
            "rare_rank": rank.numpy().astype(np.int16), "lse": torch.logsumexp(logits, dim=-1).numpy().astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", choices=["2025", "2026"])
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    a = ap.parse_args()
    hf_logging.set_verbosity_error()
    split = load_split()
    classes = known_codes(split)
    c2i = {c: i for i, c in enumerate(classes)}
    rare = rare_classes(split, classes)
    pool = POOL_2026 if a.year == "2026" else POOL_2025
    total = sum(json.loads(line)["code"] in c2i for line in pool.open(encoding="utf-8"))
    dev, want, suffix = "cuda", None, ""
    if a.pilot:
        test = load_test_2026() if a.year == "2026" else load_test_2025()  # the balanced test, which has an encoder CSV
        want, total, dev, suffix = {t["key"] for t in test[:a.pilot]}, a.pilot, a.device or "cpu", "_pilot"
        torch.set_num_threads(2)
    out_dir = OUT / a.tag
    parts = out_dir / f"{a.year}{suffix}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    wdir = WEIGHTS / a.tag  # a run of another encoder family keeps its tokenizer beside its weights
    tok = AutoTokenizer.from_pretrained(wdir if (wdir / "tokenizer_config.json").exists()
                                        else ("roberta-large" if "-large" in a.tag else BASE_NAME))
    model = AutoModelForSequenceClassification.from_pretrained(WEIGHTS / a.tag).to(dev).eval()
    assert model.config.num_labels == len(classes), (model.config.num_labels, len(classes))
    print(f"[{a.year}] {total:,} reports, {len(rare)} rare codes, device {dev}, parts of {PART:,}", flush=True)
    t0, state = time.time(), {"done": 0, "new": 0}

    def flush(buf, no):
        path = parts / f"part_{no:04d}.npz"
        if path.exists():  # finished by an earlier run of this script
            z = np.load(path)
            if len(z["keys"]) == len(buf) and int(z["keys"][0]) == buf[0][0] and int(z["keys"][-1]) == buf[-1][0]:
                state["done"] += len(buf)
                print(f"[{a.year}] part {no}: already there ({state['done']:,}/{total:,})", flush=True)
                return
        ids = tok([t for _, _, t in buf], truncation=True, max_length=MAX_LEN)["input_ids"]
        logits = gpu_logits(model, ids, tok.pad_token_id) if dev == "cuda" else cpu_logits(model, ids, tok.pad_token_id)
        tmp = parts / f"part_{no:04d}.tmp.npz"
        np.savez(tmp, keys=np.array([k for k, _, _ in buf], dtype=np.int64), gold=np.array([g for _, g, _ in buf], dtype=np.int16),
                 **summarize(logits, rare))
        tmp.replace(path)
        state["done"] += len(buf)
        state["new"] += len(buf)
        rate = state["new"] / max(time.time() - t0, 1e-6)
        print(f"[{a.year}] part {no}: {state['done']:,}/{total:,}  {rate:.0f}/s  eta {(total - state['done']) / rate / 60:.1f} min", flush=True)

    buf, no = [], 0
    for line in pool.open(encoding="utf-8"):
        r = json.loads(line)
        if r["code"] not in c2i:
            continue
        k = int(r["report_key"])
        if want is not None and k not in want:
            continue
        buf.append((k, c2i[r["code"]], canonical(r["text"])))
        if len(buf) == PART:
            flush(buf, no)
            buf, no = [], no + 1
    if buf:
        flush(buf, no)
        no += 1
    Z = [np.load(parts / f"part_{i:04d}.npz") for i in range(no)]
    M = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}
    if len(M["keys"]) != total or len(set(M["keys"].tolist())) != total:
        raise SystemExit(f"{len(M['keys'])} reports merged, {total} expected")
    np.savez_compressed(out_dir / f"{a.year}{suffix}.npz", classes=np.array(classes), rare=rare.astype(np.int16), **M)
    acc = float((M["top_idx"][:, 0] == M["gold"]).mean())
    short = {k: int((M["rare_rank"] < k).sum()) for k in (1, 3, 5, 10)}
    print(f"[{a.year}] wrote {out_dir / (a.year + suffix + '.npz')}: {total:,} reports, encoder accuracy {acc:.4f}, "
          f"best rare code inside the top k: {short}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
