"""Full softmax of the selector's encoder (class-balanced RoBERTa-large, full N, seed 42; the
retrained weights in ~/models/maude_encoders/cb-large_full_seed42 that produced every selector
input) on the three selector sets, so a revised selector can ask how plausible the encoder finds
the code each LLM proposes. Computing these inputs scores nothing.

Sets: 2024 = the 3,829 known 2024 test items (selector fitting data), 2025 = the balanced 2025
test (4,017), 2025nat = the natural-distribution sample (5,814). Runs on the CPU by default; the
GPU is shared with the user's own experiments and is used only with their go-ahead.

Writes runs/maude_encprobs/<set>.npz: keys, classes (the encoder's label order, known_codes),
logp float16 [n, n_classes], argmax.
Usage: python maude_encoder_probs.py 2024 2025 2025nat [--device cpu] [--threads 16]

--tag cb-large-conv_full_seed42 (2026-11-09, the rare-answer check of notes/rare-check-2026-11-09.md):
the redesign's encoder; files go to runs/maude_encprobs/<tag>/<set>.npz (with lse = logsumexp of the
logits) and, for 2025nat, the selector's encoder CSV runs/maude2025nat/encoder_<tag>.csv is written
from the same pass. Sets 2026 and 2026nat are added. CPU float32: predictions can differ from the
GPU bf16 CSVs in a few borderline reports.

MAINT_CORPUS=sdr (2026-11-14, notes/selector-vs-rules-2026-11-11.md): the SDR selector's encoder (--tag
cb-conv_full_seed42, roberta-base tokenizer), files under runs/sdr_encprobs/<tag>/. The natural-sample CSV is
written only when none exists (SDR's came from the GPU pass of maude_encoder_predict.py).
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from common import CORPUS, RUNS
from encoder_finetune import MODEL_NAME as BASE_NAME
from maude_data import (POOL, POOL_2025, POOL_2026, class_info, known_codes, load_items, load_natural_2025,
                        load_natural_2026, load_split, load_test_2025, load_test_2026)
from maude_encoder import MAX_LEN, WEIGHTS

TAG = "cb-large_full_seed42"
OUT = RUNS / f"{CORPUS}_encprobs"


def keys_for(which, split):
    info = class_info(split)
    if which == "2024":
        return [t["key"] for t in split["test"] if not info[t["code"]]["hidden"]], POOL
    if which == "2025":
        return [t["key"] for t in load_test_2025()], POOL_2025
    if which == "2025nat":
        return [t["key"] for t in load_natural_2025()], POOL_2025
    if which == "2026":
        return [t["key"] for t in load_test_2026()], POOL_2026
    if which == "2026nat":
        return [t["key"] for t in load_natural_2026()], POOL_2026
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", choices=["2024", "2025", "2025nat", "2026", "2026nat"])
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    hf_logging.set_verbosity_error()
    OUT.mkdir(parents=True, exist_ok=True)
    split = load_split()
    classes = known_codes(split)
    tok = AutoTokenizer.from_pretrained("roberta-large" if "-large" in a.tag else BASE_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(WEIGHTS / a.tag).to(a.device).eval()
    assert model.config.num_labels == len(classes), (model.config.num_labels, len(classes))
    out_dir = OUT if a.tag == TAG else OUT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for which in a.sets:
        keys, pool = keys_for(which, split)
        items = load_items(keys, pool=pool)
        ids = tok([items[k][0] for k in keys], truncation=True, max_length=MAX_LEN)["input_ids"]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        logp = np.zeros((len(ids), len(classes)), dtype=np.float16)
        lse = np.zeros(len(ids), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, len(order), a.batch):
                idx = order[s:s + a.batch]
                L = max(len(ids[i]) for i in idx)
                x = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
                m = torch.zeros((len(idx), L), dtype=torch.long)
                for r, i in enumerate(idx):
                    x[r, :len(ids[i])] = torch.tensor(ids[i])
                    m[r, :len(ids[i])] = 1
                lg = model(input_ids=x.to(a.device), attention_mask=m.to(a.device)).logits.float().cpu()
                logp[idx] = torch.log_softmax(lg, dim=-1).numpy().astype(np.float16)
                lse[idx] = torch.logsumexp(lg, dim=-1).numpy()
                done = s + len(idx)
                if (s // a.batch) % 20 == 0:
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"{which}: {done}/{len(ids)}  {rate:.1f}/s  eta {(len(ids) - done) / rate / 60:.1f} min", flush=True)
        argmax = logp.astype(np.float32).argmax(axis=1)
        np.savez_compressed(out_dir / f"{which}.npz", keys=np.array(keys), classes=np.array(classes), logp=logp,
                            argmax=argmax, lse=lse)
        print(f"{which}: wrote {out_dir / (which + '.npz')} in {(time.time() - t0) / 60:.1f} min", flush=True)
        from common import rdir
        if which == "2025nat" and a.tag != TAG and not (rdir("maude2025nat") / f"encoder_{a.tag}.csv").exists():
            import csv  # the selector's encoder input for the 2025 natural sample
            info = class_info(split)
            code_of = {t["key"]: t["code"] for t in load_natural_2025()}
            path = rdir("maude2025nat") / f"encoder_{a.tag}.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["key", "gold", "is_novel", "tier", "pred", "msp_novelty", "energy_novelty"])
                for i, k in enumerate(keys):
                    c = info[code_of[k]]
                    msp = 1.0 - float(np.exp(logp[i].astype(np.float32).max()))
                    w.writerow([k, code_of[k], int(c["hidden"]), c["tier"], classes[argmax[i]], f"{msp:.6f}", f"{-lse[i]:.6f}"])
            print(f"{which}: wrote {path}", flush=True)


if __name__ == "__main__":
    main()
