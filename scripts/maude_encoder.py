"""MAUDE encoder arms: roberta-base, CE or class-balanced CE, at a training size N.

Pre-registered in notes/prereg-maude-2026-10-07.md. The Stage 1a recipe
(encoder_finetune.py) carried over where it scales; what changes at 656k
training items, and why, is in the note:
- max_len 192 on lowercased text (95% of narratives fit), batches of 64
  built from length buckets, bf16 autocast
- epochs and patience by N: 1k / 10k -> up to 15, patience 3 (Stage 1a);
  100k -> up to 4, patience 1; full -> up to 2, patience 1
- lr in {2e-5, 5e-5} chosen by validation macro-F1 at every N <= 100k; at full
  N, the lr the CE seed-42 run chose at 100k
- "large" (notes/prereg-maude-rag-large-2026-10-09.md): roberta-large, lr in
  {2e-5, 3e-5}, 1e-5 if both collapse; at full N, the lr its own 100k seed-42 run chose
One model per run serves both tasks: it never sees the hidden classes, so it
classifies known test items and scores every test item for novelty (MSP and
energy, as in Stage 2a).

Usage: python maude_encoder.py <ce|cb> <1k|10k|100k|full> <seed> [eval2025] [large] [long|conv]
"""
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.utils import logging as hf_logging

from common import RUNS, rdir
from encoder_finetune import CB_BETA, MODEL_NAME, set_seed
from maude_data import (POOL_2025, class_info, known_codes, load_items, load_split,
                        load_test_2025, train_keys)

MAX_LEN = 192
BATCH = 64
EVAL_BATCH = 256
BUCKET = 50  # batches per length-sorted mega-batch
LR_GRID = [2e-5, 5e-5]
LR_GRID_LARGE = [2e-5, 3e-5]
LR_FALLBACK_LARGE = 1e-5  # tried only when the grid collapses
COLLAPSED = 0.01  # best val macro-F1 below this = a collapsed run
SIZES = {"1k": 1_000, "10k": 10_000, "100k": 100_000, "full": "full"}
SCHEDULE = {"1k": (15, 3), "10k": (15, 3), "100k": (4, 1), "full": (2, 1)}  # max epochs, patience
OUT = rdir("maude")
from common import CORPUS  # noqa: E402
WEIGHTS = Path.home() / "models" / f"{CORPUS}_encoders"  # outside OneDrive; ~500 MB each
# Another encoder family under the same recipe (2026-11-18, a check asked for after the paper's review):
# MAINT_ENC_MODEL = its Hugging Face name, MAINT_ENC_NAME = its tag part ("mbert" -> cb-mbert-conv_full_seed42).
ALT_MODEL, ALT_NAME = os.environ.get("MAINT_ENC_MODEL"), os.environ.get("MAINT_ENC_NAME")


def batches(lengths, rng):
    """Length-bucketed batches in random order (HF group_by_length style)."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    out = []
    for s in range(0, len(order), BATCH * BUCKET):
        mega = sorted(order[s:s + BATCH * BUCKET], key=lambda i: lengths[i])
        out += [mega[j:j + BATCH] for j in range(0, len(mega), BATCH)]
    rng.shuffle(out)
    return out


def collate(ids, idx, pad):
    L = max(len(ids[i]) for i in idx)
    x = torch.full((len(idx), L), pad, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        x[r, :len(ids[i])] = torch.tensor(ids[i])
        m[r, :len(ids[i])] = 1
    return x, m


def logits_for(model, ids, pad, device):
    model.eval()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.empty((len(ids), model.config.num_labels), dtype=torch.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), EVAL_BATCH):
            idx = order[s:s + EVAL_BATCH]
            x, m = collate(ids, idx, pad)
            out[idx] = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.float().cpu()
    return out


def cb_weights(train_y, n_classes):
    """Stage 1a's class-balanced weights (Cui et al., beta 0.999), from this
    run's own training subset; classes absent from it get weight 0."""
    counts = np.bincount(train_y, minlength=n_classes)
    w = np.where(counts > 0, (1 - CB_BETA) / (1 - CB_BETA ** np.maximum(counts, 1)), 0.0)
    w[w > 0] /= w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


def train_one(tr_ids, tr_y, va_ids, va_y, n_classes, lr, weights, size, seed, pad, device):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes).to(device)
    if MODEL_NAME == "roberta-large":
        # Recompute activations in the backward pass: same gradients, but without it
        # the run outgrows the 3090's 24 GB and spills into system RAM (~3x slower).
        model.gradient_checkpointing_enable()
    max_epochs, patience = SCHEDULE[size]
    lengths = [len(x) for x in tr_ids]
    steps = sum(1 for _ in range(0, len(tr_ids), BATCH)) * max_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    rng = random.Random(seed)
    best_f1, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        t0 = time.time()
        for idx in batches(lengths, rng):
            x, m = collate(tr_ids, idx, pad)
            y = torch.tensor([tr_y[i] for i in idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=x.to(device), attention_mask=m.to(device))
            loss = lossf(out.logits.float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        pred = logits_for(model, va_ids, pad, device).argmax(-1).tolist()
        f1 = f1_score(va_y, pred, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "val_macro_f1": round(f1, 4),
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(f"    lr={lr} epoch {epoch + 1}: val macro-F1 {f1:.4f} "
              f"({history[-1]['minutes']} min)", flush=True)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1, history


def main():
    global MODEL_NAME
    variant, size, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if variant not in ("ce", "cb") or size not in SIZES:
        sys.exit(__doc__)
    flags = sys.argv[4:]
    # eval2025 (notes/prereg-maude-2025-router-2026-10-08.md): retrain with the
    # same recipe, predict the 2025 test set too, keep the weights, and write the
    # 2024 predictions apart from the committed ones for a reproducibility check.
    eval2025 = "eval2025" in flags
    # large (notes/prereg-maude-rag-large-2026-10-09.md): roberta-large, the same
    # recipe and epoch budget; lr in {2e-5, 3e-5} (the base runs were limited by
    # this budget, so the grid is not lowered for the larger model), 1e-5 if both
    # collapse; outputs carry a "-large" tag beside the base-model ones.
    large = "large" in flags
    if large:
        MODEL_NAME = "roberta-large"
    tag = f"{variant}-large" if large else variant
    if ALT_MODEL:
        if large or not ALT_NAME:
            sys.exit("MAINT_ENC_MODEL needs MAINT_ENC_NAME and excludes the large flag")
        MODEL_NAME = ALT_MODEL
        tag = f"{variant}-{ALT_NAME}"
    # long (notes/prereg-sdr-amendment-2026-10-27.md): the full-N run on a second corpus
    # takes the 10k epoch budget (up to 15 epochs, patience 3) instead of MAUDE's 2-epoch
    # full-N schedule, which was written for 656k reports; outputs carry a "-long" tag.
    long_run = "long" in flags
    if long_run:
        if size != "full" or CORPUS == "maude":
            sys.exit("the long schedule is for a second corpus at full N only")
        tag += "-long"
        SCHEDULE["full"] = SCHEDULE["10k"]
    # conv (notes/redesign-2026-11-06.md, notes/prereg-redesign-2026-11-06.md): train until the
    # validation macro-F1 stops rising, under a stopping rule fixed before any run: stop after
    # "patience" epochs without a gain and keep the best epoch. MAUDE: 100k up to 10 epochs and
    # full N up to 6, patience 2; a second corpus at full N: up to 40, patience 5. 1k and 10k keep
    # the 15/3 schedule. The lr rule is unchanged; outputs carry a "-conv" tag.
    conv = "conv" in flags
    if conv:
        if long_run or size in ("1k", "10k") or (CORPUS != "maude" and size != "full"):
            sys.exit("conv is for MAUDE 100k/full or a second corpus at full N, and not with long")
        tag += "-conv"
        SCHEDULE.update({"100k": (10, 2), "full": (6, 2)} if CORPUS == "maude" else {"full": (40, 5)})
    lr_grid = LR_GRID_LARGE if large else LR_GRID
    out_dir = rdir("maude2025") / "rerun2024" if eval2025 else OUT
    out_csv = out_dir / f"encoder_{tag}_{size}_seed{seed}.csv"
    out_2025 = rdir("maude2025") / f"encoder_{tag}_{size}_seed{seed}.csv"
    out_meta = (rdir("maude2025") if eval2025 else OUT) / f"encoder_{tag}_{size}_seed{seed}_meta.json"
    # The meta file is written last, so a run cut off after its predictions reruns.
    if (out_meta if eval2025 else out_csv).exists():
        print(f"{tag}_{size}_seed{seed} already done; skipping", flush=True)
        return
    hf_logging.set_verbosity_error()
    device = "cuda"
    split = load_split()
    info = class_info(split)
    classes = known_codes(split)
    c2i = {c: i for i, c in enumerate(classes)}

    tr_keys = train_keys(split, SIZES[size])
    test = split["test"]
    items = load_items(list(tr_keys) + split["val"] + [t["key"] for t in test])
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(keys):
        return tok([items[k][0] for k in keys], truncation=True, max_length=MAX_LEN)["input_ids"]

    tr_ids, va_ids, te_ids = tokenize(tr_keys), tokenize(split["val"]), tokenize([t["key"] for t in test])
    tr_y = [c2i[items[k][1]] for k in tr_keys]
    va_y = [c2i[items[k][1]] for k in split["val"]]
    weights = cb_weights(tr_y, len(classes)) if variant == "cb" else None
    print(f"[maude {variant} N={size} seed={seed}] train {len(tr_keys):,} "
          f"({len(set(tr_y))}/{len(classes)} known classes present), val {len(va_y):,}, "
          f"test {len(test):,}", flush=True)

    if size == "full":
        ref_tag = tag.removesuffix("-conv")  # a conv run takes the lr that its namesake's rule names
        if CORPUS == "maude":
            ref_path = (out_meta.parent / f"encoder_{ref_tag}_100k_seed42_meta.json" if large
                        else OUT / "encoder_ce_100k_seed42_meta.json")
        else:  # a second corpus has no 100k run: full N takes the lr its own 10k seed-42 run chose (notes/prereg-sdr-2026-10-24.md)
            ref_path = out_meta.parent / f"encoder_{ref_tag.removesuffix('-long')}_10k_seed42_meta.json"
        grid = [json.loads(ref_path.read_text())["lr"]]
    else:
        grid = list(lr_grid)
    best, runs = None, {}
    for lr in grid:
        model, vf1, hist = train_one(tr_ids, tr_y, va_ids, va_y, len(classes), lr, weights,
                                     size, seed, tok.pad_token_id, device)
        runs[str(lr)] = {"val_macro_f1": round(vf1, 4), "epochs": hist}
        if best is None or vf1 > best[1]:
            best = (model, vf1, lr)  # the previous best loses its last reference here
        else:
            del model
        torch.cuda.empty_cache()
        if large and lr == grid[-1] and best[1] < COLLAPSED and LR_FALLBACK_LARGE not in grid:
            grid.append(LR_FALLBACK_LARGE)
    model, vf1, lr = best
    best_hist = runs[str(lr)]["epochs"]
    best_epoch = max(best_hist, key=lambda h: h["val_macro_f1"])["epoch"]

    def write_preds(path, test_items, ids):
        logits = logits_for(model, ids, tok.pad_token_id, device)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(-1).tolist()
        msp = (1.0 - probs.max(dim=-1).values).tolist()
        energy = (-torch.logsumexp(logits, dim=-1)).tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "gold", "is_novel", "tier", "pred", "msp_novelty", "energy_novelty"])
            for t, p, a, e in zip(test_items, pred, msp, energy):
                c = info[t["code"]]
                w.writerow([t["key"], t["code"], int(c["hidden"]), c["tier"], classes[p],
                            f"{a:.6f}", f"{e:.6f}"])

    meta = {"variant": variant, "model": MODEL_NAME, "size": size, "seed": seed, "lr": lr,
            "val_macro_f1": round(vf1, 4), "n_train": len(tr_keys), "classes_present": len(set(tr_y)),
            "lr_runs": runs, "max_len": MAX_LEN, "batch": BATCH, "schedule": SCHEDULE[size],
            "collapsed": vf1 < COLLAPSED,
            # epoch budget binding: the chosen lr tops a grid of 2+ and/or the best epoch was the last one run
            "lr_at_top_of_grid": len(runs) > 1 and lr == max(float(k) for k in runs),
            "best_at_last_epoch": best_epoch == best_hist[-1]["epoch"]}
    write_preds(out_csv, test, te_ids)
    if eval2025:
        test25 = load_test_2025()
        items25 = load_items([t["key"] for t in test25], pool=POOL_2025)
        ids25 = tok([items25[t["key"]][0] for t in test25], truncation=True,
                    max_length=MAX_LEN)["input_ids"]
        write_preds(out_2025, test25, ids25)
        weights_dir = WEIGHTS / f"{tag}_{size}_seed{seed}"
        model.save_pretrained(weights_dir)
        tok.save_pretrained(weights_dir)  # so that a later pass loads the tokenizer of this encoder family
        meta["weights"] = str(weights_dir)
    out_meta.write_text(json.dumps(meta, indent=1))
    print(f"  wrote {out_csv.name}{' + 2025 predictions + weights' if eval2025 else ''} "
          f"(lr={lr}, val macro-F1 {vf1:.4f})", flush=True)


if __name__ == "__main__":
    main()
