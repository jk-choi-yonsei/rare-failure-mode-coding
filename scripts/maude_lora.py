"""MAUDE LoRA arm: a small decoder LLM fine-tuned as a classifier.

Pre-registered in notes/prereg-maude-lora-2026-10-09.md. The class-balanced
encoder recipe (maude_encoder.py) with the backbone swapped for Qwen3-4B-Base
and LoRA adapters. The same: lowercased narratives cut at 192 tokens,
length-bucketed batches of 64, bf16, class-balanced cross-entropy (beta 0.999)
over this run's own training subset, AdamW with 10% warmup and linear decay,
the epoch kept by 2024 validation macro-F1, and the outputs (2025 and 2024 test
predictions with MSP and energy, in the encoder CSV format). What differs:
- LoRA r 16, alpha 32, dropout 0.05 on every linear layer of the decoder; the
  new classification head (on the last token) is trained in full, starts at
  zero and has its own lr, HEAD_LR: the decoder's last hidden state has a norm
  near 180 (roberta's head input is a tanh layer), so at the LoRA lr every
  Adam step moved the logits by more than one and the pilot never learned;
  trainable weights in fp32, the frozen 4B in bf16; weights pinned to
  MODEL_REVISION
- lr in LR_GRID chosen by validation macro-F1 at 10k and at 100k (seed 42);
  other 10k seeds reuse the 10k choice, full N reuses the 100k choice (as the
  encoders reused 100k's lr at full N)
- epochs: SCHEDULE (a 4B epoch over the full training set takes ~10 h)
- gradient checkpointing (memory only; the same gradients); batches padded to a
  multiple of 16 tokens and the CUDA cache emptied every 200 steps (fewer tensor
  shapes, so the allocator does not fragment past the card's 24 GB and spill
  into system RAM; the attention mask makes the padding inert); peft's cast of
  every adapter input to fp32 is off (autocast runs those matmuls in bf16
  either way; +15% speed)
- collapse: the best validation macro-F1 below half the CB-base mean at that N;
  a collapsed 10k / 100k run tries LR_FALLBACK, and if that collapses too the
  run is marked collapsed (the driver replaces a collapsed 10k seed; a collapsed
  100k or full run leaves its cells incomplete); full N gets no fallback
- full N also logs validation at 25 / 50 / 75% of its one epoch (monitoring only)
- crash safety: an hourly checkpoint per (N, seed, lr) resumes an interrupted
  run, and a finished lr is kept until the run's meta file is written

Usage: python maude_lora.py <10k|100k|full> <seed>
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
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from common import RUNS
from encoder_finetune import set_seed
from maude_data import POOL_2025, class_info, known_codes, load_items, load_split, load_test_2025, train_keys
from maude_encoder import BATCH, MAX_LEN, SIZES, batches, cb_weights

MODEL_ID = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
TAG = "cb-lora"
LR_GRID = [1e-4, 3e-4]  # the LoRA adapters'
HEAD_LR = 1e-5
LR_FALLBACK = 5e-5
GRID_RUNS = {("10k", 42), ("100k", 42)}
INHERIT = {"10k": "10k", "100k": "100k", "full": "100k"}  # whose seed-42 lr a non-grid run reuses
# Half the CB-base mean validation macro-F1 at N (0.2348 / 0.4314 / 0.5669).
COLLAPSE = {"10k": 0.117, "100k": 0.216, "full": 0.283}
SCHEDULE = {"10k": (10, 2), "100k": (2, 1), "full": (1, 1)}  # max epochs, patience
PARTIAL_VAL = (0.25, 0.5, 0.75)  # full N only
PAD_MULTIPLE = 16
EMPTY_CACHE_STEPS = 200
CKPT_MINUTES = 60
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear")
EVAL_BATCH = 128
OUT = RUNS / "maude2025"
WEIGHTS = Path.home() / "models" / "maude_encoders"  # outside OneDrive; adapters ~140 MB


def collate(ids, idx, pad):
    L = max(len(ids[i]) for i in idx)
    L = -(-L // PAD_MULTIPLE) * PAD_MULTIPLE
    x = torch.full((len(idx), L), pad, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        x[r, :len(ids[i])] = torch.tensor(ids[i])
        m[r, :len(ids[i])] = 1
    return x, m


def build_model(n_classes, pad_id, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, num_labels=n_classes, dtype=torch.bfloat16)
    model.config.pad_token_id = pad_id  # the head reads the last non-pad token
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA))
    for mod in model.modules():  # no fp32 copy of every adapter input; autocast computes in bf16 anyway
        if isinstance(mod, BaseTunerLayer):
            mod.cast_input_dtype_enabled = False
    for name, p in model.named_parameters():  # adapters and head train in fp32; the head starts at zero
        if p.requires_grad:
            p.data = p.data.float()
            if "score" in name:
                p.data.zero_()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def load_trainable(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v)


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def logits_for(model, ids, pad, device):
    model.eval()
    torch.cuda.empty_cache()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.empty((len(ids), model.config.num_labels), dtype=torch.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), EVAL_BATCH):
            idx = order[s:s + EVAL_BATCH]
            x, m = collate(ids, idx, pad)
            out[idx] = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.float().cpu()
    return out


def val_f1(model, va_ids, va_y, pad, device):
    pred = logits_for(model, va_ids, pad, device).argmax(-1).tolist()
    return f1_score(va_y, pred, average="macro", zero_division=0)


def train_one(tr_ids, tr_y, va_ids, va_y, n_classes, lr, weights, size, seed, pad, device, ckpt):
    """(model with its best epoch restored, best val macro-F1, history, partial vals, best trainable state)."""
    set_seed(seed)
    model = build_model(n_classes, pad, device)
    head = [p for n, p in model.named_parameters() if p.requires_grad and "score" in n]
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and "score" not in n]
    max_epochs, patience = SCHEDULE[size]
    lengths = [len(x) for x in tr_ids]
    n_batches = sum(1 for _ in range(0, len(tr_ids), BATCH))
    opt = torch.optim.AdamW([{"params": adapters, "lr": lr}, {"params": head, "lr": HEAD_LR}], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * n_batches * max_epochs), n_batches * max_epochs)
    lossf = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    rng = random.Random(seed)
    st = {"epoch": 0, "step": 0, "rng": rng.getstate(), "history": [], "partial": [],
          "best_f1": -1.0, "best_state": None, "bad": 0}
    if ckpt.exists():
        ck = torch.load(ckpt, weights_only=False)
        load_trainable(model, ck["trainable"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        st = ck["st"]
        print(f"    lr={lr} resumed at epoch {st['epoch'] + 1}, step {st['step']}", flush=True)

    def save_ckpt():
        torch.cuda.empty_cache()
        atomic_save({"trainable": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "st": st}, ckpt)

    last_ckpt = time.time()
    while st["epoch"] < max_epochs:
        rng.setstate(st["rng"])  # the epoch's batch order, also when resuming inside it
        order = batches(lengths, rng)
        marks = {int(q * len(order)): q for q in PARTIAL_VAL} if size == "full" else {}
        model.train()
        t0, seen = time.time(), 0
        for step in range(st["step"], len(order)):
            idx = order[step]
            x, m = collate(tr_ids, idx, pad)
            y = torch.tensor([tr_y[i] for i in idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=x.to(device), attention_mask=m.to(device))
            loss = lossf(out.logits.float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            seen += len(idx)
            st["step"] = step + 1
            if st["step"] % EMPTY_CACHE_STEPS == 0:
                torch.cuda.empty_cache()
            if (step + 1) % 500 == 0 or seen == 50 * BATCH:  # speed and memory (a spill into system RAM shows here)
                print(f"    lr={lr} epoch {st['epoch'] + 1} step {step + 1}/{len(order)}: loss {loss.item():.3f}, "
                      f"{seen / (time.time() - t0):.1f} items/s, max reserved "
                      f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)
            if st["step"] in marks and all(p["fraction"] != marks[st["step"]] for p in st["partial"]):
                f1 = val_f1(model, va_ids, va_y, pad, device)
                st["partial"].append({"fraction": marks[st["step"]], "val_macro_f1": round(f1, 4)})
                print(f"    lr={lr} at {marks[st['step']]:.0%} of the epoch: val macro-F1 {f1:.4f} (monitoring only)",
                      flush=True)
                model.train()
            if time.time() - last_ckpt > CKPT_MINUTES * 60:
                save_ckpt()
                last_ckpt = time.time()
        f1 = val_f1(model, va_ids, va_y, pad, device)
        st["history"].append({"epoch": st["epoch"] + 1, "val_macro_f1": round(f1, 4),
                              "minutes": round((time.time() - t0) / 60, 1)})
        print(f"    lr={lr} epoch {st['epoch'] + 1}: val macro-F1 {f1:.4f} ({st['history'][-1]['minutes']} min)",
              flush=True)
        if f1 > st["best_f1"]:
            st["best_f1"], st["bad"], st["best_state"] = f1, 0, trainable_state(model)
        else:
            st["bad"] += 1
        st["epoch"] += 1
        st["step"] = 0
        st["rng"] = rng.getstate()
        stop = st["bad"] >= patience
        if st["epoch"] < max_epochs and not stop:
            save_ckpt()
            last_ckpt = time.time()
        if stop:
            break
    load_trainable(model, st["best_state"])
    return model, st["best_f1"], st["history"], st["partial"], st["best_state"]


def chosen_lr(size):
    """The lr the seed-42 grid run chose at INHERIT[size]; none if that run collapsed."""
    meta = json.loads((OUT / f"encoder_{TAG}_{INHERIT[size]}_seed42_meta.json").read_text())
    if meta["collapsed"]:
        sys.exit(f"{INHERIT[size]} seed 42 collapsed at every lr; no lr to reuse, so this run is not made")
    return meta["lr"]


def main():
    size, seed = sys.argv[1], int(sys.argv[2])
    if size not in SCHEDULE:
        sys.exit(__doc__)
    out_2025 = OUT / f"encoder_{TAG}_{size}_seed{seed}.csv"
    out_2024 = OUT / "rerun2024" / f"encoder_{TAG}_{size}_seed{seed}.csv"
    out_meta = OUT / f"encoder_{TAG}_{size}_seed{seed}_meta.json"
    if out_meta.exists():  # written last, so a run cut off after its predictions reruns
        print(f"{TAG}_{size}_seed{seed} already done; skipping", flush=True)
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
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def tokenize(texts):
        return tok(texts, truncation=True, max_length=MAX_LEN)["input_ids"]

    tr_ids = tokenize([items[k][0] for k in tr_keys])
    va_ids = tokenize([items[k][0] for k in split["val"]])
    te_ids = tokenize([items[t["key"]][0] for t in test])
    tr_y = [c2i[items[k][1]] for k in tr_keys]
    va_y = [c2i[items[k][1]] for k in split["val"]]
    weights = cb_weights(tr_y, len(classes))
    print(f"[maude lora {MODEL_ID}@{MODEL_REVISION[:8]} N={size} seed={seed}] train {len(tr_keys):,} "
          f"({len(set(tr_y))}/{len(classes)} known classes present), val {len(va_y):,}, "
          f"test {len(test):,}", flush=True)

    grid_run = (size, seed) in GRID_RUNS
    grid = list(LR_GRID) if grid_run else [chosen_lr(size)]
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    # Only the trainable weights of each lr's best epoch are kept (on the CPU and,
    # once an lr finishes, on disk): two 4B models at once would not fit the card.
    best, runs, model, model_lr = None, {}, None, None  # best = (state, val F1, lr, not collapsed)
    for lr in grid:
        done = WEIGHTS / f"done_{TAG}_{size}_seed{seed}_lr{lr}.pt"
        if done.exists():
            d = torch.load(done, weights_only=False)
            vf1, hist, partial, state = d["val_f1"], d["history"], d["partial"], d["state"]
            print(f"    lr={lr} already trained (val macro-F1 {vf1:.4f})", flush=True)
        else:
            if model is not None:
                del model
                model = None
                torch.cuda.empty_cache()
            ckpt = WEIGHTS / f"ckpt_{TAG}_{size}_seed{seed}_lr{lr}.pt"
            model, vf1, hist, partial, state = train_one(tr_ids, tr_y, va_ids, va_y, len(classes), lr, weights,
                                                         size, seed, pad, device, ckpt)
            model_lr = lr
            atomic_save({"val_f1": vf1, "history": hist, "partial": partial, "state": state}, done)
            ckpt.unlink(missing_ok=True)
        ok = vf1 >= COLLAPSE[size]
        runs[str(lr)] = {"val_macro_f1": round(vf1, 4), "epochs": hist, "partial": partial, "collapsed": not ok}
        if best is None or (ok and (not best[3] or vf1 > best[1])):
            best = (state, vf1, lr, ok)
        if lr == grid[-1] and not best[3] and size != "full" and LR_FALLBACK not in grid:
            grid.append(LR_FALLBACK)
    state, vf1, lr, ok = best
    if model is None or model_lr != lr:  # rebuild the chosen lr's model from its kept weights
        if model is not None:
            del model
            torch.cuda.empty_cache()
        set_seed(seed)
        model = build_model(len(classes), pad, device)
        load_trainable(model, state)
    best_hist = runs[str(lr)]["epochs"]
    best_epoch = max(best_hist, key=lambda h: h["val_macro_f1"])["epoch"]

    def write_preds(path, test_items, ids):
        logits = logits_for(model, ids, pad, device)
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

    write_preds(out_2024, test, te_ids)
    test25 = load_test_2025()
    items25 = load_items([t["key"] for t in test25], pool=POOL_2025)
    write_preds(out_2025, test25, tokenize([items25[t["key"]][0] for t in test25]))
    weights_dir = WEIGHTS / f"{TAG}_{size}_seed{seed}"
    model.save_pretrained(weights_dir)
    meta = {"variant": "cb", "model": MODEL_ID, "model_revision": MODEL_REVISION, "lora": LORA, "size": size,
            "seed": seed, "lr": lr, "head_lr": HEAD_LR, "lr_inherited_from": None if grid_run else f"{INHERIT[size]}_seed42",
            "val_macro_f1": round(vf1, 4), "collapse_threshold": COLLAPSE[size], "n_train": len(tr_keys),
            "classes_present": len(set(tr_y)), "lr_runs": runs, "max_len": MAX_LEN, "batch": BATCH,
            "schedule": SCHEDULE[size], "collapsed": not ok,
            "lr_at_top_of_grid": grid_run and lr == max(LR_GRID),
            "best_at_last_epoch": best_epoch == best_hist[-1]["epoch"], "weights": str(weights_dir)}
    out_meta.write_text(json.dumps(meta, indent=1))
    for lr_ in grid:
        (WEIGHTS / f"done_{TAG}_{size}_seed{seed}_lr{lr_}.pt").unlink(missing_ok=True)
    print(f"  wrote {out_2025.name} + 2024 predictions + adapter (lr={lr}, val macro-F1 {vf1:.4f}"
          f"{', COLLAPSED' if not ok else ''})", flush=True)


if __name__ == "__main__":
    main()
