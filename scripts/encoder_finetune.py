"""Stage 1a: fine-tuned encoder arms (CE / class-balanced CE) with OOF preds.

Pre-registered in notes/prereg-stage1a-encoder-2026-08-18.md.
Usage: python encoder_finetune.py <fmc|maintie_m1> <seed>
"""
import csv
import json
import random
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import RUNS, SPLITS, load_fmc, load_maintie_gold

# Pre-registered fallback (notes/prereg-stage1a-encoder-2026-08-18.md):
# microsoft/deberta-v3-base ships only pytorch_model.bin, which transformers
# 5.x refuses to load (safetensors required). Falling back to roberta-base,
# which publishes model.safetensors. Reason recorded in the journal commit.
MODEL_NAME = "roberta-base"
MAX_LEN = 64
BATCH = 16
EVAL_BATCH = 64
MAX_EPOCHS = 15
PATIENCE = 3
LR_GRID = [2e-5, 5e-5]
CB_BETA = 0.999


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_task(name):
    """Return (ids, texts, labels, folds, classes)."""
    if name == "fmc":
        rows = {r["gid"]: r for r in load_fmc()}
        task = json.loads((SPLITS / "fmc_task.json").read_text(encoding="utf-8"))
        recs = [(r["gid"], rows[r["gid"]]["text"], r["label"], r["fold"])
                for r in task["records"]]
    elif name == "maintie_m1":
        gold = load_maintie_gold()
        task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
        recs = [(r["idx"], gold[r["idx"]]["text"], r["label"], r["fold"])
                for r in task["records"]]
    else:
        sys.exit(f"unknown dataset {name}")
    ids, texts, labels, folds = zip(*recs)
    classes = sorted(set(labels))
    return list(ids), list(texts), list(labels), list(folds), classes


def encode(tok, texts):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def cb_weights(train_labels, classes):
    counts = {c: 0 for c in classes}
    for l in train_labels:
        counts[l] += 1
    w = np.zeros(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        n = counts[c]
        w[i] = (1 - CB_BETA) / (1 - CB_BETA ** n) if n > 0 else 0.0
    nz = w[w > 0]
    if len(nz):
        w[w > 0] = w[w > 0] / nz.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_one(tok, device, tr_texts, tr_y, va_texts, va_y, n_classes, lr, weights):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes).to(device)
    ids, am = encode(tok, tr_texts)
    ds = TensorDataset(ids, am, torch.tensor(tr_y))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(0))
    steps = len(dl) * MAX_EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    vi, va_am = encode(tok, va_texts)
    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for b_ids, b_am, b_y in dl:
            opt.zero_grad()
            out = model(input_ids=b_ids.to(device), attention_mask=b_am.to(device))
            loss = lossf(out.logits, b_y.to(device))
            loss.backward()
            opt.step()
            sched.step()
        preds = predict(model, device, vi, va_am)
        f1 = f1_score(va_y, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def predict(model, device, ids, am):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), EVAL_BATCH):
            logits = model(input_ids=ids[i:i + EVAL_BATCH].to(device),
                           attention_mask=am[i:i + EVAL_BATCH].to(device)).logits
            out.extend(logits.argmax(-1).cpu().tolist())
    return out


def inner_split(tr_i, y, seed):
    """Stratified 90/10 validation split of a training fold.

    Classes with a single member in this fold cannot be stratified, so they
    are assigned to train (where they are at least learnable) and excluded
    from the stratification; every remaining class has >= 2 members, which
    is what train_test_split requires.
    """
    counts = {}
    for i in tr_i:
        counts[y[i]] = counts.get(y[i], 0) + 1
    splittable = [i for i in tr_i if counts[y[i]] >= 2]
    singletons = [i for i in tr_i if counts[y[i]] < 2]
    strat = [y[i] for i in splittable]
    # A stratified split needs at least one validation slot per class. On
    # MaintIE (116 classes) 10% falls below that, so the validation set is
    # floored at the class count -- 10% -> 10.4% there, and untouched on FMC.
    n_val = max(int(round(0.1 * len(splittable))), len(set(strat)))
    tr_j, va_j = train_test_split(splittable, test_size=n_val,
                                  random_state=seed, stratify=strat)
    return tr_j + singletons, va_j


def main():
    ds_name, seed = sys.argv[1], int(sys.argv[2])
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{ds_name} seed={seed}] device={device}")
    ids, texts, labels, folds, classes = load_task(ds_name)
    c2i = {c: i for i, c in enumerate(classes)}
    y = [c2i[l] for l in labels]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    out_dir = RUNS / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in ("ce", "cb"):
        preds_all, meta = {}, {}
        for f in range(5):
            tr_i = [i for i, fl in enumerate(folds) if fl != f]
            te_i = [i for i, fl in enumerate(folds) if fl == f]
            trr, var = inner_split(tr_i, y, seed)
            weights = cb_weights([labels[i] for i in trr], classes) if variant == "cb" else None
            best = (None, -1.0, None)
            for lr in LR_GRID:
                set_seed(seed)
                model, vf1 = train_one(tok, device,
                                       [texts[i] for i in trr], [y[i] for i in trr],
                                       [texts[i] for i in var], [y[i] for i in var],
                                       len(classes), lr, weights)
                if vf1 > best[1]:
                    best = (model, vf1, lr)
                else:
                    del model
                torch.cuda.empty_cache()
            model, vf1, lr = best
            ti, ta = encode(tok, [texts[i] for i in te_i])
            for i, p in zip(te_i, predict(model, device, ti, ta)):
                preds_all[ids[i]] = classes[p]
            meta[f"fold{f}"] = {"lr": lr, "val_macro_f1": round(vf1, 4)}
            print(f"  {variant} fold{f}: lr={lr} val_mF1={vf1:.3f}")
            del model
            torch.cuda.empty_cache()
        out = out_dir / f"encoder_{ds_name}_{variant}_seed{seed}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "pred"])
            for k in sorted(preds_all):
                w.writerow([k, preds_all[k]])
        (out_dir / f"encoder_{ds_name}_{variant}_seed{seed}_meta.json").write_text(
            json.dumps(meta, indent=1))
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
