"""Stage 2a encoder arms: train with classes hidden, score novelty (MSP / energy).

Pre-registered in notes/prereg-stage2-novelmode-2026-08-19.md.
Usage: python novelmode_encoder.py <fmc|maintie_m1> <rotation 0-2> <seed>
"""
import csv
import json
import sys

import torch
from sklearn.model_selection import train_test_split

from common import RUNS, SPLITS
from encoder_finetune import (MODEL_NAME, encode, inner_split, load_task,
                              predict, set_seed, train_one)
from transformers import AutoTokenizer

ROT_FILES = {"fmc": "fmc_novel_rotations.json",
             "maintie_m1": "maintie_m1_novel_rotations.json"}

# The known/eval 80-20 split is held fixed across training seeds so that every
# seed — and the LLM arms — score the identical item set. Only model training
# varies with the seed. Fixing it at 42 leaves the seed-42 item set (and the
# already-submitted LLM batches, which were built from it) unchanged.
SPLIT_SEED = 42


def novelty_scores(model, device, ids, am):
    """Return (pred_idx, msp_novelty, energy_novelty) per item.

    msp_novelty = 1 - max softmax; energy_novelty = -logsumexp(logits)
    (higher = more novel for both)."""
    model.eval()
    preds, msp, energy = [], [], []
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            logits = model(input_ids=ids[i:i + 64].to(device),
                           attention_mask=am[i:i + 64].to(device)).logits.float()
            probs = torch.softmax(logits, dim=-1)
            preds.extend(logits.argmax(-1).cpu().tolist())
            msp.extend((1.0 - probs.max(dim=-1).values).cpu().tolist())
            energy.extend((-torch.logsumexp(logits, dim=-1)).cpu().tolist())
    return preds, msp, energy


def main():
    ds, rot, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids_all, texts, labels, folds, _ = load_task(ds)
    hidden = set(json.loads((SPLITS / ROT_FILES[ds]).read_text(encoding="utf-8"))
                 ["rotations"][rot])
    print(f"[{ds} rot{rot} seed={seed}] device={device} hiding {sorted(hidden)}")

    known_i = [i for i, l in enumerate(labels) if l not in hidden]
    novel_i = [i for i, l in enumerate(labels) if l in hidden]
    classes = sorted({labels[i] for i in known_i})
    c2i = {c: j for j, c in enumerate(classes)}
    y = [c2i.get(l, -1) for l in labels]

    # 80/20 split of known records, stratified where possible
    counts = {}
    for i in known_i:
        counts[y[i]] = counts.get(y[i], 0) + 1
    strat_ok = [i for i in known_i if counts[y[i]] >= 2]
    singles = [i for i in known_i if counts[y[i]] < 2]
    n_test = max(int(round(0.2 * len(strat_ok))), len(set(y[i] for i in strat_ok)))
    fit_i, test_known = train_test_split(
        strat_ok, test_size=n_test, random_state=SPLIT_SEED,
        stratify=[y[i] for i in strat_ok])
    fit_i = fit_i + singles

    trr, var = inner_split(fit_i, y, seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    best = (None, -1.0, None)
    for lr in (2e-5, 5e-5):
        set_seed(seed)
        model, vf1 = train_one(tok, device,
                               [texts[i] for i in trr], [y[i] for i in trr],
                               [texts[i] for i in var], [y[i] for i in var],
                               len(classes), lr, None)
        if vf1 > best[1]:
            best = (model, vf1, lr)
        else:
            del model
        torch.cuda.empty_cache()
    model, vf1, lr = best
    print(f"  trained on {len(trr)} (val {len(var)}), lr={lr} val_mF1={vf1:.3f}, "
          f"{len(classes)} known classes")

    eval_i = test_known + novel_i
    e_ids, e_am = encode(tok, [texts[i] for i in eval_i])
    pred, msp, energy = novelty_scores(model, device, e_ids, e_am)

    out_dir = RUNS / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"encoder_{ds}_rot{rot}_seed{seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "is_novel", "gold", "pred", "msp_novelty", "energy_novelty"])
        for k, i in enumerate(eval_i):
            w.writerow([ids_all[i], int(i in set(novel_i)), labels[i],
                        classes[pred[k]], f"{msp[k]:.6f}", f"{energy[k]:.6f}"])
    (out_dir / f"encoder_{ds}_rot{rot}_seed{seed}_meta.json").write_text(json.dumps(
        {"hidden": sorted(hidden), "lr": lr, "val_macro_f1": round(vf1, 4),
         "n_known_classes": len(classes), "n_train": len(trr),
         "n_test_known": len(test_known), "n_novel": len(novel_i)}, indent=1))
    print(f"  wrote {path.name} ({len(test_known)} known + {len(novel_i)} novel)")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
