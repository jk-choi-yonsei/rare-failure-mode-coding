"""Predict with a saved MAUDE encoder on the natural-distribution 2025 sample
(notes/prereg-maude-natural-2026-10-23.md). The weights are the ones maude_encoder.py
saved with eval2025 (~/models/maude_encoders/<tag>), so these are the same models
that produced the balanced 2025 test predictions. Output CSV has the same columns.

Usage: python maude_encoder_predict.py <tag> [natural|2026|2026nat|2026novel]     tag e.g. cb-large_full_seed42
2026nat: the natural-distribution 2026 sample; 2026novel: the 2026 reports of the hidden codes,
written as encoder_novel_<tag>.csv (notes/prereg-redesign-2026-11-06.md).
"""
import csv
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from common import RUNS, rdir
from encoder_finetune import MODEL_NAME as BASE_NAME
from maude_data import POOL_2025, class_info, known_codes, load_items, load_natural_2025, load_split
from maude_encoder import MAX_LEN, WEIGHTS, logits_for


def main():
    tag = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "natural"
    if which not in ("natural", "2026", "2026nat", "2026novel", "2024val"):
        sys.exit(__doc__)
    hf_logging.set_verbosity_error()
    split = load_split()
    info = class_info(split)
    classes = known_codes(split)
    if which == "2026":  # notes/prereg-maude-2026-2026-11-05.md
        from maude_data import POOL_2026, load_test_2026
        sample = load_test_2026()
        items = load_items([t["key"] for t in sample], pool=POOL_2026)
    elif which == "2024val":  # notes/natural-selector-2026-11-09.md
        from maude_data import POOL, load_val_2024
        sample = load_val_2024()
        items = load_items([t["key"] for t in sample], pool=POOL)
    elif which in ("2026nat", "2026novel"):  # notes/prereg-redesign-2026-11-06.md
        from maude_data import POOL_2026, load_natural_2026, load_novel_2026
        sample = load_natural_2026() if which == "2026nat" else load_novel_2026()
        items = load_items([t["key"] for t in sample], pool=POOL_2026)
    else:
        sample = load_natural_2025()
        items = load_items([t["key"] for t in sample], pool=POOL_2025)
    name = "roberta-large" if "-large" in tag else BASE_NAME
    tok = AutoTokenizer.from_pretrained(name)
    weights_dir = WEIGHTS / tag
    model = AutoModelForSequenceClassification.from_pretrained(weights_dir).to("cuda")
    assert model.config.num_labels == len(classes), (model.config.num_labels, len(classes))
    ids = tok([items[t["key"]][0] for t in sample], truncation=True, max_length=MAX_LEN)["input_ids"]
    logits = logits_for(model, ids, tok.pad_token_id, "cuda")
    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(-1).tolist()
    msp = (1.0 - probs.max(dim=-1).values).tolist()
    energy = (-torch.logsumexp(logits, dim=-1)).tolist()
    out = {"2026": rdir("maude2026") / f"encoder_{tag}.csv",
           "2026nat": rdir("maude2026nat") / f"encoder_{tag}.csv",
           "2026novel": rdir("maude2026") / f"encoder_novel_{tag}.csv",
           "2024val": rdir("maude2024val") / f"encoder_{tag}.csv",
           "natural": rdir("maude2025nat") / f"encoder_{tag}.csv"}[which]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "gold", "is_novel", "tier", "pred", "msp_novelty", "energy_novelty"])
        for t, p, a, e in zip(sample, pred, msp, energy):
            c = info[t["code"]]
            w.writerow([t["key"], t["code"], int(c["hidden"]), c["tier"], classes[p], f"{a:.6f}", f"{e:.6f}"])
    acc = sum(classes[p] == t["code"] for t, p in zip(sample, pred)) / len(sample)
    print(f"{tag}: {len(sample):,} {which} predictions -> {out.name} (accuracy {acc:.4f})", flush=True)


if __name__ == "__main__":
    main()
