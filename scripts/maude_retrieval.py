"""Nearest labelled training reports for each 2025 test item (retrieval for the
few-shot LLM arm and the kNN-vote baseline).

Pre-registered in notes/prereg-maude-rag-large-2026-10-09.md. The index is the
TF-IDF of maude_neardup.py (word 1-2-grams, English stop words, terms in more
than half the documents dropped, sublinear tf), fit on the full 2024 training
set. For a training size N, the labelled candidates are the first N keys of the
committed training order, the reports an encoder trained at N saw; the term
weights (vocabulary, IDF) come from all 2024 training text, without labels.

Writes runs/maude2025/neighbors_{N}.json: {test key: [[train key, code, cosine], ...]}
with the K most similar first. Usage: python maude_retrieval.py <10k|100k|full> [...]

With the extra argument "2024" the queries are the known 2024 test items instead
(selector training data, notes/prereg-maude-selector-2026-10-23.md) and the file
goes to runs/maude/neighbors_{N}.json. The candidates are training keys only, so a
2024 test item never retrieves itself.
"""
import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import RUNS, rdir
from maude_data import POOL_2025, load_items, load_split, load_test_2025, train_keys

K = 10
CHUNK = 64
SIZES = {"10k": 10_000, "100k": 100_000, "full": "full"}


def main():
    args = sys.argv[1:]
    which = ("2024val" if "2024val" in args else "2024" if "2024" in args else "natural" if "natural" in args
             else "2026nat" if "2026nat" in args else "2026" if "2026" in args else "2025")
    sizes = [a for a in args if a in SIZES] or ["10k", "full"]
    split = load_split()
    full = train_keys(split, "full")
    items = load_items(full)
    if which == "2025":
        test = [t["key"] for t in load_test_2025()]
        test_items = load_items(test, pool=POOL_2025)
        out_dir = rdir("maude2025")
    elif which == "2026":  # notes/prereg-maude-2026-2026-11-05.md
        from maude_data import POOL_2026, load_test_2026
        test = [t["key"] for t in load_test_2026()]
        test_items = load_items(test, pool=POOL_2026)
        out_dir = rdir("maude2026")
    elif which == "2024val":  # the training-year validation reports (notes/natural-selector-2026-11-09.md)
        from maude_data import load_val_2024
        test = [t["key"] for t in load_val_2024()]
        test_items = load_items(test)
        out_dir = rdir("maude2024val")
    elif which == "2026nat":  # the natural-distribution 2026 sample (notes/prereg-redesign-2026-11-06.md)
        from maude_data import POOL_2026, load_natural_2026
        test = [t["key"] for t in load_natural_2026()]
        test_items = load_items(test, pool=POOL_2026)
        out_dir = rdir("maude2026nat")
    elif which == "natural":  # the natural-distribution 2025 sample (notes/prereg-maude-natural-2026-10-23.md)
        from maude_data import load_natural_2025
        test = [t["key"] for t in load_natural_2025()]
        test_items = load_items(test, pool=POOL_2025)
        out_dir = rdir("maude2025nat")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        hidden = {c["code"] for c in split["classes"] if c["hidden"]}
        test = [t["key"] for t in split["test"] if t["code"] not in hidden]
        test_items = load_items(test)
        out_dir = rdir("maude")
    out_dir.mkdir(parents=True, exist_ok=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                          stop_words="english", dtype=np.float32)
    X = vec.fit_transform([items[k][0] for k in full])
    Q = vec.transform([test_items[k][0] for k in test])
    row_of = {k: i for i, k in enumerate(full)}
    for size in sizes:
        keys = full if SIZES[size] == "full" else split["train_order_prefix"][:SIZES[size]]
        XT = X[[row_of[k] for k in keys]].T.tocsr()
        out = {}
        for s in range(0, Q.shape[0], CHUNK):
            sims = (Q[s:s + CHUNK] @ XT).tocsr()
            for r in range(sims.shape[0]):
                row = sims.getrow(r)
                top = np.argsort(-row.data)[:K]
                out[test[s + r]] = [[keys[row.indices[j]], items[keys[row.indices[j]]][1],
                                     round(float(row.data[j]), 4)] for j in top]
        path = out_dir / f"neighbors_{size}.json"
        path.write_text(json.dumps(out))
        n_full = sum(len(v) == K for v in out.values())
        print(f"N={size}: {len(keys):,} candidates, {n_full}/{len(out)} test items with {K} neighbours "
              f"-> {path.name}", flush=True)


if __name__ == "__main__":
    main()
