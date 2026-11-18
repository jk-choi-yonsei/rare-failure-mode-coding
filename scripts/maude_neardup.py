"""Near-duplicate check for the MAUDE split (pre-registered sensitivity analysis).

Deduplication removed exact and digit-masked copies, but template variants that
differ by a word or a product name remain and can sit on both sides of the
split, which favours a model that memorizes. For each test item this finds the
most similar item in the full training set by TF-IDF cosine (word 1-2-grams,
English stop words removed, terms in more than half the documents dropped,
sublinear tf), and maude_score.py re-runs the primary contrasts without test
items at cosine >= 0.9.

Writes runs/maude/neardup.json {test key: [max cosine, nearest training key]}.
Usage: python maude_neardup.py
"""
import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import RUNS
from maude_data import POOL_2025, load_items, load_split, load_test_2025, train_keys

CHUNK = 64


def main():
    """`python maude_neardup.py 2025` checks the out-of-time test items against
    the same 2024 training set and writes runs/maude2025/neardup.json."""
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    split = load_split()
    train = train_keys(split, "full")
    items = load_items(train)
    if year == "2025":
        test = [t["key"] for t in load_test_2025()]
        test_items = load_items(test, pool=POOL_2025)
    else:
        test = [t["key"] for t in split["test"]]
        test_items = load_items(test)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                          stop_words="english", dtype=np.float32)
    X = vec.fit_transform([items[k][0] for k in train])
    Q = vec.transform([test_items[k][0] for k in test])
    print(f"vocabulary {len(vec.vocabulary_):,}; train {X.shape[0]:,} x nnz {X.nnz:,}", flush=True)
    XT = X.T.tocsr()
    best = {}
    for s in range(0, Q.shape[0], CHUNK):
        sims = (Q[s:s + CHUNK] @ XT).tocsr()
        for r in range(sims.shape[0]):
            row = sims.getrow(r)
            if row.nnz:
                j = row.indices[row.data.argmax()]
                best[test[s + r]] = [round(float(row.data.max()), 4), train[j]]
            else:
                best[test[s + r]] = [0.0, None]
        if (s // CHUNK) % 10 == 0:
            print(f"  {s + sims.shape[0]}/{Q.shape[0]}", flush=True)
    out = RUNS / ("maude2025" if year == "2025" else "maude") / "neardup.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best))
    cos = np.array([v[0] for v in best.values()])
    print(f"wrote {out.name}: share of test items with a training neighbour at cosine >= 0.9: "
          f"{(cos >= 0.9).mean():.1%}; >= 0.8: {(cos >= 0.8).mean():.1%}; median {np.median(cos):.3f}")


if __name__ == "__main__":
    main()
