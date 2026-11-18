"""TF-IDF + linear classifier baseline on MAUDE, a second supervised arm.

Pre-registered in notes/prereg-maude-rag-large-2026-10-09.md. At training size
N: TF-IDF fit on the N training reports (word 1-2-grams, English stop words,
terms in more than half the documents dropped, sublinear tf, the 200k most
frequent terms), then a linear SVM trained by SGD with balanced class weights
(one-vs-rest). alpha in {1e-5, 1e-6, 1e-7} is chosen by 2024 validation macro-F1 at
every N. Predicts the 2025 test set.

Writes runs/maude2025/linear_{N}.csv (key, gold, pred) and linear_{N}_meta.json.
Usage: python maude_linear.py <10k|100k|full> [...]
"""
import csv
import json
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from common import RUNS
from maude_data import POOL_2025, known_codes, load_items, load_split, load_test_2025, train_keys

SIZES = {"10k": 10_000, "100k": 100_000, "full": "full"}
ALPHAS = [1e-5, 1e-6, 1e-7]
MAX_FEATURES = 200_000
SEED = 42
OUT = RUNS / "maude2025"


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    split = load_split()
    classes = set(known_codes(split))
    test = load_test_2025()
    test_items = load_items([t["key"] for t in test], pool=POOL_2025)
    for size in sys.argv[1:] or ["10k", "100k", "full"]:
        out_csv, out_meta = OUT / f"linear_{size}.csv", OUT / f"linear_{size}_meta.json"
        if out_meta.exists():
            print(f"linear_{size} already done; skipping", flush=True)
            continue
        t0 = time.time()
        tr_keys = train_keys(split, SIZES[size])
        items = load_items(list(tr_keys) + split["val"])
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                              stop_words="english", max_features=MAX_FEATURES, dtype=np.float32)
        X = vec.fit_transform([items[k][0] for k in tr_keys])
        y = np.array([items[k][1] for k in tr_keys])
        Xv = vec.transform([items[k][0] for k in split["val"]])
        yv = np.array([items[k][1] for k in split["val"]])
        assert set(y) <= classes
        runs, best = {}, None
        for alpha in ALPHAS:
            clf = SGDClassifier(loss="hinge", alpha=alpha, class_weight="balanced", max_iter=20,
                                tol=1e-4, random_state=SEED, n_jobs=-1)
            clf.fit(X, y)
            f1 = f1_score(yv, clf.predict(Xv), average="macro", zero_division=0)
            runs[str(alpha)] = round(float(f1), 4)
            print(f"  linear N={size} alpha={alpha}: val macro-F1 {f1:.4f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            if best is None or f1 > best[1]:
                best = (clf, f1, alpha)
        clf, f1, alpha = best
        pred = clf.predict(vec.transform([test_items[t["key"]][0] for t in test]))
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "gold", "pred"])
            w.writerows([t["key"], t["code"], p] for t, p in zip(test, pred))
        out_meta.write_text(json.dumps({
            "size": size, "n_train": len(tr_keys), "n_features": len(vec.vocabulary_),
            "alpha": alpha, "val_macro_f1": round(float(f1), 4), "alpha_runs": runs,
            "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
        print(f"linear N={size}: alpha={alpha}, val macro-F1 {f1:.4f} -> {out_csv.name}", flush=True)


if __name__ == "__main__":
    main()
