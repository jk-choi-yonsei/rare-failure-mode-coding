"""Stage 0 baseline: TF-IDF + LinearSVC, 5-fold out-of-fold predictions on FMC.

Configuration is pre-registered and fixed: word 1-2 grams, min_df=1,
LinearSVC(C=1.0, random_state=42). No tuning at Stage 0.
"""
import csv
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from common import RUNS, SEED, SPLITS, load_fmc


def main():
    fmc = load_fmc()
    task = json.loads((SPLITS / "fmc_task.json").read_text(encoding="utf-8"))
    fold_of = {r["gid"]: r["fold"] for r in task["records"]}
    out_dir = RUNS / "stage0"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = {}
    for f in range(5):
        train = [r for r in fmc if fold_of[r["gid"]] != f]
        test = [r for r in fmc if fold_of[r["gid"]] == f]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
        X = vec.fit_transform([r["text"] for r in train])
        clf = LinearSVC(C=1.0, random_state=SEED)
        clf.fit(X, [r["label"] for r in train])
        for r, p in zip(test, clf.predict(vec.transform([r["text"] for r in test]))):
            preds[r["gid"]] = p

    with (out_dir / "svm_fmc_preds.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gid", "pred"])
        for gid in sorted(preds):
            w.writerow([gid, preds[gid]])
    print(f"wrote {out_dir / 'svm_fmc_preds.csv'} ({len(preds)} OOF predictions)")


if __name__ == "__main__":
    main()
