"""Development of a revised selector on 2024 only (cross-validation); reads no 2025 file.

Why: the frozen selector (v1) was fitted with every 2024 test item weighted equally, so it
learned to trust answers that name rare codes. At natural proportions that yields many false
rare-mode placements (notes/maude-downstream-results-2026-11-05.md). Two candidate fixes:
  weights   fit the per-coder models with natural-proportion weights (label-shift correction):
            item i of gold code c gets (share of c among 2024 training reports) / (share of c in
            the 2024 test items), tempered by an exponent alpha (0 = v1, 1 = fully natural)
  encview   add the encoder's view of each LLM answer: the log-probability and log-rank the
            encoder gives the code the zero-shot and the retrieval LLM propose
            (runs/maude_encprobs/2024.npz)
Every design is scored with the same 5 folds as v1 (stratified by training-count bin, seed 42):
  balanced  macro-F1 with every item weighted equally (v1's objective)
  natural   estimated with the natural-proportion weights: accuracy, population macro-F1 (every
            code weighted equally, false placements at their real volume), rare-mode (1-5)
            precision and false placements per 100k reports
  rare      share of the rare-code items coded right, rare codes with at least one item right

Usage: python maude_selector_v2_dev.py [--llm opus55]
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import RUNS, SEED
from maude_selector import BINS, Learned, bin_index, choose, f1_of, features, load_coders, targets

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def natural_weights(D):
    """Label-shift weights: natural share of the gold code / its share in this set; mean 1."""
    gold, counts = D["gold"], D["counts"]
    n_b = np.bincount(gold, minlength=len(counts)).astype(float)
    w = (counts[gold] / counts.sum()) / (n_b[gold] / len(gold))
    return w / w.mean()


def encview(D, path):
    """Encoder log-probability and log-rank of the code each LLM proposes; -1 rank / floor
    log-probability when the LLM answer is invalid (its invalid flag is already a feature)."""
    z = np.load(path, allow_pickle=False)
    keys = [int(k) for k in z["keys"]]
    assert sorted(keys) == sorted(int(k) for k in D["keys"]), "encoder probabilities cover other items"
    row = {k: i for i, k in enumerate(keys)}
    pos = {c: i for i, c in enumerate(z["classes"].tolist())}
    L = z["logp"].astype(np.float32)[[row[int(k)] for k in D["keys"]]]
    rank = (-L).argsort(axis=1).argsort(axis=1)  # 0 = the encoder's top code
    floor = float(L.min()) - 1.0
    cols = []
    for arm in ("zs", "rag"):
        a = D[arm]
        lp = np.full(len(a), floor)
        rk = np.full(len(a), float(L.shape[1]))
        for i, c in enumerate(a):
            if c >= 0 and D["codes"][c] in pos:
                lp[i] = L[i, pos[D["codes"][c]]]
                rk[i] = rank[i, pos[D["codes"][c]]]
        cols += [lp, np.log1p(rk)]
    return np.column_stack(cols)


def nbsupport(D, path):
    """How many of the 10 nearest coded training reports carry each coder's answer."""
    import json
    nb = json.loads(Path(path).read_text())
    idx = {c: i for i, c in enumerate(D["codes"])}
    cols = []
    for arm in ("enc", "zs", "rag"):
        a = D[arm]
        cols.append(np.array([sum(1 for _, c, _ in nb[str(k)] if idx[c] == a[i]) if a[i] >= 0 else 0
                              for i, k in enumerate(D["keys"])], dtype=float))
    return np.column_stack(cols)


def natural_scores(D, pred, w, rare_codes):
    gold = D["gold"]
    ok = pred == gold
    acc = float((w * ok).sum() / w.sum())
    labels = np.unique(gold)
    f1s = []
    for c in labels:
        tp = float(w[(gold == c) & ok].sum())
        fp = float(w[(gold != c) & (pred == c)].sum())
        fn = float(w[(gold == c) & ~ok].sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0)
    in_rare_pred = np.isin(pred, rare_codes)
    tp_r = float(w[in_rare_pred & ok].sum())
    fp_r = float(w[in_rare_pred & ~ok].sum())
    rare_items = np.isin(gold, rare_codes)
    vis = len(set(gold[rare_items & ok].tolist()))
    return {"nat_acc": acc, "pop_macro_f1": float(np.mean(f1s)),
            "rare_prec": tp_r / (tp_r + fp_r) if tp_r + fp_r > 0 else 0.0,
            "rare_fp_per_100k": fp_r / w.sum() * 1e5,
            "rare_recall": float(ok[rare_items].mean()), "rare_visible": vis, "rare_codes": len(set(gold[rare_items].tolist()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="opus55")
    a = ap.parse_args()
    D = load_coders("2024", a.llm)
    X1, Y = features(D), targets(D)
    w_nat = natural_weights(D)
    rare_codes = np.where((D["counts"] >= 1) & (D["counts"] <= 5))[0]
    enc_path = RUNS / "maude_encprobs" / "2024.npz"
    feats = {"v1": X1}
    if enc_path.exists():
        E = encview(D, enc_path)
        feats["v1+encview"] = np.column_stack([X1, E])
        from common import rdir
        Nb = nbsupport(D, rdir("maude") / "neighbors_full.json")
        feats["v1+encview+nbsupport"] = np.column_stack([X1, E, Nb])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X1, bin_index(D)))
    arms = {"enc": D["enc"], "zs": D["zs"], "rag": D["rag"]}
    rows = []
    for name, pred in arms.items():
        r = natural_scores(D, np.where(pred >= 0, pred, -1), w_nat, rare_codes)
        rows.append((f"single: {name}", f1_of(D, pred), r))
    for fname, X in feats.items():
        for alpha in ALPHAS:
            oof = np.full(len(D["gold"]), -1)
            for tr, te in folds:
                sw = w_nat[tr] ** alpha
                sw = sw / sw.mean()
                m = Learned("S2")
                m.models, m.constant = [], []
                for j in range(Y.shape[1]):
                    y = Y[tr, j]
                    if y.min() == y.max():
                        m.models.append(None)
                        m.constant.append(float(y[0]))
                        continue
                    from maude_selector import make_model
                    mdl = make_model("S2")
                    mdl.fit(X[tr], y, logisticregression__sample_weight=sw)
                    m.models.append(mdl)
                    m.constant.append(None)
                pred, _ = choose(m.proba(X[te]), {k: v[te] for k, v in arms.items()})
                oof[te] = pred
            rows.append((f"{fname}, alpha {alpha}", f1_of(D, oof), natural_scores(D, oof, w_nat, rare_codes)))
    print(f"2024 cross-validation, {len(D['gold'])} items, llm {a.llm}; natural weights: effective n "
          f"{w_nat.sum() ** 2 / (w_nat ** 2).sum():.0f}")
    print(f"{'design':32s} {'bal mF1':>8s} {'nat acc':>8s} {'pop mF1':>8s} {'rare prec':>9s} {'rare FP/100k':>12s} "
          f"{'rare rec':>8s} {'visible':>8s}")
    for name, bal, r in rows:
        print(f"{name:32s} {bal:8.4f} {r['nat_acc']:8.4f} {r['pop_macro_f1']:8.4f} {r['rare_prec']:9.3f} "
              f"{r['rare_fp_per_100k']:12.1f} {r['rare_recall']:8.3f} {r['rare_visible']:4d}/{r['rare_codes']}")


if __name__ == "__main__":
    main()
