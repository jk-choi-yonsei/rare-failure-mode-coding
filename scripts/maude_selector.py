"""Starvation-aware selector (notes/prereg-maude-selector-2026-10-23.md).

Per report, choose among three coders at full N: the class-balanced roberta-large
(enc), the zero-shot LLM (zs) and the retrieval few-shot LLM (rag, the 10 nearest
labelled 2024 training reports in the prompt). The selector is trained on the known
2024 test items (3,829; the retrained encoders' rerun2024 outputs, the LLM's 2024
outputs, runs/maude/neighbors_full.json), frozen to splits/maude_selector_v1.json
(+ .joblib for the fitted sklearn models), and scored once on the 2025 test by
maude_selector_score.py. Nothing here reads a 2025 label.

Families (one is chosen by 5-fold cross-validated macro-F1 on 2024, folds stratified
by the gold code's training-count bin):
  S1  hand rule: rag if the 10 neighbours' top label share >= t; else zs if the zs
      answer names a code with <= k training examples; else enc. (t, k) by grid.
  S2  three logistic regressions (one per coder, "is this coder right?"), pick the
      coder with the highest probability; ties and invalid answers fall to enc.
  S3  three HistGradientBoosting classifiers, same decision rule.
Also frozen: S-cost (call the LLM only when the encoder predicts a code with <= k
examples or its confidence < p, call rate <= 50% on 2024), llm_rare(k*) and the
best single arm on 2024, all for the 2025 comparison.

Usage: python maude_selector.py fit [--llm opus55]
       python maude_selector.py features <2024|2025> [--llm opus55]   (shapes only)
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import maude_llm as ml
from common import RUNS, SEED, SPLITS, rdir
from maude_data import POOL, POOL_2025, POOL_2026, load_items, load_split
from maude_router import K_GRID, P_GRID, load_eval, macro_f1

BINS = {"1-5": (1, 5), "6-20": (6, 20), "21-89": (21, 89), "90+": (90, 10**9)}
T_GRID = [round(x, 1) for x in np.arange(0.3, 1.01, 0.1)]
KZ_GRID = [0, 5, 20, 89, 10**9]
FEATURES = ["log_count_enc", "enc_msp", "log_count_zs", "zs_invalid", "log_count_rag", "rag_invalid",
            "nb_vote_share", "nb_vote_is_rag", "nb_top1_cos", "agree_enc_zs", "agree_enc_rag",
            "agree_zs_rag", "agree_all", "log_text_len"]
ARMS = ("enc", "zs", "rag")
# The encoder arm's tag: cb-large (roberta-large, MAUDE) unless MAINT_SELECTOR_ENC says
# otherwise (the SDR replication uses the class-balanced roberta-base, "cb", first).
import os  # noqa: E402

SELECTOR_ENC = os.environ.get("MAINT_SELECTOR_ENC", "cb-large")
ENC_CSV = f"encoder_{SELECTOR_ENC}_full_seed42.csv"


def frozen_paths(llm):
    """(manifest, joblib) for the selector built on one LLM; Opus keeps the original name."""
    from common import CORPUS
    stem = f"{CORPUS}_selector_v1" + ("" if llm == "opus55" else f"_{llm}") + ("" if SELECTOR_ENC == "cb-large" else f"_{SELECTOR_ENC}")
    return SPLITS / f"{stem}.json", SPLITS / f"{stem}.joblib"


# ---------------------------------------------------------------- loading
def load_llm_file(ev, path):
    """Predictions (code index, -1 if invalid or missing) from a runner output file."""
    if not path.exists():
        return None, {"present": False}
    split = load_split()
    legal = sorted(ml.known_labels(split))
    path_to_code = ml.known_labels(split)
    idx = {c: i for i, c in enumerate(ev["codes"])}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[int(r["custom_id"][1:])] = r
    pred, missing, invalid = [], 0, 0
    for k in ev["keys"]:
        r = rows.get(k)
        if r is None:
            missing += 1
            pred.append(-1)
            continue
        code = path_to_code.get(ml.normalize(r.get("text", ""), legal)) if r["type"] == "succeeded" else None
        invalid += code is None
        pred.append(idx[code] if code else -1)
    return np.array(pred), {"present": True, "missing": missing, "invalid": invalid, "complete": missing == 0}


def load_neighbours(path, ev):
    nb = json.loads(path.read_text())
    idx = {c: i for i, c in enumerate(ev["codes"])}
    share, vote, cos = np.zeros(len(ev["keys"])), np.full(len(ev["keys"]), -1), np.zeros(len(ev["keys"]))
    for i, k in enumerate(ev["keys"]):
        lst = nb[str(k)]
        if not lst:
            continue
        labs = np.array([idx[c] for _, c, _ in lst])
        vals, cnt = np.unique(labs, return_counts=True)
        share[i], vote[i], cos[i] = cnt.max() / len(labs), vals[cnt.argmax()], lst[0][2]
    return share, vote, cos


def load_coders(which, llm):
    """Everything the selector needs for one evaluation set, in the fixed item order."""
    if which == "2024":
        ev = load_eval("2024rerun")
        enc_path = rdir("maude2025") / "rerun2024" / ENC_CSV
        llm_dir, nb_path, pool = rdir("maude"), rdir("maude") / "neighbors_full.json", POOL
    elif which == "2025":
        ev = load_eval("2025")
        enc_path = rdir("maude2025") / ENC_CSV
        llm_dir, nb_path, pool = rdir("maude2025"), rdir("maude2025") / "neighbors_full.json", POOL_2025
    elif which == "2026":  # notes/prereg-maude-2026-2026-11-05.md
        ev = load_eval("2026")
        enc_path = rdir("maude2026") / ENC_CSV
        llm_dir, nb_path, pool = rdir("maude2026"), rdir("maude2026") / "neighbors_full.json", POOL_2026
    elif which == "2024val":  # the training-year validation reports (notes/natural-selector-2026-11-09.md)
        from maude_data import class_info, load_val_2024
        from maude_router import training_counts
        split = load_split()
        info = class_info(split)
        codes = sorted(info)
        idx = {c: i for i, c in enumerate(codes)}
        val = load_val_2024()
        gold = np.array([idx[t["code"]] for t in val])
        ev = {"keys": [t["key"] for t in val], "gold": gold, "codes": codes,
              "labels": np.array(sorted(set(gold.tolist()))), "counts": training_counts(split, codes)}
        d = rdir("maude2024val")
        enc_path = d / ENC_CSV
        llm_dir, nb_path, pool = d, d / "neighbors_full.json", POOL
    elif which in ("2025nat", "2026nat"):  # natural-distribution samples (notes/prereg-maude-natural-2026-10-23.md;
        # 2026: notes/prereg-redesign-2026-11-06.md)
        from maude_data import class_info, load_natural_2025, load_natural_2026
        from maude_router import training_counts
        split = load_split()
        info = class_info(split)
        codes = sorted(info)
        idx = {c: i for i, c in enumerate(codes)}
        nat = load_natural_2025() if which == "2025nat" else load_natural_2026()
        gold = np.array([idx[t["code"]] for t in nat])
        ev = {"keys": [t["key"] for t in nat], "gold": gold, "codes": codes,
              "labels": np.array(sorted(set(gold.tolist()))), "counts": training_counts(split, codes)}
        d = rdir("maude" + which)
        enc_path = d / ENC_CSV
        llm_dir, nb_path, pool = d, d / "neighbors_full.json", (POOL_2025 if which == "2025nat" else POOL_2026)
    else:
        raise ValueError(which)
    from maude_extra_score import read_enc_csv
    enc, enc_p = read_enc_csv(ev, enc_path)
    zs, zs_st = load_llm_file(ev, llm_dir / f"llm_cls_{llm}.jsonl")
    rag, rag_st = load_llm_file(ev, llm_dir / f"llm_ragfull_{llm}.jsonl")
    for name, st in (("zero-shot", zs_st), ("retrieval", rag_st)):
        if not st.get("complete"):
            raise SystemExit(f"{which} {llm} {name} output incomplete: {st}")
    share, vote, cos = load_neighbours(nb_path, ev)
    texts = load_items(ev["keys"], pool=pool)
    text_len = np.array([len(texts[k][0]) for k in ev["keys"]])
    counts = ev["counts"]["full"]
    bins = {b: np.array([c for c in ev["labels"] if lo <= counts[c] <= hi]) for b, (lo, hi) in BINS.items()}
    return {"which": which, "llm": llm, "keys": ev["keys"], "gold": ev["gold"], "labels": ev["labels"], "codes": ev["codes"],
            "counts": counts, "bins": bins, "enc": enc, "enc_p": enc_p, "zs": zs, "rag": rag,
            "nb_share": share, "nb_vote": vote, "nb_cos": cos, "text_len": text_len,
            "status": {"zs": zs_st, "rag": rag_st}}


# ---------------------------------------------------------------- features and decisions
def features(D):
    c = D["counts"]
    enc, zs, rag = D["enc"], D["zs"], D["rag"]

    def logc(pred):
        return np.where(pred >= 0, np.log1p(c[np.maximum(pred, 0)]), -1.0)
    X = np.column_stack([
        logc(enc), D["enc_p"], logc(zs), (zs < 0).astype(float), logc(rag), (rag < 0).astype(float),
        D["nb_share"], ((D["nb_vote"] == rag) & (rag >= 0)).astype(float), D["nb_cos"],
        ((enc == zs) & (zs >= 0)).astype(float), ((enc == rag) & (rag >= 0)).astype(float),
        ((zs == rag) & (zs >= 0)).astype(float), ((enc == zs) & (zs == rag) & (zs >= 0)).astype(float),
        np.log1p(D["text_len"]),
    ])
    assert X.shape[1] == len(FEATURES)
    return X


def targets(D):
    return np.column_stack([D[a] == D["gold"] for a in ARMS]).astype(int)


def arms_matrix(D):
    return np.stack([D["enc"], D["zs"], D["rag"]], axis=1)


def choose(P, D):
    """Pick the coder with the highest probability of being right; invalid answers are
    never chosen; ties go to the first arm in ARMS order (enc)."""
    arms = arms_matrix(D)
    P = np.where(arms >= 0, P, -1.0)
    j = np.argmax(P, axis=1)
    return arms[np.arange(len(j)), j], j


def rule3(D, t, k):
    use_rag = (D["nb_share"] >= t) & (D["rag"] >= 0)
    use_zs = ~use_rag & (D["zs"] >= 0) & (D["counts"][np.maximum(D["zs"], 0)] <= k)
    return np.where(use_rag, D["rag"], np.where(use_zs, D["zs"], D["enc"]))


def llm_rare(D, k):
    use = (D["zs"] >= 0) & (D["counts"][np.maximum(D["zs"], 0)] <= k)
    return np.where(use, D["zs"], D["enc"])


def gate(D, k, p):
    return (D["counts"][D["enc"]] <= k) | (D["enc_p"] < p)


def f1_of(D, pred, sample=None):
    return macro_f1(D["gold"], pred, D["labels"], sample)


def bin_index(D):
    out = np.zeros(len(D["gold"]), int)
    for i, (lo, hi) in enumerate(BINS.values()):
        c = D["counts"][D["gold"]]
        out[(c >= lo) & (c <= hi)] = i
    return out


# ---------------------------------------------------------------- learners
def make_model(family):
    if family == "S2":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if family == "S3":
        return HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.1, random_state=SEED)
    raise ValueError(family)


class Learned:
    """Three binary models, one per coder."""

    def __init__(self, family):
        self.family, self.models, self.constant = family, [], []

    def fit(self, X, Y):
        self.models, self.constant = [], []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if y.min() == y.max():  # a coder that is always right or always wrong in this fold
                self.models.append(None)
                self.constant.append(float(y[0]))
            else:
                m = make_model(self.family)
                m.fit(X, y)
                self.models.append(m)
                self.constant.append(None)
        return self

    def proba(self, X):
        cols = []
        for m, c in zip(self.models, self.constant):
            cols.append(np.full(len(X), c) if m is None else m.predict_proba(X)[:, 1])
        return np.column_stack(cols)


def fit_s1(D, idx):
    """Best (t, k) of the hand rule on the items idx."""
    best = max(((t, k) for t in T_GRID for k in KZ_GRID), key=lambda tk: f1_of(D, rule3(D, *tk), idx))
    return {"t": best[0], "k": best[1]}


def cross_validate(D, X, Y):
    """Out-of-fold predictions and macro-F1 for each family; folds stratified by bin."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    strata = bin_index(D)
    oof = {f: np.full(len(D["gold"]), -1) for f in ("S1", "S2", "S3")}
    chosen = {f: np.full(len(D["gold"]), -1) for f in ("S2", "S3")}
    folds = []
    for tr, te in skf.split(X, strata):
        p1 = fit_s1(D, tr)
        oof["S1"][te] = rule3(D, p1["t"], p1["k"])[te]
        folds.append(p1)
        for f in ("S2", "S3"):
            m = Learned(f).fit(X[tr], Y[tr])
            pred, j = choose(m.proba(X[te]), {"enc": D["enc"][te], "zs": D["zs"][te], "rag": D["rag"][te]})
            oof[f][te], chosen[f][te] = pred, j
    cv = {f: round(f1_of(D, oof[f]), 4) for f in oof}
    cv_bins = {f: {b: round(macro_f1(D["gold"], oof[f], D["bins"][b]), 4) for b in BINS} for f in oof}
    pred = oof["S1"]
    chosen["S1"] = np.where(pred == D["rag"], 2, np.where(pred == D["zs"], 1, 0))  # by answer identity
    return cv, cv_bins, oof, folds, chosen


# ---------------------------------------------------------------- freeze
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=str(SPLITS.parent)).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def fit(llm):
    D = load_coders("2024", llm)
    X, Y = features(D), targets(D)
    n = len(D["gold"])
    singles = {a: round(f1_of(D, D[a]), 4) for a in ARMS}
    best_single = max(singles, key=singles.get)
    oracle = round(f1_of(D, np.where(Y.any(axis=1), D["gold"], D["enc"])), 4)
    cv, cv_bins, oof, folds, chosen = cross_validate(D, X, Y)
    family = max(cv, key=cv.get)
    print(f"2024 n={n}  singles {singles}  oracle {oracle}")
    print(f"CV macro-F1: {cv}  -> main family {family}")
    for f in cv_bins:
        print(f"  {f} by bin: {cv_bins[f]}")

    frozen = {"version": "maude_selector_v1", "date": str(date.today()), "git_head_at_fit": git_head(), "llm": llm,
              "encoder": f"{SELECTOR_ENC} full seed42 (rerun2024 outputs for 2024)", "encoder_tag": SELECTOR_ENC, "n_train": n,
              "features": FEATURES, "arms": list(ARMS), "cv_macro_f1": cv, "cv_macro_f1_by_bin": cv_bins,
              "cv_folds_S1": folds, "main_family": family, "singles_2024": singles, "best_single_arm_2024": best_single,
              "oracle_2024": oracle, "notes": "folds stratified by the gold code's training-count bin; ties and "
              "invalid answers fall to enc; the hand rule's (t, k) on all 2024 items is below"}
    frozen["S1"] = fit_s1(D, np.arange(n))
    models = {}
    for f in ("S2", "S3"):
        models[f] = Learned(f).fit(X, Y)
    if family == "S2":
        lr = [m.named_steps["logisticregression"] if m is not None else None for m in models["S2"].models]
        frozen["S2_coefficients"] = [None if m is None else {"coef": m.coef_[0].round(6).tolist(), "intercept": round(float(m.intercept_[0]), 6)}
                                     for m in lr]
    # main-family predictions on 2024 for the gate and llm_rare choices (out-of-fold for the learned families)
    main_oof = oof[family]
    # S-cost: gate parameters with call rate <= 50%, maximizing 2024 macro-F1 of (gated -> main selector, else enc)
    best_gate, best_f = None, -1.0
    for k in K_GRID:
        for p in P_GRID:
            g = gate(D, k, p)
            if g.mean() > 0.5:
                continue
            f = f1_of(D, np.where(g, main_oof, D["enc"]))
            if f > best_f:
                best_gate, best_f = {"k": k, "p": p, "call_rate_2024": round(float(g.mean()), 4), "macro_f1_2024_oof": round(f, 4)}, f
    frozen["S_cost"] = best_gate
    k_star = max(K_GRID, key=lambda k: f1_of(D, llm_rare(D, k)))
    frozen["llm_rare_k_star"] = {"k": k_star, "macro_f1_2024": round(f1_of(D, llm_rare(D, k_star)), 4)}
    frozen["choice_share_2024_oof"] = {a: round(float((chosen[family] == i).mean()), 4) for i, a in enumerate(ARMS)}
    fj, fb = frozen_paths(llm)
    joblib.dump({"S2": models["S2"], "S3": models["S3"], "features": FEATURES}, fb)
    frozen["joblib_sha256"] = sha256(fb)
    fj.write_text(json.dumps(frozen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"frozen -> {fj.name}, {fb.name}; S1 {frozen['S1']}; S-cost {best_gate}; llm_rare k*={k_star}")


def load_frozen(llm="opus55"):
    fj, fb = frozen_paths(llm)
    frozen = json.loads(fj.read_text(encoding="utf-8"))
    if sha256(fb) != frozen["joblib_sha256"]:
        raise SystemExit(f"{fb.name} does not match the frozen manifest")
    return frozen, joblib.load(fb)


def predict_frozen(D, frozen, models, family=None):
    """Main-selector predictions (and chosen arm index) for an evaluation set."""
    family = family or frozen["main_family"]
    if family == "S1":
        pred = rule3(D, frozen["S1"]["t"], frozen["S1"]["k"])
        j = np.where(pred == D["rag"], 2, np.where(pred == D["zs"], 1, 0))
        return pred, j
    return choose(models[family].proba(features(D)), D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "features"])
    ap.add_argument("which", nargs="?", default="2024", choices=["2024", "2025"])
    ap.add_argument("--llm", default="opus55")
    a = ap.parse_args()
    if a.cmd == "features":
        D = load_coders(a.which, a.llm)
        X = features(D)
        print(f"{a.which} {a.llm}: X {X.shape}; invalid zs {int((D['zs'] < 0).sum())}, rag {int((D['rag'] < 0).sum())}; "
              f"feature means {dict(zip(FEATURES, np.round(X.mean(axis=0), 3).tolist()))}")
        return
    fit(a.llm)


if __name__ == "__main__":
    # Run through the imported module so the pickled models reference
    # maude_selector.Learned, not __main__.Learned (the scorer must load them).
    import maude_selector as _module
    _module.main()
