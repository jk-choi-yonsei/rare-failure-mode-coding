"""Rare-answer check for the selector in natural proportions (notes/rare-check-2026-11-09.md).

The frozen selector (Opus 5.5, encoder cb-large-conv) was fitted on a balanced practice set, where an
answer naming a rare code is usually right. In a year of reports, rare-mode reports are about one in
1,200, and most of the selector's rare-code answers are false. The check is a second stage: when the
selector keeps an answer that names a rare code (training support 1-5), a logistic model estimates
whether it is right from evidence the selector does not read, and below a threshold the selector's
best answer naming a non-rare code is kept instead (the encoder's answer if there is none).

Evidence (one row per report whose selector answer names a rare code):
  sel_p             the selector's probability for its answer
  n_llm             how many of the two LLM coders name this code (1 or 2)
  enc_logp, enc_rank  the encoder's log-probability and log rank of the code (full softmax)
  enc_top           the encoder's log-probability of its own answer
  proto             largest TF-IDF cosine between the report and the training reports of the code
  proto_gap         proto minus the cosine of the nearest training report overall
  in_nb             share of the ten nearest training reports filed under the code
  attract           how often the LLM coders named this code for reports of other codes on the
                    training-year balanced test (per 1,000 reports)
  log_n             log(1 + training support of the code)
Development uses 2025 only: the balanced test and the natural sample give the rows; the natural
sample, weighted to the whole year as in maude_downstream.py, gives the threshold. 2026 is scored
once with the frozen model and threshold.

Usage: python rare_check.py dev      -> fits on 2025, writes splits/rare_check_v1.joblib/.json
       python rare_check.py test     -> applies the frozen check to 2026, results/rare_check/metrics_2026.json
       --v2 (step 2): adds "verify", Opus's choice between the rare code and the selector's best non-rare
       code when shown both codes' training reports (rare_verify.py; 1 rare, 0 other, 0.5 unparsed);
       files rare_check_v2.* and metrics_2026_v2.json. The rule for the threshold is unchanged.
"""
import argparse
import hashlib
import json
import os
import sys

import joblib
import numpy as np

os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")
import maude_selector as ms  # noqa: E402
from common import ROOT, SEED, SPLITS, rdir  # noqa: E402
from maude_data import (POOL, POOL_2025, POOL_2026, SPLIT_NATURAL, SPLIT_NATURAL_2026, load_items,  # noqa: E402
                        load_split, train_keys)
from maude_router import macro_f1  # noqa: E402

LLM = "opus55"
ENC_TAG = "cb-large-conv_full_seed42"
PROBS = ROOT / "runs" / "maude_encprobs" / ENC_TAG
FEATS = ["sel_p", "n_llm", "enc_logp", "enc_rank", "enc_top", "proto", "proto_gap", "in_nb", "attract", "log_n"]
FROZEN_J = SPLITS / "rare_check_v1.json"
FROZEN_B = SPLITS / "rare_check_v1.joblib"
RARE_MAX = 5
V2 = "--v2" in sys.argv
if V2:
    FEATS = FEATS + ["verify"]
    FROZEN_J = SPLITS / "rare_check_v2.json"
    FROZEN_B = SPLITS / "rare_check_v2.joblib"


def verdicts(which):
    path = ROOT / "runs" / "rare_verify" / f"{which}_opus55.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[int(r["key"])] = {"rare": 1.0, "other": 0.0}.get(r.get("verdict"), 0.5)
    return out


# ---------------------------------------------------------------- shared pieces
class Tfidf:
    """The retrieval index of maude_retrieval.py: TF-IDF on the training narratives, rows by code."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        split = load_split()
        self.keys = train_keys(split, "full")
        items = load_items(self.keys)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                                   stop_words="english", dtype=np.float32)
        self.X = self.vec.fit_transform([items[k][0] for k in self.keys])
        self.rows = {}
        for i, k in enumerate(self.keys):
            self.rows.setdefault(items[k][1], []).append(i)

    def proto(self, texts, codes):
        Q = self.vec.transform(texts)
        out = np.zeros(len(codes))
        for i, c in enumerate(codes):
            r = self.rows.get(c)
            if r:
                out[i] = float((Q[i] @ self.X[r].T).max())
        return out


def attraction(codes_idx):
    """Per code: LLM answers naming it for reports of other codes, per 1,000 training-year test reports."""
    D = ms.load_coders("2024", LLM)
    n = len(D["gold"])
    out = np.zeros(len(D["codes"]))
    for a in ("zs", "rag"):
        wrong = (D[a] >= 0) & (D[a] != D["gold"])
        np.add.at(out, D[a][wrong], 1.0)
    return 1000.0 * out / n


def selector_outputs(D, frozen, models):
    X = ms.features(D)
    P = models[frozen["main_family"]].proba(X)
    arms = ms.arms_matrix(D)
    Pm = np.where(arms >= 0, P, -1.0)
    j = Pm.argmax(axis=1)
    pick = arms[np.arange(len(j)), j]
    return P, Pm, arms, pick, j


def fallback(D, Pm, arms):
    """The selector's best answer naming a non-rare code; the encoder's answer if none."""
    c = D["counts"]
    ok = (arms >= 0) & (c[np.maximum(arms, 0)] > RARE_MAX)
    Q = np.where(ok, Pm, -2.0)
    k = Q.argmax(axis=1)
    alt = arms[np.arange(len(k)), k]
    return np.where(ok.any(axis=1), alt, D["enc"])


def encprobs(which, keys):
    z = np.load(PROBS / f"{which}.npz")
    pos = {int(k): i for i, k in enumerate(z["keys"])}
    rows = np.array([pos[k] for k in keys])
    return z["logp"][rows].astype(np.float32), list(z["classes"])


def build(which, tf, attract, frozen, models):
    """Selector outputs and the check's evidence for one set."""
    D = ms.load_coders(which, LLM)
    P, Pm, arms, pick, j = selector_outputs(D, frozen, models)
    c = D["counts"]
    rare = (pick >= 0) & (c[np.maximum(pick, 0)] <= RARE_MAX) & (c[np.maximum(pick, 0)] >= 1)
    idx = np.where(rare)[0]
    logp, classes = encprobs(which, D["keys"])
    col = {code: i for i, code in enumerate(classes)}
    codes = [D["codes"][pick[i]] for i in idx]
    pool = {"2025": POOL_2025, "2025nat": POOL_2025, "2026": POOL_2026, "2026nat": POOL_2026}[which]
    texts = load_items([D["keys"][i] for i in idx], pool=pool)
    nb = json.loads((rdir("maude" + which.replace("nat", "nat")) / "neighbors_full.json").read_text())
    F = np.zeros((len(idx), len(FEATS)))
    proto = tf.proto([texts[D["keys"][i]][0] for i in idx], codes)
    ver = verdicts(which) if V2 else None
    for r, i in enumerate(idx):
        code = codes[r]
        lp = logp[i]
        cc = col[code]
        neigh = nb[str(D["keys"][i])]
        top1 = neigh[0][2] if neigh else 0.0
        vals = [Pm[i, j[i]], int(D["zs"][i] == pick[i]) + int(D["rag"][i] == pick[i]),
                lp[cc], np.log1p(int((lp > lp[cc]).sum())), lp.max(),
                proto[r], proto[r] - top1, np.mean([cd == code for _, cd, _ in neigh]) if neigh else 0.0,
                attract[pick[i]], np.log1p(c[pick[i]])]
        if V2:
            vals.append(ver.get(int(D["keys"][i]), 0.5))
        F[r] = vals
    y = (pick[idx] == D["gold"][idx]).astype(int)
    return {"which": which, "D": D, "pick": pick, "alt": fallback(D, Pm, arms), "idx": idx, "F": F, "y": y}


def natural_weights(S, year):
    """Whole-year weights of the natural sample's reports, as in maude_downstream.py."""
    nat = json.loads((SPLIT_NATURAL if year == "2025" else SPLIT_NATURAL_2026).read_text(encoding="utf-8"))
    D = S["D"]
    pos = {k: i for i, k in enumerate(D["keys"])}
    A = np.array([pos[k] for k in nat["strata"]["rare_1-5"]])
    B = np.array([pos[k] for k in nat["strata"]["rare_6-20"]])
    U = np.array([pos[k] for k in nat["strata"]["uniform"]])
    gb = D["counts"][D["gold"]]
    C = U[gb[U] >= 21]
    n_cand = nat["counts"]["candidates"]
    wC = (n_cand - len(A) - nat["counts"]["rare_6-20_population"]) / len(C)
    w = np.zeros(len(D["keys"]))
    w[A] = 1.0
    w[B] = nat["weights"]["rare_6-20"]
    w[C] = wC
    return w, U


def year_counts(Sb, Sn, wn, final_b, final_n):
    """(correct rare placements, false rare placements, rare modes visible) over the whole year."""
    out = []
    for S, f, w in ((Sb, final_b, np.ones(len(final_b))), (Sn, final_n, wn)):
        D = S["D"]
        c = D["counts"]
        placed = (f >= 0) & (c[np.maximum(f, 0)] <= RARE_MAX) & (c[np.maximum(f, 0)] >= 1)
        right = placed & (f == D["gold"])
        out.append((float((w * right).sum()), float((w * (placed & ~right)).sum()),
                    set(D["gold"][right & (w > 0)].tolist())))
    tp = out[0][0] + out[1][0]
    fp = out[0][1] + out[1][1]
    return tp, fp, len(out[0][2] | out[1][2])


def apply_check(S, model, t):
    final = S["pick"].copy()
    if len(S["idx"]):
        p = model.predict_proba(S["F"])[:, 1]
        drop = S["idx"][p < t]
        final[drop] = S["alt"][drop]
    return final


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- development (2025)
def dev():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    frozen, models = ms.load_frozen(LLM)
    tf = Tfidf()
    att = attraction(None)
    Sb, Sn = build("2025", tf, att, frozen, models), build("2025nat", tf, att, frozen, models)
    wn, U = natural_weights(Sn, "2025")
    F = np.vstack([Sb["F"], Sn["F"]])
    y = np.concatenate([Sb["y"], Sn["y"]])
    print(f"2025 rows: balanced {len(Sb['y'])} ({Sb['y'].sum()} right), natural {len(Sn['y'])} ({Sn['y'].sum()} right)")

    def make():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    # cross-validated scores on the development rows, to choose the threshold without in-sample optimism
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(F, y):
        oof[te] = make().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
    print(f"check AUROC (5-fold, 2025 rows): {roc_auc_score(y, oof):.3f}")
    nb_ = len(Sb["y"])

    class Fixed:  # out-of-fold probabilities in place of a model, for the threshold sweep
        def __init__(self, p):
            self.p = p

        def predict_proba(self, X):
            return np.column_stack([1 - self.p, self.p])
    base = year_counts(Sb, Sn, wn, Sb["pick"], Sn["pick"])
    enc = year_counts(Sb, Sn, wn, Sb["D"]["enc"], Sn["D"]["enc"])
    rows, best = [], None
    for t in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        fb = apply_check(Sb, Fixed(oof[:nb_]), t)
        fn = apply_check(Sn, Fixed(oof[nb_:]), t)
        tp, fp, vis = year_counts(Sb, Sn, wn, fb, fn)
        prec = tp / (tp + fp) if tp + fp else 0.0
        total = float(np.isin(Sb["D"]["gold"], np.where((Sb["D"]["counts"] >= 1) & (Sb["D"]["counts"] <= RARE_MAX))[0]).sum()
                      + (wn * ((Sn["D"]["counts"][Sn["D"]["gold"]] <= RARE_MAX) & (Sn["D"]["counts"][Sn["D"]["gold"]] >= 1))).sum())
        rec = tp / total
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = float((fn[U] == Sn["D"]["gold"][U]).mean())
        rows.append({"t": float(t), "correct": round(tp, 1), "false": round(fp, 1), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4), "modes_visible": vis, "accuracy_uniform": round(acc, 4)})
        if best is None or f1 > best[1]:
            best = (float(t), f1)
    for r in rows:
        print(r)
    print("selector as is:", base, " encoder:", enc)
    model = make().fit(F, y)
    lr = model.named_steps["logisticregression"]
    out = {"note": "notes/rare-check-2026-11-09.md", "llm": LLM, "encoder": ENC_TAG, "rare_max": RARE_MAX,
           "features": FEATS, "threshold": best[0], "threshold_rule": "largest whole-year F1 of rare placements, "
           "2025 natural sample weighted to the year, out-of-fold check probabilities",
           "coef": dict(zip(FEATS, np.round(lr.coef_[0], 4).tolist())), "intercept": round(float(lr.intercept_[0]), 4),
           "dev_auroc_oof": round(float(roc_auc_score(y, oof)), 4), "dev_sweep": rows,
           "dev_selector_as_is": {"correct": base[0], "false": base[1], "modes_visible": base[2]},
           "dev_encoder": {"correct": enc[0], "false": enc[1], "modes_visible": enc[2]},
           "rows": {"balanced": int(len(Sb["y"])), "natural": int(len(Sn["y"]))}}
    joblib.dump({"model": model, "features": FEATS}, FROZEN_B)
    out["joblib_sha256"] = sha(FROZEN_B)
    FROZEN_J.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"threshold {best[0]} (F1 {best[1]:.4f}); coefficients {out['coef']}; frozen -> {FROZEN_J.name}")


# ---------------------------------------------------------------- test (2026), once
def test():
    frozen, models = ms.load_frozen(LLM)
    fj = json.loads(FROZEN_J.read_text(encoding="utf-8"))
    if sha(FROZEN_B) != fj["joblib_sha256"]:
        raise SystemExit("rare_check_v1.joblib does not match its manifest")
    model = joblib.load(FROZEN_B)["model"]
    t = fj["threshold"]
    tf = Tfidf()
    att = attraction(None)
    Sb, Sn = build("2026", tf, att, frozen, models), build("2026nat", tf, att, frozen, models)
    wn, U = natural_weights(Sn, "2026")
    fb, fn = apply_check(Sb, model, t), apply_check(Sn, model, t)
    res = {"note": "notes/rare-check-2026-11-09.md", "threshold": t, "git_head": ms.git_head()}
    for name, (b, n) in {"encoder": (Sb["D"]["enc"], Sn["D"]["enc"]), "selector": (Sb["pick"], Sn["pick"]),
                         "selector_with_check": (fb, fn)}.items():
        tp, fp, vis = year_counts(Sb, Sn, wn, b, n)
        D = Sn["D"]
        res[name] = {"correct": round(tp, 1), "false": round(fp, 1), "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                     "modes_visible": vis, "accuracy_uniform": round(float((n[U] == D["gold"][U]).mean()), 4),
                     "macro_f1_uniform": round(macro_f1(D["gold"], n, np.array(sorted(set(D["gold"][U].tolist()))), U), 4),
                     "macro_f1_balanced": round(macro_f1(Sb["D"]["gold"], b, Sb["D"]["labels"]), 4)}
        print(name, res[name])
    if V2:  # the Opus verdict alone, for reference: keep the rare answer only when Opus picks the rare code
        vb, vn = Sb["pick"].copy(), Sn["pick"].copy()
        for S, f in ((Sb, vb), (Sn, vn)):
            if len(S["idx"]):
                drop = S["idx"][S["F"][:, FEATS.index("verify")] < 1.0]
                f[drop] = S["alt"][drop]
        tp, fp, vis = year_counts(Sb, Sn, wn, vb, vn)
        res["opus_verdict_only"] = {"correct": round(tp, 1), "false": round(fp, 1),
                                    "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0, "modes_visible": vis,
                                    "accuracy_uniform": round(float((vn[U] == Sn["D"]["gold"][U]).mean()), 4)}
        print("opus_verdict_only", res["opus_verdict_only"])
    out = ROOT / "results" / "rare_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("metrics_2026_v2.json" if V2 else "metrics_2026.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "test"])
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    {"dev": dev, "test": test}[a.cmd]()
