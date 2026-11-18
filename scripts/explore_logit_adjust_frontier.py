"""EXPLORATORY (not pre-registered; 2025 has been looked at many times): does the selector find
more rare-mode reports per false rare placement than the standard long-tail fix applied to the
encoder alone, or than a confidence cascade (the LLM answers when the encoder is unsure)?

The standard fix is post-hoc logit adjustment (Menon et al., 2021): predict argmax over codes of
log p_enc(c) - tau * log pi(c), with pi the training share of code c. tau = 0 is the encoder as
used; larger tau tilts it toward rare codes. For each tau, the 2025 failure-mode counts are
estimated exactly as in maude_downstream.py (rare reports right: exact; false rare placements:
stratified estimate; accuracy on the uniform stratum). The selector, its cost-gated variant and
the frequency rule are the points from results/maude2025nat/downstream_metrics.json.

Usage: python explore_logit_adjust_frontier.py
"""
import json

import numpy as np

from common import ROOT, RUNS
from maude_data import SPLIT_NATURAL
from maude_selector import load_coders

TAUS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def logp_matrix(D, which):
    z = np.load(RUNS / "maude_encprobs" / f"{which}.npz")
    row = {int(k): i for i, k in enumerate(z["keys"])}
    cols = {c: j for j, c in enumerate(z["classes"].tolist())}
    L = z["logp"].astype(np.float32)[[row[int(k)] for k in D["keys"]]]
    # columns in D["codes"] order; codes the encoder does not have get -inf
    M = np.full((len(D["keys"]), len(D["codes"])), -np.inf, dtype=np.float32)
    for c_idx, code in enumerate(D["codes"]):
        if code in cols:
            M[:, c_idx] = L[:, cols[code]]
    return M


def main():
    Db, Dn = load_coders("2025", "opus55"), load_coders("2025nat", "opus55")
    counts = Dn["counts"].astype(float)
    with np.errstate(divide="ignore"):
        log_prior = np.where(counts > 0, np.log(counts / counts.sum()), 0.0)
    Mb, Mn = logp_matrix(Db, "2025"), logp_matrix(Dn, "2025nat")
    nat = json.loads(SPLIT_NATURAL.read_text(encoding="utf-8"))
    pos = {k: i for i, k in enumerate(Dn["keys"])}
    A = np.array([pos[k] for k in nat["strata"]["rare_1-5"]])
    Bs = np.array([pos[k] for k in nat["strata"]["rare_6-20"]])
    U = np.array([pos[k] for k in nat["strata"]["uniform"]])
    gn, gb = Dn["gold"], Db["gold"]
    C = U[counts[gn[U]] >= 21]
    wB = nat["weights"]["rare_6-20"]
    wC = (758_343 - len(A) - nat["counts"]["rare_6-20_population"]) / len(C)
    rare = np.where((counts >= 1) & (counts <= 5))[0]
    rare_set = set(rare.tolist())
    print("tau   rare right /597  visible /46  false rare placements  precision  accuracy (uniform)")
    for tau in TAUS:
        pb = (Mb - tau * log_prior).argmax(axis=1)
        pn = (Mn - tau * log_prior).argmax(axis=1)
        rb, ra = np.isin(gb, rare), A
        tp_items = np.concatenate([gb[rb][pb[rb] == gb[rb]], gn[ra][pn[ra] == gn[ra]]])
        tp = len(tp_items)
        assigned = 0.0
        for pred, ix, w in ((pb, np.arange(len(gb)), 1.0), (pn, A, 1.0), (pn, Bs, wB), (pn, C, wC)):
            assigned += w * np.isin(pred[ix], rare).sum()
        fp = assigned - tp
        acc = float((pn[U] == gn[U]).mean())
        print(f"{tau:4.2f}  {tp:15d}  {len(set(tp_items.tolist())):11d}  {fp:21.0f}  {tp / max(tp + fp, 1e-9):9.3f}  {acc:18.4f}")
    print("\nconfidence cascade (the LLM answer replaces the encoder's when the encoder's softmax score < theta):")
    print("arm  theta  rare right /597  visible /46  false rare placements  precision  accuracy (uniform)")
    for arm in ("zs", "rag"):
        for theta in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pb = np.where((Db["enc_p"] < theta) & (Db[arm] >= 0), Db[arm], Db["enc"])
            pn = np.where((Dn["enc_p"] < theta) & (Dn[arm] >= 0), Dn[arm], Dn["enc"])
            rb = np.isin(gb, rare)
            tp_items = np.concatenate([gb[rb][pb[rb] == gb[rb]], gn[A][pn[A] == gn[A]]])
            tp = len(tp_items)
            assigned = 0.0
            for pred, ix, w in ((pb, np.arange(len(gb)), 1.0), (pn, A, 1.0), (pn, Bs, wB), (pn, C, wC)):
                assigned += w * np.isin(pred[ix], rare).sum()
            fp = assigned - tp
            acc = float((pn[U] == gn[U]).mean())
            print(f"{arm:4s} {theta:5.2f}  {tp:15d}  {len(set(tp_items.tolist())):11d}  {fp:21.0f}  {tp / max(tp + fp, 1e-9):9.3f}  {acc:18.4f}")
    d = json.loads((ROOT / "results" / "maude2025nat" / "downstream_metrics.json").read_text(encoding="utf-8"))
    nm = json.loads((ROOT / "results" / "maude2025nat" / "natural_metrics.json").read_text(encoding="utf-8"))
    print("\nselector points (from the pre-registered downstream run):")
    for k, lab in (("enc", "encoder"), ("selector", "selector v1"), ("S_cost", "cost-gated"), ("llm_rare", "frequency rule")):
        r = d["arms"][k]
        acc = nm["uniform"].get({"llm_rare": "llm_rare_k_star"}.get(k, k), {}).get("micro_f1")
        print(f"  {lab:15s} rare right {r['rare_reports_right']:4d}  visible {r['D1_visible_rare_modes']:2d}  "
              f"false {r['D2_false_rare_placements']:8.0f}  precision {r['D3_precision_rare']:.3f}  accuracy {acc}")
    _ = rare_set


if __name__ == "__main__":
    main()
