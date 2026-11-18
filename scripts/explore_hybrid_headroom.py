"""Headroom check for a count-aware hybrid coder, offline, from saved 2025 predictions.
Exploratory only (not pre-registered): is there room above the best single arm?
"""
import json, random, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))  # run from anywhere: python scripts/explore_hybrid_headroom.py
from maude_router import load_eval, macro_f1          # noqa: E402
from maude_extra_score import load_llm, load_neighbours, read_enc_csv, OUT, BINS  # noqa: E402

ev = load_eval("2025")
gold, labels, counts = ev["gold"], ev["labels"], ev["counts"]["full"]
n = len(gold)
test_codes = set(labels.tolist())
bins = {b: np.array([c for c in labels if lo <= counts[c] <= hi]) for b, (lo, hi) in BINS.items() if b != "0"}

def f1(pred, lab=labels, sample=None):
    return macro_f1(gold, pred, lab, sample)

def row(name, pred, sample=None):
    cells = [f1(pred, labels, sample)] + [f1(pred, bins[b], sample) for b in ("1-5", "6-20", "21-89", "90+")]
    return f"{name:38s} " + " ".join(f"{c:6.3f}" for c in cells)

hdr = f"{'arm':38s} {'all':>6s} {'1-5':>6s} {'6-20':>6s} {'21-89':>6s} {'90+':>6s}"

# arms at full N
large, large_p = read_enc_csv(ev, OUT / "encoder_cb-large_full_seed42.csv")
base, base_p = ev["enc"]["full"][0]
zs = {m: ev["llm"][m] for m in ("opus55", "qwen38_27b", "gemma4_31b", "sonnet5") if m in ev["llm"]}
rag = {}
for m in ("opus55", "qwen38_27b", "gemma4_31b", "sonnet5"):
    p, st = load_llm(ev, f"llm_ragfull_{m}")
    if p is not None:
        rag[m] = p
nb_raw = json.loads((OUT / "neighbors_full.json").read_text())
idx = {c: i for i, c in enumerate(ev["codes"])}
top1cos = np.array([nb_raw[str(k)][0][2] if nb_raw[str(k)] else 0.0 for k in ev["keys"]])
# share of the 10 neighbours that carry the retrieval answer's label (agreement with the answer)
def agree_share(pred):
    out = np.zeros(n)
    for i, k in enumerate(ev["keys"]):
        lst = [idx[c] for _, c, _ in nb_raw[str(k)]]
        out[i] = (sum(1 for c in lst if c == pred[i]) / len(lst)) if lst and pred[i] >= 0 else 0.0
    return out
# label-vote share of the top neighbour label (independent of the LLM)
vote_share = np.zeros(n); vote_label = np.full(n, -1)
for i, k in enumerate(ev["keys"]):
    lst = [idx[c] for _, c, _ in nb_raw[str(k)]]
    if lst:
        vals, cnt = np.unique(lst, return_counts=True)
        vote_share[i] = cnt.max() / len(lst); vote_label[i] = vals[cnt.argmax()]

print("== 2025 test, full N; macro-F1 over all items, bin codes as labels")
print(hdr)
print(row("CB-large (seed42)", large))
print(row("CB-base (seed42)", base))
for m in zs: print(row(f"{m} zero-shot", zs[m]))
for m in rag: print(row(f"{m} + retrieval", rag[m]))

print("\n== item-level oracles (upper bounds): right if ANY listed arm is right")
def oracle(arms, fallback):
    ok = np.zeros(n, bool)
    for a in arms: ok |= (a == gold)
    return np.where(ok, gold, fallback)
print(hdr)
print(row("oracle{large, opus zs}", oracle([large, zs["opus55"]], large)))
print(row("oracle{large, opus rag}", oracle([large, rag["opus55"]], large)))
print(row("oracle{large, opus zs, opus rag}", oracle([large, zs["opus55"], rag["opus55"]], large)))
print(row("oracle{opus zs, opus rag}", oracle([zs["opus55"], rag["opus55"]], rag["opus55"])))
if "qwen38_27b" in rag:
    print(row("oracle{large, qwen zs, qwen rag}", oracle([large, zs["qwen38_27b"], rag["qwen38_27b"]], large)))
    print(row("oracle{base, qwen zs, qwen rag}", oracle([base, zs["qwen38_27b"], rag["qwen38_27b"]], base)))

print("\n== simple count-aware rules (no tuning; k = training count of the LLM's answer)")
def llm_rare(llm, enc, k):
    use = (llm >= 0) & (counts[np.maximum(llm, 0)] <= k)
    return np.where(use, llm, enc)
print(hdr)
for k in (5, 20, 89):
    print(row(f"large <- opus zs if count<={k}", llm_rare(zs["opus55"], large, k)))
for k in (5, 20, 89):
    print(row(f"large <- opus rag if count<={k}", llm_rare(rag["opus55"], large, k)))
for k in (5, 20):
    print(row(f"opus rag <- opus zs if count<={k}", llm_rare(zs["opus55"], rag["opus55"], k)))

print("\n== agreement-gated retrieval: rag answer when neighbours agree, else zero-shot (split-half tuning)")
rng = random.Random(42)
codes_sorted = sorted(test_codes)
rng.shuffle(codes_sorted)
A_codes = set(codes_sorted[: len(codes_sorted) // 2])
A = np.array([g in A_codes for g in gold]); Bm = ~A
def gated(m, thr, signal):
    use = (signal >= thr) & (rag[m] >= 0)
    return np.where(use, rag[m], np.where(zs[m] >= 0, zs[m], rag[m]))
def f1_on(pred, mask, lab=None):
    sample = np.where(mask)[0]
    lab = labels if lab is None else lab
    lab = np.array([c for c in lab if c in set(gold[sample].tolist())]) if len(lab) else lab
    return macro_f1(gold, pred, lab, sample)
for m in [x for x in ("opus55", "qwen38_27b") if x in rag]:
    for sname, signal in (("vote_share", vote_share), ("agree_share", agree_share(rag[m])), ("top1cos", top1cos)):
        grid = sorted(set(np.round(signal, 2))) if sname != "top1cos" else [round(x, 2) for x in np.arange(0.05, 0.6, 0.025)]
        res = []
        for tune, test in ((A, Bm), (Bm, A)):
            best = max(grid, key=lambda t: f1_on(gated(m, t, signal), tune))
            res.append((best, f1_on(gated(m, best, signal), test), f1_on(rag[m], test), f1_on(zs[m], test), f1_on(large, test),
                        f1_on(gated(m, best, signal), test, bins["1-5"]), f1_on(rag[m], test, bins["1-5"])))
        for (thr, g, r, z, l, g15, r15), half in zip(res, ("B", "A")):
            print(f"{m:11s} {sname:12s} thr={thr:<5} test half {half}: gated {g:.3f} | rag {r:.3f} | zs {z:.3f} | CB-large {l:.3f} || 1-5: gated {g15:.3f} rag {r15:.3f}")

print("\n== three-way, tuned on half, tested on other half: rag if agree>=t; elif zs answer rare (count<=k): zs; else CB-large")
def three_way(m, t, k, signal):
    use_rag = (signal >= t) & (rag[m] >= 0)
    use_zs = (~use_rag) & (zs[m] >= 0) & (counts[np.maximum(zs[m], 0)] <= k)
    return np.where(use_rag, rag[m], np.where(use_zs, zs[m], large))
for m in [x for x in ("opus55", "qwen38_27b") if x in rag]:
    signal = vote_share
    for tune, test, half in ((A, Bm, "B"), (Bm, A, "A")):
        best = max(((t, k) for t in sorted(set(np.round(signal, 2))) for k in (0, 5, 20, 89, 10**9)),
                   key=lambda tk: f1_on(three_way(m, tk[0], tk[1], signal), tune))
        pred = three_way(m, best[0], best[1], signal)
        print(f"{m:11s} t={best[0]:<5} k={best[1]:<10} test half {half}: 3-way {f1_on(pred, test):.3f} | rag {f1_on(rag[m], test):.3f} | CB-large {f1_on(large, test):.3f}"
              f" || 1-5 {f1_on(pred, test, bins['1-5']):.3f} vs large {f1_on(large, test, bins['1-5']):.3f} | 21-89 {f1_on(pred, test, bins['21-89']):.3f} vs large {f1_on(large, test, bins['21-89']):.3f} | 90+ {f1_on(pred, test, bins['90+']):.3f} vs large {f1_on(large, test, bins['90+']):.3f}")
