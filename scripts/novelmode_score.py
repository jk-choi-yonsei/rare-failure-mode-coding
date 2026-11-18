"""Stage 2a scoring: novel-mode detection AUROC across arms.

Pre-registered in notes/prereg-stage2-novelmode-2026-08-19.md.
Usage: python novelmode_score.py <fmc|maintie_m1>
"""
import csv
import json
import random
import re
import sys

from sklearn.metrics import average_precision_score, roc_auc_score

from common import RUNS, SEED

SEEDS = [42, 43, 44]
B = 1000
LABEL_RE = re.compile(r"LABEL:\s*(.+)", re.IGNORECASE)
NOVELTY_RE = re.compile(r"NOVELTY:\s*(\d+)", re.IGNORECASE)


def load_encoder(ds, rot):
    """{id: (is_novel, msp, energy)} averaged over seeds."""
    per_seed = []
    for s in SEEDS:
        p = RUNS / "stage2" / f"encoder_{ds}_rot{rot}_seed{s}.csv"
        with p.open(encoding="utf-8") as fh:
            per_seed.append({int(r["id"]): (int(r["is_novel"]),
                                            float(r["msp_novelty"]),
                                            float(r["energy_novelty"]))
                             for r in csv.DictReader(fh)})
    ids = sorted(per_seed[0])
    out = {}
    for i in ids:
        out[i] = (per_seed[0][i][0],
                  sum(d[i][1] for d in per_seed) / len(per_seed),
                  sum(d[i][2] for d in per_seed) / len(per_seed))
    return out


def load_llm(ds, mk):
    """{(rot, id): (novelty_score, said_none, valid)}"""
    p = RUNS / "stage2" / f"llm_{ds}_{mk}.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        m = re.match(r"r(\d)i(\d+)", row["custom_id"])
        key = (int(m.group(1)), int(m.group(2)))
        if row["type"] != "succeeded":
            out[key] = (50.0, False, False)
            continue
        text = row.get("text", "")
        nm = NOVELTY_RE.search(text)
        lm = LABEL_RE.search(text)
        said_none = bool(lm and "NONE_OF_THESE" in lm.group(1).upper())
        if nm:
            out[key] = (float(min(100, int(nm.group(1)))), said_none, True)
        else:
            out[key] = (50.0, said_none, False)
    return out


def boot_auroc(y, s, resamples):
    """AUROC per resample, kept in resample order so that two arms pair
    item-for-item. (Until 2026-10-06 this returned the values sorted, which
    paired the arms by rank instead of by resample and shrank every contrast
    interval.)"""
    vals = []
    for idx in resamples:
        yy = [y[i] for i in idx]
        vals.append(roc_auc_score(yy, [s[i] for i in idx]) if len(set(yy)) > 1 else None)
    return vals


def main():
    ds = sys.argv[1]
    rng = random.Random(SEED)
    report = {"rotations": {}, "pooled": {}}
    pooled = {"y": [], "msp": [], "energy": [], "haiku45": [], "sonnet5": []}
    llm = {mk: load_llm(ds, mk) for mk in ("haiku45", "sonnet5")}

    for rot in range(3):
        enc = load_encoder(ds, rot)
        ids = sorted(enc)
        y = [enc[i][0] for i in ids]
        arms = {"msp": [enc[i][1] for i in ids],
                "energy": [enc[i][2] for i in ids]}
        valid = {}
        for mk in ("haiku45", "sonnet5"):
            if llm[mk] is None:
                continue
            arms[mk] = [llm[mk].get((rot, i), (50.0, False, False))[0] for i in ids]
            valid[mk] = round(sum(llm[mk].get((rot, i), (0, 0, False))[2]
                                  for i in ids) / len(ids), 4)
        report["rotations"][f"rot{rot}"] = {
            "n_known": sum(1 for v in y if v == 0), "n_novel": sum(y),
            "validity": valid,
            "auroc": {k: round(roc_auc_score(y, v), 4) for k, v in arms.items()},
            "aupr_novel": {k: round(average_precision_score(y, v), 4)
                           for k, v in arms.items()},
        }
        pooled["y"].extend(y)
        for k in arms:
            pooled[k].extend(arms[k])

    n = len(pooled["y"])
    resamples = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]
    present = [k for k in ("msp", "energy", "haiku45", "sonnet5") if pooled[k]]
    report["pooled"]["n"] = n
    report["pooled"]["n_novel"] = sum(pooled["y"])
    report["pooled"]["auroc"] = {k: round(roc_auc_score(pooled["y"], pooled[k]), 4)
                                 for k in present}
    print(f"== {ds} pooled n={n} novel={sum(pooled['y'])}")
    for k in present:
        print(f"   AUROC {k:9s} {report['pooled']['auroc'][k]:.4f}")

    llms = [k for k in ("haiku45", "sonnet5") if k in present]
    if llms:
        best = max(llms, key=lambda k: report["pooled"]["auroc"][k])
        contrasts = [(best, "msp"), (best, "energy"), ("msp", "energy")]
        if len(llms) == 2:
            contrasts.append(("haiku45", "sonnet5"))
        out = {}
        for a, b in contrasts:
            point = report["pooled"]["auroc"][a] - report["pooled"]["auroc"][b]
            da = boot_auroc(pooled["y"], pooled[a], resamples)
            db = boot_auroc(pooled["y"], pooled[b], resamples)
            diffs = sorted(x - z for x, z in zip(da, db) if x is not None and z is not None)
            out[f"{a}_minus_{b}"] = {
                "point": round(point, 4),
                "ci95": [round(diffs[24], 4), round(diffs[974], 4)],
                "excludes_zero": diffs[24] > 0 or diffs[974] < 0,
            }
            print(f"   {a} - {b}: {point:+.4f} 95% CI "
                  f"[{diffs[24]:+.4f}, {diffs[974]:+.4f}]"
                  f"{'  *' if (diffs[24] > 0 or diffs[974] < 0) else '   (spans 0)'}")
        report["pooled"]["contrasts"] = out

    (RUNS / "stage2" / f"metrics_{ds}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["rotations"], indent=1))


if __name__ == "__main__":
    main()
