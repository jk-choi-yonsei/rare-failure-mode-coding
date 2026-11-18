"""Score the open-weight arms next to the committed Claude and encoder arms.

The stage scorers stay untouched, so their committed outputs cannot move;
this module imports their pre-registered normalizers and novelty parser and
adds, for each open model o with outputs present (notes/prereg-openweight-
2026-10-06.md):
  FMC        tail: o - encoder_cb, o - sonnet5      head: encoder_cb - o
  MaintIE M1 tail: o - encoder_ce, o - encoder_cb, o - sonnet5
             head: encoder_ce - o
  Stage 2a   pooled novelty AUROC: o - msp, o - energy, o - sonnet5
Item bootstrap, 1000 resamples, seed 42, encoder seeds averaged inside each
resample -- the same estimator as the committed contrasts.

Usage: python openweight_score.py
"""
import csv
import json
import random
import re
from collections import Counter

from sklearn.metrics import f1_score, roc_auc_score

from common import RUNS, SEED, SPLITS, load_fmc
from novelmode_score import NOVELTY_RE, load_encoder, load_llm
from openweight_fetch import MODELS as OPEN_MODELS
from stage0_score import normalize
from stage1b_score import normalize_maintie

SEEDS = [42, 43, 44]
B = 1000
TIERS = ("head", "body", "tail")


def read_jsonl_preds(path, id_from, norm):
    preds, ok, cut = {}, 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        lab = norm(row.get("text", "")) if row["type"] == "succeeded" else None
        cut += row.get("finish_reason") == "length"
        ok += lab is not None
        preds[id_from(row["custom_id"])] = lab or "__invalid__"
    return preds, {"validity": round(ok / len(preds), 4), "n": len(preds), "truncated": cut}


def read_encoder(ds, variant):
    seeds = []
    for s in SEEDS:
        with (RUNS / "stage1" / f"encoder_{ds}_{variant}_seed{s}.csv").open(encoding="utf-8") as fh:
            seeds.append({int(r["id"]): r["pred"] for r in csv.DictReader(fh)})
    return seeds


def classification(name, gold, tier_cls, legal, arms, stage_dir, norm, contrasts_for):
    """Tier metrics for every arm, bootstrap contrasts for each open arm."""
    open_arms, meta = [], {}
    for mk in OPEN_MODELS:
        p = RUNS / stage_dir / f"{mk}_preds.jsonl"
        if not p.exists():
            continue
        preds, check = read_jsonl_preds(p, lambda c: int(c[3:]), norm)
        if len(preds) < len(gold):  # a run still in progress would score missing items as wrong
            print(f"   (skipping {mk}: {len(preds)}/{len(gold)} items, run incomplete)")
            continue
        arms[mk], meta[mk] = [preds], check
        open_arms.append(mk)
    ids = sorted(gold)

    def tier_f1(sample, arm, tier):
        yt = [gold[g] for g in sample]
        vals = [f1_score(yt, [pr.get(g, "__m__") for g in sample], labels=tier_cls[tier],
                         average="macro", zero_division=0) for pr in arms[arm]]
        return sum(vals) / len(vals)

    report = {"output_checks": meta, "arms": {}, "contrasts": {}}
    for arm, seeds in arms.items():
        accs = [sum(gold[g] == pr.get(g) for g in ids) / len(ids) for pr in seeds]
        report["arms"][arm] = {"overall_acc": round(sum(accs) / len(accs), 4),
                               **{f"{t}_macro_f1": round(tier_f1(ids, arm, t), 4) for t in TIERS}}
    rng = random.Random(SEED)
    resamples = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(B)]
    for o in open_arms:
        for a, b, tier in contrasts_for(o):
            point = tier_f1(ids, a, tier) - tier_f1(ids, b, tier)
            diffs = sorted(tier_f1(s, a, tier) - tier_f1(s, b, tier) for s in resamples)
            report["contrasts"][f"{a}_minus_{b}_{tier}"] = {
                "point": round(point, 4), "ci95": [round(diffs[24], 4), round(diffs[974], 4)],
                "excludes_zero": diffs[24] > 0 or diffs[974] < 0}
    print(f"== {name}")
    for arm, m in report["arms"].items():
        print(f"   {arm:12} acc {m['overall_acc']:.3f}  head {m['head_macro_f1']:.3f}  "
              f"tail {m['tail_macro_f1']:.3f}")
    for k, c in report["contrasts"].items():
        print(f"   {k:40} {c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
              f"{'  *' if c['excludes_zero'] else ''}")
    return report


def fmc():
    gold = {r["gid"]: r["label"] for r in load_fmc()}
    tiers = json.loads((SPLITS / "fmc_tiers.json").read_text(encoding="utf-8"))
    legal = sorted(tiers["class_counts"])
    tier_cls = {t: [c for c, tt in tiers["class_tiers"].items() if tt == t] for t in TIERS}
    arms = {f"encoder_{v}": read_encoder("fmc", v) for v in ("ce", "cb")}
    for mk in ("haiku45", "sonnet5"):
        arms[mk] = [read_jsonl_preds(RUNS / "stage0" / f"{mk}_preds.jsonl",
                                     lambda c: int(c[3:]), lambda t: normalize(t, legal))[0]]
    return classification("FMC (Stage 0 prompt)", gold, tier_cls, legal, arms, "stage0",
                          lambda t: normalize(t, legal),
                          lambda o: [(o, "encoder_cb", "tail"), (o, "sonnet5", "tail"),
                                     ("encoder_cb", o, "head")])


def maintie():
    task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
    gold = {r["idx"]: r["label"] for r in task["records"]}
    legal = sorted(task["class_counts"])
    tier_cls = {t: [c for c, tt in task["class_tiers"].items() if tt == t] for t in TIERS}
    arms = {f"encoder_{v}": read_encoder("maintie_m1", v) for v in ("ce", "cb")}
    for mk in ("haiku45", "sonnet5"):
        arms[mk] = [read_jsonl_preds(RUNS / "stage1b" / f"{mk}_preds.jsonl",
                                     lambda c: int(c[3:]),
                                     lambda t: normalize_maintie(t, legal))[0]]
    return classification("MaintIE M1 (Stage 1b prompt)", gold, tier_cls, legal, arms, "stage1b",
                          lambda t: normalize_maintie(t, legal),
                          lambda o: [(o, "encoder_ce", "tail"), (o, "encoder_cb", "tail"),
                                     (o, "sonnet5", "tail"), ("encoder_ce", o, "head")])


def complete_llm_arms(ds):
    """LLM novelty outputs whose file covers every item; a run still in
    progress would impute 50 for its missing items and distort the AUROC."""
    n_items = sum(len(load_encoder(ds, rot)) for rot in range(3))
    out = {}
    for mk in ("haiku45", "sonnet5", *OPEN_MODELS):
        scores = load_llm(ds, mk)
        if scores is None:
            continue
        if len(scores) < n_items:
            print(f"   (skipping {mk}: {len(scores)}/{n_items} items, run incomplete)")
            continue
        out[mk] = scores
    return out


def novelty(ds):
    arms = {"msp": [], "energy": []}
    llm = complete_llm_arms(ds)
    for mk in llm:
        arms[mk] = []
    y, validity = [], {mk: 0 for mk in llm}
    for rot in range(3):
        enc = load_encoder(ds, rot)
        for i in sorted(enc):
            y.append(enc[i][0])
            arms["msp"].append(enc[i][1])
            arms["energy"].append(enc[i][2])
            for mk, scores in llm.items():
                s, _, valid = scores.get((rot, i), (50.0, False, False))
                arms[mk].append(s)
                validity[mk] += valid
    n = len(y)
    auroc = {k: round(roc_auc_score(y, v), 4) for k, v in arms.items()}
    rng = random.Random(SEED)
    resamples = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]

    def boot(k):
        out = []
        for idx in resamples:
            yy = [y[i] for i in idx]
            out.append(roc_auc_score(yy, [arms[k][i] for i in idx]) if len(set(yy)) > 1 else None)
        return out

    cache, contrasts = {}, {}
    for o in (mk for mk in OPEN_MODELS if mk in arms):
        for b in ("msp", "energy", "sonnet5"):
            for k in (o, b):
                cache.setdefault(k, boot(k))
            diffs = sorted(x - z for x, z in zip(cache[o], cache[b])
                           if x is not None and z is not None)
            lo, hi = diffs[24], diffs[974]  # same quantiles as novelmode_score
            contrasts[f"{o}_minus_{b}"] = {"point": round(auroc[o] - auroc[b], 4),
                                           "ci95": [round(lo, 4), round(hi, 4)],
                                           "excludes_zero": lo > 0 or hi < 0}
    # Descriptive diagnostic, as in the Stage 2a note: does the reject signal
    # move between known and novel items at all? Parsed outputs only -- a parse
    # failure states no novelty, so its imputed 50 would only blur the means.
    diag = {}
    for mk, scores in llm.items():
        parts = {0: [], 1: []}
        for rot in range(3):
            enc = load_encoder(ds, rot)
            for i in sorted(enc):
                s, said_none, valid = scores.get((rot, i), (50.0, False, False))
                if valid:
                    parts[enc[i][0]].append((s, said_none))
        diag[mk] = {f"{name}_{part}": round(stat(parts[flag]), 3)
                    for flag, part in ((0, "known"), (1, "novel"))
                    for name, stat in (("mean_novelty", lambda v: sum(s for s, _ in v) / len(v)),
                                       ("none_rate", lambda v: sum(x for _, x in v) / len(v)))}

    print(f"== novelty {ds} (pooled n={n})")
    for k, v in auroc.items():
        print(f"   AUROC {k:12} {v:.4f}")
    for k, c in contrasts.items():
        print(f"   {k:30} {c['point']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'  *' if c['excludes_zero'] else ''}")
    for mk, d in diag.items():
        print(f"   {mk:12} NOVELTY {d['mean_novelty_known']:5.1f} -> {d['mean_novelty_novel']:5.1f}"
              f"   NONE_OF_THESE {d['none_rate_known']:.1%} -> {d['none_rate_novel']:.1%}")
    return {"n": n, "validity": {mk: round(v / n, 4) for mk, v in validity.items()},
            "auroc": auroc, "contrasts": contrasts, "diagnostic": diag}


def known_tier_auroc(ds):
    """The Stage 2a pre-registered confound check, run for every arm: AUROC of
    novel items against known items of one frequency tier at a time. If an
    arm flags rare known classes as novel, its AUROC against known-tail
    items drops below its AUROC against known-head items."""
    if ds == "fmc":
        tiers = json.loads((SPLITS / "fmc_tiers.json").read_text(encoding="utf-8"))["class_tiers"]
    else:
        tiers = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(
            encoding="utf-8"))["class_tiers"]
    llm = complete_llm_arms(ds)
    rows = []  # (tier or "novel", {arm: score})
    for rot in range(3):
        enc = load_encoder(ds, rot)
        with (RUNS / "stage2" / f"encoder_{ds}_rot{rot}_seed42.csv").open(encoding="utf-8") as fh:
            gold = {int(r["id"]): r["gold"] for r in csv.DictReader(fh)}
        for i in sorted(enc):
            group = "novel" if enc[i][0] else tiers[gold[i]]
            scores = {"msp": enc[i][1], "energy": enc[i][2]}
            scores.update({mk: s.get((rot, i), (50.0, False, False))[0] for mk, s in llm.items()})
            rows.append((group, scores))
    out = {"n": dict(Counter(g for g, _ in rows)), "auroc": {}, "minus_msp": {}}
    arms = list(rows[0][1])
    for t in TIERS:
        sub = [(1 if g == "novel" else 0, s) for g, s in rows if g in ("novel", t)]
        y = [v for v, _ in sub]
        for arm in arms:
            out["auroc"].setdefault(arm, {})[t] = round(
                roc_auc_score(y, [s[arm] for _, s in sub]), 4)
        # Paired bootstrap of (arm - MSP) on this subset, same estimator as above.
        rng = random.Random(SEED)
        res = [[rng.randrange(len(sub)) for _ in sub] for _ in range(B)]
        per = {arm: [roc_auc_score([y[i] for i in idx], [sub[i][1][arm] for i in idx])
                     for idx in res] for arm in arms}
        for arm in arms:
            if arm == "msp":
                continue
            d = sorted(a - m for a, m in zip(per[arm], per["msp"]))
            out["minus_msp"].setdefault(arm, {})[t] = {
                "point": round(out["auroc"][arm][t] - out["auroc"]["msp"][t], 4),
                "ci95": [round(d[24], 4), round(d[974], 4)],
                "excludes_zero": d[24] > 0 or d[974] < 0}
    print(f"== known-tier check {ds}  (items: {out['n']})")
    for arm in arms:
        cells = []
        for t in TIERS:
            c = out["minus_msp"].get(arm, {}).get(t)
            ci = f" ({c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" \
                 f"{'*' if c['excludes_zero'] else ''})" if c else ""
            cells.append(f"vs known-{t} {out['auroc'][arm][t]:.3f}{ci}")
        print(f"   {arm:11} " + "  ".join(cells))
    return out


def repeat_variance(ds, mk):
    """Run-to-run check (not pre-registered): the same Stage 2a requests sent
    again (openweight_llm.py --repeat). Agreement of raw answers and the
    pooled AUROC of each run."""
    p = RUNS / "openweight" / "repeat" / f"stage2_{ds}_{mk}.jsonl"
    first = load_llm(ds, mk)
    if not p.exists() or first is None:
        return None
    second, texts = {}, {}
    for line in p.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rot, i = (int(x) for x in re.match(r"r(\d)i(\d+)", row["custom_id"]).groups())
        nm = NOVELTY_RE.search(row.get("text", "")) if row["type"] == "succeeded" else None
        second[(rot, i)] = float(min(100, int(nm.group(1)))) if nm else 50.0
        texts[(rot, i)] = row.get("text", "")
    base = {}
    for line in (RUNS / "stage2" / f"llm_{ds}_{mk}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rot, i = (int(x) for x in re.match(r"r(\d)i(\d+)", row["custom_id"]).groups())
        base[(rot, i)] = row.get("text", "")
    y, s1, s2 = [], [], []
    for rot in range(3):
        enc = load_encoder(ds, rot)
        for i in sorted(enc):
            y.append(enc[i][0])
            s1.append(first[(rot, i)][0])
            s2.append(second[(rot, i)])
    same = sum(base[k] == texts[k] for k in texts)
    out = {"n": len(texts), "identical_answers": same,
           "auroc_run1": round(roc_auc_score(y, s1), 4), "auroc_run2": round(roc_auc_score(y, s2), 4),
           "auroc_mean_of_runs": round(roc_auc_score(y, [(a + b) / 2 for a, b in zip(s1, s2)]), 4)}
    print(f"== repeat {ds}/{mk}: identical {same}/{len(texts)}, AUROC run1 {out['auroc_run1']:.4f}"
          f" run2 {out['auroc_run2']:.4f} mean-of-runs {out['auroc_mean_of_runs']:.4f}")
    return out


def main():
    report = {"fmc": fmc(), "maintie_m1": maintie(),
              "novelty_fmc": novelty("fmc"), "novelty_maintie_m1": novelty("maintie_m1"),
              "known_tier_fmc": known_tier_auroc("fmc"),
              "known_tier_maintie_m1": known_tier_auroc("maintie_m1"),
              "repeat": {f"{ds}/{mk}": r for ds in ("fmc", "maintie_m1") for mk in OPEN_MODELS
                         if (r := repeat_variance(ds, mk)) is not None}}
    out = RUNS / "openweight" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
