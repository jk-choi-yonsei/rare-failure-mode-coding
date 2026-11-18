"""S6a, second half (notes/prereg-sdr-2026-10-24.md, 2191e39): the selector on the leakage-free subset.

The pre-registration asked that, on the 2025 SDR test items whose gold term's first five characters are absent
from the narrative, the signs of S1 (full N) and S5 hold with intervals excluding zero. sdr_score.py scored the
S1 half (verdict "holds", 831 items); the S5 half was registered but never computed. This script computes it once.

S5 on the subset: selector (Opus coders) - the best single arm chosen on SDR 2024 (Opus + retrieval), macro-F1
over the codes present in the subset. Item-level interval: bootstrap over the subset's items with the S6a seeds
(SEED + 7 + i), as for the S1 half; code-level interval: the two-stage version (items, then codes). Both are
reported; the S1 half used the item-level interval only, and we also require the code-level one.
The pre-registered selector (encoder tag cb) is the test; the amended selector (cb-long) is descriptive.

Usage: MAINT_CORPUS=sdr MAINT_SELECTOR_ENC=cb python sdr_s6a_selector.py
       MAINT_CORPUS=sdr MAINT_SELECTOR_ENC=cb-long python sdr_s6a_selector.py
"""
import csv
import json

import numpy as np

from common import CORPUS, ROOT, SEED, rdir
from maude_claude_score import per_code_f1
from maude_router import macro_f1
from maude_selector import SELECTOR_ENC, git_head, load_coders, load_frozen, predict_frozen

B = 1000


def main():
    if CORPUS != "sdr":
        raise SystemExit("set MAINT_CORPUS=sdr")
    frozen, models = load_frozen("opus55")
    if frozen.get("encoder_tag", "cb-large") != SELECTOR_ENC:
        raise SystemExit(f"frozen selector uses {frozen.get('encoder_tag')}, MAINT_SELECTOR_ENC is {SELECTOR_ENC}")
    D = load_coders("2025", "opus55")
    with (rdir("maude2025") / "keyword_2025.csv").open(encoding="utf-8") as fh:
        leaked_by_key = {int(r["key"]): int(r["leaked"]) for r in csv.DictReader(fh)}
    leaked = np.array([leaked_by_key[int(k)] for k in D["keys"]])
    clean = np.where(leaked == 0)[0]
    gold = D["gold"]
    labels = np.array(sorted(set(gold[clean].tolist())))
    sel, _ = predict_frozen(D, frozen, models)
    best = D[frozen["best_single_arm_2024"]]

    point = macro_f1(gold, sel, labels, clean) - macro_f1(gold, best, labels, clean)
    subs = [clean[np.random.default_rng(SEED + 7 + i).integers(0, len(clean), len(clean))] for i in range(B)]
    d_item = sorted(macro_f1(gold, sel, labels, s) - macro_f1(gold, best, labels, s) for s in subs)
    rng = np.random.default_rng(SEED + 1)
    k = len(labels)
    d_code = sorted(float((per_code_f1(gold, [sel], labels, s) - per_code_f1(gold, [best], labels, s))[rng.integers(0, k, k)].mean())
                    for s in subs)
    lo, hi = round(0.025 * B) - 1, round(0.975 * B) - 1
    rep = {"prereg": "notes/prereg-sdr-2026-10-24.md (S6a, S5 half)", "git_head_at_score": git_head(),
           "encoder_tag": SELECTOR_ENC, "role": "test" if SELECTOR_ENC == "cb" else "descriptive (amended selector)",
           "best_single_arm_2024": frozen["best_single_arm_2024"], "n_leak_free": int(len(clean)),
           "codes_present": int(k), "macro_f1_leak_free": {"selector": round(macro_f1(gold, sel, labels, clean), 4),
                                                           "best_single": round(macro_f1(gold, best, labels, clean), 4)},
           "S5_leak_free": {"point": round(point, 4), "item_ci95": [round(d_item[lo], 4), round(d_item[hi], 4)],
                            "code_ci95": [round(d_code[lo], 4), round(d_code[hi], 4)]}}
    rep["holds"] = rep["S5_leak_free"]["item_ci95"][0] > 0 and rep["S5_leak_free"]["code_ci95"][0] > 0
    name = f"s6a_selector_{SELECTOR_ENC}.json"
    (rdir("maude2025") / name).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    res = ROOT / "results" / rdir("maude2025").name
    res.mkdir(parents=True, exist_ok=True)
    (res / name).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
