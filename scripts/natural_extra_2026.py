"""Descriptive rows of the paper's natural-proportion table that maude_natural_score.py does not
write: the majority vote and the frozen confidence cascade on the 2026 natural sample (Opus 5.5).

Majority vote (MV-enc, notes/prereg-majority-vote-2026-11-06.md): the code two or more valid
answers share, else the encoder's. Cascade: the arm and theta frozen in splits/redesign_v1_maude.json.
Accuracy on the uniform stratum and rare-mode reports placed under the right code in the 1-5 census.
Writes results/maude2026nat/natural_descriptive_extra.json. Usage:
  MAINT_SELECTOR_ENC=cb-large-conv python natural_extra_2026.py
"""
import json
import os

import numpy as np

os.environ.setdefault("MAINT_SELECTOR_ENC", "cb-large-conv")
import maude_selector as ms  # noqa: E402
from common import ROOT, SPLITS  # noqa: E402
from majority_vote_score import majority  # noqa: E402
from maude_data import SPLIT_NATURAL_2026  # noqa: E402


def main():
    D = ms.load_coders("2026nat", "opus55")
    nat = json.loads(SPLIT_NATURAL_2026.read_text(encoding="utf-8"))
    pos = {k: i for i, k in enumerate(D["keys"])}
    U = np.array([pos[k] for k in nat["strata"]["uniform"]])
    R = np.array([pos[k] for k in nat["strata"]["rare_1-5"]])
    cz = json.loads((SPLITS / "redesign_v1_maude.json").read_text(encoding="utf-8"))["cascade"]["opus55"]
    arms = {"majority_vote": majority(D, D["enc"]),
            "confidence_cascade": np.where((D["enc_p"] < cz["theta"]) & (D[cz["arm"]] >= 0), D[cz["arm"]], D["enc"]),
            "encoder": D["enc"]}
    g = D["gold"]
    out = {"llm": "opus55", "selector_encoder": ms.SELECTOR_ENC, "cascade": cz, "git_head": ms.git_head(),
           "arms": {k: {"accuracy_uniform": round(float((p[U] == g[U]).mean()), 4),
                        "placed_rare_1-5": int((p[R] == g[R]).sum()), "rare_1-5_n": int(len(R))} for k, p in arms.items()}}
    path = ROOT / "results" / "maude2026nat" / "natural_descriptive_extra.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["arms"]))


if __name__ == "__main__":
    main()
