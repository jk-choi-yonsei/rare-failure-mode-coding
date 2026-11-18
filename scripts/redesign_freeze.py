"""Fit and freeze the redesign's selectors and comparison rules on the training year
(notes/prereg-redesign-2026-11-06.md, section 2). Reads no test-year label.

1. The selector's encoder: MAUDE, the roberta-large CB conv run; SDR, the base or the large
   CB conv run, whichever has the higher validation macro-F1 (training-year validation set).
2. maude_selector.py fit for Opus 5.5, Qwen3.8-27B and Gemma 4 31B with that encoder
   (splits/<corpus>_selector_v1[_<llm>]_<tag>.json/.joblib).
3. The confidence cascade (arm and theta by training-year macro-F1), per LLM.
Writes splits/redesign_v1_<corpus>.json with the choices and the frozen files' hashes.

Usage: python redesign_freeze.py            (MAUDE)
       MAINT_CORPUS=sdr python redesign_freeze.py
"""
import json
import os
import subprocess
import sys

from common import CORPUS, SPLITS, rdir

LLMS = ("opus55", "qwen38_27b", "gemma4_31b")
THETAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def meta_summary(tag):
    m = json.loads((rdir("maude2025") / f"encoder_{tag}_full_seed42_meta.json").read_text())
    hist = m["lr_runs"][str(m["lr"])]["epochs"]
    best = max(hist, key=lambda h: h["val_macro_f1"])
    return {"lr": m["lr"], "val_macro_f1": m["val_macro_f1"], "schedule": m["schedule"], "epochs_run": len(hist),
            "best_epoch": best["epoch"], "best_at_last_epoch": m.get("best_at_last_epoch"),
            "val_by_epoch": [h["val_macro_f1"] for h in hist]}


def main():
    if CORPUS == "maude":
        tag, choice = "cb-large-conv", {"rule": "MAUDE: the roberta-large CB conv run (fixed)",
                                        "cb-large-conv": meta_summary("cb-large-conv")}
    else:
        cand = {t: meta_summary(t) for t in ("cb-conv", "cb-large-conv")}
        tag = max(cand, key=lambda t: (cand[t]["val_macro_f1"], t == "cb-conv"))  # a tie goes to base
        choice = {"rule": "higher validation macro-F1 of cb-conv and cb-large-conv (tie: cb-conv)", **cand}
    print(f"[{CORPUS}] selector encoder: {tag}", flush=True)
    env = dict(os.environ, MAINT_SELECTOR_ENC=tag, PYTHONIOENCODING="utf-8")
    here = os.path.dirname(os.path.abspath(__file__))
    for llm in LLMS:
        r = subprocess.run([sys.executable, "maude_selector.py", "fit", "--llm", llm], env=env, cwd=here)
        if r.returncode != 0:
            raise SystemExit(f"selector fit failed for {llm}")
    os.environ["MAINT_SELECTOR_ENC"] = tag
    import maude_selector as ms  # after the environment names the encoder
    from maude_router import macro_f1
    import numpy as np
    out = {"prereg": "notes/prereg-redesign-2026-11-06.md", "corpus": CORPUS, "selector_encoder": tag,
           "encoder_choice": choice, "git_head_at_freeze": ms.git_head(), "selectors": {}, "cascade": {}}
    for llm in LLMS:
        fj, fb = ms.frozen_paths(llm)
        fr = json.loads(fj.read_text(encoding="utf-8"))
        out["selectors"][llm] = {"file": fj.name, "sha256": ms.sha256(fj), "joblib": fb.name, "joblib_sha256": ms.sha256(fb),
                                 "main_family": fr["main_family"], "cv_macro_f1": fr["cv_macro_f1"],
                                 "best_single_arm_2024": fr["best_single_arm_2024"], "singles_2024": fr["singles_2024"],
                                 "llm_rare_k_star": fr["llm_rare_k_star"], "S_cost": fr["S_cost"], "oracle_2024": fr["oracle_2024"]}
        D = ms.load_coders("2024", llm)

        def cascade(arm, th):
            return np.where((D["enc_p"] < th) & (D[arm] >= 0), D[arm], D["enc"])
        grid = {(a, t): macro_f1(D["gold"], cascade(a, t), D["labels"]) for a in ("zs", "rag") for t in THETAS}
        arm, th = max(grid, key=grid.get)
        out["cascade"][llm] = {"arm": arm, "theta": th, "macro_f1_2024": round(grid[(arm, th)], 4)}
        print(f"  {llm}: family {fr['main_family']} cv {fr['cv_macro_f1']}; best single {fr['best_single_arm_2024']}; "
              f"k* {fr['llm_rare_k_star']['k']}; cascade {arm}@{th}", flush=True)
    path = SPLITS / f"redesign_v1_{CORPUS}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
