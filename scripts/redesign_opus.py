"""Opus 5.5 batches of the redesign (notes/prereg-redesign-2026-11-06.md, section 6), run detached.

Stages in the registered priority order: SDR 2026 zero-shot and retrieval, the MAUDE 2026
never-seen-mode set (novelty prompt), the MAUDE 2026 natural sample zero-shot and retrieval.
Each submission goes through maude_claude.py (seed batch first, 1-hour cache marker). The
MAUDE stages carry an 11.6k-token label list, so a cache miss on every request would cost
about 40x a hit: their first batch is a chunk of CHUNK requests, and the rest is sent only if
that chunk read the cache on at least MIN_HIT of its requests. A stage whose hit rate falls
below MIN_HIT stops the driver (LOW_CACHE_HIT in the log). Errored or expired rows are
resubmitted, up to ROUNDS submissions per stage.

Usage: python redesign_opus.py   (log: runs/redesign_opus.log)
"""
import json
import os
import re
import subprocess
import sys
import time

from common import RUNS

PY = sys.executable
LOG = RUNS / "redesign_opus.log"
STAGES = [("sdr", "maude2026_cls", None), ("sdr", "maude2026_ragfull", None),
          ("maude", "maude2026novel_novel", 400), ("maude", "maude2026nat_cls", 400),
          ("maude", "maude2026nat_ragfull", 400)]
MIN_HIT = 0.8
ROUNDS = 4
POLL_SECONDS = 300


def log(msg):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")


def run(corpus, *args):
    env = dict(os.environ, MAINT_CORPUS=corpus, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    r = subprocess.run([PY, "maude_claude.py", *args], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=os.path.dirname(os.path.abspath(__file__)))
    out = (r.stdout or "") + (r.stderr or "")
    log(f"[{corpus}] maude_claude.py {' '.join(args)} -> rc={r.returncode}\n{out.strip()}")
    return r.returncode, out


def book(corpus):
    path = RUNS / "claude" / ("batches.json" if corpus == "maude" else f"batches_{corpus}.json")
    return json.loads(path.read_text()) if path.exists() else []


def wait_all(corpus):
    while True:
        rc, out = run(corpus, "poll")
        if rc == 0 and "ALL_ENDED" in out:
            return
        time.sleep(POLL_SECONDS)


def main():
    log("START")
    for corpus, stage, chunk in STAGES:
        for rnd in range(1, ROUNDS + 1):
            args = ["submit", "opus55", stage, "--seed", "8"] + (["--chunk", str(chunk)] if chunk else [])
            rc, out = run(corpus, *args)
            m = re.search(r"(\d+) to submit", out)
            if rc != 0 or not m:
                log(f"SUBMIT_FAILED {corpus}/{stage} round {rnd}")
                time.sleep(600)
                wait_all(corpus)
                continue
            if int(m.group(1)) == 0:
                log(f"STAGE_DONE {corpus}/{stage}")
                break
            wait_all(corpus)
            mine = [b for b in book(corpus) if b["model_key"] == "opus55" and b["stage"] == stage
                    and not b.get("seed") and b.get("tag") is None and b.get("collected")]
            hit = mine[-1].get("cache_hit", 1.0) if mine else 1.0
            log(f"ROUND_ENDED {corpus}/{stage} round {rnd}: cache hit {hit}, cost {mine[-1].get('cost_usd') if mine else None}")
            if hit < MIN_HIT and (mine[-1].get("n", 0) if mine else 0) >= 50:
                log(f"LOW_CACHE_HIT {corpus}/{stage}: {hit}; stopping before more submissions")
                sys.exit(2)
            chunk = None
        else:
            log(f"STAGE_INCOMPLETE {corpus}/{stage} after {ROUNDS} rounds")
    spent = sum(b.get("cost_usd", 0) for c in ("maude", "sdr") for b in book(c))
    log(f"ALL_DONE (all-time collected cost in both books ${spent:.2f})")


if __name__ == "__main__":
    main()
