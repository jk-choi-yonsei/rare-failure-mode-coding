"""Open-weight LLM arms on a local llama.cpp server.

The requests are the Claude arms' requests, built by the same pre-registered
builders (stage0_batch, stage1b_batch, novelmode_llm), so prompt text and
item sets are identical; only the model and the delivery channel differ.
Requests go one at a time to a single server slot with greedy decoding, and
the server's prompt cache reuses the shared label-list prefix between them.
Rows are appended as they finish, in the JSONL shape the Claude arms wrote,
so an interrupted run resumes where it stopped.

Usage:
  python openweight_llm.py <model_key> <stage> [<stage> ...] [--limit N] [--pilot | --repeat]
  stage: stage0 | stage1b | stage2_fmc | stage2_maintie_m1 | maude_cls | maude_novel
         | maude2025_cls | maude2025_rag{10k,100k,full} | maude2025_labelsfull
"""
import argparse
import json
import subprocess
import time
import urllib.request

import maude_llm
import novelmode_llm
import stage0_batch
import stage1b_batch
from common import RUNS, rdir
from openweight_fetch import LLAMA_BUILD, SERVER_EXE, model_path

PORT = 8089
URL = f"http://127.0.0.1:{PORT}"
CTX = 8192
CTX_MAUDE = 16384  # 424 IMDRF paths alone are ~7.3k tokens
SEED = 42
STAGES = ["stage0", "stage1b", "stage2_fmc", "stage2_maintie_m1", "maude_cls", "maude_novel",
          "maude2025_cls", "maude2025_rag10k", "maude2025_rag100k", "maude2025_ragfull",
          "maude2025_labelsfull",
          "maude_ragfull",  # retrieval few-shot on the known 2024 test items (selector training)
          "maude2025nat_cls", "maude2025nat_ragfull",  # the natural-distribution 2025 sample
          "maude2026_cls", "maude2026_ragfull",  # the 2026 out-of-time test (notes/prereg-maude-2026-2026-11-05.md)
          # notes/prereg-redesign-2026-11-06.md: the 2026 natural sample and the 2026 never-seen-mode set
          "maude2026nat_cls", "maude2026nat_ragfull", "maude2026novel_novel",
          # notes/natural-selector-2026-11-09.md: the training-year validation reports (natural proportions)
          "maude2024val_cls", "maude2024val_ragfull"]
# notes/selector-vs-rules-2026-11-11.md: the reports of a year with a rare code inside the encoder's top k (rare_year.py);
# a larger k holds the smaller one's reports, and all k of a year share one output file, so finished rows are skipped
STAGES += [f"maude{y}short{k}_{kind}" for y in (2025, 2026) for k in (1, 3, 5, 10) for kind in ("cls", "ragfull")]
# notes/why-it-works-plan-2026-11-16.md (experiment 3): the same reports shown the encoder's top five as candidates
STAGES += [f"maude{y}cand5_cand" for y in (2025, 2026)]

# Thinking off where the chat template allows it, matching the Claude arms;
# gpt-oss cannot turn reasoning off, so it runs at its lowest effort and its
# reasoning is kept apart from the answer text that gets scored.
TEMPLATE_KWARGS = {
    "gptoss20b": {"reasoning_effort": "low"},
    "qwen38_27b": {"enable_thinking": False},
    "gemma4_31b": {"enable_thinking": False},
    "qwen3_4b": {"enable_thinking": False},
}
# Output budget per model; None keeps the Claude arms' max_tokens. gpt-oss
# spends tokens on reasoning first: at 1024, one of 200 pilot calls looped
# in its reasoning and was cut before answering.
MAX_TOKENS = {"gptoss20b": 2048, "qwen38_27b": None, "gemma4_31b": None, "qwen3_4b": None}


def stage_requests(stage):
    """(requests, output path) for one stage, with the Claude arms' builders."""
    if stage == "stage0":
        return stage0_batch.build_requests("open", "open"), RUNS / "stage0" / "{mk}_preds.jsonl"
    if "cand" in stage:  # e.g. maude2026cand5_cand
        ts = stage.removeprefix("maude").split("_")[0]
        return maude_llm.build_requests("cand", "open", "open", test_set=ts), rdir(f"maude{ts[:4]}short") / "llm_cand5_{mk}.jsonl"
    if "short" in stage:  # e.g. maude2026short5_cls, maude2025short3_ragfull
        ts, kind = stage.removeprefix("maude").split("_")
        d = maude_llm.short_dir(ts[:4])
        if kind == "cls":
            return maude_llm.build_requests("cls", "open", "open", test_set=ts), d / "llm_cls_{mk}.jsonl"
        n = kind.removeprefix("rag")
        return maude_llm.build_requests("cls", "open", "open", test_set=ts, rag=n), d / f"llm_rag{n}_{{mk}}.jsonl"
    if stage == "stage1b":
        return stage1b_batch.build_requests("open", "open"), RUNS / "stage1b" / "{mk}_preds.jsonl"
    if stage == "maude2025_cls":
        return (maude_llm.build_requests("cls", "open", "open", test_set="2025"),
                rdir("maude2025") / "llm_cls_{mk}.jsonl")
    if stage == "maude2024val_cls":  # notes/natural-selector-2026-11-09.md
        return (maude_llm.build_requests("cls", "open", "open", test_set="2024val"),
                rdir("maude2024val") / "llm_cls_{mk}.jsonl")
    if stage.startswith("maude2024val_rag"):
        n = stage.removeprefix("maude2024val_rag")
        return (maude_llm.build_requests("cls", "open", "open", test_set="2024val", rag=n),
                rdir("maude2024val") / f"llm_rag{n}_{{mk}}.jsonl")
    if stage == "maude2026novel_novel":  # notes/prereg-redesign-2026-11-06.md (test E)
        return (maude_llm.build_requests("novel", "open", "open", test_set="2026novel"),
                rdir("maude2026") / "llm_novel_{mk}.jsonl")
    if stage == "maude2026nat_cls":
        return (maude_llm.build_requests("cls", "open", "open", test_set="2026nat"),
                rdir("maude2026nat") / "llm_cls_{mk}.jsonl")
    if stage.startswith("maude2026nat_rag"):
        n = stage.removeprefix("maude2026nat_rag")
        return (maude_llm.build_requests("cls", "open", "open", test_set="2026nat", rag=n),
                rdir("maude2026nat") / f"llm_rag{n}_{{mk}}.jsonl")
    if stage == "maude2026_cls":
        return (maude_llm.build_requests("cls", "open", "open", test_set="2026"),
                rdir("maude2026") / "llm_cls_{mk}.jsonl")
    if stage.startswith("maude2026_rag"):
        n = stage.removeprefix("maude2026_rag")
        return (maude_llm.build_requests("cls", "open", "open", test_set="2026", rag=n),
                rdir("maude2026") / f"llm_rag{n}_{{mk}}.jsonl")
    if stage == "maude2025nat_cls":
        return (maude_llm.build_requests("cls", "open", "open", test_set="2025nat"),
                rdir("maude2025nat") / "llm_cls_{mk}.jsonl")
    if stage.startswith("maude2025nat_rag"):
        n = stage.removeprefix("maude2025nat_rag")
        return (maude_llm.build_requests("cls", "open", "open", test_set="2025nat", rag=n),
                rdir("maude2025nat") / f"llm_rag{n}_{{mk}}.jsonl")
    if stage.startswith("maude2025_rag"):
        n = stage.removeprefix("maude2025_rag")
        return (maude_llm.build_requests("cls", "open", "open", test_set="2025", rag=n),
                rdir("maude2025") / f"llm_rag{n}_{{mk}}.jsonl")
    if stage.startswith("maude2025_labels"):
        n = stage.removeprefix("maude2025_labels")
        return (maude_llm.build_requests("cls", "open", "open", test_set="2025", rag=n, labels_only=True),
                rdir("maude2025") / f"llm_labels{n}_{{mk}}.jsonl")
    if stage.startswith("maude_rag"):
        n = stage.removeprefix("maude_rag")
        return (maude_llm.build_requests("cls", "open", "open", test_set="2024", rag=n),
                rdir("maude") / f"llm_rag{n}_{{mk}}.jsonl")
    if stage.startswith("maude_"):
        kind = stage.removeprefix("maude_")
        return maude_llm.build_requests(kind, "open", "open"), rdir("maude") / f"llm_{kind}_{{mk}}.jsonl"
    ds = stage.removeprefix("stage2_")
    return novelmode_llm.build_requests(ds, "open", "open"), RUNS / "stage2" / f"llm_{ds}_{{mk}}.jsonl"


def to_chat(params):
    messages = []
    if params.get("system"):
        messages.append({"role": "system",
                         "content": "".join(b["text"] for b in params["system"])})
    messages += [{"role": m["role"], "content": m["content"]} for m in params["messages"]]
    return messages


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start_server(mk, log_path, ctx=CTX):
    log = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [str(SERVER_EXE), "-m", str(model_path(mk)), "--host", "127.0.0.1",
         "--port", str(PORT), "-c", str(ctx), "-np", "1", "-ngl", "999",
         "-fa", "on", "--jinja", "--seed", str(SEED), "--no-webui"],
        stdout=log, stderr=subprocess.STDOUT,
        # the driver lowers the runner to below-normal priority; without this the server inherits it
        creationflags=getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0))
    for _ in range(600):
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited early (rc={proc.returncode}); see {log_path}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except OSError:
            pass
        time.sleep(1)
    proc.kill()
    raise SystemExit("llama-server did not become healthy in 10 minutes")


def run_one(mk, req):
    p = req["params"]
    body = {
        "messages": to_chat(p),
        "max_tokens": MAX_TOKENS[mk] or p["max_tokens"],
        "temperature": 0.0, "top_k": 1, "seed": SEED, "cache_prompt": True,
        "chat_template_kwargs": TEMPLATE_KWARGS[mk],
    }
    t0 = time.time()
    try:
        r = post("/v1/chat/completions", body)
    except Exception as e:  # a failed item is recorded, not retried silently
        return {"custom_id": req["custom_id"], "type": "errored", "error": str(e)[:300]}
    msg = r["choices"][0]["message"]
    row = {"custom_id": req["custom_id"], "type": "succeeded",
           "text": (msg.get("content") or "").strip(),
           "finish_reason": r["choices"][0].get("finish_reason"),
           "usage": {"in": r["usage"]["prompt_tokens"], "out": r["usage"]["completion_tokens"],
                     "cached": r.get("timings", {}).get("cache_n"),
                     "seconds": round(time.time() - t0, 3)}}
    if msg.get("reasoning_content"):
        row["reasoning"] = msg["reasoning_content"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key", choices=sorted(TEMPLATE_KWARGS))
    ap.add_argument("stages", nargs="+", choices=STAGES)
    ap.add_argument("--limit", type=int, help="first N requests per stage only")
    ap.add_argument("--max-new", type=int, help="stop after N new requests in this run; exit code 3 if some remain "
                                                  "(a driver restarts the server between chunks)")
    ap.add_argument("--pilot", action="store_true", help="write under runs/openweight/pilot")
    ap.add_argument("--repeat", action="store_true",
                    help="write under runs/openweight/repeat (run-to-run variance check)")
    args = ap.parse_args()
    mk = args.model_key
    side_dir = "pilot" if args.pilot else "repeat" if args.repeat else None

    jobs = []  # (stage, output path, requests still to run)
    for stage in args.stages:
        reqs, out = stage_requests(stage)
        out = RUNS / "openweight" / side_dir / f"{stage}_{mk}.jsonl" if side_dir \
            else out.parent / out.name.format(mk=mk)
        if args.limit:
            reqs = reqs[:args.limit]
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
            done = {r["custom_id"] for r in rows if r["type"] == "succeeded"}
        todo = [r for r in reqs if r["custom_id"] not in done]
        print(f"{mk}/{stage}: {len(reqs)} requests, {len(done)} already done, "
              f"{len(todo)} to run -> {out}", flush=True)
        if todo:
            jobs.append((stage, out, todo))
    if not jobs:
        return

    # One server for all stages: loading the weights is the slow part.
    # The server writes a log line per request; inside the OneDrive folder the sync client locks the growing file
    # and the server stalls for about a minute at a time (2026-11-17: 1 request in 37 took 50-235 s), so the log goes
    # to the local temp folder.
    import tempfile
    from pathlib import Path
    log_path = Path(tempfile.gettempdir()) / f"maintlog_server_{mk}.log"
    ctx = CTX_MAUDE if any(s.startswith("maude") for s, _, _ in jobs) else CTX
    proc = start_server(mk, log_path, ctx)
    errored, ran, left = 0, 0, False
    try:
        props = json.loads(urllib.request.urlopen(URL + "/props", timeout=30).read())
        print(f"server up: llama.cpp {LLAMA_BUILD}, model {model_path(mk).name}, "
              f"ctx {props.get('default_generation_settings', {}).get('n_ctx')}", flush=True)
        for stage, out, todo in jobs:
            if out.exists():  # an earlier stage of this run may have written to the same file (the shortlist stages share one)
                done = {r["custom_id"] for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())
                        if r["type"] == "succeeded"}
                todo = [r for r in todo if r["custom_id"] not in done]
            t0 = time.time()
            with out.open("a", encoding="utf-8") as fh:
                for n, req in enumerate(todo, 1):
                    if args.max_new and ran >= args.max_new:
                        left = True
                        break
                    ran += 1
                    row = run_one(mk, req)
                    errored += row["type"] == "errored"
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    if n % 100 == 0 or n == len(todo):
                        print(f"  {stage} {n}/{len(todo)}  "
                              f"{n / (time.time() - t0):.2f} req/s", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    if errored:  # a nonzero exit marks the step failed; rerunning retries only these rows
        raise SystemExit(f"{errored} requests errored; rerun to retry them")
    if left:
        print(f"stopped after {ran} new requests; more remain", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
