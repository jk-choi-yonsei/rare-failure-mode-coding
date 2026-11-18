"""Stage 2a Sonnet arm via synchronous concurrent calls.

Identical requests to the cancelled batch (same model, prompt, decoding);
only the delivery channel differs, so the measurement is unchanged. Used
because the batch sat 7.5h at zero completions.

Usage: python novelmode_sync.py <fmc|maintie_m1>
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from common import RUNS
from novelmode_llm import build_requests
from stage0_batch import get_client

WORKERS = 12
MODEL_KEY = "sonnet5"
MODEL_ID = "claude-sonnet-5"

lock = threading.Lock()
done = [0]


def run_one(client, req, total):
    p = req["params"]
    for attempt in range(4):
        try:
            m = client.messages.create(**p)
            u = m.usage
            row = {
                "custom_id": req["custom_id"], "type": "succeeded",
                "text": "".join(b.text for b in m.content if b.type == "text").strip(),
                "usage": {"in": u.input_tokens, "out": u.output_tokens,
                          "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                          "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0},
            }
            break
        except Exception as e:  # rate limits / transient errors
            if attempt == 3:
                row = {"custom_id": req["custom_id"], "type": "errored",
                       "error": str(e)[:200]}
                break
            threading.Event().wait(2 ** attempt * 2)
    with lock:
        done[0] += 1
        if done[0] % 100 == 0 or done[0] == total:
            print(f"  {done[0]}/{total}", flush=True)
    return row


def main():
    ds = sys.argv[1]
    client = get_client()
    reqs = build_requests(ds, MODEL_KEY, MODEL_ID)
    total = len(reqs)
    print(f"{ds}: {total} sync requests, {WORKERS} workers")

    # Warm the cache once per rotation so the concurrent fan-out reads rather
    # than each worker writing its own copy of the prefix.
    seen = set()
    for r in reqs:
        rot = r["custom_id"][:2]
        if rot in seen:
            continue
        seen.add(rot)
        client.messages.create(**r["params"])
    print(f"  cache warmed for {len(seen)} rotations", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(lambda r: run_one(client, r, total), reqs))

    out = RUNS / "stage2" / f"llm_{ds}_{MODEL_KEY}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for r in rows if r["type"] == "succeeded")
    print(f"wrote {out.name}: {ok}/{total} succeeded")


if __name__ == "__main__":
    main()
