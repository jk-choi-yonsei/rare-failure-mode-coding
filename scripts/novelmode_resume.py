"""Re-run only the failed requests of a Stage 2a LLM arm and merge in place.

Succeeded rows are kept as they are, so nothing is paid for twice.
Usage: python novelmode_resume.py <fmc|maintie_m1> <haiku45|sonnet5>
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from common import RUNS
from novelmode_llm import build_requests
from novelmode_sync import WORKERS, run_one
from stage0_batch import MODELS, get_client


def main():
    ds, mk = sys.argv[1], sys.argv[2]
    out = RUNS / "stage2" / f"llm_{ds}_{mk}.jsonl"
    existing = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        existing[row["custom_id"]] = row

    reqs = build_requests(ds, mk, MODELS[mk])
    todo = [r for r in reqs
            if existing.get(r["custom_id"], {}).get("type") != "succeeded"]
    print(f"{ds}/{mk}: {len(todo)} to redo, {len(existing) - len(todo)} kept")
    if not todo:
        return

    client = get_client()
    seen = set()
    for r in todo:  # warm each rotation's cached prefix before fanning out
        rot = r["custom_id"][:2]
        if rot not in seen:
            seen.add(rot)
            client.messages.create(**r["params"])
    print(f"  cache warmed for {len(seen)} rotations", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for row in ex.map(lambda r: run_one(client, r, len(todo)), todo):
            existing[row["custom_id"]] = row

    order = [r["custom_id"] for r in reqs]
    with out.open("w", encoding="utf-8") as fh:
        for cid in order:
            fh.write(json.dumps(existing[cid], ensure_ascii=False) + "\n")
    ok = sum(1 for c in order if existing[c]["type"] == "succeeded")
    print(f"wrote {out.name}: {ok}/{len(order)} succeeded")


if __name__ == "__main__":
    main()
