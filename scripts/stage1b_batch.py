"""Stage 1b: zero-shot LLM arms on MaintIE M1 via the Batch API.

Pre-registered in notes/prereg-stage1b-maintie-llm-2026-08-19.md.
Usage:
  python stage1b_batch.py submit
  python stage1b_batch.py poll
"""
import json
import sys

from common import RUNS, SPLITS, load_maintie_gold
from stage0_batch import MODELS, get_client

MAX_TOKENS = 48

SYSTEM_TEMPLATE = """You classify maintenance work order texts by the maintenance subject they refer to: the physical object the work order is about.

Choose exactly one label from this list of {n} labels:
{labels}

Respond with exactly one label from the list above, copied verbatim, and nothing else."""


def build_requests(model_key, model_id):
    gold = load_maintie_gold()
    task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
    labels = sorted(task["class_counts"])
    system = SYSTEM_TEMPLATE.format(n=len(labels), labels="\n".join(labels))
    reqs = []
    for r in task["records"]:
        params = {
            "model": model_id,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": gold[r["idx"]]["text"]}],
        }
        if model_key == "sonnet5":
            params["thinking"] = {"type": "disabled"}
        else:
            params["temperature"] = 0.0
        reqs.append({"custom_id": f"mie{r['idx']:05d}", "params": params})
    return reqs


def submit():
    client = get_client()
    out_dir = RUNS / "stage1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = {}
    for mk, mid in MODELS.items():
        reqs = build_requests(mk, mid)
        batch = client.messages.batches.create(requests=reqs)
        ids[mk] = batch.id
        print(f"submitted {mk} ({mid}): {len(reqs)} requests, batch {batch.id}")
    (out_dir / "batch_ids.json").write_text(json.dumps(ids, indent=1))


def poll():
    client = get_client()
    out_dir = RUNS / "stage1b"
    ids = json.loads((out_dir / "batch_ids.json").read_text())
    all_done = True
    for mk, bid in ids.items():
        b = client.messages.batches.retrieve(bid)
        print(f"{mk}: {b.processing_status} {dict(b.request_counts)}")
        if b.processing_status != "ended":
            all_done = False
            continue
        out = out_dir / f"{mk}_preds.jsonl"
        if out.exists():
            continue
        with out.open("w", encoding="utf-8") as fh:
            for res in client.messages.batches.results(bid):
                row = {"custom_id": res.custom_id, "type": res.result.type}
                if res.result.type == "succeeded":
                    msg = res.result.message
                    row["text"] = "".join(
                        blk.text for blk in msg.content if blk.type == "text").strip()
                    u = msg.usage
                    row["usage"] = {
                        "in": u.input_tokens, "out": u.output_tokens,
                        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                    }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {out.name}")
    print("ALL_ENDED" if all_done else "STILL_RUNNING")


if __name__ == "__main__":
    {"submit": submit, "poll": poll}[sys.argv[1]]()
