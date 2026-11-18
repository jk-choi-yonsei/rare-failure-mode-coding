"""Stage 0 LLM arm: zero-shot FMC classification via the Anthropic Batch API.

Pre-registered prompt and decoding config — fixed before execution.
Usage:
  python stage0_batch.py submit   # builds + submits one batch per model
  python stage0_batch.py poll     # checks status; downloads results when ended

Cost cap: Stage 0 must stay under $10 (estimated <$1 with Batch discount).
"""
import json
import os
import sys

from common import RUNS, SPLITS, load_fmc

MODELS = {
    "haiku45": "claude-haiku-4-5",
    "sonnet5": "claude-sonnet-5",
}
MAX_TOKENS = 24

PROMPT_TEMPLATE = """You are classifying a maintenance work order observation into ISO 14224 failure modes.

Choose exactly one label from this list:
{labels}

Observation: "{text}"

Respond with exactly one label from the list above and nothing else."""


def get_client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:  # fall back to User-scope env var set via `setx`
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")[0]
        except OSError:
            key = None
    if not key:
        sys.exit("ANTHROPIC_API_KEY not set (env or user-scope). Aborting.")
    return anthropic.Anthropic(api_key=key)


def build_requests(model_key, model_id):
    fmc = load_fmc()
    tiers = json.loads((SPLITS / "fmc_tiers.json").read_text(encoding="utf-8"))
    label_list = "\n".join(sorted(tiers["class_counts"]))
    reqs = []
    for r in fmc:
        params = {
            "model": model_id,
            "max_tokens": MAX_TOKENS,
            "messages": [{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(labels=label_list, text=r["text"]),
            }],
        }
        if model_key == "sonnet5":
            params["thinking"] = {"type": "disabled"}
        else:  # haiku 4.5: no thinking by omission; deterministic decoding
            params["temperature"] = 0.0
        reqs.append({"custom_id": f"fmc{r['gid']:04d}", "params": params})
    return reqs


def submit():
    client = get_client()
    out_dir = RUNS / "stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = {}
    for mk, mid in MODELS.items():
        batch = client.messages.batches.create(requests=build_requests(mk, mid))
        ids[mk] = batch.id
        print(f"submitted {mk} ({mid}): batch {batch.id}")
    (out_dir / "batch_ids.json").write_text(json.dumps(ids, indent=1))


def poll():
    client = get_client()
    out_dir = RUNS / "stage0"
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
                    row["usage"] = {
                        "in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {out.name}")
    print("ALL_ENDED" if all_done else "STILL_RUNNING")


if __name__ == "__main__":
    {"submit": submit, "poll": poll}[sys.argv[1]]()
