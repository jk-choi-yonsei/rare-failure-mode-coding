"""Stage 2a LLM arms: zero-shot classification with an explicit reject option.

Evaluated on exactly the item set the encoder arms scored, read back from
their CSVs, so the arms are compared on identical items.

Usage:
  python novelmode_llm.py submit <fmc|maintie_m1>
  python novelmode_llm.py poll <fmc|maintie_m1>
"""
import csv
import json
import sys

from common import RUNS, SPLITS, load_maintie_gold
from common import load_fmc
from stage0_batch import MODELS, get_client

MAX_TOKENS = 96  # long ontology paths plus the NOVELTY line; 32 truncated 88% of Sonnet/MaintIE
REF_SEED = 42  # item set is seed-invariant; read it from one encoder run

SYSTEM_TEMPLATE = """You classify maintenance work order texts.

Choose exactly one label from this list of {n} known labels, or answer NONE_OF_THESE if the text does not belong to any of them:
{labels}

Some texts describe categories that are deliberately absent from this list. Say NONE_OF_THESE for those.

Respond with exactly two lines and nothing else:
LABEL: <one label from the list, or NONE_OF_THESE>
NOVELTY: <integer 0-100, the chance that this text belongs to none of the listed labels>"""


def item_texts(ds):
    if ds == "fmc":
        return {r["gid"]: r["text"] for r in load_fmc()}
    gold = load_maintie_gold()
    task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
    return {r["idx"]: gold[r["idx"]]["text"] for r in task["records"]}


def rotation_items(ds, rot):
    """(ids, known_labels) for one rotation, from the encoder run."""
    p = RUNS / "stage2" / f"encoder_{ds}_rot{rot}_seed{REF_SEED}.csv"
    with p.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads((RUNS / "stage2" /
                       f"encoder_{ds}_rot{rot}_seed{REF_SEED}_meta.json").read_text())
    hidden = set(meta["hidden"])
    if ds == "fmc":
        all_labels = sorted(json.loads(
            (SPLITS / "fmc_tiers.json").read_text(encoding="utf-8"))["class_counts"])
    else:
        all_labels = sorted(json.loads(
            (SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))["class_counts"])
    known = [l for l in all_labels if l not in hidden]
    return [int(r["id"]) for r in rows], known


def build_requests(ds, model_key, model_id):
    texts = item_texts(ds)
    reqs = []
    for rot in range(3):
        ids, known = rotation_items(ds, rot)
        system = SYSTEM_TEMPLATE.format(n=len(known), labels="\n".join(known))
        for i in ids:
            params = {
                "model": model_id,
                "max_tokens": MAX_TOKENS,
                "system": [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": texts[i]}],
            }
            if model_key == "sonnet5":
                params["thinking"] = {"type": "disabled"}
            else:
                params["temperature"] = 0.0
            reqs.append({"custom_id": f"r{rot}i{i:06d}", "params": params})
    return reqs


def submit(ds):
    client = get_client()
    out_dir = RUNS / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = {}
    for mk, mid in MODELS.items():
        reqs = build_requests(ds, mk, mid)
        batch = client.messages.batches.create(requests=reqs)
        ids[mk] = batch.id
        print(f"submitted {ds}/{mk}: {len(reqs)} requests, batch {batch.id}")
    (out_dir / f"batch_ids_{ds}.json").write_text(json.dumps(ids, indent=1))


def poll(ds):
    client = get_client()
    out_dir = RUNS / "stage2"
    ids = json.loads((out_dir / f"batch_ids_{ds}.json").read_text())
    all_done = True
    for mk, bid in ids.items():
        b = client.messages.batches.retrieve(bid)
        print(f"{ds}/{mk}: {b.processing_status} {dict(b.request_counts)}")
        if b.processing_status != "ended":
            all_done = False
            continue
        out = out_dir / f"llm_{ds}_{mk}.jsonl"
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
    {"submit": submit, "poll": poll}[sys.argv[1]](sys.argv[2])
