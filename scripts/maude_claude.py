"""Claude arms on MAUDE through the Message Batches API (50% off, cache-aware).

Pre-registered in notes/prereg-maude-claude-2026-10-22.md. The requests are the
open models' requests (openweight_llm.stage_requests -> maude_llm.build_requests),
so prompts and item sets are identical; only the model id, the per-model
decoding settings and the output budget (maude_llm.CLAUDE_MODELS) differ. Rows
land in the JSONL shape the open-weight runner wrote, under the same paths, so
every scorer reads them unchanged. Submissions are recorded in
runs/claude/batches.json; polling appends finished results; a resubmit sends
only the items without a succeeded row (errored, expired, canceled, missing).

Cache: a batch starts its requests together, so without an existing entry most
of them write the 11.6k-token system prompt instead of reading it (pilot hit
rates 2-34%; a Fable write costs 50x a read). Each submission therefore sends a
small seed batch first and submits the rest only after the seed has ended; every
request carries a 1-hour cache marker so the entry outlives queue waits.

Usage:
  python maude_claude.py submit <model_key> <stage> [<stage> ...] [--limit N] [--pilot]
  python maude_claude.py poll                      # collect every ended batch, report usage and cost
  python maude_claude.py status                    # batch states without downloading
  stage: maude_cls | maude_novel | maude2025_cls | maude2025_rag{10k,100k,full} | maude2025_labelsfull
"""
import argparse
import json
import sys
import time
from pathlib import Path

import common  # noqa: F401  (truststore for corporate TLS)
import maude_llm
import openweight_llm
from common import RUNS
from stage0_batch import get_client

from common import CORPUS  # noqa: E402
BOOK = RUNS / "claude" / ("batches.json" if CORPUS == "maude" else f"batches_{CORPUS}.json")  # one book per corpus
PILOT_DIR = RUNS / "claude" / "pilot"
REPEAT_DIR = RUNS / "claude" / "repeat"  # a second run of a cell (run-to-run variation)
# $ per million tokens: input, output, cache read, cache write
PRICES = {
    "haiku45": (1.0, 5.0, 0.10, 2.0), "sonnet5": (2.0, 10.0, 0.20, 4.0), "haiku55": (0.10, 0.50, 0.01, 0.20),
    "opus55": (4.0, 20.0, 0.20, 8.0), "fable51": (10.0, 50.0, 0.25, 20.0), "sonnet5think": (2.0, 10.0, 0.20, 4.0),
    "sonnet5max": (2.0, 10.0, 0.20, 4.0), "opus55high": (4.0, 20.0, 0.20, 8.0),
}  # cache writes at the 1-hour rate (2x input)
BATCH_DISCOUNT = 0.5


def shape(req, mk):
    """The open models' request with this Claude model's id, decoding settings and output budget."""
    model_id, extras, max_tokens = maude_llm.CLAUDE_MODELS[mk]
    p = {k: v for k, v in req["params"].items() if k not in ("thinking", "temperature", "output_config")}
    p.update(model=model_id, max_tokens=max_tokens, **extras)
    # A 1-hour cache entry (the prompt text is unchanged): a batch may wait in the
    # queue longer than the default 5 minutes, and the entry must outlive the wait.
    p["system"] = [dict(p["system"][0], cache_control={"type": "ephemeral", "ttl": "1h"})]
    return {"custom_id": req["custom_id"], "params": p}


def out_path(stage, mk, tag):
    _, out = openweight_llm.stage_requests(stage)
    if tag == "pilot":
        return PILOT_DIR / f"{stage}_{mk}.jsonl"
    if tag == "repeat":
        return REPEAT_DIR / f"{stage}_{mk}.jsonl"
    return out.parent / out.name.format(mk=mk)


def tag_of(args):
    return "pilot" if args.pilot else "repeat" if args.repeat else None


def succeeded_ids(path):
    if not path.exists():
        return set()
    return {r["custom_id"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r["type"] == "succeeded"}


def load_book():
    return json.loads(BOOK.read_text()) if BOOK.exists() else []


def save_book(book):
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    BOOK.write_text(json.dumps(book, indent=1))


def pending(book, mk, stage, tag):
    return [b for b in book if b["model_key"] == mk and b["stage"] == stage and b.get("tag", "pilot" if b.get("pilot") else None) == tag
            and not b["collected"]]


def create_batch(client, book, entry, requests):
    """One batch, recorded before and after the call, with no SDK retry: a retried
    upload whose first response was lost would be a second, unrecorded, billed batch."""
    entry = dict(entry, batch_id=None, n=len(requests), submitted=time.strftime("%Y-%m-%dT%H:%M:%S"), collected=False)
    book.append(entry)
    save_book(book)
    batch = client.with_options(max_retries=0, timeout=1800).messages.batches.create(requests=requests)
    entry["batch_id"] = batch.id
    save_book(book)
    return batch


def submit(args):
    client = get_client()
    book = load_book()
    for stage in args.stages:
        if pending(book, args.model_key, stage, tag_of(args)) and not args.force:
            sys.exit(f"{args.model_key}/{stage}: a batch is already submitted and not collected; "
                     f"run poll first (or --force to submit anyway)")
        reqs, _ = openweight_llm.stage_requests(stage)
        if args.limit:
            reqs = reqs[:args.limit]
        out = out_path(stage, args.model_key, tag_of(args))
        done = succeeded_ids(out)
        todo = [shape(r, args.model_key) for r in reqs if r["custom_id"] not in done]
        print(f"{args.model_key}/{stage}: {len(reqs)} requests, {len(done)} already succeeded, "
              f"{len(todo)} to submit -> {out}", flush=True)
        if not todo:
            continue
        # Seed batch: a few requests write the cache entry and must END before the
        # rest is submitted (entries written by one batch are read by later batches;
        # a sync pre-warm was not: 36% hits in a warmed 100-item Fable pilot).
        entry = {"model_key": args.model_key, "stage": stage, "pilot": bool(args.pilot), "tag": tag_of(args), "out": str(out)}
        if args.seed and len(todo) > args.seed:
            seed_batch = create_batch(client, book, dict(entry, seed=True), todo[:args.seed])
            print(f"  seed batch {seed_batch.id} ({args.seed} requests); waiting for it to end", flush=True)
            while client.messages.batches.retrieve(seed_batch.id).processing_status != "ended":
                time.sleep(20)
            todo = todo[args.seed:]
        if args.chunk and len(todo) > args.chunk:
            print(f"  sending the first {args.chunk} of {len(todo)}; submit again for the rest after polling", flush=True)
            todo = todo[:args.chunk]
        batch = create_batch(client, book, entry, todo)
        print(f"  submitted batch {batch.id} ({len(todo)} requests)", flush=True)


def status(args):
    client = get_client()
    for b in load_book():
        if b["collected"] or not b.get("batch_id"):
            continue
        r = client.messages.batches.retrieve(b["batch_id"])
        print(f"{b['model_key']}/{b['stage']}{' (' + b['tag'] + ')' if b.get('tag') else (' (pilot)' if b.get('pilot') else '')} {b['batch_id']}: "
              f"{r.processing_status} {dict(r.request_counts)}", flush=True)


def row_of(res):
    row = {"custom_id": res.custom_id, "type": res.result.type}
    if res.result.type == "succeeded":
        msg = res.result.message
        u = msg.usage
        row.update(text="".join(b.text for b in msg.content if b.type == "text").strip(),
                   finish_reason="length" if msg.stop_reason == "max_tokens" else msg.stop_reason,
                   thinking_blocks=sum(1 for b in msg.content if b.type == "thinking"),
                   usage={"in": u.input_tokens, "out": u.output_tokens,
                          "cache_read": u.cache_read_input_tokens or 0, "cache_write": u.cache_creation_input_tokens or 0})
    elif res.result.type == "errored":
        row["error"] = str(res.result.error)[:300]
    return row


def cost(rows, mk):
    pin, pout, pread, pwrite = PRICES[mk]
    u = [r["usage"] for r in rows if r["type"] == "succeeded"]
    dollars = BATCH_DISCOUNT * sum(x["in"] * pin + x["out"] * pout + x["cache_read"] * pread + x["cache_write"] * pwrite
                                   for x in u) / 1e6
    hit = sum(1 for x in u if x["cache_read"] > 0) / max(1, len(u))
    out = sorted(x["out"] for x in u)
    return dollars, hit, (out[len(out) // 2] if out else 0), (out[-1] if out else 0)


def poll(args):
    client = get_client()
    book = load_book()
    all_done = True
    for b in book:
        if b["collected"]:
            continue
        if not b.get("batch_id"):
            print(f"{b['model_key']}/{b['stage']}: a submission was interrupted before its id was recorded -- "
                  f"check the batch list in the Console before resubmitting", flush=True)
            all_done = False
            continue
        r = client.messages.batches.retrieve(b["batch_id"])
        tag = f"{b['model_key']}/{b['stage']}{' (' + b['tag'] + ')' if b.get('tag') else (' (pilot)' if b.get('pilot') else '')}"
        if r.processing_status != "ended":
            print(f"{tag}: {r.processing_status} {dict(r.request_counts)}", flush=True)
            all_done = False
            continue
        out = Path(b["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        have = succeeded_ids(out)
        rows, dup = [], 0
        with out.open("a", encoding="utf-8") as fh:
            for res in client.messages.batches.results(b["batch_id"]):
                row = row_of(res)
                rows.append(row)  # every returned row counts toward the cost tally
                if row["custom_id"] in have and row["type"] == "succeeded":
                    dup += 1  # a resubmission that raced an earlier success: billed, not written
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        b["collected"] = True
        save_book(book)
        n_ok = sum(r["type"] == "succeeded" for r in rows)
        n_cut = sum(r.get("finish_reason") == "length" for r in rows)
        n_ref = sum(r.get("finish_reason") == "refusal" for r in rows)
        n_think = sum(r.get("thinking_blocks", 0) > 0 for r in rows)
        dollars, hit, out_med, out_max = cost(rows, b["model_key"])
        b.update(cost_usd=round(dollars, 3), cache_hit=round(hit, 4), succeeded=n_ok, truncated=n_cut,
                 refusals=n_ref, duplicates=dup)
        save_book(book)
        print(f"{tag}: ended; {n_ok}/{len(rows)} succeeded, {len(rows) - n_ok} not, {n_cut} truncated, {n_ref} refused, "
              f"{n_think} with thinking, {dup} duplicates; cache hit {hit:.1%}; out tokens median {out_med} max {out_max}; "
              f"cost ${dollars:.2f} -> {out.name}", flush=True)
        if hit < 0.9 and not b.get("seed"):
            print("  WARNING: cache hit rate below 90% -- stop and investigate before submitting more", flush=True)
    spent = sum(b.get("cost_usd", 0) for b in book)
    print(f"total collected cost so far: ${spent:.2f}")
    print("ALL_ENDED" if all_done else "STILL_RUNNING")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("model_key", choices=sorted(maude_llm.CLAUDE_MODELS))
    s.add_argument("stages", nargs="+", choices=[x for x in openweight_llm.STAGES if x.startswith("maude")])
    s.add_argument("--limit", type=int, help="first N requests per stage only")
    s.add_argument("--pilot", action="store_true", help="write under runs/claude/pilot")
    s.add_argument("--repeat", action="store_true", help="a second run of the cell, written under runs/claude/repeat")
    s.add_argument("--seed", type=int, default=8, help="requests in the seed batch that writes the cache first (0: none)")
    s.add_argument("--chunk", type=int, help="send at most this many requests now (check the hit rate, then submit the rest)")
    s.add_argument("--force", action="store_true", help="submit although an uncollected batch of this cell exists")
    s.set_defaults(fn=submit)
    sub.add_parser("poll").set_defaults(fn=poll)
    sub.add_parser("status").set_defaults(fn=status)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
