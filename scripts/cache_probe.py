"""Does prompt caching actually engage for the Stage 1b prefix?

Sends three synchronous requests sharing one cached system prefix and
reports the cache write/read counters. Costs a few cents; settles
empirically whether the 0% cache rate observed in the batch run is a
batch-API property or a property of this prefix.
"""
import json

from common import SPLITS
from stage0_batch import get_client
from stage1b_batch import SYSTEM_TEMPLATE

task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
labels = sorted(task["class_counts"])
system = SYSTEM_TEMPLATE.format(n=len(labels), labels="\n".join(labels))

client = get_client()
sysblk = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

for i, txt in enumerate(["air conditioner thermostat not working",
                         "pump leaking oil",
                         "replace worn bucket teeth"]):
    r = client.messages.create(model="claude-haiku-4-5", max_tokens=48,
                               system=sysblk, temperature=0.0,
                               messages=[{"role": "user", "content": txt}])
    u = r.usage
    w = getattr(u, "cache_creation_input_tokens", 0) or 0
    rd = getattr(u, "cache_read_input_tokens", 0) or 0
    out = r.content[0].text.strip() if r.content else ""
    print(f"req{i}: input={u.input_tokens} cache_write={w} cache_read={rd} "
          f"pred={out[:50]!r}")
