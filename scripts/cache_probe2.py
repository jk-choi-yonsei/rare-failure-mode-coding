"""Locate the cache threshold: same prefix on Sonnet, and a padded prefix on Haiku."""
import json

from common import SPLITS
from stage0_batch import get_client
from stage1b_batch import SYSTEM_TEMPLATE

task = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
labels = sorted(task["class_counts"])
system = SYSTEM_TEMPLATE.format(n=len(labels), labels="\n".join(labels))
client = get_client()


def probe(model, sys_text, tag):
    blk = [{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}]
    for i in range(2):
        r = client.messages.create(model=model, max_tokens=16, system=blk,
                                   messages=[{"role": "user", "content": "pump leaking oil"}])
        u = r.usage
        print(f"{tag} req{i}: input={u.input_tokens} "
              f"write={getattr(u, 'cache_creation_input_tokens', 0) or 0} "
              f"read={getattr(u, 'cache_read_input_tokens', 0) or 0}")


probe("claude-sonnet-5", system, "sonnet5/2.1k-prefix")
padded = system + "\n\nReference notes (ontology guidance):\n" + "\n".join(
    f"- {l} denotes the class {l.split('/')[-1]}, a subtype of {l.split('/')[-2] if '/' in l else 'the root'}."
    for l in labels)
probe("claude-haiku-4-5", padded, "haiku45/padded")
