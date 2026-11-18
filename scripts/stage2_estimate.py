"""Cost/power estimate for the novel-mode split before committing to it."""
import collections
import json

from common import SPLITS, load_fmc
from stage0_batch import get_client

client = get_client()


def report(name, task_file, rot_file, counts, label_fmt):
    rots = json.loads((SPLITS / rot_file).read_text(encoding="utf-8"))["rotations"]
    print(f"\n=== {name}: {sum(counts.values())} items, {len(counts)} classes")
    total_calls = 0
    for r, hidden in enumerate(rots):
        novel = sum(counts[c] for c in hidden)
        known = sum(counts.values()) - novel
        test_known = round(known * 0.2)
        calls = test_known + novel
        total_calls += calls
        short = [h.split("/")[-1] for h in hidden]
        print(f"  rot{r}: hide {short} -> novel={novel} items, "
              f"known-test={test_known}, LLM calls={calls}")
    # prompt size with a reduced label list (hidden classes removed)
    kept = [c for c in counts if c not in rots[0]]
    labels = "\n".join(sorted(kept))
    system = (f"Classify the maintenance text into one of {len(kept)} labels, "
              f"or answer NONE_OF_THESE if it fits none.\n\n{labels}\n\n"
              "Respond with one label or NONE_OF_THESE.")
    for m in ("claude-haiku-4-5", "claude-sonnet-5"):
        t = client.messages.count_tokens(
            model=m, system=[{"type": "text", "text": system}],
            messages=[{"role": "user", "content": "pump leaking oil"}]).input_tokens
        print(f"  prompt tokens ({m}): {t}")
    print(f"  TOTAL LLM calls per model: {total_calls}")
    return total_calls


fmc_counts = collections.Counter(r["label"] for r in load_fmc())
n_fmc = report("FMC", "fmc_task.json", "fmc_novel_rotations.json", fmc_counts, None)

mie = json.loads((SPLITS / "maintie_m1_first_object.json").read_text(encoding="utf-8"))
n_mie = report("MaintIE M1", "maintie_m1_first_object.json",
               "maintie_m1_novel_rotations.json",
               collections.Counter(mie["class_counts"]), None)

print(f"\nGRAND TOTAL calls per model: {n_fmc + n_mie}")
