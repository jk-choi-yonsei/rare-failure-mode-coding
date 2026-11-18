"""Shared loaders, label rules, and tier definitions.

Pre-registered 2026-08-18. Changes after Stage 0 execution require a
journal commit explaining why.
"""
import hashlib
import json
from pathlib import Path

try:  # corporate TLS interception: use the Windows certificate store
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = ROOT / "splits"
RUNS = ROOT / "runs"

# Corpus switch (notes/prereg-sdr-2026-10-24.md): the MAUDE pipeline runs unchanged on a
# second corpus when MAINT_CORPUS names it ("sdr"). Data, split and run paths that the
# scripts spell with "maude" are redirected; the default keeps every MAUDE path as it was.
import os  # noqa: E402

CORPUS = os.environ.get("MAINT_CORPUS", "maude")


def rdir(name):
    """Run directory for a MAUDE-named stage under the active corpus:
    rdir("maude2025") -> runs/maude2025, or runs/sdr2025 when MAINT_CORPUS=sdr."""
    return RUNS / (name if CORPUS == "maude" else CORPUS + name[len("maude"):])

# ---------------------------------------------------------------- FMC-MWO2KG

FMC_DIR = DATA / "fmc_mwo2kg"
FMC_FILES = ["train.txt", "dev.txt", "test.txt"]  # pooled in this order


def load_fmc():
    """Pooled FMC records: [{gid, origin, text, label, sha1}]. 632 rows."""
    rows = []
    gid = 0
    for name in FMC_FILES:
        for line in (FMC_DIR / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            text, label = line.rsplit(",", 1)
            rows.append({
                "gid": gid,
                "origin": name.split(".")[0],
                "text": text.strip(),
                "label": label.strip(),
                "sha1": hashlib.sha1(line.encode("utf-8")).hexdigest()[:12],
            })
            gid += 1
    return rows


# FMC tier thresholds on pooled class counts (pre-registered):
FMC_HEAD_MIN = 50   # pooled count >= 50 -> head
FMC_TAIL_MAX = 14   # pooled count < 15 -> tail; else body

# MaintIE tier thresholds on pooled class counts (pre-registered):
MAINTIE_HEAD_MIN = 30
MAINTIE_TAIL_MAX = 4


def tier_of(count, head_min, tail_max):
    if count >= head_min:
        return "head"
    if count <= tail_max:
        return "tail"
    return "body"


# ----------------------------------------------------------------- MaintIE

MAINTIE_GOLD = DATA / "maintie" / "gold_release.json"


def load_maintie_gold():
    return json.loads(MAINTIE_GOLD.read_text(encoding="utf-8"))


def maintie_label_m1(rec):
    """M1 (main task): fine-grained type of the FIRST-mentioned PhysicalObject.

    Rationale (2026-08-18): coverage 1,046/1,076, 116 classes, healthy
    long tail. Honest naming: maintenance-subject (component/entity)
    classification, not failure-mode classification.
    """
    for e in sorted(rec["entities"], key=lambda e: e["start"]):
        if e["type"].startswith("PhysicalObject"):
            return e["type"]
    return None


def maintie_label_m2(rec):
    """M2 (secondary task): fine-grained type of the PhysicalObject linked
    via a hasParticipant* relation to an Undesirable State/Process — the
    failing component. Coverage 505/1,076, 94 classes."""
    ents = rec["entities"]

    def is_undesirable(t):
        return t.startswith("State/UndesirableState") or t.startswith(
            "Process/UndesirableProcess")

    for rel in rec.get("relations", []):
        if not rel["type"].startswith("hasParticipant"):
            continue
        h, t = ents[rel["head"]], ents[rel["tail"]]
        if is_undesirable(h["type"]) and t["type"].startswith("PhysicalObject"):
            return t["type"]
        if is_undesirable(t["type"]) and h["type"].startswith("PhysicalObject"):
            return h["type"]
    return None


# ------------------------------------------------------- novel-mode rotations

def novel_rotations(class_counts, n_rotations=3, k=4):
    """Deterministic held-out class sets. Classes ranked by pooled count
    desc, ties broken alphabetically. Rotation r holds out ranks (1-indexed)
    [3+r, ceil(0.3C)+r, ceil(0.6C)+r, ceil(0.85C)+r] — a head/body/tail mix
    that never touches the top-2 classes."""
    import math
    ranked = sorted(class_counts, key=lambda c: (-class_counts[c], c))
    C = len(ranked)
    rotations = []
    for r in range(n_rotations):
        pos = [3 + r, math.ceil(0.30 * C) + r, math.ceil(0.60 * C) + r,
               math.ceil(0.85 * C) + r]
        pos = sorted(set(min(p, C) for p in pos))[:k]
        rotations.append([ranked[p - 1] for p in pos])
    return rotations
