"""Shared loaders for the MAUDE stage (splits/maude_v1.json over data/maude/pool.jsonl).

Pre-registered in notes/prereg-maude-2026-10-07.md. Every arm sees the same
canonical text: the pool narrative lowercased. MAUDE narratives are filed in
uniform upper case, so case carries no information, and lowercasing cuts
roberta's tokens per narrative by 40% (median 122 -> 74).
"""
import hashlib
import json

from common import CORPUS, DATA, SPLITS

# Under MAINT_CORPUS=sdr these point at the SDR corpus (data/sdr, splits/sdr_*), built by
# sdr_data.py in the same schema; the default is MAUDE, unchanged.
POOL = DATA / CORPUS / "pool.jsonl"
POOL_2025 = DATA / CORPUS / "pool_2025.jsonl"
POOL_2026 = DATA / CORPUS / "pool_2026.jsonl"  # notes/prereg-maude-2026-2026-11-05.md
SPLIT = SPLITS / f"{CORPUS}_v1.json"
SPLIT_2025 = SPLITS / f"{CORPUS}_2025_test.json"
SPLIT_2026 = SPLITS / f"{CORPUS}_2026_test.json"
SPLIT_NATURAL = SPLITS / f"{CORPUS}_2025_natural.json"  # notes/prereg-maude-natural-2026-10-23.md
# notes/prereg-redesign-2026-11-06.md: the 2026 test year's never-seen-mode set and natural sample
SPLIT_NOVEL_2026 = SPLITS / f"{CORPUS}_2026_novel.json"
SPLIT_NATURAL_2026 = SPLITS / f"{CORPUS}_2026_natural.json"
TIERS = ("head", "body", "tail")


def load_split():
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def canonical(text):
    return text.lower()


def known_codes(split):
    """Known (non-hidden) codes, numerically sorted: the encoder's label index."""
    return sorted((c["code"] for c in split["classes"] if not c["hidden"]), key=int)


def class_info(split):
    return {c["code"]: c for c in split["classes"]}


def train_keys(split, n):
    """Training keys for subset size n (an int, or "full"), verified against the split."""
    if n != "full":
        return split["train_order_prefix"][:n]
    test = {t["key"] for t in split["test"]}
    val = set(split["val"])
    hidden = {c["code"] for c in split["classes"] if c["hidden"]}
    keys = []
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        k = int(r["report_key"])
        if r["code"] not in hidden and k not in test and k not in val:
            keys.append(k)
    digest = hashlib.sha256("\n".join(map(str, sorted(keys))).encode()).hexdigest()
    if digest != split["train_full_sha256"]:
        raise SystemExit("full training set does not match splits/maude_v1.json")
    return keys


def load_test_2025():
    """The out-of-time test items, [{key, code}] (splits/maude_2025_test.json)."""
    return json.loads(SPLIT_2025.read_text(encoding="utf-8"))["test"]


def load_test_2026():
    """The second out-of-time test (splits/maude_2026_test.json)."""
    return json.loads(SPLIT_2026.read_text(encoding="utf-8"))["test"]


def load_natural_2025():
    """The natural-distribution 2025 sample, [{key, code}] (splits/maude_2025_natural.json)."""
    return json.loads(SPLIT_NATURAL.read_text(encoding="utf-8"))["sample"]


def load_val_2024():
    """The 10,000 training-year validation reports (natural proportions; the encoder used them only to
    pick its learning rate and stopping epoch), [{key, code}] (notes/natural-selector-2026-11-09.md)."""
    split = load_split()
    want = set(split["val"])
    out = []
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        k = int(r["report_key"])
        if k in want:
            out.append({"key": k, "code": r["code"]})
    return sorted(out, key=lambda t: t["key"])


def load_novel_2026():
    """2026 reports of the hidden codes, up to ten per code (splits/maude_2026_novel.json)."""
    return json.loads(SPLIT_NOVEL_2026.read_text(encoding="utf-8"))["test"]


def load_natural_2026():
    """The natural-distribution 2026 sample, [{key, code, stratum}] (splits/maude_2026_natural.json)."""
    return json.loads(SPLIT_NATURAL_2026.read_text(encoding="utf-8"))["sample"]


def load_items(keys, pool=POOL):
    """{key: (canonical text, code)} for the requested report keys."""
    want = set(keys)
    out = {}
    for line in pool.open(encoding="utf-8"):
        r = json.loads(line)
        k = int(r["report_key"])
        if k in want:
            out[k] = (canonical(r["text"]), r["code"])
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} keys missing from the pool")
    return out
