"""MAUDE LLM arms: request builders, label normalization and novelty parsing.

Pre-registered in notes/prereg-maude-2026-10-07.md. Two prompts over the same
424 known labels (full IMDRF paths, alphabetical, so siblings sit together):
- cls:   classification, the Stage 1b template adapted to device problems;
         run on the known test items
- novel: classification with a reject option and a 0-100 novelty score, the
         Stage 2a template adapted; run on every test item, known and novel
The open-weight runner (openweight_llm.py) and any later API runner call
build_requests, so every model receives identical prompts.

Few-shot variant (notes/prereg-maude-rag-large-2026-10-09.md): the cls prompt
unchanged, plus the K nearest labelled 2024 training reports at training size N
(maude_retrieval.py) shown in the user turn, most similar last.
"""
import json
import re

from common import CORPUS, RUNS, rdir
from maude_data import POOL_2025, class_info, load_items, load_split, load_test_2025

MAX_TOKENS = {"cls": 64, "novel": 96, "cand": 64}
RAG_EXAMPLE_CHARS = 300

# Claude arms (notes/prereg-maude-claude-2026-10-22.md), run through the Message
# Batches API by maude_claude.py with the requests built here. key -> (model id,
# per-request settings, output budget). The settings are each model's closest
# analogue of the open models' no-reasoning greedy decoding: Haiku 4.5 takes
# temperature 0 (as in Stages 0-2); Sonnet 5 and Haiku 5.5 accept thinking
# disabled; Opus 5.5 and Fable 5.1 cannot disable thinking, so they run at the
# lowest effort (no thinking blocks in a 15-item probe). The output budget is
# above the open models' 64/96 because these models sometimes restate the
# report or correct themselves before the label; the parser recovers the label.
CLAUDE_MODELS = {
    "haiku45": ("claude-haiku-4-5", {"temperature": 0.0}, 256),
    "sonnet5": ("claude-sonnet-5", {"thinking": {"type": "disabled"}}, 256),
    "haiku55": ("claude-haiku-5-5", {"thinking": {"type": "disabled"}}, 256),
    "opus55": ("claude-opus-5-5", {"output_config": {"effort": "low"}}, 1024),
    "fable51": ("claude-fable-5-1", {"output_config": {"effort": "low"}}, 1024),
    # Reasoning arms, piloted under a rule written before the pilots (the note):
    # Sonnet 5 at effort high rarely thought (3/50) and at max once ran to the
    # cap; Opus 5.5 at effort high thought on 33/50 items, briefly, so the
    # reasoning contrast is opus55high vs opus55. The Sonnet keys stay for the
    # pilot files only.
    "sonnet5think": ("claude-sonnet-5", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}, 8192),
    "sonnet5max": ("claude-sonnet-5", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}, 16384),
    "opus55high": ("claude-opus-5-5", {"output_config": {"effort": "high"}}, 16384),
}


def llm_keys():
    """Every LLM arm key the scorers look for: open-weight models, then Claude."""
    from openweight_fetch import MODELS as open_models
    return tuple(open_models) + tuple(CLAUDE_MODELS)

# The first sentence names the corpus; everything after it is identical across corpora
# (notes/prereg-sdr-2026-10-24.md keeps the SDR prompt a one-sentence variant of MAUDE's).
_DOMAIN = {
    "maude": ("medical device adverse event reports by the device problem they describe: what went wrong with the device",
              "device problems"),
    "sdr": ("aircraft service difficulty reports (maintenance discrepancy reports) by the condition of the part that "
            "failed, malfunctioned or was found defective", "part conditions"),
}[CORPUS]

SYSTEM_CLS = """You classify """ + _DOMAIN[0] + """.

Choose exactly one label from this list of {n} labels:
{labels}

Respond with exactly one label from the list above, copied verbatim, and nothing else."""

SYSTEM_NOVEL = """You classify """ + _DOMAIN[0] + """.

Choose exactly one label from this list of {n} known labels, or answer NONE_OF_THESE if the report does not belong to any of them:
{labels}

Some reports describe """ + _DOMAIN[1] + """ that are deliberately absent from this list. Say NONE_OF_THESE for those.

Respond with exactly two lines and nothing else:
LABEL: <one label from the list, or NONE_OF_THESE>
NOVELTY: <integer 0-100, the chance that this report belongs to none of the listed labels>"""

# Shown-candidates prompt (notes/why-it-works-plan-2026-11-16.md, experiment 3): the LLM sees only the encoder's top
# five codes for the report, alphabetically and without scores, plus NONE_OF_THESE, as in filter-then-rerank (Ma et al.,
# Findings of EMNLP 2023). It tests the gate's choice of an LLM answer given without the encoder's candidates.
SYSTEM_CAND = """You classify """ + _DOMAIN[0] + """.

Each request lists candidate labels for one report. Choose the one candidate that fits the report, or answer NONE_OF_THESE if none of them fits.

Respond with exactly one candidate label, copied verbatim, or NONE_OF_THESE, and nothing else."""

# \W* tolerates markdown around the field name ("**LABEL:** ..."); the LAST match
# counts, so a model that reconsiders is scored on its final answer.
LABEL_RE = re.compile(r"LABEL\W*(.+)", re.IGNORECASE)
NOVELTY_RE = re.compile(r"NOVELTY\W*(\d+)", re.IGNORECASE)


def short_dir(year):
    """Directory of a year's shortlisted reports. MAINT_SHORT_SUFFIX names the shortlist of another encoder
    (2026-11-18), so that its files stay apart from those of the paper's encoder."""
    import os
    return rdir(f"maude{year}short" + os.environ.get("MAINT_SHORT_SUFFIX", ""))


def known_labels(split):
    """{IMDRF path: code} for the known codes."""
    return {c["path"]: c["code"] for c in split["classes"] if not c["hidden"]}


def rag_message(text, neighbours, examples, info, labels_only=False):
    """User turn with the retrieved labelled reports (most similar last), then the report.
    labels_only: the same labels in the same order without the example texts (a control
    separating the label shortlist from the example reports)."""
    if labels_only:
        parts = ["Here are the labels of reports similar to the one to classify, the most similar last:", ""]
        parts += [f"Label: {info[code]['path']}" for _, code, _ in reversed(neighbours)] + [""]
    else:
        parts = ["Here are labeled reports similar to the one to classify, the most similar last:", ""]
        for key, code, _ in reversed(neighbours):
            ex = examples[key][0]
            ex = ex if len(ex) <= RAG_EXAMPLE_CHARS else ex[:RAG_EXAMPLE_CHARS].rstrip() + " ..."
            parts += [f"Report: {ex}", f"Label: {info[code]['path']}", ""]
    parts += ["Classify this report:", text]
    return "\n".join(parts)


def build_requests(kind, model_key, model_id, test_set="2024", rag=None, labels_only=False):
    """test_set "2025": the out-of-time items (classification only), same prompt.
    rag "10k" / "100k" / "full": add the retrieved examples at that training size
    (labels_only: their labels without the texts)."""
    split = load_split()
    info = class_info(split)
    labels = sorted(known_labels(split))
    system = {"cls": SYSTEM_CLS, "novel": SYSTEM_NOVEL, "cand": SYSTEM_CAND}[kind].format(
        n=len(labels), labels="\n".join(labels))
    if test_set == "2026novel":  # notes/prereg-redesign-2026-11-06.md: the 2026 known test, then the 2026 reports of the hidden codes
        if kind != "novel":
            raise ValueError("the 2026 never-seen set takes the novelty prompt")
        from maude_data import POOL_2026, load_novel_2026, load_test_2026
        test = load_test_2026() + load_novel_2026()
        texts = load_items([t["key"] for t in test], pool=POOL_2026)
    elif "cand" in test_set:  # e.g. "2026cand5": every report of the year with a rare code in the encoder's top 5
        if kind != "cand":
            raise ValueError("the shown-candidates set takes the candidates prompt")
        import numpy as np
        from maude_data import POOL_2026
        year, k = test_set[:4], int(test_set.split("cand")[1])
        tag = {"maude": "cb-large-conv_full_seed42", "sdr": "cb-conv_full_seed42"}[CORPUS]
        z = np.load(RUNS / f"{CORPUS}_encyear" / tag / f"{year}.npz")
        cls = [str(c) for c in z["classes"]]
        sel = np.nonzero(z["rare_rank"].astype(int) < k)[0]
        test = [{"key": int(z["keys"][i]), "cands": sorted(info[cls[c]]["path"] for c in z["top_idx"][i, :k])} for i in sel]
        texts = load_items([t["key"] for t in test], pool=POOL_2026 if year == "2026" else POOL_2025)
    elif "short" in test_set:  # e.g. "2026short5": reports with a rare code inside the encoder's top 5 (rare_year.py)
        if kind != "cls":
            raise ValueError("the shortlisted reports are coded with the classification prompt")
        from maude_data import POOL_2026
        year, k = test_set[:4], int(test_set.split("short")[1])
        short = json.loads((short_dir(year) / "shortlist.json").read_text(encoding="utf-8"))["items"]
        test = [t for t in short if t["rank"] < k]
        texts = load_items([t["key"] for t in test], pool=POOL_2026 if year == "2026" else POOL_2025)
    elif test_set == "2024val":  # the training-year validation reports (notes/natural-selector-2026-11-09.md)
        if kind != "cls":
            raise ValueError("the validation reports are coded with the classification prompt")
        from maude_data import load_val_2024
        test = load_val_2024()
        texts = load_items([t["key"] for t in test])
    elif test_set == "2026nat":  # the natural-distribution 2026 sample (notes/prereg-redesign-2026-11-06.md)
        if kind != "cls":
            raise ValueError("the 2026 natural sample has known codes only")
        from maude_data import POOL_2026, load_natural_2026
        test = load_natural_2026()
        texts = load_items([t["key"] for t in test], pool=POOL_2026)
    elif test_set == "2026":  # notes/prereg-maude-2026-2026-11-05.md
        from maude_data import POOL_2026, load_test_2026
        test = load_test_2026()
        texts = load_items([t["key"] for t in test], pool=POOL_2026)
    elif test_set in ("2025", "2025nat"):
        if kind != "cls":
            raise ValueError("the 2025 sets have known codes only")
        # 2025nat: the natural-distribution sample (notes/prereg-maude-natural-2026-10-23.md)
        from maude_data import load_natural_2025
        test = load_test_2025() if test_set == "2025" else load_natural_2025()
        texts = load_items([t["key"] for t in test], pool=POOL_2025)
    else:
        test = [t for t in split["test"] if kind == "novel" or not info[t["code"]]["hidden"]]
        texts = load_items([t["key"] for t in test])
    if rag:
        if kind != "cls":
            raise ValueError("retrieval prompts are classification prompts")
        # 2025: the out-of-time test; 2024: the known test items, the selector's training
        # data (notes/prereg-maude-selector-2026-10-23.md). Same prompt, same K.
        nb_dir = short_dir(test_set[:4]) if "short" in test_set else rdir(
            {"2025": "maude2025", "2025nat": "maude2025nat", "2024": "maude", "2026": "maude2026",
             "2026nat": "maude2026nat", "2024val": "maude2024val"}[test_set])
        neighbours = json.loads((nb_dir / f"neighbors_{rag}.json").read_text())
        examples = load_items({k for v in neighbours.values() for k, _, _ in v})
    reqs = []
    for t in test:
        text = texts[t["key"]][0]
        content = rag_message(text, neighbours[str(t["key"])], examples, info, labels_only) if rag else text
        if kind == "cand":
            content = "\n".join(["Candidate labels:"] + t["cands"] + ["NONE_OF_THESE", "", "Report:", text])
        params = {
            "model": model_id,
            "max_tokens": MAX_TOKENS[kind],
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
        }
        if model_key.startswith("sonnet") or model_key.startswith("opus"):
            params["thinking"] = {"type": "disabled"}
        elif model_key.startswith("haiku"):
            params["temperature"] = 0.0
        prefix = ("l" if labels_only else "r") if rag else {"cls": "c", "novel": "v", "cand": "k"}[kind]
        reqs.append({"custom_id": f"{prefix}{t['key']}", "params": params})
    return reqs


def normalize(text, legal):
    """Stage 1b's rule with IMDRF paths: full path exact -> leaf exact ->
    unique substring (longest wins) -> invalid. Returns a legal path or None.

    Final-line rule (2026-10-22, notes/prereg-maude-claude-2026-10-22.md): the
    last non-empty line is tried first, so a model that names a label, reconsiders
    and ends with another is scored on its final answer; the whole text is the
    fallback. Re-scoring every open-model file changed no prediction."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if lines:
        hit = _match(re.sub(r"^\s*label\W*", "", lines[-1], flags=re.IGNORECASE), legal)
        if hit:
            return hit
    return _match(text, legal)


def _match(text, legal):
    t = " ".join(text.strip().split()).rstrip(".").strip("\"'`*")
    low = t.lower()
    by_full = {l.lower(): l for l in legal}
    if low in by_full:
        return by_full[low]
    by_leaf = {l.split(" > ")[-1].lower(): l for l in legal}
    if low in by_leaf:
        return by_leaf[low]
    hits = [l for l in legal if l.lower() in low]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        longest = max(hits, key=len)
        if sum(1 for h in hits if len(h) == len(longest)) == 1:
            return longest
    return None


def parse_novel(text, legal):
    """(label path or "NONE_OF_THESE" or None, novelty 0-100 or None)."""
    lms, nms = LABEL_RE.findall(text), NOVELTY_RE.findall(text)
    lm = re.match(r"(.+)", lms[-1]) if lms else None
    nm = re.match(r"(\d+)", nms[-1]) if nms else None
    label = None
    if lm:
        raw = lm.group(1).strip()
        label = "NONE_OF_THESE" if "NONE_OF_THESE" in raw.upper() else normalize(raw, legal)
    novelty = float(min(100, int(nm.group(1)))) if nm else None
    return label, novelty
