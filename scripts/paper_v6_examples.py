"""Table 4 of the RESS manuscript: six reports of 2026 with the answer of every coder.

One report per stream and outcome of Ours (the agreement rule, depth k = 5, either LLM coder, Qwen3.8-27B), drawn at
random over every report of the test year 2026:
  correct  Ours places the filed rare code
  false    Ours places a rare code that is not the filed code
  none     a rare report on which Ours places no rare code
The draw uses numpy's default_rng(42) (SEED of common.py), one draw per stream in the order device, aircraft and per
group in the order correct, false, none, each from the reports of that group in the order of their report key.

The narrative excerpts are the one hand-made part: EXCERPTS holds, for each drawn report, verbatim fragments of its
narrative (lower-cased and with runs of spaces collapsed, as the coders read it) that keep the words that matter for the
codes; the script checks each fragment against the pool and joins the fragments with ellipses. If the draw changes,
the script stops and prints the new narratives so that their fragments can be chosen.

Every row is checked against every source that holds it:
  data/<corpus>/pool_2026.jsonl                      narrative and filed code
  runs/<corpus>_encyear/<tag>/2026.npz               classifier ranking (first answer, rank of the filed code, top rare code)
  runs/<corpus>2026{,nat,short}/llm_{cls,ragfull}_<llm>.jsonl   LLM answers, parsed with maude_llm.normalize; when a
                                                     report has answers in several files, the later file is used, as in
                                                     rare_year.llm_answers, and any disagreement is reported
  runs/<corpus>2026short/shortlist.json              filed code, top rare code and its rank of shortlisted reports
  runs/<corpus>2026{,nat}/encoder_<tag>.csv          the classifier's answer saved for the balanced test and random sample
  results/why_it_works/<corpus>.json                 stored random false placements (placed, filed, classifier, text)
  results/rare_year/{metrics,sdr_metrics}.json       whole-year correct / false counts of Ours (Table 2)
  splits/<corpus>_v1.json + data/<corpus>/pool.jsonl training counts; the rare codes (1 to 5 training reports) must
                                                     equal the npz's rare set
Outputs (paper/RESS/figures/candidates/): tab_examples.tex (table snippet for the manuscript), tab_examples_preview.pdf
and .png (the snippet compiled in the cas-sc layout, cropped; PNG at 150 dpi), tab_examples_rows.json (every fact of
every row, the group sizes and the checks).
Usage: python paper_v6_examples.py      (CPU only; reads the 2024 and 2026 MAUDE pools, about a minute)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import maude_llm as ml  # noqa: E402  (normalize() only, which does not depend on MAINT_CORPUS)

SEED = 42
YEAR = "2026"
K = 5
LLM = "qwen38_27b"
LLMS = {"qwen38_27b": "Qwen3.8-27B", "gemma4_31b": "Gemma 4 31B"}
KINDS = {"zs": "cls", "rag": "ragfull"}
TAG = {"maude": "cb-large-conv_full_seed42", "sdr": "cb-conv_full_seed42"}
STREAMS = (("maude", "Device"), ("sdr", "Aircraft"))
GROUPS = ("correct", "false", "none")
OUT = ROOT / "paper" / "RESS" / "figures" / "candidates"
TEX_BIN = Path("C:/texlive/2026/bin/windows")
MARK = r"$^\ddagger$"  # rare code; Table 2 of the manuscript already uses the single dagger for estimated counts

# Verbatim fragments of the drawn narratives, chosen by hand after the draw (see the module docstring).
EXCERPTS = {
    ("maude", 24314269): ["two patients were observed to have radiographic stem subsidence exceeding approximately 3 mm."],
    ("maude", 50686095): ["during procedure, the oxygen supply from electronic gas blender egb was temporarely interrupted. "
                          "no error was displayed."],
    ("maude", 50048672): ["the patient experienced sound intermittencies with the implanted device since device activation.",
                          "the device was subsequently explanted"],
    ("sdr", 178297): ["on final approach the nr 2 engine experienced a flame out condition"],
    ("sdr", 181468): ["right wing nr 21 l/e v/c flap exterior surface lsa several areas found chipped.",
                      "lsa damaged areas repaired"],
    ("sdr", 202460): ["nr 1 engine auto shutdown no exceedance noted.", "no faults noted."],
}

CAPTION = ("Six reports of 2026, drawn at random for each stream: a correct placement by Ours where the classifier ranks "
           "the filed code second to fifth, a false placement, and a rare report that Ours leaves without a rare code.")
NOTE = (MARK + "Rare code. Parentheses: rank of an LLM coder's rare answer in the classifier's ranking. The LLM coders "
        "use Qwen3.8-27B. Device codes show the last level of their path.")

CHECKS = []


def check(row, what, ok, detail=""):
    CHECKS.append({"row": row, "check": what, "ok": bool(ok), "detail": detail})
    if not ok:
        print(f"CHECK FAILED [{row}] {what}: {detail}", flush=True)


def canon(text):
    """The coders' text (maude_data.canonical: lower case) with runs of white space collapsed for display."""
    return " ".join(text.lower().split())


# ------------------------------------------------------------------ data of one stream
def pool_rows(path, keys):
    want = {str(int(k)) for k in keys}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            head = line[:40]
            if not any(k in head for k in want):
                continue
            r = json.loads(line)
            if str(r["report_key"]) in want:
                out[int(r["report_key"])] = r
    return out


def training_counts(c, split):
    """Training reports of every code: the training-year pool without the held-out test and validation reports and
    without hidden codes, as maude_router.training_counts(split)["full"]."""
    held = {int(t["key"]) for t in split["test"]} | {int(k) for k in split["val"]}
    hidden = {x["code"] for x in split["classes"] if x["hidden"]}
    n = Counter()
    with open(ROOT / "data" / c / "pool.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if int(r["report_key"]) not in held and r["code"] not in hidden:
                n[r["code"]] += 1
    return n


def load_stream(c):
    z = np.load(ROOT / "runs" / f"{c}_encyear" / TAG[c] / f"{YEAR}.npz")
    cls = [str(x) for x in z["classes"]]
    c2i = {x: i for i, x in enumerate(cls)}
    split = json.loads((ROOT / "splits" / f"{c}_v1.json").read_text(encoding="utf-8"))
    p2c = {x["path"]: x["code"] for x in split["classes"] if not x["hidden"]}
    legal = sorted(p2c)
    path_of = {v: k for k, v in p2c.items()}
    keys, gold, top = z["keys"].astype(np.int64), z["gold"].astype(int), z["top_idx"].astype(int)
    N = len(keys)
    is_rare = np.zeros(len(cls), bool)
    is_rare[z["rare"].astype(int)] = True
    Y = {"c": c, "cls": cls, "c2i": c2i, "split": split, "paths": [path_of[x] for x in cls], "keys": keys, "gold": gold,
         "top": top, "is_rare": is_rare, "best": z["rare_best"].astype(int), "rank": z["rare_rank"].astype(int),
         "enc": top[:, 0], "nonrare": top[np.arange(N), (~is_rare[top]).argmax(1)], "N": N,
         "pos": {int(k): i for i, k in enumerate(keys)}, "A": {}, "raw": {}}
    for mk in LLMS:
        for name, kind in KINDS.items():
            a = np.full(N, -2)  # -2: no answer on file, -1: invalid answer
            for d in (f"{c}{YEAR}", f"{c}{YEAR}nat", f"{c}{YEAR}short"):  # the order of rare_year.llm_answers
                p = ROOT / "runs" / d / f"llm_{kind}_{mk}.jsonl"
                for line in p.read_text(encoding="utf-8").splitlines():
                    r = json.loads(line)
                    if r["type"] != "succeeded":
                        continue
                    k = int(r["custom_id"][1:])
                    code = p2c.get(ml.normalize(r.get("text", ""), legal))
                    v = c2i[code] if code else -1
                    Y["raw"].setdefault((mk, name, k), []).append({"file": f"runs/{d}/{p.name}", "text": r.get("text", ""),
                                                                   "code": cls[v] if v >= 0 else None})
                    if k in Y["pos"]:
                        a[Y["pos"][k]] = v
            Y["A"][(mk, name)] = a
    return Y


def ours(Y, mk):
    """The agreement rule of Eq. (3) with k = 5 and either LLM coder (why_it_works.gate with its defaults)."""
    fire = (Y["rank"] < K) & ((Y["A"][(mk, "zs")] == Y["best"]) | (Y["A"][(mk, "rag")] == Y["best"]))
    other = np.where(Y["is_rare"][Y["enc"]], Y["nonrare"], Y["enc"])
    return np.where(fire, Y["best"], other)


def groups_of(Y, final):
    g, r = Y["gold"], Y["is_rare"]
    return {"correct": r[final] & (final == g), "false": r[final] & (final != g), "none": r[g] & ~r[final]}


def relation(pa, pb):  # why_it_works.relation
    a, b = pa.split(" > "), pb.split(" > ")
    if a[:-1] == b[:-1]:
        return "sibling"
    if a[:len(b)] == b or b[:len(a)] == a:
        return "ancestor_or_descendant"
    if a[0] == b[0]:
        return "same_top_level"
    return "unrelated"


# ------------------------------------------------------------------ names, excerpts and LaTeX
def short_name(Y, j):
    """Device codes by the last level of their path (unique among the known codes), aircraft codes as filed."""
    return Y["paths"][j].split(" > ")[-1]


def tex(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    s = "".join(rep.get(ch, ch) for ch in s)
    return s.replace("--", "-{}-").replace("/", r"/\allowbreak{}")


def excerpt(text, frags, row):
    """The fragments joined by ellipses, with an ellipsis wherever the narrative is cut; a full stop right before an
    ellipsis is dropped, so that no cut shows four dots."""
    t = canon(text)
    at, out = 0, []
    for f in frags:
        i = t.find(f, at)
        check(row, f"excerpt fragment in the narrative: '{f[:40]}'", i >= 0, "found" if i >= 0 else "missing")
        if i < 0:
            return None, 0
        out.append((i, i + len(f)))
        at = i + len(f)
    words = sum(len(f.split()) for f in frags)
    lead = out[0][0] > 0
    trail = t[out[-1][1]:].strip() != ""
    parts = [f[:-1] if f.endswith(".") and (n < len(frags) - 1 or trail) else f for n, f in enumerate(frags)]
    body = r" \ldots{} ".join(tex(f) for f in parts)
    return (r"\ldots{} " if lead else "") + body + (r" \ldots" if trail else ""), words


def code_cell(Y, j):
    if j is None:
        return "none"
    return tex(short_name(Y, j)) + (MARK if Y["is_rare"][j] else "")


def answer_cell(Y, v, i):
    v = int(v)
    if v < 0:
        return {-2: "no answer", -1: "invalid"}[v]
    if not Y["is_rare"][v]:
        return code_cell(Y, v)
    hit = np.flatnonzero(Y["top"][i] == v)
    return code_cell(Y, v) + (f" ({hit[0] + 1})" if len(hit) else r" ($>$20)")


def table_tex(rows):
    # widths in pt (text width 468.3pt): each code column holds its longest word (Intermittently and SHUTDOWN with the
    # mark: 52.1pt; SHUTDOWN: 48.6pt; Medical Gas: 43.3pt), the excerpt takes the rest, 10pt apart from the codes
    P = r">{\raggedright\arraybackslash}p{%s}"
    spec = ("@{}" + P % "140pt" + r"@{\hspace{10pt}\extracolsep{\fill}}" + P % "49pt" + "c" + P % "52.5pt" + P % "52.5pt"
            + P % "45pt" + P % "52.5pt" + "@{}")
    L = ["% Candidate table for Section 5.3 (scripts/paper_v6_examples.py; reports drawn with numpy default_rng(42);",
         "% sources listed in the script and in tab_examples_rows.json)",
         r"\begin{table}[pos=!htbp]",
         r"\caption{" + CAPTION + "}",
         r"\label{tab:examples}",
         r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\tblwidth}{" + spec + "}",
         r"\toprule",
         r" & \multicolumn{2}{c}{Classifier} & & & & \\",
         r"\cmidrule(lr){2-3}",
         r"\makecell[lb]{Narrative (excerpt)} & \makecell[lb]{First\\answer} & \makecell[cb]{Rank of\\filed code} & "
         r"\makecell[lb]{Zero-shot\\LLM} & \makecell[lb]{Retrieval\\LLM} & \makecell[lb]{Ours} & \makecell[lb]{Filed\\code} \\",
         r"\midrule"]
    for s, (c, title) in enumerate(STREAMS):
        if s:
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{7}{@{}l}{\textit{" + title + r" reports}} \\")
        rs = [r for r in rows if r["corpus"] == c]
        for i, r in enumerate(rs):
            L.append(" & ".join([r["tex"]["excerpt"], r["tex"]["first"], r["tex"]["rank"], r["tex"]["zs"], r["tex"]["rag"],
                                 r["tex"]["ours"], r["tex"]["filed"]]) + r" \\" + (r"[6pt]" if i < len(rs) - 1 else ""))
    L += [r"\bottomrule", r"\end{tabular*}",
          r"\par\smallskip{\footnotesize\raggedright\noindent " + NOTE + r"\par}",
          r"\end{table}", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ checks against the stored results
def check_whole_year(Y, finals):
    c = Y["c"]
    M = json.loads((ROOT / "results" / "rare_year" / ("metrics.json" if c == "maude" else "sdr_metrics.json"))
                   .read_text(encoding="utf-8"))["years"][YEAR]
    check(c, "year size and rare reports as results/rare_year", M["reports"] == Y["N"] and
          M["rare_reports"] == int(Y["is_rare"][Y["gold"]].sum()), f"{Y['N']} reports")
    for mk in LLMS:
        G = groups_of(Y, finals[mk])
        want = M["llms"][mk]["rules"]["k5/either"]["confirmed_only"]
        got = {"right": int(G["correct"].sum()), "false": int(G["false"].sum())}
        check(c, f"Ours correct / false = results/rare_year ({mk})", got == {k: want[k] for k in got}, str(got))
    W = json.loads((ROOT / "results" / "why_it_works" / f"{c}.json").read_text(encoding="utf-8"))["years"][YEAR]
    for mk in LLMS:
        f = np.flatnonzero(groups_of(Y, finals[mk])["false"])
        stored = W[mk]["errors"]["gate"]
        if c == "maude":
            rel = Counter(relation(Y["paths"][finals[mk][i]], Y["paths"][Y["gold"][i]]) for i in f)
            check(c, f"false placements by place in the code hierarchy = why_it_works ({mk})",
                  all(rel[k] == v for k, v in stored.items()), str(dict(rel)))
        else:
            T = pool_rows(ROOT / "data" / c / f"pool_{YEAR}.jsonl", Y["keys"][f])
            said = sum(short_name(Y, finals[mk][i]).lower()[:5] in T[int(Y["keys"][i])]["text"].lower() for i in f)
            check(c, f"false placements with the placed term in the narrative = why_it_works ({mk})",
                  stored == {"false": len(f), "placed_term_in_narrative": said}, f"{len(f)} false, {said} with the term")
    return W


def check_row(Y, i, row, W, pool):
    c, k = Y["c"], int(Y["keys"][i])
    cls, g = Y["cls"], int(Y["gold"][i])
    check(row, "pool code = classifier file's filed code", pool["code"] == cls[g], f"{pool['code']} vs {cls[g]}")
    if c == "maude":
        check(row, "pool path = split path of the filed code", pool.get("path") == Y["paths"][g], pool.get("path", ""))
    sl = json.loads((ROOT / "runs" / f"{c}{YEAR}short" / "shortlist.json").read_text(encoding="utf-8"))
    item = next((t for t in sl["items"] if t["key"] == k), None)
    if item:
        ok = item["code"] == cls[g] and item["rare"] == cls[Y["best"][i]] and item["rank"] == int(Y["rank"][i])
        check(row, "shortlist.json: filed code, top rare code and its rank", ok, str(item))
    else:
        had = any(r["file"].split("/")[1] in (f"{c}{YEAR}", f"{c}{YEAR}nat") for r in Y["raw"].get((LLM, "zs", k), []))
        check(row, "not in shortlist.json because its top rare code is below the top 10 or it had LLM answers already",
              int(Y["rank"][i]) >= 10 or had, f"top rare code rank {int(Y['rank'][i]) + 1}, earlier answers {had}")
    for d in (f"{c}{YEAR}", f"{c}{YEAR}nat"):
        p = ROOT / "runs" / d / f"encoder_{TAG[c]}.csv"
        for line in p.read_text(encoding="utf-8").splitlines()[1:]:
            q = line.split(",")
            if q[0] == str(k):
                check(row, f"saved classifier answer in runs/{d}/{p.name}", q[4] == cls[Y["enc"][i]] and q[1] == cls[g],
                      f"pred {q[4]}, gold {q[1]}")
    for mk in LLMS:
        for name in KINDS:
            recs = Y["raw"].get((mk, name, k), [])
            used = Y["A"][(mk, name)][i]
            check(row, f"{mk} {name}: answer on file", len(recs) > 0, ", ".join(r["file"] for r in recs))
            codes = {r["code"] for r in recs}
            if len(codes) > 1:
                check(row, f"{mk} {name}: files disagree; the later file is used, as in the paper", True,
                      "; ".join(f"{r['file']}: {r['text']}" for r in recs))
            check(row, f"{mk} {name}: parsed answer = last file's answer", recs and
                  recs[-1]["code"] == (cls[used] if used >= 0 else None), recs[-1]["text"] if recs else "")
    for mk in LLMS:
        for s in W[mk]["errors"]["sample"]:
            if s["key"] == k:
                fin = ours(Y, mk)[i]
                ok = (s["placed"] == Y["paths"][fin] and s["filed"] == Y["paths"][g] and
                      s["encoder_top1"] == Y["paths"][Y["enc"][i]] and canon(s["text"]) == canon(pool["text"])[:len(canon(s["text"]))])
                check(row, f"stored false placement in results/why_it_works/{c}.json ({mk})", ok,
                      f"placed {s['placed']}, filed {s['filed']}")


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, summary = [], {}
    for c, title in STREAMS:
        Y = load_stream(c)
        n_train = training_counts(c, Y["split"])
        known = [x["code"] for x in Y["split"]["classes"] if not x["hidden"]]
        check(c, "training reports sum to the split's train_full", sum(n_train[x] for x in known) ==
              Y["split"]["counts"]["train_full"], str(sum(n_train[x] for x in known)))
        rare_codes = {x for x in known if 1 <= n_train[x] <= 5}
        check(c, "rare codes (1 to 5 training reports) = the classifier file's rare set",
              rare_codes == {Y["cls"][j] for j in np.flatnonzero(Y["is_rare"])}, f"{len(rare_codes)} rare codes")
        if c == "maude":
            leaves = Counter(p.split(" > ")[-1].lower() for p in Y["paths"])
            check(c, "last levels of the device paths are unique", max(leaves.values()) == 1, "")
        finals = {mk: ours(Y, mk) for mk in LLMS}
        W = check_whole_year(Y, finals)
        G = groups_of(Y, finals[LLM])
        keys = Y["keys"]
        filed_rank = np.full(Y["N"], 99)
        hit = Y["top"] == Y["gold"][:, None]
        filed_rank[hit.any(1)] = hit.argmax(1)[hit.any(1)] + 1
        summary[c] = {"reports": Y["N"], "rare_reports": int(Y["is_rare"][Y["gold"]].sum()),
                      "group_sizes": {g: int(m.sum()) for g, m in G.items()},
                      "correct_with_filed_code_first": int((G["correct"] & (filed_rank == 1)).sum()),
                      "none_by_rank_of_filed_code": {"1-5": int((G["none"] & (filed_rank <= 5)).sum()),
                                                     "6-20": int((G["none"] & (filed_rank > 5) & (filed_rank <= 20)).sum()),
                                                     ">20": int((G["none"] & (filed_rank > 20)).sum())},
                      "none_named_by_an_llm_coder": int((G["none"] & ((Y["A"][(LLM, "zs")] == Y["gold"]) |
                                                                      (Y["A"][(LLM, "rag")] == Y["gold"]))).sum())}
        picks = []
        for g in GROUPS:
            idx = np.flatnonzero(G[g] & (filed_rank >= 2) if g == "correct" else G[g])
            idx = idx[np.argsort(keys[idx], kind="stable")]
            picks.append((g, int(rng.choice(idx))))
        pool = pool_rows(ROOT / "data" / c / f"pool_{YEAR}.jsonl", [keys[i] for _, i in picks])
        missing = [(c, int(keys[i])) for _, i in picks if (c, int(keys[i])) not in EXCERPTS]
        if missing:
            for g, i in picks:
                print(f"\n{c} {g} {int(keys[i])}: {canon(pool[int(keys[i])]['text'])}")
            raise SystemExit(f"choose the excerpt fragments of {missing} in EXCERPTS")
        dates = json.loads((ROOT / "data" / "maude" / f"dates_{YEAR}.json").read_text(encoding="utf-8")) if c == "maude" else {}
        for g, i in picks:
            k = int(keys[i])
            row = f"{title} {g} {k}"
            P = pool[k]
            fin = int(finals[LLM][i])
            placed = fin if Y["is_rare"][fin] else None
            check(row, "group of the row", {"correct": placed is not None and fin == Y["gold"][i],
                                            "false": placed is not None and fin != Y["gold"][i],
                                            "none": placed is None and Y["is_rare"][Y["gold"][i]]}[g], g)
            check_row(Y, i, row, W, P)
            ex, words = excerpt(P["text"], EXCERPTS[(c, k)], row)
            check(row, "excerpt has at most 20 words", words <= 20, f"{words} words")
            fr = int(filed_rank[i])

            def info(j):
                j = int(j)
                if j < 0:
                    return {-2: "no answer", -1: "invalid"}[j]
                return {"code": Y["cls"][j], "path": Y["paths"][j], "name": short_name(Y, j), "rare": bool(Y["is_rare"][j]),
                        "training_reports": int(n_train[Y["cls"][j]])}
            gem = int(finals["gemma4_31b"][i])
            rows.append({
                "corpus": c, "stream": title.lower(), "group": g, "key": k, "date": P.get("date") or dates.get(str(k)),
                "narrative": canon(P["text"]), "excerpt_fragments": EXCERPTS[(c, k)], "excerpt_words": words,
                "classifier_first": info(Y["enc"][i]), "filed_rank": fr if fr <= 20 else ">20",
                "classifier_top5": [info(j) for j in Y["top"][i, :5]],
                "top_rare_code": info(Y["best"][i]) | {"rank": int(Y["rank"][i]) + 1},
                "zero_shot": info(Y["A"][(LLM, "zs")][i]), "retrieval": info(Y["A"][(LLM, "rag")][i]),
                "ours": info(placed) if placed is not None else "none", "filed": info(Y["gold"][i]),
                "llm_files": {f"{mk} {n}": Y["raw"].get((mk, n, k), []) for mk in LLMS for n in KINDS},
                "with_gemma4_31b": {"zero_shot": info(Y["A"][("gemma4_31b", "zs")][i]),
                                    "retrieval": info(Y["A"][("gemma4_31b", "rag")][i]),
                                    "ours": info(gem) if Y["is_rare"][gem] else "none"},
                "tex": {"excerpt": ex, "first": code_cell(Y, int(Y["enc"][i])), "rank": str(fr) if fr <= 20 else "$>$20",
                        "zs": answer_cell(Y, Y["A"][(LLM, "zs")][i], i), "rag": answer_cell(Y, Y["A"][(LLM, "rag")][i], i),
                        "ours": code_cell(Y, placed), "filed": code_cell(Y, int(Y["gold"][i]))}})
            print(f"[{row}] first {rows[-1]['tex']['first']} | rank {rows[-1]['tex']['rank']} | zs {rows[-1]['tex']['zs']} | "
                  f"rag {rows[-1]['tex']['rag']} | Ours {rows[-1]['tex']['ours']} | filed {rows[-1]['tex']['filed']}", flush=True)
    for words, what, cap in ((CAPTION, "caption", 40), (re.sub(r"\$[^$]*\$", "", NOTE), "table note", 30)):
        n = len(words.split())
        check("text", f"{what} within {cap} words", n <= cap, f"{n} words")

    snippet = table_tex(rows)
    (OUT / "tab_examples.tex").write_text(snippet, encoding="utf-8")
    compile_preview(snippet)
    failed = [x for x in CHECKS if not x["ok"]]
    (OUT / "tab_examples_rows.json").write_text(json.dumps(
        {"script": "scripts/paper_v6_examples.py", "seed": SEED, "year": YEAR, "llm": LLMS[LLM],
         "rule": "Ours: top rare code within the classifier's top 5, named by the zero-shot or the retrieval LLM",
         "draw": "numpy default_rng(42); streams device, aircraft; groups correct, false, none; reports by key",
         "caption": CAPTION, "note": NOTE, "groups": summary, "rows": rows, "checks": CHECKS,
         "checks_failed": len(failed)}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(CHECKS)} checks, {len(failed)} failed; wrote {OUT / 'tab_examples.tex'}, tab_examples_preview.pdf/.png, "
          f"tab_examples_rows.json", flush=True)
    if failed:
        raise SystemExit(1)


def compile_preview(snippet):
    """The snippet in the manuscript's class (cas-sc), cropped to the table; PNG at 150 dpi."""
    env = dict(os.environ, PATH=str(TEX_BIN) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "tab_examples.tex").write_text(snippet, encoding="utf-8")
        (tmp / "preview.tex").write_text("\n".join([
            r"\documentclass[a4paper,fleqn]{cas-sc}", r"\usepackage{amsmath,amssymb}", r"\begin{document}",
            r"\thispagestyle{empty}", r"\setcounter{table}{3}", r"\input{tab_examples}", r"\end{document}", ""]),
            encoding="utf-8")
        run = subprocess.run([str(TEX_BIN / "pdflatex.exe"), "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
                             cwd=tmp, env=env, capture_output=True, text=True, encoding="latin-1")
        log = (tmp / "preview.log").read_text(encoding="latin-1") if (tmp / "preview.log").exists() else run.stdout
        check("latex", "preview compiles", run.returncode == 0, run.stdout[-800:] if run.returncode else "")
        boxes = re.findall(r"(Overfull \\hbox \([^)]*\)[^\n]*)", log)
        check("latex", "no overfull boxes", not boxes, "; ".join(boxes))
        if run.returncode:
            return
        subprocess.run([str(TEX_BIN / "pdfcrop.exe"), "--margins", "10", "preview.pdf", "cropped.pdf"], cwd=tmp, env=env,
                       capture_output=True, check=True)
        shutil.copyfile(tmp / "cropped.pdf", OUT / "tab_examples_preview.pdf")
        subprocess.run([str(TEX_BIN / "rungs.exe"), "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r150",
                        "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-sOutputFile=" + str(OUT / "tab_examples_preview.png"),
                        str(OUT / "tab_examples_preview.pdf")], env=env, capture_output=True, check=True)


if __name__ == "__main__":
    main()
