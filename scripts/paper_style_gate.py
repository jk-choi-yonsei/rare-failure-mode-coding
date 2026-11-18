"""Countable checks for the list of un-paper-like problems (notes/paper-style-concerns-2026-11-18.md).

Reads a manuscript (or one section file) and counts, per item of the list: captions and table notes over their word
budget, facts stated in several places, name variants that the term table bans, abbreviations used before or without
a definition, strong words, defensive wording, comparisons with a dropped noun, long and number-heavy sentences,
long paragraphs, section pointers and number-format slips. A line that breaks a budget starts with FLAG.

Usage: python paper_style_gate.py <file.tex> [--list]     (--list prints every offending line, not only counts)
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CAPTION_MAX, NOTE_MAX, LONG, VERY_LONG, PARA_MAX, POINTER_MAX = 40, 30, 35, 45, 130, 8

# name variants that the term table bans in running prose (regex, reason)
BANNED = [
    (r"\bencoder", "say classifier"),
    (r"five most likely codes|(?<!top )\bfive codes\b|\btop 5\b(?![^&]*&)|depth of (five|three|ten)", "say top five / top k"),
    (r"uniform draw|uniformly drawn", "say random sample"),
    (r"coder selector", "say DCS"),
    (r"condition term", "say code / part condition"),
    (r"\bconfirm", "say names"),
    (r"reference code|reference IMDRF", "say filed code"),
    (r"combinations of stream", "say cases"),
    (r"reports of common failure modes|reports of other codes|reports of another code", "say ordinary reports"),
    (r"the rare code that the (encoder|classifier) ranks highest(?!,)", "say the top rare code"),
    (r"in the proportions in which", "say natural proportions (define once)"),
    (r"\breaders?\b", "say reviewer"),
    (r"code tree", "say code hierarchy"),
    (r"\bOurs\b", "legend label used in prose"),
]
STRONG = [r"\bnever\b", r"\balways\b", r"\bcannot\b", r"no proof", r"do(es)? not solve", r"too many", r"hundredfold",
          r"\bthousands\b", r"\bfar (more|less|fewer|higher|rarer|outnumber|below|above)", r"\bclearly\b", r"\bobviously\b",
          r"every comparison", r"\bmust\b", r"\bnothing\b", r"\bat all\b", r"\bby far\b", r"\bdramatic", r"\bstriking"]
DEFENSIVE = [r"in view\b", r"could not be run", r"diverged", r"not computed", r"\bOne seed\b", r"in use the model",
             r"we found no", r"is taken as the correct", r"reference for scoring", r"second test"]
ELLIPTIC = [r"no more falsely", r"(fewer|more) falsely", r"of the four\b(?! cases)", r"in these three\b", r"the rare one\b",
            r"only [\d.]+\\?% to [\d.]+\\?% do\b", r"\bdoes the reverse\b", r"the two are level", r"nothing measurable"]
FACTS = {"LLM call share (0.65%)": r"0\.65\\?%", "2026 ends in October": r"October", "rare-code definition": r"one to five training reports",
         "LLM has not seen the top five": r"(not seen|without seeing|does not see)", "logit adjustment defined": r"share of the training reports",
         "rare share 1 in 1,200": r"1,200", "Implant subsidence": r"Implant subsidence", "review load 2.2 to 3.4": r"2\.2 to 3\.4",
         "gain 0.5 to 2.1": r"0\.5 to 2\.1", "gain 0.9 to 1.7": r"0\.9 to 1\.7", "1.4 to 2.4 times": r"1\.4 to 2\.4",
         "learns from reports / from the name": r"from (its|their|the code's) names?"}
KNOWN_NAMES = {"EQN", "GB", "RoBERTa-base", "RoBERTa-large", "RoBERTa", "EnergyRoute", "MADE", "GPT", "ISO", "US", "NONE", "NOT", "NO", "TEST", "CORRODED", "CRACKED", "UNKNOWN",
               "DISCREPANCY", "REPORTED", "ENTERED", "REQUIRED", "LLMs", "LLM-based", "ModernBERT", "AdamW", "GGUF", "CAS", "II",
               "A24", "A27", "ICD-10", "GPT-4", "GPT-3", "VerifiNER", "SuperICL", "CARE", "L2D-Clinical", "FrugalGPT", "OF", "THESE",
               "NONE_OF_THESE", "UD-Q4", "Q4", "QAT", "L2"}


def clean(s):
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\(ref|label)\{[^}]*\}", "X", s)
    s = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{(equation|align)\*?\}", " EQN ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", "M", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    return re.sub(r"\s+", " ", re.sub(r"[{}~\\]", " ", s)).strip()


def sentences(text):
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if len(x.split()) > 3]


def nums(s):
    """Quantities in a sentence: a range ('38% to 55%', 'from 32 to 77') counts once, years are not counted."""
    s = re.sub(r"(\d[\d,.]*)\s*%?\s+to\s+(\d[\d,.]*)", r"\1", s)
    return [n for n in re.findall(r"(?<![A-Za-z])\d[\d,.]*", s) if not re.fullmatch(r"20(2[3-6])[.,]?", n)]


def main():
    path = sys.argv[1]
    show = "--list" in sys.argv
    t = io.open(path, encoding="utf-8").read()
    a = t.find("\\section{Introduction}")
    b = t.find("\\section*{Data and code availability}")
    main_t = t[a if a >= 0 else 0:b if b >= 0 else len(t)]
    floats = r"(?s)\\begin\{(table|figure)\}.*?\\end\{(table|figure)\}"
    prose_raw = re.sub(floats, "\n\n", main_t)
    prose = clean(prose_raw)
    lines = t.splitlines()

    quoted, inq = set(), False  # the prompts quoted in the appendix are not the paper's own wording
    for i, l in enumerate(lines):
        inq = (inq or "\\begin{quote}" in l) and "\\end{quote}" not in l
        if inq:
            quoted.add(i)

    def where(pat, text_lines=lines, flags=re.I):
        return [(i + 1, l.strip()) for i, l in enumerate(text_lines)
                if i not in quoted and re.search(pat, re.sub(r"\\(ref|label)\{[^}]*\}", "", l), flags)]

    print(f"file: {path}\nwords in running prose (Introduction to Conclusion): {len(prose.split())}")

    print("\n[6] captions (max %d words) and table notes (max %d words)" % (CAPTION_MAX, NOTE_MAX))
    for m in re.finditer(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}\s*\\label\{([^}]*)\}", t):
        n = len(clean(m.group(1)).split())
        print(f"  {'FLAG' if n > CAPTION_MAX else 'ok  '} caption {m.group(2):20s} {n:4d}")
    for m in re.finditer(r"\\par\\smallskip\{\\footnotesize\\raggedright\\noindent (.*?)\\par\}", t, flags=re.S):
        n = len(clean(m.group(1)).split())
        print(f"  {'FLAG' if n > NOTE_MAX else 'ok  '} note    {clean(m.group(1))[:20]:20s} {n:4d}")

    print("\n[2] one fact in several places (whole file; a fact has one home, plus abstract, Introduction and Conclusion)")
    for k, pat in FACTS.items():
        hits = where(pat, flags=0)
        print(f"  {len(hits):3d}  {k}" + ("  lines " + ",".join(str(i) for i, _ in hits) if show or len(hits) > 4 else ""))

    print("\n[3] banned name variants in the whole file (0 wanted)")
    for pat, why in BANNED:
        hits = where(pat)
        if pat == r"\bOurs\b":  # the label is fine inside tables and when quoted as a method name with a bracket
            hits = [(i, l) for i, l in hits if "&" not in l and "Ours (" not in l and "label=" not in l]
        if hits:
            print(f"  FLAG {len(hits):3d}  {pat[:44]:44s} {why}" + ("" if not show else "".join(f"\n        L{i}: {l[:110]}" for i, l in hits)))
    print("  (lines with a variant are listed with --list)")

    print("\n[5] abbreviations in the main text: used before the definition, never defined, or defined twice")
    txt = prose  # floats left out: a table may label a row with an abbreviation that the text defines
    for ab in sorted(set(re.findall(r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z0-9-]*\b|\b[A-Z]{2,}[0-9-]*\b", txt))):
        if ab in KNOWN_NAMES or ab.rstrip("s") in KNOWN_NAMES:
            continue
        uses = [m.start() for m in re.finditer(r"(?<![A-Za-z])" + re.escape(ab) + r"(?![A-Za-z])", txt)]
        defs = [m.start() for m in re.finditer(r"\(" + re.escape(ab) + r"\)", txt)]
        flag = []
        if not defs and len(uses) >= 1:
            flag.append("never defined")
        if defs and uses and uses[0] < defs[0] - 1:
            flag.append("used before its definition")
        if len(defs) > 1:
            flag.append(f"defined {len(defs)} times")
        if flag:
            print(f"  FLAG {ab:14s} uses {len(uses):3d}  {', '.join(flag)}")

    print("\n[8] strong words, [9] defensive wording, [7] comparisons with a dropped noun (0 wanted)")
    for name, pats in (("strong", STRONG), ("defensive", DEFENSIVE), ("dropped noun", ELLIPTIC)):
        for pat in pats:
            hits = where(pat)
            if hits:
                print(f"  FLAG {name:12s} {len(hits):3d}  {pat}" + ("" if not show else "".join(f"\n        L{i}: {l[:110]}" for i, l in hits)))

    print("\n[7] sentences by section: n, mean words, share over %d words, over %d words, with 4+ quantities (a range counts once, years not counted)" % (LONG, VERY_LONG))
    secs = [(m.start(), m.group(1)) for m in re.finditer(r"\\section\{([^}]*)\}", main_t)] + [(len(main_t), "END")]
    if len(secs) == 1:
        secs = [(0, "(file)"), (len(main_t), "END")]
    for (s0, name), (s1, _) in zip(secs, secs[1:]):
        body = re.sub(floats, "\n\n", main_t[s0:s1])
        ss = sentences(clean(body))
        if not ss:
            continue
        w = [len(x.split()) for x in ss]
        n4 = [x for x in ss if len(nums(x)) >= 4]
        long_share = sum(x > LONG for x in w) / len(w)
        limit = 0.07 if name in ("Introduction", "Problem formulation", "Method", "Proposed method") else 0.20
        paras = [clean(p) for p in re.split(r"\n\s*\n", body)]
        pw = [len(p.split()) for p in paras if len(p.split()) > 25]
        flag = "FLAG" if long_share > limit or sum(x > VERY_LONG for x in w) or (name == "Results" and len(n4) / len(ss) > 0.05) else "ok  "
        print(f"  {flag} {name[:24]:24s} n {len(ss):3d}  mean {sum(w) / len(w):4.1f}  >{LONG}: {long_share:4.0%}  >{VERY_LONG}: {sum(x > VERY_LONG for x in w):2d}"
              f"  4+ quantities: {len(n4):3d} ({len(n4) / len(ss):.0%})  paragraphs {len(pw):2d}, mean {sum(pw) / max(1, len(pw)):4.0f} words, over {PARA_MAX}: {sum(x > PARA_MAX for x in pw)}")
        if show:
            for x in ss:
                if len(x.split()) > VERY_LONG:
                    print(f"        very long ({len(x.split())}): {x[:120]}")

    print("\n[7] repeated sentence openers in running prose")
    ss = sentences(prose)
    for p in ("On the device reports", "On the aircraft reports", "With ", "Without ", "In ", "Of the ", "When "):
        print(f"  {sum(s.startswith(p) for s in ss):3d}  {p}")
    allc = clean(main_t)
    print(f"  model name in prose and captions: Qwen3.8-27B {allc.count('Qwen3.8-27B')}, Gemma 4 31B {allc.count('Gemma 4 31B')}")

    print("\n[10] section pointers in running prose (max %d)" % POINTER_MAX)
    n = len(re.findall(r"Sections?~\\ref", prose_raw))
    print(f"  {'FLAG' if n > POINTER_MAX else 'ok  '} Section pointers {n}; table and figure pointers {len(re.findall(r'(Tables?|Figs?[.])~\\ref', prose_raw))}; "
          f"appendix pointers {len(re.findall(r'Appendix~\\ref', prose_raw))}")

    print("\n[13] formats")
    print(f"  year order: '2025 and 2026' {allc.count('2025 and 2026')}, '2026 and 2025' {allc.count('2026 and 2025')}")
    print(f"  rare share written as: '1 in ' {allc.count('1 in ')}, '1 to 1,' {allc.count('1 to 1,')}, ' to 1' {len(re.findall(r'[0-9] to 1[^0-9.,]', allc))}")
    print(f"  semicolons {prose.count(';')}, colons {prose.count(': ')}, em dashes {prose_raw.count('---')}")
    print(f"  displayed equations {len(re.findall(r'\\begin\{(equation|align)', t))}, list blocks {len(re.findall(r'\\begin\{(itemize|enumerate)\}', main_t))}")


if __name__ == "__main__":
    main()
