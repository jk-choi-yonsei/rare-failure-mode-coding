"""Keyword-rule arm for the SDR corpus (notes/prereg-sdr-2026-10-24.md, S6b), and the
leakage flags the scorer uses (S6a).

Prediction: among the known PartCondition terms, those that appear verbatim in the
narrative as whole words (case-insensitive); the longest wins; none -> abstain (-1,
scored as wrong). Leakage flag per test item: the first five characters of the gold
term appear in the narrative (case-insensitive), the measure of the corpus search.

Writes runs/sdr2025/keyword_2025.csv and runs/sdr/keyword_2024.csv (key, gold, pred,
leaked) for the 2025 test and the known 2024 test. Usage: MAINT_CORPUS=sdr python sdr_keyword.py
"""
import csv
import re

from common import CORPUS, rdir
from maude_data import POOL_2025, POOL_2026, class_info, load_items, load_split, load_test_2025, load_test_2026

if CORPUS != "sdr":
    raise SystemExit("set MAINT_CORPUS=sdr")


def main():
    split = load_split()
    info = class_info(split)
    known = {c["code"]: c["path"].upper() for c in split["classes"] if not c["hidden"]}
    patterns = sorted(((re.compile(r"(?<![A-Z0-9])" + re.escape(t) + r"(?![A-Z0-9])"), code, t) for code, t in known.items()),
                      key=lambda x: -len(x[2]))  # longest term first

    def predict(text):
        up = text.upper()
        for pat, code, _ in patterns:
            if pat.search(up):
                return code
        return "-1"

    def leaked(text, code):
        return int(info[code]["path"].upper()[:5] in text.upper())

    # 2026 (notes/prereg-redesign-2026-11-06.md): the leakage flag defines test D8's subset
    for name, test, pool, out in (("2026", load_test_2026(), POOL_2026, rdir("maude2026") / "keyword_2026.csv"),
                                  ("2025", load_test_2025(), POOL_2025, rdir("maude2025") / "keyword_2025.csv"),
                                  ("2024", [t for t in split["test"] if not info[t["code"]]["hidden"]], None,
                                   rdir("maude") / "keyword_2024.csv")):
        items = load_items([t["key"] for t in test], pool=pool) if pool else load_items([t["key"] for t in test])
        out.parent.mkdir(parents=True, exist_ok=True)
        hits = correct = leak = 0
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "gold", "pred", "leaked"])
            for t in test:
                text = items[t["key"]][0]
                p, lk = predict(text), leaked(text, t["code"])
                hits += p != "-1"
                correct += p == t["code"]
                leak += lk
                w.writerow([t["key"], t["code"], p, lk])
        n = len(test)
        print(f"{name}: {n} items; keyword rule predicts {hits / n:.1%}, accuracy {correct / n:.3f}; "
              f"gold-term stem in narrative {leak / n:.1%} -> {out.name}")


if __name__ == "__main__":
    main()
