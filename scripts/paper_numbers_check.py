"""Compare the numbers of two versions of the manuscript, so that a rewrite cannot change or invent a result.

Usage: python paper_numbers_check.py <old.tex> <new.tex>
Prints the numbers of the new file that the old file does not hold (each with its line), and how many numbers of
the old file are no longer in the new one. Citation keys, labels, comments and layout lengths are left out.
"""
import collections
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def strip(line):
    line = re.sub(r"(?<!\\)%.*$", "", line)
    line = re.sub(r"\\(cite|ref|label|includegraphics|bibliography|bibliographystyle|usepackage|documentclass|setcounter|setlength)(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", " ", line)
    line = re.sub(r"\\(renewcommand|def)\S*", " ", line)
    line = re.sub(r"\\\\\[[^\]]*\]", " ", line)
    line = re.sub(r"(width|height)=[\d.]+\\?[a-z]*", " ", line)
    line = re.sub(r"p\{[\d.]+cm\}", " ", line)
    return line.replace("$-$", "-").replace("\\%", "%").replace("{,}", ",")


def numbers(path):
    out = collections.defaultdict(list)
    for i, raw in enumerate(io.open(path, encoding="utf-8").read().splitlines(), 1):
        for n in re.findall(r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?", strip(raw)):
            n = n.rstrip(",.")
            if n:
                out[n].append(i)
    return out


def main():
    old, new = numbers(sys.argv[1]), numbers(sys.argv[2])
    added = {n: ls for n, ls in new.items() if n not in old}
    lost = {n: ls for n, ls in old.items() if n not in new}
    print(f"numbers: old {len(old)}, new {len(new)}; new ones absent from the old file: {len(added)}; old ones no longer present: {len(lost)}")
    src = io.open(sys.argv[2], encoding="utf-8").read().splitlines()
    for n, ls in sorted(added.items(), key=lambda kv: kv[1][0]):
        print(f"  NEW {n:>10s}  L{ls[0]}: {src[ls[0] - 1].strip()[:120]}")
    if "--lost" in sys.argv:
        osrc = io.open(sys.argv[1], encoding="utf-8").read().splitlines()
        for n, ls in sorted(lost.items(), key=lambda kv: kv[1][0]):
            print(f"  LOST {n:>10s}  old L{ls[0]}: {osrc[ls[0] - 1].strip()[:110]}")


if __name__ == "__main__":
    main()
