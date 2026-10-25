#!/usr/bin/env python
"""Check that every number in report/paper.tex and report/paper_long.tex also appears in report/report.tex.

report.tex is the checked source (experiments/verify_report.py ties each of its
numbers to a run artifact). The two papers are shorter versions and must not
introduce numbers of their own. This is a presence check: it confirms each number
exists in report.tex, not that it is used in the same sentence, so read any
number you move between contexts.

    python experiments/check_paper_numbers.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"\d+(?:\\,\d{3})*(?:\.\d+)?")


def body(path):
    text = path.read_text(encoding="utf-8")
    text = text.split(r"\begin{document}", 1)[1]
    text = text.split(r"\begin{thebibliography}", 1)[0]
    lines = []
    for line in text.splitlines():
        line = re.sub(r"(?<!\\)%.*", "", line)  # comments, but not \%
        # file names, labels and refs carry digits that are not results
        line = re.sub(r"\\(includegraphics|label|ref|cite)(\[[^\]]*\])?\{[^}]*\}", "", line)
        lines.append(line)
    return lines


def numbers(lines):
    return {m.group(0).replace(r"\,", "") for line in lines for m in NUM.finditer(line)}


def check(name, known):
    paper = body(ROOT / "report" / name)
    missing = []
    for line in paper:
        for m in NUM.finditer(line):
            n = m.group(0).replace(r"\,", "")
            if n not in known:
                missing.append((n, line.strip()))
    for n, line in missing:
        print(f"{name}: NOT IN report.tex  {n:>10}  {line[:80]}")
    if not missing:
        print(f"OK: all {len(numbers(paper))} distinct numbers in {name} appear in report.tex")
    return not missing


def main():
    known = numbers(body(ROOT / "report" / "report.tex"))
    ok = all([check(name, known) for name in ("paper.tex", "paper_long.tex")])
    if not ok:
        print("\nFAIL: some numbers are not in report.tex")
        sys.exit(1)


if __name__ == "__main__":
    main()
