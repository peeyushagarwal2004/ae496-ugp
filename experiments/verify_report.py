#!/usr/bin/env python
"""Verify every number printed in report/report.tex against its run artifact.

The report must not contain a hand-typed number that nothing can regenerate.
report/numbers.json maps each printed value to the artifact it came from, the
key inside it, and the reduction applied. This script checks two things per
entry, which fail in different ways and must both hold:

  1. TEX   -- the literal still appears on the recorded line of report.tex.
              Catches the report being edited away from the manifest.
  2. VALUE -- recomputing from the artifact reproduces the printed value to
              within half the last printed digit. Catches the artifact being
              regenerated without the report following, which is how the
              multiperiod table went stale after the St estimator changed.

Entries whose source is {"kind": "manual"} are numbers with no machine-readable
artifact. They are listed as UNCHECKED with a provenance string rather than
silently counted as passing; --strict makes them fail.

    python experiments/verify_report.py             # check
    python experiments/verify_report.py -v          # list every number
    python experiments/verify_report.py --strict    # UNCHECKED also fails
    python experiments/verify_report.py --resync    # rewrite moved line hints
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "report" / "report.tex"
MANIFEST = ROOT / "report" / "numbers.json"

PASS, MISMATCH, TEXDRIFT, TEXMOVED, UNCHECKED, ERROR = (
    "PASS", "MISMATCH", "TEX-DRIFT", "TEX-MOVED", "UNCHECKED", "ERROR")

_cache: dict[Path, object] = {}


def load(rel: str):
    p = ROOT / rel
    if p not in _cache:
        if not p.exists():
            raise FileNotFoundError(rel)
        _cache[p] = json.loads(p.read_text())
    return _cache[p]


# --------------------------------------------------------------------------
# reducers: each returns the raw value, before `unit` and `vs` are applied
# --------------------------------------------------------------------------

def _reduce(values, how):
    if not values:
        raise ValueError("no rows matched")
    return {"mean": statistics.mean, "min": min, "max": max,
            "first": lambda v: v[0], "last": lambda v: v[-1],
            "spread_pct": lambda v: (max(v) - min(v)) / min(v) * 100.0,
            }[how](values)


def src_list(s):
    """List-of-dicts summary (runs/*/summary.json): filter, pluck, reduce."""
    rows = load(s["file"])
    where = s.get("where", {})
    hits = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if not hits:
        raise ValueError("no row matching %s" % where)
    return _reduce([r[s["key"]] for r in hits], s.get("reduce", "mean"))


def src_dict(s):
    """Dict-of-dicts summary (runs/re_sweep/summary.json) keyed by a label.

    `keys` reduces across several keys of the same record, which is how the
    per-seed fields (all_rel_l2_s0, all_rel_l2_s1) are combined.
    """
    d = load(s["file"])[s["at"]]
    keys = s.get("keys", [s["key"]] if "key" in s else [])
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError("key(s) %s absent at %r" % (missing, s["at"]))
    return _reduce([d[k] for k in keys], s.get("reduce", "mean"))


def src_item(s):
    """One record out of a list (runs/*/shedding_accuracy.json)."""
    return load(s["file"])[s.get("index", 0)][s["key"]]


def src_scalar(s):
    """Flat dict (a dataset *_meta.json)."""
    d = load(s["file"])
    if s["key"] not in d:
        raise ValueError("key %r absent" % s["key"])
    return d[s["key"]]


def src_fno(s):
    """FNO rollout curve.

    Reproduces the reduction experiments/plot_rollout.py draws as the bold
    line, which is what the report quotes: mean over seeds first, then the
    pairwise mean of adjacent steps plotted at the later step, so the value at
    step n is mean(eps[n-1], eps[n]) of the seed-averaged curve. Without this
    the parity oscillation (30-45 % between even and odd steps) makes any
    single step's number meaningless.
    """
    metric = s.get("metric", "eps_full")
    curves = []
    for seed in s.get("seeds", [0, 1]):
        blob = load("runs/%s/%s_s%d/results.json"
                    % (s.get("run", "fno"), s["variant"], seed))
        curves.append(blob["curve"])
    steps = curves[0]["steps"]
    n = len(curves)
    eps = [sum(c[metric][i] for c in curves) / n for i in range(len(steps))]
    i = steps.index(s["step"])
    if s.get("pairwise", True):
        if i == 0:
            raise ValueError("step 1 has no preceding step to pair with")
        return 0.5 * (eps[i - 1] + eps[i])
    return eps[i]


def src_convergence(s):
    """data/convergence.json: a resolution series and a blockage series."""
    rows = load(s["file"])
    series = s["series"]
    hits = [r for r in rows if r["series"] in (series, "both")]
    if "blockage_pct" in s:
        hits = [r for r in hits
                if abs(r["blockage_pct"] - s["blockage_pct"]) < 1e-6]
    if "D_lattice" in s:
        hits = [r for r in hits if r["D_lattice"] == s["D_lattice"]]
    if not hits:
        raise ValueError("no convergence row matched")
    if s.get("reduce") == "zero_blockage_linear":
        # Linear extrapolation through the two LOWEST blockages only. The
        # three-point set is visibly non-linear -- 5.0 -> 3.3 moves Cd about
        # six times as much per point of blockage as 3.3 -> 2.5 -- so a fit
        # through all three would be dominated by the most confined case.
        pts = sorted((r["blockage_pct"], r[s["key"]]) for r in hits)[:2]
        (b0, c0), (b1, c1) = pts
        slope = (c1 - c0) / (b1 - b0)
        return c0 - slope * b0
    return _reduce([r[s["key"]] for r in hits], s.get("reduce", "mean"))


def src_path(s):
    """A dotted key path into any JSON artifact.

    Used for the files that exist only to make previously hand-computed
    numbers regenerable: runs/derived_metrics.json (written by
    experiments/derived_metrics.py) and runs/operator_verification.json
    (written by tests/test_ns2d_unsteady.py).
    """
    node = load(s["file"])
    for key in s["path"].split("."):
        if isinstance(node, list):
            node = node[int(key)]
        elif key in node:
            node = node[key]
        else:
            raise ValueError("no key %r along path %r in %s"
                             % (key, s["path"], s["file"]))
    if not isinstance(node, (int, float)):
        raise ValueError("path %r resolved to %s, not a number"
                         % (s["path"], type(node).__name__))
    return float(node)


def src_ratio(s):
    """A ratio of two other sources, or of a source and a constant.

    {"kind": "ratio", "num": {<src>}, "den": {<src>} | <number>,
     "mode": "pct" | "ratio"}

    mode "pct" returns (num / den - 1) * 100 -- a percentage change, which is
    how the report quotes effects like "moving the inlet removes 4.7 % of
    drag". mode "ratio" returns num / den.
    """
    num = SOURCES[s["num"]["kind"]](s["num"])
    den = s["den"]
    if isinstance(den, dict):
        den = SOURCES[den["kind"]](den)
    mode = s.get("mode", "pct")
    if mode == "pct":
        return (num / den - 1.0) * 100.0
    if mode == "product":
        # e.g. a frequency error times the number of periods it acted over
        # gives the phase error accumulated across a run
        return num * den
    return num / den


def src_logline(s):
    """A number printed into a committed log file, matched by regex group 1."""
    text = (ROOT / s["file"]).read_text(errors="replace")
    m = re.search(s["pattern"], text)
    if not m:
        raise ValueError("pattern %r not found in %s" % (s["pattern"], s["file"]))
    return float(m.group(1))


SOURCES = {"list": src_list, "dict": src_dict, "item": src_item,
           "scalar": src_scalar, "fno": src_fno, "path": src_path,
           "ratio": src_ratio,
           "convergence": src_convergence, "logline": src_logline}


def compute(entry):
    s = entry["src"]
    raw = SOURCES[s["kind"]](s)
    unit = entry.get("unit", "raw")
    if unit == "pct":
        raw *= 100.0
    elif unit == "deg_from_periods":
        raw *= 360.0
    elif unit != "raw":
        raise ValueError("unknown unit %r" % unit)
    if "vs" in entry:                       # express the value as a % error
        raw = (raw / entry["vs"] - 1.0) * 100.0
    return raw


# --------------------------------------------------------------------------

def find_literal(literal, lines):
    """1-based line numbers where `literal` occurs.

    TeX source wraps sentences, so a literal may contain one newline; it is
    then matched against each line joined to the next and reported at the
    first of the two. report/build_manifest.py uses the same rule.
    """
    if "\n" in literal:
        return [i + 1 for i in range(len(lines) - 1)
                if literal in lines[i] + "\n" + lines[i + 1]]
    return [i for i, ln in enumerate(lines, 1) if literal in ln]


def check(entry, lines):
    literal = entry["literal"]
    want = entry["tex"]
    found = find_literal(literal, lines)

    tex_status, tex_msg = PASS, ""
    if not found:
        tex_status = TEXDRIFT
        tex_msg = "literal %r no longer appears in report.tex" % literal
    elif want not in found:
        tex_status = TEXMOVED
        tex_msg = "literal moved: expected line %d, found on %s" % (want, found)

    if entry["src"]["kind"] == "manual":
        return (UNCHECKED if tex_status == PASS else tex_status, None,
                tex_msg or entry["src"].get("provenance", ""))

    try:
        got = compute(entry)
    except Exception as exc:                # noqa: BLE001 - reported, not raised
        return ERROR, None, "%s: %s" % (type(exc).__name__, exc)

    if abs(got - entry["printed"]) > entry["tol"]:
        return (MISMATCH, got,
                "printed %s, artifact gives %.6g (tol %s)%s"
                % (entry["printed"], got, entry["tol"],
                   "; " + tex_msg if tex_msg else ""))
    return (PASS if tex_status == PASS else tex_status), got, tex_msg


def _untracked_sources(entries):
    """Source files the manifest reads that git is not tracking.

    An artifact that only exists on one laptop is not a reproducible source,
    whatever the checker says about it today. This caught data/*_meta.json and
    data/convergence.json, which a blanket `data/` ignore rule was hiding
    along with the 400 MB of .npz it was meant to exclude.
    """
    wanted = {}

    def files_of(src):
        if not isinstance(src, dict):
            return []
        out = [src["file"]] if src.get("file") else []
        for k in ("num", "den"):
            out += files_of(src.get(k))
        return out

    for e in entries:
        for f in files_of(e["src"]):
            wanted[f] = wanted.get(f, 0) + 1
    if not wanted:
        return {}
    try:
        import subprocess
        out = subprocess.run(["git", "ls-files", "--", *wanted],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=20)
        if out.returncode != 0:
            return {}
        tracked = {line.strip().replace("\\", "/")
                   for line in out.stdout.splitlines() if line.strip()}
    except Exception:                       # noqa: BLE001 - git is optional
        return {}
    return {f: n for f, n in wanted.items()
            if f.replace("\\", "/") not in tracked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="UNCHECKED entries fail too")
    ap.add_argument("--resync", action="store_true",
                    help="rewrite TEX-MOVED line hints into the manifest")
    ap.add_argument("--only", default=None, help="substring filter on entry id")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print PASS lines as well")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["entries"]
    lines = TEX.read_text(errors="replace").splitlines()
    if args.only:
        entries = [e for e in entries if args.only in e["id"]]

    counts, bad, moved = {}, [], []
    for e in entries:
        status, got, msg = check(e, lines)
        counts[status] = counts.get(status, 0) + 1
        # A line hint can be stale on an entry that also MISMATCHes, so decide
        # resyncability from the literal alone rather than from the status.
        found = find_literal(e["literal"], lines)
        if found and e["tex"] not in found:
            moved.append(e)
        if status != PASS or args.verbose:
            shown = "%.6g" % got if got is not None else "--"
            print("%-10s %-40s tex:%-4d printed=%-10s artifact=%-10s %s"
                  % (status, e["id"], e["tex"], e["printed"], shown, msg))
        if status in (MISMATCH, TEXDRIFT, ERROR):
            bad.append(e)

    if args.resync and moved:
        for e in moved:
            e["tex"] = find_literal(e["literal"], lines)[0]
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print("\nresynced %d line hint(s) in %s" % (len(moved), MANIFEST.name))

    print("\n%d numbers checked: %s"
          % (len(entries),
             ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items()))))

    untracked = _untracked_sources(entries)
    if untracked:
        print("\nWARNING: %d source file(s) are not tracked by git, so the "
              "numbers that come from them cannot be re-verified from a fresh "
              "clone:" % len(untracked))
        for f, n in sorted(untracked.items()):
            print("  %-46s %d number(s)" % (f, n))

    unchecked = counts.get(UNCHECKED, 0)
    if unchecked:
        print("%d number(s) have no machine-readable artifact "
              "(-v lists their provenance)." % unchecked)
    if bad:
        print("\nFAIL: %d number(s) do not reconcile." % len(bad))
        return 1
    if args.strict and unchecked:
        print("\nFAIL (--strict): unverifiable numbers present.")
        return 1
    print("\nOK: every checked number reproduces from its artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
