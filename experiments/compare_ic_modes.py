#!/usr/bin/env python
"""Does causal marching survive when the chain is actually chained?

The ablation in runs/ablation gave every window a fresh LBM snapshot as its
initial condition, so a four-window run consumed four ground-truth slices
against one for the single-window configurations it was compared with. The
runs in runs/ablation_icnet are identical except that each window after the
first takes its target from the previous window's network, so the whole run
touches ground truth once.

This prints the two side by side. The comparison it supports is narrow and
worth stating plainly: the difference between the columns is the extra
ground truth, because nothing else differs.

    python experiments/compare_ic_modes.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    ("sigma 2", "runs/ablation/summary.json", "runs/ablation_icnet/summary.json"),
    ("sigma 8", "runs/ablation_sigma8/summary.json",
     "runs/ablation_icnet_sigma8/summary.json"),
]


def load(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text())


def agg(rows, config, key):
    vals = [r[key] for r in rows if r["config"] == config and r.get(key) is not None]
    return (statistics.mean(vals), vals) if vals else (None, [])


def gt_budget(rows, config):
    """Ground-truth points a config consumed, if the run recorded it."""
    for r in rows:
        if r["config"] == config:
            ic = r.get("gt_ic_points")
            dat = r.get("gt_data_points") or 0
            if ic is not None:
                return ic + dat, r.get("gt_ic_slices")
    return None, None


def main():
    any_found = False
    for label, dataset_rel, network_rel in PAIRS:
        base, icnet = load(dataset_rel), load(network_rel)
        print("=" * 78)
        print("%s   %s  vs  %s" % (label, dataset_rel, network_rel))
        print("=" * 78)
        if base is None:
            print("  baseline run missing: %s\n" % dataset_rel)
            continue
        if icnet is None:
            print("  chained run not finished yet: %s\n" % network_rel)
            continue
        any_found = True

        configs = [c for c in ("march", "all")
                   if any(r["config"] == c for r in icnet)]
        print("%-10s %22s %22s %12s" %
              ("config", "dataset ICs (before)", "chained (after)", "change"))
        print("-" * 78)
        for c in configs:
            for key, fmt, name in (("shedding", "%.1f%%", "shedding retained"),
                                   ("rel_l2", "%.4f", "relative L2")):
                mb, vb = agg(base, c, key)
                mn, vn = agg(icnet, c, key)
                if mb is None or mn is None:
                    continue
                scale = 100.0 if key == "shedding" else 1.0
                delta = (mn - mb) * scale
                print("%-10s %-18s %10s %22s %+11.1f" % (
                    c if key == "shedding" else "", name,
                    fmt % (mb * scale), fmt % (mn * scale), delta)
                    if key == "shedding" else
                    "%-10s %-18s %10s %22s %+11.4f" % (
                        "", name, fmt % (mb * scale), fmt % (mn * scale), delta))
            gb, sb = gt_budget(base, c)
            gn, sn = gt_budget(icnet, c)
            if gn is not None:
                print("%-10s %-18s %10s %22s" % (
                    "", "ground-truth pts",
                    "%s" % (gb if gb is not None else "not recorded"),
                    "%d (%s slice)" % (gn, sn)))
            print()

        # the control the comparison rests on: the one-window configs, whose
        # information budget did not change
        for c in ("baseline", "fourier", "decompose"):
            m, v = agg(base, c, "shedding")
            if m is not None:
                print("  reference: %-10s %5.1f%%  (1 window, unchanged by this flag)"
                      % (c, m * 100))
        print()

    if not any_found:
        print("Nothing to compare yet. Run queue9.sh on the GPU box first.")
        return 1

    print("Reading the result:")
    print("  chained stays near the dataset-IC value -> causality did the work,")
    print("    and the claim now holds on a quarter of the reference data.")
    print("  chained collapses toward the one-window configs -> the extra")
    print("    ground-truth slices were doing the work, not the causal structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
