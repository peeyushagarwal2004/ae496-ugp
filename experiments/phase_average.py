#!/usr/bin/env python
"""Phase-averaged shedding retention for the headline configurations.

A single-window baseline's "shedding retained" swings by a factor of four with
nothing but the starting phase of its initial condition (runs/v5/phase, on the
v4 dataset: 9.1, 25.6, 13.1, 34.9 % at quarter-period steps). Any comparison
quoted from one starting instant is therefore a comparison at that phase. This
gathers each configuration at all four quarter-period phases and both seeds
and reports the phase average with its spread.

Phase 0 is the default start time, which the main ablation already ran:
baseline and march from runs/v5/ablation, all (sigma 8) from
runs/v5/ablation_sigma8. Phases 1-3 come from runs/v5/phase_core.

    python experiments/phase_average.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("baseline", "march", "all", "baseline80k", "fd8_80k")
PHASE0 = {"baseline": "runs/v5/ablation/baseline_s{s}",
          "march": "runs/v5/ablation/march_s{s}",
          "all": "runs/v5/ablation_sigma8/all_s{s}",
          # equal-budget controls (queue18): one window at 80k epochs
          "baseline80k": "runs/v5/ctrl_budget80k/baseline_s{s}",
          "fd8_80k": "runs/v5/phase_core/fd8_80k_k0_s{s}"}
CORE = "runs/v5/phase_core/{c}_k{k}_s{s}"


def overall(path):
    p = ROOT / path / "results.json"
    if not p.exists():
        return None
    o = json.loads(p.read_text())["overall"]
    return {"shedding": o["shedding_ratio"], "rel_l2": o["rel_l2_mean"],
            "gt_ic_slices": o.get("gt_ic_slices")}


def gather():
    table = {}
    for c in CONFIGS:
        for k in range(4):
            for s in (0, 1):
                path = PHASE0[c].format(s=s) if k == 0 else CORE.format(c=c, k=k, s=s)
                r = overall(path)
                if r is not None:
                    table[(c, k, s)] = dict(r, path=path)
    return table


def summarise(table):
    out = {}
    for c in CONFIGS:
        rows = [v for (cc, _, _), v in table.items() if cc == c]
        if not rows:
            continue
        per_phase = {}
        for k in range(4):
            vals = [table[(c, k, s)]["shedding"] for s in (0, 1) if (c, k, s) in table]
            if vals:
                per_phase[k] = statistics.mean(vals)
        sh = [r["shedding"] for r in rows]
        l2 = [r["rel_l2"] for r in rows]
        out[c] = {
            "n_runs": len(rows),
            "phases_covered": sorted(per_phase),
            "shedding_by_phase": {str(k): v for k, v in per_phase.items()},
            "shedding_mean": statistics.mean(sh),
            "shedding_min": min(sh), "shedding_max": max(sh),
            "shedding_phase_mean_min": min(per_phase.values()),
            "shedding_phase_mean_max": max(per_phase.values()),
            "rel_l2_mean": statistics.mean(l2),
            "gt_ic_slices": sorted({r["gt_ic_slices"] for r in rows
                                    if r["gt_ic_slices"] is not None}),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "v5" / "phase_average.json"))
    args = ap.parse_args()

    table = gather()
    summ = summarise(table)
    print(f"{'config':9s} {'runs':>4s} " + "".join(f"{'phase ' + str(k):>10s}" for k in range(4))
          + f"{'mean':>9s} {'range':>15s} {'rel L2':>8s}")
    for c, r in summ.items():
        cells = "".join(f"{100 * r['shedding_by_phase'][str(k)]:9.1f}%"
                        if str(k) in r["shedding_by_phase"] else f"{'--':>10s}"
                        for k in range(4))
        print(f"{c:9s} {r['n_runs']:4d} {cells}{100 * r['shedding_mean']:8.1f}% "
              f"{100 * r['shedding_min']:6.1f}-{100 * r['shedding_max']:5.1f}% "
              f"{r['rel_l2_mean']:8.4f}")
    complete = all(len(r["phases_covered"]) == 4 and r["n_runs"] == 8
                   for r in summ.values()) and len(summ) == len(CONFIGS)

    # Paired by phase and seed: the effect of each mechanism at equal compute.
    pairs = {"causality (march vs baseline80k)": ("march", "baseline80k"),
             "causality with the other two mechanisms (all vs fd8_80k)":
                 ("all", "fd8_80k"),
             "compute (baseline80k vs baseline)": ("baseline80k", "baseline")}
    paired = {}
    for name, (a, b) in pairs.items():
        d = [table[(a, k, s)]["shedding"] - table[(b, k, s)]["shedding"]
             for k in range(4) for s in (0, 1)
             if (a, k, s) in table and (b, k, s) in table]
        if d:
            paired[name] = {"n": len(d), "mean_pts": 100 * statistics.mean(d),
                            "min_pts": 100 * min(d), "max_pts": 100 * max(d),
                            "all_positive": all(x > 0 for x in d)}
            print(f"{name:58s} n={len(d)}  mean {100 * statistics.mean(d):+5.1f} pts"
                  f"  range {100 * min(d):+5.1f} .. {100 * max(d):+5.1f}")
    print("\ncomplete (3 configs x 4 phases x 2 seeds):", complete)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(
        {"complete": complete, "summary": summ, "paired": paired,
         "runs": {f"{c}_k{k}_s{s}": v for (c, k, s), v in table.items()}},
        indent=2) + "\n")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
