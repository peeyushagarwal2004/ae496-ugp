#!/usr/bin/env python
"""Architecture study: Transformer / RNN / CNN against the project's baselines.

Two tables, one per track, both built only from saved run artifacts.

PINN track (queue19 part A), phase-averaged exactly like phase_average.py:

    tf, gru              single window, data free -- does the architecture
                         alone stop the collapse?  paired against `baseline`
    tf_march, gru_march  marched -- does it add anything on top of causal
                         time-marching?  paired against `march`

The MLP rows are read from the same paths phase_average.py uses, so the
pairing is by identical (phase, seed) on identical data.

Operator track (queue19 part B): rollout error at the quoted horizons for the
FNO (runs/fno), U-Net (runs/fno_unet) and ConvGRU (runs/fno_convgru), data
and physics variants, mean over seeds.

    python experiments/arch_compare.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from phase_average import CORE, PHASE0, overall

ROOT = Path(__file__).resolve().parents[1]
MLP = ("baseline", "march")
NEW = ("tf", "gru", "tf_march", "gru_march")
PAIRS = {"tf": "baseline", "gru": "baseline",
         "tf_march": "march", "gru_march": "march"}
OPERATORS = {"fno": "runs/fno", "unet": "runs/fno_unet", "convgru": "runs/fno_convgru"}
HORIZONS = (1, 10, 20, 50)


def pinn_table():
    table = {}
    for c in MLP + NEW:
        for k in range(4):
            for s in (0, 1):
                path = (PHASE0[c].format(s=s) if (k == 0 and c in PHASE0)
                        else CORE.format(c=c, k=k, s=s))
                r = overall(path)
                if r is not None:
                    blob = json.loads((ROOT / path / "results.json").read_text())
                    o = blob["overall"]
                    r.update(path=path, n_params=o.get("n_params"),
                             train_minutes=(o["train_seconds"] / 60.0
                                            if "train_seconds" in o else None))
                    table[(c, k, s)] = r
    return table


def operator_table():
    out = {}
    for arch, d in OPERATORS.items():
        for f in sorted((ROOT / d).glob("*_s*/results.json")):
            blob = json.loads(f.read_text())
            variant = blob["config"]["variant"]
            out.setdefault((arch, variant), []).append({
                "seed": blob["config"]["seed"], "n_params": blob.get("n_params"),
                "minutes": (blob["train_seconds"] / 60.0 if "train_seconds" in blob
                            else blob["history"][-1]["seconds"] / 60.0),
                **{f"eps_full@{h}": blob["headline"].get(f"eps_full@{h}") for h in HORIZONS},
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "v5" / "arch_compare.json"))
    args = ap.parse_args()

    # ---- PINN track ------------------------------------------------------
    table = pinn_table()
    print("PINN track: shedding retained, phase-averaged (re100_v5, data free)\n")
    print(f"{'config':10s} {'runs':>4s} " + "".join(f"{'phase ' + str(k):>10s}" for k in range(4))
          + f"{'mean':>9s} {'rel L2':>8s} {'params':>9s} {'min/run':>8s}")
    pinn = {}
    for c in MLP + NEW:
        rows = {(k, s): v for (cc, k, s), v in table.items() if cc == c}
        if not rows:
            continue
        cells = ""
        for k in range(4):
            v = [rows[(k, s)]["shedding"] for s in (0, 1) if (k, s) in rows]
            cells += f"{100 * statistics.mean(v):9.1f}%" if v else f"{'--':>10s}"
        sh = [v["shedding"] for v in rows.values()]
        l2 = [v["rel_l2"] for v in rows.values()]
        npar = next((v["n_params"] for v in rows.values() if v["n_params"]), None)
        mins = [v["train_minutes"] for v in rows.values() if v["train_minutes"]]
        pinn[c] = {"n_runs": len(rows), "shedding_mean": statistics.mean(sh),
                   "rel_l2_mean": statistics.mean(l2), "n_params": npar,
                   "train_minutes_mean": statistics.mean(mins) if mins else None}
        print(f"{c:10s} {len(rows):4d} {cells}{100 * statistics.mean(sh):8.1f}% "
              f"{statistics.mean(l2):8.4f} {npar or 0:>9,} "
              f"{(statistics.mean(mins) if mins else float('nan')):8.1f}")

    paired = {}
    print()
    for a, b in PAIRS.items():
        d = [table[(a, k, s)]["shedding"] - table[(b, k, s)]["shedding"]
             for k in range(4) for s in (0, 1) if (a, k, s) in table and (b, k, s) in table]
        if d:
            paired[f"{a} vs {b}"] = {"n": len(d), "mean_pts": 100 * statistics.mean(d),
                                     "min_pts": 100 * min(d), "max_pts": 100 * max(d)}
            print(f"  {a:10s} vs {b:9s} n={len(d)}  mean {100 * statistics.mean(d):+6.1f} pts"
                  f"  range {100 * min(d):+6.1f} .. {100 * max(d):+6.1f}")

    # ---- operator track --------------------------------------------------
    ops = operator_table()
    print("\nOperator track: rollout error eps_full@N (held-out ICs, mean over seeds)\n")
    print(f"{'arch':8s} {'variant':8s} {'seeds':>5s} "
          + "".join(f"{'@' + str(h):>9s}" for h in HORIZONS) + f" {'params':>11s} {'min':>6s}")
    op_summary = {}
    for (arch, variant), rows in sorted(ops.items()):
        means = {h: statistics.mean(r[f"eps_full@{h}"] for r in rows) for h in HORIZONS}
        npar = rows[0]["n_params"]
        op_summary[f"{arch}/{variant}"] = {"n_seeds": len(rows), "n_params": npar,
                                           **{f"eps_full@{h}": v for h, v in means.items()}}
        print(f"{arch:8s} {variant:8s} {len(rows):5d} "
              + "".join(f"{means[h]:9.4f}" for h in HORIZONS)
              + f" {npar or 0:>11,} {statistics.mean(r['minutes'] for r in rows):6.1f}")

    Path(args.json).write_text(json.dumps(
        {"pinn": pinn, "paired": paired, "operator": op_summary,
         "pinn_runs": {f"{c}_k{k}_s{s}": v for (c, k, s), v in table.items()}},
        indent=2) + "\n")
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
