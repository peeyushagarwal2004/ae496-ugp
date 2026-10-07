#!/usr/bin/env python
"""Architecture study: Transformer / RNN / CNN against the project's baselines.

Two tables, one per track, both built only from saved run artifacts.

PINN track (queue19 part A, queue20), phase-averaged exactly like
phase_average.py:

    tf, gru              single window, data free -- does the architecture
                         alone stop the collapse?  paired against `baseline`
    tf_march, gru_march  marched -- does it add anything on top of causal
                         time-marching?  paired against `march`

The MLP rows are read from the same paths phase_average.py uses, so the
pairing is by identical (phase, seed) on identical data.

Cost is wall-clock per run, read from the queue logs' "### <config> phase k/4
seed s ... <date>" headers: the MLP runs predate train_seconds in
results.json, and wall-clock (training plus evaluation) is the same measure
for every configuration.

Operator track (queue19 part B): rollout error for the FNO (runs/fno), U-Net
(runs/fno_unet) and ConvGRU (runs/fno_convgru) on re100_v4, data and physics
variants, plus the FNO on re100_v5 (runs/v5/fno). Errors use the report's
convention (experiments/verify_report.py, src_fno): mean over seeds, then the
pairwise mean of steps n-1 and n, which cancels the period-2 parity
oscillation of the two-frame map.

    python experiments/arch_compare.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from phase_average import CORE, PHASE0, overall

ROOT = Path(__file__).resolve().parents[1]
MLP = ("baseline", "march")
NEW = ("tf", "gru", "tf_march", "gru_march")
PAIRS = {"tf": "baseline", "gru": "baseline",
         "tf_march": "march", "gru_march": "march"}
OPERATORS = {"fno": "runs/fno", "unet": "runs/fno_unet", "convgru": "runs/fno_convgru",
             "fno_v5": "runs/v5/fno"}
HORIZONS = (10, 20, 30, 50)

LOGS = ("runs/v5/queue17.log", "runs/v5/queue19.log", "runs/v5/queue20.log")
STAMP = r"(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \w+ \d{4})"
HEADER = re.compile(r"^### (.*?)\s+" + STAMP)
DONE = re.compile(r"^QUEUE\d+_DONE " + STAMP)
RUN = re.compile(r"^(\S+) phase (\d)/4 seed (\d)")


def _when(stamp):
    parts = stamp.split()            # 'Sat Oct  3 13:14:34 IST 2026'
    return datetime.strptime(" ".join(parts[:4] + parts[5:]), "%a %b %d %H:%M:%S %Y")


def wall_minutes():
    """{(config, phase, seed): minutes} between consecutive queue-log headers."""
    out = {}
    for log in LOGS:
        marks = []
        for ln in (ROOT / log).read_text(errors="replace").splitlines():
            h, d = HEADER.match(ln), DONE.match(ln)
            if h:
                marks.append((h.group(1), _when(h.group(2))))
            elif d:
                marks.append((None, _when(d.group(1))))
        for (label, t0), (_, t1) in zip(marks, marks[1:]):
            r = RUN.match(label or "")
            if r:
                out[(r.group(1), int(r.group(2)), int(r.group(3)))] = (
                    (t1 - t0).total_seconds() / 60.0)
    return out


def mlp_params(args):
    """Parameter count of a run that predates n_params, rebuilt from its args.

    The Rescaled wrapper holds constants only, so the base network's count is
    the model's.
    """
    import sys

    import jax
    import jax.numpy as jnp
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "upinn")]
    from underPINN.nn.factory import build_model
    net = build_model({"type": args["net"],
                       "layers": [3] + [args["width"]] * args["depth"] + [3]})
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


_MLP = {}


def pinn_table():
    wall = wall_minutes()
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
                    npar = o.get("n_params")
                    if npar is None and blob["args"]["net"] == "mlp":
                        a = blob["args"]
                        key = (a["width"], a["depth"])
                        if key not in _MLP:
                            _MLP[key] = mlp_params(a)
                        npar = _MLP[key]
                    r.update(path=path, n_params=npar,
                             train_minutes=(o["train_seconds"] / 60.0
                                            if "train_seconds" in o else None),
                             wall_minutes=wall.get((c, k, s)))
                    table[(c, k, s)] = r
    return table


def pairwise(steps, eps, step):
    """Mean of steps n-1 and n, as the report quotes."""
    i = steps.index(step)
    return 0.5 * (eps[i - 1] + eps[i])


def operator_table():
    """{(arch, variant): seed-mean pairwise errors, per-seed values, cost}."""
    out = {}
    for arch, d in OPERATORS.items():
        runs = {}
        for f in sorted((ROOT / d).glob("*_s*/results.json")):
            blob = json.loads(f.read_text())
            runs.setdefault(blob["config"]["variant"], []).append(blob)
        for variant, blobs in runs.items():
            blobs.sort(key=lambda b: b["config"]["seed"])
            steps = blobs[0]["curve"]["steps"]
            mean = [statistics.mean(b["curve"]["eps_full"][i] for b in blobs)
                    for i in range(len(steps))]
            row = {"n_seeds": len(blobs), "tag": blobs[0]["config"]["tag"],
                   "n_params": blobs[0].get("n_params"),
                   "minutes": statistics.mean(
                       (b["train_seconds"] if "train_seconds" in b
                        else b["history"][-1]["seconds"]) / 60.0 for b in blobs)}
            for h in HORIZONS:
                row[f"eps@{h}"] = pairwise(steps, mean, h)
                row[f"eps@{h}_by_seed"] = [pairwise(steps, b["curve"]["eps_full"], h)
                                           for b in blobs]
            out[(arch, variant)] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "v5" / "arch_compare.json"))
    args = ap.parse_args()

    # ---- PINN track ------------------------------------------------------
    table = pinn_table()
    print("PINN track: shedding retained, phase-averaged (re100_v5, data free)\n")
    print(f"{'config':10s} {'runs':>4s} " + "".join(f"{'phase ' + str(k):>10s}" for k in range(4))
          + f"{'mean':>9s} {'rel L2':>8s} {'params':>9s} {'wall min':>9s}")
    pinn = {}
    for c in MLP + NEW:
        rows = {(k, s): v for (cc, k, s), v in table.items() if cc == c}
        if not rows:
            continue
        cells, by_phase = "", {}
        for k in range(4):
            v = [rows[(k, s)]["shedding"] for s in (0, 1) if (k, s) in rows]
            if v:
                by_phase[str(k)] = statistics.mean(v)
            cells += f"{100 * statistics.mean(v):9.1f}%" if v else f"{'--':>10s}"
        sh = [v["shedding"] for v in rows.values()]
        l2 = [v["rel_l2"] for v in rows.values()]
        npar = next((v["n_params"] for v in rows.values() if v["n_params"]), None)
        wall = [v["wall_minutes"] for v in rows.values() if v["wall_minutes"]]
        pinn[c] = {"n_runs": len(rows), "shedding_mean": statistics.mean(sh),
                   "shedding_min": min(sh), "shedding_max": max(sh),
                   "shedding_by_phase": by_phase,
                   "rel_l2_mean": statistics.mean(l2), "n_params": npar,
                   "wall_minutes_median": statistics.median(wall) if wall else None,
                   "n_wall": len(wall)}
        print(f"{c:10s} {len(rows):4d} {cells}{100 * statistics.mean(sh):8.1f}% "
              f"{statistics.mean(l2):8.4f} {npar or 0:>9,} "
              f"{(statistics.median(wall) if wall else float('nan')):9.1f}")

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
    # what marching adds to each network, paired by (phase, seed)
    for net, single, marched in (("mlp", "baseline", "march"), ("gru", "gru", "gru_march"),
                                 ("tf", "tf", "tf_march")):
        d = [table[(marched, k, s)]["shedding"] - table[(single, k, s)]["shedding"]
             for k in range(4) for s in (0, 1)
             if (marched, k, s) in table and (single, k, s) in table]
        if d:
            paired[f"march gain {net}"] = {"n": len(d), "mean_pts": 100 * statistics.mean(d),
                                           "min_pts": 100 * min(d), "max_pts": 100 * max(d)}
            print(f"  marching adds to {net:4s} n={len(d)}  mean {100 * statistics.mean(d):+6.1f} pts"
                  f"  range {100 * min(d):+6.1f} .. {100 * max(d):+6.1f}")

    cost = {}
    for a, b in PAIRS.items():
        wa = pinn.get(a, {}).get("wall_minutes_median")
        wb = pinn.get(b, {}).get("wall_minutes_median")
        if wa and wb:
            cost[f"{a} vs {b}"] = wa / wb
            print(f"  wall-clock {a} / {b}: {wa / wb:.1f}x")
    # what finishing the PINNsFormer grid would cost at the pilot's pace
    left = sum((8 - pinn[c]["n_runs"]) * pinn[c]["wall_minutes_median"] / 60.0
               for c in ("tf", "tf_march") if c in pinn)
    cost["pinnsformer_remaining_gpu_hours"] = left
    print(f"  PINNsFormer runs not made: {left:.0f} GPU-hours at the pilot's pace")

    # ---- operator track --------------------------------------------------
    ops = operator_table()
    print("\nOperator track: eps_full, pairwise mean of steps n-1 and n, mean over seeds\n")
    print(f"{'arch':8s} {'variant':12s} {'tag':9s}"
          + "".join(f"{'@' + str(h):>9s}" for h in HORIZONS) + f" {'params':>11s} {'min':>6s}")
    op_summary = {}
    for (arch, variant), row in sorted(ops.items()):
        op_summary[f"{arch}/{variant}"] = row
        print(f"{arch:8s} {variant:12s} {row['tag']:9s}"
              + "".join(f"{row[f'eps@{h}']:9.4f}" for h in HORIZONS)
              + f" {row['n_params'] or 0:>11,} {row['minutes']:6.1f}")
    physics_effect = {}
    for arch in OPERATORS:
        d, p = ops.get((arch, "data")), ops.get((arch, "physics"))
        if d and p:
            physics_effect[arch] = {str(h): 100.0 * (p[f"eps@{h}"] / d[f"eps@{h}"] - 1.0)
                                    for h in HORIZONS}
            print(f"  physics vs data, {arch:8s} "
                  + "  ".join(f"@{h} {v:+6.1f}%" for h, v in physics_effect[arch].items()))

    Path(args.json).write_text(json.dumps(
        {"pinn": pinn, "paired": paired, "wall_clock_ratio": cost,
         "operator": op_summary, "physics_effect_pct": physics_effect,
         "pinn_runs": {f"{c}_k{k}_s{s}": v for (c, k, s), v in table.items()}},
        indent=2) + "\n")
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
