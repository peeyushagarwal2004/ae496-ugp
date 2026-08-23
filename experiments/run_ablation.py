"""Phase 3 + 4 ablation: which mechanism makes a PINN shed?

Runs the experiment matrix and prints one table. The question each row answers
is not "how small is the loss" but "did the wake survive" -- reported as
shedding retained, the ratio of wake-probe transverse-velocity r.m.s. between
the PINN and the LBM reference.

    baseline        plain MLP, data free, one window
                    -> expected to collapse to steady flow (the documented
                       failure this project is built around)
    +fourier        Fourier features, attacking spectral bias
    +decompose      learn u' on a frozen mean flow (the reference paper's trick)
    +march          causal time-marching over short windows
    all             the three combined
    +data           sparse supervised points as well (the realistic winner)

Usage (GPU):
    python experiments/run_ablation.py --tag re100_v4 --epochs 20000
    python experiments/run_ablation.py --quick          # small, for a first check
    python experiments/run_ablation.py --only baseline,all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "pinn" / "cylinder_unsteady.py"

# name -> extra CLI flags. Everything else is held fixed across the matrix.
MATRIX = {
    "baseline": [],
    "fourier": ["--net", "fourier_mlp"],
    "decompose": ["--decompose"],
    "march": ["--windows", "4", "--window-len", "1.5"],
    "all": ["--decompose", "--net", "fourier_mlp", "--windows", "4",
            "--window-len", "1.5"],
    "all+data": ["--decompose", "--net", "fourier_mlp", "--windows", "4",
                 "--window-len", "1.5", "--n-data", "4000"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="re100_v4")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat each config with this many seeds")
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--quick", action="store_true",
                    help="short runs, for checking the GPU path works")
    ap.add_argument("--extra", default="", help="extra flags passed to every run")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2000

    names = args.only.split(",") if args.only else list(MATRIX)
    unknown = [n for n in names if n not in MATRIX]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}; choose from {list(MATRIX)}")

    outdir = ROOT / "runs" / "ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    done = {(r["config"], r["seed"]) for r in summary}

    for name in names:
        for seed in range(args.seeds):
            if (name, seed) in done:
                print(f"[{name} seed {seed}] already done, skipping")
                continue
            run_dir = outdir / f"{name}_s{seed}"
            cmd = [sys.executable, "-u", str(TRAINER),
                   "--tag", args.tag, "--epochs", str(args.epochs),
                   "--seed", str(seed), "--out", str(run_dir)]
            cmd += MATRIX[name]
            if args.extra:
                cmd += args.extra.split()

            print(f"\n{'=' * 70}\n[{name} seed {seed}] {' '.join(MATRIX[name]) or '(no flags)'}"
                  f"\n{'=' * 70}", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=ROOT)
            if proc.returncode != 0:
                print(f"[{name} seed {seed}] FAILED (exit {proc.returncode})")
                continue

            res = json.loads((run_dir / "results.json").read_text())["results"]
            if not res:
                print(f"[{name} seed {seed}] produced no windows")
                continue
            summary.append({
                "config": name, "seed": seed,
                "shedding": float(sum(r["shedding_ratio"] for r in res) / len(res)),
                "rel_l2": float(sum(r["rel_l2_mean"] for r in res) / len(res)),
                "windows": len(res),
                "minutes": (time.time() - t0) / 60.0,
            })
            summary_path.write_text(json.dumps(summary, indent=2))

    # ---- table -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'config':<12s} {'seed':>4s} {'shedding retained':>18s} {'rel L2':>9s} {'min':>7s}")
    print("-" * 70)
    for name in names:
        for r in [r for r in summary if r["config"] == name]:
            print(f"{r['config']:<12s} {r['seed']:>4d} {100 * r['shedding']:>17.1f}% "
                  f"{r['rel_l2']:>9.4f} {r['minutes']:>7.1f}")
    print("=" * 70)
    print("shedding retained ~ 0 %  : collapsed to the steady base flow")
    print("shedding retained ~ 100 %: wake amplitude matches the LBM reference")
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
