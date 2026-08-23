"""Grid- and domain-convergence study for the LBM cylinder solver.

The production runs sit ~2 % high on St and ~9-13 % high on C_d against the
Re = 100 benchmark. Both are biased the same way, which points at systematic
discretisation error rather than a coding fault. This sweep separates the two
candidate causes by varying one at a time:

    resolution series   D = 16, 24, 32, 40   at fixed height 20 D
    blockage series     H = 20, 30, 40 D     at fixed D = 24

Runs are force-only (`--no-snapshots`), so they cost a fraction of a
production run. Results accumulate in data/convergence.json after every case,
so a partial sweep is still usable.

Usage:
    python src/cfd/convergence_sweep.py
    python src/cfd/convergence_sweep.py --only 16,24     # subset by diameter
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "src" / "cfd" / "lbm_cylinder.py"
OUT = ROOT / "data" / "convergence.json"

U = 0.08  # fixed, so Mach number is identical across the sweep
T_STAR = 140.0  # total simulated time; saturation happens by t* ~ 70
WARMUP_PERIODS = 12.0

# (diameter, height_in_diameters, series label) -- cheapest first so the trend
# shows up early even if the sweep is interrupted.
CASES = [
    (16, 20, "resolution"),
    (24, 20, "both"),  # shared baseline of the two series
    (24, 30, "blockage"),
    (32, 20, "resolution"),
    (24, 40, "blockage"),
    (40, 20, "resolution"),
]


def run_case(D, H):
    tag = f"conv_D{D}_H{H}"
    steps = int(T_STAR * D / U)
    cmd = [
        sys.executable, "-u", str(SOLVER),
        "--diameter", str(D), "--height", str(H), "--length", "25",
        "--upstream", "8", "--u", str(U), "--steps", str(steps),
        "--warmup-periods", str(WARMUP_PERIODS), "--no-snapshots",
        "--chunk", "1000", "--tag", tag,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stdout.strip().splitlines()[-6:]))
        return None
    meta = json.loads((ROOT / "data" / f"cylinder_{tag}_meta.json").read_text())
    meta["wall_time_s"] = time.time() - t0
    meta["blockage_pct"] = 100.0 / H
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated diameters to run (default: all)")
    args = ap.parse_args()
    keep = {int(v) for v in args.only.split(",")} if args.only else None

    results = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["D_lattice"], round(1 / r["blockage"])) for r in results}

    for D, H, series in CASES:
        if keep and D not in keep:
            continue
        if (D, H) in done:
            print(f"D={D:>2d} H={H:>2d}D  already done, skipping")
            continue
        print(f"D={D:>2d} H={H:>2d}D  ({series}) ...", flush=True)
        meta = run_case(D, H)
        if meta is None:
            continue
        meta["series"] = series
        results.append(meta)
        OUT.write_text(json.dumps(results, indent=2))
        print(f"  St={meta['St']:.4f}  Cd={meta['Cd_mean']:.4f}  "
              f"Cl_rms={meta['Cl_rms']:.4f}  ({meta['wall_time_s'] / 60:.1f} min)",
              flush=True)

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"{'D':>4s} {'H/D':>5s} {'blockage':>9s} {'St':>8s} {'err%':>7s} "
          f"{'Cd':>8s} {'err%':>7s} {'Cl_rms':>8s}")
    print("-" * 74)
    for r in sorted(results, key=lambda r: (r["series"], r["D_lattice"], r["blockage"])):
        st_e = 100 * (r["St"] - 0.165) / 0.165
        cd_e = 100 * (r["Cd_mean"] - 1.375) / 1.375
        print(f"{r['D_lattice']:>4d} {round(1 / r['blockage']):>5d} "
              f"{r['blockage_pct']:>8.1f}% {r['St']:>8.4f} {st_e:>+7.2f} "
              f"{r['Cd_mean']:>8.4f} {cd_e:>+7.2f} {r['Cl_rms']:>8.4f}")
    print("=" * 74)
    print("benchmark: St = 0.165, Cd = 1.37-1.38, Cl_rms ~ 0.23")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
