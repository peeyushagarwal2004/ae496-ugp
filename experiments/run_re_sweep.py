"""Does the Re = 100 result hold across the 2-D shedding range?

The Phase 4 finding -- causal time-marching restores shedding data-free -- rests
on one Reynolds number. This script repeats the essential comparison at several,
so the claim becomes one about the shedding regime rather than about one
dataset.

Per Reynolds number, in order:

  1. generate LBM ground truth            (~1-2 h on an A100, the expensive step)
  2. validate it: measured St against Williamson's correlation, Cd recorded
  3. fit the frozen base flow u0 for the Reynolds decomposition
  4. run the two configs that matter: `baseline` (the collapse) and `all`
     (the winner), 2 seeds each
  5. measure shedding accuracy -- St, amplitude, phase -- from the saved network

Every step is skipped if its output already exists, so an interrupted sweep
resumes where it stopped.

**Window length scales with the shedding period.** At Re = 100 the ablation
used 4 windows of 1.5 t*, which is one period. A period is 7.4 t* at Re = 60
and 5.2 t* at Re = 180, so a fixed 1.5 would mean different fractions of a
period at each Re and the comparison would not be like-for-like. The window is
therefore set to (1/St)/4 from the *measured* St of each dataset.

Range: 60 to 180. Below ~47 there is no shedding to predict; above ~190 the
real wake goes three-dimensional and a 2-D model is solving the wrong physics,
which is the same argument the write-up makes about the reference paper's
Re = 240-3060.

Usage:
    python experiments/run_re_sweep.py --res 60,80,120,150,180
    python experiments/run_re_sweep.py --res 150 --stage data   # generation only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "cfd"))

from lbm_cylinder import strouhal_estimate  # noqa: E402

PY = sys.executable
STAGES = ["data", "base", "ablation", "accuracy"]


def run(*cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)
    if proc.returncode:
        raise SystemExit(f"step failed (exit {proc.returncode}): {cmd[1]}")
    return proc


# Re = 100 was generated before this script existed, under its own tag; reuse
# that dataset rather than spending an hour reproducing it.
TAG_OVERRIDE = {100: "re100_v4"}


def tag_for(re):
    return TAG_OVERRIDE.get(int(re), f"re{int(re)}_v1")


def generate(re, args):
    tag = tag_for(re)
    meta = ROOT / "data" / f"cylinder_{tag}_meta.json"
    if meta.exists():
        print(f"[Re {re:g}] ground truth already present, skipping generation")
        return json.loads(meta.read_text())
    run(PY, "-u", "src/cfd/lbm_cylinder.py", "--re", re,
        # Same domain and resolution as re100_v4, so datasets differ only in Re:
        # 2 % blockage, D = 24 lattice units, Ma = 0.087.
        "--u", "0.05", "--height", "50", "--steps", args.steps,
        "--warmup-periods", 15, "--snaps-per-period", 32, "--tag", tag)
    return json.loads(meta.read_text())


def validate(re, meta):
    """St against the empirical correlation; Cd recorded, not asserted.

    Cd at finite blockage is confined-flow drag -- at Re = 100 that cost +18 %
    before extrapolation (Phase 1), so a benchmark comparison needs the same
    blockage study per Re and is out of scope here. St is far less sensitive,
    so it is the honest one-number check that the dataset is a real wake.
    """
    st_measured = float(meta["St"])
    st_expected = strouhal_estimate(re)
    err = 100.0 * (st_measured - st_expected) / st_expected
    print(f"[Re {re:g}] St {st_measured:.4f} vs correlation {st_expected:.4f} "
          f"({err:+.1f} %)   Cd {meta['Cd_mean']:.4f} at {meta['blockage']:.1%} blockage")
    return {"St": st_measured, "St_correlation": st_expected, "St_error_pct": err,
            "Cd_mean": float(meta["Cd_mean"]), "blockage": float(meta["blockage"]),
            "n_snapshots": int(meta["n_snapshots"])}


def fit_base_flow(re, args):
    tag = tag_for(re)
    if (ROOT / "data" / f"base_flow_{tag}.pkl").exists():
        print(f"[Re {re:g}] base flow already fitted, skipping")
        return
    run(PY, "-u", "src/pinn/base_flow.py", "--tag", tag,
        "--net", "fourier_mlp", "--width", 128, "--sigma", 8,
        "--epochs", args.base_epochs)


def ablate(re, meta, args):
    """`baseline` and `all`, 2 seeds, with the window scaled to this Re's period."""
    tag = tag_for(re)
    window = (1.0 / float(meta["St"])) / 4.0  # 4 windows == one shedding period
    extra = f"--sigma 8 --window-len {window:.4f} --save-params"
    print(f"[Re {re:g}] window {window:.3f} t* (period {1 / float(meta['St']):.2f})")
    run(PY, "-u", "experiments/run_ablation.py", "--tag", tag,
        "--epochs", args.epochs, "--seeds", args.seeds,
        "--only", "baseline,all", "--run-name", f"re_sweep/{int(re)}",
        "--extra", extra)


def accuracy(re, args):
    run(PY, "-u", "experiments/shedding_accuracy.py",
        "--run-name", f"re_sweep/{int(re)}", "--runs", "all_s0",
        "--tag", tag_for(re))
    rows = json.loads(
        (ROOT / "runs" / "re_sweep" / str(int(re)) / "shedding_accuracy.json").read_text())
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="60,80,120,150,180",
                    help="comma-separated Reynolds numbers (100 is already done)")
    ap.add_argument("--stage", default=None, choices=STAGES,
                    help="stop after this stage (default: run all four)")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=102000, help="LBM lattice steps")
    ap.add_argument("--base-epochs", type=int, default=25000)
    args = ap.parse_args()

    res = [float(r) for r in args.res.split(",")]
    last = STAGES.index(args.stage) if args.stage else len(STAGES) - 1
    out = ROOT / "runs" / "re_sweep"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for re in res:
        print(f"\n{'=' * 70}\nRe = {re:g}\n{'=' * 70}", flush=True)
        t0 = time.time()
        row = dict(summary.get(str(int(re)), {}))

        meta = generate(re, args)
        row.update(validate(re, meta))
        if last >= STAGES.index("base"):
            fit_base_flow(re, args)
        if last >= STAGES.index("ablation"):
            ablate(re, meta, args)
            rows = json.loads((out / str(int(re)) / "summary.json").read_text())
            for r in rows:
                row[f"{r['config']}_shedding_s{r['seed']}"] = r["shedding"]
                row[f"{r['config']}_rel_l2_s{r['seed']}"] = r["rel_l2"]
        if last >= STAGES.index("accuracy"):
            acc = accuracy(re, args)
            row.update({k: acc[k] for k in
                        ("St_pinn", "St_error_pct", "amplitude_ratio",
                         "phase_lag_periods", "phase_drift_periods",
                         "periods_covered")})

        row["minutes"] = (time.time() - t0) / 60.0
        summary[str(int(re))] = row
        summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
