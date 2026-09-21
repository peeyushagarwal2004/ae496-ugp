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

import numpy as np

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


RE_CRIT = 47.0  # onset of shedding for a 2-D circular cylinder


def warmup_periods(re, base):
    """Shedding periods to discard before recording, scaled towards onset.

    The instability's growth rate scales roughly with $\\Rey - \\Rey_c$, so a
    warmup that saturates the wake at $\\Rey = 100$ leaves it still growing at
    $\\Rey = 60$. Measured: the $\\Rey = 60$ record generated with a flat 15
    periods had its lift r.m.s. still rising by 1.44x across the record and its
    half-period spacing scattering by 33 % (4.9 % at $\\Rey = 150$), which
    inflated the apparent St by 25 %. The factor below gives Re = 60 about
    four times the warmup of Re = 100.
    """
    return base * max(1.0, (100.0 - RE_CRIT) / max(re - RE_CRIT, 1.0))


def generate(re, args):
    tag = tag_for(re)
    meta = ROOT / "data" / f"cylinder_{tag}_meta.json"
    if meta.exists():
        print(f"[Re {re:g}] ground truth already present, skipping generation")
        return json.loads(meta.read_text())

    # Size the run in shedding periods rather than lattice steps, so that every
    # dataset records the same number of periods whatever its frequency.
    period = 24.0 / (strouhal_estimate(re) * 0.05)  # D / (St U), lattice steps
    wp = warmup_periods(re, args.warmup_periods)
    steps = int(round((wp + args.record_periods) * period))
    print(f"[Re {re:g}] warmup {wp:.0f} periods, record {args.record_periods}, "
          f"{steps} lattice steps")
    run(PY, "-u", "src/cfd/lbm_cylinder.py", "--re", re,
        # Same domain and resolution as re100_v4, so datasets differ only in Re:
        # 2 % blockage, D = 24 lattice units, Ma = 0.087.
        "--u", "0.05", "--height", "50", "--steps", steps,
        "--warmup-periods", round(wp, 2), "--snaps-per-period", 32, "--tag", tag)
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
    sat = saturation(re)
    print(f"[Re {re:g}] St {st_measured:.4f} vs correlation {st_expected:.4f} "
          f"({err:+.1f} %)   Cd {meta['Cd_mean']:.4f} at {meta['blockage']:.1%} blockage")
    print(f"[Re {re:g}] saturation: Cl rms growth {sat['cl_growth']:.2f}x, "
          f"half-period spread {sat['period_spread_pct']:.1f} %")
    if not sat["saturated"]:
        raise SystemExit(
            f"[Re {re:g}] the wake has not saturated (growth "
            f"{sat['cl_growth']:.2f}x, spread {sat['period_spread_pct']:.1f} %). "
            "Its St is not trustworthy and nothing trained on it would be "
            "either. Delete the dataset and regenerate with a longer warmup.")
    return {"St": st_measured, "St_correlation": st_expected,
            "St_dataset_error_pct": err,
            "Cd_mean": float(meta["Cd_mean"]), "blockage": float(meta["blockage"]),
            "n_snapshots": int(meta["n_snapshots"]), **sat}


def saturation(re, growth_max=1.15, spread_max=0.20):
    """Has the wake stopped growing, and is it periodic?

    Guards the failure that produced the first Re = 60 dataset: near onset the
    wake is still amplifying when recording starts, and a zero-crossing
    frequency taken from a growing, irregular signal is meaningless (it read
    25 % high). Both tests are on the recorded lift signal alone.
    """
    tag = tag_for(re)
    meta = json.loads((ROOT / "data" / f"cylinder_{tag}_meta.json").read_text())
    with np.load(ROOT / "data" / f"cylinder_{tag}.npz") as d:
        cl = np.asarray(d["cl"], dtype=np.float64)
    rec = cl[int(meta["warmup_steps"]):]
    n = rec.size
    first, last = np.std(rec[: n // 3]), np.std(rec[2 * n // 3:])
    growth = float(last / max(first, 1e-12))

    sig = rec - rec.mean()
    idx = np.where(np.diff(np.signbit(sig)))[0]
    t0 = idx + sig[idx] / (sig[idx] - sig[idx + 1])
    half = np.diff(t0)
    spread = float(half.std() / half.mean()) if half.size > 2 else 1.0

    return {"cl_growth": growth, "period_spread_pct": 100.0 * spread,
            "saturated": bool(growth < growth_max and spread < spread_max)}


def fit_base_flow(re, args):
    tag = tag_for(re)
    if (ROOT / "data" / f"base_flow_{tag}.pkl").exists():
        print(f"[Re {re:g}] base flow already fitted, skipping")
        return
    run(PY, "-u", "src/pinn/base_flow.py", "--tag", tag,
        "--net", "fourier_mlp", "--width", 128, "--sigma", 8,
        "--epochs", args.base_epochs)


def ablate(re, meta, args):
    """`baseline` and `all`, 2 seeds, both spanning exactly one shedding period.

    The two configurations need *different* window lengths to cover the same
    span: `baseline` trains one window, `all` trains four. Passing a single
    `--window-len` to both -- as this script first did -- gave `baseline` a
    quarter period, over which the initial condition alone carries the solution
    and the collapse has no time to happen. It scored 55-158 % "shedding
    retained" against 10.6 % at Re = 100, which is an artefact of the span, not
    a property of the flow.
    """
    tag = tag_for(re)
    period = 1.0 / float(meta["St"])
    run_name = f"re_sweep/{int(re)}"
    print(f"[Re {re:g}] shedding period {period:.2f} t*: "
          f"baseline 1 x {period:.3f}, all 4 x {period / 4:.3f}")
    for only, window in (("baseline", period), ("all", period / 4.0)):
        run(PY, "-u", "experiments/run_ablation.py", "--tag", tag,
            "--epochs", args.epochs, "--seeds", args.seeds,
            "--only", only, "--run-name", run_name,
            "--extra", f"--sigma 8 --window-len {window:.4f} --save-params")


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
    ap.add_argument("--warmup-periods", type=float, default=15.0,
                    help="base warmup, scaled towards onset by warmup_periods()")
    ap.add_argument("--record-periods", type=float, default=15.0,
                    help="shedding periods to record after warmup")
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
            # Prefixed: `shedding_accuracy` reports its own `St_error_pct`
            # (network versus LBM), which silently overwrote the dataset's
            # validation error against the correlation when both shared a key.
            row.update({f"pinn_{k}": acc[k] for k in
                        ("St_pinn", "St_error_pct", "amplitude_ratio",
                         "phase_lag_periods", "phase_drift_periods",
                         "periods_covered")})

        row["minutes"] = (time.time() - t0) / 60.0
        summary[str(int(re))] = row
        summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
