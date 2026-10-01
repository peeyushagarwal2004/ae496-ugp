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

from lbm_cylinder import strouhal_estimate, strouhal_spectral  # noqa: E402

PY = sys.executable
STAGES = ["data", "base", "ablation", "accuracy"]


def run(*cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)
    if proc.returncode:
        raise SystemExit(f"step failed (exit {proc.returncode}): {cmd[1]}")
    return proc


# Re = 100 was generated before this script existed, under its own tag; reuse
# that dataset rather than spending an hour reproducing it. Applies to the
# original v1 sweep only.
TAG_OVERRIDE = {100: "re100_v4"}

# Generation settings, set from the command line in main(). The defaults
# reproduce the original v1 sweep exactly, so its committed results stay
# regenerable. The v5 sweep differs in three ways, each measured before it
# was adopted (see report/report.tex, Section 2):
#
#   upstream 25 D instead of 8 -- the dominant error. At D = 32 with the
#     absorbing layer, moving the inlet from 8 to 25 diameters ahead of the
#     cylinder took Cd from +7.4 % to +2.3 % and St from +3.0 % to +0.3 %
#     against Qu et al. (2013); raising the domain height from 50 to 120 D
#     moved Cd by 0.07 %.
#   absorbing layer at the free-slip walls -- without it the transverse duct
#     mode entrains the wake given enough time (vorticity St 0.1600 over 20
#     recorded periods, 0.1132 over 38, at Re = 100).
#   u = 0.05 at every Re -- the v1 sweep lowered the lattice velocity at
#     Re = 60 and 80 to move the resonance away, which changed the Mach
#     number between datasets. With the mode damped there is nothing to
#     dodge, so every dataset now shares one Mach number.
CFG = {"suffix": "v1", "sponge": 0.0, "sponge_target": 0.05,
       "upstream": 8.0, "length": 30.0, "ic_from_network": False,
       "run_root": "re_sweep"}


def tag_for(re):
    if CFG["suffix"] == "v1":
        return TAG_OVERRIDE.get(int(re), f"re{int(re)}_v1")
    return f"re{int(re)}_{CFG['suffix']}"


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


D_LATTICE, HEIGHT_D = 24, 50.0
CS = 1.0 / np.sqrt(3.0)  # lattice speed of sound


def acoustic_strouhal(u_lattice, height_d=HEIGHT_D, d_lattice=D_LATTICE):
    """St of the half-wave acoustic resonance across the channel, in t* units.

    LBM is weakly compressible and the free-slip walls reflect perfectly, so
    the domain supports a standing wave at c_s / 2H. Converted to the flow's
    own time scale it is (c_s / 2H)(D / U), so it moves *away* from the
    shedding frequency as the lattice velocity is lowered.
    """
    return (CS / (2.0 * height_d * d_lattice)) * (d_lattice / u_lattice)


def lattice_velocity(st_shedding, separation=0.35, u_default=0.05, u_min=0.02):
    """Largest u (so, fastest run) whose acoustic mode is clear of the shedding.

    Measured on this project's datasets: with u = 0.05 the resonance sits at
    St = 0.1155, which is 30 % below the shedding at Re = 100 (clean, measured
    -3 % against the correlation) but only 24 % below it at Re = 80 and 15 % at
    Re = 60 -- and at both of those the *wake vorticity itself*, which cannot
    see irrotational sound, oscillates at 0.113. The wake locks onto the
    resonance. Keeping them `separation` apart avoids the lock-in; the cost is
    runtime, since a smaller u needs proportionally more steps per t*.
    """
    if CFG["sponge"] > 0:
        return u_default        # the mode is damped; one Mach number for all Re
    if abs(acoustic_strouhal(u_default) - st_shedding) / st_shedding >= separation:
        return u_default
    # Push the resonance above the shedding band rather than below it: the
    # alternative (raising u) would also raise the Mach number.
    u = CS / (2.0 * HEIGHT_D) / ((1.0 + separation) * st_shedding)
    return max(round(u, 4), u_min)


def _lbm(re, tag, st, args):
    """One generation pass, with the record and the lattice velocity from `st`."""
    u = lattice_velocity(st)
    period = D_LATTICE / (st * u)  # D / (St U), in lattice steps
    wp = warmup_periods(re, args.warmup_periods)
    steps = int(round((wp + args.record_periods) * period))
    print(f"[Re {re:g}] St assumed {st:.4f}: u {u:g} (acoustic mode at St "
          f"{acoustic_strouhal(u):.4f}), warmup {wp:.0f} periods, "
          f"{steps} steps total")
    geometry = ["--length", CFG["length"], "--upstream", CFG["upstream"]]
    if CFG["sponge"] > 0:
        geometry += ["--sponge", CFG["sponge"],
                     "--sponge-target", CFG["sponge_target"]]
    run(PY, "-u", "src/cfd/lbm_cylinder.py", "--re", re,
        # One domain and resolution for every Re -- 2 % blockage, D = 24
        # lattice units -- so datasets differ only in Re (and, in the v1
        # sweep only, in the lattice velocity).
        "--u", u, "--height", "50", "--steps", steps, "--st", round(st, 5),
        "--warmup-periods", round(wp, 2), "--snaps-per-period", 32,
        *geometry, "--tag", tag)
    return json.loads((ROOT / "data" / f"cylinder_{tag}_meta.json").read_text())


def generate(re, args, tol=0.10):
    """Generate, then regenerate once if the warmup was sized from a bad guess.

    The warmup is counted in shedding periods, but the period is not known
    until the run is over. At Re = 80 the correlation over-estimated it, so
    "24 periods" of warmup was 18 real ones, the wake was still spinning up
    when recording began, and it shed at St = 0.115 against an expected 0.153.
    Sizing the second pass from the measured frequency closes that loop.
    """
    tag = tag_for(re)
    meta_path = ROOT / "data" / f"cylinder_{tag}_meta.json"
    if meta_path.exists():
        print(f"[Re {re:g}] ground truth already present, skipping generation")
        return json.loads(meta_path.read_text())

    meta = _lbm(re, tag, strouhal_estimate(re), args)
    measured = float(meta["St"])
    drift = abs(measured - strouhal_estimate(re)) / strouhal_estimate(re)
    if drift > tol:
        print(f"[Re {re:g}] measured St {measured:.4f} is {100 * drift:.0f} % "
              f"from the estimate the warmup was sized with -- regenerating "
              f"with the measured value")
        for f in (meta_path, ROOT / "data" / f"cylinder_{tag}.npz"):
            f.unlink(missing_ok=True)
        meta = _lbm(re, tag, measured, args)
    return meta


def validate(re, meta):
    """St against the empirical correlation; Cd recorded, not asserted.

    Cd at finite blockage is confined-flow drag -- at Re = 100 that cost +18 %
    before extrapolation (Phase 1), so a benchmark comparison needs the same
    blockage study per Re and is out of scope here. St is far less sensitive,
    so it is the honest one-number check that the dataset is a real wake.
    """
    sat = saturation(re)
    # Reported St is the vorticity one: the lift value in the metadata is
    # acoustically contaminated (see `saturation`).
    st_measured = float(sat["St_vorticity"])
    st_expected = strouhal_estimate(re)
    err = 100.0 * (st_measured - st_expected) / st_expected
    print(f"[Re {re:g}] St {st_measured:.4f} vs correlation {st_expected:.4f} "
          f"({err:+.1f} %)   Cd {meta['Cd_mean']:.4f} at {meta['blockage']:.1%} blockage")
    print(f"[Re {re:g}] probe r.m.s. growth {sat['probe_growth']:.2f}x   "
          f"St: vorticity {sat['St_vorticity']:.4f}, probe {sat['St_probe']:.4f} "
          f"({sat['St_instrument_disagreement_pct']:.1f} % apart), "
          f"lift {sat['St_lift']:.4f}")
    if not sat["saturated"]:
        raise SystemExit(
            f"[Re {re:g}] dataset rejected: probe growth "
            f"{sat['probe_growth']:.2f}x, vorticity and probe disagree on St by "
            f"{sat['St_instrument_disagreement_pct']:.1f} %. Nothing trained "
            "on it would be trustworthy. Delete the dataset and regenerate "
            "with a longer warmup.")
    return {"St": st_measured, "St_lift_metadata": float(meta["St"]),
            "St_correlation": st_expected,
            "St_dataset_error_pct": err,
            "Cd_mean": float(meta["Cd_mean"]), "blockage": float(meta["blockage"]),
            "n_snapshots": int(meta["n_snapshots"]), **sat}


def saturation(re, growth_max=1.15, agree_max=0.05):
    """Is the wake saturated, and do two hydrodynamic instruments agree on it?

    Three frequencies are measured and two are trusted:

    * **wake vorticity**, averaged over many points in the wake. Sound is
      irrotational, so this sees the shedding and nothing else. It is the
      reported value.
    * **wake-probe velocity** at one point -- an independent estimate that
      should match it.
    * **lift**, kept as a diagnostic only. It is a surface pressure integral
      and carries the domain's acoustic ringing: at Re = 80 it reads 0.1619
      where the vorticity says 0.1511 and the correlation 0.1528. Gating on
      the lift rejected a perfectly good dataset, which is what this
      arrangement fixes.

    Plus a growth test: the probe r.m.s. in the last third of the record
    against the first, which catches a wake still amplifying (the first
    Re = 60 dataset was growing 1.44x).
    """
    tag = tag_for(re)
    meta = json.loads((ROOT / "data" / f"cylinder_{tag}_meta.json").read_text())
    with np.load(ROOT / "data" / f"cylinder_{tag}.npz") as d:
        cl = np.asarray(d["cl"], dtype=np.float64)
        x, y, t = np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["t"])
        snaps = np.asarray(d["snapshots"], dtype=np.float64)

    dt = float(t[1] - t[0])
    i = int(np.argmin(np.abs(x - 2.0)))  # the wake probe every metric uses
    j = int(np.argmin(np.abs(y - 0.0)))
    v = snaps[:, 1, i, j]

    first, last = np.std(v[: v.size // 3]), np.std(v[2 * v.size // 3:])
    growth = float(last / max(first, 1e-12))

    st_vort = wake_strouhal(snaps, x, y, dt)
    st_probe = strouhal_spectral(v, dt)
    st_lift = strouhal_spectral(cl[int(meta["warmup_steps"]):],
                                float(meta["dt_star_per_step"]))
    agree = abs(st_vort - st_probe) / max(st_probe, 1e-12)

    return {"probe_growth": growth, "St_vorticity": st_vort,
            "St_probe": st_probe, "St_lift": st_lift,
            "St_instrument_disagreement_pct": 100.0 * agree,
            "saturated": bool(growth < growth_max and agree < agree_max)}


def wake_strouhal(snaps, x, y, dt, x_range=(1.5, 8.0), y_max=1.5):
    """Shedding frequency from vorticity averaged over the wake.

    Vorticity is blind to irrotational sound, and averaging the power spectra
    of many points rather than trusting one probe suppresses local noise. This
    is the instrument that showed the low-Re wakes were locked to the domain
    resonance while every pressure-based measure disagreed with itself.
    """
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    u, v = snaps[:, 0], snaps[:, 1]
    w = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=2)

    xs = np.where((x >= x_range[0]) & (x <= x_range[1]))[0]
    ys = np.where(np.abs(y) <= y_max)[0]
    sig = w[:, xs[:, None], ys[None, :]].reshape(w.shape[0], -1)
    sig = (sig - sig.mean(axis=0, keepdims=True)) * np.hanning(w.shape[0])[:, None]

    n = 1 << int(np.ceil(np.log2(w.shape[0] * 8)))
    power = (np.abs(np.fft.rfft(sig, n=n, axis=0)) ** 2).mean(axis=1)
    freq = np.fft.rfftfreq(n, d=dt)
    k = int(np.argmax(power[1:]) + 1)
    if k + 1 >= power.size:
        return float(freq[k])
    a, b, c = power[k - 1], power[k], power[k + 1]
    den = a - 2.0 * b + c
    shift = 0.5 * (a - c) / den if den else 0.0
    return float(freq[k] + shift * (freq[1] - freq[0]))


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
    # The period comes from the trusted instrument. meta["St"] is the lift,
    # which the v1 sweep used here and which validate() itself documents as
    # acoustically contaminated -- on a locked dataset it would size the
    # windows from the duct mode instead of the wake.
    period = 1.0 / float(meta.get("St_vorticity", meta["St"]))
    run_name = f"{CFG['run_root']}/{int(re)}"
    chain = " --ic-from-network" if CFG["ic_from_network"] else ""
    print(f"[Re {re:g}] shedding period {period:.2f} t*: "
          f"baseline 1 x {period:.3f}, all 4 x {period / 4:.3f}"
          f"{'  (chained ICs)' if chain else ''}")
    for only, window in (("baseline", period), ("all", period / 4.0)):
        run(PY, "-u", "experiments/run_ablation.py", "--tag", tag,
            "--epochs", args.epochs, "--seeds", args.seeds,
            "--only", only, "--run-name", run_name,
            "--extra", f"--sigma 8 --window-len {window:.4f} --save-params{chain}")


def accuracy(re, args):
    run(PY, "-u", "experiments/shedding_accuracy.py",
        "--run-name", f"{CFG['run_root']}/{int(re)}", "--runs", "all_s0",
        "--tag", tag_for(re))
    rows = json.loads(
        (ROOT / "runs" / CFG["run_root"] / str(int(re))
         / "shedding_accuracy.json").read_text())
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
    ap.add_argument("--suffix", default="v1",
                    help="dataset tag suffix; v1 reproduces the original sweep")
    ap.add_argument("--sponge", type=float, default=0.0,
                    help="absorbing layer at the free-slip walls, diameters")
    ap.add_argument("--sponge-target", type=float, default=0.05)
    ap.add_argument("--upstream", type=float, default=8.0,
                    help="inlet distance ahead of the cylinder, diameters")
    ap.add_argument("--length", type=float, default=30.0,
                    help="domain length, diameters")
    ap.add_argument("--ic-from-network", action="store_true",
                    help="chain the marching windows (one ground-truth slice)")
    ap.add_argument("--run-root", default="re_sweep",
                    help="results go under runs/<run-root>/<Re>")
    args = ap.parse_args()
    CFG.update(suffix=args.suffix, sponge=args.sponge,
               sponge_target=args.sponge_target, upstream=args.upstream,
               length=args.length, ic_from_network=args.ic_from_network,
               run_root=args.run_root)

    res = [float(r) for r in args.res.split(",")]
    last = STAGES.index(args.stage) if args.stage else len(STAGES) - 1
    out = ROOT / "runs" / CFG["run_root"]
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for re in res:
        print(f"\n{'=' * 70}\nRe = {re:g}\n{'=' * 70}", flush=True)
        t0 = time.time()
        row = dict(summary.get(str(int(re)), {}))

        meta = generate(re, args)
        val = validate(re, meta)
        row.update(val)
        if CFG["suffix"] != "v1":
            # Size the marching windows from the vorticity St, not the lift.
            # The v1 sweep used the lift; kept that way so it stays faithful.
            meta = {**meta, "St_vorticity": val["St_vorticity"]}
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
