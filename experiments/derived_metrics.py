#!/usr/bin/env python
"""Emit the report numbers that were previously only computed by hand.

Several quantities quoted in report.tex existed nowhere on disk: the FNO
crossover step, the parity oscillation, the physics improvement ratios, the
steps-to-periods conversion, the acoustic resonance frequency, and the
divergence error of the ground-truth dataset. Each was read off a figure or
worked out interactively, which means nothing could check them and nothing
could regenerate them when an upstream artifact changed.

This script recomputes all of them from artifacts that are already committed
and writes runs/derived_metrics.json, so experiments/verify_report.py can
source them like any other number.

    python experiments/derived_metrics.py                  # everything
    python experiments/derived_metrics.py --skip-dataset   # no .npz needed

The dataset section needs data/cylinder_re100_v4.npz (190 MB), which is not in
the repository; --skip-dataset leaves those keys absent and the checker will
report them as errors rather than silently passing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FNO_DIR = ROOT / "runs" / "fno"
OUT = ROOT / "runs" / "derived_metrics.json"

VARIANTS = ("data", "physics", "physics_div")
SEEDS = (0, 1)


# --------------------------------------------------------------------------
# FNO rollout curves
# --------------------------------------------------------------------------

def _curve(variant, seed, metric="eps_full"):
    blob = json.loads((FNO_DIR / f"{variant}_s{seed}" / "results.json").read_text())
    c = blob["curve"]
    return np.asarray(c["steps"]), np.asarray(c[metric], dtype=float)


def _pairwise(eps):
    """The bold line in plot_rollout.py: mean of steps (n-1, n), placed at n.

    Returned aligned to steps[1:], so index i corresponds to steps[i + 1].
    """
    return np.convolve(eps, [0.5, 0.5], mode="valid")


def _seed_mean(variant, metric="eps_full"):
    steps = None
    stack = []
    for s in SEEDS:
        st, e = _curve(variant, s, metric)
        steps = st if steps is None else steps
        stack.append(e)
    return steps, np.mean(stack, axis=0)


def crossover(a="physics", b="data", metric="eps_full"):
    """First step at which variant `a` stops being better than `b`.

    Evaluated on the parity-smoothed curve, because on the raw curve the
    answer depends on which parity the step lands on -- which is the whole
    point of the parity note in the report.
    """
    out = {}
    for seed in SEEDS:
        steps, ea = _curve(a, seed, metric)
        _, eb = _curve(b, seed, metric)
        sa, sb = _pairwise(ea), _pairwise(eb)
        at = steps[1:]
        worse = np.where(sa > sb)[0]
        out[f"seed{seed}"] = int(at[worse[0]]) if worse.size else None
    steps, ea = _seed_mean(a, metric)
    _, eb = _seed_mean(b, metric)
    sa, sb = _pairwise(ea), _pairwise(eb)
    worse = np.where(sa > sb)[0]
    out["seed_mean"] = int(steps[1:][worse[0]]) if worse.size else None
    vals = [v for k, v in out.items() if k.startswith("seed") and k != "seed_mean"
            and v is not None]
    out["min"], out["max"] = (min(vals), max(vals)) if vals else (None, None)
    return out


def parity_oscillation(metric="eps_full"):
    """How much the error alternates between even and odd steps.

    For each consecutive pair the relative jump is
    |e[n+1] - e[n]| / min(e[n+1], e[n]); the median over the rollout is
    reported per variant and seed. The median rather than the max, because a
    single late-horizon blow-up would otherwise dominate.
    """
    out = {}
    for v in VARIANTS:
        per_seed = []
        for s in SEEDS:
            _, e = _curve(v, s, metric)
            jump = np.abs(np.diff(e)) / np.minimum(e[1:], e[:-1])
            per_seed.append(float(np.median(jump) * 100.0))
        out[v] = {"seed0_pct": per_seed[0], "seed1_pct": per_seed[1],
                  "mean_pct": float(np.mean(per_seed))}
    allv = [out[v]["mean_pct"] for v in VARIANTS]
    out["min_pct"], out["max_pct"] = float(min(allv)), float(max(allv))
    return out


def improvement(step, a="physics", b="data", metric="eps_full"):
    """Percentage by which `a` beats `b` at `step`, on the smoothed curve."""
    steps, ea = _seed_mean(a, metric)
    _, eb = _seed_mean(b, metric)
    i = int(np.where(steps == step)[0][0])
    va, vb = _pairwise(ea)[i - 1], _pairwise(eb)[i - 1]
    return float((1.0 - va / vb) * 100.0)


def steps_to_periods(meta, st_key, steps=(10, 20, 30, 50)):
    """Forecast steps expressed in shedding periods: n * dt_star * St."""
    dt = meta["snapshot_dt_star"]
    st = meta[st_key] if st_key in meta else st_key
    return {str(n): float(n * dt * st) for n in steps}


def fft_bin_width(meta):
    """Candidate readings of the report's "FFT bin width (0.014)".

    NONE OF THESE REPRODUCES 0.014, which is why they are all listed. The
    honest resolutions are either to recompute the figure from whichever
    record the FFT was actually run on and state it, or to drop the number and
    keep the argument (a zero-crossing estimator resolves the period far more
    finely than an FFT bin over this record length), which does not depend on
    the exact value.
    """
    dt = meta["snapshot_dt_star"]
    n = meta["n_snapshots"]
    st = meta["St"]
    span = n * dt
    periods = span / (1.0 / st)
    out = {
        "snapshot_record": {
            "n_snapshots": n, "dt_star": dt, "span_t_star": span,
            "periods_in_record": periods,
            "bin_width_St": 1.0 / span,
        },
        "force_record": {
            "n_steps": meta.get("steps"),
            "dt_star_per_step": meta.get("dt_star_per_step"),
            "bin_width_St": (1.0 / (meta["steps"] * meta["dt_star_per_step"])
                             if meta.get("steps") and
                             meta.get("dt_star_per_step") else None),
        },
        "st_over_12_periods": st / 12.0,
        "st_over_record_periods": st / periods,
        "_note": "1/span = %.5f; St/12 = %.5f; St/periods = %.5f. The report "
                 "prints 0.014, closest to St/12, but the saved record holds "
                 "%.1f periods, not 12." % (1.0 / span, st / 12.0,
                                            st / periods, periods),
    }
    return out


# --------------------------------------------------------------------------
# Acoustic resonance
# --------------------------------------------------------------------------

def acoustic_table(sweep):
    """The resonance separation per Reynolds number, both ways of measuring it.

    Two different normalisations are in play and they must not be confused.

    The report's "separation from the resonance" is
        (St_natural - St_res) / St_natural,
    i.e. the gap measured as a fraction of the SHEDDING frequency. That is the
    definition that reproduces the printed 15 / 24 / 30 / 33-40 %.

    The acoustic-resonance literature (Hong et al., JFM 884 A42) instead
    quantifies lock-in by the RATIO f_s / f_a, with a beta-mode dominant band
    out to 1.14. Both are reported here so the comparison in the report is
    made in the literature's own units rather than in ours.

    St_natural is Williamson's correlation, not the measured value, because at
    the Reynolds numbers that locked, the measured value IS the locked one --
    using it would define the anomaly away.
    """
    cs = 1.0 / math.sqrt(3.0)
    rows = {}
    for re_key, rec in sweep.items():
        meta = _sweep_meta(re_key)
        if meta is None:
            continue
        H, D = meta["grid"][1], meta["D_lattice"]
        u_now = meta["U_lattice"]
        st_nat = rec["St_correlation"]
        entry = {"U_lattice": u_now, "H_lattice": H, "D_lattice": D,
                 "St_correlation": st_nat}
        for label, u in (("as_built", u_now), ("at_u005", 0.05)):
            st_res = (cs / (2.0 * H)) * (D / u)
            entry[label] = {
                "U_lattice": u,
                "St_resonance": float(st_res),
                "separation_pct": float((st_nat - st_res) / st_nat * 100.0),
                "ratio_fs_over_fa": float(st_nat / st_res),
            }
        rows[re_key] = entry
    return rows


# Qu, Norberg, Davidson, Peng and Wang (2013), J. Fluids Struct. 39, 347-370,
# Table 3: two-dimensional results at large domains (H = 100-200). Transcribed
# here so the report's benchmark column is a checkable artifact rather than a
# remembered figure, and so the whole sweep can be scored against it rather
# than only Re = 100. Their Table 1 compiles eight independent studies at
# Re = 100 with Cd from 1.318 to 1.336, which is the range the report quotes.
QU2013_TABLE3 = {
    "50":  {"H": 200, "Cd": 1.397, "Cl_rms": 0.0399, "St": 0.1240},
    "60":  {"H": 160, "Cd": 1.377, "Cl_rms": 0.0871, "St": 0.1354},
    "80":  {"H": 160, "Cd": 1.336, "Cl_rms": 0.1616, "St": 0.1528},
    "100": {"H": 120, "Cd": 1.317, "Cl_rms": 0.2224, "St": 0.1649},
    "120": {"H": 120, "Cd": 1.306, "Cl_rms": 0.2776, "St": 0.1739},
    "150": {"H": 160, "Cd": 1.301, "Cl_rms": 0.3529, "St": 0.1837},
    "180": {"H": 100, "Cd": 1.310, "Cl_rms": 0.4251, "St": 0.1919},
    "200": {"H": 100, "Cd": 1.316, "Cl_rms": 0.4678, "St": 0.1958},
}
QU2013_RE100_CD_RANGE = (1.318, 1.336)
RE_C_3D = 188.5          # Barkley and Henderson (1996), JFM 322, 215-241


def benchmark_comparison(sweep):
    """Score every sweep dataset against Qu et al. (2013) Table 3."""
    rows, cd_over, st_err = {}, {}, {}
    for re_key, rec in sweep.items():
        ref = QU2013_TABLE3.get(re_key)
        if ref is None:
            continue
        cd = rec.get("Cd_mean")
        st = rec.get("St_vorticity", rec.get("St"))
        row = {"reference": ref}
        if cd is not None:
            row["Cd"] = cd
            row["Cd_error_pct"] = float((cd / ref["Cd"] - 1.0) * 100.0)
            cd_over[re_key] = row["Cd_error_pct"]
        if st is not None:
            row["St"] = st
            row["St_error_pct"] = float((st / ref["St"] - 1.0) * 100.0)
            st_err[re_key] = row["St_error_pct"]
        rows[re_key] = row
    out = {"qu2013": QU2013_TABLE3,
           "qu2013_Re100_Cd_range": list(QU2013_RE100_CD_RANGE),
           "Re_c_3d": RE_C_3D,
           "per_Re": rows}
    if cd_over:
        out["cd_overshoot_pct"] = {"min": float(min(cd_over.values())),
                                   "max": float(max(cd_over.values())),
                                   "per_Re": cd_over}
    if st_err:
        out["st_abs_error_pct"] = {
            "max": float(max(abs(v) for v in st_err.values())),
            "per_Re": st_err}
    return out


SWEEP_SUFFIXES = ["v1", "v4"]   # set from --suffix in main()


def _sweep_meta(re_key):
    """Metadata for one sweep dataset, by Reynolds number label."""
    for name in [f"cylinder_re{re_key}_{sfx}_meta.json"
                 for sfx in SWEEP_SUFFIXES]:
        p = ROOT / "data" / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def acoustic_resonance(meta):
    """Strouhal number of the transverse standing half-wave, in flow units.

    The free-slip lateral walls reflect perfectly, so the domain supports a
    standing wave across its height H at frequency c_s / 2H in lattice units,
    with c_s = 1/sqrt(3) for the D2Q9 lattice. Converting to flow units with
    the cylinder diameter and the lattice velocity gives

        St_res = (c_s / 2H) * (D / U).

    Lowering U raises St_res, which is why regenerating the low-Re datasets at
    a smaller lattice velocity moves the resonance out of the shedding band
    rather than merely weakening it.
    """
    grid = meta["grid"]
    H = grid[1]                       # lattice cells across the domain height
    D = meta["D_lattice"]
    U = meta["U_lattice"]
    cs = 1.0 / math.sqrt(3.0)
    return {
        "H_lattice": H, "D_lattice": D, "U_lattice": U,
        "c_s": cs,
        "St_resonance": float((cs / (2.0 * H)) * (D / U)),
        "blockage_check": float(D / H),
    }


# --------------------------------------------------------------------------
# Dataset divergence
# --------------------------------------------------------------------------

def divergence_error(npz_path, halo=2, edge=2):
    """Residual divergence of the saved field, relative to |grad u|.

    Lattice Boltzmann is weakly compressible, so the recovered velocity is not
    exactly solenoidal and the residual divergence is expected to scale as
    O(Ma^2).

    CAUTION, and the reason the halo sweep below is reported alongside the
    headline number: this ratio is not a well-defined property of the dataset
    until BOTH the mask and the norm are fixed. Centred differences next to the
    cylinder reach across the wall, so the divergence there is a stencil
    artefact rather than compressibility. Excluding more of that halo drops the
    ratio by an order of magnitude -- roughly 4.1 % at halo 0, 1.3 % at halo 2,
    0.6 % at halo 4, 0.4 % at halo 8 -- and switching from an r.m.s. ratio to a
    mean-absolute ratio moves it again by a factor of about 2.5.

    The headline uses `halo`/`edge` = 2/2, which is this project's own
    convention from src/fno/physics.py:interior_mask, and reports both norms.
    Any number quoted in the report must name the convention it used.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from fno.physics import interior_mask

    with np.load(npz_path) as d:
        snaps = d["snapshots"]            # (t, field, nx, ny), field = u, v, p
        x, y = d["x"], d["y"]
        solid = d["solid"]
        n_t = snaps.shape[0]

        halos = sorted({0, 2, 4, 6, 8, halo})
        masks = {h: interior_mask(solid, halo=h, edge=edge).astype(bool)
                 for h in halos}
        acc = {h: dict(d2=0.0, g2=0.0, ad=0.0, ag=0.0, n=0) for h in halos}

        # Accumulated in time chunks: the full gradient stack for 641
        # snapshots would be several hundred megabytes at once.
        for lo in range(0, n_t, 64):
            blk = np.asarray(snaps[lo:lo + 64], dtype=np.float64)
            u, v = blk[:, 0], blk[:, 1]
            ux = np.gradient(u, x, axis=1)
            uy = np.gradient(u, y, axis=2)
            vx = np.gradient(v, x, axis=1)
            vy = np.gradient(v, y, axis=2)
            div = ux + vy
            gmag = np.sqrt(ux**2 + uy**2 + vx**2 + vy**2)
            for h, m in masks.items():
                dm, gm = div[:, m], gmag[:, m]
                a = acc[h]
                a["d2"] += float(np.sum(dm**2))
                a["g2"] += float(np.sum(gm**2))
                a["ad"] += float(np.sum(np.abs(dm)))
                a["ag"] += float(np.sum(gm))
                a["n"] += dm.size

    def ratios(h):
        a = acc[h]
        rms = math.sqrt(a["d2"] / a["n"]) / math.sqrt(a["g2"] / a["n"]) * 100.0
        mab = (a["ad"] / a["n"]) / (a["ag"] / a["n"]) * 100.0
        return {"rms_ratio_pct": rms, "mean_abs_ratio_pct": mab,
                "div_rms": math.sqrt(a["d2"] / a["n"]),
                "grad_rms": math.sqrt(a["g2"] / a["n"]),
                "n_points_kept": int(masks[h].sum())}

    head = ratios(halo)
    return {
        "ma_squared_pct": None,            # filled by the caller, which has meta
        "headline_halo": halo,
        "headline_edge": edge,
        "div_over_grad_pct": head["rms_ratio_pct"],
        "div_over_grad_mean_abs_pct": head["mean_abs_ratio_pct"],
        "div_rms": head["div_rms"],
        "grad_rms": head["grad_rms"],
        "n_snapshots": int(n_t),
        "n_points_kept": head["n_points_kept"],
        "halo_sensitivity": {str(h): ratios(h) for h in halos},
        "_definition": "r.m.s.(du/dx + dv/dy) / r.m.s.(|grad u|) over "
                       "src/fno/physics.py:interior_mask(halo, edge); "
                       "|grad u| = sqrt(ux^2 + uy^2 + vx^2 + vy^2). The "
                       "mean-absolute variant is given alongside because the "
                       "report does not say which norm it used, and the two "
                       "differ by about a factor of 2.5.",
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dataset", action="store_true",
                    help="do not open the .npz (which is not in the repo)")
    ap.add_argument("--tag", default="re100_v4")
    ap.add_argument("--fno-dir", default="runs/fno",
                    help="FNO runs to analyse, relative to the repo root")
    ap.add_argument("--sweep-dir", default="runs/re_sweep",
                    help="Reynolds sweep whose summary.json to read")
    ap.add_argument("--suffix", default=None,
                    help="dataset suffix of the sweep (default: v1 then v4)")
    ap.add_argument("--out", default="runs/derived_metrics.json")
    args = ap.parse_args()

    global FNO_DIR, OUT
    FNO_DIR = ROOT / args.fno_dir
    OUT = ROOT / args.out
    if args.suffix:
        SWEEP_SUFFIXES[:] = [args.suffix]
    sweep_path = ROOT / args.sweep_dir / "summary.json"
    sweep_doc = json.loads(sweep_path.read_text()) if sweep_path.exists() else {}
    # The vorticity St of the Re = 100 dataset, from the sweep's own
    # validation, rather than the v4 value that used to be written in here.
    st_vort_100 = float(sweep_doc.get("100", {}).get("St_vorticity",
                                                     0.15947208294485055))

    meta = json.loads(
        (ROOT / "data" / f"cylinder_{args.tag}_meta.json").read_text())

    out = {
        "_about": "Generated by experiments/derived_metrics.py. Numbers here "
                  "were previously computed by hand or read off a figure; "
                  "report/numbers.json sources them from this file.",
        "acoustic": acoustic_resonance(meta),
        "fft_bin_width": fft_bin_width(meta),
    }

    # The FNO section needs every variant and seed of the rollout runs. Before
    # those exist -- the regeneration queue runs the FNO last -- skip it rather
    # than fail, so the dataset and sweep numbers can be produced first.
    if all((FNO_DIR / f"{v}_s{s_}" / "results.json").exists()
           for v in VARIANTS for s_ in SEEDS):
        out["fno"] = {
                "crossover_physics_vs_data": crossover("physics", "data"),
                "crossover_physics_div_vs_data": crossover("physics_div", "data"),
                "crossover_all_variants": {},   # filled in below
                "parity_oscillation": parity_oscillation(),
                "improvement_pct": {
                    "step10": improvement(10),
                    "step20": improvement(20),
                    "step10_div_only": improvement(10, a="physics_div"),
                    "step20_div_only": improvement(20, a="physics_div"),
                },
                "periods_per_step_St_lift": steps_to_periods(meta, "St"),
                "periods_per_step_St_vorticity": steps_to_periods(
                    meta, st_vort_100),
                "data_residuals": json.loads(
                    (FNO_DIR / "data_s0" / "results.json").read_text())["data_residuals"],
            }

        # The report quotes a crossover range "across seeds and variants", so
        # the range has to be taken over both physics variants together.
        both = [out["fno"]["crossover_physics_vs_data"],
                out["fno"]["crossover_physics_div_vs_data"]]
        vals = [v for c in both for k, v in c.items()
                if k.startswith("seed") and k != "seed_mean" and v is not None]
        out["fno"]["crossover_all_variants"] = {
            "min": min(vals), "max": max(vals), "n_curves": len(vals)}
    else:
        print(f"skipping FNO section: no complete set of runs in {FNO_DIR}")

    if sweep_doc:
        s = sweep_doc
        dis = {k: v["St_instrument_disagreement_pct"]
               for k, v in s.items() if "St_instrument_disagreement_pct" in v}
        out["acoustic"]["instrument_disagreement_pct"] = {
            "per_Re": dis,
            "max": float(max(dis.values())),
            "median": float(np.median(list(dis.values()))),
        }
        table = acoustic_table(s)
        if table:
            out["acoustic"]["per_Re"] = table
            clean = [v["at_u005"]["separation_pct"]
                     for k, v in table.items() if k in ("120", "150", "180")]
            if clean:
                out["acoustic"]["clean_band_separation_pct"] = {
                    "min": float(min(clean)), "max": float(max(clean))}
        out["benchmark"] = benchmark_comparison(s)

    npz = ROOT / "data" / f"cylinder_{args.tag}.npz"
    if args.skip_dataset:
        print(f"skipping dataset section (--skip-dataset)")
    elif not npz.exists():
        print(f"skipping dataset section: {npz.name} not present")
    else:
        print(f"reading {npz.name} ...")
        out["dataset"] = divergence_error(npz)
        out["dataset"]["ma_squared_pct"] = float(meta["Ma"] ** 2 * 100.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    try:
        shown = OUT.relative_to(ROOT)
    except ValueError:
        shown = OUT
    print(f"wrote {shown}")

    f = out.get("fno")
    if f:
        print(f"\n  crossover (physics vs data): seeds "
              f"{f['crossover_physics_vs_data']['min']}-"
              f"{f['crossover_physics_vs_data']['max']}, "
              f"seed-mean {f['crossover_physics_vs_data']['seed_mean']}")
        print(f"  parity oscillation: "
              f"{f['parity_oscillation']['min_pct']:.1f}-"
              f"{f['parity_oscillation']['max_pct']:.1f} %")
        print(f"  improvement: step 10 {f['improvement_pct']['step10']:.1f} %, "
              f"step 20 {f['improvement_pct']['step20']:.1f} %")
        print(f"  periods at step 10: "
              f"{f['periods_per_step_St_lift']['10']:.4f} (lift St), "
              f"{f['periods_per_step_St_vorticity']['10']:.4f} (vorticity St)")
    print(f"  acoustic resonance St: {out['acoustic']['St_resonance']:.5f}")
    if "dataset" in out:
        print(f"  divergence error: "
              f"{out['dataset']['div_over_grad_pct']:.3f} % of |grad u|")


if __name__ == "__main__":
    main()
