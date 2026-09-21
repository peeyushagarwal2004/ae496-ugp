"""How accurately does the PINN shed -- frequency, amplitude and phase?

`shedding retained` (the ablation's metric) is an amplitude ratio: it answers
"is there a wake at all", which is the right question for the collapse, and the
wrong one for "is the prediction accurate". A network could hold the right
amplitude while shedding at the wrong rate, or drift out of phase within the
window, and the ablation table would not notice.

This script reports the three numbers that do notice, from the wake-probe
transverse velocity of a saved run against the LBM reference over the same
span:

    St          shedding frequency, from mean zero-crossing spacing (the same
                estimator `lbm_cylinder.strouhal` uses on the lift signal, and
                deliberately not an FFT peak -- see its docstring)
    amplitude   r.m.s. ratio, PINN / LBM  (= the ablation's metric)
    phase       lag of the best cross-correlation alignment, as a fraction of a
                shedding period, plus the drift between the first and second
                half of the span

Read the caveat printed with the output: a run trained over one shedding
period cannot pin `St` the way a 12-period LBM record can.

Usage:
    python experiments/shedding_accuracy.py --run-name fields_best --runs all_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "pinn"))
sys.path.insert(0, str(ROOT / "src" / "cfd"))
sys.path.insert(0, str(ROOT / "upinn"))
sys.path.insert(0, str(ROOT / "experiments"))

import cylinder_unsteady as cu  # noqa: E402
from lbm_cylinder import strouhal_spectral  # noqa: E402
from pinn.data import load  # noqa: E402
from plot_fields import probe_series, restore  # noqa: E402


def zero_crossing_strouhal(sig, dt):
    """Shedding frequency, spectral, with the crossing count for context.

    Was zero-crossing timing, which returns NaN on a span holding fewer than
    two crossings -- exactly what a one-period evaluation window gives at low
    Re. The spectral estimate (same function the LBM datasets are measured
    with) degrades gracefully instead, and the crossing count is still
    reported so a short record is visible as such.
    """
    sig = np.asarray(sig, dtype=np.float64)
    centred = sig - sig.mean()
    idx = np.where(np.diff(np.signbit(centred)))[0]
    return strouhal_spectral(sig, dt), idx.size


def phase_lag(pinn, lbm, dt, period):
    """Lag (in periods) of the cross-correlation peak, refined parabolically."""
    a = np.asarray(pinn) - np.mean(pinn)
    b = np.asarray(lbm) - np.mean(lbm)
    corr = np.correlate(a, b, mode="full")
    k = int(np.argmax(corr))
    if 0 < k < corr.size - 1:  # sub-sample peak from the parabola through 3 points
        c0, c1, c2 = corr[k - 1], corr[k], corr[k + 1]
        denom = c0 - 2 * c1 + c2
        k = k + (0.5 * (c0 - c2) / denom if denom != 0 else 0.0)
    lag_samples = k - (len(b) - 1)
    return float(lag_samples * dt / period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="fields_best")
    ap.add_argument("--runs", default="all_s0", help="comma-separated run dirs")
    ap.add_argument("--tag", default=None, help="dataset tag (default: the run's own)")
    ap.add_argument("--n-times", type=int, default=601,
                    help="probe samples across the span; the PINN is continuous in t")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = [(ROOT / "runs" / args.run_name / r if not Path(r).is_absolute() else Path(r))
            for r in args.runs.split(",")]
    tag = args.tag or json.loads((dirs[0] / "results.json").read_text())["args"]["tag"]
    ds = load(tag)

    rows = []
    for d in dirs:
        name, pde, segments, saved = restore(d, ds)
        t0 = segments[0][0]
        t1 = segments[-1][1]
        tt = np.linspace(t0, t1, args.n_times)
        dt = float(tt[1] - tt[0])

        v_pinn = probe_series(pde, segments, tt)

        # LBM at the same instants. The probe sits on a grid node, so this is
        # interpolation in time only -- no spatial interpolation error.
        i = int(np.argmin(np.abs(ds.x - cu.PROBE[0])))
        j = int(np.argmin(np.abs(ds.y - cu.PROBE[1])))
        sel = (ds.t >= t0 - 1e-9) & (ds.t <= t1 + 1e-9)
        v_lbm = np.interp(tt, ds.t[sel], ds.fields[sel, 1, i, j])

        st_pinn, n_cross_pinn = zero_crossing_strouhal(v_pinn, dt)
        st_lbm, n_cross_lbm = zero_crossing_strouhal(v_lbm, dt)
        period = 1.0 / st_lbm if np.isfinite(st_lbm) and st_lbm > 0 else np.nan

        amp = float(np.std(v_pinn) / np.std(v_lbm))
        lag = phase_lag(v_pinn, v_lbm, dt, period)
        half = len(tt) // 2
        lag_first = phase_lag(v_pinn[:half], v_lbm[:half], dt, period)
        lag_second = phase_lag(v_pinn[half:], v_lbm[half:], dt, period)

        row = {
            "run": name, "dir": str(d), "t_span": [float(t0), float(t1)],
            "periods_covered": float((t1 - t0) * st_lbm),
            "St_pinn": st_pinn, "St_lbm": st_lbm,
            "St_error_pct": float(100.0 * (st_pinn - st_lbm) / st_lbm),
            "zero_crossings_pinn": int(n_cross_pinn),
            "amplitude_ratio": amp,
            "phase_lag_periods": lag,
            "phase_drift_periods": float(lag_second - lag_first),
            "rel_l2_saved": saved.get("rel_l2_mean"),
            "shedding_saved": saved.get("shedding_ratio"),
        }
        rows.append(row)

        print(f"\n{name}   t* {t0:.2f} .. {t1:.2f}  "
              f"({row['periods_covered']:.2f} shedding periods)")
        print(f"  St            PINN {st_pinn:.4f}   LBM {st_lbm:.4f}   "
              f"({row['St_error_pct']:+.1f} %)")
        print(f"  amplitude     {100 * amp:.1f} % of LBM")
        print(f"  phase lag     {lag:+.3f} periods ({360 * lag:+.1f} deg)")
        print(f"  phase drift   {row['phase_drift_periods']:+.3f} periods "
              f"between the first and second half of the span")
        print(f"  field rel L2  {row['rel_l2_saved']:.4f}")

    print(f"\n  caveat: {rows[0]['periods_covered']:.1f} periods of signal and "
          f"{rows[0]['zero_crossings_pinn']} zero crossings. St from so short a "
          "record\n  carries roughly a percent of uncertainty per crossing -- it "
          "is a consistency\n  check, not a validated frequency like the LBM's "
          "12-period value.")

    out = Path(args.out) if args.out else dirs[0].parent / "shedding_accuracy.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
