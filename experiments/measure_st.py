#!/usr/bin/env python
"""Report the shedding frequency of a dataset from all three instruments.

run_re_sweep.py measures St three ways and keeps the vorticity one, because
lattice Boltzmann carries sound and a lift-based estimate can lock onto a
domain acoustic mode while the wake itself does not. That machinery is tied to
the sweep's tag convention; this exposes it for any tag, so a question like
"does this run lock, or does only its lift signal lock?" can be answered
without a sweep.

    python experiments/measure_st.py --tag confm_H50 acoustic_long

Prints, per tag:
  lift       zero-crossing-free spectral estimate of the momentum-exchange lift
  probe      single-point transverse velocity in the wake
  vorticity  wake vorticity averaged over many points -- blind to irrotational
             sound, and therefore the instrument that decides the question
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from cfd.lbm_cylinder import strouhal_spectral  # noqa: E402
from run_re_sweep import wake_strouhal  # noqa: E402


def measure(tag, datadir):
    meta_p = Path(datadir) / f"cylinder_{tag}_meta.json"
    npz_p = Path(datadir) / f"cylinder_{tag}.npz"
    meta = json.loads(meta_p.read_text())

    out = {"tag": tag, "Re": meta["Re"], "U_lattice": meta["U_lattice"],
           "H_over_D": meta["grid"][1] / meta["D_lattice"],
           "St_lift_meta": meta.get("St"), "Cd_mean": meta.get("Cd_mean"),
           "Cl_rms": meta.get("Cl_rms")}

    period = meta["D_lattice"] / (meta["U_lattice"] * meta["St_target"])
    out["warmup_periods"] = meta["warmup_steps"] / period
    out["record_periods"] = (meta["steps"] - meta["warmup_steps"]) / period

    # the transverse acoustic half-wave, in flow units
    cs = 1.0 / math.sqrt(3.0)
    out["St_resonance"] = ((cs / (2.0 * meta["grid"][1]))
                           * (meta["D_lattice"] / meta["U_lattice"]))

    if not npz_p.exists():
        out["note"] = "no .npz (run used --no-snapshots): lift only"
        return out

    with np.load(npz_p) as d:
        if "snapshots" not in d.files or d["snapshots"].shape[0] < 8:
            out["note"] = "no usable snapshots: lift only"
            return out
        cl = np.asarray(d["cl"], dtype=np.float64)
        x, y, t = np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["t"])
        snaps = np.asarray(d["snapshots"], dtype=np.float64)

    dt = float(t[1] - t[0])
    i = int(np.argmin(np.abs(x - 2.0)))
    j = int(np.argmin(np.abs(y - 0.0)))
    v = snaps[:, 1, i, j]

    first, last = np.std(v[: v.size // 3]), np.std(v[2 * v.size // 3:])
    out["probe_growth"] = float(last / max(first, 1e-12))
    out["St_vorticity"] = float(wake_strouhal(snaps, x, y, dt))
    out["St_probe"] = float(strouhal_spectral(v, dt))
    out["St_lift"] = float(strouhal_spectral(
        cl[int(meta["warmup_steps"]):], float(meta["dt_star_per_step"])))
    out["instrument_disagreement_pct"] = 100.0 * abs(
        out["St_vorticity"] - out["St_probe"]) / max(out["St_probe"], 1e-12)
    out["n_snapshots"] = int(snaps.shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--datadir", default=str(ROOT / "data"))
    ap.add_argument("--json", default=None, help="also write results here")
    args = ap.parse_args()

    rows = []
    for tag in args.tags:
        try:
            rows.append(measure(tag, args.datadir))
        except FileNotFoundError as exc:
            print(f"{tag}: {exc}")

    hdr = ("tag", "H/D", "warm", "rec", "lift", "probe", "vort", "St_res", "Cd")
    print("%-16s %5s %6s %6s %8s %8s %8s %8s %8s" % hdr)
    print("-" * 82)
    for r in rows:
        print("%-16s %5.0f %6.1f %6.1f %8s %8s %8s %8.4f %8.4f" % (
            r["tag"], r["H_over_D"], r["warmup_periods"], r["record_periods"],
            "%.4f" % r["St_lift"] if "St_lift" in r else "%.4f*" % r["St_lift_meta"],
            "%.4f" % r["St_probe"] if "St_probe" in r else "--",
            "%.4f" % r["St_vorticity"] if "St_vorticity" in r else "--",
            r["St_resonance"], r["Cd_mean"]))
    print("\n* lift value from metadata (no snapshots saved)")
    print("vorticity is the instrument that decides lock-in: it cannot see "
          "irrotational sound.")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
