#!/usr/bin/env python
"""Where does a dataset's residual divergence come from?

Lattice Boltzmann is weakly compressible, so div u is never zero. Two sources
are expected and they mean different things:

* hydrodynamic: the shedding's own pressure fluctuations, delta rho ~ Ma^2 Cp,
  which oscillate at the shedding frequency and are concentrated in the near
  wake. Harmless -- it is what a weakly compressible solver is supposed to
  produce.
* acoustic: sound travelling through the domain, at whatever frequencies the
  domain's resonances select, spread across the whole field. This is the
  thing the absorbing layer exists to remove, and if it survives in the
  interior the dataset is not as clean as its Strouhal number suggests.

The two separate in frequency. This takes the divergence at a set of points
across the crop, Fourier-transforms each in time, and reports where the power
sits relative to the shedding frequency and to the domain's candidate
acoustic modes.

    python experiments/divergence_spectrum.py --tag re100_v5 re100_v4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PROBES = {           # (x/D, y/D): near wake, far wake, off-axis, upstream
    "near wake (2, 0)": (2.0, 0.0),
    "far wake (12, 0)": (12.0, 0.0),
    "off-axis (5, 4)": (5.0, 4.0),
    "upstream (-4, 0)": (-4.0, 0.0),
    "side (0, 4.5)": (0.0, 4.5),
}


def analyse(tag):
    meta = json.loads((ROOT / "data" / f"cylinder_{tag}_meta.json").read_text())
    with np.load(ROOT / "data" / f"cylinder_{tag}.npz") as d:
        snaps = np.asarray(d["snapshots"], dtype=np.float64)
        x, y, t = np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["t"])

    u, v = snaps[:, 0], snaps[:, 1]
    div = np.gradient(u, x, axis=1) + np.gradient(v, y, axis=2)
    dt = float(t[1] - t[0])

    D, U = meta["D_lattice"], meta["U_lattice"]
    cs = 1.0 / math.sqrt(3.0)
    nx, ny = meta["grid"]
    modes = {
        "transverse c_s/2H": (cs / (2.0 * ny)) * (D / U),
        "streamwise c_s/2L": (cs / (2.0 * nx)) * (D / U),
    }

    freq = np.fft.rfftfreq(div.shape[0], d=dt)
    out = {"tag": tag, "modes": modes, "probes": {}}
    for name, (px, py) in PROBES.items():
        i = int(np.argmin(np.abs(x - px)))
        j = int(np.argmin(np.abs(y - py)))
        sig = div[:, i, j] - div[:, i, j].mean()
        p = np.abs(np.fft.rfft(sig * np.hanning(sig.size))) ** 2
        p[0] = 0.0
        k = int(np.argmax(p))
        out["probes"][name] = {
            "rms": float(np.sqrt(np.mean(sig ** 2))),
            "peak_St": float(freq[k]),
            # fraction of the power within +-10 % of the shedding frequency and
            # of its first harmonic -- the hydrodynamic share
            "f_peak_power_share": float(p[k] / p.sum()),
        }
    # time-mean divergence map statistics too: a steady offset is neither
    # sound nor shedding, it is a mean compressibility of the flow
    mean_div = div.mean(axis=0)
    out["mean_div_rms"] = float(np.sqrt(np.mean(mean_div ** 2)))
    out["fluct_div_rms"] = float(np.sqrt(np.mean((div - mean_div) ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = []
    for tag in args.tags:
        r = analyse(tag)
        res.append(r)
        st = json.loads((ROOT / "data" / f"cylinder_{tag}_meta.json")
                        .read_text())["St"]
        print(f"\n=== {tag}   shedding St (lift meta) {st:.4f}")
        for k, v in r["modes"].items():
            print(f"    candidate mode {k:20s} St {v:.4f}")
        print(f"    time-mean div rms {r['mean_div_rms']:.3e}   "
              f"fluctuating div rms {r['fluct_div_rms']:.3e}")
        for name, p in r["probes"].items():
            print(f"    {name:18s} rms {p['rms']:.3e}  peak at St {p['peak_St']:.4f}"
                  f"  ({100 * p['f_peak_power_share']:.0f} % of power in that bin)")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
