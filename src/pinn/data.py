"""Access layer for the LBM ground-truth datasets.

Everything downstream -- initial conditions, boundary data, supervised terms,
error metrics -- reads the flow through this module, so the non-dimensional
convention is defined in exactly one place:

    D = 1, U = 1, cylinder centred at the origin
    x in [-5, 15], y in [-5, 5], t* = U t / D
    fields are (u, v, p) with p in units of rho U^2

Sampling returns values at exact grid nodes and snapshot times rather than
interpolating, so supervised targets carry no interpolation error of their own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CylinderDataset:
    """A time-resolved 2-D cylinder wake on a uniform grid."""

    x: np.ndarray  # (nx,)
    y: np.ndarray  # (ny,)
    t: np.ndarray  # (nt,)
    fields: np.ndarray  # (nt, 3, nx, ny) -> u, v, p
    mean_field: np.ndarray  # (3, nx, ny) time mean over the record
    solid: np.ndarray  # (nx, ny) bool
    Re: float
    meta: dict

    # -- geometry ----------------------------------------------------------
    @property
    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the saved sub-domain."""
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    @property
    def t_range(self):
        return float(self.t[0]), float(self.t[-1])

    @property
    def fluct(self):
        """u' = u - u0, the quantity the reference paper actually learns."""
        return self.fields - self.mean_field[None]

    # -- sampling ----------------------------------------------------------
    def _fluid_index(self):
        if not hasattr(self, "_fi"):
            ii, jj = np.where(~self.solid)
            self._fi = (ii, jj)
        return self._fi

    def sample(self, n, rng, t_slice=None, fluctuation=False):
        """Random (coords, values) pairs drawn from fluid nodes.

        Returns coords (n, 3) as (x, y, t) and values (n, 3) as (u, v, p).
        `t_slice` restricts sampling to a range of snapshot indices.
        """
        ii, jj = self._fluid_index()
        lo, hi = (0, len(self.t)) if t_slice is None else t_slice
        pick = rng.integers(0, ii.size, size=n)
        kt = rng.integers(lo, hi, size=n)
        i, j = ii[pick], jj[pick]

        src = self.fluct if fluctuation else self.fields
        coords = np.stack([self.x[i], self.y[j], self.t[kt]], axis=1)
        values = src[kt, :, i, j]
        return coords.astype(np.float64), values.astype(np.float64)

    def snapshot(self, k, fluctuation=False):
        """Full field at snapshot index k as (coords (M,3), values (M,3))."""
        ii, jj = self._fluid_index()
        src = self.fluct if fluctuation else self.fields
        coords = np.stack(
            [self.x[ii], self.y[jj], np.full(ii.size, self.t[k])], axis=1
        )
        return coords.astype(np.float64), src[k, :, ii, jj].astype(np.float64)

    def nearest_time_index(self, t_star):
        return int(np.argmin(np.abs(self.t - float(t_star))))

    # -- metrics -----------------------------------------------------------
    @staticmethod
    def relative_l2(pred, true):
        """The reference paper's error metric (its eq. 3.3).

        eps = ||q* - q||_2 / ||q||_2 with q the stacked (u, v) field, so it is
        a single number per snapshot rather than a per-component error.
        """
        pred = np.asarray(pred)[..., :2].ravel()
        true = np.asarray(true)[..., :2].ravel()
        return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def load(tag="re100_v3", datadir=None, time_stride=1, center_pressure=True):
    """Load `data/cylinder_<tag>.npz` (plus its meta JSON if present).

    In an incompressible flow only pressure *gradients* are physical, and
    LBM leaves an arbitrary offset in rho (the outlet lets the mean density
    drift). With `center_pressure` the outlet-column mean is subtracted, so
    a `p = 0` outlet boundary condition agrees with the data instead of
    fighting it by a constant.
    """
    datadir = Path(datadir) if datadir else ROOT / "data"
    npz = datadir / f"cylinder_{tag}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found -- run src/cfd/lbm_cylinder.py first")
    d = np.load(npz)

    meta_path = datadir / f"cylinder_{tag}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    sl = slice(None, None, time_stride)
    fields = d["snapshots"][sl]
    mean_field = d["mean_field"]
    p_ref = 0.0
    if center_pressure:
        p_ref = float(fields[:, 2, -1, :].mean())  # outlet column, all times
        fields = fields.copy()
        fields[:, 2] -= p_ref
        mean_field = mean_field.copy()
        mean_field[2] -= p_ref
    meta = dict(meta, p_reference=p_ref)

    return CylinderDataset(
        x=d["x"].astype(np.float64),
        y=d["y"].astype(np.float64),
        t=d["t"][sl].astype(np.float64),
        fields=fields,
        mean_field=mean_field,
        solid=d["solid"],
        Re=float(d["Re"]),
        meta=meta,
    )


if __name__ == "__main__":
    import sys

    tag = sys.argv[1] if len(sys.argv) > 1 else "re100_v3"
    ds = load(tag)
    xmin, xmax, ymin, ymax = ds.bounds
    t0, t1 = ds.t_range
    print(f"tag        : {tag}")
    print(f"Re         : {ds.Re:g}   scheme {ds.meta.get('scheme', '?')}")
    print(f"grid       : {ds.x.size} x {ds.y.size}   dx = {ds.x[1] - ds.x[0]:.4f} D")
    print(f"domain     : x [{xmin:g}, {xmax:g}]   y [{ymin:g}, {ymax:g}]")
    print(f"snapshots  : {ds.t.size}   t* {t0:.2f} .. {t1:.2f}   dt* {ds.t[1] - ds.t[0]:.4f}")
    print(f"fluid nodes: {int((~ds.solid).sum())} of {ds.solid.size}")

    rng = np.random.default_rng(0)
    c, v = ds.sample(5, rng)
    print("\nsample coords (x, y, t) and values (u, v, p):")
    for ci, vi in zip(c, v):
        print(f"  ({ci[0]:+.3f}, {ci[1]:+.3f}, {ci[2]:7.3f}) -> "
              f"({vi[0]:+.4f}, {vi[1]:+.4f}, {vi[2]:+.4f})")

    f = ds.fluct
    print(f"\nfluctuation u' rms = {np.sqrt((f[:, :2] ** 2).mean()):.4f}"
          f"   mean-flow rms = {np.sqrt((ds.mean_field[:2] ** 2).mean()):.4f}")
