"""PINN vs LBM: vorticity at one instant, plus the wake-probe signal over time.

The Phase 3 figure. The top rows put the LBM reference and each trained PINN
side by side at the same t*, so a collapsed network shows up as a symmetric,
steady wake where the reference has a Karman street. The bottom panel is the
headline metric made visible: transverse velocity at the wake probe.

Needs runs trained with --save-params. Each network is rebuilt with the
trainer's own build() and re-scored; the script refuses to plot if that
score does not match the one saved at training time, so a bad reload cannot
produce a plausible-looking figure.

Usage:
    python experiments/plot_fields.py --run-name ablation_cpu_fields \
        --runs baseline_s0,all_s0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pinn"))
sys.path.insert(0, str(ROOT / "src" / "cfd"))

import cylinder_unsteady as cu  # noqa: E402
from inspect_data import vorticity  # noqa: E402
from pinn.data import load  # noqa: E402

LABELS = {"baseline": "PINN, baseline (no mechanisms)", "fourier": "PINN, Fourier features",
          "decompose": "PINN, Reynolds decomposition", "march": "PINN, time-marching",
          "all": "PINN, all three mechanisms", "all+data": "PINN, all three + data"}

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ACCENT, CONTEXT = "#2a78d6", "#898781"
VMAX = 5.0  # same vorticity scale as figures/<tag>_vorticity.png
VIEW = (-2.0, 12.0, -3.0, 3.0)  # wake region; the full 20 x 10 D domain wastes the page


def restore(run_dir, ds):
    """Rebuild a saved run: (config name, pde, segments, saved overall score)."""
    blob = pickle.loads((run_dir / "params.pkl").read_bytes())
    args = argparse.Namespace(**blob["args"])
    _, pde, _ = cu.build(args, ds)
    segments = [(a, b, p) for a, b, p in blob["segments"]]
    saved = json.loads((run_dir / "results.json").read_text())["overall"]
    name = run_dir.name.rsplit("_s", 1)[0]
    return name, pde, segments, saved


def owner(segments, t):
    ends = np.array([b for _, b, _ in segments[:-1]])
    return segments[int(np.searchsorted(ends, t, side="left"))][2]


def field_at(pde, params, ds, t):
    """(u, v) on the full saved grid at time t, shaped (nx, ny) like the LBM."""
    X, Y = np.meshgrid(ds.x, ds.y, indexing="ij")
    xyt = np.stack([X.ravel(), Y.ravel(), np.full(X.size, t)], axis=1)
    q = np.asarray(pde.u(params, jnp.asarray(xyt)))
    return q[:, 0].reshape(X.shape), q[:, 1].reshape(X.shape)


def probe_series(pde, segments, tt):
    xyt = np.stack([np.full_like(tt, cu.PROBE[0]), np.full_like(tt, cu.PROBE[1]), tt], axis=1)
    ends = np.array([b for _, b, _ in segments[:-1]])
    own = np.searchsorted(ends, tt, side="left")
    v = np.empty_like(tt)
    for s in np.unique(own):
        m = own == s
        v[m] = np.asarray(pde.u(segments[s][2], jnp.asarray(xyt[m])))[:, 1]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="ablation_cpu_fields")
    ap.add_argument("--runs", default="baseline_s0,all_s0",
                    help="comma-separated run dirs under runs/<run-name> (or absolute paths)")
    ap.add_argument("--t", type=float, default=None,
                    help="snapshot time (default: late in the span, inside the last window)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = ROOT / "runs" / args.run_name
    dirs = [Path(r) if Path(r).is_absolute() else base / r for r in args.runs.split(",")]

    first = pickle.loads((dirs[0] / "params.pkl").read_bytes())["args"]
    ds = load(first["tag"])
    runs = [restore(d, ds) for d in dirs]

    # ---- reload check: the rebuilt networks must reproduce their scores ----
    for name, pde, segs, saved in runs:
        t0, t1 = saved["t0"], saved["t1"]
        got = cu.evaluate(pde, segs, ds, t0, t1, n_times=24)
        d_shed = abs(got["shedding_ratio"] - saved["shedding_ratio"])
        d_l2 = abs(got["rel_l2_mean"] - saved["rel_l2_mean"])
        print(f"{name:<10s} shedding {100 * got['shedding_ratio']:5.1f}% "
              f"(saved {100 * saved['shedding_ratio']:5.1f}%)  rel L2 {got['rel_l2_mean']:.4f} "
              f"(saved {saved['rel_l2_mean']:.4f})")
        if d_shed > 1e-3 or d_l2 > 1e-3:
            raise SystemExit(f"{name}: reloaded network does not reproduce its saved score")

    t0, t1 = runs[0][3]["t0"], runs[0][3]["t1"]
    t_req = args.t if args.t is not None else t1 - 0.4
    k = ds.nearest_time_index(t_req)
    t_snap = float(ds.t[k])

    # ---- figure ------------------------------------------------------------
    n_rows = 1 + len(runs)
    fig = plt.figure(figsize=(8.0, 2.55 * n_rows + 3.3))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(n_rows + 1, 2, width_ratios=[40, 1],
                          height_ratios=[1] * n_rows + [1.05], hspace=0.38, wspace=0.03,
                          top=0.94)

    solid = np.hypot(*np.meshgrid(ds.x, ds.y, indexing="ij")) < 0.5
    panels = [("LBM reference", ds.fields[k, 0], ds.fields[k, 1])]
    for name, pde, segs, _ in runs:
        u, v = field_at(pde, owner(segs, t_snap), ds, t_snap)
        panels.append((LABELS.get(name, name), u, v))

    im = None
    for i, (title, u, v) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        w = np.where(solid, np.nan, vorticity(u, v, ds.x, ds.y))
        im = ax.pcolormesh(ds.x, ds.y, w.T, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX,
                           shading="auto", rasterized=True)
        ax.add_patch(plt.Circle((0, 0), 0.5, color="#52514e", zorder=5))
        ax.set_xlim(VIEW[0], VIEW[1])
        ax.set_ylim(VIEW[2], VIEW[3])
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8, length=2)
        for s in ax.spines.values():
            s.set_color(AXIS)
        ax.set_ylabel("$y/D$", color=INK_2, fontsize=9)
        if i == len(panels) - 1:
            ax.set_xlabel("$x/D$", color=INK_2, fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        if i == 0:
            ax.plot(*cu.PROBE, marker="o", ms=5, mfc="none", mec=INK, mew=1.2, zorder=6)
            ax.annotate("wake probe", cu.PROBE, xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color=INK)
    cax = fig.add_subplot(gs[:n_rows, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"vorticity $\omega_z D/U$", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    # ---- wake-probe signal ---------------------------------------------------
    ax = fig.add_subplot(gs[n_rows, 0])
    ax.set_facecolor(SURFACE)
    i = int(np.argmin(np.abs(ds.x - cu.PROBE[0])))
    j = int(np.argmin(np.abs(ds.y - cu.PROBE[1])))
    sel = (ds.t >= t0) & (ds.t <= t1)
    ax.plot(ds.t[sel], ds.fields[sel, 1, i, j], color=INK, lw=2, label="LBM reference")
    tt = np.linspace(t0, t1, 400)
    for n, (name, pde, segs, saved) in enumerate(runs):
        c = ACCENT if n == len(runs) - 1 else CONTEXT  # emphasis: last run is the point
        ax.plot(tt, probe_series(pde, segs, tt), color=c, lw=2,
                label=f"{LABELS.get(name, name)} ({100 * saved['shedding_ratio']:.1f}% retained)")
    for n_b, (_, b, _) in enumerate(runs[-1][2][:-1]):
        ax.axvline(b, color=AXIS, lw=1, zorder=0,
                   label="time-marching window boundary" if n_b == 0 else None)
    ax.axvline(t_snap, color=AXIS, ls=(0, (3, 3)), lw=1)
    ax.text(t_snap, 1.0, " snapshot above", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK_2, va="bottom")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("$t^* = Ut/D$", color=INK_2, fontsize=9)
    ax.set_ylabel("$v/U$ at probe", color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=1, frameon=False,
              fontsize=8.5, labelcolor=INK_2)
    ax.set_title("transverse velocity at the wake probe", loc="left", fontsize=10, color=INK)

    fig.suptitle(f"Re = 100 cylinder wake at $t^* = {t_snap:.2f}$", x=0.075, ha="left",
                 fontsize=12.5, color=INK, y=0.995)

    out = Path(args.out) if args.out else ROOT / "figures" / f"{args.run_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
