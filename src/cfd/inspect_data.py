"""Validate and visualise an LBM cylinder dataset.

Checks the generated flow against the published Re = 100 benchmarks and
produces the figures that document Phase 1:

    figures/<tag>_forces.png     C_l / C_d histories + lift spectrum
    figures/<tag>_vorticity.png  four snapshots spanning one shedding period
    figures/<tag>_mean.png       time-mean field u0 (the Reynolds base flow)

Usage:
    python src/cfd/inspect_data.py --tag re100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ST_TARGET = 0.165
CD_TARGET = (1.37, 1.38)


def vorticity(u, v, x, y):
    """omega_z = dv/dx - du/dy on the (possibly non-unit-spaced) saved grid."""
    dvdx = np.gradient(v, x, axis=0)
    dudy = np.gradient(u, y, axis=1)
    return dvdx - dudy


def strouhal(cl, dt_star):
    """Shedding frequency from the lift signal, via mean zero-crossing spacing.

    Deliberately not an FFT peak: with only ~12 recorded shedding periods the
    FFT bin width is ~1/T ~ 0.014, which is larger than the discrepancy being
    measured, so peak-picking can flatter or condemn a result purely by where
    the bin lands. Zero-crossing timing resolves the period to a fraction of a
    time step and averages over every cycle in the record.
    """
    sig = np.asarray(cl, dtype=np.float64)
    sig = sig - sig.mean()
    if sig.size < 64:
        return float("nan")
    idx = np.where(np.diff(np.signbit(sig)))[0]
    if idx.size < 4:
        return float("nan")
    # Refine each crossing instant by linear interpolation between samples.
    t0 = idx + sig[idx] / (sig[idx] - sig[idx + 1])
    half_period = np.mean(np.diff(t0)) * dt_star
    return float(0.5 / half_period)


def lift_spectrum(cl, dt_star):
    sig = np.asarray(cl, dtype=np.float64)
    sig = sig - sig.mean()
    win = np.hanning(sig.size)
    spec = np.abs(np.fft.rfft(sig * win))
    freq = np.fft.rfftfreq(sig.size, d=dt_star)
    spec[0] = 0.0
    return freq, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="re100")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    npz = Path(args.datadir) / f"cylinder_{args.tag}.npz"
    d = np.load(npz)
    snaps, x, y, t = d["snapshots"], d["x"], d["y"], d["t"]
    mean_field, solid = d["mean_field"], d["solid"]
    cd, cl, dt_star = d["cd"], d["cl"], float(d["dt_star"])

    meta_path = Path(args.datadir) / f"cylinder_{args.tag}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    warmup = int(meta.get("warmup_steps", 0))

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # ---- statistics over the converged window only -----------------------
    cd_t, cl_t = cd[warmup:], cl[warmup:]
    freq, spec = lift_spectrum(cl_t, dt_star)
    st = strouhal(cl_t, dt_star)  # zero-crossing estimate; the spectrum is for plotting
    cd_mean, cd_amp = float(cd_t.mean()), float(cd_t.std())
    cl_rms = float(np.sqrt(np.mean(cl_t**2)))
    cl_amp = float(np.abs(cl_t).max())

    st_err = 100 * (st - ST_TARGET) / ST_TARGET
    cd_ref = 0.5 * (CD_TARGET[0] + CD_TARGET[1])
    cd_err = 100 * (cd_mean - cd_ref) / cd_ref

    print(f"dataset : {npz}   ({snaps.shape[0]} snapshots, grid {snaps.shape[2]}x{snaps.shape[3]})")
    print(f"span    : t* = {t[0]:.2f} .. {t[-1]:.2f}   dt* = {t[1] - t[0]:.4f}")
    print(f"periods : {(t[-1] - t[0]) * st:.1f} shedding periods recorded")
    print()
    print("  quantity      computed     benchmark      error")
    print(f"  St            {st:8.4f}     {ST_TARGET:8.3f}   {st_err:+7.1f} %")
    print(f"  Cd (mean)     {cd_mean:8.4f}   {CD_TARGET[0]:.2f}-{CD_TARGET[1]:.2f}   {cd_err:+7.1f} %")
    print(f"  Cd (rms fluc) {cd_amp:8.4f}")
    print(f"  Cl (rms)      {cl_rms:8.4f}")
    print(f"  Cl (peak)     {cl_amp:8.4f}")

    # ---- figure 1: forces -------------------------------------------------
    steps = np.arange(cd.size) * dt_star
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    ax[0].plot(steps, cd, lw=0.7)
    ax[0].axvline(warmup * dt_star, color="k", ls="--", lw=0.8, label="end of warm-up")
    ax[0].axhline(cd_mean, color="C3", ls=":", lw=1.0, label=f"mean {cd_mean:.3f}")
    ax[0].set_ylabel("$C_d$")
    ax[0].set_ylim(0, max(3.0, float(np.percentile(cd, 99.5))))
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"Re = {float(d['Re']):.0f} cylinder, LBM ground truth")

    ax[1].plot(steps, cl, lw=0.7, color="C1")
    ax[1].axvline(warmup * dt_star, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("$C_l$")
    ax[1].set_xlabel("$t^* = Ut/D$")

    ax[2].plot(freq, spec / spec.max(), lw=1.0, color="C2")
    ax[2].axvline(ST_TARGET, color="k", ls="--", lw=0.8, label=f"benchmark St = {ST_TARGET}")
    ax[2].axvline(st, color="C3", ls=":", lw=1.2, label=f"computed St = {st:.4f}")
    ax[2].set_xlim(0, 0.8)
    ax[2].set_xlabel("Strouhal number $fD/U$")
    ax[2].set_ylabel("lift spectrum")
    ax[2].legend(fontsize=8)
    fig.savefig(figdir / f"{args.tag}_forces.png", dpi=140)
    plt.close(fig)

    # ---- figure 2: vorticity over one shedding period --------------------
    period_snaps = max(1, int(round(1.0 / (st * (t[1] - t[0])))))
    idx = [min(snaps.shape[0] - 1, i * period_snaps // 4) for i in range(4)]
    vmax = 5.0
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), constrained_layout=True)
    for a, i in zip(axes, idx):
        w = vorticity(snaps[i, 0], snaps[i, 1], x, y)
        w = np.where(solid, np.nan, w)
        im = a.pcolormesh(x, y, w.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        a.add_patch(plt.Circle((0, 0), 0.5, color="0.3", zorder=5))
        a.set_aspect("equal")
        a.set_ylabel("$y/D$")
        a.set_title(f"$t^* = {t[i]:.2f}$", fontsize=9)
        fig.colorbar(im, ax=a, label=r"$\omega_z D/U$", shrink=0.9)
    axes[-1].set_xlabel("$x/D$")
    fig.savefig(figdir / f"{args.tag}_vorticity.png", dpi=140)
    plt.close(fig)

    # ---- figure 3: the Reynolds base flow u0 -----------------------------
    names = ["$\\bar{u}/U$", "$\\bar{v}/U$", "$\\bar{p}/\\rho U^2$"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    for a, k, nm in zip(axes, range(3), names):
        fld = np.where(solid, np.nan, mean_field[k])
        im = a.pcolormesh(x, y, fld.T, cmap="viridis", shading="auto")
        a.add_patch(plt.Circle((0, 0), 0.5, color="0.3", zorder=5))
        a.set_aspect("equal")
        a.set_ylabel("$y/D$")
        a.set_title(nm, fontsize=10)
        fig.colorbar(im, ax=a, shrink=0.9)
    axes[-1].set_xlabel("$x/D$")
    fig.suptitle("time-mean base flow $u_0$ (Reynolds decomposition)", fontsize=11)
    fig.savefig(figdir / f"{args.tag}_mean.png", dpi=140)
    plt.close(fig)

    # ---- fluctuation energy: confirms the decomposition is meaningful ----
    fluct = snaps - mean_field[None]
    e_mean = float(np.mean(mean_field[0] ** 2 + mean_field[1] ** 2))
    e_fluc = float(np.mean(fluct[:, 0] ** 2 + fluct[:, 1] ** 2))
    print()
    print(f"  mean-flow kinetic energy      {e_mean:.4f}")
    print(f"  fluctuation kinetic energy    {e_fluc:.4f}  ({100 * e_fluc / e_mean:.1f} % of mean)")
    print(f"\nfigures written to {figdir}/")


if __name__ == "__main__":
    main()
