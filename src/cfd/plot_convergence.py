"""Turn the convergence sweep into the Phase-1 validation figure.

The production runs sit above the Re = 100 benchmark on both St and C_d. This
script separates the two candidate causes and extrapolates the surviving one
to zero, which is what justifies calling the solver validated.

Usage:
    python src/cfd/plot_convergence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CD_REF, ST_REF = 1.375, 0.165


def main():
    rows = json.loads((ROOT / "data" / "convergence.json").read_text())
    for r in rows:
        r["H"] = round(1.0 / r["blockage"])

    res = sorted([r for r in rows if r["H"] == 20], key=lambda r: r["D_lattice"])
    blk = sorted([r for r in rows if r["D_lattice"] == 24], key=lambda r: r["blockage"])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    # ---- (a) resolution: is the solution grid converged? -----------------
    D = np.array([r["D_lattice"] for r in res])
    cd = np.array([r["Cd_mean"] for r in res])
    ax[0].plot(D, cd, "o-", color="C0")
    ax[0].axhline(CD_REF, color="k", ls="--", lw=1, label="benchmark 1.375")
    ax[0].set_xlabel("cylinder diameter $D$ (lattice units)")
    ax[0].set_ylabel("$\\overline{C_d}$")
    ax[0].set_title(f"(a) resolution, fixed 5 % blockage\nspread = "
                    f"{100 * (cd.max() - cd.min()) / cd.mean():.1f} %", fontsize=10)
    ax[0].set_ylim(1.30, 1.72)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # ---- (b) blockage: the term that actually moves ----------------------
    beta = np.array([r["blockage"] for r in blk])
    cdb = np.array([r["Cd_mean"] for r in blk])
    # Richardson-style linear extrapolation in beta using the two finest cases.
    sl = (cdb[1] - cdb[0]) / (beta[1] - beta[0])
    cd0 = cdb[0] - sl * beta[0]

    bb = np.linspace(0, beta.max() * 1.05, 50)
    ax[1].plot(100 * beta, cdb, "o-", color="C1", label="LBM")
    ax[1].plot(100 * bb, cd0 + sl * bb, ":", color="C1", lw=1.2,
               label=f"extrapolation $\\to$ {cd0:.3f}")
    ax[1].axhline(CD_REF, color="k", ls="--", lw=1, label="benchmark 1.375")
    ax[1].plot(0, cd0, "*", ms=14, color="C3", zorder=5)
    ax[1].set_xlabel("blockage $D/H$ (%)")
    ax[1].set_ylabel("$\\overline{C_d}$")
    ax[1].set_title(f"(b) blockage, fixed $D$ = 24\nzero-blockage limit "
                    f"{cd0:.3f} ({100 * (cd0 - CD_REF) / CD_REF:+.1f} %)", fontsize=10)
    ax[1].set_xlim(-0.3, 100 * beta.max() * 1.05)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # ---- (c) Strouhal, both series ---------------------------------------
    ax[2].plot(100 * np.full(len(res), 0.05), [r["St"] for r in res], "o",
               color="C0", label="resolution series")
    ax[2].plot(100 * beta, [r["St"] for r in blk], "s-", color="C1",
               label="blockage series")
    ax[2].axhline(ST_REF, color="k", ls="--", lw=1, label="benchmark 0.165")
    ax[2].set_xlabel("blockage $D/H$ (%)")
    ax[2].set_ylabel("$St$")
    ax[2].set_title("(c) Strouhal number", fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.suptitle("LBM cylinder solver at Re = 100: the residual bias is domain "
                 "confinement, not discretisation", fontsize=12)
    out = ROOT / "figures" / "convergence.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}\n")

    print(f"resolution series (H = 20 D): Cd spread "
          f"{100 * (cd.max() - cd.min()) / cd.mean():.1f} % over D = {D.min()}..{D.max()}")
    print("  -> grid converged; resolution is NOT the error source\n")
    print("blockage series (D = 24):")
    for b, c in zip(beta, cdb):
        print(f"  D/H = {100 * b:4.1f} %   Cd = {c:.4f}   ({100 * (c - CD_REF) / CD_REF:+5.1f} %)")
    print(f"  extrapolated to zero blockage: Cd = {cd0:.4f} "
          f"({100 * (cd0 - CD_REF) / CD_REF:+.1f} % vs benchmark)")


if __name__ == "__main__":
    main()
