"""Section 8 figure: does physics help an operator forecast, and for which backbone?

Two panels, rollout error against forecast horizon on a log axis, every curve
the seed mean reduced by the pairwise mean of adjacent steps (the report's
convention; see experiments/verify_report.py, src_fno).

    left   the FNO on re100_v5, data against the two physics variants
    right  three backbones on re100_v4 -- FNO, U-Net, ConvGRU -- each trained
           data-only (solid) and with continuity + vorticity transport (dashed)

    python experiments/plot_operators.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
VARIANT = {"data": ("data-driven", "#898781"),
           "physics": ("+ continuity and vorticity", "#2a78d6"),
           "physics_div": ("+ continuity only", "#7aa9de")}
BACKBONE = {"fno": ("FNO", "#2a78d6"), "unet": ("U-Net", "#d9822b"),
            "convgru": ("ConvGRU", "#3b9b6d")}
RUNS = {"fno": "runs/fno", "unet": "runs/fno_unet", "convgru": "runs/fno_convgru"}
TRAINED = 10


def curve(run_dir, variant):
    """(steps[1:], pairwise mean of the seed-averaged eps_full)."""
    blobs = [json.loads(f.read_text())
             for f in sorted((ROOT / run_dir).glob(f"{variant}_s*/results.json"))]
    steps = np.asarray(blobs[0]["curve"]["steps"])
    eps = np.mean([b["curve"]["eps_full"] for b in blobs], axis=0)
    return steps[1:], 0.5 * (eps[1:] + eps[:-1])


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.axvline(TRAINED, color=AXIS, ls=(0, (4, 3)), lw=1.2)
    ax.set_xlabel("forecast step", color=INK_2, fontsize=10)
    ax.set_title(title, loc="left", fontsize=10, color=INK_2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures" / "operators.png"))
    args = ap.parse_args()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(wspace=0.18))
    fig.patch.set_facecolor(SURFACE)

    for v, (label, color) in VARIANT.items():
        x, y = curve("runs/v5/fno", v)
        a1.plot(x, y, color=color, lw=2.0, label=label)
    style(a1, "FNO, validated v5 data")
    a1.set_ylabel("relative $L_2$ on (u, v)", color=INK_2, fontsize=10)
    a1.text(TRAINED + 0.7, a1.get_ylim()[1] * 0.8, "trained horizon",
            fontsize=8.5, color=INK_2, va="top")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="center left",
              bbox_to_anchor=(0.0, 0.55))

    for arch, (label, color) in BACKBONE.items():
        for v, ls in (("data", "-"), ("physics", (0, (4, 2)))):
            x, y = curve(RUNS[arch], v)
            a2.plot(x, y, color=color, lw=2.0, ls=ls,
                    label=f"{label}, {'data' if v == 'data' else '+ physics'}")
    style(a2, "Three backbones, earlier v4 data")
    a2.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right", ncol=1)

    fig.subplots_adjust(top=0.8)
    fig.suptitle("Physics in an operator forecast: the answer depends on the backbone",
                 x=0.01, y=0.99, ha="left", va="top", fontsize=12.5, color=INK)
    fig.text(0.01, 0.9, "Held-out initial conditions · 2 seeds per curve · "
             "pairwise mean of adjacent steps · trained on 10-step rollouts",
             fontsize=9, color=INK_2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
