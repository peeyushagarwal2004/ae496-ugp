"""Ablation bar chart: shedding retained and relative L2 for every config.

Reads runs/<run-name>/summary.json (written by run_ablation.py) and writes
figures/<run-name>.png. Two panels on a shared config axis rather
than one dual-axis chart, because the two measures have unrelated scales.
Combined-mechanism configs are the accent colour; the control and the
single-mechanism configs are context, in the de-emphasis grey.

Usage:
    python experiments/plot_ablation.py --run-name ablation_cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

ORDER = ["baseline", "fourier", "decompose", "march", "all", "all+data"]
LABELS = {"baseline": "baseline (none)", "fourier": "Fourier features",
          "decompose": "Reynolds decomposition", "march": "time-marching",
          "all": "all three", "all+data": "all three + data"}
COMBINED = {"all", "all+data"}

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ACCENT, CONTEXT = "#2a78d6", "#898781"


def mean_by_config(rows):
    """Seed-mean per config, in ORDER (configs that were not run are skipped)."""
    out = {}
    for name in ORDER:
        rs = [r for r in rows if r["config"] == name]
        if rs:
            out[name] = dict(shedding=100 * np.mean([r["shedding"] for r in rs]),
                             rel_l2=np.mean([r["rel_l2"] for r in rs]), n=len(rs))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="ablation_cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.loads((ROOT / "runs" / args.run_name / "summary.json").read_text())
    m = mean_by_config(rows)
    names = list(m)[::-1]  # barh draws bottom-up; keep baseline on top
    y = np.arange(len(names))
    colors = [ACCENT if n in COMBINED else CONTEXT for n in names]

    seeds = max(v["n"] for v in m.values())
    steps = {r["windows"] * r["epochs_per_window"] for r in rows}
    budget = f"{steps.pop()} steps per config" if len(steps) == 1 else "mixed budgets"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.08))
    fig.patch.set_facecolor(SURFACE)

    shed = [m[n]["shedding"] for n in names]
    a1.barh(y, shed, height=0.62, color=colors)
    a1.axvline(100, color=AXIS, ls=(0, (4, 3)), lw=1.2)
    a1.text(99, len(names) - 0.45, "LBM reference", ha="right", va="bottom",
            fontsize=8.5, color=INK_2)
    for yi, v in zip(y, shed):
        a1.text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    a1.set_xlim(0, 105)
    a1.set_title("shedding retained (%) — higher is better", loc="left",
                 fontsize=10, color=INK_2)

    l2 = [m[n]["rel_l2"] for n in names]
    a2.barh(y, l2, height=0.62, color=colors)
    for yi, v in zip(y, l2):
        a2.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    a2.set_xlim(0, max(l2) * 1.22)
    a2.set_title("relative L2 error — lower is better", loc="left",
                 fontsize=10, color=INK_2)

    for a in (a1, a2):
        style(a)
    a1.set_yticks(y, [LABELS[n] for n in names], color=INK, fontsize=9.5)

    # Header band: reserve the top of the figure so titles never meet the panels.
    fig.subplots_adjust(top=0.78)
    fig.suptitle("Only the combined mechanisms pull the PINN out of the steady collapse",
                 x=0.01, y=0.99, ha="left", va="top", fontsize=12.5, color=INK)
    fig.text(0.01, 0.905,
             f"Re = 100 cylinder wake · data-free unless noted · {seeds} seed{'' if seeds == 1 else 's'} · {budget}"
             " · wake-probe v r.m.s. of PINN vs LBM",
             fontsize=9, color=INK_2)
    fig.legend(handles=[Patch(color=ACCENT, label="combined mechanisms"),
                        Patch(color=CONTEXT, label="control / single mechanism")],
               loc="lower left", bbox_to_anchor=(0.005, -0.06), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_2)

    out = Path(args.out) if args.out else ROOT / "figures" / f"{args.run_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
