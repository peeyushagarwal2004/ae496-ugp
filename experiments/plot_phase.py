#!/usr/bin/env python
"""Shedding retained against the start phase, for the headline configurations.

Reads runs/v5/phase_average.json (written by experiments/phase_average.py).
Each configuration is one line through its seed means, with every individual
run plotted behind it, so the half-cycle swing of the plain PINN and of
marching alone is visible next to the flat line of all three mechanisms.

    python experiments/phase_average.py && python experiments/plot_phase.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Categorical slots 1-3, validated. Slot 3 is below 3:1 against the surface,
# so every series is also direct-labelled and carries its own marker shape,
# and the report gives the same numbers as a table.
SERIES = [
    ("baseline", "plain PINN", "#2a78d6", "o"),
    ("march", "+ causal marching", "#eb6834", "s"),
    ("all", "+ Fourier features + decomposition", "#1baf7a", "D"),
]
PHASE_LABELS = ["0", "1/4", "1/2", "3/4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "runs" / "v5" / "phase_average.json"))
    ap.add_argument("--out", default=str(ROOT / "figures" / "phase.png"))
    args = ap.parse_args()

    doc = json.loads(Path(args.data).read_text())
    runs, summ = doc["runs"], doc["summary"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for cfg, label, col, mk in SERIES:
        if cfg not in summ:
            continue
        # every run, behind
        for key, r in runs.items():
            c, k, _ = key.rsplit("_", 2)
            if c != cfg:
                continue
            ax.scatter(int(k[1:]), 100 * r["shedding"], s=22, color=col,
                       alpha=0.35, marker=mk, linewidth=0, zorder=2)
        by_phase = summ[cfg]["shedding_by_phase"]
        xs = [int(k) for k in sorted(by_phase, key=int)]
        ys = [100 * by_phase[str(k)] for k in xs]
        ax.plot(xs, ys, color=col, lw=2, zorder=3)
        ax.scatter(xs, ys, s=64, color=col, marker=mk, edgecolor=SURFACE,
                   linewidth=2, zorder=4, label=label)
        mean = 100 * summ[cfg]["shedding_mean"]
        ax.text(xs[-1] + 0.12, ys[-1], f"{label}\nmean {mean:.1f} %",
                va="center", ha="left", fontsize=8.5, color=INK_2)

    ax.set_xticks(range(4))
    ax.set_xticklabels([f"phase {p}" for p in PHASE_LABELS], fontsize=9,
                       color=INK_2)
    ax.set_xlim(-0.3, 4.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("shedding retained (%)", fontsize=9.5, color=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK_2)
    ax.set_title("Retention by the phase at which the run starts "
                 "(Re = 100, one ground-truth slice per run)",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
