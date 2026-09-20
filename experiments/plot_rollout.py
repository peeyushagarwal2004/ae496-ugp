"""Phase 5 figure: rollout error against forecast horizon, with and without physics.

One panel, one line per variant, error on a log axis because the interesting
behaviour is the growth rate. The trained horizon is marked: inside it the
curves are expected to sit on top of each other, and any separation outside it
is the result.

Reads runs/<run-name>/<variant>_s<seed>/results.json and writes
figures/<run-name>.png.

Usage:
    python experiments/plot_rollout.py --run-name fno
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

LABELS = {"data": "data-driven (paper recipe)",
          "physics": "+ continuity and vorticity transport",
          "physics_div": "+ continuity only"}
COLORS = {"data": "#898781", "physics": "#2a78d6", "physics_div": "#7aa9de"}

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def collect(run_dir, metric):
    """{variant: (steps, mean curve over seeds)} for the runs present."""
    out = {}
    for d in sorted(run_dir.glob("*_s*")):
        blob = json.loads((d / "results.json").read_text())
        variant = d.name.rsplit("_s", 1)[0]
        out.setdefault(variant, []).append(blob["curve"])
    return {v: (np.asarray(cs[0]["steps"]),
                np.mean([c[metric] for c in cs], axis=0), len(cs))
            for v, cs in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="fno")
    ap.add_argument("--metric", default="eps_full",
                    choices=["eps_full", "eps_fluct"],
                    help="error on the velocity field, or on the fluctuation alone")
    ap.add_argument("--quick", action="store_true",
                    help="read runs_quick/ -- for checking the figure renders")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = ROOT / ("runs_quick" if args.quick else "runs") / args.run_name
    curves = collect(run_dir, args.metric)
    if not curves:
        raise SystemExit(f"no results under {run_dir}")

    trained = json.loads(
        next(run_dir.glob("*_s*/results.json")).read_text())["config"]["rollout"]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for variant, (steps, eps, n_seeds) in curves.items():
        label = LABELS.get(variant, variant)
        if n_seeds > 1:
            label += f" ({n_seeds} seeds)"
        ax.plot(steps, eps, lw=2.0, color=COLORS.get(variant, INK_2), label=label)

    ax.axvspan(0, trained, color=GRID, alpha=0.55, lw=0)
    ax.text(trained, ax.get_ylim()[1], " trained horizon ", ha="left", va="top",
            fontsize=8.5, color=INK_2)

    ax.set_yscale("log")
    ax.set_xlabel("recursive forecast steps", color=INK_2, fontsize=10)
    ax.set_ylabel("relative L2 on (u, v)", color=INK_2, fontsize=10)
    ax.set_xlim(1, max(s[-1] for s, _, _ in curves.values()))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")

    fig.subplots_adjust(top=0.82)
    fig.suptitle("Does a physics term buy rollout stability beyond the trained horizon?",
                 x=0.01, y=0.99, ha="left", va="top", fontsize=12.5, color=INK)
    fig.text(0.01, 0.9,
             f"Re = 100 cylinder wake · FNO2D · trained on {trained} recursive steps"
             " · held-out initial conditions",
             fontsize=9, color=INK_2)

    out = (Path(args.out) if args.out else
           ROOT / "figures" / f"{args.run_name}{'_quick' if args.quick else ''}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
