#!/usr/bin/env python
"""How the ground truth was validated, in three panels.

Every value is read from a committed artifact; nothing here is typed in except
the literature reference, which is cited.

  (a) the transverse duct mode entrains the wake given time, and only an
      absorbing layer that relaxes toward the free stream releases it --
      wake-vorticity St, which cannot see irrotational sound
  (b) the inlet distance, not the domain height, was the dominant geometric
      error -- Cd against Qu et al. (2013) for the 2 x 2 cross at D = 32
  (c) the solution is grid converged, at D = 32 and 48

    python experiments/plot_validation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Same surface and ink tokens as every other figure in the report.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"   # categorical slots 1, 2 (validated)

# Qu, Norberg, Davidson, Peng and Wang (2013), J. Fluids Struct. 39, Table 3,
# Re = 100 at H = 120.
QU_CD, QU_ST = 1.317, 0.1649


def meta(tag):
    return json.loads((ROOT / "data" / f"cylinder_{tag}_meta.json").read_text())


def vorticity_st():
    """{tag: St_vorticity} from the committed measure_st outputs."""
    out = {}
    for name in ("st_instruments.json", "st_tsponge.json"):
        p = ROOT / "runs" / name
        if p.exists():
            for row in json.loads(p.read_text()):
                if "St_vorticity" in row:
                    out[row["tag"]] = row["St_vorticity"]
    return out


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)


def ref_line(ax, value, label, horizontal, color=INK_2, side="right",
             y_frac=0.03):
    """Dashed reference line, labelled inside the plot area.

    Labels sit at the bottom-left of a vertical line or the left end of a
    horizontal one, so they never collide with the panel title or run off the
    figure edge.
    """
    kw = dict(color=color, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if horizontal:
        ax.axhline(value, **kw)
        x0 = ax.get_xlim()[0]
        span = ax.get_xlim()[1] - x0
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.text(x0 + 0.02 * span, value + 0.012 * yspan, label, va="bottom",
                ha="left", fontsize=8, color=color)
    else:
        ax.axvline(value, **kw)
        y0, y1 = ax.get_ylim()
        pad = 0.0015 if side == "right" else -0.0015
        ax.text(value + pad, y0 + y_frac * (y1 - y0), label, va="bottom",
                ha="left" if side == "right" else "right",
                fontsize=8, color=color)


def panel_acoustic(ax):
    st = vorticity_st()
    rows = [  # (label, tag), top to bottom
        ("no layer, 20 periods", "acoustic_short"),
        ("no layer, 38 periods", "acoustic_long"),
        ("viscous layer, 38", "sponge_long"),
        ("relaxation layer, 38", "tsponge_long"),
    ]
    rows = [(lab, st[t]) for lab, t in rows if t in st]
    ys = list(range(len(rows)))[::-1]
    ax.scatter([v for _, v in rows], ys, s=64, color=SERIES_1,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    for (_, v), y in zip(rows, ys):
        ax.text(v, y + 0.28, f"{v:.4f}", ha="center", fontsize=8, color=INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=8.5, color=INK_2)
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.set_xlim(0.105, 0.178)
    ax.set_xlabel("wake-vorticity St", fontsize=9, color=INK_2)
    resonance = meta("acoustic_long")
    from math import sqrt
    st_a = (1 / sqrt(3)) / (2 * resonance["grid"][1]) * (
        resonance["D_lattice"] / resonance["U_lattice"])
    ref_line(ax, st_a, f"duct mode {st_a:.4f}", horizontal=False, color=MUTED,
             y_frac=0.91)
    ref_line(ax, QU_ST, f"Qu et al. {QU_ST}", horizontal=False, side="left")
    style(ax, "(a) the duct mode captures the wake")


def panel_geometry(ax):
    cells = {  # (inlet D, H/D): tag
        (8, 50): "geo_base", (8, 120): "geo_tall",
        (25, 50): "geo_long", (25, 120): "geo_tall_long",
    }
    err = {k: 100 * (meta(t)["Cd_mean"] / QU_CD - 1) for k, t in cells.items()}
    w = 0.34
    for i, (h, col) in enumerate(((50, SERIES_1), (120, SERIES_2))):
        xs = [j + (i - 0.5) * (w + 0.04) for j in range(2)]   # 2px-ish gap
        vals = [err[(xu, h)] for xu in (8, 25)]
        ax.bar(xs, vals, width=w, color=col, zorder=2, label=f"H = {h} D")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.15, f"+{v:.1f} %", ha="center", fontsize=8,
                    color=INK_2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["inlet 8 D ahead", "inlet 25 D ahead"], fontsize=8.5,
                       color=INK_2)
    ax.set_ylabel("Cd error against Qu et al. (%)", fontsize=9, color=INK_2)
    ax.set_ylim(0, 9)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    style(ax, "(b) inlet distance, not blockage")


def panel_resolution(ax):
    pts = sorted((meta(f"res_D{d}")["D_lattice"], meta(f"res_D{d}")["Cd_mean"])
                 for d in (24, 32, 48))
    xs = [d for d, _ in pts]
    ys = [c for _, c in pts]
    ax.plot(xs, ys, color=SERIES_1, lw=2, zorder=2)
    ax.scatter(xs, ys, s=64, color=SERIES_1, edgecolor=SURFACE, linewidth=2,
               zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.006, f"{y:.4f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"D = {x}" for x in xs], fontsize=8.5, color=INK_2)
    ax.set_xlim(20, 54)
    ax.set_ylim(1.30, 1.46)
    ax.set_ylabel("Cd   (inlet 8 D, H = 50 D)", fontsize=9, color=INK_2)
    ref_line(ax, QU_CD, f"Qu et al. {QU_CD}", horizontal=True)
    style(ax, "(c) grid converged at D = 32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures" / "validation.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    fig.patch.set_facecolor(SURFACE)
    panel_acoustic(axes[0])
    panel_geometry(axes[1])
    panel_resolution(axes[2])
    fig.subplots_adjust(left=0.13, right=0.94, bottom=0.17, top=0.84, wspace=0.42)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
