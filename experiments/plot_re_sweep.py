"""Does the Phase 4 result hold across the shedding range?

Two panels. Left: shedding retained against Reynolds number, for the control
and for the mechanism-equipped configuration, every seed plotted. Right: the
datasets' own validation -- measured Strouhal number against Qu et al. (2013),
Table 3, with Williamson's correlation as context -- because the left panel
means nothing if the ground truth at each Re is not itself a real wake.

Each run starts at the beginning of its dataset's record, which falls at an
uncontrolled phase of the shedding cycle. That matters for the control, whose
retention swings with start phase, and not for `all` at sigma = 8, which does
not (report, Section 4).

Re = 100 comes from the main ablation (`runs/ablation`, `runs/ablation_sigma8`)
rather than the sweep, since that dataset predates it; the configuration is the
same.

Usage:
    python experiments/plot_re_sweep.py
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

import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from derived_metrics import QU2013_TABLE3  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ACCENT, CONTEXT = "#2a78d6", "#898781"


def strouhal_correlation(re):
    """Williamson (1988), the laminar 2-D shedding range."""
    return -3.3265 / re + 0.1816 + 1.6e-4 * re


def collect(sweep_dir="re_sweep"):
    """{Re: {'baseline': [...], 'all': [...], 'St': float}} across all sources."""
    out = {}
    sweep = json.loads((ROOT / "runs" / sweep_dir / "summary.json").read_text())
    for re, row in sweep.items():
        out[float(re)] = {
            "baseline": [100 * v for k, v in sorted(row.items())
                         if k.startswith("baseline_shedding")],
            "all": [100 * v for k, v in sorted(row.items())
                    if k.startswith("all_shedding")],
            "St": row.get("St"),
        }

    # Re = 100: the v1 sweep reused runs made before the sweep existed. The
    # v5 sweep runs Re = 100 itself, so only graft when it is missing.
    if out.get(100.0, {}).get("all"):
        return out
    for name, cfg in (("ablation", "baseline"), ("ablation_sigma8", "all")):
        path = ROOT / "runs" / name / "summary.json"
        if path.exists():
            rows = json.loads(path.read_text())
            out.setdefault(100.0, {})[cfg] = [
                100 * r["shedding"] for r in rows if r["config"] == cfg]
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep-dir", default="re_sweep")
    args = ap.parse_args()

    data = collect(args.sweep_dir)
    res = sorted(data)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                 gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.22))
    fig.patch.set_facecolor(SURFACE)

    for cfg, color, label in (("baseline", CONTEXT, "baseline (start phase uncontrolled)"),
                              ("all", ACCENT, "all three mechanisms, data-free")):
        means, xs = [], []
        for re in res:
            vals = data[re].get(cfg) or []
            if not vals:
                continue
            xs.append(re)
            means.append(np.mean(vals))
            a1.scatter([re] * len(vals), vals, s=18, color=color, alpha=0.55,
                       zorder=3, linewidths=0)
        a1.plot(xs, means, color=color, lw=2.0, zorder=2, label=label)

    a1.axhline(100, color=AXIS, ls=(0, (4, 3)), lw=1.2)
    a1.text(res[-1], 101, "LBM reference", ha="right", va="bottom",
            fontsize=8.5, color=INK_2)
    a1.set_ylim(0, 115)
    a1.set_xlabel("Reynolds number", color=INK_2, fontsize=10)
    a1.set_ylabel("shedding retained (%)", color=INK_2, fontsize=10)
    a1.set_title("Shedding retained over one period", loc="left",
                 fontsize=10, color=INK_2)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="center right")

    fine = np.linspace(min(res) - 5, max(res) + 5, 200)
    a2.plot(fine, strouhal_correlation(fine), color=GRID, lw=2.5, zorder=1,
            label="Williamson (1988) correlation")
    qu_re = [float(k) for k in QU2013_TABLE3 if min(res) <= float(k) <= max(res)]
    a2.scatter(qu_re, [QU2013_TABLE3[f"{int(r)}"]["St"] for r in qu_re], s=60,
               facecolors="none", edgecolors=INK_2, lw=1.1, zorder=2,
               label="Qu et al. (2013), Table 3")
    st = [data[re]["St"] for re in res if data[re].get("St")]
    st_res = [re for re in res if data[re].get("St")]
    a2.scatter(st_res, st, s=22, color=ACCENT, zorder=3, linewidths=0,
               label="LBM datasets (wake vorticity)")
    for re, v in zip(st_res, st):
        ref = QU2013_TABLE3[f"{int(re)}"]["St"]
        err = 100 * (v - ref) / ref
        a2.annotate(f"{err:+.2f}%" if abs(err) < 0.05 else f"{err:+.1f}%",
                    (re, v), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=7.5, color=INK_2)
    a2.set_xlabel("Reynolds number", color=INK_2, fontsize=10)
    a2.set_ylabel("Strouhal number", color=INK_2, fontsize=10)
    a2.set_title("Ground truth against Qu et al.", loc="left",
                 fontsize=10, color=INK_2)
    a2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")

    for a in (a1, a2):
        style(a)

    fig.subplots_adjust(top=0.8)
    fig.suptitle("The Reynolds sweep: data-free PINNs against validated ground truth",
                 x=0.01, y=0.99, ha="left", va="top", fontsize=12.5, color=INK)
    fig.text(0.01, 0.9,
             f"2-D cylinder wake, Re = {int(min(res))}-{int(max(res))} · "
             "2 seeds per point · one shedding period per run, "
             "one ground-truth slice",
             fontsize=9, color=INK_2)

    out = Path(args.out) if args.out else ROOT / "figures" / "re_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
