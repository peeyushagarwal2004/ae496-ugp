#!/usr/bin/env python
"""Build report/numbers.json: every number printed in report.tex, with its source.

The manifest was first written by hand. Regenerating the ground truth changed
almost every source path, so it is now declared here and generated, which
keeps the declarations short enough to read and review line by line.

Each entry says: this literal, on whichever line of report.tex holds it,
prints this value, which must be reproducible from this artifact to within
half the last printed digit. experiments/verify_report.py checks it.

    python report/build_manifest.py          # write report/numbers.json
    python experiments/verify_report.py      # check it

Conventions
-----------
tol      half the last printed digit, unless a note says otherwise
unit     "pct" multiplies the artifact by 100; "deg_from_periods" by 360
vs       the printed value is (artifact / vs - 1) * 100
manual   a number with no machine-readable source; reported as UNCHECKED
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "report" / "report.tex"
OUT = ROOT / "report" / "numbers.json"

V5 = "re100_v5"
QU_CD, QU_ST, QU_CL = 1.317, 0.1649, 0.2224     # Qu et al. (2013), Table 3
DM = "runs/v5/derived_metrics.json"
SWEEP5 = "runs/re_sweep_v5/summary.json"

# ---------------------------------------------------------------------------
# source constructors
# ---------------------------------------------------------------------------

def meta(tag, key):
    return {"kind": "scalar", "file": f"data/cylinder_{tag}_meta.json", "key": key}


def path(file, p):
    return {"kind": "path", "file": file, "path": p}


def lst(file, key, reduce="mean", **where):
    return {"kind": "list", "file": file, "where": where, "key": key,
            "reduce": reduce}


def dct(file, at, key=None, keys=None, reduce="mean"):
    s = {"kind": "dict", "file": file, "at": at, "reduce": reduce}
    if keys:
        s["keys"] = keys
    else:
        s["key"] = key
    return s


def ratio(num, den, mode="pct"):
    return {"kind": "ratio", "num": num, "den": den, "mode": mode}


def manual(provenance):
    return {"kind": "manual", "provenance": provenance}


ENTRIES = []


def E(id, literal, printed, tol, src, unit=None, vs=None, note=None):
    e = {"id": id, "tex": 0, "literal": literal, "printed": printed}
    if unit:
        e["unit"] = unit
    if vs is not None:
        e["vs"] = vs
    e["tol"] = tol
    if note:
        e["note"] = note
    e["src"] = src
    ENTRIES.append(e)


BS = "\\"
PCT = BS + "%"

# ---------------------------------------------------------------------------
# Section 2 -- ground truth
# ---------------------------------------------------------------------------

def section2():
    E("gt.layer_width", "An absorbing layer} of $8D$", 8.0, 0.5,
      meta(V5, "sponge_d"))
    E("gt.layer_rate", "equilibrium at a peak rate of\n", 0.05, 0.005,
      meta(V5, "sponge_target"))
    E("gt.length", "in a domain $60D$ long", 60.0, 0.5,
      ratio(path(f"data/cylinder_{V5}_meta.json", "grid.0"),
            meta(V5, "D_lattice"), mode="ratio"))
    E("gt.upstream", "The inlet $25D$ ahead of the cylinder", 25.0, 0.5,
      manual("Not recorded in the v5 metadata -- lbm_cylinder.py began writing "
             "upstream_d after this dataset was made. Set by --upstream 25 in "
             "the SWEEP flags of queue15.sh, which generated re100_v5."))
    E("gt.D", "$" + BS + "Rey = 100$, $D = 24$ lattice units", 24, 0.5,
      meta(V5, "D_lattice"))
    E("gt.Ma", "$" + BS + "mathrm{Ma} = 0.087$", 0.087, 0.0005, meta(V5, "Ma"))
    E("gt.grid_x", "grid $1440 " + BS + "times 1200$", 1440, 0.5,
      path(f"data/cylinder_{V5}_meta.json", "grid.0"))
    E("gt.grid_y", "grid $1440 " + BS + "times 1200$", 1200, 0.5,
      path(f"data/cylinder_{V5}_meta.json", "grid.1"))
    E("gt.blockage", "blockage $D/H = 2" + PCT + "$", 2.0, 0.5,
      meta(V5, "blockage"), unit="pct")
    E("gt.n_snapshots", "record is 641 snapshots", 641, 0.5,
      meta(V5, "n_snapshots"))
    E("gt.record_periods", "covering 20 shedding periods", 20, 0.5,
      path(DM, "fft_bin_width.snapshot_record.periods_in_record"))
    E("gt.dx", "$" + BS + "Delta x = 0.083D$", 0.083, 0.0005, meta(V5, "dx_over_D"))
    E("gt.dt", "$" + BS + "Delta " + BS + "tstar = 0.19$", 0.19, 0.005,
      meta(V5, "snapshot_dt_star"))

    # validation table
    E("gt.Cd", "& 1.3634 & 1.318--1.336", 1.3634, 0.00005, meta(V5, "Cd_mean"))
    E("gt.Cd_err", "& 1.3634 & 1.318--1.336 & $+3.5" + PCT + "$", 3.5, 0.05,
      meta(V5, "Cd_mean"), vs=QU_CD)
    E("gt.St", "& 0.1648 & 0.164--0.166", 0.1648, 0.00005,
      dct(SWEEP5, "100", "St_vorticity"))
    E("gt.St_err", "& 0.1648 & 0.164--0.166 & $-0.04" + PCT + "$", -0.04, 0.005,
      dct(SWEEP5, "100", "St_vorticity"), vs=QU_ST)
    E("gt.Cl", "& 0.2362 & 0.222--0.235", 0.2362, 0.00005, meta(V5, "Cl_rms"))
    E("gt.Cl_err", "& 0.2362 & 0.222--0.235 & $+6.2" + PCT + "$", 6.2, 0.05,
      meta(V5, "Cl_rms"), vs=QU_CL)
    E("gt.instrument_agree", "give the same value to within $0.002" + PCT + "$",
      0.002, 0.001, dct(SWEEP5, "100", "St_instrument_disagreement_pct"),
      note="A bound ('within'): measured 0.0016 %.")
    E("gt.Cd_D32", "geometry gives $+2.3" + PCT + "$", 2.3, 0.05,
      meta("geo_long", "Cd_mean"), vs=QU_CD)

    # the earlier dataset, as history
    E("gt.v4_St", "It read $" + BS + "St = 0.1595$", 0.1595, 0.00005,
      dct("runs/re_sweep/summary.json", "100", "St_vorticity"))
    E("gt.v4_St_err", "It read $" + BS + "St = 0.1595$ ($-3.3" + PCT + "$)", -3.3,
      0.05, dct("runs/re_sweep/summary.json", "100", "St_vorticity"), vs=QU_ST)
    E("gt.v4_Cd", "and $" + BS + "Cd = 1.3798$", 1.3798, 0.00005,
      meta("re100_v4", "Cd_mean"))
    E("gt.v4_claim", "1.37--1.38 to $0.3" + PCT + "$", 0.3, 0.05,
      meta("re100_v4", "Cd_mean"), vs=1.375,
      note="The earlier draft's claim, against the midpoint of its uncited "
           "range. Reproduced to show what was claimed, not as a validation.")

    # the duct mode
    E("ac.St_a", "$" + BS + "St_a = (c_s/2H)(D/U) = 0.1155$", 0.1155, 0.00005,
      path(DM, "acoustic.St_resonance"))
    E("ac.short", "sheds at $" + BS + "St = 0.1600$ over 20 recorded", 0.1600,
      0.00005, lst("runs/st_instruments.json", "St_vorticity",
                   tag="acoustic_short"))
    E("ac.long", "and at $0.1132$ over 38", 0.1132, 0.00005,
      lst("runs/st_tsponge.json", "St_vorticity", tag="acoustic_long"))
    E("ac.viscous", "does nothing here (0.1131)", 0.1131, 0.00005,
      lst("runs/st_tsponge.json", "St_vorticity", tag="sponge_long"))
    E("ac.visc_absorb", "removes about $0.04" + PCT + "$ of the", 0.04, 0.005,
      path(DM, "viscous_layer.pct_absorbed_per_pass"))
    E("ac.released", "releases the wake ($0.1694$ over the same", 0.1694,
      0.00005, lst("runs/st_tsponge.json", "St_vorticity", tag="tsponge_long"))
    E("ac.spread_cd", "$" + BS + "Cd$ moves by $0.1" + PCT + "$ and", 0.1, 0.05,
      path(DM, "sponge_sensitivity.Cd_spread_pct"))
    E("ac.spread_st", "$" + BS + "St$ by $0.2" + PCT + "$, and 38 and 120", 0.2,
      0.05, path(DM, "sponge_sensitivity.St_spread_pct"))
    E("ac.hong_band", "$" + BS + "St/" + BS + "St_a " + BS + "le 1.14$", 1.14,
      0.005, path(DM, "benchmark.literature.hong2020_lockin_ratio_max"))
    E("ac.ratio_100", "At $" + BS + "Rey = 100$ the ratio here is $1.43$", 1.43,
      0.005, ratio(dct(SWEEP5, "100", "St_vorticity"),
                   path(DM, "acoustic.St_resonance"), mode="ratio"))

    # geometry cross
    E("geo.inlet_effect", "lowers $" + BS + "Cd$ by\n", -4.7, 0.05,
      ratio(meta("geo_long", "Cd_mean"), meta("geo_base", "Cd_mean")),
      note="Printed as a magnitude ('lowers ... by 4.7 %'); the artifact "
           "gives the signed change.")
    E("geo.base_cd", "from $+7.4" + PCT + "$ to $+2.3" + PCT + "$ against", 7.4,
      0.05, meta("geo_base", "Cd_mean"), vs=QU_CD)
    E("geo.long_cd", "from $+7.4" + PCT + "$ to $+2.3" + PCT + "$ against", 2.3,
      0.05, meta("geo_long", "Cd_mean"), vs=QU_CD)
    E("geo.base_st", "$+3.0" + PCT + "$ to $+0.3" + PCT + "$; raising", 3.0, 0.05,
      meta("geo_base", "St"), vs=QU_ST)
    E("geo.long_st", "$+3.0" + PCT + "$ to $+0.3" + PCT + "$; raising", 0.3, 0.05,
      meta("geo_long", "St"), vs=QU_ST)
    E("geo.height_xu8", "moves $" + BS + "Cd$ by $0.36" + PCT + "$", -0.36, 0.005,
      ratio(meta("geo_tall", "Cd_mean"), meta("geo_base", "Cd_mean")))
    E("geo.height_xu25", "and by $0.07" + PCT + "$ once the inlet", -0.07, 0.005,
      ratio(meta("geo_tall_long", "Cd_mean"), meta("geo_long", "Cd_mean")))
    E("geo.posdziech", "require more than $20D$", 20, 0.5,
      path(DM, "benchmark.literature.posdziech2007_upstream_min_d"))

    # resolution
    E("res.D24", "$" + BS + "Cd = 1.4334$, $1.4139$ and $1.4147$", 1.4334, 0.00005,
      meta("res_D24", "Cd_mean"))
    E("res.D32", "$" + BS + "Cd = 1.4334$, $1.4139$ and $1.4147$", 1.4139, 0.00005,
      meta("res_D32", "Cd_mean"))
    E("res.D48", "$" + BS + "Cd = 1.4334$, $1.4139$ and $1.4147$", 1.4147, 0.00005,
      meta("res_D48", "Cd_mean"))
    E("res.D24_cd_high", "with $D = 24$ about $1.4" + PCT + "$", 1.4, 0.05,
      ratio(meta("res_D24", "Cd_mean"), meta("res_D32", "Cd_mean")))
    E("res.D24_st", "within $0.3" + PCT + "$ in $" + BS + "St$", -0.3, 0.05,
      ratio(meta("res_D24", "St"), meta("res_D32", "St")),
      note="'Within 0.3 %': the signed change is -0.27 %.")

    # divergence
    E("div.rms", "its r.m.s.\\ is $2.46" + PCT + "$ of", 2.46, 0.005,
      path(DM, "dataset.div_over_grad_pct"))
    E("div.ma2", "against $" + BS + "mathrm{Ma}^2 = 0.75" + PCT + "$.", 0.75, 0.005,
      path(DM, "dataset.ma_squared_pct"))

    # methodological note
    E("gt.instrument_agree2", "the three agree to $0.002" + PCT + "$", 0.002, 0.001,
      dct(SWEEP5, "100", "St_instrument_disagreement_pct"))
    E("gt.v4_disagree", "disagreed by about $1" + PCT + "$ because the lift", 1.0,
      0.5, ratio(meta("re100_v4", "St"),
                 dct("runs/re_sweep/summary.json", "100", "St_vorticity")),
      note="v4 lift St 0.1611 against vorticity 0.1595: +1.0 %.")


# ---------------------------------------------------------------------------
# Section 3 -- the unsteady operator (unchanged by the regeneration)
# ---------------------------------------------------------------------------

def section3():
    OPV = "runs/operator_verification.json"
    E("operator.taylor_green", "& $1.6" + BS + "times 10^{-16}$", 1.6e-16, 5e-18,
      path(OPV, "test_taylor_green_is_annihilated.value"))
    E("operator.backends", "& $2.8" + BS + "times 10^{-17}$", 2.8e-17, 5e-19,
      path(OPV, "test_backends_agree_on_a_network.value"))
    E("operator.steady", "& $5.6" + BS + "times 10^{-17}$", 5.6e-17, 5e-19,
      path(OPV, "test_steady_limit_matches_shipped_operator.value"))
    E("operator.speedup", "speed-up of the fast backend & ${" + BS + "approx}2.5"
      + BS + "times$", 2.5, 0.25, path(OPV, "benchmark_backends.value"),
      note="Wall-clock ratio; tolerance is run-to-run variance (2.52, 2.58), "
           "not printed precision.")


# ---------------------------------------------------------------------------
# Sections 4-5 -- the collapse and the ablation (re100_v5, chained ICs)
# ---------------------------------------------------------------------------

PA = "runs/v5/phase_average.json"
PH = "runs/v5/phase/summary.json"


def section4():
    row = ("shedding retained & 41.1" + PCT + " & 19.3" + PCT + " & 42.5" + PCT
           + " & 17.5" + PCT + " & " + BS + "textbf{30.1" + PCT + "}")
    for k, v in enumerate((41.1, 19.3, 42.5, 17.5)):
        E(f"collapse.phase{k}", row, v, 0.05,
          path(PA, f"summary.baseline.shedding_by_phase.{k}"), unit="pct")
    E("collapse.mean", row, 30.1, 0.05, path(PA, "summary.baseline.shedding_mean"),
      unit="pct")
    E("collapse.range_lo", "keeps between 17" + PCT + " and 42" + PCT, 17.0, 0.5,
      path(PA, "summary.baseline.shedding_phase_mean_min"), unit="pct",
      note="Range of the four phase means, 17.49 to 42.49 %.")
    E("collapse.range_hi", "keeps between 17" + PCT + " and 42" + PCT, 42.0, 0.5,
      path(PA, "summary.baseline.shedding_phase_mean_max"), unit="pct")
    E("collapse.l2", "with a field error of 0.17:", 0.17, 0.005,
      path(PA, "summary.baseline.rel_l2_mean"))
    v4 = "the same configuration gives 9.1" + PCT + ", 25.6" + PCT + ", 13.1" + PCT
    for k, v in enumerate((9.1, 25.6, 13.1)):
        E(f"collapse.v4_phase{k}", v4, v, 0.05,
          path(PH, f"re100_v4.runs.{k}.shedding"), unit="pct")
    E("collapse.v4_phase3", "34.9" + PCT + " at the four phases", 34.9, 0.05,
      path(PH, "re100_v4.runs.3.shedding"), unit="pct")
    # equal-budget control: the plain PINN at 80 000 epochs
    b80 = "summary.baseline80k."
    E("ctrl.b80.mean", "the same network keeps 47.4" + PCT + " on average", 47.4,
      0.05, path(PA, b80 + "shedding_mean"), unit="pct")
    E("ctrl.b80.l2", "field error of 0.116.", 0.116, 0.0005,
      path(PA, b80 + "rel_l2_mean"))
    E("ctrl.compute_gain", "by 17.3 points phase-averaged", 17.3, 0.05,
      path(PA, "paired.compute (baseline80k vs baseline).mean_pts"))
    for k, v, lit in ((0, 61.6, "the half-cycle swing survives it: 61.6" + PCT),
                      (2, 59.8, "and 59.8" + PCT + " at phases 0 and"),
                      (1, 33.9, "33.9" + PCT + " and 34.4" + PCT + " at the other two"),
                      (3, 34.4, "33.9" + PCT + " and 34.4" + PCT + " at the other two")):
        E(f"ctrl.b80.phase{k}", lit, v, 0.05,
          path(PA, b80 + f"shedding_by_phase.{k}"), unit="pct")
    E("collapse.old_claim", "``10.6" + PCT + " of the wake''", 10.6, 0.05,
      lst("runs/ablation/summary.json", "shedding", config="baseline"), unit="pct",
      note="The earlier report's figure, from the v4 dataset at one phase.")


def section5():
    AB = "runs/v5/ablation/summary.json"
    E("abl.baseflow", "$L_2 = 2.2" + BS + "times 10^{-3}$", 0.0022, 0.00005,
      {"kind": "scalar", "file": "data/base_flow_re100_v5.json", "key": "rel_l2"})
    rows = [("baseline", "41.1", "0.173"), ("fourier", "36.9", "0.149"),
            ("decompose", "36.1", "0.086"), ("march", "72.2", "0.156"),
            ("all", "77.3", "0.052")]
    for cfg, sh, l2 in rows:
        lit = BS + "texttt{" + cfg + "}" + " " * (10 - len(cfg)) + f"& {sh}{PCT} & {l2}"
        E(f"abl.{cfg}.shedding", lit, float(sh), 0.05,
          lst(AB, "shedding", config=cfg), unit="pct")
        E(f"abl.{cfg}.l2", lit, float(l2), 0.0005,
          lst(AB, "rel_l2", config=cfg))
    E("abl.all+data.shedding", "& 99.1" + PCT + " & " + BS + "textbf{0.006}", 99.1,
      0.05, lst(AB, "shedding", config="all+data"), unit="pct")
    E("abl.all+data.l2", "& 99.1" + PCT + " & " + BS + "textbf{0.006}", 0.006,
      0.0005, lst(AB, "rel_l2", config="all+data"))

    # the phase-averaged core table
    core = {"baseline": (("41.1", "19.3", "42.5", "17.5"), "30.1", "0.171", False),
            "march": (("72.2", "42.1", "72.4", "40.1"), "56.7", "0.158", False),
            "all": (("76.4", "72.9", "77.8", "73.2"), "75.1", "0.031", True)}
    for cfg, (phases, mean, l2, bold) in core.items():
        if bold:
            tail = (" & " + BS + "textbf{" + mean + PCT + "} & " + BS + "textbf{"
                    + l2 + "}")
        else:
            tail = f" & {mean}{PCT} & {l2}"
        lit = (BS + "texttt{" + cfg + "}" + " " * (8 - len(cfg)) + " & "
               + " & ".join(p + PCT for p in phases) + tail)
        for k, v in enumerate(phases):
            E(f"core.{cfg}.phase{k}", lit, float(v), 0.05,
              path(PA, f"summary.{cfg}.shedding_by_phase.{k}"), unit="pct")
        E(f"core.{cfg}.mean", lit, float(mean), 0.05,
          path(PA, f"summary.{cfg}.shedding_mean"), unit="pct")
        E(f"core.{cfg}.l2", lit, float(l2), 0.0005,
          path(PA, f"summary.{cfg}.rel_l2_mean"))

    E("abl.text.fourier", "(37" + PCT + " and 36" + PCT + " against 41" + PCT, 37.0,
      0.5, lst(AB, "shedding", config="fourier"), unit="pct")
    E("abl.text.decompose", "(37" + PCT + " and 36" + PCT + " against 41" + PCT,
      36.0, 0.5, lst(AB, "shedding", config="decompose"), unit="pct")
    E("abl.text.baseline", "(37" + PCT + " and 36" + PCT + " against 41" + PCT,
      41.0, 0.5, lst(AB, "shedding", config="baseline"), unit="pct")
    E("abl.text.march_lo", "swing, between 40" + PCT + " and 72" + PCT, 40.0, 0.5,
      path(PA, "summary.march.shedding_phase_mean_min"), unit="pct")
    E("abl.text.march_hi", "swing, between 40" + PCT + " and 72" + PCT, 72.0, 0.5,
      path(PA, "summary.march.shedding_phase_mean_max"), unit="pct")
    E("abl.text.all_lo", "73--78" + PCT + " at every phase", 73.0, 0.5,
      path(PA, "summary.all.shedding_phase_mean_min"), unit="pct")
    E("abl.text.all_hi", "73--78" + PCT + " at every phase", 78.0, 0.5,
      path(PA, "summary.all.shedding_phase_mean_max"), unit="pct")
    E("abl.text.all_l2", "at a field\nerror of 0.031", 0.031, 0.0005,
      path(PA, "summary.all.rel_l2_mean"))

    # what the extra ground truth was worth
    E("dsic.march", "retain 93.0" + PCT + " (" + BS + "texttt{march})", 93.0, 0.05,
      lst("runs/v5/ablation_dsic/summary.json", "shedding", config="march"),
      unit="pct")
    E("dsic.all", "and 94.5" + PCT + " (" + BS + "texttt{all})", 94.5, 0.05,
      lst("runs/v5/ablation_dsic/summary.json", "shedding", config="all"),
      unit="pct")
    E("dsic.march_chained", "against 72.2" + PCT + " and 77.3" + PCT + " chained",
      72.2, 0.05, lst(AB, "shedding", config="march"), unit="pct")
    E("dsic.all_chained", "against 72.2" + PCT + " and 77.3" + PCT + " chained",
      77.3, 0.05, lst(AB, "shedding", config="all"), unit="pct")

    # equal compute
    CM = "paired.causality (march vs baseline80k)."
    CA = "paired.causality with the other two mechanisms (all vs fd8_80k)."
    r1 = ("& 56.7" + PCT + " & 47.4" + PCT + " & $+9.3$ & $+3.7$ to $+13.8$")
    E("budget.march", r1, 56.7, 0.05, path(PA, "summary.march.shedding_mean"),
      unit="pct")
    E("budget.b80", r1, 47.4, 0.05, path(PA, "summary.baseline80k.shedding_mean"),
      unit="pct")
    E("budget.march_gain", r1, 9.3, 0.05, path(PA, CM + "mean_pts"))
    E("budget.march_min", r1, 3.7, 0.05, path(PA, CM + "min_pts"))
    E("budget.march_max", r1, 13.8, 0.05, path(PA, CM + "max_pts"))
    r2 = ("& 75.1" + PCT + " & 46.6" + PCT + " & $+28.4$ & $+20.0$ to $+37.2$")
    E("budget.all", r2, 75.1, 0.05, path(PA, "summary.all.shedding_mean"),
      unit="pct")
    E("budget.fd8", r2, 46.6, 0.05, path(PA, "summary.fd8_80k.shedding_mean"),
      unit="pct")
    E("budget.all_gain", r2, 28.4, 0.05, path(PA, CA + "mean_pts"))
    E("budget.all_min", r2, 20.0, 0.05, path(PA, CA + "min_pts"))
    E("budget.all_max", r2, 37.2, 0.05, path(PA, CA + "max_pts"))
    E("budget.text.compute", "17.3 points against 9.3", 17.3, 0.05,
      path(PA, "paired.compute (baseline80k vs baseline).mean_pts"))
    E("budget.text.causal", "17.3 points against 9.3", 9.3, 0.05,
      path(PA, CM + "mean_pts"))
    E("budget.text.fd8", "window keep 46.6" + PCT + " at", 46.6, 0.05,
      path(PA, "summary.fd8_80k.shedding_mean"), unit="pct")
    E("budget.text.all_gain", "lifts them by 28.4 points", 28.4, 0.05,
      path(PA, CA + "mean_pts"))
    E("budget.text.three_times", "Causality is worth three times", 3.0, 0.5,
      ratio(path(PA, CA + "mean_pts"), path(PA, CM + "mean_pts"), mode="ratio"),
      note="28.4 / 9.3 paired-mean points; printed as a round multiple.")

    # knobs
    S8 = "runs/v5/ablation_sigma8/summary.json"
    W1 = "runs/v5/ablation_wic100/summary.json"
    arrow = " $" + BS + "rightarrow$ "
    k1 = "& 77.3" + PCT + arrow + "76.4" + PCT
    E("knob.s8.all.before", k1, 77.3, 0.05, lst(AB, "shedding", config="all"),
      unit="pct")
    E("knob.s8.all.after", k1, 76.4, 0.05, lst(S8, "shedding", config="all"),
      unit="pct")
    k2 = "& 0.052" + arrow + BS + "textbf{0.031}"
    E("knob.s8.all.l2_before", k2, 0.052, 0.0005, lst(AB, "rel_l2", config="all"))
    E("knob.s8.all.l2_after", k2, 0.031, 0.0005, lst(S8, "rel_l2", config="all"))
    k3 = "& 36.9" + PCT + arrow + "37.6" + PCT + " & 0.149" + arrow + "0.136"
    E("knob.s8.fourier.before", k3, 36.9, 0.05,
      lst(AB, "shedding", config="fourier"), unit="pct")
    E("knob.s8.fourier.after", k3, 37.6, 0.05,
      lst(S8, "shedding", config="fourier"), unit="pct")
    E("knob.s8.fourier.l2_before", k3, 0.149, 0.0005,
      lst(AB, "rel_l2", config="fourier"))
    E("knob.s8.fourier.l2_after", k3, 0.136, 0.0005,
      lst(S8, "rel_l2", config="fourier"))
    k4 = "& 77.3" + PCT + arrow + "74.7" + PCT + " & 0.052" + arrow + "0.065"
    E("knob.wic.after", k4, 74.7, 0.05, lst(W1, "shedding", config="all"),
      unit="pct")
    E("knob.wic.l2_after", k4, 0.065, 0.0005, lst(W1, "rel_l2", config="all"))

    # supervision
    sup = [(0, "77.3", "0.052", AB, "all"),
           (100, "87.4", "0.020", "runs/v5/data_n100/summary.json", "all+data"),
           (500, "96.9", "0.010", "runs/v5/data_n500/summary.json", "all+data"),
           (2000, "98.1", "0.007", "runs/v5/data_n2000/summary.json", "all+data"),
           (4000, "99.1", "0.006", AB, "all+data")]
    for n, sh, l2, f, cfg in sup:
        lit = f"{n}" + " " * (5 - len(str(n))) + f"& {sh}{PCT} & {l2}"
        E(f"sup.{n}.shedding", lit, float(sh), 0.05,
          lst(f, "shedding", config=cfg), unit="pct")
        E(f"sup.{n}.l2", lit, float(l2), 0.0005, lst(f, "rel_l2", config=cfg))


# ---------------------------------------------------------------------------
# Section 6 -- accuracy over one period and five
# ---------------------------------------------------------------------------

def section6():
    ONE = "runs/v5/ablation_sigma8/shedding_accuracy.json"
    FIVE = "runs/v5/multiperiod/shedding_accuracy.json"

    def it(f, key):
        return {"kind": "item", "file": f, "index": 0, "key": key}

    E("acc5.St", "& --- & 0.1476 ($-10.4" + PCT + "$)", 0.1476, 0.00005,
      it(FIVE, "St_pinn"))
    E("acc5.St_err", "& --- & 0.1476 ($-10.4" + PCT + "$)", -10.4, 0.05,
      it(FIVE, "St_error_pct"))
    E("acc5.St_ref", "reference $" + BS + "St$ over the same span & --- & 0.1648",
      0.1648, 0.00005, it(FIVE, "St_lbm"))
    E("acc.crossings1", "zero crossings available & 2 & 9", 2, 0.5,
      it(ONE, "zero_crossings_pinn"))
    E("acc.crossings5", "zero crossings available & 2 & 9", 9, 0.5,
      it(FIVE, "zero_crossings_pinn"))
    E("acc1.amp", "wake amplitude           & 73.5" + PCT + " & 63.8" + PCT, 73.5,
      0.05, it(ONE, "amplitude_ratio"), unit="pct")
    E("acc5.amp", "wake amplitude           & 73.5" + PCT + " & 63.8" + PCT, 63.8,
      0.05, it(FIVE, "amplitude_ratio"), unit="pct")
    E("acc1.lag", "phase lag                & $0.1^" + BS + "circ$", 0.1, 0.05,
      it(ONE, "phase_lag_periods"), unit="deg_from_periods")
    E("acc5.phase_err",
      "phase error by the end   & --- & ${" + BS + "approx}" + BS + ",0.5$ period",
      0.5, 0.05,
      ratio(ratio(it(FIVE, "St_error_pct"), it(FIVE, "periods_covered"),
                  mode="product"), -100.0, mode="ratio"),
      note="Frequency error times the periods covered: -10.42 % x 5.0 periods. "
           "Used instead of the cross-correlation drift, which is ambiguous "
           "modulo one period when the two frequencies differ by a tenth.")
    E("acc1.l2", "velocity-field error     & 0.034 & 0.156", 0.034, 0.0005,
      it(ONE, "rel_l2_saved"))
    E("acc5.l2", "velocity-field error     & 0.034 & 0.156", 0.156, 0.0005,
      it(FIVE, "rel_l2_saved"))

    E("acc1.text_amp", "at 73.5" + PCT + " of its amplitude and a field error of "
      "0.034", 73.5, 0.05, it(ONE, "amplitude_ratio"), unit="pct")
    E("acc1.text_l2", "at 73.5" + PCT + " of its amplitude and a field error of "
      "0.034", 0.034, 0.0005, it(ONE, "rel_l2_saved"))
    E("acc1.ref_misread", "reads even the reference at 0.139 against the",
      0.139, 0.0005, it(ONE, "St_lbm"),
      note="The estimator over a two-crossing window, applied to the LBM signal.")
    E("acc1.true_st", "dataset's 0.1648.", 0.1648, 0.00005,
      dct(SWEEP5, "100", "St_vorticity"))
    E("acc5.text_err", "10.4" + PCT + " below the reference", -10.4, 0.05,
      it(FIVE, "St_error_pct"), note="Printed as a magnitude.")
    E("acc5.text_amp", "Amplitude decays\nto 63.8" + PCT, 63.8, 0.05,
      it(FIVE, "amplitude_ratio"), unit="pct")
    E("acc5.text_l2", "and the field error grows to 0.156", 0.156, 0.0005,
      it(FIVE, "rel_l2_saved"))
    E("acc5.old_claim", "$" + BS + "St$ within $0.1" + PCT + "$\nover five periods",
      0.1, 0.05, {"kind": "item", "file": "runs/multiperiod/shedding_accuracy.json",
                  "index": 0, "key": "St_error_pct"},
      note="The earlier report's figure, from the per-window-snapshot run on v4.")


# ---------------------------------------------------------------------------
# Framing -- introduction, limitations, note on method
# ---------------------------------------------------------------------------

def framing():
    FIVE = "runs/v5/multiperiod/shedding_accuracy.json"

    def it(key):
        return {"kind": "item", "file": FIVE, "index": 0, "key": key}

    half_cycle = ratio(ratio(it("St_error_pct"), it("periods_covered"),
                             mode="product"), -100.0, mode="ratio")
    AB = "runs/v5/ablation/summary.json"

    E("intro.three_quarters", "at three-quarters of its amplitude", 75.0, 2.5,
      path(PA, "summary.all.shedding_mean"), unit="pct",
      note="75.1 % phase-averaged, printed in words.")
    E("intro.half_cycle", "it drifts about half a", 0.5, 0.05, half_cycle,
      note="Same estimate as acc5.phase_err.")
    E("lim.three_quarters", "three-quarters of the wake amplitude over a period",
      75.0, 2.5, path(PA, "summary.all.shedding_mean"), unit="pct")
    E("lim.slow", "five periods it runs 10" + PCT + " slow", -10.0, 0.5,
      it("St_error_pct"), note="Printed as a magnitude.")
    E("lim.half_cycle", "ends half a cycle out of phase", 0.5, 0.05, half_cycle)
    E("lim.seed0", "gives 69.7" + PCT + " and", 69.7, 0.05,
      lst(AB, "shedding", config="march", seed=0), unit="pct")
    E("lim.seed1", "74.6" + PCT + " on its two seeds", 74.6, 0.05,
      lst(AB, "shedding", config="march", seed=1), unit="pct")
    E("lim.swing_lo", "against a swing from 40" + PCT + " to 72" + PCT, 40.0, 0.5,
      path(PA, "summary.march.shedding_phase_mean_min"), unit="pct")
    E("lim.swing_hi", "against a swing from 40" + PCT + " to 72" + PCT, 72.0, 0.5,
      path(PA, "summary.march.shedding_phase_mean_max"), unit="pct")
    E("lim.lockin", "$" + BS + "St/" + BS + "St_a = 1.43$, outside", 1.43, 0.005,
      ratio(dct(SWEEP5, "100", "St_vorticity"),
            path(DM, "acoustic.St_resonance"), mode="ratio"))
    # abstract
    S = "summary."
    CM = "paired.causality (march vs baseline80k).mean_pts"
    CA = "paired.causality with the other two mechanisms (all vs fd8_80k).mean_pts"
    CC = "paired.compute (baseline80k vs baseline).mean_pts"
    E("abs.St", "$0.04" + PCT + "$, $" + BS + "Cd$ $3.5" + PCT + "$ high", -0.04, 0.005,
      path(DM, "benchmark.per_Re.100.St_error_pct"), note="Printed as a magnitude.")
    E("abs.Cd", "$0.04" + PCT + "$, $" + BS + "Cd$ $3.5" + PCT + "$ high", 3.5, 0.05,
      path(DM, "benchmark.per_Re.100.Cd_error_pct"))
    E("abs.baseline", "30.1" + PCT + " of the reference wake amplitude", 30.1, 0.05,
      path(PA, S + "baseline.shedding_mean"), unit="pct")
    E("abs.march", "56.7" + PCT + ", although at equal training budget it adds only 9.3",
      56.7, 0.05, path(PA, S + "march.shedding_mean"), unit="pct")
    E("abs.causal", "56.7" + PCT + ", although at equal training budget it adds only 9.3",
      9.3, 0.05, path(PA, CM))
    E("abs.compute", "the 17.3 that training longer adds", 17.3, 0.05, path(PA, CC))
    E("abs.all", "phase: 75.1" + PCT + " retained, at a velocity-field error of 0.031",
      75.1, 0.05, path(PA, S + "all.shedding_mean"), unit="pct")
    E("abs.all_l2", "phase: 75.1" + PCT + " retained, at a velocity-field error of 0.031",
      0.031, 0.0005, path(PA, S + "all.rel_l2_mean"))
    E("abs.slow", "periods that configuration sheds 10.4" + PCT + " slow", -10.4, 0.05,
      it("St_error_pct"), note="Printed as a magnitude.")
    E("abs.half_cycle", "ends about half a cycle out of", 0.5, 0.05, half_cycle)

    # conclusions
    E("concl.lo", "17" + PCT + " and 42" + PCT + " of the wake depending", 17.0, 0.5,
      path(PA, S + "baseline.shedding_phase_mean_min"), unit="pct")
    E("concl.hi", "17" + PCT + " and 42" + PCT + " of the wake depending", 42.0, 0.5,
      path(PA, S + "baseline.shedding_phase_mean_max"), unit="pct")
    E("concl.baseline", "start phase, 30.1" + PCT + " on", 30.1, 0.05,
      path(PA, S + "baseline.shedding_mean"), unit="pct")
    E("concl.b80", "raises that to 47.4" + PCT + " and leaves", 47.4, 0.05,
      path(PA, S + "baseline80k.shedding_mean"), unit="pct")
    E("concl.march", "on its own reaches 56.7" + PCT, 56.7, 0.05,
      path(PA, S + "march.shedding_mean"), unit="pct")
    E("concl.causal", "only 9.3 points. Fourier", 9.3, 0.05, path(PA, CM))
    E("concl.all", "keep 75.1" + PCT + " at every start", 75.1, 0.05,
      path(PA, S + "all.shedding_mean"), unit="pct")
    E("concl.all_l2", "phase, at a field error of 0.031, and there marching is worth 28.4",
      0.031, 0.0005, path(PA, S + "all.rel_l2_mean"))
    E("concl.all_gain", "phase, at a field error of 0.031, and there marching is worth 28.4",
      28.4, 0.05, path(PA, CA))
    E("concl.lag", "$0.1^" + BS + "circ$ over one period", 0.1, 0.05,
      {"kind": "item", "file": "runs/v5/ablation_sigma8/shedding_accuracy.json",
       "index": 0, "key": "phase_lag_periods"}, unit="deg_from_periods")
    E("concl.slow", "over five it sheds 10.4" + PCT + " slow", -10.4, 0.05,
      it("St_error_pct"), note="Printed as a magnitude.")
    E("concl.half_cycle", "about half a cycle behind", 0.5, 0.05, half_cycle)
    E("concl.sup100", "retention to 87.4" + PCT + " and 500 to 96.9" + PCT, 87.4, 0.05,
      lst("runs/v5/data_n100/summary.json", "shedding", config="all+data"),
      unit="pct")
    E("concl.sup500", "retention to 87.4" + PCT + " and 500 to 96.9" + PCT, 96.9, 0.05,
      lst("runs/v5/data_n500/summary.json", "shedding", config="all+data"),
      unit="pct")

    E("method.old_march", "causal marching alone retains 89.2" + PCT, 89.2, 0.05,
      lst("runs/ablation/summary.json", "shedding", config="march"), unit="pct",
      note="The earlier report's headline, v4 dataset, per-window snapshots, "
           "phase 0.")


SECTIONS = [section2, section3, section4, section5, section6, framing]


def resolve_lines(entries, lines):
    """Point each entry at the line holding its literal.

    A literal containing a newline spans two lines; it is matched against
    each line joined to the next and anchored to the first. A literal that
    is missing or ambiguous is left at line 0 and reported, so the problem
    surfaces when the manifest is built rather than only when it is checked.
    """
    problems = []
    for e in entries:
        lit = e["literal"]
        if "\n" in lit:
            hits = [i + 1 for i in range(len(lines) - 1)
                    if lit in lines[i] + "\n" + lines[i + 1]]
        else:
            hits = [i + 1 for i, ln in enumerate(lines) if lit in ln]
        if len(hits) >= 1:
            e["tex"] = hits[0]
        if len(hits) != 1:
            problems.append((e["id"], len(hits)))
    return problems


def main():
    for build in SECTIONS:
        build()

    ids = [e["id"] for e in ENTRIES]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"duplicate ids: {sorted(dup)}")

    lines = TEX.read_text(encoding="utf-8").splitlines()
    problems = resolve_lines(ENTRIES, lines)
    OUT.write_text(json.dumps({
        "_README": [
            "Generated by report/build_manifest.py -- edit that, not this.",
            "Every number printed in report/report.tex, with the artifact that",
            "regenerates it. Checked by experiments/verify_report.py.",
        ],
        "tex": "report/report.tex",
        "entries": ENTRIES}, indent=2) + "\n")
    print(f"wrote {len(ENTRIES)} entries to {OUT.relative_to(ROOT)}")
    for pid, n in problems:
        print(f"  literal for {pid}: {'not found' if n == 0 else f'{n} matches'}")


if __name__ == "__main__":
    main()
