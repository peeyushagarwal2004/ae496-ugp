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
    E("res.D24_cd_high", "about $1.4" + PCT + "$ high in", 1.4, 0.05,
      ratio(meta("res_D24", "Cd_mean"), meta("res_D32", "Cd_mean")))
    E("res.D24_st", "within $0.3" + PCT + "$ in $" + BS + "St$", -0.3, 0.05,
      ratio(meta("res_D24", "St"), meta("res_D32", "St")),
      note="'Within 0.3 %': the signed change is -0.27 %.")

    # divergence
    E("div.rms", "its r.m.s.\\ is $2.47" + PCT + "$ of", 2.47, 0.005,
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


SECTIONS = [section2, section3]


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
