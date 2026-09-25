#!/bin/bash
# Confinement series, and the two low-Re datasets that were overwritten.
#
# WHY THIS RUN EXISTS
#
# 1. The report attributes a systematic +2.7 % to +7.9 % Cd overshoot across
#    the sweep to domain confinement, on the strength of Qu et al. (2013)
#    reporting that blockage effects persist until H > 100 diameters. The
#    production datasets run at H = 50. That is an argument, not a measurement.
#    Extending the blockage series to H = 80 and H = 120 tests it directly: if
#    confinement is the cause, Cd should fall towards the literature value of
#    about 1.32 and St should barely move, because St is far less sensitive to
#    domain size. If Cd does not fall, the overshoot is a solver defect and the
#    report's validation section is wrong.
#
# 2. The runs at u = 0.05 for Re = 60 and 80 restore the two measurements that
#    report/numbers.json still cannot verify -- the acoustically locked wakes
#    at St = 0.114 and 0.113. Those datasets were overwritten in place when
#    they were regenerated at a lower lattice velocity, so the anomaly that
#    motivates the whole acoustic subsection currently rests on numbers nothing
#    can reproduce. These are cheap; they should never have been lost.
#
# --steps MATTERS. It defaults to 80000, but warmup alone is
# --warmup-periods (30) x the expected shedding period, which at u = 0.05 is
# about 2900 steps at Re = 100 and 3540 at Re = 60 -- so 87k to 106k steps of
# warmup against an 80k budget. The first attempt at this queue ran entirely
# inside warmup, recorded nothing, and reported St, Cd and Cl_rms as nan with
# exit status 0. 200k leaves 25-38 shedding periods of record in every case.
#
# All four record forces only. St and Cd come from the force history, which is
# kept every step regardless, and nothing here needs the velocity snapshots.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

STEPS=200000

run () {
  local label="$1"; shift
  echo "### $label start $(date)"
  $PY -u src/cfd/lbm_cylinder.py --steps $STEPS --no-snapshots "$@" 2>&1 \
      | grep -v nvptx_compiler
  echo "### $label exit ${PIPESTATUS[0]} ($(date))"
}

# -- confinement series at Re = 100, same resolution and lattice velocity as
#    the production dataset so that H is the only thing that changes ---------
run conf_H50  --re 100 --diameter 24 --u 0.05 --height 50  --length 30 \
              --tag conf_H50  --outdir data
run conf_H80  --re 100 --diameter 24 --u 0.05 --height 80  --length 30 \
              --tag conf_H80  --outdir data
run conf_H120 --re 100 --diameter 24 --u 0.05 --height 120 --length 30 \
              --tag conf_H120 --outdir data

# -- the overwritten low-Re datasets, at the ORIGINAL lattice velocity -------
run re60_u05 --re 60 --diameter 24 --u 0.05 --height 50 --length 30 \
             --tag re60_u05 --outdir data
run re80_u05 --re 80 --diameter 24 --u 0.05 --height 50 --length 30 \
             --tag re80_u05 --outdir data

echo QUEUE7_DONE $(date)
