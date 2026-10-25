#!/bin/bash
# Grid convergence is reached and the answer is still wrong. What is left?
#
#   D = 24   Cd 1.4334   St 0.1694   Cl' 0.2695
#   D = 32   Cd 1.4139   St 0.1698   Cl' 0.2588
#   D = 48   Cd 1.4147   St 0.1698   Cl' 0.2600
#
# D = 32 and D = 48 agree to 0.06 % in Cd and to four figures in St, so the
# solution is grid converged -- and it converges to Cd = 1.414 against Qu et
# al.'s 1.317, St = 0.1698 against 0.1649, Cl' = 0.260 against 0.2224. With
# the acoustics absorbed, 120 recorded periods, sponge independence checked
# and now the grid converged, that discrepancy is real and systematic.
#
# Two geometric candidates survive, and both were half-tested before under
# conditions now known to be contaminated.
#
# BLOCKAGE. The absorbing layer softens the walls but does not move them: it
# occupies 8 D of the 50 D height at each side, so the clear interior is
# 34 D, an effective half-width of 17 D. Qu et al., citing Kumar and Mittal,
# put the threshold for negligible blockage at H > 100. Nothing in this
# project has ever run near that.
#
# UPSTREAM LENGTH. Fixed at 8 diameters throughout, where Posdziech and
# Grundmann require more than 20. The one earlier test of this gave
# Cd 1.3815 -> 1.3658, but both of those runs were acoustically locked, so
# the comparison meant nothing.
#
# D = 32 throughout: grid converged, and a quarter the cost of D = 48. 40
# recorded periods, past the 38 at which the answer stopped moving.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

while pgrep -u "$USER" -f "lbm_cylinder\.py|run_ablation\.py" > /dev/null; do sleep 20; done
echo "### starting $(date)"

geo () {
  local label="$1" height="$2" length="$3" upstream="$4"
  echo "### $label start $(date)  (H=$height L=$length Xu=$upstream)"
  $PY -u src/cfd/lbm_cylinder.py --re 100 --diameter 32 --u 0.05 \
      --height "$height" --length "$length" --upstream "$upstream" \
      --sponge 8 --sponge-target 0.05 --warmup-periods 15 --steps 214000 \
      --no-snapshots --tag "$label" --outdir data 2>&1 | grep -v nvptx_compiler
  echo "### $label exit ${PIPESTATUS[0]} ($(date))"
}

geo geo_base      50  30  8     # the current setup, D = 32 reference point
geo geo_tall     120  30  8     # blockage alone
geo geo_long      50  60 25     # upstream alone
geo geo_tall_long 120 60 25     # both

echo "### measuring $(date)"
$PY -u experiments/measure_st.py res_D32 geo_base geo_tall geo_long \
    geo_tall_long --json runs/st_geometry.json 2>&1 | grep -v nvptx_compiler

echo QUEUE14_DONE $(date)
