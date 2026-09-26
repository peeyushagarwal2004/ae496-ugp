#!/bin/bash
# Two questions the last round opened and could not close.
#
# (A) IS THE ACOUSTIC LOCK REAL, OR ONLY IN THE LIFT?
#
# At the production budget Re = 100 reads St = 0.1609. The same configuration
# run for 30 warmup and 38 recorded periods instead of 15 and 20 reads 0.1132,
# which is the transverse resonance at 0.1155. Both numbers come from the lift,
# and sec:acoustic says the lift is acoustically modulated and not to be
# trusted for frequency -- so those two runs do not yet show the WAKE locking,
# only the force signal. Wake vorticity is blind to irrotational sound, so it
# decides the question, and it needs snapshots, which the last round did not
# save. acoustic_short and acoustic_long are the same run at the two durations
# with snapshots kept. If vorticity follows the lift down to ~0.115 in the long
# run, the wake genuinely locks and run length, not Reynolds number, is the
# controlling variable -- which is not what the report's separation table says.
# If vorticity stays near 0.16 while the lift falls, then only the force signal
# is contaminated and the report's low-Re lock-in story needs different
# evidence than it currently has.
#
# (B) IF NOT CONFINEMENT, WHAT?
#
# Holding the production budget and varying only the domain height gave
# Cd = 1.3815, 1.4439, 1.3939 at H = 50, 80, 120 against Qu et al.'s 1.317 --
# no convergence toward the literature value, so the confinement attribution
# now written into sec:lbm is unsupported. Every run in this project has used
# an upstream distance of 8 diameters. Posdziech and Grundmann, as reported by
# Qu et al., require more than 20. upstream25 and upstream25_H120 vary that
# instead, holding everything else fixed.
#
# Snapshots are kept only for (A). --snaps-per-period 10 is plenty to resolve
# a frequency and keeps the long run's array manageable.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

while pgrep -u "$USER" -f "lbm_cylinder\.py|run_ablation\.py" > /dev/null; do sleep 20; done
echo "### starting $(date)"

lbm () {
  local label="$1"; shift
  echo "### $label start $(date)"
  $PY -u src/cfd/lbm_cylinder.py --re 100 --diameter 24 --u 0.05 "$@" 2>&1 \
      | grep -v nvptx_compiler
  echo "### $label exit ${PIPESTATUS[0]} ($(date))"
}

# (A) same configuration, two durations, snapshots kept
lbm acoustic_short --height 50 --length 30 --warmup-periods 15 --steps 102000 \
    --snaps-per-period 10 --tag acoustic_short --outdir data
lbm acoustic_long  --height 50 --length 30 --warmup-periods 30 --steps 200000 \
    --snaps-per-period 10 --tag acoustic_long  --outdir data

# (B) upstream distance, the one geometric parameter never varied
lbm upstream25      --height 50  --length 60 --upstream 25 --warmup-periods 15 \
    --steps 102000 --no-snapshots --tag upstream25 --outdir data
lbm upstream25_H120 --height 120 --length 60 --upstream 25 --warmup-periods 15 \
    --steps 102000 --no-snapshots --tag upstream25_H120 --outdir data

echo "### measuring $(date)"
$PY -u experiments/measure_st.py acoustic_short acoustic_long confm_H50 \
    upstream25 upstream25_H120 --json runs/st_instruments.json 2>&1 \
    | grep -v nvptx_compiler

echo QUEUE10_DONE $(date)
