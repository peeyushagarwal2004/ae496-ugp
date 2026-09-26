#!/bin/bash
# Does the absorbing layer buy back the long runs?
#
# Two things follow from the last round, and the sponge tests both.
#
# (A) LOCK-IN. Without a sponge, Re = 100 at H = 50 sheds at vorticity
#     St = 0.1600 over 20 recorded periods and 0.1132 over 38 -- the wake
#     itself entrained by the transverse resonance at 0.1155. sponge_long
#     repeats the 38-period run with 8 diameters of absorbing layer at each
#     free-slip wall. If the wake now stays near 0.16, the standing wave was
#     the cause and the mechanism is demonstrated rather than argued.
#     acoustic_long, already committed, is the control.
#
# (B) CONVERGENCE. Qu et al. average over at least 80 shedding cycles; every
#     dataset in this project has had 20, because the flow stops being the
#     flow after about 30. If the sponge holds the wake, 120 cycles becomes
#     available and Cd can finally be compared with the literature on equal
#     terms. sponge_conv and sponge_conv_H120 do that at both heights;
#     nosponge_conv is the control that should be corrupted.
#
# A local CPU check at D = 16, H = 20 confirmed the layer does what it should:
# density spread halved from 2.5 % to 1.2 % while St moved 0.1514 -> 0.1513
# and Cd 1.5048 -> 1.5021. It absorbs sound and leaves the hydrodynamics
# alone. The layer is 8 D at each wall here, leaving 34 D of the 50 D height
# clear, and the wake occupies about +/- 3 D.

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
  $PY -u src/cfd/lbm_cylinder.py --re 100 --diameter 24 --u 0.05 --length 30 \
      "$@" 2>&1 | grep -v nvptx_compiler
  echo "### $label exit ${PIPESTATUS[0]} ($(date))"
}

# (A) the lock-in test: same 30 + 38 periods as acoustic_long, plus a sponge
lbm sponge_long --height 50 --sponge 8 --warmup-periods 30 --steps 200000 \
    --snaps-per-period 10 --tag sponge_long --outdir data

# (B) long averaging. 15 warmup + ~120 recorded periods at ~2921 steps each.
lbm sponge_conv      --height 50  --sponge 8 --warmup-periods 15 --steps 394000 \
    --no-snapshots --tag sponge_conv --outdir data
lbm nosponge_conv    --height 50             --warmup-periods 15 --steps 394000 \
    --no-snapshots --tag nosponge_conv --outdir data
lbm sponge_conv_H120 --height 120 --sponge 8 --warmup-periods 15 --steps 394000 \
    --no-snapshots --tag sponge_conv_H120 --outdir data

echo "### measuring $(date)"
$PY -u experiments/measure_st.py acoustic_short acoustic_long sponge_long \
    sponge_conv nosponge_conv sponge_conv_H120 \
    --json runs/st_sponge.json 2>&1 | grep -v nvptx_compiler

echo QUEUE11_DONE $(date)
