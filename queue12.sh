#!/bin/bash
# Target-state damping: does it break the lock-in, and can it be trusted?
#
# The viscosity ramp failed, and failed for a reason that was predictable and
# should have been predicted: viscous absorption goes as nu * omega^2, the
# transverse mode has a wavelength of twice the domain height, and an
# eight-diameter layer at twenty times the viscosity therefore absorbs about
# 0.04 % per pass. Measured outcome, three runs at H = 50, all identical:
# St = 0.1131 with the layer, 0.1132 without.
#
# Relaxing the populations toward a fixed free-stream equilibrium damps at a
# rate set by sigma per step regardless of frequency, which is what a
# long-wavelength mode needs. A CPU check at H = 20 showed it works -- density
# spread 2.3 % -> 0.9 % -- but also moved Cd from 1.5021 to 1.3875, which is
# the warning sign this queue is built around.
#
# THE HONESTY PROBLEM. Pulling toward the free stream near the walls is a soft
# far-field condition, not purely an acoustic treatment. It relaxes the
# blockage as well as the sound. Cd falling toward the literature value could
# therefore mean the domain now emulates an unbounded flow -- the thing we
# want -- or simply that the sponge is imposing the answer. The difference is
# testable: if Cd depends materially on the layer's width or strength, the
# sponge is setting it. tsponge_narrow and tsponge_weak exist to find that
# out, and their result decides whether any Cd from this method can be
# quoted at all.

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
      --height 50 "$@" 2>&1 | grep -v nvptx_compiler
  echo "### $label exit ${PIPESTATUS[0]} ($(date))"
}

# the lock-in test, directly against acoustic_long (0.1132) and sponge_long (0.1131)
lbm tsponge_long --sponge 8 --sponge-target 0.05 --warmup-periods 30 \
    --steps 200000 --snaps-per-period 10 --tag tsponge_long --outdir data

# 120 recorded periods, the averaging the literature uses
lbm tsponge_conv --sponge 8 --sponge-target 0.05 --warmup-periods 15 \
    --steps 394000 --no-snapshots --tag tsponge_conv --outdir data

# sensitivity: if Cd moves with these, the sponge is setting it, not revealing it
lbm tsponge_narrow --sponge 4 --sponge-target 0.05 --warmup-periods 15 \
    --steps 394000 --no-snapshots --tag tsponge_narrow --outdir data
lbm tsponge_weak   --sponge 8 --sponge-target 0.02 --warmup-periods 15 \
    --steps 394000 --no-snapshots --tag tsponge_weak --outdir data
lbm tsponge_strong --sponge 8 --sponge-target 0.10 --warmup-periods 15 \
    --steps 394000 --no-snapshots --tag tsponge_strong --outdir data

echo "### measuring $(date)"
$PY -u experiments/measure_st.py acoustic_long sponge_long tsponge_long \
    tsponge_conv tsponge_narrow tsponge_weak tsponge_strong nosponge_conv \
    --json runs/st_tsponge.json 2>&1 | grep -v nvptx_compiler

echo QUEUE12_DONE $(date)
