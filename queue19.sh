#!/bin/bash
# Architecture study: do a Transformer, an RNN or a CNN change either answer?
#
# (A) PINN TRACK -- does the architecture alone stop the collapse?
#
# PINNsFormer (Zhao et al., ICLR 2024) was proposed against PINN failure
# modes, so it is the obvious challenger to the claim that causal
# time-marching is what restores shedding. pinns_gru is the same pseudo-
# sequence model with a GRU in place of attention. Each runs single-window
# (architecture alone, against `baseline`) and marched (against `march`),
# under the exact protocol of queue17: re100_v5, 20000 epochs per window,
# --ic-from-network, all four quarter-period phases, two seeds. Phase 0 is
# included here because the main ablation never ran these networks.
#
# PINNsFormer at its published size is ~36x the MLP's step cost on CPU
# (pinns_gru ~8x), so this queue runs only a pilot -- phase 0, seed 0, all
# four configs. Its measured GPU timings decide whether the other 28 runs
# (queue20: phases 0-3 x seeds 0-1) are worth the shared card.
#
# (B) OPERATOR TRACK -- is "physics helps short, hurts long" an FNO artefact?
#
# The U-Net (CNN) and ConvGRU (RNN, PhyCRNet pattern) forecasters, trained
# exactly as runs/fno was: re100_v4, 200 epochs, rollout 10, eval 50, data
# and physics variants, two seeds.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5

while ! grep -q QUEUE18_DONE runs/v5/queue18.log 2>/dev/null; do sleep 60; done
echo "### starting $(date)"

read T0 PER < <($PY -c "
import json, numpy as np
t = np.load('data/cylinder_${TAG}.npz')['t']
st = json.load(open('runs/re_sweep_v5/summary.json'))['100']['St_vorticity']
print(float(t[0]), 1.0 / st)")
echo "t0 $T0  period $PER"

run () {   # config phase seed -- flags...
  local cfg=$1 k=$2 seed=$3; shift 3
  local out=runs/v5/phase_core/${cfg}_k${k}_s${seed}
  [ -f $out/results.json ] && { echo "skip $out"; return; }
  local ts=$($PY -c "print(round($T0 + $k * $PER / 4.0, 4))")
  echo "### $cfg phase $k/4 seed $seed  t_start=$ts  $(date)"
  $PY -u src/pinn/cylinder_unsteady.py --tag $TAG --epochs 20000 --seed $seed \
      --t-start $ts --log-every 5000 --ic-from-network --out $out "$@" 2>&1 \
      | grep --line-buffered -E "params=|ep  *(0|5000) |shedding retained|relative L2|Traceback|Error" | tail -8
}
pinn_all () {  # phase seed
  run tf        $1 $2 --net pinnsformer --window-len 6.0
  run tf_march  $1 $2 --net pinnsformer --windows 4 --window-len 1.5
  run gru       $1 $2 --net pinns_gru   --window-len 6.0
  run gru_march $1 $2 --net pinns_gru   --windows 4 --window-len 1.5
}

op () {   # arch
  echo "### operator $1 start $(date)"
  $PY -u experiments/run_fno.py --tag re100_v4 --epochs 200 --seeds 2 \
      --only data,physics --run-name fno_$1 --extra "--arch $1" 2>&1 \
      | grep --line-buffered -E "arch|params|ep  *(0|100|199) |eps_full|FAILED|Traceback|Error"
  echo "### operator $1 exit ${PIPESTATUS[0]} ($(date))"
}

op unet
op convgru
pinn_all 0 0                     # pilot
echo QUEUE19_DONE $(date)
