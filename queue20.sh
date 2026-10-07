#!/bin/bash
# Architecture study, PINN track, GRU only.
#
# The queue19 pilot (phase 0, seed 0) answered the PINNsFormer question:
# single-window it retains less shedding than the MLP, and marched it is
# only marginally ahead of the marched MLP at ~11x the compute (~5 h per
# run). The remaining 14 PINNsFormer runs (~66 GPU-hours) are dropped.
#
# pinns_gru finishes the grid: all four quarter-period phases x seeds 0-1,
# single-window and marched, under the queue17 protocol (re100_v5, 20000
# epochs per window, --ic-from-network). The pilot's k0_s0 runs are
# skipped. 14 runs, ~8 h on GPU 1.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5

while ! grep -q QUEUE19_DONE runs/v5/queue19.log 2>/dev/null; do sleep 60; done
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

for seed in 0 1; do
  for k in 0 1 2 3; do
    run gru       $k $seed --net pinns_gru --window-len 6.0
    run gru_march $k $seed --net pinns_gru --windows 4 --window-len 1.5
  done
done
echo QUEUE20_DONE $(date)
