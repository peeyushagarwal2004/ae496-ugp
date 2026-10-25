#!/bin/bash
# Phase-averaged equal-budget controls: is the marching gain causality or compute?
#
# At phase 0 on re100_v5, one ground-truth slice per run:
#   baseline, 1 window x 20k epochs       41.1 %
#   baseline, 1 window x 80k epochs       61.6 %   <- 4x compute alone: +20
#   march,    4 windows x 20k, chained    72.2 %   <- same compute, causal: +10
# On v4 the same control read 21.8 % against 89.2 %, but that march used four
# ground-truth slices and every number came from one starting phase, which is
# now known to move single-window retention by up to 4x. So the control is
# repeated at all four quarter-period phases, two seeds.
#
# Second pair, which the earlier ablation never ran: does marching still add
# anything once the other two mechanisms are present? "fd8" is Fourier features
# (sigma 8) plus the Reynolds decomposition in ONE window at 80k epochs -- the
# equal-budget counterpart of `all` at sigma 8, which is the same network
# marched over four chained windows at 20k each.
#
# Phase 0 of baseline-80k is runs/v5/ctrl_budget80k; everything else is here.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5

read T0 PER < <($PY -c "
import json, numpy as np
t = np.load('data/cylinder_${TAG}.npz')['t']
st = json.load(open('runs/re_sweep_v5/summary.json'))['100']['St_vorticity']
print(float(t[0]), 1.0 / st)")

run () {   # config phase seed -- flags...
  local cfg=$1 k=$2 seed=$3; shift 3
  local out=runs/v5/phase_core/${cfg}_k${k}_s${seed}
  [ -f $out/results.json ] && { echo "skip $out"; return; }
  local ts=$($PY -c "print(round($T0 + $k * $PER / 4.0, 4))")
  echo "### $cfg phase $k/4 seed $seed  t_start=$ts  $(date)"
  $PY -u src/pinn/cylinder_unsteady.py --tag $TAG --seed $seed \
      --t-start $ts --log-every 10000 --out $out "$@" 2>&1 \
      | grep --line-buffered -E "shedding retained|relative L2|Traceback|Error" | tail -3
}

for k in 0 1 2 3; do
  for seed in 0 1; do
    [ $k -gt 0 ] && run baseline80k $k $seed --epochs 80000 --window-len 6.0
    run fd8_80k $k $seed --epochs 80000 --window-len 6.0 \
        --decompose --net fourier_mlp --sigma 8
  done
done
echo QUEUE18_DONE $(date)
