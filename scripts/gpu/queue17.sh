#!/bin/bash
# Phase-averaged comparison of the three configurations the headline rests on.
#
# queue16 showed the single-window baseline's "shedding retained" swings with
# nothing but the starting phase of the initial condition: 9.1, 25.6, 13.1 and
# 34.9 % at quarter-period steps on the same dataset with the same seed. Every
# configuration in the ablation starts at the same instant, so comparisons at
# that instant are fair -- but a single phase can flatter or penalise one
# configuration against another by chance, and no absolute retention figure
# from one phase means much.
#
# So the headline comparison is repeated at all four quarter-period phases,
# two seeds each: baseline, march and all at sigma 8, the last two with
# chained initial conditions (one ground-truth slice per run). Phase 0 is not
# re-run here; it is exactly the configuration runs/v5/ablation (baseline,
# march) and runs/v5/ablation_sigma8 (all) already produce, at the default
# start time. experiments/phase_average.py gathers both.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5

while ! grep -q QUEUE16_DONE runs/v5/queue16.log 2>/dev/null; do sleep 30; done

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
      --t-start $ts --log-every 5000 --out $out "$@" 2>&1 \
      | grep --line-buffered -E "shedding retained|relative L2|Traceback|Error" | tail -3
}

for k in 1 2 3; do
  for seed in 0 1; do
    run baseline $k $seed --window-len 6.0
    run march    $k $seed --windows 4 --window-len 1.5 --ic-from-network
    run all      $k $seed --decompose --net fourier_mlp --sigma 8 \
                          --windows 4 --window-len 1.5 --ic-from-network
  done
done
echo QUEUE17_DONE $(date)
