#!/bin/bash
# Seeds 2-4 for the phase-averaged headline and its equal-budget controls.
#
# Every cell of the headline (report Section 4, experiments/phase_average.py)
# is two seeds per start phase. The seed spread is small beside the effects
# reported, but n = 2 puts no interval on a mean. This adds seeds 2, 3 and 4
# under exactly the queue17/queue18 protocol, so each config ends with five
# seeds at each of the four quarter-period phases (20 runs per config).
#
#   baseline     1 window x 6 t*, 20k epochs                     ~7 min
#   march        4 chained windows x 1.5 t*, 20k each            ~26 min
#   all          march + decompose + fourier_mlp sigma 8         ~35 min
#   baseline80k  baseline at 80k epochs (equal budget to march)  ~28 min
#   fd8_80k      decompose + fourier sigma 8, 1 window, 80k      ~45 min
#
# Phase 0 for seeds 0-1 came from runs/v5/ablation{,_sigma8} and
# runs/v5/ctrl_budget80k at the default start time; here phase 0 goes to
# phase_core/*_k0_* with --t-start set to that same first snapshot time.
#
# Order: the three headline configs for all new seeds first (~13.6 h), then
# the two controls (~14.6 h). Every run skips if its results.json exists.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5
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
  $PY -u src/pinn/cylinder_unsteady.py --tag $TAG --seed $seed \
      --t-start $ts --log-every 10000 --out $out "$@" 2>&1 \
      | grep --line-buffered -E "shedding retained|relative L2|Traceback|Error" | tail -3
}

for seed in 2 3 4; do
  for k in 0 1 2 3; do
    run baseline $k $seed --epochs 20000 --window-len 6.0
    run march    $k $seed --epochs 20000 --windows 4 --window-len 1.5 --ic-from-network
    run all      $k $seed --epochs 20000 --decompose --net fourier_mlp --sigma 8 \
                          --windows 4 --window-len 1.5 --ic-from-network
  done
done
echo "### HEADLINE_SEEDS_DONE $(date)"

for seed in 2 3 4; do
  for k in 0 1 2 3; do
    run baseline80k $k $seed --epochs 80000 --window-len 6.0
    run fd8_80k     $k $seed --epochs 80000 --window-len 6.0 \
                             --decompose --net fourier_mlp --sigma 8
  done
done
echo QUEUE23_DONE $(date)
