#!/bin/bash
# Sparse periodic re-anchoring of the five-period marched chain.
#
# The data-free chain (runs/v5/multiperiod/all_s0: all, sigma 8, 20 quarter-
# period windows, --ic-from-network) sheds for five periods but 10.4 % slow,
# ending about half a cycle behind. Every window re-anchored to a fresh LBM
# snapshot held the phase -- at 20 x 4000 = 80 000 ground-truth IC points.
# Question: does a few supervised points every few windows hold it too?
#
# Same run as queue15 stage 7 except --n-data N --anchor-every K: windows
# 0, K, 2K, ... get N supervised (u, v) samples spread over their own time
# span; every other window is data-free. ICs stay chained throughout, so the
# only ground truth beyond the first slice is the anchor data.
#
#   name      N    K   anchors  data points  (vs 80 000 for per-window ICs)
#   a8_n100  100   8      3        300        two periods between anchors
#   a4_n100  100   4      5        500        once per period
#   a2_n100  100   2     10       1000        twice per period
#   a1_n100  100   1     20       2000        every window: the n=100 ceiling
#   a4_n500  500   4      5       2500        once per period, more points
#   free       0   -      0          0        data-free, second seed
#
# Seed 0 first (5 runs, ~7 h), accuracy scored; then seed 1 (6 runs, ~8 h).
# ~80 min per run on GPU 1. Every run skips if its results.json exists.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5
R=runs/v5/reanchor
BASE=$D/runs/v5/multiperiod/all_s0
echo "### starting $(date)"

WIN=$($PY -c "import json;s=json.load(open('runs/re_sweep_v5/summary.json'))['100'];print(round(0.25/s['St_vorticity'],4))")
echo "window length (quarter of the vorticity period): $WIN"

run () {   # name seed -- flags...
  local name=$1 seed=$2; shift 2
  local out=$R/${name}_s${seed}
  [ -f $out/results.json ] && { echo "skip $out"; return; }
  echo "### $name seed $seed  $(date)"
  $PY -u src/pinn/cylinder_unsteady.py --tag $TAG --epochs 20000 --seed $seed \
      --decompose --net fourier_mlp --sigma 8 --windows 20 --window-len $WIN \
      --ic-from-network --save-params --log-every 5000 --out $out "$@" 2>&1 \
      | grep --line-buffered -E "params=|window +[0-9]+ |shedding retained|relative L2|Traceback|Error"
}

score () {   # seed [extra absolute run dir] -- scored against the reference probe
  local seed=$1; shift
  # absolute paths: shedding_accuracy resolves relative ones under runs/<run-name>/
  local dirs=$(ls -d $D/$R/*_s$seed "$@" 2>/dev/null | paste -sd, -)
  $PY -u experiments/shedding_accuracy.py --runs "$dirs" --tag $TAG \
      --out $R/shedding_accuracy_s$seed.json 2>&1 | grep -v nvptx_compiler
}

for seed in 0 1; do
  run a8_n100 $seed --n-data 100 --anchor-every 8
  run a4_n100 $seed --n-data 100 --anchor-every 4
  run a2_n100 $seed --n-data 100 --anchor-every 2
  run a1_n100 $seed --n-data 100 --anchor-every 1
  run a4_n500 $seed --n-data 500 --anchor-every 4
  if [ $seed = 0 ]; then
    score 0 $BASE        # seed-0 data-free run already exists (queue15)
  else
    run free $seed
    score 1
  fi
done
echo QUEUE21_DONE $(date)
