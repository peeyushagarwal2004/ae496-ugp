#!/bin/bash
# Where does sparse re-anchoring start to hold the phase?
#
# queue21 (runs/v5/reanchor) on the five-period chain: 200 supervised points
# per shedding period leaves it 3-5 % slow and 0.10-0.12 period behind at the
# end; 400 (100 every window) and 500 (500 once per period) both hold it to
# within 0.003 period, on two seeds. Two questions for the gap between:
#
#   1. at fixed anchoring frequency (every half period), where between 200
#      and 400 points per period does the phase lock?
#   2. at fixed density (300 per period), does it matter whether the points
#      come every window, every half period or once per period?
#
#   name      N    K   per period  total   question
#   a2_n125  125   2     250       1250    1
#   a2_n150  150   2     300       1500    1, 2
#   a2_n175  175   2     350       1750    1
#   a1_n75    75   1     300       1500    2
#   a4_n300  300   4     300       1500    2
#
# Same run as queue21 otherwise (all, sigma 8, 20 quarter-period windows,
# --ic-from-network). Seed 0 first (~7 h) and scored, then seed 1 (~7 h).
# ~80 min per run on GPU 1. Every run skips if its results.json exists.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5
R=runs/v5/reanchor_threshold
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

score () {   # seed -- scored against the reference probe
  local seed=$1
  # absolute paths: shedding_accuracy resolves relative ones under runs/<run-name>/
  local dirs=$(ls -d $D/$R/*_s$seed 2>/dev/null | paste -sd, -)
  $PY -u experiments/shedding_accuracy.py --runs "$dirs" --tag $TAG \
      --out $R/shedding_accuracy_s$seed.json 2>&1 | grep -v nvptx_compiler
}

mkdir -p $R
for seed in 0 1; do
  run a2_n150 $seed --n-data 150 --anchor-every 2
  run a1_n75  $seed --n-data 75  --anchor-every 1
  run a4_n300 $seed --n-data 300 --anchor-every 4
  run a2_n125 $seed --n-data 125 --anchor-every 2
  run a2_n175 $seed --n-data 175 --anchor-every 2
  score $seed
done
echo QUEUE22_DONE $(date)
