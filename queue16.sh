#!/bin/bash
# Is "shedding retained" by the single-window baseline a robust number?
#
# The same baseline configuration -- one 6 t* window, exact LBM initial
# condition, no data -- retains 9.8 / 11.3 % on cylinder_re100_v4 and
# 40.4 / 41.7 % on cylinder_re100_v5. The reference amplitude is nearly the
# same (probe v rms 0.412 vs 0.403) and the seeds agree tightly within each
# dataset, so this is not noise. Something about the dataset changes what a
# plain PINN does over one period.
#
# The obvious candidate is the phase of the initial condition. Both runs start
# at their dataset's first snapshot, t0 = 91.10 and 91.46, which catches the
# wake at different points in its cycle. A PINN that decays from its initial
# condition retains an amount over one period that could depend strongly on
# where in the cycle it was released. If it does, the report's "a plain PINN
# retains X % of the wake" is a property of an arbitrary starting time, and
# every configuration compared at one starting time is compared at that
# phase only.
#
# Four phases, a quarter period apart, on both datasets, one seed.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

for tag in re100_v4 re100_v5; do
  read T0 PER < <($PY -c "
import json, numpy as np
t = np.load('data/cylinder_${tag}.npz')['t']
st = {'re100_v4': 0.15947208294485055,
      're100_v5': json.load(open('runs/re_sweep_v5/summary.json'))['100']['St_vorticity']}['${tag}']
print(float(t[0]), 1.0 / st)")
  for k in 0 1 2 3; do
    TS=$($PY -c "print(round($T0 + $k * $PER / 4.0, 4))")
    echo "### $tag phase $k/4  t_start=$TS  $(date)"
    $PY -u src/pinn/cylinder_unsteady.py --tag $tag --epochs 20000 --seed 0 \
        --t-start $TS --window-len 6.0 --log-every 5000 \
        --out runs/v5/phase/${tag}_k$k 2>&1 | grep --line-buffered -E "OVERALL|shedding retained|relative L2" | tail -3
  done
done
echo QUEUE16_DONE $(date)
