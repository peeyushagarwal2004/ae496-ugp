#!/bin/bash
# Runs once the FNO sweep finishes: finish Re = 100, then the Re sweep.
while pgrep -f fno_sweep.sh >/dev/null; do sleep 120; done
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

# 1. Re = 100 over ~5 shedding periods, so St rests on ~10 zero crossings
#    instead of 2. Same winning config: all three mechanisms, sigma 8.
echo "### multiperiod start $(date)"
$PY -u src/pinn/cylinder_unsteady.py --tag re100_v4 --epochs 20000 \
    --decompose --net fourier_mlp --sigma 8 --windows 20 --window-len 1.5514 \
    --save-params --log-every 2000 --out runs/multiperiod/all_s0 2>&1 | grep -v nvptx_compiler
echo "### multiperiod exit ${PIPESTATUS[0]} ($(date))"

# 2. The rest of the shedding range.
echo "### re_sweep start $(date)"
$PY -u experiments/run_re_sweep.py --res 60,80,120,150,180 2>&1 | grep -v nvptx_compiler
echo "### re_sweep exit ${PIPESTATUS[0]} ($(date))"
echo AFTER_FNO_DONE $(date)
