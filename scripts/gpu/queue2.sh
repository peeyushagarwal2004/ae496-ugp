#!/bin/bash
# Order requested: second FNO seed, then finish Re = 100, then the Re sweep.
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

# 1. Seed 1 for all three FNO variants (seed 0 rows are skipped).
echo "### fno_seed1 start $(date)"
$PY -u experiments/run_fno.py --tag re100_v4 --epochs 200 --seeds 2 2>&1 | grep -v nvptx_compiler
echo "### fno_seed1 exit ${PIPESTATUS[0]} ($(date))"

# 2. Re = 100 over ~5 shedding periods, for a quotable St.
echo "### multiperiod start $(date)"
$PY -u src/pinn/cylinder_unsteady.py --tag re100_v4 --epochs 20000 \
    --decompose --net fourier_mlp --sigma 8 --windows 20 --window-len 1.5514 \
    --save-params --log-every 2000 --out runs/multiperiod/all_s0 2>&1 | grep -v nvptx_compiler
echo "### multiperiod exit ${PIPESTATUS[0]} ($(date))"

# 3. The rest of the shedding range (resumes: Re 60 data and base flow exist).
echo "### re_sweep start $(date)"
$PY -u experiments/run_re_sweep.py --res 60,80,120,150,180 2>&1 | grep -v nvptx_compiler
echo "### re_sweep exit ${PIPESTATUS[0]} ($(date))"
echo QUEUE2_DONE $(date)
