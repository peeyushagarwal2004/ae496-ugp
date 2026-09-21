#!/bin/bash
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
echo "### sweep3 start $(date)"
$PY -u experiments/run_re_sweep.py --res 60,80,120,150,180 2>&1 | grep -v nvptx_compiler
echo "### sweep3 exit ${PIPESTATUS[0]} ($(date))"
echo SWEEP3_DONE $(date)
