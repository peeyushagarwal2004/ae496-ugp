#!/bin/bash
# Phase 5 on /data (the full 641-snapshot record; /home has no room for it).
# Waits for the PINN seed sweep so the two do not share the card.
while pgrep -f gpu_dataseed.sh >/dev/null; do sleep 60; done
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
echo "### fno start $(date)"
$PY -u experiments/run_fno.py --tag re100_v4 --epochs 200 --seeds 1 2>&1 | grep -v nvptx_compiler
echo "### fno exit ${PIPESTATUS[0]} ($(date))"
df -h /data | tail -1
echo FNO_DONE $(date)
