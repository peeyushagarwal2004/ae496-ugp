#!/bin/bash
# Equal-budget control: single-window configs at the march total step count (4 x 20000).
while pgrep -f gpu_sweep.sh >/dev/null; do sleep 60; done
P=/home/phyguest/rajeshr/peeyush
cd $P/AE496_UGP
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib:$P/underPINN_GPU_Ready/underPINN_GPU:$P/AE496_UGP/src
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
echo "### stage E start $(date)"
$PY -u experiments/run_ablation.py --tag re100_v4 --epochs 80000 --seeds 2 --run-name ctrl_budget80k --only baseline,decompose 2>&1 | grep -v nvptx_compiler
echo "### stage exit ${PIPESTATUS[0]} ($(date))"
echo CONTROL_DONE $(date)
