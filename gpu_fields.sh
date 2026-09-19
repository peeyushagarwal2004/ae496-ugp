#!/bin/bash
# Save network parameters for the best data-free config, for the field figure.
while pgrep -f gpu_control.sh >/dev/null; do sleep 60; done
P=/home/phyguest/rajeshr/peeyush
cd $P/AE496_UGP
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib:$P/underPINN_GPU_Ready/underPINN_GPU:$P/AE496_UGP/src
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
echo "### fields start $(date)"
$PY -u experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 1 \
    --run-name fields_best --only all --extra "--sigma 8 --save-params" 2>&1 | grep -v nvptx_compiler
echo "### fields exit ${PIPESTATUS[0]} ($(date))"
du -sh runs/fields_best; df -h /home | tail -1
echo FIELDS_DONE $(date)
