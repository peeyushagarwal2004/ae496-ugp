#!/bin/bash
# Full-budget GPU sweep on kanaada (GPU 1). Writes only under this directory.
P=/home/phyguest/rajeshr/peeyush
cd $P/AE496_UGP
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib:$P/underPINN_GPU_Ready/underPINN_GPU:$P/AE496_UGP/src
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
A="experiments/run_ablation.py --tag re100_v4 --epochs 20000"
stage() { echo "### stage $1 start $(date)"; shift; $PY -u $A "$@" 2>&1 | grep -v nvptx_compiler; echo "### stage exit ${PIPESTATUS[0]} ($(date))"; df -h /home | tail -1; }
# A: the planned Phase 3/4 ablation, full budget, 2 seeds
stage A --seeds 2 --run-name ablation
# B: Fourier sigma knob (base-flow fit wanted 8-12; default 2)
stage B --seeds 2 --run-name ablation_sigma8 --only fourier,all --extra "--sigma 8"
# C: stronger initial condition
stage C --seeds 2 --run-name ablation_wic100 --only all --extra "--w-ic 100"
# D: how much supervision restores shedding
for n in 100 500 2000; do stage D$n --seeds 1 --run-name data_n$n --only all+data --extra "--n-data $n"; done
echo SWEEP_DONE $(date)
