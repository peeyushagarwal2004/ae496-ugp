#!/bin/bash
# Second seed for the supervision sweep (seed 0 rows are skipped automatically).
P=/home/phyguest/rajeshr/peeyush
cd $P/AE496_UGP
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib:$P/underPINN_GPU_Ready/underPINN_GPU:$P/AE496_UGP/src
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
for n in 100 500 2000; do
  echo "### seed1 n$n start $(date)"
  $PY -u experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 2 \
      --run-name data_n$n --only all+data --extra "--n-data $n" 2>&1 | grep -v nvptx_compiler
  echo "### seed1 n$n exit ${PIPESTATUS[0]} ($(date))"
done
df -h /home | tail -1
echo DATASEED_DONE $(date)
