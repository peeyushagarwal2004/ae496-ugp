#!/bin/bash
# Does causal time-marching restore shedding on ONE ground-truth slice?
#
# THE QUESTION
#
# report.tex:386 describes the chain as "solving a short window, then using its
# end state as the initial condition for the next". src/pinn/cylinder_unsteady.py
# did not do that: train_window() took every window's initial condition from
# ds.snapshot(nearest_time_index(t0)) -- a fresh LBM slice at each window start.
# A four-window run therefore consumed 4 x --n-ic = 16000 ground-truth points at
# four times spanning the interval, against 4000 at a single instant for the
# one-window configurations it was compared with. "Time-marching alone restores
# shedding, data-free" was measured against configurations given a quarter of
# the reference information, spread differently in time.
#
# --ic-from-network implements the chain as described: window 0 takes the one
# initial condition a forward solve is entitled to, and every later window
# takes its target from the previous window's network. One ground-truth slice
# for the whole run.
#
# READING THE RESULT, decided before looking:
#   ~89 % retained  -> causality does the work; the extra slices were
#                      incidental, the claim survives and gets stronger,
#                      because it now holds on a quarter of the reference data.
#   ~10 % retained  -> the extra ground truth was doing the work, and the
#                      headline of the report is wrong as written.
#   in between      -> both contribute; the report must quantify the split
#                      rather than claim either mechanism alone.
#
# The comparison is against the committed runs/ablation, which used identical
# settings with dataset ICs. Nothing else changes: same epochs, same windows,
# same seeds, same window length.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

# Wait for the LBM queue rather than contending for the same GPU.
while pgrep -u "$USER" -f "lbm_cylinder.py" > /dev/null; do sleep 30; done
echo "### lbm queue clear, starting at $(date)"

echo "### icnet march,all start $(date)"
$PY -u experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 2 \
    --only march,all --run-name ablation_icnet \
    --extra "--ic-from-network" 2>&1 | grep -v nvptx_compiler
echo "### icnet march,all exit ${PIPESTATUS[0]} ($(date))"

# sigma = 8 is the configuration the Reynolds sweep used and the one the
# report quotes as the best data-free result (97.6 % retained), so the
# question matters most here.
echo "### icnet all sigma8 start $(date)"
$PY -u experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 2 \
    --only all --run-name ablation_icnet_sigma8 \
    --extra "--ic-from-network --sigma 8" 2>&1 | grep -v nvptx_compiler
echo "### icnet all sigma8 exit ${PIPESTATUS[0]} ($(date))"

echo QUEUE8_DONE $(date)
