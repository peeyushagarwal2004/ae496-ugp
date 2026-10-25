#!/bin/bash
# Regenerate the ground truth and every result that depends on it.
#
# WHY. cylinder_re100_v4 -- the dataset every PINN and FNO number in the
# report is trained on and scored against -- put the inlet 8 diameters ahead
# of the cylinder, had no acoustic treatment, and recorded 20 periods because
# the transverse duct mode takes over at about 30. Measured since: the inlet
# distance alone was worth 4.7 % of Cd and took St from +3.0 % to +0.3 %
# against Qu et al. (2013); the duct mode entrains the wake itself on long
# runs (vorticity St 0.1600 -> 0.1132); and an absorbing layer that relaxes
# toward the free stream removes it without moving Cd or St by more than
# 0.1-0.2 % across a 2x width and 5x strength change. The v4 ground truth is
# therefore about 3.5 % slow in St, and the report's validation claim does not
# hold at the level it asserts.
#
# WHAT CHANGES. Datasets: inlet at 25 D (domain 60 D long), 8 D absorbing
# layer at each free-slip wall relaxing toward the free stream, u = 0.05 at
# every Re (one Mach number for the whole sweep), 20 recorded periods. Same
# D = 24, same crop and snapshot rate, so every downstream script runs
# unchanged against the new tag. Marched PINN runs: --ic-from-network, so a
# run touches ground truth exactly once, at the first window. The earlier
# runs gave every window a fresh LBM snapshot, so a 4-window run consumed four
# times the reference information of the 1-window runs it was compared with.
#
# ORDER. Most important first, so a stall late in the queue costs the least:
# the Re = 100 dataset, then the ablation that carries the report's headline,
# then its controls, then the accuracy runs, then the sweep, then the FNO.
# Every stage skips work whose output already exists, so this resumes.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python
TAG=re100_v5
SWEEP="--suffix v5 --sponge 8 --upstream 25 --length 60 --record-periods 20 \
       --ic-from-network --run-root re_sweep_v5"

while pgrep -u "$USER" -f "lbm_cylinder\.py|run_ablation\.py|run_fno\.py" > /dev/null; do sleep 30; done

stage () { echo; echo "######## $1  $(date)"; }
abl () {   # run_ablation with the v5 tag; "$@" is passed through
  $PY -u experiments/run_ablation.py --tag $TAG --epochs 20000 --seeds 2 "$@" \
      2>&1 | grep -v nvptx_compiler
  echo "######## exit ${PIPESTATUS[0]}"
}

# ---- 1. the Re = 100 ground truth and its base flow -----------------------
stage "1 data+base Re=100"
$PY -u experiments/run_re_sweep.py --res 100 --stage base $SWEEP 2>&1 | grep -v nvptx_compiler
[ -f data/cylinder_${TAG}_meta.json ] || { echo "FATAL: no $TAG dataset"; exit 1; }

# ---- 2. the ablation, chained: this carries the headline ------------------
stage "2 ablation (chained ICs)"
abl --run-name v5/ablation --extra="--ic-from-network"

# ---- 3. the same marched configs with per-window dataset ICs, for the
#         paired comparison that shows what the old method inflated --------
stage "3 dataset-IC contrast"
abl --only march,all --run-name v5/ablation_dsic
abl --only all --run-name v5/ablation_dsic_sigma8 --extra="--sigma 8"

# ---- 4. equal-budget control: one window, 4x the epochs, same single IC --
stage "4 equal-budget control"
$PY -u experiments/run_ablation.py --tag $TAG --epochs 80000 --seeds 2 \
    --only baseline,decompose --run-name v5/ctrl_budget80k \
    --extra="--ic-from-network" 2>&1 | grep -v nvptx_compiler

# ---- 5. knobs. sigma 8 keeps its weights: its seed 0 is the accuracy run --
stage "5 knobs"
abl --only all,fourier --run-name v5/ablation_sigma8 \
    --extra="--ic-from-network --sigma 8 --save-params"
abl --only all --run-name v5/ablation_wic100 --extra="--ic-from-network --w-ic 100"

# ---- 6. how much supervision ----------------------------------------------
stage "6 data sweep"
for n in 100 500 2000; do
  abl --only all+data --run-name v5/data_n$n --extra="--ic-from-network --n-data $n"
done

# ---- 7. accuracy over one period and five ---------------------------------
stage "7 accuracy"
$PY -u experiments/shedding_accuracy.py --run-name v5/ablation_sigma8 \
    --runs all_s0 --tag $TAG 2>&1 | grep -v nvptx_compiler
WIN=$($PY -c "import json;s=json.load(open('runs/re_sweep_v5/summary.json'))['100'];print(round(0.25/s['St_vorticity'],4))")
echo "multiperiod window length (quarter of the vorticity period): $WIN"
$PY -u src/pinn/cylinder_unsteady.py --tag $TAG --epochs 20000 \
    --decompose --net fourier_mlp --sigma 8 --windows 20 --window-len $WIN \
    --ic-from-network --save-params --log-every 2000 \
    --out runs/v5/multiperiod/all_s0 2>&1 | grep -v nvptx_compiler
$PY -u experiments/shedding_accuracy.py --run-name v5/multiperiod \
    --runs all_s0 --tag $TAG 2>&1 | grep -v nvptx_compiler

# ---- 8. the Reynolds sweep, one Re per invocation --------------------------
stage "8 Reynolds sweep"
for re in 100 60 80 120 150 180; do
  echo "--- Re $re  $(date)"
  $PY -u experiments/run_re_sweep.py --res $re $SWEEP 2>&1 | grep -v nvptx_compiler
done

# ---- 9. the operator forecaster --------------------------------------------
stage "9 FNO"
$PY -u experiments/run_fno.py --tag $TAG --epochs 200 --seeds 2 \
    --run-name v5/fno 2>&1 | grep -v nvptx_compiler

stage "QUEUE15_DONE"
