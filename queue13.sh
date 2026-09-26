#!/bin/bash
# The last axis: resolution at the cylinder surface.
#
# With the duct mode absorbed the flow is finally stationary -- 38 and 120
# recorded periods agree to four figures -- and the answer no longer depends
# on the absorbing layer: halving its width and varying its strength fivefold
# move Cd by 0.10 % and St by 0.18 %. So what is left is a real, cleanly
# measured discrepancy rather than an artefact:
#
#   Cd     1.4334  against Qu et al. 1.317    +8.8 %
#   St     0.1694  against         0.1649     +2.7 %
#   Cl'    0.2693  against         0.2224    +21 %
#
# Everything high, and high together, which is what an under-resolved
# cylinder surface looks like: separation predicted too early makes the wake
# wider, the drag larger and the lift fluctuation stronger.
#
# This project uses a uniform lattice with half-way bounce-back at D = 24, so
# the first fluid node sits 1/48 D from the surface. Qu et al. use a polar
# grid whose first spacing is 0.0054 D, roughly four times finer, and Qu
# reports first-node spacing as the parameter with the strongest influence on
# the Strouhal number in particular.
#
# The existing resolution series in data/convergence.json cannot settle this:
# it was run at 5 % blockage, where the report itself notes blockage masks
# resolution, and without any acoustic treatment. This one holds H = 50, the
# sponge and the averaging fixed and varies only D. 15 warmup + 60 recorded
# periods, well past the 38 at which the answer stopped moving.
#
# If Cd falls toward 1.32 as D rises, the discrepancy is discretisation at
# the wall and the validation question closes. If it does not, something
# structural is wrong with the solver and the report's ground truth needs a
# much harder look than a parameter sweep.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

while pgrep -u "$USER" -f "lbm_cylinder\.py|run_ablation\.py" > /dev/null; do sleep 20; done
echo "### starting $(date)"

# steps = (15 + 60) * period, period = D / (u * St) with St ~ 0.1643
res () {
  local d="$1" steps="$2"
  echo "### res_D$d start $(date)"
  $PY -u src/cfd/lbm_cylinder.py --re 100 --diameter "$d" --u 0.05 \
      --length 30 --height 50 --sponge 8 --sponge-target 0.05 \
      --warmup-periods 15 --steps "$steps" --no-snapshots \
      --tag "res_D$d" --outdir data 2>&1 | grep -v nvptx_compiler
  echo "### res_D$d exit ${PIPESTATUS[0]} ($(date))"
}

res 24 220000    # period ~2921, the production resolution
res 32 293000    # period ~3895
res 48 439000    # period ~5843

echo "### measuring $(date)"
$PY -u experiments/measure_st.py res_D24 res_D32 res_D48 tsponge_conv \
    --json runs/st_resolution.json 2>&1 | grep -v nvptx_compiler

echo QUEUE13_DONE $(date)
