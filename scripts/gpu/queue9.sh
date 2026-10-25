#!/bin/bash
# Re-queue what queue7 and queue8 got wrong, behind whatever is still running.
#
# queue8 lost its first job to an argparse quirk: --extra "--ic-from-network"
# is a single token beginning with a dash, which argparse refuses as a value.
# The sigma-8 job survived only because its value contained a space, which
# argparse accepts. Use --extra=... instead, which is unambiguous.
#
# queue7 measured the wrong thing. It ran 30 warmup periods and recorded 38,
# where the production dataset used 15 and 20. Over that longer horizon the
# free-slip domain accumulates acoustic energy with nothing to damp it: at
# H = 50 the density spread grew from 0.5 % to several percent and the lift
# frequency fell to 0.1132, essentially the resonance at 0.1155; at H = 80 it
# reached rho[0.887, 1.080] with Cl_rms = 2.91 and Cd = 1.98, which is not
# converged physics at all. Record length was therefore confounded with the
# height the series was meant to isolate. Everything below matches the
# production budget exactly -- 15 warmup periods, about 20 recorded -- so H is
# the only thing that varies.
#
# That accumulation is itself worth keeping: queue7_overlong.log is direct
# evidence that this domain has no acoustic sink, which is the mechanism
# Section sec:acoustic argues for. It is a finding, not just a failed run.

set -u
D=/data/rajeshr/peeyush/AE496_UGP
P=/home/phyguest/rajeshr/peeyush
cd $D
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$P/pylib
PY=/home/phyguest/anaconda3/envs/graphcast-env/bin/python

while pgrep -u "$USER" -f "run_ablation\.py" > /dev/null; do sleep 30; done
echo "### prior queue clear, starting $(date)"

# ---- the decisive PINN experiment, the half queue8 dropped ----------------
echo "### icnet march,all start $(date)"
$PY -u experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 2 \
    --only march,all --run-name ablation_icnet \
    --extra=--ic-from-network 2>&1 | grep -v nvptx_compiler
echo "### icnet march,all exit ${PIPESTATUS[0]} ($(date))"

# ---- confinement series, production budget, H the only variable ----------
# 15 warmup periods + ~20 recorded, as cylinder_re100_v4: period at Re = 100,
# u = 0.05 is ~2921 steps, so 102000 total.
conf () {
  local h="$1"
  echo "### conf_H$h start $(date)"
  $PY -u src/cfd/lbm_cylinder.py --re 100 --diameter 24 --u 0.05 \
      --height "$h" --length 30 --warmup-periods 15 --steps 102000 \
      --no-snapshots --tag "confm_H$h" --outdir data 2>&1 | grep -v nvptx_compiler
  echo "### conf_H$h exit ${PIPESTATUS[0]} ($(date))"
}
conf 50
conf 80
conf 120

# ---- the two overwritten low-Re datasets, at the original u = 0.05 -------
# Same 15 + 20 period convention; the step counts differ because the shedding
# period in lattice steps does.
low () {
  local re="$1" steps="$2"
  echo "### re${re}_u05 start $(date)"
  $PY -u src/cfd/lbm_cylinder.py --re "$re" --diameter 24 --u 0.05 \
      --height 50 --length 30 --warmup-periods 15 --steps "$steps" \
      --no-snapshots --tag "re${re}_u05" --outdir data 2>&1 | grep -v nvptx_compiler
  echo "### re${re}_u05 exit ${PIPESTATUS[0]} ($(date))"
}
low 60 124000   # period ~3536 steps
low 80 110000   # period ~3141 steps

echo QUEUE9_DONE $(date)
