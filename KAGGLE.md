# Running on Kaggle GPU

Phase 3 is the part that needs a GPU. Everything is JAX, so the code runs
unchanged — but read the two gotchas at the bottom before starting a long run.

**Don't upload the datasets.** The LBM solver is JAX too, so regenerating the
ground truth on the GPU takes ~5 minutes instead of the ~100 it takes on your
laptop, and avoids pushing 200 MB around. Generate it in the notebook.

## 0. Get the code there

Easiest is a GitHub repo (the project isn't one yet):

```bash
cd "C:/Users/hp/Desktop/AE496 UGP"
git init && git add src tests experiments README.md KAGGLE.md && git commit -m "UGP: PINN vortex shedding"
gh repo create ae496-ugp --private --source=. --push
```

Add `data/`, `runs/`, `figures/`, `upinn/`, `*.npz` to `.gitignore` first — the
datasets are regenerated on Kaggle and `upinn/` is installed from its own repo.

Then in a Kaggle notebook with **Accelerator = GPU T4 x2** (or P100):

```python
!git clone https://github.com/<you>/ae496-ugp.git /kaggle/working/ugp
!git clone https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN.git /kaggle/working/ugp/upinn
%cd /kaggle/working/ugp
!pip install -q flax optax
import jax; print(jax.default_backend(), jax.devices())
```

JAX with CUDA is preinstalled on Kaggle. If `default_backend()` prints `cpu`,
stop and fix that first — everything below assumes GPU.

## 1. Ground truth (~5 min on GPU)

```bash
python src/cfd/lbm_cylinder.py --u 0.05 --height 50 --steps 102000 \
    --warmup-periods 15 --snaps-per-period 32 --tag re100_v4
python src/cfd/inspect_data.py --tag re100_v4
```

Check the printed St and C_d. Expect **St ≈ 0.168** and **C_d ≈ 1.42** — both
sit above the 1.375 benchmark because of the 2 % blockage, which the
convergence study already quantified (see README). If they come out wildly
different, something is wrong with the GPU build, not the physics.

## 2. Base flow for the decomposition (~2 min on GPU)

```bash
python src/pinn/base_flow.py --tag re100_v4 --net fourier_mlp --sigma 8 --epochs 25000
```

Target **relative L2 below ~1 %**. Fourier features beat a plain MLP by 7.7×
here, so don't substitute `--net mlp`. If it lands above 2 %, raise `--sigma`
or `--epochs` — a loose base flow leaks a steady residual into u′ and defeats
the whole point of decomposing.

## 3. Calibrate before committing (~5 min)

```bash
python experiments/run_ablation.py --tag re100_v4 --quick
```

This runs all six configs at 2000 epochs. It is **not** a result — it tells you
the per-epoch cost so you can size the real run, and proves every code path
works on GPU.

## 4. The actual experiment

```bash
python experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 2
```

Six configurations × 2 seeds. Expect roughly **10–30 min per run**, so **2–6
hours total** — I'm estimating from a CPU benchmark, not measuring, so use
step 3 to get the real number. Results accumulate in
`runs/ablation/summary.json` after every run, so an interrupted session is not
wasted, and re-running skips what's already done.

## What the output means

The table's key column is **shedding retained** — wake-probe transverse-velocity
r.m.s., PINN vs LBM. Relative L2 alone would mislead you: a PINN that has
collapsed to steady flow still scores respectably, because the mean flow is
most of the field.

- `baseline` near **0 %** is the expected, published result, and it is the
  finding the rest of the report is built on. Do not treat it as a bug.
- If one of `fourier` / `decompose` / `march` lifts it substantially, that
  single mechanism is your contribution.
- If only `all` works, the mechanisms are complementary — also a real result,
  and the one the ablation exists to distinguish.
- If `all+data` works but no data-free config does, that reproduces the
  literature's conclusion that PINNs need data for this problem, which is a
  legitimate (if less exciting) finding.

## Two gotchas

**Session lifetime.** An interactive Kaggle notebook dies when you disconnect.
For anything past step 3, use **Save Version → Save & Run All (Commit)** so it
runs headless. GPU quota is 30 h/week and the per-session cap is ~9 h.

**Write to `/kaggle/working`.** Nothing outside it survives the session. If you
cloned to `/kaggle/working/ugp` as above you're fine; download `runs/` and
`figures/` at the end, or commit them back.

**Precision.** Everything defaults to float32 now, deliberately — a T4 runs
float64 at 1/32 the rate and it would dominate your quota. `--f64` exists for
precision checks only; don't use it for production runs.
