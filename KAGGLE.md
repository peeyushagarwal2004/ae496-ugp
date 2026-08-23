# Running on Kaggle GPU

Phase 3 is the part that needs a GPU. Everything is JAX, so the code runs
unchanged — but read the two gotchas at the bottom before starting a long run.

**Don't upload the datasets.** The LBM solver is JAX too, so regenerating the
ground truth on the GPU takes ~5 minutes instead of the ~100 it takes on your
laptop, and avoids pushing 200 MB around. Generate it in the notebook.

## 0. Get the code there

The local git repo is already created and committed (code only, 142 KB —
`.gitignore` keeps the datasets out). You only need to push it somewhere Kaggle
can reach.

1. Go to <https://github.com/new>, name it `ae496-ugp`, set it **Private**, and
   do **not** tick "Add a README" — the repo already has one.
2. Push (replace `<you>` with your GitHub username):

```bash
git remote add origin https://github.com/<you>/ae496-ugp.git
git branch -M main
git push -u origin main
```

When git asks for a password, give it a **personal access token**, not your
account password (GitHub → Settings → Developer settings → Personal access
tokens → generate one with `repo` scope). Plain passwords stopped working for
git years ago.

*No GitHub account?* Alternative: zip `src/`, `tests/` and `experiments/`,
upload via Kaggle's **+ Add Data → Upload**, and replace the clone line below
with a copy out of `/kaggle/input/...`. It works, but you re-upload on every
code change, which gets old fast.

Then in a Kaggle notebook with **Accelerator = GPU T4 x2** (or P100), and
**Internet switched on** in the right sidebar (needs phone verification on your
Kaggle account -- both clones fail instantly without it).

A private repo needs credentials. Add your token under **Add-ons -> Secrets ->
Add a new secret**, labelled `GH_TOKEN`, and check that it is attached to the
notebook. Then, in the first cell:

```python
import subprocess
from kaggle_secrets import UserSecretsClient

GH_USER, REPO = "<you>", "ae496-ugp"
tok = UserSecretsClient().get_secret("GH_TOKEN")

subprocess.run(["rm", "-rf", "/kaggle/working/ugp"], check=False)
subprocess.run(["git", "clone", "-q",
                f"https://{tok}@github.com/{GH_USER}/{REPO}.git",
                "/kaggle/working/ugp"], check=True)
subprocess.run(["git", "clone", "-q",
                "https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN.git",
                "/kaggle/working/ugp/upinn"], check=True)
print("cloned OK")
```

```python
!pip install -q flax optax
```

Use `subprocess.run`, not `!git clone`: the `!` magic echoes the command --
token included -- into the notebook output, and Kaggle saves outputs.

Then in a **separate** cell (a pip install does not affect modules already
imported in the same cell):

```python
%cd /kaggle/working/ugp
import jax, flax, optax
print("jax", jax.__version__, "| flax", flax.__version__, "| optax", optax.__version__)
print("backend:", jax.default_backend(), jax.devices())
!ls upinn/underPINN | head
```

Two things must be true before going further:

* **`backend: gpu`.** If it prints `cpu`, the pip install resolved a CPU JAX
  wheel over Kaggle's preinstalled CUDA one -- a common trap. Fix with
  `!pip install -q -U "jax[cuda12]"` and re-run the cell.
* **`ls upinn/underPINN` lists `core`, `nn`, `pde`, ...** If it is empty, the
  second clone landed in the wrong place and every `import underPINN` fails.

## 1. Ground truth (~5 min on GPU)

```bash
python src/cfd/lbm_cylinder.py --u 0.05 --height 50 --steps 102000 \
    --warmup-periods 15 --snaps-per-period 32 --tag re100_v4
python src/cfd/inspect_data.py --tag re100_v4
```

Check the printed numbers against what this exact command produced locally:

```
Cd = 1.3798   (benchmark 1.37-1.38,  +0.3 %)
St = 0.1611   (benchmark 0.165,      -2.3 %)
```

LBM is deterministic, so the GPU should reproduce these to within float32
round-off. If it does not, something is wrong with the GPU build — sort that
out before spending quota on training.

## 2. Base flow for the decomposition (~2 min on GPU)

```bash
python src/pinn/base_flow.py --tag re100_v4 --net fourier_mlp --sigma 12 --epochs 25000
```

Target **relative L2 below ~1 %**. The same fit reached **1.2e-3** locally, so
anything much worse means something is off. Fourier features beat a plain MLP
by 7.7x here, and sigma = 12 won a scan over 8/12/16/24 — don't substitute
`--net mlp`. A loose base flow leaks a steady residual into u' and defeats the
whole point of decomposing.

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
