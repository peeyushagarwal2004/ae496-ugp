# AE496 UGP — Vortex shedding behind a cylinder with a PINN

Predicting Kármán vortex shedding at Re = 100 with a physics-informed neural
network, built on [underPINN](https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN)
(JAX/Flax). The reference paper is Renn, Wang, Lale, Li, Anandkumar & Gharib,
*Forecasting subcritical cylinder wakes with Fourier Neural Operators*
([arXiv:2301.08290](https://arxiv.org/abs/2301.08290)).

## The problem this project actually attacks

The reference paper is **not** a PINN paper — it is a purely data-driven FNO
trained on experimental PIV, with no physics loss. And a data-free unsteady
PINN is documented to **fail** at this exact problem: it collapses onto the
steady, symmetric base flow instead of shedding
([arXiv:2306.00230](https://arxiv.org/abs/2306.00230)). The contribution here is
to reproduce that failure honestly, then attack it with three mechanisms —
Reynolds decomposition (borrowed from the reference paper), Fourier features,
and causal time-marching — and measure which one actually matters.

Note the decomposition does **not** by itself exclude the trivial answer:
u' = 0 solves the fluctuation equations too. It buys conditioning (the network
represents an O(0.13) fluctuation instead of an O(1) field). Excluding the
steady solution has to come from the initial condition plus time-marching.

Target is **Re = 100**, not the paper's Re = 240–3060: the subcritical regime is
genuinely three-dimensional (the paper says so, and blames some of its own
error on out-of-plane motion), so a 2-D PINN there would be solving the wrong
physics. Below Re ≈ 200, 2-D and 3-D results agree.

## Layout

```
src/cfd/     lattice-Boltzmann ground-truth generator + validation
src/pinn/    unsteady NS operator, data layer, base flow, the PINN experiment
tests/       correctness checks (run them; they are fast)
data/        generated datasets (not in git — regenerate with the commands below)
figures/     validation and result figures
upinn/       vendored underPINN source
```

## Status

| phase | state |
|---|---|
| 1. Ground-truth data | **done** — solver validated, see below |
| 2. Unsteady NS operator | **done** — validated to machine precision |
| 3. PINN experiment | pipeline built and smoke-tested; needs GPU to run for real |
| 4. Defeating the collapse | mechanisms implemented as switches; not yet run |
| 5. FNO comparison | not started |

### Phase 1 result: the solver is validated

`St` and `C_d` both came out high against the Re = 100 benchmark. The
convergence sweep shows why, and it is not a coding fault:

* **Resolution is not the cause.** `C_d` varies 2.4 % across D = 16 → 40
  lattice units — grid converged at the coarsest resolution tried.
* **Blockage is.** `C_d` = 1.627 / 1.459 / 1.431 at 5.0 / 3.3 / 2.5 % blockage,
  extrapolating to **1.348 at zero blockage, −2.0 % from the published 1.375**.

See `figures/convergence.png`.

### Datasets

| tag | Ma | blockage | div error | notes |
|---|---|---|---|---|
| `re100`    | 0.139 | 5.0 % | 2.12 % | first run, BGK |
| `re100_v3` | 0.087 | 3.3 % | 0.55 % | TRT; usable |
| `re100_v4` | 0.087 | 2.0 % | ~0.5 % | **use this**; 20 recorded periods |

The divergence error tracks O(Ma²) exactly, which is why the Mach number was
lowered: at 2 % the data-fit and physics-residual terms fight each other during
PINN training.

## Reproducing

```bash
# ground truth (~100 min on CPU)
python src/cfd/lbm_cylinder.py --u 0.05 --height 50 --steps 102000 \
    --warmup-periods 15 --snaps-per-period 32 --tag re100_v4
python src/cfd/inspect_data.py --tag re100_v4

# solver validation (~2 h)
python src/cfd/convergence_sweep.py && python src/cfd/plot_convergence.py

# smooth mean flow for the Reynolds decomposition
python src/pinn/base_flow.py --tag re100_v4 --net fourier_mlp --sigma 8 --epochs 25000

# the experiment
python src/pinn/cylinder_unsteady.py --smoke                    # CPU sanity check
python src/pinn/cylinder_unsteady.py --tag re100_v4 --epochs 20000            # 3a: expect collapse
python src/pinn/cylinder_unsteady.py --tag re100_v4 --decompose \
    --net fourier_mlp --windows 4 --epochs 20000                # 4: the fix
```

```bash
python tests/test_ns2d_unsteady.py && python tests/test_lbm_freeslip.py
```

## The headline metric

Relative L2 against the LBM field is not sufficient on its own: a PINN that
relaxes to steady flow still scores respectably, because the mean flow *is*
most of the field. `cylinder_unsteady.py` therefore also reports **shedding
retained** — the ratio of wake-probe transverse-velocity r.m.s. between PINN
and LBM. A value near 0 % is the documented collapse.

## Compute

CPU-only development works for phases 1–2. Phase 3 does not: a realistic
20 000-epoch run is ~8 hours on CPU, and the ablation needs a dozen. Everything
is JAX and runs unchanged on GPU (Kaggle gives 30 GPU-h/week).

## Things that turned out to be wrong

Kept deliberately, because they shaped the design:

* TRT collision was expected to fix the `C_d` bias — it moved it ~1 %.
* Under-resolution was expected to dominate the error — blockage does, by ~5×.
* `St` looked accurate to 0.3 % with an FFT peak estimator; that was the bin
  width flattering the result. Zero-crossing timing gives +2 %, consistently.
