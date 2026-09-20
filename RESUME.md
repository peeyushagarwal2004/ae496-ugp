# AE496 UGP — Resume pointers, concepts and expected results

Prediction of vortex shedding behind a circular cylinder using a
physics-informed neural network, built on
[underPINN](https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN).
Reference paper: Renn, Wang, Lale, Li, Anandkumar & Gharib, *Forecasting
subcritical cylinder wakes with Fourier Neural Operators*,
[arXiv:2301.08290](https://arxiv.org/abs/2301.08290).

Written so the project can be picked up cold. Every phase is given as
**Objective / Approach / Result**.

> ### How to read the numbers
> **[MEASURED]** — actually computed in this project. Quote these.
> **[ESTIMATE]** — not yet run. Predicted, with the reasoning stated.
> **Never put an [ESTIMATE] in the report as a result.** They exist so you can
> tell whether a run went as expected or something is broken. My estimates in
> this project have been wrong at least three times (see §10).

---

## Part I — Concepts

Everything the report needs to define, in dependency order.

### 1.1 Neural network / MLP

A **multi-layer perceptron** is a function approximator: alternating affine maps
and a nonlinearity, `x -> W_n tanh(... tanh(W_1 x + b_1) ...) + b_n`. The
weights `W, b` are fitted by gradient descent. Here the network *is* the flow
field: it maps a point in space-time to the flow variables there,

```
(x, y, t)  ->  (u, v, p)
```

so it is a **mesh-free continuous representation** — evaluable at any point, not
just grid nodes. `layers = [3, 128, 128, 128, 128, 128, 128, 3]` means 3 inputs,
6 hidden layers of width 128, 3 outputs. `tanh` is used rather than ReLU because
the PDE residual needs **second** derivatives, and ReLU's are identically zero.

### 1.2 Automatic differentiation

JAX computes exact derivatives of the network with respect to its *inputs* by
the chain rule — not finite differences, so no truncation error and no grid.
This is what makes a PINN possible: `u_t`, `u_x`, `u_xx` are obtained
analytically from the network itself.

* **Forward mode (`jvp`, `jacfwd`)** — efficient when inputs are few (3 here).
* **Reverse mode (`grad`)** — efficient when outputs are few (the scalar loss).

Used in `src/pinn/ns2d_unsteady.py`: derivatives w.r.t. inputs in forward mode,
gradient of the loss w.r.t. weights in reverse mode.

### 1.3 Physics-informed neural network (PINN)

A PINN trains the network to satisfy a PDE rather than (only) to fit data. The
loss is a sum of squared residuals evaluated at randomly sampled **collocation
points**:

```
L = w_pde * |PDE residual|^2          at interior points
  + w_bc  * |boundary condition|^2    at domain boundaries and the cylinder
  + w_ic  * |u - u(t0)|^2             at the initial time
  + w_data* |u - u_measured|^2        optional supervised term
```

With `w_data = 0` the PINN is **data-free**: it is a PDE solver that happens to
use a network as its discretisation. That is the interesting case here.

### 1.4 The governing equations

2-D unsteady incompressible Navier-Stokes, non-dimensionalised on cylinder
diameter `D` and free-stream speed `U` (so `D = U = 1` and `Re = UD/nu`):

```
continuity   u_x + v_y                                        = 0
x-momentum   u_t + u u_x + v u_y + p_x - (1/Re)(u_xx + u_yy)  = 0
y-momentum   v_t + u v_x + v v_y + p_y - (1/Re)(v_xx + v_yy)  = 0
```

underPINN ships only a **steady** version (`NavierStokesPDE`, inputs `(x,y)`),
which cannot represent shedding at all. Supplying the unsteady operator was
deliverable 2 of this project.

### 1.5 Vortex shedding and why it is hard for a PINN

Above `Re ~ 47` the steady symmetric wake becomes linearly unstable and sheds a
**Kármán vortex street** — alternating vortices at a frequency set by the
**Strouhal number** `St = f D / U`.

The difficulty is structural, and it is the whole point of the project:

1. **The steady solution is still a valid solution.** A symmetric, non-shedding
   field satisfies the equations and all boundary conditions exactly. It is an
   *unstable* solution physically, but the PINN loss has no notion of stability —
   it is a strong, easily-reachable minimum.
2. **Spectral bias** (§1.6) — networks learn low frequencies first and struggle
   with the oscillation.
3. **The initial condition is a measure-zero slice** of a space-time domain, so
   it is weakly weighted against the bulk PDE residual.

Documented consequence
([arXiv:2306.00230](https://arxiv.org/abs/2306.00230)): a data-free PINN
"behaves like a steady solver", and a data-driven one sheds only while data is
present, reverting to steady once it stops.

### 1.6 Spectral bias and Fourier features

Networks fit low-frequency content long before high-frequency content — the
neural tangent kernel decays with frequency. A wake is oscillatory, so this
directly opposes what we need.

**Random Fourier features** fix the input encoding: instead of feeding `x`, feed

```
[sin(Bx), cos(Bx)],   B ~ N(0, sigma^2)
```

`sigma` sets the frequency content the network can express cheaply. underPINN's
`FourierMLP` implements this with a *trainable* `B`. **SIREN** is the
alternative (sine activations throughout).

### 1.7 Reynolds decomposition — the idea borrowed from the reference paper

Split the field into a time mean and a fluctuation:

```
u(x, y, t) = u0(x, y) + u'(x, y, t)
```

The reference paper subtracts the mean before training its FNO (its eq. 3.1–3.2)
because the resulting equations for `u'` are homogeneous and nearly linear, and
because the fluctuation is not swamped by the large free-stream bias.

**Important and easy to get wrong:** this does **not** exclude the trivial
answer. `u' = 0` still solves the fluctuation equations. What it buys is
*conditioning* — the network represents an O(0.13) fluctuation instead of an
O(1) field dominated by its mean. Excluding the steady solution has to come from
the initial condition plus time-marching. Do not write the stronger claim.

Implementation subtlety: the momentum residual contains `grad(u0)` and
`lap(u0)`, so `u0` must be differentiable. Bilinear interpolation of a grid gives
piecewise-constant first derivatives and *zero* second derivatives — useless. So
`u0` is fitted by a separate network (`src/pinn/base_flow.py`) and frozen.

Second subtlety: the time mean of a shedding flow does **not** satisfy the
steady Navier-Stokes equations — it carries the divergence of the Reynolds
stresses. So `u0` is obtained by **regression on the measured mean**, not by a
steady PINN solve. That keeps the decomposition exact.

### 1.8 Causal time-marching

Instead of solving the whole time interval at once, solve a short window
`[t0, t0+dt]`, then use its end state as the initial condition for the next.
This strengthens the initial condition relative to the PDE residual and enforces
causality (information flows forward in time), which a naive space-time PINN
does not respect.

### 1.9 Neural operators and the FNO

A standard network approximates a *function*. A **neural operator** approximates
a *mapping between function spaces* — e.g. "flow field now" -> "flow field one
step later". This makes it **discretisation invariant**: trained on one mesh,
evaluable on another.

A **Fourier Neural Operator** implements this with layers of the form

```
v -> sigma( W v + F^-1( R . F(v) ) )
```

`F` is an FFT; `R` multiplies the lowest `modes` Fourier coefficients by learned
weights; `W` is a local linear map. The FFT path is a **global** integral
operator (every point sees every other point in one layer), which is why FNOs
capture large-scale structures such as convecting vortices efficiently. `modes`
truncates high wavenumbers (regularisation); `width` is the channel count.

**Autoregressive rollout**: the model predicts one step, and is then re-applied
to its own output, `n` times. The reference paper trains with the loss summed
over all 10 recursive steps, which is what makes the rollout stable rather than
drifting immediately.

**Crucially, the reference paper's FNO is purely data-driven — there is no
physics loss anywhere in it.** It is not a PINN paper. Bridging that gap is this
project's contribution.

### 1.10 Lattice Boltzmann method (the ground-truth solver)

We had no experimental data, so ground truth is generated. LBM evolves particle
distribution functions `f_i` on a **D2Q9** lattice (9 discrete velocities in 2-D):

```
collide:  f_i^post = f_i - (f_i - f_i^eq)/tau       (relaxation to equilibrium)
stream:   f_i(x + e_i, t+1) = f_i^post(x, t)        (exact advection)
```

Macroscopic variables recover from moments: `rho = sum f_i`,
`rho u = sum e_i f_i`, and `p = c_s^2 rho`. It solves Navier-Stokes to second
order with `nu = c_s^2 (tau - 1/2)`.

Special methods used, and why:

* **TRT (two-relaxation-time) collision.** Plain BGK has one relaxation time, and
  the wall implied by bounce-back then drifts with viscosity unless
  `Lambda = (tau+ - 1/2)(tau- - 1/2) = 3/16`. TRT relaxes symmetric and
  antisymmetric parts separately so `Lambda` can be pinned, fixing the cylinder's
  effective diameter at any `Re`.
* **Half-way bounce-back** for the no-slip cylinder: populations that would enter
  the solid are reflected back, placing the wall midway between nodes.
* **Momentum-exchange algorithm (MEA)** for the surface force:
  `F = sum over boundary links of 2 e_i f_i^post`. This is what makes `C_d`
  accurate enough to validate against published data.
* **Specular (free-slip) top/bottom walls**, enforcing `v = 0` exactly, matching
  the `symmetry` boundary underPINN's cylinder example uses. Diagonal populations
  must also advance one cell in `x` while reflecting — omitting that leaves a
  spurious tangential drag (this was a real bug, now covered by a test).
* **Weak compressibility.** LBM is not exactly incompressible; the divergence
  error scales as `O(Ma^2)`. Lowering the lattice velocity therefore directly
  improves how well the data satisfies the PINN's own equations.

### 1.11 Validation quantities

* `C_d = F_x / (0.5 rho U^2 D)` — mean drag coefficient. Benchmark at
  Re = 100: **1.37–1.38**.
* `C_l` — lift coefficient; oscillates at the shedding frequency.
  `C_d` oscillates at **twice** it (a useful correctness check).
* `St = f D / U` — benchmark at Re = 100: **0.165**.
* **Blockage** `D/H` — confinement by the finite domain height. Raises both
  `C_d` and `St`, and must be extrapolated away before comparing to published
  unbounded-flow values.
* **Relative L2** (the paper's eq. 3.3), `eps = ||q* - q||_2 / ||q||_2` over the
  stacked `(u, v)` field.
* **Shedding retained** (this project's own metric, see §5).

---

## Part II — Phases

### Phase 1 — Ground-truth data

**Objective.** Produce a validated, time-resolved Re = 100 cylinder wake to
serve as initial/boundary conditions, supervised data, and the yardstick for
every later phase. Required because no experimental data was available.

**Approach.** D2Q9 TRT lattice-Boltzmann solver in JAX
(`src/cfd/lbm_cylinder.py`), Re = 100 (laminar, unambiguously 2-D — the paper's
Re = 240–3060 is genuinely three-dimensional, so a 2-D PINN there would be
solving the wrong physics). Shedding is triggered by a brief transverse inlet
perturbation; without it the symmetric base flow persists indefinitely. Output is
written in exactly underPINN's cylinder convention (`D = 1`, `U = 1`, origin at
the cylinder, `x` in [-5, 15], `y` in [-5, 5]) at `dx = 0.083 D` and
`dt* = 0.19`, chosen to sit near the reference paper's PIV resolution (< 0.1 D)
and time step (0.17) so the Phase 5 comparison is like-for-like.

Validation is a **convergence study** separating the two candidate error sources
(`src/cfd/convergence_sweep.py`): resolution `D = 16..40` at fixed blockage, and
blockage `H/D = 20/30/40` at fixed resolution.

**Result [MEASURED].** Production dataset `re100_v4`, 641 snapshots,
`t* = 91..212`:

| quantity | computed | benchmark | error |
|---|---|---|---|
| `C_d` | 1.3798 | 1.37–1.38 | **+0.3 %** |
| `St` | 0.1611 | 0.165 | −2.3 % |
| divergence error | 0.55 % of \|grad u\| | — | tracks O(Ma²), Ma = 0.087 |

Convergence study:

```
resolution D = 16..40 at 5 % blockage : Cd spread 2.4 %  -> grid converged
blockage 5.0 / 3.3 / 2.5 %            : Cd 1.627 / 1.459 / 1.431
extrapolated to zero blockage         : Cd 1.348   (-2.0 % vs benchmark)
```

So the apparent +18 % error at 5 % blockage is **domain confinement, not a
solver defect** — the headline validation result. Figure:
`figures/convergence.png`.

Base flow for the decomposition: `data/base_flow_re100_v4.pkl`,
relative L2 = **1.10e-3** (Fourier MLP 128×5, sigma = 8).

---

### Phase 2 — The unsteady Navier-Stokes operator

**Objective.** underPINN has no unsteady incompressible NS residual. Supply one
that conforms to its `BasePDE` contract so it composes with the rest of the
framework (and is contributable upstream).

**Approach.** `src/pinn/ns2d_unsteady.py`. Network `(x,y,t) -> (u,v,p)`, packed
as a single `(N,3)` array per underPINN's convention. Two derivative backends:
a literal `jax.hessian` reference, and a forward-over-forward JVP path computing
only the six derivative groups the residual needs (a full Hessian computes 27
derivatives where 4 are wanted).

**Result [MEASURED].** All three checks pass at machine precision:

| check | result |
|---|---|
| annihilates the Taylor-Green vortex (an exact NS solution) | 1.6e-16 |
| fast JVP backend vs literal Hessian | 2.8e-17 |
| steady limit vs underPINN's own `NavierStokesPDE` | 5.6e-17 |
| speed-up of the fast backend | **2.5x** |

The third check matters most: it proves the new operator is consistent with the
framework's existing physics rather than a parallel implementation that merely
looks right.

---

### Phase 3 — Reproduce the collapse

**Objective.** Demonstrate, on our own validated data, the documented failure:
a data-free unsteady PINN converges to the steady symmetric base flow instead of
shedding. This is the negative result the rest of the project is built on.

**Approach.**

```bash
python experiments/run_ablation.py --tag re100_v4 --only baseline --epochs 20000
```

Plain MLP `[3,128x6,3]`, physics + boundary + initial conditions, **no data**,
single time window of one shedding period.

**Result [MEASURED].** A100 GPU, 20 000 epochs, 2 seeds (`runs/ablation`):

| quantity | seed 0 | seed 1 | estimate was |
|---|---|---|---|
| shedding retained | 9.8 % | 11.3 % | 0–10 % — right |
| relative L2 | 0.184 | 0.183 | 0.2–0.4 — slightly pessimistic |

The collapse reproduces: the wake keeps about a tenth of the reference
amplitude, i.e. the PINN is solving the steady problem. Raising the budget to
80 000 epochs (`runs/ctrl_budget80k`) moves it only to **21.9 %**, so the
collapse is not an under-training artefact.

**Deliverable.** One figure: PINN vs LBM vorticity at the same `t*`, showing a
symmetric steady wake where the reference has a Kármán street —
`figures/ablation_cpu_fields.png`, from `experiments/plot_fields.py`.

> If shedding retained comes out high here, **be suspicious before being
> pleased**: check that the initial condition is not simply being memorised
> across the window, and that the evaluation window really extends beyond it.

---

### Phase 4 — Defeat the collapse

**Objective.** Determine which mechanism, if any, restores shedding — and
whether the mechanisms are individually sufficient or only jointly.

**Approach.** One script, mechanisms as switches, so the ablation is controlled:

```bash
python experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 2
```

| config | mechanism | attacks |
|---|---|---|
| `baseline` | — | (control) |
| `fourier` | random Fourier features (§1.6) | spectral bias |
| `decompose` | Reynolds decomposition (§1.7) | conditioning / dynamic range |
| `march` | causal time-marching (§1.8) | weak initial condition, causality |
| `all` | all three | — |
| `all+data` | plus 4000 supervised points/window | — |

**Result [MEASURED].** A100 GPU, 20 000 epochs per window, 2 seeds each, seed
means (`runs/ablation`, figure `figures/ablation.png`):

| config | shedding retained | relative L2 | estimate was |
|---|---|---|---|
| `baseline` | 10.6 % | 0.183 | 0–10 % — right |
| `fourier` | 9.7 % | 0.167 | 10–35 % — too optimistic |
| `decompose` | 7.0 % | 0.110 | 5–25 % — right, at the bottom |
| `march` | **89.2 %** | 0.078 | 25–60 % — **badly under-estimated** |
| `all` | **90.2 %** | 0.047 | 40–85 % — under-estimated |
| `all+data` | 100.0 % | **0.015** | 80–100 % — right |

**Causal time-marching alone is sufficient, data-free.** That is the headline,
and it is the "one mechanism suffices" row of the interpretation table below.
The other two mechanisms do not restore shedding on their own, but they roughly
halve the error once marching is in place (0.078 → 0.047).

*Equal-budget control (`runs/ctrl_budget80k`).* `march` runs 4 windows, so it
takes 4x the optimiser steps of a single-window config. Re-running the
single-window configs at 80 000 epochs — the same total — gives **21.9 %**
(`baseline`) and **23.8 %** (`decompose`). Both improve, neither approaches
89 %. **The gain is causality, not compute.** Quote this control alongside the
headline; without it the comparison is confounded.

*Knobs (`runs/ablation_sigma8`, `runs/ablation_wic100`).*

| change | effect |
|---|---|
| `--sigma 8` on `all` | 90.2 % -> **97.6 %**, L2 0.047 -> **0.042** — the best data-free result |
| `--sigma 8` on `fourier` alone | 9.7 % -> 11.9 %: still collapsed. Spectral bias was never the binding constraint |
| `--w-ic 100` on `all` | 89.8 %, L2 0.046 — no change. Once marching supplies the causal chain, the IC weight does not matter |

*How little supervision is needed (`runs/data_n*`, `all` + data, 2 seeds,
seed means with the spread in brackets):*

| data points per window | shedding retained | relative L2 |
|---|---|---|
| 0 | 90.2 % (90.0, 90.5) | 0.047 |
| 100 | 96.2 % (94.6, 97.7) | 0.030 |
| 500 | 97.6 % (97.0, 98.2) | 0.021 |
| 2000 | 99.5 % (99.0, 99.9) | 0.016 |
| 4000 | 100.0 % (100.0, 100.1) | 0.015 |

Shedding saturates by 100 points; the remaining benefit of data is accuracy,
and it has largely flattened by 2000. So the physics is buying the *dynamics*,
and the data is buying the last factor of ~3 in error.

The error column is reproducible to three decimals across seeds (e.g. 0.0295
vs 0.0296 at 100 points), while shedding retained moves by up to 3 points.
That is the expected asymmetry: L2 is an average over the whole field, the
shedding ratio is one probe's r.m.s., so quote L2 when a tight number is
needed and the shedding ratio when the question is whether the wake survived.

**How to interpret whatever actually happens** (the row that fired is the first):

| outcome | meaning | how to write it |
|---|---|---|
| one mechanism suffices | that mechanism is the contribution | lead with it; ablation is the evidence |
| only `all` works | mechanisms are complementary | the ablation *is* the result |
| only `all+data` works | reproduces the literature: PINNs need data here | legitimate negative result — then quantify *how little* data suffices |
| nothing works | the collapse is robust at this budget | report honestly; sweep `--n-data 0/100/500/2000/8000` for the minimum supervision that restores shedding |

**Budget is what separates these results from the CPU sweep.** The same matrix
at 1000–4000 epochs (`runs/ablation_cpu`) gave `march` 3.7 % and `all` 35 %,
which reads as "only the combined mechanisms work". That conclusion was an
artefact of under-training. Do not quote the CPU numbers as the result.

---

### Phase 5 — The FNO comparison (code written, runs pending)

**Objective.** Two things: (a) reproduce the reference paper's method on our
data, and (b) answer the question that unites the topic with the reference —
**does adding physics to a data-driven operator improve forecast stability?**

**Approach.**

*(a) Reproduce.* underPINN ships `FNO2D` in `underPINN/nn/operators.py` and an
`OperatorSolver` in `underPINN/solver/operator.py`. The paper's exact recipe
(its Table 1 and §3.1):

```
Reynolds-decompose first: train on u' = u - u0, not u
input   : 2 consecutive time steps    output: the next step
rollout : applied recursively 10 steps; loss summed over EVERY step
width 80, modes 24, Adam lr 1e-3, 200 epochs, batch 16, StepLR(step 5, 0.90)
metric  : eps = ||q*-q||_2/||q||_2 over stacked (u,v)
```

`data.py` already exposes `.fluct` and `.sample(..., fluctuation=True)`, and
`CylinderDataset.relative_l2` already implements the paper's metric.

*(b) Contribute.* Add a physics term to the FNO loss and measure rollout error
at **10, 20 and 50 steps**, i.e. well beyond the trained horizon. Plot `eps`
vs rollout step for both variants.

**Code [WRITTEN, NOT YET RUN AT FULL BUDGET].**

| file | what |
|---|---|
| `src/fno/physics.py` | grid residuals: continuity, and vorticity transport |
| `src/fno/cylinder_fno.py` | data pipeline, FNO2D wrapper, rollout, training, rollout-error evaluation |
| `experiments/run_fno.py` | runs the variants under identical settings; resumable like `run_ablation.py` |
| `experiments/plot_rollout.py` | `eps` vs forecast horizon, trained horizon shaded |
| `tests/test_fno.py` | 9 guards, all passing |

```bash
python experiments/run_fno.py --tag re100_v4 --epochs 200     # data, physics, physics_div
python experiments/plot_rollout.py --run-name fno
python src/fno/cylinder_fno.py --quick                        # CPU sanity run, ~1 min
```

Four decisions worth knowing before reading any result it produces:

* **The residual is vorticity transport, not momentum.** The network predicts
  `(u, v)` only, so any residual containing `grad p` is unusable. Taking the
  curl of the momentum equations removes the pressure, leaving
  `w_t + u.grad(w) - (1/Re) lap(w)`. Continuity is available separately and is
  the cheap minimum (`--variant physics_div`).
* **Physics terms are normalised by the residual the LBM data itself leaves**
  (`reference_residuals`). LBM is weakly compressible and second-order, so the
  measured field does not satisfy either equation exactly; pushing a prediction
  below that floor is fitting discretisation error, not physics. After the
  normalisation `w_div = w_vort = 1` means "penalise as hard as the data's own
  violation", which is why the weights survive a change of grid or `Re`.
* **Two error numbers are reported, `eps_full` and `eps_fluct`.** The model
  learns `u'`, but the paper reports error on the velocity. Adding the frozen
  mean back shrinks the relative error by roughly the mean-to-fluctuation
  energy ratio — a factor of several — for no modelling merit. Quote
  `eps_full` for comparability with the paper and `eps_fluct` as the honest
  difficulty measure, and never mix them in one table.
* **The split is contiguous in time**, train first and test after. A random
  split would leak: consecutive snapshots of a periodic wake are nearly the
  same picture, so the model would be scored on forecasting frames it had
  effectively already seen.

One deviation from the paper's Table 1: rollout activations are recomputed
(`jax.checkpoint`) rather than stored, which is what lets the paper's batch of
16 fit a 40 GB A100 at width 80. `--downsample 2` quarters the memory again if
needed. Neither changes the recipe. **[MEASURED]** on the A100: ~38 s per
epoch, so ~2.1 h per 200-epoch variant.

**Result [ESTIMATE].**

| case | eps @ 10 steps | eps @ 50 steps | reasoning |
|---|---|---|---|
| paper, Re 240–3060, PIV | < 0.1 (their measured value) | not reported | noisy experimental data, 3-D effects |
| our FNO, Re = 100, LBM | **0.01–0.05** | 0.1–0.4 | clean periodic data, far easier than theirs — this is *not* a win over the paper and must be said explicitly |
| our physics-informed FNO | 0.01–0.05 (similar) | **0.05–0.2** | physics should matter little inside the trained horizon and increasingly beyond it, where pure data-driven rollout drifts |

The expected shape of the result — *no benefit inside the trained horizon, growing
benefit outside it* — is itself the finding. If physics helps everywhere,
suspect a bug; if it helps nowhere, that is a real (and publishable) negative
result about the value of physics priors for short-horizon forecasting.

---

### Phase 6 — Write-up

**Suggested structure.** Introduction and the PINN-shedding problem → LBM solver
and its validation → the unsteady operator → the collapse (Phase 3) →
mechanisms and ablation (Phase 4) → FNO comparison (Phase 5) → conclusions.

**Figures that already exist:**

| file | shows |
|---|---|
| `figures/convergence.png` | solver validation: resolution vs blockage, extrapolation |
| `figures/re100_vorticity.png` | the Kármán street over one shedding period |
| `figures/re100_forces.png` | `C_l`/`C_d` histories and the lift spectrum |
| `figures/re100_mean.png` | the base flow `u0` |
| `figures/ablation.png` | **the Phase 4 headline**: shedding retained and L2 per config, full GPU budget |
| `figures/ctrl_budget80k.png` | the equal-budget control that rules out "marching just trains longer" |
| `figures/ablation_sigma8.png` | the σ = 8 variant, the best data-free result |
| `figures/ablation_cpu_fields.png` | PINN vs LBM vorticity — the collapsed baseline against the reference |
| `figures/fields_best.png` | the *successful* data-free PINN: a Kármán street, and the probe trace against LBM |

**Still to generate:** FNO rollout-error curve (Phase 5) — the script is
`experiments/plot_rollout.py`, waiting on the runs.

---

## Part III — Practical

### §7 Immediate blocker

Kaggle cannot clone the private GitHub repo — git prompts for a username and
hangs. GPU, CUDA JAX and the pip install are all already confirmed working there.

**Fix:** GitHub → repo → Settings → Change visibility → **Public**. Then the
clone cell in `KAGGLE.md` works unchanged (token route also documented there).

**Fallback:** `data/cylinder_re100_v4.npz` (190 MB) and
`data/base_flow_re100_v4.pkl` (70 KB) already exist locally and are validated —
upload them as a Kaggle Dataset and skip generation entirely.

### §8 Compute

CPU is fine for Phases 1–2. Phase 3 is not: ~8 h per 20 000-epoch run on CPU,
and the ablation is a dozen runs. Everything is JAX and runs unchanged on GPU.
Kaggle gives 30 GPU-h/week, ~9 h per session; use **Save Version → Save & Run
All** so it survives disconnection. Everything defaults to **float32**
deliberately — a T4 runs float64 at 1/32 rate.

**[MEASURED] on an A100** (IITK `kanaada`, see §11): ~12 ms per epoch at the
default point counts, so 3–4 min for a single-window 20 000-epoch run and
~16 min for a 4-window one. The whole matrix above — 12 ablation runs, 4 knob
runs, 3 data-sweep runs and 4 control runs — took **under 4 h**. That is the
same work the CPU sweep spent ~10 h on at a fifth of the budget.

### §9 Sanity check after any change

```bash
python tests/test_ns2d_unsteady.py && python tests/test_lbm_freeslip.py &&   python tests/test_metrics.py && python tests/test_fno.py
```

All four must pass. They catch failure modes that are otherwise invisible: a
wrong PDE residual still trains, a leaky wall still looks like a wake, and a
transposed metric still prints a plausible number.

### §10 Things that went wrong — do not rediscover these

* **A transposed array in `relative_l2` does not raise.** It silently compared
  6 values instead of 58 102 and returned a believable number. This was live in
  `evaluate()`; now guarded by `tests/test_metrics.py`. Any metric refactor must
  keep that test.
* **`St` from an FFT peak is a trap.** With ~12 recorded periods the bin width
  (0.014) exceeds the error being measured. Use the zero-crossing estimator in
  `lbm_cylinder.strouhal()`. An early "+0.3 % accurate" claim was pure bin luck;
  the truth was +2 %.
* **TRT was expected to fix the `C_d` bias** — it moved it ~1 %. Blockage was
  the cause, by roughly 5x.
* **Under-resolution was expected to dominate** — the grid was already converged
  at `D = 16`.
* **The Reynolds decomposition does not exclude the trivial solution** (§1.7).
* The resolution series in `convergence_sweep.py` was run at 5 % blockage, where
  blockage masks resolution. The blockage series is sound; redo the resolution
  series at `H = 40 D` if a clean resolution answer is wanted.
* `Cl_rms` is not well converged across runs (0.18–0.54 depending on
  configuration, benchmark ~0.23). Do not quote it as a validation metric
  without longer averaging.
* **float64 is the wrong default on GPU** (1/32 rate on a T4).
* **An under-trained ablation gives the wrong qualitative answer, not just a
  worse number.** At 1000–4000 epochs `march` scored 3.7 % and only `all`
  worked, which reads as "the mechanisms are complementary". At 20 000 epochs
  `march` alone scores 89 % and the honest conclusion is the opposite: one
  mechanism suffices. Fix the budget before interpreting an ablation.
* **A 4-window run takes 4x the optimiser steps of a 1-window run**, so the
  ablation matrix is not budget-matched by construction. `runs/ctrl_budget80k`
  is the control that makes the marching claim defensible; do not drop it.
* **`--sigma` mattered only in combination** (97.6 % with marching, 11.9 %
  without). A knob that does nothing alone can still be worth several points on
  top of the mechanism that works — test knobs on the winning config, not the
  baseline.

### §11 The IITK GPU server

```bash
ssh iitkgpu     # 172.27.21.120, user phyguest, key ~/.ssh/iitk_gpu
```

Two A100 40 GB cards. GPU 0 is usually busy with other users, so pin
`CUDA_VISIBLE_DEVICES=1`. It is a **shared guest account** — write only inside
`/home/phyguest/rajeshr/peeyush`.

Constraints found the hard way, all still true:

* **`/home` is 100 % full** (a few MB free). No room for the 190 MB dataset, so
  the server carries a 10 MB trimmed copy: 34 snapshots covering `t* 91.1–97.4`,
  which is exactly the span every run trains and scores on. The wake-probe
  reference comes out at 0.4118 either way, so scores are comparable with the
  CPU runs.
* **No internet** (pip fails on an SSL intercept). `einops`, which underPINN
  imports, is symlinked from another env into `peeyush/pylib`.
* **No JAX in the default python.** Use the existing CUDA env:
  `~/anaconda3/envs/graphcast-env/bin/python` (jax 0.4.31, flax 0.9.0), with
  `PYTHONPATH=$P/pylib:$P/underPINN_GPU_Ready/underPINN_GPU:$P/AE496_UGP/src`.
* That env has **numpy 1.26**, so a pickle written by numpy 2 will not load.
  The server's `base_flow_re100_v4.pkl` was re-saved for it; if you refit the
  base flow locally you must convert it again.

`gpu_sweep.sh`, `gpu_control.sh` and `gpu_fields.sh` in the repo root run the
matrix unattended (`setsid nohup ./gpu_sweep.sh &`), and `run_ablation.py`
skips configs already in `summary.json`, so a killed sweep resumes.
