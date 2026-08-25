# Resume bullets — AE496 UGP

Concept-forward: each bullet names the **technical methods used** and folds in a
**special achievement** from building it. Not a results log. Numbers appear only
where already measured; the shedding/FNO outcomes are pending, so they are not
claimed.

Bold = keywords (→ `\textbf{}` in LaTeX). Use ONE variant per application.

---

## For ML / SDE roles

**Physics-Informed Neural Networks for Fluid-Flow Prediction** · [GitHub]
*Undergraduate Research Project, IIT Kanpur* — 2026

* Applying **physics-informed neural networks (PINNs)**, **automatic
  differentiation**, and **Fourier-feature / SIREN embeddings** in **JAX / Flax /
  Optax** to learn a continuous flow field from PDE residuals — spectral
  encoding chosen for a measured **7.7×** accuracy gain over plain MLPs.
* Using **Fourier Neural Operators** with **autoregressive rollout** and
  **Reynolds decomposition** to forecast flow evolution and study
  physics-regularised vs. data-only operator learning.
* Engineered a **forward-over-forward JVP** derivative scheme (4 of 27 Hessian
  entries) for a **2.5×** faster training step on a **float32, GPU-portable**
  stack, with checkpointing and a reproducible ablation harness.

## For Aerospace / Core roles

**PINN-Based Prediction of Vortex Shedding Behind a Cylinder** · [GitHub]
*Undergraduate Research Project, IIT Kanpur* — 2026

* Applying **physics-informed neural networks** to the **unsteady incompressible
  Navier–Stokes** equations, with **Reynolds decomposition** and **causal
  time-marching** to target the wake's known collapse-to-steady failure mode.
* Built a **D2Q9 lattice-Boltzmann** reference solver (**TRT** collision,
  **momentum-exchange** surface force) validated to **0.3% drag error** against
  the Re=100 benchmark, with a **convergence study** separating discretisation
  from **domain-blockage** effects.
* Bridging **PINN** and **Fourier-Neural-Operator** methods under one validated
  dataset, non-dimensionalisation, and error metric, to compare
  physics-constrained solving against data-driven forecasting.

---

## Standout achievements (drop into either variant if a line frees up)

* Contributed the **unsteady 2-D Navier–Stokes operator** absent from the
  open-source **underPINN** library, verified to **1e-16** against a closed-form
  solution.
* Diagnosed a systematic **+18% drag bias** to domain confinement (not the
  solver) through controlled grid/blockage sweeps — overturning two wrong
  initial hypotheses.

---

## Interview prep

* **"What's a PINN?"** A network `(x,y,t)→(u,v,p)` trained so its own autodiff
  derivatives satisfy the Navier–Stokes residual at sampled points — mesh-free,
  can run data-free.
* **"FNO vs PINN?"** PINN learns one flow field constrained by physics; FNO
  learns an *operator* (field-now → field-next) from data and rolls forward
  autoregressively. The project unites both.
* **"Why Fourier features?"** Networks are spectrally biased toward low
  frequencies; a `[sin(Bx),cos(Bx)]` encoding injects the high frequencies a
  vortex street needs — worth 7.7× here.
* **"Hardest part?"** The +18% drag bias: my first two hypotheses were wrong, and
  a convergence study proved blockage was the cause. Measurement over intuition.
