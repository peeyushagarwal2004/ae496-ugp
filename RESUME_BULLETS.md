# Resume bullets — AE496 UGP

Concept-forward, 1–2 lines each. Bold = keywords (→ `\textbf{}`). One variant per
application. Numbers only where measured; shedding/FNO results pending.

---

## For ML / SDE roles

**Physics-Informed Neural Networks for Fluid-Flow Prediction** · [GitHub] — 2026

* Learning a continuous flow field from PDE residuals via **PINNs** +
  **automatic differentiation** in **JAX / Flax / Optax**.
* Applied **Fourier-feature embeddings** over MLP/SIREN for a **7.7×** accuracy
  gain; **FNOs** with **autoregressive rollout** for forecasting.
* Cut training-step cost **2.5×** with a **forward-over-forward JVP** scheme on a
  **float32, GPU-portable** stack.

## For Aerospace / Core roles

**PINN-Based Prediction of Vortex Shedding Behind a Cylinder** · [GitHub] — 2026

* Solving the **unsteady incompressible Navier–Stokes** equations with **PINNs**,
  **Reynolds decomposition**, and **causal time-marching**.
* Built a **D2Q9 lattice-Boltzmann** reference solver (**TRT** collision,
  **momentum-exchange** force), validated to **0.3% drag error** at Re=100.
* Bridging **PINN** and **Fourier-Neural-Operator** methods under one validated
  dataset and error metric.

---

## Standout achievements (swap in if a line frees up)

* Contributed the **unsteady 2-D Navier–Stokes operator** missing from the
  **underPINN** library, verified to **1e-16**.
* Traced a **+18% drag bias** to domain blockage (not the solver) via a
  **convergence study**, overturning two wrong hypotheses.

---

## Planned / in-progress (method-forward; add results once measured)

High-signal techniques on the roadmap — safe to list as *methods in use* even
before results land. Pick the 1–2 that fit the target role.

* **Self-adaptive loss weighting** (NTK / gradient-norm balancing) to
  auto-tune the PDE/BC/IC/data terms — the core difficulty in multi-objective
  PINN training.
* **Residual-based adaptive collocation (RAR-D)** — resampling points where the
  PDE residual is largest, instead of uniform sampling.
* **Physics-regularised Fourier Neural Operator** — adding a divergence-free /
  NS-residual penalty to a data-driven operator to extend its stable
  **autoregressive rollout** horizon (the project's novel contribution).
* **Sparse-sensor flow reconstruction** — inverse PINN recovering the full
  field (and pressure) from a few velocity probes, the regime where physics
  priors most outperform pure ML.
* **Distributed data-parallel training** across multi-GPU (T4×2) via JAX
  `pmap`/sharding.
