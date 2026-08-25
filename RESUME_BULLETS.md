# Resume bullets — AE496 UGP

Rule: every number is **measured** and traceable to this repo. Methodology is
stated as *approach* (what the system does / how), never as an unearned result.
The vortex-shedding and FNO outcomes are not listed because those runs are
pending — add them from `RESUME.md`'s bracketed lines once measured.

Bold = keywords (→ `\textbf{}` in LaTeX). Pick ONE variant per application.

---

## For ML / SDE roles

**Physics-Informed Neural Networks for Fluid-Flow Prediction** · [GitHub]
*Undergraduate Research Project, IIT Kanpur* — 2026

* Contributed the **unsteady 2-D Navier–Stokes operator** missing from the
  **underPINN** (JAX/Flax) library, letting a network `(x,y,t)→(u,v,p)` learn a
  PDE from its own **autodiff** residual; verified to **1e-16** against a
  closed-form solution.
* Rebuilt the residual's derivative path with **forward-over-forward JVPs**,
  evaluating 4 of 27 Hessian entries for a **2.5× faster** training step, on a
  **float32, CPU/GPU-portable** stack.
* Ran a controlled **ablation harness** (spectral Fourier features vs standard
  MLP/SIREN encodings) that cut flow-field regression error **7.7×**
  (8.9e-3 vs 6.9e-2 relative L2).
* Hardened the pipeline with a regression suite that caught a **silent
  broadcasting bug** in the eval metric — comparing 6 values instead of 58,102 —
  that had returned plausible-but-wrong numbers.

## For Aerospace / Core roles

**PINN-Based Prediction of Vortex Shedding Behind a Cylinder** · [GitHub]
*Undergraduate Research Project, IIT Kanpur* — 2026

* Engineered a **D2Q9 lattice-Boltzmann** solver (**TRT** collision,
  momentum-exchange surface force, half-way bounce-back) for the Re=100 cylinder
  wake, matching the benchmark at **C_d = 1.3798 (0.3% error)** and **St = 0.161**.
* Isolated an apparent **+18% drag error** to **domain-blockage** confinement —
  not discretisation — via a grid/blockage **convergence study**, extrapolating
  to **C_d = 1.348 (2.0%)** at zero blockage.
* Implemented the **unsteady incompressible Navier–Stokes** PINN operator absent
  from underPINN, plus a **Reynolds-decomposition** scheme (frozen mean flow +
  learned fluctuation) to target the wake's known collapse-to-steady failure.
* Designed the study to **bridge PINN and Fourier-Neural-Operator** approaches
  from the reference literature under a single validated dataset and error metric.

---

## Add once the GPU runs finish (from RESUME.md, fill brackets with measured values)

* Reproduced and then **defeated the documented collapse-to-steady failure** of
  data-free PINNs, restoring **[X]%** shedding fidelity via **[mechanism]**.
* Built a physics-regularised **FNO** with **10-step autoregressive rollout**,
  extending the stable forecast horizon **[X]×** over the data-only baseline.

---

## Interview prep

* **"What's a PINN?"** A network `(x,y,t)→(u,v,p)` trained so its *own* autodiff
  derivatives satisfy the Navier–Stokes residual at sampled points — mesh-free,
  runs data-free.
* **"Why is 0.3% impressive?"** It's not model accuracy — it's *solver
  validation* against published DNS, which is what makes every downstream claim
  trustworthy.
* **"Hardest part?"** Diagnosing the +18% drag bias. My first two hypotheses
  (collision scheme, grid resolution) were wrong; the convergence study proved
  the grid was already converged and **blockage** was the real cause. Let
  measurement overrule intuition.
* **"A bug you found?"** A transposed array made NumPy silently compare 6 values
  instead of 58,102 and return a believable number — no exception. Now covered
  by a regression test.
