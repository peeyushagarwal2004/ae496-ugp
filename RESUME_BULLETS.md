# Resume bullets — AE496 UGP

Every number here is **measured**, traceable to a file in this repo, and
defensible in an interview. Nothing predicted or projected.

Bold = tech keywords (convert to `\textbf{}` in LaTeX).

---

## Version A — full entry (5 bullets)

**Physics-Informed Neural Networks for Vortex Shedding Prediction** [GitHub]
*AE496 Undergraduate Project, IIT Kanpur* — Aug'26

* Built an unsteady **Navier-Stokes PINN** framework in **JAX/Flax/Optax**,
  contributing the unsteady 2-D operator absent from the **underPINN** library;
  validated to **1e-16** against the analytic **Taylor-Green vortex**.
* Engineered a **D2Q9 lattice-Boltzmann** CFD solver (**TRT** collision,
  momentum-exchange forcing, half-way bounce-back) generating a **641-snapshot**
  ground-truth wake at **C_d = 1.3798 vs the 1.37-1.38 benchmark (0.3 % error)**.
* Diagnosed a systematic **+18 % drag bias** to domain blockage — not
  discretisation — through a 6-configuration convergence study, extrapolating to
  **1.348 at zero blockage (2.0 % error)**.
* Cut solver runtime **3x** (float32 + windowed force reduction) and PINN
  gradient cost **2.5x** by replacing full-Hessian autodiff with
  forward-over-forward **JVPs**, computing 4 derivatives instead of 27.
* Benchmarked **random Fourier features** against MLP/SIREN encodings,
  achieving **7.7x lower** relative L2 (**8.9e-3 vs 6.9e-2**) on flow-field
  regression.

---

## Version B — compact (3 bullets, if space is tight)

**Physics-Informed Neural Networks for Vortex Shedding Prediction** [GitHub]
*AE496 Undergraduate Project, IIT Kanpur* — Aug'26

* Built an unsteady **Navier-Stokes PINN** in **JAX/Flax**, contributing the
  unsteady 2-D operator missing from the **underPINN** library; validated to
  **1e-16** on the analytic **Taylor-Green vortex**.
* Engineered a **D2Q9 lattice-Boltzmann** solver (**TRT** collision, momentum
  exchange) hitting **C_d = 1.3798 vs 1.37-1.38 benchmark (0.3 % error)**, and
  traced a **+18 %** drag bias to domain blockage via a convergence study.
* Optimised training **2.5x** with forward-over-forward **JVP** autodiff and
  **7.7x** lower error using **random Fourier features** over standard MLPs.

---

## Version C — ML/SWE-leaning (if the role is not aerospace)

Reframes the same work around engineering rather than fluid mechanics.

* Built a **JAX/Flax** scientific-ML pipeline (**PINNs**, **neural operators**)
  with a **GPU-portable** float32 training path, checkpointing and a
  reproducible experiment harness across a 6-configuration ablation.
* Optimised **automatic-differentiation** cost **2.5x** by hand-deriving a
  forward-over-forward **JVP** scheme computing only 4 of 27 Hessian entries,
  and **3x** more via float32 and windowed tensor reductions.
* Wrote a physics-validated **numerical solver** in JAX verified to **1e-16**
  against an analytic solution, plus a regression suite that caught a silent
  metric bug comparing **6 values instead of 58,102**.
* Ran a systematic **convergence/ablation study** isolating a **+18 %**
  systematic error to its true cause, with **7.7x** model-accuracy improvement
  from an encoding change (**8.9e-3 vs 6.9e-2** relative L2).

---

## Bullets to ADD once Phases 3-5 finish

Do **not** use these yet. Fill the brackets with real measured numbers.

* Reproduced the documented failure of data-free PINNs on vortex shedding,
  quantifying wake collapse at **[X] %** shedding retained vs the CFD reference.
* Restored shedding to **[X] %** via **[mechanism]**, isolated through a
  6-configuration, 2-seed ablation over Fourier features, **Reynolds
  decomposition** and causal time-marching.
* Implemented a **Fourier Neural Operator** (width 80, 24 modes) with
  **10-step autoregressive rollout**, reaching **eps = [X]** relative L2, and
  showed physics-regularised training extends the stable horizon **[X]x**.

---

## Interview prep — what you will be asked

**"What is a PINN?"** A neural network that maps `(x, y, t) -> (u, v, p)` and is
trained so its own **automatic derivatives** satisfy the Navier-Stokes
residual at randomly sampled collocation points, rather than being fitted to
data. Mesh-free, and can run data-free.

**"Why is the 0.3 % number impressive?"** It is not accuracy of a model — it is
**solver validation** against published DNS benchmarks. It means the
ground-truth data is trustworthy, which is the prerequisite for every claim
downstream.

**"What was the hardest part?"** Diagnosing the +18 % drag error. My first two
hypotheses (relaxation-time scheme, grid resolution) were both wrong — the
convergence study showed the grid was already converged at the coarsest
resolution and blockage was the real cause, by roughly 5x. Good answer because
it shows you let measurement overrule intuition.

**"Tell me about a bug you found."** A transposed array in the error metric.
NumPy broadcasting made it silently compare 6 values instead of 58,102 and
return a plausible-looking number — no exception, no warning. It would have
invalidated a whole results table. Now covered by a regression test.

**"Why JAX?"** Automatic differentiation to arbitrary order (a PINN needs second
derivatives of the network w.r.t. its inputs), `jit` compilation via XLA, and
identical code on CPU and GPU.
