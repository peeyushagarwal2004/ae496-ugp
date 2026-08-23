# Resume pointers — AE496 UGP, start to finish

Written so the project can be picked up cold: by you in a month, by a fresh
Claude session, or by your supervisor. Read `README.md` first for what the
project *is*; this file is where it stands and what remains.

---

## 1. State of play

| phase | state | evidence |
|---|---|---|
| 1. Ground-truth CFD | **done, validated** | `figures/convergence.png`, table below |
| 2. Unsteady NS operator | **done, validated** | `tests/test_ns2d_unsteady.py`, 1e-16 |
| 3. Reproduce the collapse | code ready, **not run** | needs GPU |
| 4. Defeat the collapse | code ready, **not run** | needs GPU |
| 5. FNO comparison | **not written** | see §5 |
| 6. Write-up | not started | see §6 |

Everything through phase 2 is finished. Phases 3 and 4 are a matter of running
`experiments/run_ablation.py` on a GPU. Phase 5 is the only remaining code.

### Numbers you can quote

Production dataset `re100_v4` (D2Q9 TRT lattice-Boltzmann, Re = 100):

```
Cd = 1.3798   benchmark 1.37-1.38    +0.3 %
St = 0.1611   benchmark 0.165        -2.3 %
divergence error 0.55 % of |grad u|  (tracks O(Ma^2), Ma = 0.087)
641 snapshots, t* = 91 .. 212, dt* = 0.1896, grid 241 x 121 at dx = 0.083 D
```

Solver validation (`data/convergence.json`):

```
resolution D = 16..40 at 5 % blockage : Cd spread 2.4 %   -> grid converged
blockage 5.0 / 3.3 / 2.5 %            : Cd 1.627 / 1.459 / 1.431
extrapolated to zero blockage         : Cd 1.348  (-2.0 % vs benchmark)
```

Base flow for the Reynolds decomposition: `data/base_flow_re100_v4.pkl`,
relative L2 = **1.10e-3** (Fourier MLP 128x5, sigma = 12).

---

## 2. Immediate blocker

The Kaggle notebook cannot clone the private GitHub repo — git prompts for a
username and hangs. GPU, CUDA JAX and the pip install are all already
confirmed working there.

**Fix:** GitHub -> repo -> Settings -> Change visibility -> Public. Then the
clone cell in `KAGGLE.md` works unchanged. (Token route also documented there
if it must stay private.)

Fallback if Kaggle keeps fighting: `data/cylinder_re100_v4.npz` (190 MB) and
`data/base_flow_re100_v4.pkl` (70 KB) already exist locally and are validated —
upload them as a Kaggle Dataset and skip steps 1–2 entirely.

---

## 3. Phase 3 — reproduce the collapse

```bash
python experiments/run_ablation.py --tag re100_v4 --quick        # calibrate
python experiments/run_ablation.py --tag re100_v4 --only baseline --epochs 20000
```

**Expected:** `shedding retained` near 0 %. That is the *published* result
([arXiv:2306.00230](https://arxiv.org/abs/2306.00230)) and the foundation of the
report — not a bug to fix. Do not "improve" it away.

Deliverable: one figure of PINN vs LBM at the same t*, showing a symmetric
steady wake where the reference has a Kármán street. That figure justifies
everything in phase 4.

**Not yet written:** a plotting script for PINN fields. `src/cfd/inspect_data.py`
does this for LBM data and is the obvious thing to adapt — it already handles
vorticity, the cylinder mask and the non-dimensional axes.

---

## 4. Phase 4 — defeat it

```bash
python experiments/run_ablation.py --tag re100_v4 --epochs 20000 --seeds 2
```

Six configs: `baseline`, `fourier`, `decompose`, `march`, `all`, `all+data`.
Results accumulate in `runs/ablation/summary.json`; re-running skips completed
entries, so an interrupted Kaggle session is not wasted.

### Read the outcome like this

| outcome | what it means | what to write |
|---|---|---|
| one mechanism lifts shedding alone | that mechanism is the contribution | lead with it; ablation is the evidence |
| only `all` works | mechanisms are complementary | the ablation *is* the result |
| only `all+data` works | reproduces the literature: PINNs need data here | still legitimate; frame as a negative result with a quantified data threshold |
| nothing works | the collapse is robust | report it; then sweep `--n-data` to find the minimum data that rescues shedding |

The last row is a real possibility, not a failure of the project. If it
happens, the interesting follow-up is *how little* data suffices — a sweep over
`--n-data 0 / 100 / 500 / 2000 / 8000` gives a curve of shedding retained vs
supervision, which is a genuinely publishable observation and directly answers
"is physics buying anything?".

### Knobs worth trying before concluding failure

* `--w-ic` (default 10) — the initial condition is what excludes the trivial
  steady solution. If shedding dies, raise it hard (100, 1000).
* `--window-len` (default 6.0, one shedding period; `march` uses 1.5) — shorter
  windows make the causal chain stronger.
* `--sigma` on the unsteady net (default 2.0). The base-flow fit wanted 12.
  The unsteady field has similar length scales, so 2.0 is probably too low —
  **this is the single most likely quick win.**
* `--n-interior` / `--n-wake` — 8000/4000 is modest for a 3-D (x,y,t) domain.

---

## 5. Phase 5 — the FNO comparison (only remaining code)

This is where the reference paper comes back in. Two jobs:

**(a) Reproduce the paper on our data.** underPINN ships `FNO2D` in
`underPINN/nn/operators.py` and an `OperatorSolver` in `underPINN/solver/operator.py`.
The paper's exact recipe (its Table 1 and §3.1):

```
Reynolds decomposition first: train on u' = u - u0, not u
input  : 2 consecutive time steps      output: the next step
rollout: applied recursively 10 steps, loss summed over every step
width 80, modes 24, Adam lr 1e-3, 200 epochs, batch 16, StepLR(5, 0.90)
metric : eps = ||q* - q||_2 / ||q||_2 over stacked (u, v)  -- already
         implemented as CylinderDataset.relative_l2
```

Our `dt* = 0.1896` was chosen deliberately to sit near the paper's 0.17, and
`dx = 0.083 D` near their PIV resolution of < 0.1 D, so the comparison is
close to like-for-like. `data.py` already exposes `.fluct` for the
decomposition and `.sample(..., fluctuation=True)`.

Their result to compare against: **eps < 0.1 across all Re over 10 steps.**

**(b) The actual contribution.** Add a physics residual (or at minimum a
divergence-free penalty) to the FNO loss and ask whether it improves rollout
stability *beyond* the trained 10-step horizon. Pure data-driven FNO is known
to drift; if physics regularisation extends the usable horizon, that is the
result that unites your topic (PINN) with your reference (FNO). Test at 10, 20,
50 steps and plot eps vs rollout step for both.

Note: at Re = 100 the flow is periodic and much easier than the paper's
Re = 240–3060, so expect eps well below theirs. That is not a win over the
paper — say so explicitly.

---

## 6. Phase 6 — write-up skeleton

Figures that already exist:

* `figures/convergence.png` — solver validation (§1)
* `figures/re100_vorticity.png` — the Kármán street
* `figures/re100_forces.png` — Cl/Cd histories and lift spectrum
* `figures/re100_mean.png` — the base flow u0

Still to generate: PINN-vs-LBM field comparison, ablation bar chart
(shedding retained per config), FNO rollout-error curve.

Suggested structure: introduction and the PINN-shedding problem → LBM solver
and its validation → the unsteady operator → the collapse (phase 3, with the
figure) → mechanisms and ablation (phase 4) → FNO comparison (phase 5) →
conclusions.

---

## 7. Things that went wrong — do not rediscover these

* **`St` from an FFT peak is a trap.** With ~12 recorded periods the bin width
  (0.014) exceeds the error being measured. The zero-crossing estimator in
  `lbm_cylinder.strouhal()` is the correct one. An early "+0.3 % accurate"
  claim was pure bin luck; the true figure was +2 %.
* **A transposed array in `relative_l2` does not raise.** It silently compares
  two grid points instead of 58 102 values, and returns a plausible-looking
  number. This bug was live in `evaluate()` and is now guarded by
  `tests/test_metrics.py`. Any metric refactor must keep that test.
* **float64 is the wrong default on GPU** (1/32 rate on a T4). Everything
  defaults to float32; `--f64` is for precision checks only.
* **TRT collision was expected to fix the C_d bias** — it moved it ~1 %.
  Blockage was the cause, by roughly 5x.
* **Under-resolution was expected to dominate** — the grid was already
  converged at D = 16.
* **The Reynolds decomposition does not exclude the trivial solution.**
  u' = 0 solves the fluctuation equations. It buys conditioning only; the
  initial condition plus time-marching is what rules out steady flow. Do not
  write the stronger claim in the report.
* The resolution series in `convergence_sweep.py` was run at 5 % blockage,
  where blockage dominates and masks resolution. The blockage series is sound;
  redo the resolution series at H = 40 D if a clean resolution answer is
  needed.
* `Cl_rms` is not well converged across runs (0.18–0.54 depending on
  configuration, benchmark ~0.23). Do not quote it as a validation metric
  without longer averaging.

---

## 8. Sanity check after any change

```bash
python tests/test_ns2d_unsteady.py && python tests/test_lbm_freeslip.py && python tests/test_metrics.py
```

All three must pass. They are fast and catch the failure modes that are
otherwise invisible: a wrong PDE residual still trains, a leaky wall still
looks like a wake, and a transposed metric still prints a number.
