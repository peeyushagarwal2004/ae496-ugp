# Literature survey and novelty assessment

Survey date 2026-09-22, second pass. Companion to [`references.bib`](references.bib),
which carries the per-entry verification status and the raw numbers quoted here.

**Verification convention.** `[V]` = title/authors/venue read off the arXiv or
publisher page. `[VR]` = the paper itself was opened and the stated content read
from it. `[U]` = unverified, or known only through another paper's citation of
it — yours to confirm. `[NF]` = not found after a deliberate multi-round search,
which is weak evidence of novelty and not proof.

Five papers were read rather than skimmed on this pass: Chuang & Barba,
Shah, Krishnapriyan et al. (§5), Qu et al. (Tables 1–3), and Blevins (p. 1).

---

## 0. Third pass (2026-10-02): what the regenerated results change

Sections 1–13 below were written against the v4 results. The v4 headline did
not survive the confound checks (report §4–§6), and several positions taken
below are now wrong. They are left in place, marked **[superseded]**, so the
history is readable; this section says what replaces them. Every number here is
in `report/numbers.json` and checked by `experiments/verify_report.py`.

**The Shah position reverses (§1, §9.1).** The v4 data-free result (89–90 %
retained) was inflated twice: marched runs got a fresh reference snapshot at
every window (four ground-truth slices against the baseline's one, worth about
twenty points), and every configuration was scored at one start phase of a
quantity that swings with a half-cycle period. With one ground-truth slice and
four phases averaged: plain PINN 30.1 %, `march` 56.7 %, `all` (σ = 8) 75.1 %
at L₂ 0.031. Over five chained periods `all` runs 10.4 % slow and ends about
half a cycle out of phase. So:

- Do **not** claim to contradict Shah or Chuang & Barba. A data-free chain
  sustains a wake at three-quarters amplitude but does not forecast it.
- Shah's title, *Complementary, Not Cumulative*, now **agrees** with the
  finding here: only all three mechanisms together give a phase-independent
  wake. Cite him in support of the interaction effect.
- What remains distinct from Shah: hard window marching (he tests loss
  reweighting over time bins), phase-averaged scoring, and the
  optimiser-step-matched control.

**The equal-budget control now cuts the other way (§3, §9.3).** At matched
optimiser steps (80 000 epochs, four phases, two seeds), training budget
accounts for more of `march`'s gain than causality does: +17.3 points from
training alone, +9.3 from causality. Within the combination causality is worth
+28.4. The distinction from Krishnapriyan et al. (collocation-point matching
vs optimiser-step matching) still stands; the claim "the gain is causality, not
compute" does not.

**The Cd story changes (§4 points 1–2, §9.4, §11.5).** The v4 overshoot was
mostly the inlet 8D ahead of the cylinder, not the domain height. With the
inlet at 25D and an absorbing layer: Re = 100 gives Cd +3.5 %, St −0.04 % vs Qu
et al.; raising H from 50D to 120D moves Cd by only 0.07 % once the inlet is far
enough upstream. Posdziech & Grundmann's >20D upstream requirement (via Qu) is
the relevant citation, not the H > 100 blockage threshold. The resolution
series (§11.7) has been redone at 2 % blockage: converged at D = 32.

**The acoustic section moved and was resolved (§6, §11.3–4).** A target-state
absorbing layer (relaxation toward free-stream equilibrium) removes the duct
mode; a viscous layer does not, for a quantified reason (≈0.04 % per pass). At
Re = 100 without the layer the wake locks at St/St_a = 1.43, well outside Hong
et al.'s 1.14, given enough time. The band discrepancy remains open and is
listed as such in the report; the lattice-velocity-lowering workaround is no
longer used.

**Novelty, re-ranked for the v5 evidence.**

1. **A controlled, phase-averaged decomposition of what defeats the collapse**:
   causality, compute and the two other mechanisms separated at matched
   optimiser steps, with one ground-truth slice per run.
2. **Two confounds that inflate PINN-marching results** — per-window
   ground-truth ICs and single-phase scoring — each measured. Of methodological
   interest beyond this problem.
3. **The acoustic lock-in diagnosis** and the target-state layer fix.
4. **Inlet distance, not blockage**, as the dominant Cd error for an LBM
   cylinder in a free-slip duct, by a crossed geometry test.
5. Reynolds-sweep generalisation (v5 sweep done: 63–79 % vs 24–45 % at
   Re 60–180), FNO physics residual (backbone-dependent, report §9), sequence
   PINNs (no substitute for marching, report §8), supervision volume —
   supporting results.

---

## 1. Headline: the competing paper does not achieve what you achieve **[superseded — see §0]**

**Shah, arXiv:2608.19632 (20 Aug 2026)** `[VR]` is not the threat it looked like
on the abstract. Reading the paper changes the conclusion:

- Their 4.1 % relative L₂ **requires data supervision.**
- Data-free at Re = 100, *none* of their six techniques or combinations escapes
  the collapse. They conclude: *"consistent with [Chuang and Barba (2023)], some
  form of data supervision proves necessary for learning this problem at all."*
- Their "causal weighting" is **loss reweighting over 20 time bins**. They never
  test hard time-window marching with a separate network per window and the end
  state as the next initial condition — which is precisely your mechanism.
- No equal-compute control. No seeds — single deterministic run per config.
- Two Reynolds numbers (20 and 100), not one.
- They *do* report wake amplitude ratio, wake correlation and Strouhal number.
  **Correct my earlier claim: they are not limited to field L₂.**

So your data-free result — 90.2 % shedding retained, relative L₂ 0.047, and
0.042 at σ = 8 — directly contradicts a negative claim made independently by
the two most relevant papers in the literature, one of them a month old. And the
mechanism your refutation rests on is the one neither of them tried.

That is a much stronger paper than "one of my three mechanisms worked."

**Suggested framing:** *Two independent studies conclude that data supervision
is necessary to sustain vortex shedding in a PINN. Both reach that conclusion
without testing hard causal time-window marching. With marching, and with no
data at all, the wake survives at 90 % of reference amplitude and the field
error matches the best data-supervised result in the literature.*

State plainly that the benchmarks differ — Shah uses the confined
DFG/Schäfer–Turek channel against OpenFOAM, you use a free-slip domain at 2 %
blockage against LBM — so the error figures are not directly comparable. Claim
the qualitative contradiction, not a numerical win.

---

## 2. Chuang & Barba is at Re = 200, and their PINN solves one 40-period window

Read in full `[VR]`. Facts your report needs:

- **The cylinder case is Re = 200**, not Re = 100. Domain [−8,25]×[−8,8],
  t ∈ [0,200], ν = 0.005. Re = 40 is their steady validation; the Re = 100
  case is a Taylor–Green vortex for verification only.
- **Their unsteady PINN solves a single time window spanning t ∈ [0,200]** —
  roughly 40 shedding periods in one optimisation, 6×512 MLP, 10⁶ Adam
  iterations, ~31 h on an A100.
- Cd at Re = 200: PetIBM 1.38, steady PINN 0.95, unsteady PINN 0.95.
- Reference solver is **PetIBM**, their own code — not OpenFOAM.
- Their data-driven variant replaces the IC losses with 16 snapshots over ~3
  periods; shedding stops when the data does. They conclude *"the perturbation
  is not the primary factor responsible for the steady-state behavior."*

Two consequences.

**You must say which Reynolds number you reproduce.** Your report claims to
reproduce "the documented failure"; the documented case is Re = 200 and your
sweep tops out at 180. A referee who knows the paper will ask immediately. The
honest and defensible statement is that you reproduce the *phenomenon* at
Re = 100 and across 60–180, on independently validated ground truth, not their
specific case. You can add that Re = 200 lies above the three-dimensional
transition at Re_c = 188.5 ± 1.0 (Barkley & Henderson 1996 `[V]`), so a 2D
simulation there is itself an idealisation — a fair observation, made once,
without picking a fight.

**Your baseline is a much more favourable configuration than theirs, and still
collapses.** One window of *one* shedding period versus their single window of
forty. That strengthens your reproduction considerably and is worth stating.

**Be careful contradicting their spectral-bias diagnosis.** Their attribution is
backed by Koopman/DMD analysis showing damped modes inside the unit circle and
frequencies absent from the data — a claim about *learned dynamic modes*. Your
`fourier` row shows that attacking spectral bias *in the input embedding* does
not restore shedding. Those are not the same claim, and overstating the
contradiction is an easy thing for a referee to puncture. The safe, stronger
version: marching restores shedding **without addressing spectral bias at all**,
which points to causality and conditioning as the binding constraint — exactly
what Krishnapriyan et al. argue from loss-landscape analysis.

---

## 3. Your equal-budget control is genuinely a different control **[conclusion superseded — see §0]**

Read §5.1–5.2 and Figure 5 of **Krishnapriyan et al. (2021)** `[VR]`.

They *do* run a fairness control, but it is over **collocation points**, not
optimiser steps. Figure 5 caption: *"To allow fair comparison, we keep the total
number of collocation points to be exactly the same in either approach."*
Footnote 4: *"for T = [0,1], if we use 1000 collocation points to predict the
entire state space, then for Δt = 0.1 we use 100 collocation points per
section."* Nothing in the paper matches total iterations or wall-clock between
seq2seq and single-shot.

So your 80 000-epoch single-window control answers a question they left open.
Say it precisely — *"prior work matched collocation-point count; we match
optimiser steps"* — rather than claiming nobody controlled anything. Precision
here is free credibility.

Their seq2seq tests: 1D convection, 1D reaction, 1D reaction–diffusion. **No
Navier–Stokes, no vortex shedding** `[VR]`. Combined with §1, the position is:
the mechanism is established on 1D problems; you are the first I can find to
show it suffices, data-free, on 2D Navier–Stokes wake shedding, with an
optimiser-matched control, across six Reynolds numbers.

---

## 4. Your Cd validation claim is wrong, and the fix is a better result **[cause superseded — see §0]**

This is the most consequential finding of the survey.

**Qu, Norberg, Davidson, Peng & Wang (2013),** *J. Fluids Struct.* **39**,
347–370 `[VR]` — Tables 1 and 3 read directly.

Their Table 1 compiles **eight independent studies at Re = 100** with large
domains. Cd: 1.33 (Park 1998), 1.32 (Kravchenko & Moin 1998), 1.318 (Shi 2004),
1.322 (Mittal 2005), 1.32 (Stålberg 2006), 1.325 (Posdziech & Grundmann 2007),
1.336 (Li 2009, an LBM study), 1.319 (their own).

**The Re = 100 consensus is Cd ≈ 1.32, not 1.37–1.38.** Your 1.3798 is therefore
about **+4.8 %**, not +0.3 %. Your zero-blockage extrapolation of 1.348 is +2.1 %,
not −2.0 %. The report's validation table currently overstates agreement, and the
`gt.Cd_error_pct` entry the number-checker already flagged as ambiguous is worse
than ambiguous — the reference itself is wrong. (1.38 is close to the *Re = 200*
value, which is where it may have come from.)

Two benchmarks in your report survive intact and are now citable from this
paper: **St = 0.165** (their range 0.164–0.166) and **Cl′ ≈ 0.23** (their range
0.222–0.235, with 0.2224 at H = 120).

### Their Table 3 covers every Reynolds number in your sweep

| Re | H | Cd | Cl′ | −Cpb | Lc | θs | St |
|---|---|---|---|---|---|---|---|
| 50 | 200 | 1.397 | 0.0399 | 0.511 | 2.46 | 124.0 | 0.1240 |
| 60 | 160 | 1.377 | 0.0871 | 0.560 | 2.11 | 123.0 | 0.1354 |
| 80 | 160 | 1.336 | 0.1616 | 0.640 | 1.67 | 120.3 | 0.1528 |
| 100 | 120 | 1.317 | 0.2224 | 0.707 | 1.42 | 118.0 | 0.1649 |
| 120 | 120 | 1.306 | 0.2776 | 0.764 | 1.25 | 116.4 | 0.1739 |
| 150 | 160 | 1.301 | 0.3529 | 0.840 | 1.06 | 114.7 | 0.1837 |
| 180 | 100 | 1.310 | 0.4251 | 0.921 | 0.93 | 113.5 | 0.1919 |
| 200 | 100 | 1.316 | 0.4678 | 0.966 | 0.86 | 112.9 | 0.1958 |

Validate the **whole sweep** against this, not just Re = 100. Comparing your
`runs/re_sweep/summary.json` against it:

| Re | your Cd | Qu Cd | Δ | your St | Qu St | Δ | Williamson corr. vs Qu |
|---|---|---|---|---|---|---|---|
| 60 | 1.4419 | 1.377 | **+4.7 %** | 0.1366 | 0.1354 | +0.9 % | +0.26 % |
| 80 | 1.3726 | 1.336 | **+2.7 %** | 0.1511 | 0.1528 | −1.1 % | +0.01 % |
| 100 | 1.3798 | 1.317 | **+4.8 %** | 0.1595 | 0.1649 | −3.3 % | −0.34 % |
| 120 | 1.4097 | 1.306 | **+7.9 %** | 0.1774 | 0.1739 | +2.0 % | −0.47 % |
| 150 | 1.3849 | 1.301 | **+6.5 %** | 0.1828 | 0.1837 | −0.5 % | −0.15 % |
| 180 | 1.3770 | 1.310 | **+5.1 %** | 0.1889 | 0.1919 | −1.6 % | +0.01 % |

Three things fall out, and together they are *better* than what the report
currently claims:

1. **Cd is systematically +2.7 % to +7.9 % high at every Reynolds number.** Not
   noise, not one bad dataset — a consistent bias. Your report's uncited
   +0.3 % agreement at Re = 100 was masking it.
2. **There is a clean, citable explanation.** Qu et al. report — via Kumar &
   Mittal (2006) `[U]` — that blockage effects are negligible only for
   **H > 100**, and via Posdziech & Grundmann (2007) `[U]` that *"the force
   coefficient and the base suction coefficient are strongly dependent on the
   resolution and even more on the size of the computational domain"* while St
   is not. Your H = 50 (2 % blockage) is below that threshold and below every
   domain in their tables (H = 100–200). Their lateral condition — free stream,
   u = 1, v = 0 — is your free-slip setup, so the comparison is apples to
   apples. **This turns your blockage finding from an observation into a
   quantitatively supported result**: Cd is confinement-biased by ~5 % at
   H = 50, St is not, exactly as the literature predicts.
3. **Your Strouhal numbers are good and your correlation is validated.** St
   agrees to within 3.3 % everywhere, and the Williamson correlation you
   implement agrees with Qu et al.'s independent high-resolution CFD to within
   **0.5 % at all six Reynolds numbers**. That is a strong independent check on
   `strouhal_estimate` and worth a sentence in the report.

Chase **Posdziech & Grundmann (2007)** `[U]`: Qu reports they published
high-order polynomial fits for Cd(Re), −Cpb(Re), St(Re) and lift amplitude in
the laminar shedding regime at H = 140 and H = 8000. That would be the ideal
single benchmark curve for your sweep. Get the real citation from Qu's
bibliography.

---

## 5. The Strouhal correlation source is resolved (probably)

`src/cfd/lbm_cylinder.py` implements `St = -3.3265/Re + 0.1816 + 1.6e-4·Re` and
credits "Williamson (1988)". That paper exists and fits `[V]`:

> Williamson, C. H. K. (1988), *Defining a universal and continuous
> Strouhal–Reynolds number relationship for the laminar vortex shedding of a
> circular cylinder*, Phys. Fluids **31**(10), 2742.

Title, year, venue and subject (the continuous parallel-shedding Strouhal curve
for laminar shedding) all match, and the code's own docstring already names it.
**Williamson & Brown (1998) is the wrong citation** — that fits a series in
1/√Re, a different functional form.

Still to close: the coefficients were not read from the paper. It is a
three-page Letter — open it, confirm the fit and its stated Reynolds range, and
then the citation is safe. The 0.5 % agreement with Qu et al. above is strong
independent evidence the implementation is correct regardless, and the report
now cites it on that basis.

**A separate resolution, from the recovered sweep metadata.** The acoustic
section's "separation from the resonance" turned out to be
`(St − St_a)/St` with St from the correlation — that definition reproduces the
printed 15 / 24 / 30 / 33–40 % exactly, and it is now stated in the report and
computed by `derived_metrics.py`. Converting to the ratio Hong et al. use gives
St/St_a = 1.18 and 1.32 for the two cases that locked, against their band
limit of 1.14. So the discrepancy in §6 is real but small for Re = 60 and
substantial for Re = 80; the report now names it and attributes it to lossless
walls, with an absorbing-boundary run listed as the way to settle it.

An earlier secondary source (a Stanford report, TF-062) attributes these exact
coefficients to "Williamson (1991)" `[U]`. Probably a slip; do not propagate it.

---

## 6. The acoustic section: now properly anchored, with one real tension **[partly superseded — see §0]**

Two references turn this from an anecdote into a result.

**Blevins (1985),** *JFM* **161**, 217–237 `[VR]` — the experimental precedent.
A **transverse** sound wave acting on a cylinder **in a duct**; the shedding
frequency can be shifted by sound applied above *or* below the natural
frequency; and — the detail worth quoting — *"This entrainment is produced by the
velocity induced by the sound wave rather than by the sound pressure."* His
introduction notes the resonant modes in such systems are transverse to both the
mean flow and the cylinder axis, which is exactly your `c_s/2H` mode. Caveat to
state: Re = 20 000–40 000, far above your range.

**Hong, Wang, Jing & Sun (2020),** *JFM* **884** A42 `[V]` — the mechanism and
the band. Bidirectional vortex–sound coupling, transverse standing β-mode,
with lock-in quantified: synchronous region 0.98 ≤ f_s/f_a ≤ 1.02, β-mode
dominant region out to f_s/f_a ≤ 1.14.

**The reframing.** You did not find a numerical artefact. Free-slip walls are
perfectly reflecting duct walls, LBM carries real sound, so you built a duct and
it resonated — a known physical lock-in, accidentally enabled by an
unphysically lossless boundary. That is a better story and it comes with a
citation trail.

**The tension you must address.** Your table reports locking at 15 % and 24 %
separation; Hong et al.'s outer bound is 14 %. At Re = 80 your wake sat at 0.113
against a correlation value of 0.1528, so it was pulled ~26 % *down* to reach a
resonance at 0.1155 — a very large entrainment. Check the definition of
"separation" against their frequency ratio f_s/f_a first; if the definitions
agree, then the wider band is itself a finding, plausibly because your walls are
lossless where a real duct is not. Either outcome is publishable. Leaving it
unreconciled is not.

Novelty: three rounds of searching found the general phenomena well documented —
spurious boundary reflection in LBM, non-reflecting and sponge-layer boundary
conditions as standard practice in LBM aeroacoustics — but **no paper describing
a domain acoustic mode capturing the shedding frequency itself in an LBM
cylinder simulation, nor the lattice-velocity-lowering fix** `[NF]`. Reasonable
confidence, not proof.

---

## 7. Reynolds decomposition: unclaimed in your specific form

Three search rounds `[NF]`. What exists is adjacent but different:

- **RANS-PINNs** — solving the Reynolds-averaged equations and inferring closure
  terms (e.g. arXiv:2107.10711 `[V]`, and turbulence-model-augmented
  mean-flow reconstruction). Different problem: they solve for the mean.
- **Mean-flow reconstruction from sparse data** — inferring time-averaged fields
  or unsteady RANS forcing from measurements (*Data-Centric Engineering* `[V]`).
  Different problem: data-driven inference, not a data-free forward solve.
- **Renn et al. (2023)** Reynolds-decompose before the FNO `[V]` — which is where
  your FNO section's decomposition comes from, and you already credit it.

Nothing found uses a mean/fluctuation split with a **differentiable fitted mean
network** as a *conditioning device inside the residual of a data-free unsteady
forward PINN*. Your two implementation observations are the substantive part and
both are correct and sharp:

- the momentum residual needs ∇u₀ and ∇²u₀, so bilinear interpolation of a grid
  is unusable (piecewise-constant first derivatives, identically zero second);
- the mean must be regressed on the *measured* mean, not obtained from a steady
  solve, because the time mean of a shedding flow does not satisfy the steady
  equations — it carries the divergence of the Reynolds stresses.

The second is the kind of point referees remember. Claim the implementation, not
the concept, and cite Renn et al. for the decomposition itself.

---

## 8. FNO residual normalisation: sits in a crowded area

Two search rounds `[NF]` for normalising a physics residual by **the residual
the training data itself leaves**. But loss-term balancing is a crowded field —
adaptive weighting, gradient pathologies, H⁻¹ discretisations to rebalance the
PDE loss — so the surrounding territory is well occupied.

Frame this as a *principled choice of the residual weight* (don't drive a
prediction below the physical consistency of the data it is trained on), not as
a new mechanism. It is a good idea and a one-paragraph contribution, not a
headline. Position the whole FNO section against **PINO** `[V]`, which already
adds physics residuals to neural operators; your increments are the
pressure-elimination argument for a velocity-only output and this weight choice.

---

## 9. Defensible novelty, re-ranked after reading the papers **[superseded — see §0]**

1. **Data-free vortex shedding via causal time-window marching, contradicting
   two independent published conclusions that data supervision is necessary.**
   Promoted to first on this pass. Load-bearing, current, and the mechanism is
   untested by both papers making the claim.
2. **The acoustic lock-in diagnosis**, now properly anchored in Blevins and
   Hong et al., with the band discrepancy as an open question worth resolving.
3. **Optimiser-step-matched attribution of the gain to causality, not compute** —
   complementary to Krishnapriyan et al.'s collocation-point control.
4. **Quantified confinement bias across the shedding range**: Cd high by 2.7–7.9 %
   at H = 50 while St is unaffected, consistent with the H > 100 threshold in
   the literature. Upgraded — this is now a supported result, not an aside.
5. **Generalisation across Re = 60–180** with the marching window scaled to each
   dataset's measured period, validated against Qu et al.'s Table 3.
6. **The "shedding retained" amplitude metric.** Note Shah also uses a wake
   amplitude ratio, so present this as the right metric rather than a new one —
   and keep your argument that field L₂ cannot distinguish a dead wake from a
   slightly wrong one, which your own `decompose` row proves (L₂ improves
   0.183 → 0.110 while the wake stays collapsed).
7. Pressure-free curl residual for a velocity-only operator — incremental.
8. Supervision-volume result — clean, not a headline.

---

## 10. What changed in the report — status (second pass, v4)

All of the following were applied on 2026-09-22. `experiments/verify_report.py`
is green: **170 numbers, 168 pass, 0 mismatch, 2 unverifiable.** Every change
below is a diff for you to read, not a substitute for reading it.

| # | Change | Status |
|---|---|---|
| 1 | Cd benchmark 1.37–1.38 → Qu et al. (1.318–1.336, reference 1.317); error +0.3 % → **+4.8 %**; reframed as confinement bias with the H > 100 threshold | done |
| 2 | Citation error: arXiv:2306.00230 is Chuang & Barba, not the FEM paper | done |
| 3 | Krishnapriyan et al. cited; marching prior art stated with the collocation-vs-optimiser-steps distinction | done |
| 4 | Chuang & Barba's case named as Re = 200, single 40-period window | done |
| 5 | Shah cited; the data-free contradiction is now the framing in §1 | done |
| 6 | Whole sweep scored against Qu et al. Table 3 (transcribed into `derived_metrics.py`) | done |
| 7 | Williamson (1988) cited for the correlation | done (see §5) |
| 8 | Barkley & Henderson cited for Re_c = 188.5 ± 1.0 | done |
| 9 | Acoustic section reframed around Blevins and Hong et al.; band discrepancy named | done (see §6) |
| 10 | Stale Limitations section rewritten | done |
| 11 | Spectral-bias contradiction softened to the precise version | done |

Fourteen transcription errors were corrected alongside these: the 80k control
(21.9→21.8, 23.8→23.7), the σ=8 Fourier L₂ (0.147→0.148), `w_ic`=100
(89.8→89.7, "no change"→0.046), the 2000-point row (99.5→99.4), the multiperiod
St (0.1622→0.1627), the one-period phase lag (2.9°→3.0°), the Re=120 and 180
L₂ columns (0.035→0.034, 0.027→0.026), the FNO improvement (20→21 %), the
parity range (30–45→24–44 %), the Cl′ range (0.54→0.56), and the regenerated
lattice velocity (0.0315→0.0277, read from the recovered metadata).

**Two numbers remain unverifiable** and only two: the locked St = 0.114 and
0.113 at Re = 60 and 80. The GPU server holds only the regenerated datasets;
the superseded u = 0.05 runs were overwritten in place. Re-running those two at
u = 0.05 would take minutes and would close the last gap.

---

## 11. Gaps worth filling for a journal version

1. **Hard-constrained window continuity.** Roy & Castonguay (2024) `[V]` enforce
   inter-window continuity exactly via a solution ansatz and test advection,
   Allen–Cahn, KdV and wave equations — **not Navier–Stokes, not shedding.**
   Your report already documents the handoff steps as a visible artefact.
   Applying their ansatz closes your own stated limitation and is well-scoped.
2. **Seed count.** Everything is n = 2 and `decompose` has seeds of 9.3 % and
   4.6 %. Three to five seeds with reported spread on the main ablation removes
   the easiest referee objection you have. Cheapest credibility available.
3. **Resolve the lock-in band discrepancy** (§6).
4. **A sponge or non-reflecting lateral boundary** as the principled alternative
   to lowering the lattice velocity. Absorbing-layer formulations exist in the
   LBM aeroacoustics literature `[U]`. Showing the anomaly vanishes with a
   sponge layer proves the mechanism instead of sidestepping it. Also worth a
   look: arXiv:2605.02185, "Lattice Boltzmann methodology for unconfined
   flows" `[U]`, which may bear directly on your confinement bias.
5. **A larger domain at H > 100** for one Reynolds number, to confirm the Cd
   bias is confinement and recover the literature value. This is the single
   experiment that would close the validation question properly.
6. **Longer horizon** — tens of periods, which you name as a limitation.
7. **Resolution series at low blockage** — your own limitation; the current
   series is at 5 % blockage where blockage masks resolution.

---

## 12. Venue **[framing superseded — see §0]**

With items 1–3 and 5 above: **Physics of Fluids**, **Computers & Fluids**, or
**Machine Learning: Science and Technology** (IOP — best fit for the methods
framing). *JCP* would want more numerical analysis. A Springer SciML book
chapter is lower-risk and suits the "three mechanisms, one works, here is the
controlled evidence" structure.

The data-free-versus-Shah-and-Chuang-&-Barba framing is a stronger hook than
the ablation framing, and it is timely — Shah is a month old. If you want the
priority claim to count, that argues for moving reasonably quickly.

All venue judgements `[U]` — opinion, not checked against scope statements.
Your supervisor's read is worth more than mine here.

---

## 13. Reading queue

Read in this order. The first four change what you write.

1. **Qu et al. (2013)**, *J. Fluids Struct.* 39, 347–370 — Tables 1 and 3. Fix
   your benchmark, then validate the whole sweep. Get the real citations for
   Posdziech & Grundmann (2007), Kumar & Mittal (2006) and Li et al. (2009) out
   of its bibliography.
2. **Shah**, arXiv:2608.19632 — confirm the data-supervision reading in §1 for
   yourself before you build a headline on it.
3. **Chuang & Barba**, arXiv:2306.00230 — §4 for the Re = 200 setup, §5 for the
   spectral-bias argument you are qualifying.
4. **Krishnapriyan et al.**, arXiv:2109.01050 — §5.2 and Figure 5 for the
   control distinction.
5. **Williamson (1988)**, Phys. Fluids 31(10), 2742 — three pages, one formula
   to confirm.
6. **Hong et al.**, *JFM* 884 A42 — lock-in mechanism and band; reconcile with
   your separation table.
7. **Blevins (1985)**, *JFM* 161, 217–237 — the experimental precedent.
8. **Roy & Castonguay**, arXiv:2403.03223 — the extension in §11.1.
9. **Wang, Wang & Perdikaris**, *CMAME* 384:113938 — the Fourier-feature
   citation to use in a PINN context.
10. **Li et al. (PINO)**, arXiv:2111.03794 — position the FNO section against it.
11. **Barkley & Henderson**, *JFM* 322, 215–241 — one number, the Re limit.
