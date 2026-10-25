# Predicting Vortex Shedding Behind a Circular Cylinder with Physics-Informed Neural Networks

**AE496 Undergraduate Project, Department of Aerospace Engineering, IIT Kanpur**
Peeyush Agarwal

Built on [underPINN](https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN)
(JAX/Flax). Reference paper: Renn et al., *Forecasting subcritical cylinder
wakes with Fourier neural operators*,
[arXiv:2301.08290](https://arxiv.org/abs/2301.08290).

## Submission documents

| Document | Description |
|---|---|
| [`report/paper.pdf`](report/paper.pdf) | Conference paper, IEEE format, 4 pages |
| [`report/paper_long.pdf`](report/paper_long.pdf) | Extended paper, 9 pages |
| [`report/report.pdf`](report/report.pdf) | Full technical report with every result and control, 19 pages |
| [`report/brief.pdf`](report/brief.pdf) | Two-page summary for the supervisor |

## Summary

A physics-informed neural network (PINN) trained without data on the unsteady
cylinder wake tends toward the steady symmetric solution, which satisfies the
Navier-Stokes equations exactly. This project reproduces that failure at
Re = 100 on validated lattice Boltzmann reference data and measures which
training strategy prevents it.

| Result | Value |
|---|---|
| Reference data vs Qu et al. (2013) | St within 0.04 %, C_d 3.5 % high |
| Plain PINN, wake amplitude retained (4 start phases x 5 seeds) | 31.1 % |
| Causal time-marching | 56.4 % (+7.7 points over an equal-budget baseline) |
| Marching + Fourier features + Reynolds decomposition | 72.9 % (70-77 % at every start phase), L2 error 0.031 |
| Prediction from physics alone | about one shedding period; over five, 10.4 % slow and half a cycle out of phase |
| Supervised points per period needed to hold the phase (reconstruction) | 350 |
| Reynolds-number range tested | 60-180 |
| FNO forecast error at 10 steps, validated data | 0.0010 |

Physics added to the operator forecast helps the FNO only at short horizons on
the harder, earlier dataset, and does not help a U-Net or a convolutional GRU.

## Repository layout

```
src/cfd/        lattice Boltzmann reference solver (D2Q9, TRT) and validation
src/pinn/       unsteady Navier-Stokes operator, PINN trainer, sequence networks
src/fno/        FNO, U-Net and ConvGRU forecasting with physics residuals
experiments/    ablation, phase averaging, Re sweep, plotting, report checks
tests/          correctness tests for the operator, solver and networks
runs/           results of every run (JSON), the source of all reported numbers
figures/        figures used in the report and papers
data/           dataset metadata (the .npz fields are regenerated, not tracked)
report/         LaTeX sources and PDFs, references, number manifest
scripts/gpu/    run queues used on the IIT Kanpur A100 server
logs/           server logs referenced by the analysis
docs/           working notes, literature survey, Colab and Kaggle setup
```

## Reproducing

Requirements: Python 3.10+, JAX with CUDA, Flax, NumPy, Matplotlib, and
underPINN cloned beside this repository (see `docs/COLAB.md`).

Correctness tests (fast, CPU):

```bash
python tests/test_ns2d_unsteady.py
python tests/test_lbm_freeslip.py
```

The full-budget runs were made on an A100 using the scripts in `scripts/gpu/`;
each script states its purpose in its header comment. The main ones are:

| Script | Produces |
|---|---|
| `queue15.sh` | production dataset re100_v5, phase-0 ablation, Re sweep, FNO runs |
| `queue17.sh`, `queue18.sh`, `queue23.sh` | phase-averaged comparison and equal-budget controls, five seeds |
| `queue19.sh`, `queue20.sh` | sequence-PINN comparison and U-Net / ConvGRU operator backbones |
| `queue21.sh`, `queue22.sh` | sparse re-anchoring of the five-period chain |

## Verifying the reported numbers

Every number in `report/report.tex` is registered in `report/numbers.json`
against the run file that produces it.

```bash
python experiments/verify_report.py --strict
python experiments/check_paper_numbers.py
```

The first recomputes all 552 numbers in the report from `runs/` and `data/`.
The second confirms that every number in the two papers also appears in the
report. Build any document with `cd report && pdflatex <name>.tex` (run twice).
