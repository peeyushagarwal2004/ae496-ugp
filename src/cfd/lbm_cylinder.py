"""
D2Q9 lattice-Boltzmann solver for 2-D flow past a circular cylinder.

Generates the ground-truth vortex-shedding dataset for the PINN study.
Validation targets at Re = 100 (2-D, laminar, periodic shedding):

    St  = 0.165          (Strouhal number, from the C_l spectrum)
    C_d = 1.37 - 1.38    (mean drag coefficient)

Numerics
--------
* Two-relaxation-time (TRT) collision with Lambda = 3/16, which fixes the
  bounce-back wall exactly halfway between nodes independently of
  viscosity. Under plain BGK the effective cylinder diameter drifts with
  tau, which biases C_d. `--bgk` restores single-relaxation-time.
* Half-way bounce-back on the cylinder, with the momentum-exchange
  algorithm (MEA) for the surface force -- this is what makes C_d
  trustworthy enough to validate against the benchmark.
* Free-slip (specular) top/bottom walls, which enforce v = 0 exactly and
  therefore match the `symmetry` boundary used by underPINN's
  examples/cylinder/config.yaml.
* Equilibrium velocity inlet, zero-gradient outlet.
* Shedding is triggered by a brief transverse inlet perturbation; without
  it the symmetric (unstable) base flow persists for a very long time.

Output convention
-----------------
Snapshots are written in the SAME non-dimensional frame as underPINN's
cylinder example, so the PINN can consume them without rescaling:

    cylinder diameter D = 1, centred at the origin
    free stream U = 1
    x in [-5, 15], y in [-5, 5]
    t* = U t / D
    p  = (rho - 1) / 3  in units of rho_phys U^2

Usage
-----
    python src/cfd/lbm_cylinder.py --steps 2000 --tag smoke
    python src/cfd/lbm_cylinder.py                       # full production run
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import sys

import jax

# Precision must be chosen before any jnp array is created, so peek at argv.
# float32 is standard practice for LBM (rho stays O(1), the pressure signal
# rho-1 is O(1e-2), so ~1e-7 round-off is 5 orders below the signal).
USE_F64 = "--f64" in sys.argv
jax.config.update("jax_enable_x64", USE_F64)

import jax.numpy as jnp
import numpy as np

DTYPE = np.float64 if USE_F64 else np.float32

# --------------------------------------------------------------------------
# D2Q9 lattice
# --------------------------------------------------------------------------
# index:      0       1       2       3        4       5       6        7        8
# direction: rest    +x      +y      -x       -y     +x+y    -x+y     -x-y     +x-y
E = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=DTYPE)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])  # e_i -> -e_i
SPEC = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5])  # (ex, ey) -> (ex, -ey)

# Populations with a +y / -y component, needed for the free-slip walls.
UP = np.array([2, 5, 6])  # ey = +1
DOWN = np.array([4, 7, 8])  # ey = -1


def equilibrium(rho, ux, uy):
    """D2Q9 second-order equilibrium distribution."""
    eu = 3.0 * (E[:, 0][:, None, None] * ux + E[:, 1][:, None, None] * uy)
    usq = 1.5 * (ux * ux + uy * uy)
    return W[:, None, None] * rho * (1.0 + eu + 0.5 * eu * eu - usq)


def macroscopic(f):
    rho = jnp.sum(f, axis=0)
    ux = jnp.tensordot(E[:, 0], f, axes=(0, 0)) / rho
    uy = jnp.tensordot(E[:, 1], f, axes=(0, 0)) / rho
    return rho, ux, uy


# --------------------------------------------------------------------------
# One LBM time step
# --------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("win",))
def step(f, solid, links, tau_p, tau_m, u_in_x, u_in_y, win):
    """Advance the distribution `f` by one lattice time step.

    `links[i]` marks fluid nodes whose neighbour in direction i is solid.
    Returns the new populations and the momentum-exchange force (Fx, Fy).

    Collision is two-relaxation-time (TRT): the even (symmetric) part of the
    distribution relaxes at `tau_p`, which sets the viscosity, and the odd
    (antisymmetric) part at `tau_m`. Setting tau_m = tau_p recovers BGK.
    """
    rho, ux, uy = macroscopic(f)

    # --- collide (TRT) --------------------------------------------------
    # With BGK the wall implied by bounce-back drifts with viscosity unless
    # Lambda = (tau_p - 1/2)(tau_m - 1/2) = 3/16. TRT pins that product, so
    # the cylinder keeps its nominal diameter at any Reynolds number.
    feq = equilibrium(rho, ux, uy)
    f_o, feq_o = f[OPP], feq[OPP]
    d_sym = 0.5 * ((f + f_o) - (feq + feq_o))
    d_asym = 0.5 * ((f - f_o) - (feq - feq_o))
    fpost = f - d_sym / tau_p - d_asym / tau_m

    # Solid nodes carry no meaningful physics; keep them finite.
    fpost = jnp.where(solid[None, :, :], f, fpost)

    # --- momentum exchange on the cylinder ------------------------------
    # Half-way bounce-back => each blocked link transfers 2 * e_i * f_i^post.
    # Only a small window around the body has non-zero links, so reduce there
    # instead of over the whole grid.
    x0, x1, y0, y1 = win
    blocked = jnp.where(links[:, x0:x1, y0:y1], fpost[:, x0:x1, y0:y1], 0.0)
    Fx = 2.0 * jnp.sum(jnp.tensordot(E[:, 0], blocked, axes=(0, 0)))
    Fy = 2.0 * jnp.sum(jnp.tensordot(E[:, 1], blocked, axes=(0, 0)))

    # --- stream (periodic roll; boundaries fixed up below) --------------
    fs = jnp.stack(
        [jnp.roll(fpost[i], shift=(int(E[i, 0]), int(E[i, 1])), axis=(0, 1)) for i in range(9)]
    )

    # --- half-way bounce-back on the cylinder ---------------------------
    # A population that would have entered the solid returns to its origin
    # reversed, overwriting the garbage streamed out of the solid node.
    fs = fs.at[OPP].set(jnp.where(links, fpost, fs[OPP]))

    # --- free-slip (specular) top and bottom walls ----------------------
    # At y = 0 the unknown up-going populations are reflections of the local
    # down-going ones (and vice versa at y = ny-1), enforcing v = 0 without a
    # no-slip boundary layer. A diagonal population also advances one cell
    # along x while it reflects, so the reflected value is taken from the
    # neighbour upstream in x -- dropping that shift leaves a spurious
    # tangential drag on the wall.
    for i in (2, 5, 6):  # bottom wall: unknown up-going populations
        fs = fs.at[i, :, 0].set(jnp.roll(fpost[SPEC[i], :, 0], int(E[i, 0])))
    for i in (4, 7, 8):  # top wall: unknown down-going populations
        fs = fs.at[i, :, -1].set(jnp.roll(fpost[SPEC[i], :, -1], int(E[i, 0])))

    # --- inlet: equilibrium at (rho = 1, u = u_in) ----------------------
    nx, ny = solid.shape
    ones = jnp.ones((1, ny), dtype=f.dtype)
    f_in = equilibrium(ones, jnp.full((1, ny), u_in_x, dtype=f.dtype),
                       jnp.full((1, ny), u_in_y, dtype=f.dtype))
    fs = fs.at[:, 0, :].set(f_in[:, 0, :])

    # --- outlet: zero gradient ------------------------------------------
    fs = fs.at[:, -1, :].set(fs[:, -2, :])

    return fs, Fx, Fy


@partial(jax.jit, static_argnames=("n", "win"))
def run_chunk(f, solid, links, tau_p, tau_m, u_in_x, u_in_y, n, win):
    """Run `n` steps inside one XLA program (avoids Python-loop overhead)."""

    def body(carry, _):
        f, = carry
        f, Fx, Fy = step(f, solid, links, tau_p, tau_m, u_in_x, u_in_y, win)
        return (f,), jnp.array([Fx, Fy])

    (f,), forces = jax.lax.scan(body, (f,), None, length=n)
    return f, forces


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def build_geometry(nx, ny, cx, cy, radius):
    """Solid mask for the cylinder plus the per-direction link masks."""
    xs = np.arange(nx)[:, None]
    ys = np.arange(ny)[None, :]
    solid = ((xs - cx) ** 2 + (ys - cy) ** 2) <= radius**2

    fluid = ~solid
    links = np.zeros((9, nx, ny), dtype=bool)
    for i in range(1, 9):
        # neighbour_is_solid[x] == solid[x + e_i]
        neighbour_is_solid = np.roll(solid, shift=(-E[i, 0], -E[i, 1]), axis=(0, 1))
        links[i] = fluid & neighbour_is_solid
    return solid, links


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
def strouhal(cl, dt_star):
    """Shedding frequency from the lift signal, via mean zero-crossing spacing.

    Deliberately not an FFT peak: with only ~12 recorded shedding periods the
    FFT bin width is ~1/T ~ 0.014, which is larger than the discrepancy being
    measured, so peak-picking can flatter or condemn a result purely by where
    the bin lands. Zero-crossing timing resolves the period to a fraction of a
    time step and averages over every cycle in the record.
    """
    sig = np.asarray(cl, dtype=np.float64)
    sig = sig - sig.mean()
    if sig.size < 64:
        return float("nan")
    idx = np.where(np.diff(np.signbit(sig)))[0]
    if idx.size < 4:
        return float("nan")
    # Refine each crossing instant by linear interpolation between samples.
    t0 = idx + sig[idx] / (sig[idx] - sig[idx + 1])
    half_period = np.mean(np.diff(t0)) * dt_star
    return float(0.5 / half_period)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=100.0, help="Reynolds number U*D/nu")
    ap.add_argument("--diameter", type=int, default=24, help="cylinder diameter, lattice units")
    ap.add_argument("--u", type=float, default=0.08, help="inlet velocity, lattice units")
    ap.add_argument("--steps", type=int, default=80000, help="total time steps")
    ap.add_argument("--warmup-periods", type=float, default=30.0,
                    help="shedding periods to discard before recording")
    ap.add_argument("--snaps-per-period", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=200, help="steps fused per XLA call")
    ap.add_argument("--length", type=float, default=30.0, help="domain length, diameters")
    ap.add_argument("--height", type=float, default=20.0, help="domain height, diameters")
    ap.add_argument("--upstream", type=float, default=8.0,
                    help="inlet-to-cylinder distance, diameters")
    ap.add_argument("--magic", type=float, default=3.0 / 16.0,
                    help="TRT magic parameter Lambda; 3/16 puts the bounce-back "
                         "wall exactly halfway between nodes")
    ap.add_argument("--bgk", action="store_true",
                    help="use single-relaxation-time BGK instead of TRT")
    ap.add_argument("--crop", type=float, nargs=4,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    default=[-5.0, 15.0, -5.0, 5.0],
                    help="saved sub-domain in diameters (default: the underPINN "
                         "cylinder example domain)")
    ap.add_argument("--stride", type=int, default=2,
                    help="spatial stride for saved snapshots (2 -> ~0.08 D, "
                         "comparable with the PIV resolution in the reference paper)")
    ap.add_argument("--no-snapshots", action="store_true",
                    help="record forces only -- for cheap convergence sweeps")
    ap.add_argument("--f64", action="store_true",
                    help="run in float64 (~2x slower; for precision checks)")
    ap.add_argument("--tag", type=str, default="re100")
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    D = args.diameter
    U = args.u
    nu = U * D / args.re
    tau_p = 3.0 * nu + 0.5
    tau_m = tau_p if args.bgk else 0.5 + args.magic / (tau_p - 0.5)
    lam = (tau_p - 0.5) * (tau_m - 0.5)

    # A wide domain keeps blockage low so C_d is comparable with the
    # unbounded-flow benchmark; the PINN sub-domain is cropped from this later.
    nx, ny = int(args.length * D), int(args.height * D)
    cx = args.upstream * D
    cy = (ny - 1) / 2.0  # free-slip walls sit ON nodes 0 and ny-1

    if tau_p <= 0.505:
        raise SystemExit(f"tau_p = {tau_p:.4f} is too close to 1/2; lower --u or raise --diameter.")

    solid_np, links_np = build_geometry(nx, ny, cx, cy, D / 2.0)
    # Bounding box (with margin) enclosing every boundary link.
    pad = 3
    xs_i, ys_i = np.where(links_np.any(axis=0))
    win = (int(xs_i.min() - pad), int(xs_i.max() + pad + 1),
           int(ys_i.min() - pad), int(ys_i.max() + pad + 1))
    solid = jnp.asarray(solid_np)
    links = jnp.asarray(links_np)  # bool

    # Shedding period in lattice steps, from the expected Strouhal number.
    period = D / (0.165 * U)
    warmup = int(args.warmup_periods * period)
    snap_every = max(1, int(round(period / args.snaps_per_period)))

    scheme = "BGK" if args.bgk else "TRT"
    print(f"Re={args.re:g}  D={D}  U={U:g}  nu={nu:.5f}  Ma={U * 3 ** 0.5:.4f}")
    print(f"{scheme}: tau_p={tau_p:.5f}  tau_m={tau_m:.5f}  Lambda={lam:.5f}")
    print(f"grid {nx} x {ny}   cylinder centre ({cx:.1f}, {cy:.1f})   dtype={DTYPE().dtype}")
    print(f"blockage D/H = {1.0 / args.height:.3f}   force window {win}")
    print(f"expected shedding period ~{period:.0f} steps; warmup {warmup} steps")
    print(f"recording every {snap_every} steps after warmup")

    # --- initial condition: uniform free stream --------------------------
    rho0 = jnp.ones((nx, ny), dtype=DTYPE)
    ux0 = jnp.full((nx, ny), U, dtype=DTYPE)
    uy0 = jnp.zeros((nx, ny), dtype=DTYPE)
    f = equilibrium(rho0, ux0, uy0)

    # Perturbation window that breaks the wake symmetry and seeds shedding.
    pert_start, pert_end = int(0.5 * period), int(1.5 * period)
    pert_amp = 0.05 * U

    coef = 0.5 * U * U * D  # 0.5 * rho * U^2 * D, with rho = 1
    dt_star = U / D  # one lattice step in units of D/U

    # Physical (non-dimensional) node coordinates: D = 1, U = 1, origin at
    # the cylinder centre -- the frame underPINN's cylinder example uses.
    x_all = (np.arange(nx) - cx) / D
    y_all = (np.arange(ny) - cy) / D
    cxmin, cxmax, cymin, cymax = args.crop
    ix0, ix1 = int(np.searchsorted(x_all, cxmin)), int(np.searchsorted(x_all, cxmax)) + 1
    iy0, iy1 = int(np.searchsorted(y_all, cymin)), int(np.searchsorted(y_all, cymax)) + 1
    st_ = args.stride
    sx, sy = slice(ix0, ix1, st_), slice(iy0, iy1, st_)
    x_out, y_out = x_all[sx], y_all[sy]
    print(f"saving crop x[{x_out[0]:.2f},{x_out[-1]:.2f}] y[{y_out[0]:.2f},{y_out[-1]:.2f}]"
          f"  grid {x_out.size} x {y_out.size}  dx={st_ / D:.4f} D")

    forces, snaps, snap_steps = [], [], []
    mean_acc = np.zeros((3, x_out.size, y_out.size), dtype=np.float64)
    fluid_np = ~solid_np
    t0 = last_report = time.time()
    done = 0
    while done < args.steps:
        # Before warmup, run long fused chunks for speed. Once recording, the
        # chunk length IS the snapshot interval, so every chunk ends exactly on
        # a snapshot and the sampling rate is exact rather than chunk-aliased.
        if done < warmup:
            n = min(args.chunk, warmup - done, args.steps - done)
        elif args.no_snapshots:
            n = min(args.chunk, args.steps - done)
        else:
            n = min(snap_every, args.steps - done)

        uy_in = pert_amp if pert_start <= done < pert_end else 0.0
        f, F = run_chunk(f, solid, links, tau_p, tau_m, U, uy_in, n, win)
        forces.append(np.asarray(F))
        done += n

        if done >= warmup and n == snap_every and not args.no_snapshots:
            rho, ux, uy = macroscopic(f)
            field = np.stack(
                [
                    np.where(fluid_np, np.asarray(ux) / U, 0.0),
                    np.where(fluid_np, np.asarray(uy) / U, 0.0),
                    np.where(fluid_np, (np.asarray(rho) - 1.0) / 3.0 / (U * U), 0.0),
                ]
            )[:, sx, sy]
            snaps.append(field.astype(np.float32))
            mean_acc += field  # for the Reynolds decomposition u = u0 + u'
            snap_steps.append(done)
        if time.time() - last_report > 30.0:
            last_report = time.time()
            rho, ux, uy = macroscopic(f)
            rmin, rmax = float(jnp.min(rho)), float(jnp.max(rho))
            if not np.isfinite(rmin) or not np.isfinite(rmax):
                raise SystemExit(f"solver diverged at step {done}")
            rate = done / (time.time() - t0)
            print(
                f"  step {done:>7d}/{args.steps}  rho[{rmin:.4f},{rmax:.4f}]  "
                f"{rate:6.0f} steps/s  snaps={len(snaps)}",
                flush=True,
            )

    F = np.concatenate(forces, axis=0)
    cd, cl = F[:, 0] / coef, F[:, 1] / coef

    # Statistics over the recorded (post-transient) window only.
    tail = slice(warmup, None)
    st = strouhal(cl[tail], dt_star)
    cd_mean = float(np.mean(cd[tail]))
    cl_rms = float(np.sqrt(np.mean(cl[tail] ** 2)))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_star = np.asarray(snap_steps, dtype=np.float64) * dt_star
    mean_field = (mean_acc / max(len(snaps), 1)).astype(np.float32)

    npz = outdir / f"cylinder_{args.tag}.npz"
    np.savez_compressed(
        npz,
        snapshots=np.asarray(snaps),  # (n_t, 3, nx, ny) -> u, v, p
        mean_field=mean_field,  # (3, nx, ny) time-mean u0 over the recorded window
        x=x_out,
        y=y_out,
        t=t_star,
        solid=solid_np[sx, sy],
        cd=cd,
        cl=cl,
        dt_star=dt_star,
        Re=args.re,
    )

    meta = {
        "Re": args.re,
        "D_lattice": D,
        "U_lattice": U,
        "nu_lattice": nu,
        "tau_p": tau_p,
        "tau_m": tau_m,
        "Lambda": lam,
        "scheme": scheme,
        "Ma": U * 3 ** 0.5,
        "grid": [nx, ny],
        "steps": args.steps,
        "warmup_steps": warmup,
        "n_snapshots": len(snaps),
        "snapshot_grid": [int(x_out.size), int(y_out.size)],
        "dx_over_D": st_ / D,
        "crop": list(args.crop),
        "dt_star_per_step": dt_star,
        "snapshot_dt_star": snap_every * dt_star,
        "St": st,
        "Cd_mean": cd_mean,
        "Cl_rms": cl_rms,
        "St_target": 0.165,
        "Cd_target": [1.37, 1.38],
        "runtime_s": time.time() - t0,
        "dtype": str(DTYPE().dtype),
        "blockage": 1.0 / args.height,
    }
    (outdir / f"cylinder_{args.tag}_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n--- validation -------------------------------------------")
    print(f"  St      = {st:.4f}   (target 0.165)")
    print(f"  Cd_mean = {cd_mean:.4f}   (target 1.37-1.38)")
    print(f"  Cl_rms  = {cl_rms:.4f}")
    print(f"  snapshots: {len(snaps)}  ->  {npz}")


if __name__ == "__main__":
    main()
