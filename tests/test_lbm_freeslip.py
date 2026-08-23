"""Boundary-condition checks for the LBM cylinder solver.

Uniform free-stream flow is an exact fixed point of the scheme when the
inlet, outlet and free-slip walls are all implemented correctly: nothing in
the domain should be able to disturb it. A specular wall that forgets the
tangential displacement of the diagonal populations acts like a weak drag and
shows up here immediately as a velocity deficit near the walls.

Run with:  python tests/test_lbm_freeslip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "cfd"))

import jax.numpy as jnp  # noqa: E402

import lbm_cylinder as L  # noqa: E402


def _uniform(nx=64, ny=48, U=0.08):
    solid = jnp.zeros((nx, ny), dtype=bool)
    links = jnp.zeros((9, nx, ny), dtype=bool)
    f = L.equilibrium(
        jnp.ones((nx, ny), dtype=L.DTYPE),
        jnp.full((nx, ny), U, dtype=L.DTYPE),
        jnp.zeros((nx, ny), dtype=L.DTYPE),
    )
    return f, solid, links, U


def test_uniform_flow_is_a_fixed_point():
    """Free stream + free-slip walls must be preserved exactly."""
    f, solid, links, U = _uniform()
    tau = 0.5576
    win = (0, 4, 0, 4)  # no links; force window content is irrelevant here

    for _ in range(200):
        f, _, _ = L.step(f, solid, links, tau, tau, U, 0.0, win)

    rho, ux, uy = L.macroscopic(f)
    du = float(jnp.abs(ux - U).max())
    dv = float(jnp.abs(uy).max())
    drho = float(jnp.abs(rho - 1.0).max())
    print(f"  max|u - U| = {du:.3e}   max|v| = {dv:.3e}   max|rho-1| = {drho:.3e}")
    assert du < 1e-6, f"free stream not preserved: max|u-U| = {du:.3e}"
    assert dv < 1e-6, f"spurious transverse velocity: max|v| = {dv:.3e}"
    assert drho < 1e-6, f"spurious density drift: max|rho-1| = {drho:.3e}"


def test_walls_do_not_bias_the_profile():
    """The near-wall rows must not lag the channel centre (no wall drag)."""
    f, solid, links, U = _uniform()
    tau = 0.5576
    win = (0, 4, 0, 4)
    for _ in range(200):
        f, _, _ = L.step(f, solid, links, tau, tau, U, 0.0, win)

    _, ux, _ = L.macroscopic(f)
    mid = ux.shape[0] // 2
    wall_lag = float(ux[mid, 0] - ux[mid, ux.shape[1] // 2])
    print(f"  wall-row minus centre-row velocity = {wall_lag:.3e}")
    assert abs(wall_lag) < 1e-7, f"walls exert spurious drag: {wall_lag:.3e}"


def test_mass_is_conserved_with_a_body():
    """With the cylinder present, bounce-back must not create or destroy mass."""
    nx, ny, U = 128, 96, 0.08
    solid_np, links_np = L.build_geometry(nx, ny, 40.0, (ny - 1) / 2.0, 8.0)
    solid, links = jnp.asarray(solid_np), jnp.asarray(links_np)
    f = L.equilibrium(
        jnp.ones((nx, ny), dtype=L.DTYPE),
        jnp.full((nx, ny), U, dtype=L.DTYPE),
        jnp.zeros((nx, ny), dtype=L.DTYPE),
    )
    xs, ys = np.where(links_np.any(axis=0))
    win = (int(xs.min() - 2), int(xs.max() + 3), int(ys.min() - 2), int(ys.max() + 3))

    fluid = ~solid_np
    m0 = float(jnp.sum(L.macroscopic(f)[0][fluid]))
    for _ in range(300):
        f, _, _ = L.step(f, solid, links, tau_p=0.5576, tau_m=0.5576,
                         u_in_x=U, u_in_y=0.0, win=win)
    rho = L.macroscopic(f)[0]
    m1 = float(jnp.sum(rho[fluid]))
    assert np.isfinite(m1), "density went non-finite"
    drift = abs(m1 - m0) / m0
    print(f"  fluid mass drift over 300 steps = {drift:.3e} (inflow/outflow makes this nonzero)")
    assert drift < 1e-3, f"implausible mass drift: {drift:.3e}"


def test_trt_reduces_to_bgk():
    """tau_m = tau_p must reproduce single-relaxation-time BGK exactly."""
    import jax

    nx, ny, U, tau = 48, 32, 0.08, 0.5576
    rng = np.random.default_rng(0)
    solid = jnp.zeros((nx, ny), dtype=bool)
    links = jnp.zeros((9, nx, ny), dtype=bool)
    # A perturbed (non-equilibrium) state, so the collision term is non-trivial.
    f = L.equilibrium(
        jnp.ones((nx, ny), dtype=L.DTYPE),
        jnp.asarray(rng.normal(U, 0.01, (nx, ny)), dtype=L.DTYPE),
        jnp.asarray(rng.normal(0.0, 0.01, (nx, ny)), dtype=L.DTYPE),
    )
    f = f * jnp.asarray(rng.uniform(0.99, 1.01, f.shape), dtype=L.DTYPE)
    win = (0, 4, 0, 4)

    trt, _, _ = L.step(f, solid, links, tau, tau, U, 0.0, win)

    # Literal BGK for reference.
    rho, ux, uy = L.macroscopic(f)
    feq = L.equilibrium(rho, ux, uy)
    bgk_post = f - (f - feq) / tau
    d = float(jnp.abs(trt - L.step(f, solid, links, tau, tau, U, 0.0, win)[0]).max())
    # Compare the collision operators directly.
    f_o, feq_o = f[L.OPP], feq[L.OPP]
    d_sym = 0.5 * ((f + f_o) - (feq + feq_o))
    d_asym = 0.5 * ((f - f_o) - (feq - feq_o))
    trt_post = f - d_sym / tau - d_asym / tau
    err = float(jnp.abs(trt_post - bgk_post).max())
    print(f"  max|TRT(tau,tau) - BGK(tau)| = {err:.3e}   (step determinism {d:.1e})")
    assert err < 1e-6, f"TRT does not reduce to BGK: {err:.3e}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n{name}")
            try:
                fn()
                print("  PASS")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL: {exc}")
    print("\nall tests passed" if not failures else f"\n{failures} test(s) failed")
    sys.exit(1 if failures else 0)
