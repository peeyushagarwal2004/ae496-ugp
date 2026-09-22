"""Validation for the unsteady 2-D Navier-Stokes residual.

Run with:  python tests/test_ns2d_unsteady.py

Besides asserting, each check RETURNS the value it measured, and running this
file as a script writes them to runs/operator_verification.json. The report
quotes these residuals; before, they were printed and then transcribed by hand,
so nothing could verify them and nothing regenerated them when the operator
changed. experiments/verify_report.py now sources them from that file.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upinn"))

from pinn.ns2d_unsteady import NavierStokes2DUnsteadyPDE, TaylorGreen  # noqa: E402

from underPINN.nn.mlp import MLP  # noqa: E402


def _points(n=256, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        np.stack(
            [
                rng.uniform(0.0, 2 * np.pi, n),
                rng.uniform(0.0, 2 * np.pi, n),
                rng.uniform(0.0, 2.0, n),
            ],
            axis=1,
        )
    )


def test_taylor_green_is_annihilated():
    """An exact NS solution must drive the residual to zero."""
    Re = 100.0
    xyt = _points()
    measured = {}
    for fast in (True, False):
        pde = NavierStokes2DUnsteadyPDE(TaylorGreen(Re), Re=Re, fast=fast)
        r = np.asarray(pde.residual({}, xyt))
        worst = np.abs(r).max()
        print(f"  fast={fast!s:5s}  max|residual| on Taylor-Green = {worst:.3e}")
        assert worst < 1e-10, f"Taylor-Green residual too large: {worst:.3e}"

        # Sanity: the analytic field really is divergence free.
        cont = np.abs(r[:, 0]).max()
        assert cont < 1e-12, f"continuity not satisfied: {cont:.3e}"
        measured["fast" if fast else "slow"] = float(worst)
        measured[("fast" if fast else "slow") + "_continuity"] = float(cont)
    # The report quotes the fast backend, which is the one the solvers use.
    measured["value"] = measured["fast"]
    return measured


def test_backends_agree_on_a_network():
    """The fast JVP backend must match the literal Hessian backend."""
    xyt = _points(128, seed=1)
    net = MLP([3, 32, 32, 3])
    params = net.init(jax.random.PRNGKey(0), xyt)

    fast = NavierStokes2DUnsteadyPDE(net, Re=100.0, fast=True).residual(params, xyt)
    slow = NavierStokes2DUnsteadyPDE(net, Re=100.0, fast=False).residual(params, xyt)

    diff = float(jnp.abs(fast - slow).max())
    scale = float(jnp.abs(slow).max())
    print(f"  backend max abs diff = {diff:.3e}  (residual scale {scale:.3e})")
    assert diff < 1e-9 * max(scale, 1.0), f"backends disagree by {diff:.3e}"
    return {"value": diff, "residual_scale": scale}


def test_steady_limit_matches_shipped_operator():
    """With a time-independent network, our residual must reduce to underPINN's
    steady NavierStokesPDE (which is written in conservative form -- equivalent
    to ours only because the field is divergence free, so this also checks that
    the continuity constraint is consistent between the two)."""
    from underPINN.pde.navier_stokes import NavierStokesPDE

    Re = 100.0
    xy = _points(128, seed=2)[:, :2]
    xyt = jnp.concatenate([xy, jnp.zeros((xy.shape[0], 1))], axis=1)

    class SteadyTG:
        """Taylor-Green frozen at t = 0: steady in time, divergence free."""

        def apply(self, params, z):  # noqa: ARG002
            x, y = z[:, 0], z[:, 1]
            u = -jnp.cos(x) * jnp.sin(y)
            v = jnp.sin(x) * jnp.cos(y)
            p = -0.25 * (jnp.cos(2 * x) + jnp.cos(2 * y))
            return jnp.stack([u, v, p], axis=1)

    field = SteadyTG()
    ours = np.asarray(NavierStokes2DUnsteadyPDE(field, Re=Re).residual({}, xyt))
    theirs = np.asarray(NavierStokesPDE(field, Re=Re).residual({}, xy))

    diff = np.abs(ours - theirs).max()
    print(f"  steady-limit max abs diff vs shipped NavierStokesPDE = {diff:.3e}")
    assert diff < 1e-9, f"steady limit disagrees with underPINN by {diff:.3e}"
    return {"value": float(diff)}


def benchmark_backends(n=4096, repeats=20):
    """Wall-clock speed-up of the fast JVP backend over the literal Hessian.

    Not a test -- it asserts nothing, because a timing ratio depends on the
    machine. It exists so the speed-up quoted in the report is regenerated
    rather than remembered, and so the number carries the hardware it was
    measured on. Both paths are jitted and warmed up first, otherwise the
    comparison is dominated by compilation.
    """
    xyt = _points(n, seed=3)
    net = MLP([3, 128, 128, 128, 3])
    params = net.init(jax.random.PRNGKey(0), xyt)

    timings = {}
    for fast in (True, False):
        pde = NavierStokes2DUnsteadyPDE(net, Re=100.0, fast=fast)
        fn = jax.jit(lambda p, z, _pde=pde: _pde.residual(p, z))
        fn(params, xyt).block_until_ready()          # compile
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(params, xyt).block_until_ready()
            best = min(best, time.perf_counter() - t0)
        timings["fast" if fast else "slow"] = best

    speedup = timings["slow"] / timings["fast"]
    print(f"  fast {timings['fast']*1e3:.2f} ms   "
          f"slow {timings['slow']*1e3:.2f} ms   speed-up {speedup:.2f}x")
    return {
        "value": speedup,
        "fast_seconds": timings["fast"],
        "slow_seconds": timings["slow"],
        "n_points": n,
        "repeats": repeats,
        "network": "MLP([3, 128, 128, 128, 3])",
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "_caveat": "Best-of-N wall clock on whatever machine ran this. A "
                   "timing ratio is hardware dependent; quote it with the "
                   "backend recorded here.",
    }


if __name__ == "__main__":
    failures = 0
    recorded = {}
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n{name}")
            try:
                got = fn()
                if isinstance(got, dict):
                    recorded[name] = got
                print("  PASS")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL: {exc}")

    print("\nbenchmark_backends")
    try:
        recorded["benchmark_backends"] = benchmark_backends()
    except Exception as exc:                # noqa: BLE001 - benchmark is optional
        print(f"  SKIPPED: {type(exc).__name__}: {exc}")

    if recorded:
        out = ROOT / "runs" / "operator_verification.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(recorded, indent=2) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}")

    print("\nall tests passed" if not failures else f"\n{failures} test(s) failed")
    sys.exit(1 if failures else 0)
