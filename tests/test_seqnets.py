"""Guards for the sequence PINNs (PINNsFormer and its GRU analogue).

The PDE operator differentiates the network one point at a time, so a network
that let one query point see another -- attention over the batch instead of
over the point's own pseudo-sequence -- would train, and its residual would be
wrong. Both properties are checked here, plus agreement of the two derivative
backends on these architectures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upinn"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from pinn.ns2d_unsteady import NavierStokes2DUnsteadyPDE  # noqa: E402
from pinn.seqnets import SEQ_NETS, build_seqnet, pseudo_sequence  # noqa: E402


def _nets():
    for kind in SEQ_NETS:
        net = build_seqnet(kind, d_model=8, d_hidden=16, k=5, dt=0.05)
        params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
        yield kind, net, params


def test_pseudo_sequence_shifts_only_time():
    x = jnp.array([[0.1, 0.2, 0.3]])
    seq = pseudo_sequence(x, 4, 0.5)
    assert seq.shape == (1, 4, 3)
    np.testing.assert_allclose(seq[0, :, :2], np.tile([0.1, 0.2], (4, 1)))
    np.testing.assert_allclose(seq[0, :, 2], [0.3, 0.8, 1.3, 1.8])


def test_no_mixing_across_query_points():
    """Point 0's output must not change when the rest of the batch does."""
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(6, 3)))
    b = a.at[1:].set(jnp.asarray(rng.normal(size=(5, 3))))
    for kind, net, params in _nets():
        ya, yb = net.apply(params, a)[0], net.apply(params, b)[0]
        alone = net.apply(params, a[:1])[0]
        d = max(float(jnp.abs(ya - yb).max()), float(jnp.abs(ya - alone).max()))
        print(f"  {kind}: max change in point 0 = {d:.2e}")
        assert d < 1e-12, (kind, d)


def test_derivative_backends_agree():
    """Forward-over-forward JVPs vs jax.hessian, on the sequence nets."""
    xyt = jnp.asarray(np.random.default_rng(1).normal(size=(7, 3)))
    for kind, net, params in _nets():
        fast = NavierStokes2DUnsteadyPDE(net, Re=100.0, fast=True).residual(params, xyt)
        ref = NavierStokes2DUnsteadyPDE(net, Re=100.0, fast=False).residual(params, xyt)
        d = float(jnp.abs(fast - ref).max())
        print(f"  {kind}: max |fast - hessian| = {d:.2e}")
        assert d < 1e-10, (kind, d)


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
