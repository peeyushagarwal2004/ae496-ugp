"""Guards for the Phase 5 operator code.

The failure modes here are all silent. A finite-difference residual with the
wrong spacing still trains; a rollout that quietly re-feeds the ground truth
instead of its own prediction produces a beautiful error curve that means
nothing; and a train/test split that overlaps in time turns forecasting into
recall. Each of those is checked against a case with a known answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upinn"))

import jax  # noqa: E402

# The residuals are checked against exact solutions, so the test must not be
# limited by single precision -- training still runs in float32.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from fno import physics as phys  # noqa: E402
from fno.cylinder_fno import (  # noqa: E402
    make_rollout_fn, relative_l2, rollout_starts, split_indices,
)


def _taylor_green(nx=64, ny=64, t=0.0, Re=100.0, L=2 * np.pi):
    """An exact unsteady NS solution: divergence free, and a known vorticity.

    u = -cos x sin y e^{-2t/Re},  v = sin x cos y e^{-2t/Re},  w = -2 u_0-style
    decay. Used because both residuals must annihilate it, so a wrong spacing
    or a transposed axis shows up immediately.
    """
    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    decay = np.exp(-2.0 * t / Re)
    u = -np.cos(X) * np.sin(Y) * decay
    v = np.sin(X) * np.cos(Y) * decay
    return (np.stack([u, v], axis=-1).astype(np.float64),
            float(x[1] - x[0]), float(y[1] - y[0]))


def test_divergence_is_zero_on_an_incompressible_field():
    uv, dx, dy = _taylor_green()
    div = np.asarray(phys.divergence(jnp.asarray(uv), dx, dy))
    interior = np.ones(uv.shape[:2], np.float32)  # periodic: every point is valid
    worst = np.abs(div).max()
    print(f"  max |div| on Taylor-Green = {worst:.3e}")
    assert worst < 1e-12, worst
    assert float(phys.masked_mean_square(jnp.asarray(div), jnp.asarray(interior))) < 1e-20


def test_divergence_is_exact_on_a_known_compressible_field():
    """u = x, v = 2y has divergence 3 everywhere -- a spacing bug changes it."""
    nx, ny = 64, 48
    x = np.linspace(0.0, 3.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    uv = np.stack([X, 2.0 * Y], axis=-1)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    div = np.asarray(phys.divergence(jnp.asarray(uv), dx, dy))
    inner = div[2:-2, 2:-2]  # jnp.roll wraps, so skip the ring it corrupts
    print(f"  divergence of (x, 2y): got {inner.mean():.6f}, want 3.000000")
    assert np.allclose(inner, 3.0, atol=1e-9), inner.mean()


def _tg_residual(nx, Re=100.0, dt=1e-3):
    prev, dx, dy = _taylor_green(nx=nx, ny=nx, t=-dt, Re=Re)
    now, _, _ = _taylor_green(nx=nx, ny=nx, t=0.0, Re=Re)
    nxt, _, _ = _taylor_green(nx=nx, ny=nx, t=dt, Re=Re)
    r = np.asarray(phys.vorticity_transport_residual(
        jnp.asarray(prev), jnp.asarray(now), jnp.asarray(nxt), dx, dy, dt, Re))
    scale = np.abs(np.asarray(phys.vorticity(jnp.asarray(now), dx, dy))).max()
    return float(np.abs(r).max() / scale)


def test_vorticity_transport_annihilates_taylor_green():
    """Residual must vanish on an exact solution -- at the discretisation order.

    Central differences leave an O(dx^2) truncation error, so the residual is
    small but not zero. Halving the spacing must cut it by ~4; a wrong sign,
    spacing or axis gives an O(1) residual that does not converge at all, which
    a single fixed tolerance would not distinguish from a merely coarse grid.
    """
    coarse, fine = _tg_residual(32), _tg_residual(64)
    ratio = coarse / fine
    print(f"  residual/|w|: {coarse:.3e} (nx=32) -> {fine:.3e} (nx=64), "
          f"ratio {ratio:.2f} (want ~4)")
    assert fine < 1e-2, fine
    assert 3.0 < ratio < 5.0, ratio


def test_interior_mask_excludes_the_wall_and_the_edges():
    solid = np.zeros((40, 30), bool)
    solid[18:22, 13:17] = True
    mask = phys.interior_mask(solid, halo=2, edge=2)
    assert mask[19, 15] == 0.0, "solid itself must be excluded"
    assert mask[17, 15] == 0.0 and mask[16, 15] == 0.0, "halo must be excluded"
    assert mask[0, :].sum() == 0.0 and mask[:, -1].sum() == 0.0, "edges must be excluded"
    assert mask[10, 8] == 1.0, "far field must survive"
    print(f"  mask keeps {int(mask.sum())} of {mask.size} points")


def test_rollout_feeds_its_own_prediction_forward():
    """A rollout that secretly re-used its input would pass a one-step test.

    With a step that returns `now + 1`, the n-th prediction must be `now + n`.
    """
    def step(params, prev, now):
        return now + 1.0

    rollout = make_rollout_fn(step, remat=False)
    prev = jnp.zeros((2, 4, 4, 2))
    now = jnp.zeros((2, 4, 4, 2))
    preds = rollout({}, prev, now, 5)
    got = [float(preds[i].mean()) for i in range(5)]
    print(f"  recursive means: {got}  (want 1..5)")
    assert got == [1.0, 2.0, 3.0, 4.0, 5.0], got


def test_rollout_matches_hand_composition():
    """The scanned rollout must equal applying the step function by hand."""
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (2, 2)) * 0.1

    def step(params, prev, now):
        return jnp.tanh(now @ w) + 0.1 * prev

    rollout = make_rollout_fn(step, remat=True)
    prev = jax.random.normal(jax.random.PRNGKey(1), (3, 5, 5, 2))
    now = jax.random.normal(jax.random.PRNGKey(2), (3, 5, 5, 2))

    manual, p, c = [], prev, now
    for _ in range(4):
        nxt = step({}, p, c)
        manual.append(nxt)
        p, c = c, nxt
    manual = jnp.stack(manual)

    got = rollout({}, prev, now, 4)
    worst = float(jnp.abs(got - manual).max())
    print(f"  max |scan - hand-composed| = {worst:.3e}")
    assert worst < 1e-6, worst


def test_relative_l2_agrees_with_the_project_metric():
    """Must match `CylinderDataset.relative_l2`, which the PINN side is scored with."""
    from pinn.data import CylinderDataset

    rng = np.random.default_rng(0)
    true = rng.normal(size=(1, 8, 6, 2))
    pred = true * 1.10
    got = float(relative_l2(jnp.asarray(pred), jnp.asarray(true), (-3, -2, -1))[0])
    want = CylinderDataset.relative_l2(pred.reshape(-1, 2), true.reshape(-1, 2))
    print(f"  fno {got:.6f} vs data.py {want:.6f}")
    assert abs(got - want) < 1e-6, (got, want)
    assert abs(got - 0.10) < 1e-6, got


def test_split_is_contiguous_and_does_not_leak():
    train, test = split_indices(100, 0.7)
    assert train[-1] < test[0], "test must lie strictly after train in time"
    assert len(train) == 70 and len(test) == 30

    # Every training sample, inputs and targets alike, must stay in train.
    starts = rollout_starts(train, rollout=10)
    assert starts[0] - 1 >= train[0]
    assert starts[-1] + 10 <= train[-1]
    print(f"  {len(starts)} training samples, last target frame "
          f"{starts[-1] + 10} <= {train[-1]}")


def test_wake_frames_crops_to_even_and_blanks_the_cylinder():
    """Real data, if present: the grid must be even (SpectralConv2d asserts it)."""
    try:
        from pinn.data import load

        from fno.cylinder_fno import WakeFrames
        ds = load("re100_v4")
    except (FileNotFoundError, ImportError):
        print("  skipped (no dataset present)")
        return

    frames = WakeFrames(ds, downsample=2)
    nx, ny = frames.grid
    print(f"  grid {nx} x {ny}, dt* {frames.dt:.4f}")
    assert nx % 2 == 0 and ny % 2 == 0, (nx, ny)
    assert np.abs(frames.fluct[:, frames.solid]).max() == 0.0, "cylinder must be blank"
    assert frames.interior[frames.solid].sum() == 0.0


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
