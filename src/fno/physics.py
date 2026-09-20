"""Grid physics residuals for the operator forecaster (Phase 5).

The PINN side of this project gets its derivatives from automatic
differentiation of a continuous network (`src/pinn/ns2d_unsteady.py`). An FNO
does not represent a continuous field -- it maps one grid to another -- so the
physics has to be imposed with finite differences on that grid instead.

Two residuals are available, and neither needs the pressure:

* **continuity**, `u_x + v_y = 0`. Cheap, and the one the write-up names as the
  minimum viable physics term.
* **vorticity transport**, `w_t + u w_x + v w_y - (1/Re)(w_xx + w_yy) = 0`,
  obtained by taking the curl of the momentum equations. The pressure drops out
  of the curl, which is exactly why this form is used: the network never
  predicts `p`, so any residual containing `grad p` would be unusable.

Both are evaluated on the **full** velocity `u = u0 + u'`, not on the
fluctuation alone. The equations govern the total field; the decomposition is a
representation choice, and `u'` alone satisfies neither residual.

Second derivatives on a finite-difference grid amplify noise, so every residual
is evaluated on a **weight mask** that excludes the domain edges and a halo
around the cylinder, where the stencils would otherwise reach across a boundary
(see `interior_mask`).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def interior_mask(solid: np.ndarray, halo: int = 2, edge: int = 2) -> np.ndarray:
    """Points where a centred second-derivative stencil is safe to evaluate.

    Drops `edge` cells at the domain boundary and dilates the solid by `halo`
    cells, so no stencil straddles the cylinder wall. Returned as float32 so it
    can weight a loss directly.
    """
    keep = ~solid.astype(bool)
    for _ in range(halo):
        blocked = ~keep
        grown = blocked.copy()
        grown[1:, :] |= blocked[:-1, :]
        grown[:-1, :] |= blocked[1:, :]
        grown[:, 1:] |= blocked[:, :-1]
        grown[:, :-1] |= blocked[:, 1:]
        keep = ~grown
    if edge:
        keep[:edge, :] = keep[-edge:, :] = False
        keep[:, :edge] = keep[:, -edge:] = False
    return keep.astype(np.float32)


def _d_dx(f, dx):
    """Second-order central difference along axis -2 (x)."""
    return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2.0 * dx)


def _d_dy(f, dy):
    """Second-order central difference along axis -1 (y)."""
    return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dy)


def _laplacian(f, dx, dy):
    fxx = (jnp.roll(f, -1, axis=-2) - 2.0 * f + jnp.roll(f, 1, axis=-2)) / dx**2
    fyy = (jnp.roll(f, -1, axis=-1) - 2.0 * f + jnp.roll(f, 1, axis=-1)) / dy**2
    return fxx + fyy


# `jnp.roll` wraps around, so the two outermost rings hold garbage. That is
# harmless because `interior_mask` zeroes them, and wrapping keeps every
# operation shape-stable, which matters inside a scanned rollout.


def divergence(uv, dx, dy):
    """`u_x + v_y` for a field packed as (..., nx, ny, 2)."""
    u, v = uv[..., 0], uv[..., 1]
    return _d_dx(u, dx) + _d_dy(v, dy)


def vorticity(uv, dx, dy):
    """`w = v_x - u_y` for a field packed as (..., nx, ny, 2)."""
    u, v = uv[..., 0], uv[..., 1]
    return _d_dx(v, dx) - _d_dy(u, dy)


def vorticity_transport_residual(uv_prev, uv_now, uv_next, dx, dy, dt, Re):
    """Residual of `w_t + u.grad(w) - (1/Re) lap(w)` at the middle time level.

    The time derivative is the centred difference over `uv_prev` and `uv_next`,
    so a one-step prediction is constrained against the two frames that
    produced it. Advection and diffusion are evaluated at `uv_now`, which is
    known input rather than network output -- that keeps the residual a
    constraint on the prediction instead of a self-consistency condition the
    network could satisfy by drifting.
    """
    w_prev = vorticity(uv_prev, dx, dy)
    w_now = vorticity(uv_now, dx, dy)
    w_next = vorticity(uv_next, dx, dy)

    w_t = (w_next - w_prev) / (2.0 * dt)
    u, v = uv_now[..., 0], uv_now[..., 1]
    adv = u * _d_dx(w_now, dx) + v * _d_dy(w_now, dy)
    visc = _laplacian(w_now, dx, dy) / Re
    return w_t + adv - visc


def masked_mean_square(r, mask):
    """Mean of `r**2` over the masked points (mask broadcasts over leading axes)."""
    w = mask
    return jnp.sum((r**2) * w) / jnp.maximum(jnp.sum(w) * (r.size / mask.size), 1.0)
