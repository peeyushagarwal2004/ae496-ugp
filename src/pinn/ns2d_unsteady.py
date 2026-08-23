"""2-D unsteady incompressible Navier-Stokes residual for underPINN.

underPINN ships `NavierStokesPDE`, but it is steady: the network maps
(x, y) -> (u, v, p) and the residual has no time derivative, so it cannot
represent vortex shedding at all. This module adds the unsteady operator.

    network:  (x, y, t) -> (u, v, p)

    continuity   u_x + v_y                                            = 0
    x-momentum   u_t + u u_x + v u_y + p_x - (1/Re)(u_xx + u_yy)      = 0
    y-momentum   v_t + u v_x + v v_y + p_y - (1/Re)(v_xx + v_yy)      = 0

Non-dimensionalised on the cylinder diameter D and free stream U, matching
the convention used by the LBM ground-truth generator in src/cfd.

Coordinate packing follows underPINN's `BasePDE` contract: a single
(N, 3) array with xyt[:, 0:2] = (x, y) and xyt[:, 2] = t.

Two derivative backends are provided:

* ``fast=True``  (default) -- forward-over-forward JVPs that compute only the
  six derivative groups the residual actually needs.
* ``fast=False`` -- a literal `jax.hessian` implementation, kept because it is
  transparently correct. `tests/test_ns2d_unsteady.py` asserts the two agree
  and that both annihilate the Taylor-Green vortex.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from underPINN.core.base import BasePDE

_EX = jnp.array([1.0, 0.0, 0.0])
_EY = jnp.array([0.0, 1.0, 0.0])
_ET = jnp.array([0.0, 0.0, 1.0])


class NavierStokes2DUnsteadyPDE(BasePDE):
    """Unsteady incompressible Navier-Stokes in primitive variables.

    Parameters
    ----------
    model : Flax module mapping (N, 3) -> (N, 3), i.e. (x, y, t) -> (u, v, p).
    Re    : Reynolds number U D / nu.
    fast  : use the JVP derivative backend (see module docstring).
    """

    def __init__(self, model, Re: float = 100.0, fast: bool = True):
        self.model = model
        self.Re = Re
        self.fast = fast

    # -- forward evaluation -------------------------------------------------
    def u(self, params, xyt):
        """Evaluate (u, v, p) at packed coordinates xyt of shape (N, 3)."""
        return self.model.apply(params, xyt)

    def _point_fn(self, params):
        """Single-point view of the network: (3,) -> (3,)."""

        def f(z):
            return self.model.apply(params, z[None, :])[0]

        return f

    # -- derivative backends ------------------------------------------------
    def _derivs_fast(self, params, xyt):
        """Forward-over-forward JVPs; returns (out, d_x, d_y, d_t, d_xx, d_yy)."""
        f = self._point_fn(params)

        def per_point(z):
            out, d_x = jax.jvp(f, (z,), (_EX,))
            _, d_y = jax.jvp(f, (z,), (_EY,))
            _, d_t = jax.jvp(f, (z,), (_ET,))

            # Second derivatives along a single axis: differentiate the
            # directional derivative again in the same direction.
            d_xx = jax.jvp(lambda w: jax.jvp(f, (w,), (_EX,))[1], (z,), (_EX,))[1]
            d_yy = jax.jvp(lambda w: jax.jvp(f, (w,), (_EY,))[1], (z,), (_EY,))[1]
            return out, d_x, d_y, d_t, d_xx, d_yy

        return jax.vmap(per_point)(xyt)

    def _derivs_hessian(self, params, xyt):
        """Reference backend built on jax.jacfwd / jax.hessian."""
        f = self._point_fn(params)

        out = jax.vmap(f)(xyt)  # (N, 3)
        J = jax.vmap(jax.jacfwd(f))(xyt)  # (N, 3 out, 3 in)
        H = jax.vmap(jax.hessian(f))(xyt)  # (N, 3 out, 3 in, 3 in)

        d_x, d_y, d_t = J[:, :, 0], J[:, :, 1], J[:, :, 2]
        d_xx, d_yy = H[:, :, 0, 0], H[:, :, 1, 1]
        return out, d_x, d_y, d_t, d_xx, d_yy

    # -- residual -----------------------------------------------------------
    def residual(self, params, xyt):
        """Return the (N, 3) stack [continuity, x-momentum, y-momentum]."""
        derivs = self._derivs_fast if self.fast else self._derivs_hessian
        out, d_x, d_y, d_t, d_xx, d_yy = derivs(params, xyt)

        u, v = out[:, 0], out[:, 1]
        u_x, v_x, p_x = d_x[:, 0], d_x[:, 1], d_x[:, 2]
        u_y, v_y, p_y = d_y[:, 0], d_y[:, 1], d_y[:, 2]
        u_t, v_t = d_t[:, 0], d_t[:, 1]
        u_xx, v_xx = d_xx[:, 0], d_xx[:, 1]
        u_yy, v_yy = d_yy[:, 0], d_yy[:, 1]

        nu = 1.0 / self.Re
        cont = u_x + v_y
        mom_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return jnp.stack([cont, mom_x, mom_y], axis=1)


# --------------------------------------------------------------------------
# Taylor-Green vortex: an exact unsteady solution used to validate the residual
# --------------------------------------------------------------------------
class TaylorGreen:
    """Analytic decaying Taylor-Green vortex on (x, y) in [0, 2pi]^2.

        u = -cos(x) sin(y) e^{-2t/Re}
        v =  sin(x) cos(y) e^{-2t/Re}
        p = -(cos(2x) + cos(2y)) e^{-4t/Re} / 4

    This is an exact solution of the incompressible Navier-Stokes equations,
    so a correct residual must vanish on it to round-off. It duck-types the
    `.apply(params, xyt)` interface of a Flax module so it can be dropped
    straight into `NavierStokes2DUnsteadyPDE` in place of a network.
    """

    def __init__(self, Re: float = 100.0):
        self.Re = Re

    def apply(self, params, xyt):  # noqa: ARG002 - params unused by design
        x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]
        F = jnp.exp(-2.0 * t / self.Re)
        u = -jnp.cos(x) * jnp.sin(y) * F
        v = jnp.sin(x) * jnp.cos(y) * F
        p = -0.25 * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * F**2
        return jnp.stack([u, v, p], axis=1)
