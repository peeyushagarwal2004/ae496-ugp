"""Smooth analytic representation of the time-mean base flow u0.

The Reynolds decomposition u = u0 + u' needs u0 at arbitrary points *with
derivatives*, because the momentum residual contains grad(u0) and lap(u0).
Bilinear interpolation of the LBM mean field would give piecewise-constant
first derivatives and zero second derivatives, which is useless inside a PDE
residual. So we fit a small MLP to the mean field by plain supervised
regression -- no physics involved -- and then differentiate it analytically.

Note this is deliberately *not* a steady-NS PINN solve. The time mean of a
shedding flow does not satisfy the steady Navier-Stokes equations: it carries
the divergence of the Reynolds stresses. Fitting the measured mean keeps the
decomposition exact, and pushes that term into the u' equation where it
belongs.

Usage:
    python src/pinn/base_flow.py --tag re100_v3 --epochs 20000
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import jax

# Precision must be set before the first jnp array exists, so peek at argv.
# Default float32: PINNs train in single precision as standard practice, and
# FP64 runs at 1/32 rate on a T4, which would dominate the GPU budget here.
jax.config.update("jax_enable_x64", "--f64" in sys.argv)

import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upinn"))

from pinn.data import load  # noqa: E402

from underPINN.nn.factory import build_model  # noqa: E402


class BaseFlowNet:
    """Frozen (x, y) -> (u0, v0, p0) network with an affine input rescaling.

    Duck-types `.apply(params, xy)` so it can be composed with the unsteady
    network and handed to the PDE operator unchanged.
    """

    def __init__(self, net, params, shift, scale):
        self.net = net
        self.params = params
        self.shift = jnp.asarray(shift)
        self.scale = jnp.asarray(scale)

    def apply(self, params, xy):
        """Evaluate at physical coordinates xy (N, 2); `params` is ignored."""
        return self.net.apply(self.params, (xy - self.shift) / self.scale)

    def __call__(self, xy):
        return self.apply(None, xy)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(
                {"params": self.params, "shift": np.asarray(self.shift),
                 "scale": np.asarray(self.scale), "cfg": self.cfg}, fh)

    @staticmethod
    def restore(path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        net = build_model(blob["cfg"])
        obj = BaseFlowNet(net, blob["params"], blob["shift"], blob["scale"])
        obj.cfg = blob["cfg"]
        return obj


def fit(tag="re100_v3", width=64, depth=5, epochs=20000, lr=1e-3,
        batch=4096, seed=0, verbose=True, net_type="mlp", n_fourier=64,
        sigma=3.0):
    """Regress the LBM time-mean field with a tanh MLP."""
    ds = load(tag)
    xmin, xmax, ymin, ymax = ds.bounds

    ii, jj = np.where(~ds.solid)
    xy = np.stack([ds.x[ii], ds.y[jj]], axis=1)
    target = ds.mean_field[:, ii, jj].T  # (M, 3)

    # Rescale inputs to ~[-1, 1]; tanh networks are badly conditioned otherwise.
    shift = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
    scale = np.array([0.5 * (xmax - xmin), 0.5 * (ymax - ymin)])

    cfg = {"type": net_type, "layers": [2] + [width] * depth + [3]}
    if net_type in ("fourier_mlp", "fourier"):
        cfg.update(n_fourier=n_fourier, sigma=sigma)
    net = build_model(cfg)
    xy_j = jnp.asarray(xy)
    tgt_j = jnp.asarray(target)
    shift_j, scale_j = jnp.asarray(shift), jnp.asarray(scale)

    params = net.init(jax.random.PRNGKey(seed), (xy_j[:1] - shift_j) / scale_j)
    sched = optax.exponential_decay(lr, transition_steps=max(epochs // 8, 1),
                                    decay_rate=0.5, staircase=True)
    opt = optax.adam(sched)
    state = opt.init(params)

    def loss_fn(p, xb, yb):
        pred = net.apply(p, (xb - shift_j) / scale_j)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def step(p, st, xb, yb):
        v, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, v

    key = jax.random.PRNGKey(seed + 1)
    n = xy_j.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch,), 0, n)
        params, state, v = step(params, state, xy_j[idx], tgt_j[idx])
        if verbose and (ep % max(epochs // 10, 1) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>6d}  mse {float(v):.3e}  ({time.time() - t0:5.1f}s)",
                  flush=True)

    obj = BaseFlowNet(net, params, shift_j, scale_j)
    obj.cfg = cfg

    pred = np.asarray(net.apply(params, (xy_j - shift_j) / scale_j))
    err = pred - target
    rel = np.linalg.norm(err[:, :2]) / np.linalg.norm(target[:, :2])
    if verbose:
        print(f"\n  relative L2 on (u0, v0) = {rel:.4e}")
        for k, nm in enumerate("uvp"):
            print(f"    {nm}0: max abs err {np.abs(err[:, k]).max():.4e}"
                  f"   rms {np.sqrt((err[:, k] ** 2).mean()):.4e}")
    return obj, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="re100_v3")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--net", default="mlp",
                    choices=["mlp", "gated_mlp", "fourier_mlp", "siren"])
    ap.add_argument("--n-fourier", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--f64", action="store_true", help="float64 (slow on GPU)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"fitting base flow u0 for {args.tag} "
          f"({args.net} {args.width}x{args.depth}, {args.epochs} epochs)")
    obj, rel = fit(args.tag, args.width, args.depth, args.epochs, args.lr,
                   args.batch, net_type=args.net, n_fourier=args.n_fourier,
                   sigma=args.sigma)

    out = Path(args.out) if args.out else ROOT / "data" / f"base_flow_{args.tag}.pkl"
    obj.save(out)
    print(f"\nsaved -> {out}")

    # Derivatives must be smooth for the residual to be meaningful: report the
    # magnitude of the Laplacian the PDE will actually see.
    def single(z):
        return obj.apply(None, z[None, :])[0]

    pts = jnp.asarray(np.stack(np.meshgrid(
        np.linspace(-4, 14, 40), np.linspace(-4, 4, 20), indexing="ij"),
        axis=-1).reshape(-1, 2))
    H = jax.vmap(jax.hessian(single))(pts)
    lap = H[:, :, 0, 0] + H[:, :, 1, 1]
    print(f"laplacian of u0 over the domain: "
          f"max|lap u| {float(jnp.abs(lap[:, 0]).max()):.3f}, "
          f"max|lap v| {float(jnp.abs(lap[:, 1]).max()):.3f}")


if __name__ == "__main__":
    main()
