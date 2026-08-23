"""Unsteady PINN for vortex shedding behind a cylinder at Re = 100.

This is the experiment the project turns on. A data-free unsteady PINN is
documented to collapse onto the steady, symmetric base flow instead of
shedding (Chuang & Barba, arXiv:2306.00230), and the whole point of the study
is to reproduce that failure honestly and then try to defeat it.

Every mechanism is a switch, so a single script produces the whole ablation:

    --decompose     learn the fluctuation u' on top of a frozen mean flow u0
                    (the trick the reference FNO paper uses, ported to a PINN)
    --net           mlp | fourier_mlp | siren -- attacks spectral bias
    --data-points   supervised samples from the LBM field (0 = data free)
    --windows N     causal time-marching: N short windows solved in sequence,
                    each initialised from the previous one's end state

Headline diagnostic
-------------------
Relative L2 against the LBM field is not enough on its own: a PINN that
relaxes to the steady base flow can still score a respectable L2, because the
mean flow *is* most of the field. So we also report the r.m.s. of the
transverse velocity at a wake probe. If that collapses toward zero while the
reference stays O(0.3), the network has stopped shedding -- which is exactly
the failure mode under study.

Usage:
    python src/pinn/cylinder_unsteady.py --smoke              # CPU sanity run
    python src/pinn/cylinder_unsteady.py --epochs 40000       # real run (GPU)
    python src/pinn/cylinder_unsteady.py --decompose --net fourier_mlp --windows 4
"""

from __future__ import annotations

import argparse
import json
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

from pinn.base_flow import BaseFlowNet  # noqa: E402
from pinn.data import load  # noqa: E402
from pinn.ns2d_unsteady import NavierStokes2DUnsteadyPDE  # noqa: E402

from underPINN.geometry.cylinder import Cylinder2D  # noqa: E402
from underPINN.nn.factory import build_model  # noqa: E402

PROBE = (2.0, 0.0)  # wake probe, ~2 D downstream on the centreline


# --------------------------------------------------------------------------
# Model wrappers (both duck-type `.apply(params, xyt)` for the PDE operator)
# --------------------------------------------------------------------------
class Rescaled:
    """Affine input rescaling in front of a network.

    t* runs to ~160 in the dataset, and feeding that raw into a tanh MLP
    saturates it immediately. Autodiff carries the chain rule through this
    wrapper, so the PDE residual stays in physical units.
    """

    def __init__(self, net, shift, scale):
        self.net = net
        self.shift = jnp.asarray(shift)
        self.scale = jnp.asarray(scale)

    def init(self, key, xyt):
        return self.net.init(key, (jnp.asarray(xyt) - self.shift) / self.scale)

    def apply(self, params, xyt):
        return self.net.apply(params, (xyt - self.shift) / self.scale)


class MeanPlusFluctuation:
    """u(x, y, t) = u0(x, y) + u'(x, y, t) with u0 frozen.

    Handing the *sum* to the PDE operator means the residual is the full
    Navier-Stokes residual, so no separate u' equation has to be derived --
    grad(u0) and lap(u0) come out of the frozen network by autodiff.
    """

    def __init__(self, base: BaseFlowNet, fluct: Rescaled):
        self.base = base
        self.fluct = fluct

    def init(self, key, xyt):
        return self.fluct.init(key, xyt)

    def apply(self, params, xyt):
        return self.base.apply(None, xyt[:, :2]) + self.fluct.apply(params, xyt)


# --------------------------------------------------------------------------
# Collocation / boundary sampling
# --------------------------------------------------------------------------
def sample_points(cfg, cyl, rng, t0, t1):
    """Interior, boundary and cylinder-surface points over [t0, t1]."""
    xmin, xmax, ymin, ymax = cfg["bounds"]

    def with_time(xy, n):
        t = rng.uniform(t0, t1, size=(n, 1))
        return np.concatenate([xy, t], axis=1)

    # interior: uniform background + a denser wake pool
    n_bg = cfg["n_interior"]
    n_wake = cfg["n_wake"]
    bg = cyl.sample_exterior(n_bg, xmin, xmax, ymin, ymax, seed=int(rng.integers(1 << 30)))
    wk = cyl.sample_exterior(n_wake, -1.0, 8.0, -2.5, 2.5, seed=int(rng.integers(1 << 30)))
    interior = np.concatenate([with_time(bg, n_bg), with_time(wk, n_wake)], axis=0)

    nb = cfg["n_bc"]
    inlet = with_time(np.stack([np.full(nb, xmin), rng.uniform(ymin, ymax, nb)], 1), nb)
    outlet = with_time(np.stack([np.full(nb, xmax), rng.uniform(ymin, ymax, nb)], 1), nb)
    wall_lo = with_time(np.stack([rng.uniform(xmin, xmax, nb), np.full(nb, ymin)], 1), nb)
    wall_hi = with_time(np.stack([rng.uniform(xmin, xmax, nb), np.full(nb, ymax)], 1), nb)
    surf = with_time(cyl.surface_points(nb).astype(np.float64), nb)

    return {k: jnp.asarray(v) for k, v in dict(
        interior=interior, inlet=inlet, outlet=outlet,
        wall_lo=wall_lo, wall_hi=wall_hi, surface=surf).items()}


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def make_loss(pde, w):
    """Total loss and its components, given weight dict `w`."""

    def terms(params, pts, ic, data):
        r = pde.residual(params, pts["interior"])
        l_pde = jnp.mean(r**2)

        q_in = pde.u(params, pts["inlet"])
        q_out = pde.u(params, pts["outlet"])
        q_lo = pde.u(params, pts["wall_lo"])
        q_hi = pde.u(params, pts["wall_hi"])
        q_s = pde.u(params, pts["surface"])

        l_bc = (
            jnp.mean((q_in[:, 0] - 1.0) ** 2) + jnp.mean(q_in[:, 1] ** 2)  # inlet u=1, v=0
            + jnp.mean(q_out[:, 2] ** 2)  # outlet p = 0
            + jnp.mean(q_lo[:, 1] ** 2) + jnp.mean(q_hi[:, 1] ** 2)  # symmetry v = 0
            + jnp.mean(q_s[:, 0] ** 2) + jnp.mean(q_s[:, 1] ** 2)  # no slip
        )

        ic_xyt, ic_val = ic
        l_ic = jnp.mean((pde.u(params, ic_xyt)[:, :2] - ic_val[:, :2]) ** 2)

        if data is None:
            l_dat = jnp.array(0.0)
        else:
            d_xyt, d_val = data
            l_dat = jnp.mean((pde.u(params, d_xyt)[:, :2] - d_val[:, :2]) ** 2)

        return l_pde, l_bc, l_ic, l_dat

    def total(params, pts, ic, data):
        l_pde, l_bc, l_ic, l_dat = terms(params, pts, ic, data)
        return (w["pde"] * l_pde + w["bc"] * l_bc
                + w["ic"] * l_ic + w["data"] * l_dat), (l_pde, l_bc, l_ic, l_dat)

    return total


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(pde, params, ds, t0, t1, n_times=8):
    """Relative L2 against LBM, plus the wake-probe shedding amplitude."""
    k0, k1 = ds.nearest_time_index(t0), ds.nearest_time_index(t1)
    ks = np.unique(np.linspace(k0, k1, n_times).astype(int))

    errs = []
    for k in ks:
        coords, vals = ds.snapshot(k)
        pred = np.asarray(pde.u(params, jnp.asarray(coords)))
        errs.append(ds.relative_l2(pred.T, vals.T) if pred.shape[0] == vals.shape[0]
                    else ds.relative_l2(pred, vals))

    # Shedding amplitude: r.m.s. of v(t) at the wake probe.
    tt = np.linspace(t0, t1, 200)
    probe = np.stack([np.full_like(tt, PROBE[0]), np.full_like(tt, PROBE[1]), tt], axis=1)
    v_pred = np.asarray(pde.u(params, jnp.asarray(probe)))[:, 1]

    i = int(np.argmin(np.abs(ds.x - PROBE[0])))
    j = int(np.argmin(np.abs(ds.y - PROBE[1])))
    sel = (ds.t >= t0) & (ds.t <= t1)
    v_true = ds.fields[sel, 1, i, j]

    return {
        "rel_l2_mean": float(np.mean(errs)),
        "rel_l2_max": float(np.max(errs)),
        "probe_v_rms_pinn": float(np.std(v_pred)),
        "probe_v_rms_lbm": float(np.std(v_true)),
        "shedding_ratio": float(np.std(v_pred) / max(np.std(v_true), 1e-12)),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_window(pde, params, opt, state, step_fn, cfg, cyl, ds, rng,
                 t0, t1, epochs, log_every, label):
    ic_k = ds.nearest_time_index(t0)
    ic_coords, ic_vals = ds.snapshot(ic_k)
    sub = rng.choice(ic_coords.shape[0], size=min(cfg["n_ic"], ic_coords.shape[0]),
                     replace=False)
    ic = (jnp.asarray(ic_coords[sub]), jnp.asarray(ic_vals[sub]))

    data = None
    if cfg["n_data"] > 0:
        lo, hi = ds.nearest_time_index(t0), ds.nearest_time_index(t1) + 1
        d_c, d_v = ds.sample(cfg["n_data"], rng, t_slice=(lo, max(hi, lo + 1)))
        data = (jnp.asarray(d_c), jnp.asarray(d_v))

    pts = sample_points(cfg, cyl, rng, t0, t1)
    hist = []
    t_start = time.time()
    for ep in range(epochs):
        if cfg["resample_every"] and ep and ep % cfg["resample_every"] == 0:
            pts = sample_points(cfg, cyl, rng, t0, t1)
        params, state, l, parts = step_fn(params, state, pts, ic, data)
        if ep % log_every == 0 or ep == epochs - 1:
            lp, lb, li, ld = (float(x) for x in parts)
            hist.append(dict(epoch=ep, loss=float(l), pde=lp, bc=lb, ic=li, data=ld))
            print(f"  [{label}] ep {ep:>6d}  loss {float(l):.3e}  "
                  f"pde {lp:.2e}  bc {lb:.2e}  ic {li:.2e}  dat {ld:.2e}"
                  f"  ({time.time() - t_start:5.1f}s)", flush=True)
    return params, state, hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="re100_v3")
    ap.add_argument("--net", default="mlp",
                    choices=["mlp", "gated_mlp", "fourier_mlp", "siren"])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-fourier", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--decompose", action="store_true",
                    help="learn u' on top of the frozen mean flow")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--windows", type=int, default=1, help="causal time-marching windows")
    ap.add_argument("--window-len", type=float, default=6.0,
                    help="length of each window in t* (one shedding period ~ 6)")
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--n-interior", type=int, default=8000)
    ap.add_argument("--n-wake", type=int, default=4000)
    ap.add_argument("--n-bc", type=int, default=400)
    ap.add_argument("--n-ic", type=int, default=4000)
    ap.add_argument("--n-data", type=int, default=0,
                    help="supervised samples per window (0 = data free)")
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=10.0)
    ap.add_argument("--w-ic", type=float, default=10.0)
    ap.add_argument("--w-data", type=float, default=10.0)
    ap.add_argument("--resample-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f64", action="store_true",
                    help="float64 (slow on GPU; for precision checks only)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.n_interior, args.n_wake = 300, 800, 400
        args.n_bc, args.n_ic, args.log_every = 100, 500, 50
        args.width, args.depth = 32, 3

    ds = load(args.tag)
    xmin, xmax, ymin, ymax = ds.bounds
    t_lo, t_hi = ds.t_range
    t_start = args.t_start if args.t_start is not None else t_lo
    cyl = Cylinder2D(radius=0.5, center=(0.0, 0.0))

    cfg = dict(bounds=(xmin, xmax, ymin, ymax), n_interior=args.n_interior,
               n_wake=args.n_wake, n_bc=args.n_bc, n_ic=args.n_ic,
               n_data=args.n_data, resample_every=args.resample_every)

    # ---- model ----------------------------------------------------------
    net_cfg = {"type": args.net, "layers": [3] + [args.width] * args.depth + [3]}
    if args.net in ("fourier_mlp", "fourier"):
        net_cfg.update(n_fourier=args.n_fourier, sigma=args.sigma)
    base_net = build_model(net_cfg)

    span = args.windows * args.window_len
    shift = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax), t_start + 0.5 * span])
    scale = np.array([0.5 * (xmax - xmin), 0.5 * (ymax - ymin), max(0.5 * span, 1e-6)])
    model = Rescaled(base_net, shift, scale)

    if args.decompose:
        bf_path = ROOT / "data" / f"base_flow_{args.tag}.pkl"
        if not bf_path.exists():
            raise SystemExit(f"{bf_path} missing -- run src/pinn/base_flow.py first")
        model = MeanPlusFluctuation(BaseFlowNet.restore(bf_path), model)

    pde = NavierStokes2DUnsteadyPDE(model, Re=ds.Re, fast=True)

    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.zeros((1, 3)))

    sched = optax.exponential_decay(args.lr, transition_steps=max(args.epochs // 5, 1),
                                    decay_rate=0.5, staircase=True)
    opt = optax.adam(sched)
    state = opt.init(params)

    w = dict(pde=args.w_pde, bc=args.w_bc, ic=args.w_ic, data=args.w_data)
    loss_fn = make_loss(pde, w)

    @jax.jit
    def step_fn(p, st, pts, ic, data):
        (l, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(p, pts, ic, data)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, l, parts

    # ---- run ------------------------------------------------------------
    mode = ("decomposed " if args.decompose else "") + args.net
    print(f"Re={ds.Re:g}  mode={mode}  windows={args.windows} x {args.window_len} t*"
          f"  data={args.n_data}  epochs/window={args.epochs}")
    print(f"domain x[{xmin:g},{xmax:g}] y[{ymin:g},{ymax:g}]  "
          f"t* {t_start:.2f} .. {min(t_start + span, t_hi):.2f}\n")

    rng = np.random.default_rng(args.seed)
    all_hist, results = [], []
    for wdw in range(args.windows):
        a = t_start + wdw * args.window_len
        b = min(a + args.window_len, t_hi)
        if a >= t_hi:
            break
        params, state, hist = train_window(
            pde, params, opt, state, step_fn, cfg, cyl, ds, rng,
            a, b, args.epochs, args.log_every, f"win {wdw + 1}/{args.windows}")
        all_hist.append(hist)
        m = evaluate(pde, params, ds, a, b)
        results.append(dict(window=wdw, t0=a, t1=b, **m))
        print(f"  -> rel L2 {m['rel_l2_mean']:.4f}   "
              f"probe v rms: PINN {m['probe_v_rms_pinn']:.4f} vs "
              f"LBM {m['probe_v_rms_lbm']:.4f}   "
              f"shedding retained {100 * m['shedding_ratio']:.1f} %\n", flush=True)

    # ---- report ---------------------------------------------------------
    out = Path(args.out) if args.out else ROOT / "runs" / (
        f"{'decomp_' if args.decompose else ''}{args.net}"
        f"_d{args.n_data}_w{args.windows}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "results": results, "history": all_hist}, indent=2))

    print("=" * 68)
    for r in results:
        print(f"  window {r['window']}  t* {r['t0']:.1f}-{r['t1']:.1f}   "
              f"relL2 {r['rel_l2_mean']:.4f}   shedding {100 * r['shedding_ratio']:5.1f} %")
    if results:
        avg = np.mean([r["shedding_ratio"] for r in results])
        print("=" * 68)
        print(f"  mean shedding retained: {100 * avg:.1f} %")
        print("  (a value near 0 % is the documented collapse to steady flow)")
    print(f"\nresults -> {out}/results.json")


if __name__ == "__main__":
    main()
