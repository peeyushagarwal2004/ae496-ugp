"""Phase 5: FNO forecasting of the Re = 100 wake, with and without physics.

Two questions, one script:

  (a) **Reproduce.** Train underPINN's `FNO2D` the way the reference paper
      trains its operator (its Table 1 and section 3.1): Reynolds-decompose
      first, map two consecutive frames to the next one, apply the model
      recursively for 10 steps during training and sum the loss over every
      step.

  (b) **Contribute.** Add a physics term to that loss -- continuity and
      pressure-free vorticity transport, both by finite differences on the
      grid (`src/fno/physics.py`) -- and compare rollout error *well beyond*
      the trained horizon, at 10, 20 and 50 steps.

The expected shape of the answer is the finding: physics should matter little
inside the trained horizon and increasingly outside it, where a purely
data-driven rollout drifts. If it helps everywhere, suspect a bug; if it helps
nowhere, that is a real negative result about physics priors for short-horizon
forecasting, and should be written up as one.

Usage:
    python src/fno/cylinder_fno.py --tag re100_v4 --variant data
    python src/fno/cylinder_fno.py --tag re100_v4 --variant physics
    python src/fno/cylinder_fno.py --quick          # tiny CPU sanity run

Note on data: this needs a long record -- the 50-step rollout alone is 50
snapshots after the training split. The trimmed 34-snapshot copy on the IITK
server (RESUME section 11) is *not* enough; use the full
`data/cylinder_re100_v4.npz` (641 snapshots).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upinn"))

from fno import physics as phys  # noqa: E402
from pinn.data import load  # noqa: E402

from underPINN.nn.operators import FNO2D  # noqa: E402


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

class WakeFrames:
    """Snapshot velocities laid out the way an operator wants them.

    `fluct` is (nt, nx, ny, 2) -- the paper learns the fluctuation, not the
    field -- and `mean` is (nx, ny, 2), added back whenever physics or an error
    metric needs the real velocity.

    The grid is cropped to even side lengths because underPINN's
    `SpectralConv2d` only accepts those, and optionally strided to cut memory.
    """

    def __init__(self, ds, downsample: int = 1):
        vel = np.transpose(ds.fields[:, :2], (0, 2, 3, 1))  # (nt, nx, ny, 2)
        mean = np.transpose(ds.mean_field[:2], (1, 2, 0))  # (nx, ny, 2)
        solid = ds.solid
        x, y = ds.x, ds.y

        s = max(1, int(downsample))
        vel, mean, solid, x, y = vel[:, ::s, ::s], mean[::s, ::s], solid[::s, ::s], x[::s], y[::s]

        nx, ny = (vel.shape[1] // 2) * 2, (vel.shape[2] // 2) * 2
        self.vel = np.ascontiguousarray(vel[:, :nx, :ny], dtype=np.float32)
        self.mean = np.ascontiguousarray(mean[:nx, :ny], dtype=np.float32)
        self.solid = np.ascontiguousarray(solid[:nx, :ny])
        self.fluid = (~self.solid).astype(np.float32)[..., None]  # (nx, ny, 1)

        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])
        self.dt = float(ds.t[1] - ds.t[0])
        self.t = np.asarray(ds.t, dtype=np.float64)
        self.Re = float(ds.Re)
        self.interior = phys.interior_mask(self.solid)

        self.fluct = self.vel - self.mean[None]
        # The cylinder interior holds whatever the LBM left there; it is not
        # flow, so it must not enter a loss or a norm.
        self.fluct *= self.fluid[None]

    @property
    def n_times(self):
        return self.fluct.shape[0]

    @property
    def grid(self):
        return self.fluct.shape[1], self.fluct.shape[2]


def split_indices(n_times, train_frac):
    """Contiguous train/test split in time -- never interleaved.

    A random split would let the model see frame k+1 in training and be scored
    on forecasting it from frame k, which is leakage: consecutive snapshots of
    a periodic wake are nearly the same picture.
    """
    n_train = int(round(train_frac * n_times))
    return np.arange(0, n_train), np.arange(n_train, n_times)


def rollout_starts(idx, rollout, stride=1):
    """Frame indices `k` usable as the second input frame of a sample.

    A sample needs `k-1` and `k` as input and `k+1 .. k+rollout` as targets,
    all inside `idx`.
    """
    lo, hi = int(idx[0]), int(idx[-1])
    return np.arange(lo + 1, hi - rollout + 1, stride)


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

def build(cfg, frames):
    """FNO2D mapping two stacked frames (4 channels) to the next one (2)."""
    nx, ny = frames.grid
    model = FNO2D(
        modes1=min(cfg["modes"], nx // 2 + 1),
        modes2=min(cfg["modes"], ny // 2 + 1),
        width=cfg["width"],
        depth=cfg["depth"],
        channels_last_proj=cfg["proj"],
        out_channels=2,
        # The wake is not periodic in either direction, and a spectral
        # convolution assumes it is; padding is the standard FNO remedy.
        padding=cfg["padding"],
    )
    key = jax.random.PRNGKey(cfg["seed"])
    params = model.init(key, jnp.zeros((1, nx, ny, 4), jnp.float32))
    return model, params


def make_step_fn(model, frames):
    """One forecast step: (prev, now) -> next, in normalised fluctuation units.

    Normalising by the training r.m.s. (see `train`) keeps the network's inputs
    and outputs O(1) without touching the physics, which always runs on the
    rescaled field.
    """
    fluid = jnp.asarray(frames.fluid)

    def step(params, prev, now):
        x = jnp.concatenate([prev, now], axis=-1)
        nxt = model.apply(params, x)
        # Nothing flows inside the cylinder; enforcing that here stops the
        # solid interior from feeding garbage back in during a rollout.
        return nxt * fluid

    return step


def make_rollout_fn(step, remat=True):
    """Apply `step` recursively, returning every intermediate prediction.

    `jax.checkpoint` trades a second forward pass for not storing the
    activations of all `n` steps -- without it a 10-step rollout at width 80
    exhausts a 40 GB card.
    """
    inner = jax.checkpoint(step) if remat else step

    def rollout(params, prev, now, n):
        def body(carry, _):
            p, c = carry
            nxt = inner(params, p, c)
            return (c, nxt), nxt

        _, preds = jax.lax.scan(body, (prev, now), None, length=n)
        return preds  # (n, batch, nx, ny, 2)

    return rollout


# ---------------------------------------------------------------------------
#  Losses
# ---------------------------------------------------------------------------

def relative_l2(pred, true, axis):
    """The paper's eq. 3.3, per sample: ||q* - q|| / ||q|| over stacked (u, v)."""
    num = jnp.sqrt(jnp.sum((pred - true) ** 2, axis=axis))
    den = jnp.sqrt(jnp.sum(true**2, axis=axis)) + 1e-12
    return num / den


def reference_residuals(frames, idx, n_samples=16):
    """The residuals the *ground-truth data* itself leaves on this grid.

    LBM is weakly compressible and second-order accurate, so the measured field
    does not satisfy either residual exactly (continuity error tracks O(Ma^2),
    see RESUME section 1.10). Driving a prediction below that floor is not
    physics, it is fitting the discretisation error.

    Both physics terms are therefore divided by these numbers, which makes
    `w_div = w_vort = 1` mean "penalise the prediction as hard as the data's
    own violation" -- a scale-free default that survives a change of grid,
    time step or Reynolds number.
    """
    ks = np.linspace(idx[0] + 1, idx[-1] - 1, min(n_samples, len(idx) - 2)).astype(int)
    prev = jnp.asarray(frames.vel[ks - 1])
    now = jnp.asarray(frames.vel[ks])
    nxt = jnp.asarray(frames.vel[ks + 1])
    mask = jnp.asarray(frames.interior)
    div = phys.masked_mean_square(phys.divergence(nxt, frames.dx, frames.dy), mask)
    vort = phys.masked_mean_square(
        phys.vorticity_transport_residual(prev, now, nxt, frames.dx, frames.dy,
                                          frames.dt, frames.Re), mask)
    return {"div": float(div), "vort": float(vort)}


def make_loss_fn(cfg, frames, rollout, scale, reference):
    """Data loss (always) plus physics residuals (when weighted above zero).

    The data term is the relative-L2 *summed over rollout steps*, which is what
    makes the recursive application stable rather than merely one-step accurate.
    """
    mean = jnp.asarray(frames.mean)
    interior = jnp.asarray(frames.interior)
    dx, dy, dt, Re = frames.dx, frames.dy, frames.dt, frames.Re
    w_div, w_vort = cfg["w_div"], cfg["w_vort"]
    # Residuals are reported as multiples of the data's own (see
    # `reference_residuals`), so the weights are dimensionless.
    ref_div = max(reference["div"], 1e-30)
    ref_vort = max(reference["vort"], 1e-30)
    use_physics = (w_div > 0.0) or (w_vort > 0.0)
    grid_axes = (-3, -2, -1)

    def full_field(fluct_norm):
        """Normalised fluctuation -> physical velocity `u0 + u'`."""
        return fluct_norm * scale + mean

    def loss_fn(params, prev, now, targets):
        # targets: (n_steps, batch, nx, ny, 2), normalised fluctuation
        n_steps = targets.shape[0]
        preds = rollout(params, prev, now, n_steps)
        l_data = jnp.sum(jnp.mean(relative_l2(preds, targets, grid_axes), axis=-1))

        l_div = l_vort = 0.0
        if use_physics:
            # Physical fields for the whole chain: the two inputs, then every
            # prediction. Residuals need three consecutive frames.
            chain = jnp.concatenate([prev[None], now[None], preds], axis=0)
            chain = full_field(chain)
            if w_div > 0.0:
                l_div = phys.masked_mean_square(
                    phys.divergence(chain[2:], dx, dy), interior) / ref_div
            if w_vort > 0.0:
                l_vort = phys.masked_mean_square(
                    phys.vorticity_transport_residual(
                        chain[:-2], chain[1:-1], chain[2:], dx, dy, dt, Re),
                    interior) / ref_vort

        total = l_data + w_div * l_div + w_vort * l_vort
        return total, {"data": l_data, "div": l_div, "vort": l_vort}

    return loss_fn


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def batches(starts, batch, rng):
    order = rng.permutation(starts)
    for i in range(0, len(order) - batch + 1, batch):
        yield order[i:i + batch]


def gather(fluct_norm, ks, rollout):
    """Assemble (prev, now, targets) for the sample starts `ks`."""
    prev = fluct_norm[ks - 1]
    now = fluct_norm[ks]
    targets = np.stack([fluct_norm[ks + s] for s in range(1, rollout + 1)], axis=0)
    return jnp.asarray(prev), jnp.asarray(now), jnp.asarray(targets)


def train(cfg, frames, verbose=True):
    train_idx, test_idx = split_indices(frames.n_times, cfg["train_frac"])
    if len(test_idx) < cfg["eval_steps"] + 2:
        raise SystemExit(
            f"test split has {len(test_idx)} frames but the rollout needs "
            f"{cfg['eval_steps'] + 2}; use a longer record or a smaller "
            f"--train-frac / --eval-steps")

    scale = float(np.sqrt(np.mean(frames.fluct[train_idx] ** 2)))
    fluct_norm = frames.fluct / scale

    model, params = build(cfg, frames)
    step = make_step_fn(model, frames)
    rollout = make_rollout_fn(step, remat=cfg["remat"])
    reference = reference_residuals(frames, train_idx)
    if verbose:
        print(f"  data's own residuals: div {reference['div']:.3e}  "
              f"vort {reference['vort']:.3e}  (physics terms are relative to these)")
    loss_fn = make_loss_fn(cfg, frames, rollout, scale, reference)

    starts = rollout_starts(train_idx, cfg["rollout"])
    steps_per_epoch = max(len(starts) // cfg["batch"], 1)
    sched = optax.exponential_decay(
        cfg["lr"], transition_steps=cfg["decay_every"] * steps_per_epoch,
        decay_rate=cfg["decay"], staircase=True)
    opt = optax.adam(sched)
    opt_state = opt.init(params)

    @jax.jit
    def update(params, opt_state, prev, now, targets):
        (total, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, prev, now, targets)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, total, parts

    rng = np.random.default_rng(cfg["seed"])
    history, t0 = [], time.time()
    for epoch in range(cfg["epochs"]):
        epoch_loss, parts_sum, n = 0.0, {"data": 0.0, "div": 0.0, "vort": 0.0}, 0
        for ks in batches(starts, cfg["batch"], rng):
            prev, now, targets = gather(fluct_norm, ks, cfg["rollout"])
            params, opt_state, total, parts = update(
                params, opt_state, prev, now, targets)
            epoch_loss += float(total)
            for k in parts_sum:
                parts_sum[k] += float(parts[k])
            n += 1
        row = {"epoch": epoch, "loss": epoch_loss / max(n, 1),
               **{k: v / max(n, 1) for k, v in parts_sum.items()},
               "seconds": time.time() - t0}
        history.append(row)
        if verbose and (epoch % cfg["log_every"] == 0 or epoch == cfg["epochs"] - 1):
            print(f"  ep {epoch:4d}  loss {row['loss']:.4e}  data {row['data']:.4e}"
                  f"  div {row['div']:.3e}  vort {row['vort']:.3e}"
                  f"  ({row['seconds']:6.1f}s)", flush=True)

    return dict(params=params, scale=scale, model=model, step=step,
                rollout=rollout, history=history, reference=reference,
                train_idx=train_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate_rollout(trained, frames, cfg):
    """Relative L2 per rollout step, averaged over held-out initial conditions.

    Reported two ways, because they differ by roughly the ratio of mean to
    fluctuation energy and quoting the wrong one flatters the model:

      `eps_full`   on `u = u0 + u'` -- comparable to the reference paper, which
                   reports error on the velocity field.
      `eps_fluct`  on `u'` alone -- the harder, more honest number, since the
                   frozen mean contributes no error of its own.
    """
    n = cfg["eval_steps"]
    fluct_norm = frames.fluct / trained["scale"]
    starts = rollout_starts(trained["test_idx"], n)
    if cfg["n_eval_ic"] and len(starts) > cfg["n_eval_ic"]:
        starts = np.linspace(starts[0], starts[-1], cfg["n_eval_ic"]).astype(int)

    prev, now, targets = gather(fluct_norm, starts, n)
    preds = trained["rollout"](trained["params"], prev, now, n)

    mean = jnp.asarray(frames.mean)
    scale = trained["scale"]
    grid_axes = (-3, -2, -1)
    eps_fluct = relative_l2(preds, targets, grid_axes)  # (n, n_ic)
    eps_full = relative_l2(preds * scale + mean, targets * scale + mean, grid_axes)

    return {
        "steps": list(range(1, n + 1)),
        "eps_fluct": np.asarray(eps_fluct.mean(axis=-1)).tolist(),
        "eps_full": np.asarray(eps_full.mean(axis=-1)).tolist(),
        "n_initial_conditions": int(len(starts)),
        "t_start": [float(frames.t[k]) for k in starts],
    }


def headline(curve, marks=(1, 10, 20, 50)):
    """The few rollout horizons the write-up actually quotes, plus the last one."""
    out = {}
    for m in sorted({*marks, len(curve["steps"])}):
        if m <= len(curve["steps"]):
            out[f"eps_full@{m}"] = curve["eps_full"][m - 1]
            out[f"eps_fluct@{m}"] = curve["eps_fluct"][m - 1]
    return out


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

VARIANTS = {
    # The reference paper's recipe: no physics anywhere in the loss.
    "data": dict(w_div=0.0, w_vort=0.0),
    # This project's contribution. Both residuals are already normalised by the
    # data's own violation (`reference_residuals`), so a weight is "how many
    # times the data's own error do we forbid". These values put both terms
    # within a factor of a few of the data term at initialisation -- check the
    # printed `div`/`vort` columns if the grid or the dataset changes.
    "physics": dict(w_div=1.0, w_vort=0.01),
    # Ablation of the contribution: continuity only, the cheapest possible term.
    "physics_div": dict(w_div=1.0, w_vort=0.0),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="re100_v4")
    ap.add_argument("--variant", default="data", choices=sorted(VARIANTS))
    # Paper's Table 1.
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--padding", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8,
                    help="paper uses 16; 8 fits a 40 GB card at width 80")
    ap.add_argument("--rollout", type=int, default=10,
                    help="recursive steps trained through, loss summed over all")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--decay", type=float, default=0.90)
    ap.add_argument("--decay-every", type=int, default=5, help="epochs per lr step")
    # Physics weights default to the variant; override to sweep them.
    ap.add_argument("--w-div", type=float, default=None)
    ap.add_argument("--w-vort", type=float, default=None)
    # Evaluation.
    ap.add_argument("--eval-steps", type=int, default=50,
                    help="rollout horizon at evaluation (5x the trained one)")
    ap.add_argument("--n-eval-ic", type=int, default=16,
                    help="held-out initial conditions to average over")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--downsample", type=int, default=1,
                    help="spatial stride; 2 quarters the memory")
    ap.add_argument("--no-remat", dest="remat", action="store_false",
                    help="keep rollout activations (faster, much more memory)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-params", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny CPU sanity run: minutes, not a result")
    args = ap.parse_args(argv)

    if args.quick:
        args.epochs, args.width, args.modes, args.depth = 2, 8, 6, 2
        args.batch, args.rollout, args.eval_steps = 2, 2, 6
        args.n_eval_ic, args.downsample, args.log_every = 2, 4, 1
    return args


def config_from(args):
    cfg = dict(vars(args))
    cfg.update(VARIANTS[args.variant])
    if args.w_div is not None:
        cfg["w_div"] = args.w_div
    if args.w_vort is not None:
        cfg["w_vort"] = args.w_vort
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = config_from(args)

    ds = load(args.tag)
    frames = WakeFrames(ds, downsample=cfg["downsample"])
    nx, ny = frames.grid
    print(f"grid {nx} x {ny}   dt* {frames.dt:.4f}   Re {frames.Re:g}   "
          f"snapshots {frames.n_times}")
    print(f"variant {args.variant}  w_div={cfg['w_div']}  w_vort={cfg['w_vort']}  "
          f"rollout {cfg['rollout']} -> eval {cfg['eval_steps']}\n")

    trained = train(cfg, frames)
    curve = evaluate_rollout(trained, frames, cfg)
    marks = headline(curve)

    print("\n  rollout error (relative L2, held-out initial conditions)")
    for k in sorted(marks, key=lambda s: (s.split("@")[0], int(s.split("@")[1]))):
        print(f"    {k:>16s}  {marks[k]:.4f}")

    out = Path(args.out) if args.out else ROOT / "runs" / "fno" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    blob = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "curve": curve,
        "headline": marks,
        "history": trained["history"],
        "scale": trained["scale"],
        "data_residuals": trained["reference"],
        "n_train": int(len(trained["train_idx"])),
        "n_test": int(len(trained["test_idx"])),
    }
    (out / "results.json").write_text(json.dumps(blob, indent=2))
    if args.save_params:
        with open(out / "params.pkl", "wb") as fh:
            pickle.dump({"params": trained["params"], "scale": trained["scale"],
                         "cfg": cfg}, fh)
    print(f"\nresults -> {out / 'results.json'}")
    return blob


if __name__ == "__main__":
    main()
