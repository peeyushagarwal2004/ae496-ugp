# Running on Google Colab (results auto-pushed to GitHub)

Colab VMs are destroyed on disconnect, and free Colab disconnects readily. So
this workflow **pushes results to GitHub after every stage** — a dropped session
costs you the current run, never the completed ones.

Ground truth is regenerated on the GPU (~5 min) rather than uploaded: it is
deterministic, and 190 MB does not belong in a repo.

## Before you start

1. **Runtime → Change runtime type → T4 GPU.**
2. **Secrets** (key icon, left sidebar) → **Add new secret**
   * Name `GH_TOKEN`, value = your GitHub personal access token (`repo` scope)
   * Toggle **Notebook access** on — without this `userdata.get` raises.

The repo is public, so the clone needs no token. The token is only for *pushing*
results back, which always requires authentication.

---

## Cell 1 — clone and install

The repo is public, so cloning needs no credentials. The token is still required
later — pushing always authenticates, public or not.

```python
import subprocess, os
from google.colab import userdata

GH_USER, GH_REPO = "peeyushagarwal2004", "ae496-ugp"
TOKEN = userdata.get('GH_TOKEN')      # needed only for pushing results

os.chdir("/content")      # must leave /content/ugp before deleting it
subprocess.run(["rm", "-rf", "/content/ugp"], check=False)
for url, dest in [
    (f"https://github.com/{GH_USER}/{GH_REPO}.git", "/content/ugp"),
    ("https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN.git",
     "/content/ugp/upinn"),
]:
    p = subprocess.run(["git", "clone", url, dest], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"clone failed ({p.returncode}):
{p.stderr}")
os.chdir("/content/ugp")
print(subprocess.run(["ls"], capture_output=True, text=True).stdout)
```

```python
!pip install -q -U "flax>=0.12.9" "optax>=0.2.8"
```

**Then Runtime -> Restart session.** Colab ships a flax that predates JAX 0.11
and calls `jax.core.get_opaque_trace_state`, removed in JAX 0.11.0 — every
network build fails with an `AttributeError` until flax is upgraded. The
restart is required because jax/flax are already imported; without it the
upgrade has no effect on the running kernel.

`src` must appear in that listing. If it does not, the clone failed and nothing
below will work.

## Cell 2 — verify GPU, define helpers

Must be a **separate cell** — a `pip install` does not affect modules already
imported in the same cell.

```python
import subprocess, sys, os, time
os.chdir("/content/ugp"); sys.path.insert(0, "/content/ugp")

import jax
print("jax", jax.__version__, "|", jax.default_backend(), jax.devices())

from experiments.push_results import push

def run(*args, timeout=None):
    """Run a project script, streaming the tail of its output."""
    t0 = time.time()
    p = subprocess.run([sys.executable, "-u", *args], cwd="/content/ugp",
                       capture_output=True, text=True, timeout=timeout)
    print(p.stdout[-4000:])
    if p.returncode:
        print("--- STDERR ---\n", p.stderr[-3000:])
    print(f"[{args[0]} finished in {(time.time()-t0)/60:.1f} min, exit {p.returncode}]")
    return p.returncode

def save(msg):
    push(token=TOKEN, user=GH_USER, repo=GH_REPO, message=msg)
```

**`backend` must print `gpu`.** If it says `cpu`, run
`!pip install -q -U "jax[cuda12]"`, then **Runtime → Restart session**, and
re-run Cells 1–2. Everything below assumes GPU.

`subprocess` is used rather than `!` because the `!` magic forks a process that
already holds JAX's threads, which can deadlock on long runs.

## Cell 3 — ground truth (~5 min)

```python
run("src/cfd/lbm_cylinder.py", "--u", "0.05", "--height", "50",
    "--steps", "102000", "--warmup-periods", "15",
    "--snaps-per-period", "32", "--tag", "re100_v4")
run("src/cfd/inspect_data.py", "--tag", "re100_v4")
save("LBM ground truth re100_v4 (Colab)")
```

Compare against the values this exact command produced locally:

```
Cd = 1.3798   (benchmark 1.37-1.38,  +0.3 %)
St = 0.1611   (benchmark 0.165,      -2.3 %)
```

LBM is deterministic, so the GPU should reproduce these to float32 round-off.
If it does not, stop — the GPU build is wrong, and training on it wastes quota.

## Cell 4 — base flow for the decomposition (~2 min)

```python
run("src/pinn/base_flow.py", "--tag", "re100_v4",
    "--net", "fourier_mlp", "--sigma", "12", "--epochs", "25000")
save("base flow fit re100_v4 (Colab)")
```

Target **relative L2 below ~1 %**; locally this reached **1.2e-3**. A loose base
flow leaks a steady residual into u' and defeats the decomposition, so do not
proceed if it lands above ~2 %.

## Cell 5 — calibrate (~5 min)

```python
run("experiments/run_ablation.py", "--tag", "re100_v4", "--quick")
```

Six configs at 2000 epochs. **Not a result** — it measures per-run cost so you
can size Cell 6, and proves every code path works on GPU. Read the `min` column
and multiply by ~10.

## Cell 6 — the real experiment (push after each config)

Run one configuration at a time and push after each, so a disconnect costs at
most one config rather than the whole sweep.

```python
CONFIGS = ["baseline", "fourier", "decompose", "march", "all", "all+data"]

for cfg in CONFIGS:
    print(f"\n{'='*70}\n{cfg}\n{'='*70}", flush=True)
    rc = run("experiments/run_ablation.py", "--tag", "re100_v4",
             "--epochs", "20000", "--seeds", "2", "--only", cfg)
    save(f"ablation: {cfg} (Colab)")
```

`run_ablation.py` appends to `runs/ablation/summary.json` and **skips
configurations already present**, so re-running this cell after a disconnect
resumes rather than repeating.

## Cell 7 — final table

```python
import json
rows = json.load(open("runs/ablation/summary.json"))
print(f"{'config':<12s}{'seed':>5s}{'shedding':>11s}{'rel L2':>9s}{'min':>7s}")
for r in sorted(rows, key=lambda r: (r["config"], r["seed"])):
    print(f"{r['config']:<12s}{r['seed']:>5d}{100*r['shedding']:>10.1f}%"
          f"{r['rel_l2']:>9.4f}{r['minutes']:>7.1f}")
save("final ablation table (Colab)")
```

---

## Reading the result

The column that matters is **shedding retained** (wake-probe transverse-velocity
r.m.s., PINN vs LBM), not relative L2 — a PINN that has collapsed to steady flow
still scores a respectable L2, because the mean flow is most of the field.

`baseline` near **0 %** is the expected, published outcome and the foundation of
the report — not a bug. The question is which of `fourier` / `decompose` /
`march` lifts it, and whether any works alone.

## Notes

* **Token safety.** `push_results.py` never writes the token to `.git/config`
  and scrubs it from any error output, because Colab saves notebook outputs.
  Never `print(TOKEN)` or use `!git clone https://{TOKEN}@...` — the `!` magic
  echoes the command, token included.
* **What gets pushed.** `runs/` (JSON) and `figures/` (PNG) only. `data/` stays
  gitignored — the 190 MB dataset is regenerated, not stored.
* **Idle disconnects.** Free Colab drops idle sessions after ~90 min. Keep the
  tab open, or accept that Cell 6 resumes from the last pushed config.
* **Precision.** Everything defaults to float32 deliberately; `--f64` is for
  precision checks only and is far slower on a T4.
