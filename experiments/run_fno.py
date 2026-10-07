"""Phase 5 comparison: does physics buy rollout stability beyond the horizon?

Trains the variants of `src/fno/cylinder_fno.py` under identical settings --
same architecture, same optimiser, same seed, same data -- so the only
difference between rows is whether the loss carries a physics term. Writes
`runs/<run-name>/<variant>/results.json` and a `summary.json` alongside, and
skips variants already present, so an interrupted sweep resumes.

Usage:
    python experiments/run_fno.py --tag re100_v4 --epochs 200
    python experiments/run_fno.py --only data,physics --seeds 2
    python experiments/run_fno.py --quick          # CPU sanity check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "fno" / "cylinder_fno.py"

VARIANTS = ["data", "physics", "physics_div"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="re100_v4")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to check the path works -- runs_quick/, not a result")
    ap.add_argument("--extra", default="", help="extra flags passed to every run")
    ap.add_argument("--run-name", default="fno")
    args = ap.parse_args()

    names = args.only.split(",") if args.only else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}; choose from {VARIANTS}")

    outdir = ROOT / ("runs_quick" if args.quick else "runs") / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    done = {(r["variant"], r["seed"]) for r in summary}

    for name in names:
        for seed in range(args.seeds):
            if (name, seed) in done:
                print(f"[{name} seed {seed}] already done, skipping")
                continue
            run_dir = outdir / f"{name}_s{seed}"
            cmd = [sys.executable, "-u", str(TRAINER),
                   "--tag", args.tag, "--variant", name,
                   "--epochs", str(args.epochs), "--seed", str(seed),
                   "--out", str(run_dir)]
            if args.quick:
                cmd.append("--quick")
            if args.extra:
                cmd += args.extra.split()

            print(f"\n{'=' * 70}\n[{name} seed {seed}]\n{'=' * 70}", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=ROOT)
            if proc.returncode != 0:
                print(f"[{name} seed {seed}] FAILED (exit {proc.returncode})")
                continue

            blob = json.loads((run_dir / "results.json").read_text())
            summary.append({
                "variant": name, "seed": seed,
                "arch": blob["config"].get("arch", "fno"),
                "n_params": blob.get("n_params"),
                **blob["headline"],
                "epochs": args.epochs,
                "trained_rollout": blob["config"]["rollout"],
                "eval_steps": blob["config"]["eval_steps"],
                "minutes": (time.time() - t0) / 60.0,
            })
            summary_path.write_text(json.dumps(summary, indent=2))

    # ---- table -----------------------------------------------------------
    if not summary:
        return
    marks = [k for k in summary[0] if k.startswith("eps_full@")]
    marks.sort(key=lambda s: int(s.split("@")[1]))
    print("\n" + "=" * 70)
    print(f"{'variant':<14s} {'seed':>4s} " + " ".join(f"{m:>12s}" for m in marks)
          + f" {'min':>7s}")
    print("-" * 70)
    for r in summary:
        print(f"{r['variant']:<14s} {r['seed']:>4d} "
              + " ".join(f"{r.get(m, float('nan')):>12.4f}" for m in marks)
              + f" {r['minutes']:>7.1f}")
    print("=" * 70)
    trained = summary[0]["trained_rollout"]
    print("eps_full@N: relative L2 on (u, v) after N recursive steps, held-out ICs.")
    print(f"The trained horizon is {trained} steps -- the columns beyond it are the point.")
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
