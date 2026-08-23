"""Guard for the relative-L2 metric and its call convention.

A transposed argument here does not raise -- it silently compares two grid
points instead of every velocity component, which looks like a plausible
error value and would quietly invalidate an entire results table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn.data import CylinderDataset, load  # noqa: E402


def test_relative_l2_is_exact_on_a_known_error():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(5000, 3))
    pred = true.copy()
    pred[:, :2] *= 1.10  # uniform +10 % on (u, v)
    got = CylinderDataset.relative_l2(pred, true)
    print(f"  rel_l2 on a uniform +10 % error = {got:.6f}")
    assert abs(got - 0.10) < 1e-9, got


def test_relative_l2_ignores_pressure():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(1000, 3))
    pred = true.copy()
    pred[:, 2] += 50.0  # enormous pressure error, no velocity error
    got = CylinderDataset.relative_l2(pred, true)
    print(f"  rel_l2 with pressure-only error = {got:.3e}  (must be ~0)")
    assert got < 1e-12, got


def test_snapshot_orientation_matches_prediction():
    """Real data: snapshot() must return (M, 3) so predictions line up."""
    try:
        ds = load("re100_v4")
    except FileNotFoundError:
        print("  skipped (no dataset present)")
        return
    coords, vals = ds.snapshot(0)
    assert coords.ndim == 2 and coords.shape[1] == 3, coords.shape
    assert vals.shape == coords.shape, (vals.shape, coords.shape)

    # The whole point: transposing must NOT quietly still work.
    n_correct = vals[..., :2].size
    n_transposed = vals.T[..., :2].size
    print(f"  correct orientation compares {n_correct} values, "
          f"transposed compares {n_transposed}")
    assert n_correct > n_transposed * 100


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
