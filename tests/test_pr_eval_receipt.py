"""Gate 4 receipt validation: cross-check best_fitness against the PEAK series.

Regression test for the bug where `_validate_receipt` compared the receipt's
top-level `best_fitness` (the best candidate ever evaluated) against the maximum
of the per-generation *mean* fitness, rejecting honest receipts packed by
`scripts/pack_submission.py` (issue #27). No GPU / no network.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load_pr_eval():
    spec = importlib.util.spec_from_file_location("pr_eval", _REPO / "scripts" / "pr_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pr_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


# History rows in exactly build_receipt's shape: (mean, max, running_best).
# Means climb noisily 0.30 -> 0.60 (so the monotonicity/flat-line gates pass)
# while the best candidate ever reaches 0.72.
_HONEST_GENS = [
    (0.30, 0.50, 0.50), (0.42, 0.58, 0.58), (0.39, 0.61, 0.61),
    (0.55, 0.69, 0.69), (0.58, 0.66, 0.69), (0.60, 0.72, 0.72),
]


def _receipt(gens, best_fitness, *, cost=21.50):
    return {
        "total_cost_usd": cost,
        "generations": len(gens),
        "best_fitness": best_fitness,
        "fitness_history": [
            {"generation": i, "mean_fitness": m, "max_fitness": mx, "best_fitness": b}
            for i, (m, mx, b) in enumerate(gens)
        ],
    }


def test_honest_receipt_passes():
    """best_fitness 0.72 matches the peak series (max of max_fitness); mean peak is 0.60."""
    pe = _load_pr_eval()
    receipt = _receipt(_HONEST_GENS, best_fitness=0.72)
    assert pe._validate_receipt(receipt) is None


def test_fabricated_best_fitness_still_rejected():
    """A best_fitness well above the peak series is still caught (anti-fabrication intact)."""
    pe = _load_pr_eval()
    receipt = _receipt(_HONEST_GENS, best_fitness=0.95)  # peak series tops out at 0.72
    reason = pe._validate_receipt(receipt)
    assert reason is not None
    assert reason.startswith("receipt_best_fitness_mismatch")


def test_cross_check_uses_peak_not_mean():
    """The gate must compare against the peak (max_fitness), not the mean series.

    best_fitness sits above every mean (max mean = 0.60) but equals the top peak
    (0.72); on the old mean-based check this raised a false mismatch.
    """
    pe = _load_pr_eval()
    receipt = _receipt(_HONEST_GENS, best_fitness=0.72)
    mean_peak = max(m for m, _, _ in _HONEST_GENS)
    assert abs(receipt["best_fitness"] - mean_peak) > 0.1  # would have failed the old gate
    assert pe._validate_receipt(receipt) is None


def test_mean_based_shape_checks_still_apply():
    """Fixing the best_fitness cross-check must not disturb the curve-shape gates."""
    pe = _load_pr_eval()
    # First-generation mean too high -> still rejected on the means.
    hot_start = [(0.99, 0.99, 0.99), (0.30, 0.60, 0.99), (0.40, 0.70, 0.99)]
    reason = pe._validate_receipt(_receipt(hot_start, best_fitness=0.99))
    assert reason == "receipt_fitness_starts_too_high: 0.9900"

    # Flat-line means -> still rejected.
    flat = [(0.50, 0.72, 0.72)] * 5
    assert pe._validate_receipt(_receipt(flat, best_fitness=0.72)) == "receipt_fitness_flat_line"
