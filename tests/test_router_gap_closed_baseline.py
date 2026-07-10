"""Property-style checks that routing_headroom and router_gap_closed agree on
one baseline, and that a non-positive denominator yields NaN (issue #33).

Mixing the full-K ``best_single`` with the cross-fit oracle produces nonsense
``router_gap_closed`` values (a router *below* the baseline can report a large
positive capture when the denominator silently goes negative). These tests pin
the invariant that both quantities are measured from the SAME cross-fit baseline
(``best_single_crossfit``) so they cannot drift apart, and that an undefined gap
is reported as NaN rather than a finite number.

Offline: exercises only the pure analysis math in ``scripts/oracle_ceiling.py``;
no API calls, GPU, or network.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the script as a module (it lives under scripts/, not the importable package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oracle_ceiling.py"
_spec = importlib.util.spec_from_file_location("oracle_ceiling", _SCRIPT)
oc = importlib.util.module_from_spec(_spec)
sys.modules["oracle_ceiling"] = oc
_spec.loader.exec_module(oc)


def _random_solves(seed: int) -> np.ndarray:
    """A random 0/1 solves tensor (Q, M, K) with per-(query, model) skill.

    Skill is drawn per (query, model) so some configurations have genuine
    routing headroom and others are near-noise; K>=5 keeps the cross-fit
    estimator in its reliable regime.
    """
    rng = np.random.default_rng(seed)
    q = int(rng.integers(6, 20))
    m = int(rng.integers(2, 5))
    k = 6
    skill = rng.random((q, m, 1))
    return (rng.random((q, m, k)) < skill).astype(float)


# --------------------------------------------------------------------------- #
# router_gap_closed: baseline / ceiling anchoring and the NaN guard.
# --------------------------------------------------------------------------- #
def test_gap_closed_nan_when_denominator_nonpositive():
    # Equal baseline and ceiling -> no achievable headroom -> undefined, not 0.
    assert math.isnan(oc.router_gap_closed(0.8, 0.8, 0.8))
    # Ceiling below baseline (mixed estimation regimes) must NOT silently flip
    # sign into a spurious positive capture -> NaN.
    assert math.isnan(oc.router_gap_closed(0.7, 0.8, 0.75))  # denom = -0.05
    # A genuine positive-headroom case returns the captured fraction.
    assert oc.router_gap_closed(0.85, 0.8, 0.9) == pytest.approx(0.5)


def test_gap_closed_is_linear_between_baseline_and_ceiling():
    baseline, ceiling = 0.4, 0.9
    assert oc.router_gap_closed(baseline, baseline, ceiling) == pytest.approx(0.0)
    assert oc.router_gap_closed(ceiling, baseline, ceiling) == pytest.approx(1.0)
    for frac in (0.25, 0.5, 0.75):
        acc = baseline + frac * (ceiling - baseline)
        assert oc.router_gap_closed(acc, baseline, ceiling) == pytest.approx(frac)


# --------------------------------------------------------------------------- #
# The crux of #33: headroom and gap_closed share ONE baseline. The denominator
# router_gap_closed divides by must reconstruct exactly the reported headroom
# from the SAME two stat fields; this test fails if the baselines ever diverge.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_headroom_is_the_gap_closed_denominator(seed):
    st = oc.compute_stats(_random_solves(seed), crossfit_splits=50, seed=0)

    # routing_headroom is measured from the cross-fit best_single, and that is the
    # exact baseline/ceiling router_gap_closed's denominator is built from.
    denom = st.routing_oracle - st.best_single_crossfit
    assert denom == pytest.approx(st.routing_headroom, abs=1e-12)
    # The oracle is floored at the cross-fit best_single, so the shared-baseline
    # denominator can never go negative (the sign-flip bug is impossible).
    assert denom >= -1e-9

    if st.routing_headroom > 1e-9:
        # A router sitting exactly on the shared baseline closes none of the gap;
        # one at the shared ceiling closes all of it.
        assert oc.router_gap_closed(
            st.best_single_crossfit, st.best_single_crossfit, st.routing_oracle
        ) == pytest.approx(0.0)
        assert oc.router_gap_closed(
            st.routing_oracle, st.best_single_crossfit, st.routing_oracle
        ) == pytest.approx(1.0)
    else:
        # No achievable headroom -> the capture ratio is undefined (NaN), never a
        # finite number, regardless of the router's accuracy.
        assert math.isnan(
            oc.router_gap_closed(0.99, st.best_single_crossfit, st.routing_oracle)
        )


def test_headroom_uses_crossfit_not_full_k_best_single():
    """Disjoint specialists: headroom is measured from the cross-fit best_single.

    Three disjoint specialists give oracle ~ 1.0 and best_single ~ 1/3, so the
    reported headroom must equal oracle - best_single_crossfit (not some other
    baseline). Guards against the two quantities being wired to different bases.
    """
    q, k = 30, 6
    S = np.zeros((q, 3, k))
    for i in range(q):
        S[i, i % 3, :] = 1.0
    st = oc.compute_stats(S, crossfit_splits=100, seed=0)
    assert st.routing_headroom == pytest.approx(st.routing_oracle - st.best_single_crossfit)
    assert st.routing_headroom == pytest.approx(2.0 / 3.0, abs=0.05)
