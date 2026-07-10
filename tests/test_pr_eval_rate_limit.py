"""Offline unit tests for the pr_eval rate-limit gate (anti-cheat Gate 1).

These tests exercise ONLY the pure timestamp / rate-limit helpers in
scripts/pr_eval.py. They make NO API calls and need no GPU/network (torch is
imported lazily inside pr_eval, never at module load).

Regression target: leaderboard timestamps are written in UTC (``time.gmtime`` +
trailing ``Z``), so they must be read back as UTC. Parsing them with
``time.mktime`` interprets the struct as *local* time and skews the epoch by the
host's UTC offset, silently shifting the 7-day rate-limit window on any non-UTC
maintainer box. The gate must behave identically regardless of the host timezone.
"""
import importlib.util
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Load the script as a module (it lives under scripts/, not the importable package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pr_eval.py"
_spec = importlib.util.spec_from_file_location("pr_eval", _SCRIPT)
pr_eval = importlib.util.module_from_spec(_spec)
sys.modules["pr_eval"] = pr_eval
_spec.loader.exec_module(pr_eval)


def _utc_stamp(epoch: float) -> str:
    """Render a Unix epoch as the leaderboard's UTC ``...Z`` string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def test_parse_utc_timestamp_is_true_utc():
    """A ``...Z`` stamp parses to the correct UTC epoch, independent of the impl."""
    # Independent ground truth via datetime (not the parser under test).
    expected = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert pr_eval._parse_utc_timestamp("2026-01-01T00:00:00Z") == expected


def test_parse_utc_timestamp_ignores_host_timezone():
    """The parse must not depend on the host TZ (the mktime bug did).

    On platforms with ``time.tzset`` (POSIX/CI), force a non-UTC zone and confirm
    the result is unchanged. On Windows (no ``tzset``) this degrades to the plain
    UTC-epoch assertion above.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset unavailable (Windows); covered by the UTC-epoch test")

    s = "2026-01-01T00:00:00Z"
    expected = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    saved_tz = os.environ.get("TZ")
    try:
        for tz in ("Asia/Tokyo", "America/Los_Angeles", "UTC"):
            os.environ["TZ"] = tz
            time.tzset()
            assert pr_eval._parse_utc_timestamp(s) == expected, f"TZ={tz}"
    finally:
        if saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved_tz
        time.tzset()


def test_parse_utc_timestamp_rejects_bad_input():
    assert pr_eval._parse_utc_timestamp("") is None
    assert pr_eval._parse_utc_timestamp("not-a-timestamp") is None
    assert pr_eval._parse_utc_timestamp("2026-01-01 00:00:00") is None  # missing Z/T


def _leaderboard_with(miner: str, ages_days: list[float]) -> dict:
    now = time.time()
    history = [
        {"miner": miner, "timestamp": _utc_stamp(now - d * 86400)}
        for d in ages_days
    ]
    return {"benchmarks": {"math500": {"history": history}}}


def test_recent_submission_is_rate_limited():
    """A submission one day ago blocks a new one (window is 7 days)."""
    lb = _leaderboard_with("alice", [1.0])
    err = pr_eval._check_rate_limit("alice", "math500", lb)
    assert err is not None and "rate_limited" in err


def test_old_submission_is_allowed():
    """A submission 10 days ago is outside the 7-day window."""
    lb = _leaderboard_with("alice", [10.0])
    assert pr_eval._check_rate_limit("alice", "math500", lb) is None


def test_other_miner_does_not_count():
    lb = _leaderboard_with("bob", [1.0])
    assert pr_eval._check_rate_limit("alice", "math500", lb) is None


def test_boundary_entry_counts_as_utc_regardless_of_host_tz():
    """An entry ~6 days old must count as recent on any host timezone.

    Under the old ``time.mktime`` parse, a host east of UTC reads the stamp as
    hours *older* than reality; this pins the behaviour to true UTC so the gate
    is deterministic across deploy environments.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset unavailable (Windows)")

    lb = _leaderboard_with("alice", [6.0])  # inside the 7-day window
    saved_tz = os.environ.get("TZ")
    try:
        for tz in ("Asia/Tokyo", "America/Los_Angeles", "UTC"):
            os.environ["TZ"] = tz
            time.tzset()
            err = pr_eval._check_rate_limit("alice", "math500", lb)
            assert err is not None and "rate_limited" in err, f"TZ={tz}"
    finally:
        if saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved_tz
        time.tzset()
