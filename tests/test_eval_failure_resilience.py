"""One failed trajectory must not discard a whole evaluation.

`optim/fitness.py` already gathers with ``return_exceptions=True`` so a trajectory
that exhausts its retries degrades to reward 0 instead of crashing training (see
docs/JOURNAL.md, 2026-06-22). `eval.py` never got the same treatment: a single
transient timeout across ~100 tasks x N reps aborted the entire eval, discarding
the TRINITY number *and* every baseline already computed.

These tests use fakes only -- no network.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from trinity import eval as trinity_eval
from trinity.eval import _mean_scoring_failures_as_zero


class _Boom(RuntimeError):
    """Stands in for a retry-exhausted httpx.ReadTimeout."""


@pytest.fixture(autouse=True)
def _stub_httpx(monkeypatch):
    """`eval.py` opens an `httpx.AsyncClient`; these tests never issue requests."""

    class _AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    module = types.ModuleType("httpx")
    module.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", module)


# --------------------------------------------------------------------------- #
# _mean_scoring_failures_as_zero
# --------------------------------------------------------------------------- #
def test_all_successes_is_a_plain_mean():
    assert _mean_scoring_failures_as_zero([1.0, 0.0, 1.0, 1.0], float, "x") == 0.75


def test_failures_score_zero_and_stay_in_the_denominator():
    """3 good (all correct) + 1 failed -> 3/4, not 3/3."""
    outcomes = [1.0, 1.0, 1.0, _Boom("timeout")]
    assert _mean_scoring_failures_as_zero(outcomes, float, "x") == 0.75


def test_a_single_failure_does_not_raise():
    outcomes = [1.0, _Boom("timeout")]
    assert _mean_scoring_failures_as_zero(outcomes, float, "x") == 0.5


def test_total_failure_raises_rather_than_reporting_zero():
    """A dead API must not masquerade as a score of 0.0."""
    outcomes = [_Boom("timeout"), _Boom("timeout")]
    with pytest.raises(RuntimeError, match="all 2 tasks failed"):
        _mean_scoring_failures_as_zero(outcomes, float, "trinity")


def test_total_failure_message_names_the_condition_and_first_error():
    with pytest.raises(RuntimeError, match="single::gpt-x"):
        _mean_scoring_failures_as_zero([_Boom("boom")], float, "single::gpt-x")


def test_empty_task_list_raises():
    with pytest.raises(RuntimeError, match="no tasks to score"):
        _mean_scoring_failures_as_zero([], float, "trinity")


def test_failure_count_is_reported(capsys):
    _mean_scoring_failures_as_zero([1.0, _Boom("t")], float, "trinity")
    assert "1/2 tasks failed" in capsys.readouterr().out


def test_no_warning_when_nothing_failed(capsys):
    _mean_scoring_failures_as_zero([1.0, 0.0], float, "trinity")
    assert "failed" not in capsys.readouterr().out


def test_scorer_is_applied_to_successful_outcomes():
    """The score callable, not the raw outcome, decides the value."""
    outcomes = ["yes", "no", _Boom("t")]
    score = lambda o: 1.0 if o == "yes" else 0.0  # noqa: E731
    assert _mean_scoring_failures_as_zero(outcomes, score, "x") == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# _score_policy: the regression
# --------------------------------------------------------------------------- #
class _Adapter:
    def score_trajectory(self, traj):
        return float(traj)

    def build_prompt(self, task):
        return str(task)

    def score_output(self, text, answer):
        return 1.0 if text == answer else 0.0


class _Task:
    def __init__(self, tid):
        self.task_id = tid
        self.answer = "ok"


def _patch_run_trajectory(monkeypatch, fail_on):
    async def fake_run_trajectory(task, *a, **kw):
        if task.task_id in fail_on:
            raise _Boom(f"retries exhausted on {task.task_id}")
        return 1.0

    monkeypatch.setattr(trinity_eval, "run_trajectory", fake_run_trajectory)


def test_score_policy_survives_one_failed_trajectory(monkeypatch):
    """The regression: previously this raised and the whole eval died."""
    _patch_run_trajectory(monkeypatch, fail_on={"t2"})
    tasks = [_Task(f"t{i}") for i in range(4)]

    score = asyncio.run(
        trinity_eval._score_policy(
            tasks, policy=None, pool=None, pool_models=[],
            adapter=_Adapter(), sample=False,
        )
    )

    # t0, t1, t3 scored 1.0; t2 failed -> 0.0
    assert score == 0.75


def test_score_policy_is_unaffected_when_nothing_fails(monkeypatch):
    _patch_run_trajectory(monkeypatch, fail_on=set())
    tasks = [_Task(f"t{i}") for i in range(3)]

    score = asyncio.run(
        trinity_eval._score_policy(
            tasks, policy=None, pool=None, pool_models=[],
            adapter=_Adapter(), sample=False,
        )
    )

    assert score == 1.0


def test_score_policy_raises_when_every_trajectory_fails(monkeypatch):
    _patch_run_trajectory(monkeypatch, fail_on={"t0", "t1"})
    tasks = [_Task(f"t{i}") for i in range(2)]

    with pytest.raises(RuntimeError, match="all 2 tasks failed"):
        asyncio.run(
            trinity_eval._score_policy(
                tasks, policy=None, pool=None, pool_models=[],
                adapter=_Adapter(), sample=False,
            )
        )


# --------------------------------------------------------------------------- #
# _score_single_model: same treatment, so baselines survive too
# --------------------------------------------------------------------------- #
class _Result:
    def __init__(self, text):
        self.text = text


class _Pool:
    def __init__(self, fail_on):
        self.fail_on = fail_on

    async def chat(self, model, msgs, **kw):
        prompt = msgs[-1]["content"] if isinstance(msgs[-1], dict) else str(msgs[-1])
        if any(f in prompt for f in self.fail_on):
            raise _Boom("retries exhausted")
        return _Result("ok")


def test_score_single_model_survives_one_failed_task(monkeypatch):
    monkeypatch.setattr(
        "trinity.roles.prompts.build_messages",
        lambda role, prompt, hist: [{"role": "user", "content": prompt}],
    )
    tasks = [_Task(f"t{i}") for i in range(4)]
    # _Adapter.build_prompt returns str(task); make one prompt identifiable.
    monkeypatch.setattr(_Adapter, "build_prompt", lambda self, t: t.task_id)

    score = asyncio.run(
        trinity_eval._score_single_model(
            tasks, _Pool(fail_on={"t2"}), "m", _Adapter(),
            max_tokens=16, reasoning=None,
        )
    )

    assert score == 0.75
