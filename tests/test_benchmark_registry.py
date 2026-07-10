"""Tests for the benchmark adapter registry (issue #9).

All offline: task loading falls back to the built-in toy sets in
``dataset.py`` when ``datasets``/network are unavailable, so these run with zero
network exactly like the other smoke tests.
"""
from __future__ import annotations

import pytest

from trinity.adapters import (
    BenchmarkAdapter,
    DelegatingBenchmarkAdapter,
    TaskType,
    available_adapters,
    clear_registry,
    get_adapter,
    is_registered,
    register_adapter,
    register_builtin_adapters,
    register_mmlu_pro_adapter,
    register_swebench_adapter,
)
from trinity.orchestration.dataset import SUPPORTED_BENCHMARKS

# Known-correct / known-wrong pairs per built-in benchmark, mirroring the toy
# sets the offline loader returns.
_SCORING_CASES = {
    "math500": ("The answer is \\boxed{4}.", "\\boxed{9}.", "4"),
    "mmlu": ("The answer is B.", "The answer is A.", "B"),
    "gpqa": ("Answer: B", "Answer: C", "B"),
}


def test_builtins_registered_for_every_supported_benchmark():
    for name in SUPPORTED_BENCHMARKS:
        assert is_registered(name)
        assert isinstance(get_adapter(name), BenchmarkAdapter)
    # The registry exposes at least the supported benchmarks; additional adapters
    # (e.g. the SWE-bench Verified (#17) and MMLU-Pro (#12) adapters) may also be
    # registered.
    assert set(SUPPORTED_BENCHMARKS) <= set(available_adapters())


def test_lookup_is_case_insensitive():
    assert get_adapter("MATH500") is get_adapter("math500")


def test_get_adapter_unknown_lists_available():
    with pytest.raises(KeyError) as excinfo:
        get_adapter("does-not-exist")
    assert "math500" in str(excinfo.value)


def test_task_type_matches_family():
    assert get_adapter("math500").task_type() is TaskType.MATH
    assert get_adapter("mmlu").task_type() is TaskType.MCQ
    assert get_adapter("gpqa").task_type() is TaskType.MCQ
    assert get_adapter("livecodebench").task_type() is TaskType.CODE


def test_load_tasks_deterministic_and_capped():
    adapter = get_adapter("math500")
    a = adapter.load_tasks("test", max_items=2, seed=42)
    b = adapter.load_tasks("test", max_items=2, seed=42)
    assert len(a) <= 2
    assert [t.task_id for t in a] == [t.task_id for t in b]
    for t in a:
        assert t.benchmark == "math500"


@pytest.mark.parametrize("name", list(_SCORING_CASES))
def test_score_output_binary_correct_and_wrong(name):
    adapter = get_adapter(name)
    right, wrong, ref = _SCORING_CASES[name]
    assert adapter.score_output(right, ref) == 1.0
    assert adapter.score_output(wrong, ref) == 0.0


def test_score_output_code_runs_tests():
    adapter = get_adapter("livecodebench")
    spec = {
        "tests": [{"input": "3\n", "output": "9"}, {"input": "5\n", "output": "25"}],
        "fn_name": None,
        "starter_code": None,
    }
    good = "n = int(input())\nprint(n * n)\n"
    bad = "n = int(input())\nprint(n + 1)\n"
    assert adapter.score_output(good, spec) == 1.0
    assert adapter.score_output(bad, spec) == 0.0


def test_serialize_task_is_json_safe():
    import json

    adapter = get_adapter("mmlu")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    item = adapter.serialize_task(task)
    assert item["task_id"] == task.task_id
    assert item["benchmark"] == "mmlu"
    assert item["task_type"] == TaskType.MCQ.value
    assert item["reference"] == task.answer
    # Must round-trip through JSON (frozen hidden-benchmark item format).
    assert json.loads(json.dumps(item))["task_id"] == task.task_id


def test_cache_baselines_default_is_noop():
    adapter = get_adapter("math500")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    assert adapter.cache_baselines(task, pool=None) is None


def test_delegating_adapter_rejects_unknown_benchmark():
    with pytest.raises(ValueError):
        DelegatingBenchmarkAdapter("totally-unknown-benchmark")


def test_register_duplicate_name_raises():
    with pytest.raises(ValueError):
        register_adapter("math500", DelegatingBenchmarkAdapter("math500"))


def test_score_trajectory_matches_reward_score_for_delegating_adapter():
    """The routed path (score_trajectory) must equal the legacy reward.score."""
    from trinity.orchestration import reward
    from trinity.types import Task, Trajectory

    adapter = get_adapter("math500")
    task = Task(task_id="t", benchmark="math500", prompt="2+2?", answer="4")
    traj = Trajectory(task=task, final_answer="The answer is \\boxed{4}.")
    assert adapter.score_trajectory(traj) == reward.score(traj) == 1.0

    traj_wrong = Trajectory(task=task, final_answer="The answer is \\boxed{9}.")
    assert adapter.score_trajectory(traj_wrong) == reward.score(traj_wrong) == 0.0


def test_score_trajectory_honors_custom_score_output():
    """A custom score_output is used on the multi-turn path, not just single-turn."""
    from trinity.types import Task, Trajectory

    class _AlwaysZero(DelegatingBenchmarkAdapter):
        def score_output(self, output, reference):
            return 0.0  # deliberately disagrees with the real scorer

    adapter = _AlwaysZero("math500")
    task = Task(task_id="t", benchmark="math500", prompt="2+2?", answer="4")
    # final_answer is objectively correct; the custom scorer must still win.
    traj = Trajectory(task=task, final_answer="The answer is \\boxed{4}.")
    assert adapter.score_trajectory(traj) == 0.0


def test_build_prompt_drives_routed_trajectory():
    """build_prompt is the real prompt seam: run_trajectory must present it to
    both the coordinator (policy) and the pool model, not raw task.prompt."""
    import asyncio

    from trinity.orchestration.session import run_trajectory
    from trinity.types import Role, Task

    marker = "ZZ_UNIQUE_PROMPT_MARKER_ZZ"

    class _MarkerAdapter(DelegatingBenchmarkAdapter):
        def build_prompt(self, task):
            return f"{marker} :: {task.prompt}"

    seen = {"policy_text": None, "pool_messages": None}

    class _StubPolicy:
        def decide(self, transcript_text, *, sample, rng=None):
            seen["policy_text"] = transcript_text
            return 0, Role.WORKER

    class _Res:
        text = "\\boxed{4}"
        prompt_tokens = 0
        completion_tokens = 0

    class _StubPool:
        async def chat(self, model, messages, **kwargs):
            seen["pool_messages"] = messages
            return _Res()

    adapter = _MarkerAdapter("math500")
    task = Task(task_id="t", benchmark="math500", prompt="What is 2+2?", answer="4")
    asyncio.run(
        run_trajectory(task, _StubPolicy(), _StubPool(), ["m0"],
                       max_turns=1, adapter=adapter)
    )
    # The coordinator saw the adapter-rendered query...
    assert marker in seen["policy_text"]
    # ...and so did the pool model (marker appears somewhere in the messages).
    assert marker in "".join(m.get("content", "") for m in seen["pool_messages"])


def test_decorator_registration_and_registry_isolation():
    """A throwaway registration on a cleared registry, then restored built-ins."""
    try:
        clear_registry()
        assert available_adapters() == ()

        @register_adapter("unit-test-bench")
        class _StubAdapter(BenchmarkAdapter):
            name = "unit-test-bench"

            def load_tasks(self, split, max_items, seed=0):
                return []

            def build_prompt(self, task):
                return task.prompt

            def score_output(self, output, reference):
                return 1.0 if output == reference else 0.0

            def task_type(self):
                return TaskType.MATH

            def serialize_task(self, task):
                return {"task_id": task.task_id}

        assert is_registered("unit-test-bench")
        assert get_adapter("unit-test-bench").score_output("x", "x") == 1.0
    finally:
        # Never leak test state into other tests: restore the real registry
        # (the built-in benchmarks plus the SWE-bench (#17) and MMLU-Pro (#12)
        # adapters).
        clear_registry()
        register_builtin_adapters()
        register_swebench_adapter()
        register_mmlu_pro_adapter()

    assert set(SUPPORTED_BENCHMARKS) <= set(available_adapters())
