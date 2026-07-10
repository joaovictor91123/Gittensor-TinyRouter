"""Concrete per-benchmark adapters (issue #10).

Verifies the four shipped benchmarks are each served by a dedicated
:class:`BenchmarkAdapter` subclass resolved through the registry, and that the
refactor is behavior-preserving vs the ``dataset.load_tasks`` entry point.
Offline via the toy fallback.
"""
from __future__ import annotations

import pytest

from trinity.adapters import (
    GpqaAdapter,
    LiveCodeBenchAdapter,
    Math500Adapter,
    MmluAdapter,
    TaskType,
    get_adapter,
)
from trinity.adapters import loaders as _loaders
from trinity.orchestration.dataset import SUPPORTED_BENCHMARKS, load_tasks

_EXPECTED = {
    "math500": (Math500Adapter, TaskType.MATH),
    "mmlu": (MmluAdapter, TaskType.MCQ),
    "gpqa": (GpqaAdapter, TaskType.MCQ),
    "livecodebench": (LiveCodeBenchAdapter, TaskType.CODE),
}


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_each_benchmark_has_dedicated_adapter_class(name):
    cls, ttype = _EXPECTED[name]
    adapter = get_adapter(name)
    assert isinstance(adapter, cls)
    assert adapter.name == name
    assert adapter.task_type() is ttype


def test_every_supported_benchmark_is_registered():
    assert set(_EXPECTED) == set(SUPPORTED_BENCHMARKS)


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_load_tasks_matches_registry_path(name):
    """dataset.load_tasks (registry shim) == the adapter's own load_tasks."""
    via_shim = load_tasks(name, "test", max_items=2, seed=7)
    via_adapter = get_adapter(name).load_tasks("test", max_items=2, seed=7)
    assert [t.task_id for t in via_shim] == [t.task_id for t in via_adapter]
    for t in via_shim:
        assert t.benchmark == name


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_load_tasks_matches_raw_loader(name):
    """The refactor is behavior-preserving vs the moved raw loader path."""
    via_adapter = get_adapter(name).load_tasks("test", max_items=3, seed=11)
    via_loader = _loaders.load_split(name, "test", 3, seed=11)
    assert [t.task_id for t in via_adapter] == [t.task_id for t in via_loader]


def test_scoring_still_binary():
    assert get_adapter("math500").score_output("\\boxed{4}", "4") == 1.0
    assert get_adapter("math500").score_output("\\boxed{5}", "4") == 0.0
    assert get_adapter("mmlu").score_output("The answer is B.", "B") == 1.0


def test_serialize_task_uses_adapter_task_type():
    adapter = get_adapter("gpqa")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    item = adapter.serialize_task(task)
    assert item["benchmark"] == "gpqa"
    assert item["task_type"] == TaskType.MCQ.value
    assert item["reference"] == task.answer


def test_no_duplicated_load_logic():
    """The concrete adapters share one loading path (loaders.load_split), so all
    four resolve to the same underlying function object, not per-benchmark copies."""
    fns = {get_adapter(n).load_tasks.__func__ for n in _EXPECTED}
    assert len(fns) == 1  # one shared _BuiltinAdapter.load_tasks
