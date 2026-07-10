"""Concrete benchmark adapters for the four built-in benchmarks (issue #10).

Each benchmark the repo ships — ``math500``, ``mmlu``, ``gpqa``,
``livecodebench`` — gets its own :class:`~trinity.adapters.base.BenchmarkAdapter`
subclass, replacing the single parametrised ``DelegatingBenchmarkAdapter`` with a
named class per benchmark. The benchmark-specific *loading* logic lives in
:mod:`trinity.adapters.loaders`; the *scoring* stays in
:mod:`trinity.orchestration.reward`. The shared, benchmark-agnostic behaviour
(prompt passthrough, reward delegation, serialization) lives once on
:class:`_BuiltinAdapter`, so nothing is duplicated per benchmark — a subclass only
declares its ``name`` and :class:`~trinity.adapters.base.TaskType`.

These adapters import neither torch nor the dataset module: loading goes through
``loaders.load_split`` (which the back-compat ``dataset.load_tasks`` shim also
calls), so there is no import cycle and both paths share one loading contract.
"""
from __future__ import annotations

from typing import Any

from trinity.orchestration import reward as _reward
from trinity.types import Task

from . import loaders
from .base import BenchmarkAdapter, TaskType
from .registry import register_adapter

__all__ = [
    "Math500Adapter",
    "MmluAdapter",
    "GpqaAdapter",
    "LiveCodeBenchAdapter",
    "register_builtin_adapters",
]


class _BuiltinAdapter(BenchmarkAdapter):
    """Shared behaviour for the four shipped benchmarks.

    Subclasses set :attr:`name` and :attr:`_TASK_TYPE`. Loading delegates to
    :func:`trinity.adapters.loaders.load_split` (raw HF loader with offline toy
    fallback + deterministic shuffle/truncate); scoring delegates to the shared
    :func:`trinity.orchestration.reward.score_text`. Prompts are produced by the
    loaders (MCQ formatting, boxed-answer instructions, ...), so
    :meth:`build_prompt` returns the task prompt unchanged.
    """

    _TASK_TYPE: TaskType

    def load_tasks(self, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
        return loaders.load_split(self.name, split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        return _reward.score_text(self.name, output, reference)

    def task_type(self) -> TaskType:
        return self._TASK_TYPE

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": self._TASK_TYPE.value,
            "meta": dict(task.meta),
        }


class Math500Adapter(_BuiltinAdapter):
    """MATH-500: free-form answer graded by boxed/last-number extraction."""

    name = "math500"
    _TASK_TYPE = TaskType.MATH


class MmluAdapter(_BuiltinAdapter):
    """MMLU: single multiple-choice letter."""

    name = "mmlu"
    _TASK_TYPE = TaskType.MCQ


class GpqaAdapter(_BuiltinAdapter):
    """GPQA-Diamond: single multiple-choice letter (options shuffled per row)."""

    name = "gpqa"
    _TASK_TYPE = TaskType.MCQ


class LiveCodeBenchAdapter(_BuiltinAdapter):
    """LiveCodeBench: code executed against tests (pass@1)."""

    name = "livecodebench"
    _TASK_TYPE = TaskType.CODE


#: The concrete adapter classes, registered under their ``name``.
_BUILTIN_ADAPTERS: tuple[type[_BuiltinAdapter], ...] = (
    Math500Adapter,
    MmluAdapter,
    GpqaAdapter,
    LiveCodeBenchAdapter,
)


def register_builtin_adapters() -> None:
    """Register one concrete adapter per built-in benchmark.

    Skips names already registered so re-import (or a test that registered its
    own adapter for a built-in name) does not raise.
    """
    from .registry import is_registered

    for cls in _BUILTIN_ADAPTERS:
        if not is_registered(cls.name):
            register_adapter(cls.name, cls())
