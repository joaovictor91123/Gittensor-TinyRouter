"""Generic delegating adapter + built-in registration.

The four shipped benchmarks (``math500``, ``mmlu``, ``gpqa``, ``livecodebench``)
now each have a dedicated :class:`~trinity.adapters.base.BenchmarkAdapter`
subclass in :mod:`trinity.adapters.benchmarks` (issue #10); this module keeps the
generic :class:`DelegatingBenchmarkAdapter` — a name-parametrised wrapper still
useful for ad-hoc registration and tests — and re-exports the built-in
registration entry point.

:class:`DelegatingBenchmarkAdapter` routes:

* loading to :func:`trinity.adapters.loaders.load_split` (the same raw-or-toy +
  deterministic shuffle/truncate path the concrete adapters use), so it takes no
  dependency on the dataset module and cannot form an import cycle;
* scoring to :func:`trinity.orchestration.reward.score_text`;
* its task-type family to ``reward.py``'s own dispatch frozensets, so the
  taxonomy cannot drift from the scorer.
"""
from __future__ import annotations

from typing import Any

from trinity.orchestration import reward as _reward
from trinity.types import Task

from . import loaders
from .base import BenchmarkAdapter, TaskType
from .benchmarks import register_builtin_adapters

__all__ = ["DelegatingBenchmarkAdapter", "register_builtin_adapters"]


def _task_type_for(benchmark: str) -> TaskType:
    """Classify a benchmark using ``reward.py``'s own dispatch tables.

    Reusing the scorer's frozensets (rather than a second hand-kept map) means a
    benchmark can never be classified one way here and scored another way there.
    """
    key = (benchmark or "").strip().lower()
    if key in _reward.MATH_BENCHMARKS:
        return TaskType.MATH
    if key in _reward.CHOICE_BENCHMARKS:
        return TaskType.MCQ
    if key in _reward.CODE_BENCHMARKS:
        return TaskType.CODE
    raise ValueError(
        f"Benchmark {benchmark!r} has no known task type; add it to the "
        f"reward.py dispatch tables or give it a dedicated adapter."
    )


class DelegatingBenchmarkAdapter(BenchmarkAdapter):
    """Adapter that routes every call to the shared dataset/reward pipeline.

    This is the faithful wrapper for benchmarks already served end-to-end by
    ``dataset.load_tasks`` + ``reward.score_text``. Prompts are produced by the
    loaders (which already format MCQ options, boxed-answer instructions, etc.),
    so :meth:`build_prompt` simply returns the task's prompt.
    """

    def __init__(self, name: str):
        self.name = (name or "").strip().lower()
        if not self.name:
            raise ValueError("DelegatingBenchmarkAdapter requires a benchmark name.")
        # Validate up front so a misconfigured registration fails at import, not
        # deep inside an eval run.
        self._task_type = _task_type_for(self.name)

    def load_tasks(
        self,
        split: str,
        max_items: int | None,
        seed: int = 0,
    ) -> list[Task]:
        return loaders.load_split(self.name, split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        return _reward.score_text(self.name, output, reference)

    def task_type(self) -> TaskType:
        return self._task_type

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": self._task_type.value,
            "meta": dict(task.meta),
        }
