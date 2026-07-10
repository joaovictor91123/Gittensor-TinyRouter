"""Benchmark adapter registry — the unified seam for the evaluation pipeline.

Importing this package registers the built-in adapters, so a caller only needs::

    from trinity.adapters import get_adapter

    adapter = get_adapter("math500")
    tasks = adapter.load_tasks("test", max_items=100, seed=42)
    score = adapter.score_output(model_output, tasks[0].answer)

The core evaluator resolves a benchmark to an adapter once and then drives the
adapter's interface, so it never branches on the benchmark name. Adding a new
benchmark means implementing :class:`BenchmarkAdapter` and registering it —
nothing in the shared evaluator changes. See :mod:`trinity.adapters.base` for
the interface and :mod:`trinity.adapters.registry` for the lookup.
"""
from __future__ import annotations

from .base import BenchmarkAdapter, TaskType
from .benchmarks import (
    GpqaAdapter,
    LiveCodeBenchAdapter,
    Math500Adapter,
    MmluAdapter,
    register_builtin_adapters,
)
from .builtin import DelegatingBenchmarkAdapter
from .mmlu_pro import MmluProAdapter, register_mmlu_pro_adapter
from .registry import (
    available_adapters,
    clear_registry,
    get_adapter,
    is_registered,
    register_adapter,
)
from .swebench import PatchReference, SweBenchAdapter, register_swebench_adapter

__all__ = [
    "BenchmarkAdapter",
    "TaskType",
    "DelegatingBenchmarkAdapter",
    "Math500Adapter",
    "MmluAdapter",
    "GpqaAdapter",
    "LiveCodeBenchAdapter",
    "SweBenchAdapter",
    "PatchReference",
    "MmluProAdapter",
    "register_adapter",
    "get_adapter",
    "is_registered",
    "available_adapters",
    "clear_registry",
    "register_builtin_adapters",
    "register_swebench_adapter",
    "register_mmlu_pro_adapter",
]

# Register the built-in benchmarks on import so `get_adapter("math500")` works
# without the caller having to bootstrap the registry.
register_builtin_adapters()
register_swebench_adapter()
register_mmlu_pro_adapter()
