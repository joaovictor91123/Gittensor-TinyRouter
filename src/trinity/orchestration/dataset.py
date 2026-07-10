"""Benchmark task sampling for TRINITY training/eval.

Turns benchmark datasets into canonical :class:`trinity.types.Task` objects and
samples minibatches for the CMA inner loop.

The per-benchmark dataset parsing moved behind the benchmark-adapter interface
(issue #10): it now lives in :mod:`trinity.adapters.loaders` and is reached
through the registry. :func:`load_tasks` is kept as the stable, back-compatible
entry point every script/test already imports; it resolves the benchmark to its
registered adapter and asks it to load, so there is exactly one loading path.

Public API
----------
- ``load_tasks(benchmark, split, max_items, seed=0) -> list[Task]``
- ``sample_minibatch(tasks, m, rng) -> list[Task]``
- ``SUPPORTED_BENCHMARKS`` (tuple[str, ...])
"""
from __future__ import annotations

import random

from trinity.types import Task

__all__ = ["load_tasks", "sample_minibatch", "SUPPORTED_BENCHMARKS"]

SUPPORTED_BENCHMARKS: tuple[str, ...] = (
    "math500",
    "mmlu",
    "gpqa",
    "livecodebench",
)


def load_tasks(
    benchmark: str,
    split: str,
    max_items: int | None,
    seed: int = 0,
) -> list[Task]:
    """Load a benchmark as a deterministic list of :class:`Task`.

    Resolves ``benchmark`` to its registered adapter and delegates loading to it
    (the adapter's loader tries HuggingFace ``datasets`` and falls back to a tiny
    offline toy set, then applies a ``seed``-seeded shuffle and truncates to
    ``max_items``). Repeated calls with identical arguments yield identical lists.

    Raises
    ------
    ValueError
        If ``benchmark`` is not a registered benchmark.
    """
    # Imported lazily so importing this module never triggers adapter import at
    # module load (and so the adapters package can import cleanly on its own).
    from trinity.adapters import get_adapter

    try:
        adapter = get_adapter(benchmark)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    return adapter.load_tasks(split, max_items, seed=seed)


def sample_minibatch(
    tasks: list[Task],
    m: int,
    rng: random.Random,
) -> list[Task]:
    """Sample ``m`` distinct task instances for one CMA candidate evaluation.

    Per SPEC §5.2 each of the ``m_CMA`` replications uses a different randomly
    sampled task instance (a minibatch of distinct problems per candidate,
    re-sampled per iteration). Sampling is *without replacement* when enough
    tasks exist, otherwise it falls back to sampling *with replacement* so a tiny
    toy set still yields a full minibatch for smoke tests.

    Parameters
    ----------
    tasks:
        The pool of tasks to draw from (typically the training split).
    m:
        Number of instances to draw (``m_CMA``, e.g. 16).
    rng:
        Caller-owned :class:`random.Random` so the optimizer controls determinism
        (e.g. re-seeded per CMA iteration).

    Returns
    -------
    list[Task]
        ``m`` sampled tasks (distinct where possible).

    Raises
    ------
    ValueError
        If ``tasks`` is empty or ``m`` is not positive.
    """
    if not tasks:
        raise ValueError("Cannot sample a minibatch from an empty task list.")
    if m <= 0:
        raise ValueError(f"Minibatch size m must be positive, got {m}.")

    if m <= len(tasks):
        return rng.sample(tasks, m)
    # Not enough distinct tasks (toy set): sample with replacement.
    return [rng.choice(tasks) for _ in range(m)]
