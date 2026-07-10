#!/usr/bin/env python3
"""The frozen hidden-benchmark sampling protocol.

This module is the single source of truth for **how** a hidden benchmark is
sampled and sealed, so `build_benchmark.py` follows one documented, deterministic
rule and any rebuild is auditable. The full protocol is written up in
`docs/BENCHMARK_PROTOCOL.md`; this file is its executable form.

It is deliberately **pure** — no network, no encryption, no filesystem — so the
protocol can be unit-tested offline and reused by any auditor. `build_benchmark.py`
supplies the (impure) task loader, model pool, and encryption around it.

The protocol has five frozen parts (issue #14):

1. Sealed seed        — `SEALED_SEED`, fixed forever; determines the question set.
2. Split policy       — `SPLIT_ORDER` + `split_counts()`; disjoint eval/audit/live
                        slices taken in order from one deterministically sampled pool.
3. Sample counts      — per-benchmark counts (default 150/50/20) + a fixed margin.
4. Rebuild/integrity  — `manifest_hash()` pins the FULL protocol-relevant content
                        AND each item's split assignment, so a rebuild that reshuffles
                        the same questions across splits changes the hash.
5. Public metadata    — `build_manifest()` produces the committable, deterministic
                        audit record (hash, seed, per-split counts, id list).
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable, Iterable, Mapping

# --------------------------------------------------------------------------- #
# 1. Sealed seed — NEVER change after the first build.
# --------------------------------------------------------------------------- #
#: First 9 digits of e. Arbitrary but fixed forever: it fully determines which
#: questions are drawn and how they split. Changing it invalidates every score.
SEALED_SEED: int = 271828182

# --------------------------------------------------------------------------- #
# 2 & 3. Split policy and sample counts.
# --------------------------------------------------------------------------- #
#: The canonical split names, in the order they are carved from the pool. The
#: pool is sampled once; eval takes the first N_eval, audit the next N_audit,
#: live the next N_live — so the three sets are disjoint by construction.
SPLIT_ORDER: tuple[str, ...] = ("eval", "audit", "live")

#: Default per-split question counts. A benchmark may override via
#: ``_SPLIT_COUNT_OVERRIDES``; everything else uses this.
DEFAULT_SPLIT_COUNTS: Mapping[str, int] = {"eval": 150, "audit": 50, "live": 20}

#: Per-benchmark overrides (benchmark name -> {split: count}). Empty by default;
#: kept explicit so a count change is a visible, reviewable protocol edit.
_SPLIT_COUNT_OVERRIDES: Mapping[str, Mapping[str, int]] = {}

#: Extra tasks loaded beyond the total needed, so that items which fail to parse
#: or dedupe do not shrink a split below its count. Fixed so the pool size — and
#: therefore the selection — is deterministic.
SAMPLE_MARGIN: int = 50


def split_counts(benchmark: str) -> dict[str, int]:
    """Return the frozen per-split question counts for ``benchmark``.

    Falls back to :data:`DEFAULT_SPLIT_COUNTS` when the benchmark has no override.
    The returned dict is a fresh copy ordered by :data:`SPLIT_ORDER`.
    """
    key = (benchmark or "").strip().lower()
    counts = dict(_SPLIT_COUNT_OVERRIDES.get(key, DEFAULT_SPLIT_COUNTS))
    return {name: int(counts[name]) for name in SPLIT_ORDER}


def total_needed(counts: Mapping[str, int]) -> int:
    """Total questions across all splits (excludes :data:`SAMPLE_MARGIN`)."""
    return sum(int(counts[name]) for name in SPLIT_ORDER)


def pool_size(counts: Mapping[str, int]) -> int:
    """Number of tasks to sample into the pool before splitting (counts + margin)."""
    return total_needed(counts) + SAMPLE_MARGIN


# --------------------------------------------------------------------------- #
# Deterministic sampling + splitting.
# --------------------------------------------------------------------------- #
def sample_pool(
    load_tasks: Callable[..., list[Any]],
    benchmark: str,
    counts: Mapping[str, int],
    *,
    seed: int = SEALED_SEED,
) -> list[Any]:
    """Sample the deterministic, ordered task pool for ``benchmark``.

    ``load_tasks`` is the project's ``trinity.orchestration.dataset.load_tasks``
    (injected so this module stays free of dataset/network imports). It is called
    for the ``"train"`` split with ``max_items = pool_size * 3`` and the sealed
    ``seed``; the result is then shuffled once more with a ``seed``-seeded RNG and
    truncated to ``pool_size``. Both steps are seeded, so the pool is identical on
    every rebuild.

    Args:
        load_tasks: ``(benchmark, split, max_items, seed) -> list`` task loader.
        benchmark: Benchmark name.
        counts: Per-split counts (from :func:`split_counts`).
        seed: Sampling seed (defaults to the sealed seed).

    Returns:
        The ordered task pool, length ``min(pool_size, available)``.
    """
    size = pool_size(counts)
    tasks = list(load_tasks(benchmark, "train", max_items=size * 3, seed=seed))
    random.Random(seed).shuffle(tasks)
    return tasks[:size]


def select_splits(tasks: list[Any], counts: Mapping[str, int]) -> dict[str, list[Any]]:
    """Carve ``tasks`` into disjoint eval/audit/live splits, in :data:`SPLIT_ORDER`.

    Splits are contiguous, non-overlapping slices taken in order, so any given
    task lands in exactly one split. Raises if the pool is too small to satisfy
    the counts (a silent short split would corrupt the frozen protocol).

    Raises:
        ValueError: If ``tasks`` has fewer items than ``total_needed(counts)``.
    """
    need = total_needed(counts)
    if len(tasks) < need:
        raise ValueError(
            f"pool has {len(tasks)} tasks but the protocol needs {need} "
            f"({', '.join(f'{n}={counts[n]}' for n in SPLIT_ORDER)})"
        )
    splits: dict[str, list[Any]] = {}
    cursor = 0
    for name in SPLIT_ORDER:
        n = int(counts[name])
        splits[name] = tasks[cursor : cursor + n]
        cursor += n
    return splits


# --------------------------------------------------------------------------- #
# Stable identifiers (no builtin hash() — that is per-process randomised).
# --------------------------------------------------------------------------- #
def question_id(benchmark: str, index: int, prompt: str, existing: Any = None) -> str:
    """Return a stable question id, identical across processes and rebuilds.

    Prefers a task's own ``task_id`` (``existing``) when present — those are
    already deterministic (``math500-3``, ``mmlu-17``, ...). Otherwise derives one
    from ``sha256(benchmark, index, prompt)``; crucially it never uses the builtin
    ``hash()``, whose value depends on ``PYTHONHASHSEED`` and so differs run to run,
    which would make the audit hash non-reproducible.
    """
    if existing:
        return str(existing)
    digest = hashlib.sha256(f"{benchmark}\x1f{index}\x1f{prompt}".encode("utf-8")).hexdigest()
    return f"{(benchmark or 'q').strip().lower()}-{digest[:12]}"


#: Benchmark -> task-type classification, frozen as part of the protocol so a
#: rebuild labels items identically. Mirrors the reward module's families.
_MATH = frozenset({"math500", "math", "aime", "aime2025"})
_KNOWLEDGE = frozenset({"mmlu", "gpqa", "gpqa-diamond", "gpqa_diamond"})


def task_type(benchmark: str) -> str:
    """Classify a benchmark into the frozen item task-type ('math'|'knowledge'|'code')."""
    key = (benchmark or "").strip().lower()
    if key in _MATH:
        return "math"
    if key in _KNOWLEDGE:
        return "knowledge"
    return "code"


# --------------------------------------------------------------------------- #
# 4. Rebuild / integrity hash.
# --------------------------------------------------------------------------- #
#: Item fields that are part of the frozen identity of a benchmark. Cached model
#: answers/scores are deliberately excluded — they are populated by live API calls
#: and must not change the integrity hash.
_HASHED_ITEM_FIELDS: tuple[str, ...] = (
    "question_id",
    "benchmark",
    "task_type",
    "question_text",
    "correct_answer",
)


def _canonical_item(split: str, item: Mapping[str, Any]) -> dict[str, Any]:
    """Project one item down to its hash-relevant fields, tagged with its split."""
    canon: dict[str, Any] = {"split": split}
    for field in _HASHED_ITEM_FIELDS:
        canon[field] = item.get(field)
    return canon


def manifest_hash(splits: Mapping[str, Iterable[Mapping[str, Any]]]) -> str:
    """Return the SHA-256 audit hash over all items AND their split assignment.

    Unlike a hash over question text alone, this pins *which* questions are in
    *which* split: reshuffling the same pool across eval/audit/live changes the
    hash, so a rebuild is auditable down to the split. Items are canonicalised to
    :data:`_HASHED_ITEM_FIELDS`, tagged with their split, and sorted by
    ``(split rank, question_id)`` before hashing, so the result is invariant to
    input ordering but sensitive to content and placement.

    Args:
        splits: Mapping of split name -> iterable of item dicts.

    Returns:
        Hex SHA-256 digest.
    """
    rank = {name: i for i, name in enumerate(SPLIT_ORDER)}
    canon = [
        _canonical_item(split, item)
        for split in splits
        for item in splits[split]
    ]
    canon.sort(key=lambda c: (rank.get(c["split"], len(rank)), str(c["question_id"])))
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 5. Public metadata / manifest.
# --------------------------------------------------------------------------- #
def build_manifest(
    benchmark: str,
    splits: Mapping[str, list[Mapping[str, Any]]],
    *,
    seed: int = SEALED_SEED,
    pool_models: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the public, committable audit record for a built benchmark.

    Everything here except ``created_at`` and ``pool_models`` is a deterministic
    function of the sampled splits, so two rebuilds agree on ``content_hash``,
    per-split counts, and the full sorted ``question_ids`` per split.

    Args:
        benchmark: Benchmark name.
        splits: The built splits (split name -> item dicts).
        seed: The sampling seed used (recorded for the audit trail).
        pool_models: Model names whose answers were cached (informational).
        created_at: ISO timestamp; supplied by the caller (kept out of the hash).

    Returns:
        A JSON-serialisable manifest dict.
    """
    counts = {name: len(splits.get(name, [])) for name in SPLIT_ORDER}
    ids = {
        name: sorted(str(item.get("question_id")) for item in splits.get(name, []))
        for name in SPLIT_ORDER
    }
    return {
        "benchmark": benchmark,
        "seed": seed,
        "protocol_version": 1,
        "split_order": list(SPLIT_ORDER),
        "counts": counts,
        "content_hash": manifest_hash(splits),
        "question_ids": ids,
        "pool_models": list(pool_models or []),
        "created_at": created_at,
    }
