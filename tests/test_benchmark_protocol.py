"""Frozen hidden-benchmark sampling protocol (issue #14).

Offline: exercises determinism, disjoint splits, stable ids, and the auditable
integrity hash without any network, GPU, or encryption. The task pool uses
`trinity.orchestration.dataset.load_tasks`, which falls back to the built-in toy
sets when `datasets`/network are unavailable.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "benchmark_protocol", _REPO / "scripts" / "benchmark_protocol.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_protocol"] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load_protocol()


def _fake_task(i: int, benchmark: str = "math500"):
    """A minimal task-like object (duck-typed: task_id / benchmark / prompt / answer)."""
    class _T:
        task_id = f"{benchmark}-{i}"
        prompt = f"question number {i}?"
        answer = str(i)
    _T.benchmark = benchmark
    return _T()


def _loader(pool):
    """A load_tasks stand-in returning a deterministic slice of `pool`."""
    def load_tasks(benchmark, split, max_items, seed=0):
        return list(pool[:max_items])
    return load_tasks


def _small_counts():
    # Small counts so the pool fits the toy/fake sets; margin still applies.
    return {"eval": 3, "audit": 2, "live": 1}


# --------------------------------------------------------------------------- #
# Counts / pool sizing
# --------------------------------------------------------------------------- #
def test_split_counts_default_and_order():
    c = P.split_counts("math500")
    assert list(c) == list(P.SPLIT_ORDER)          # ordered eval, audit, live
    assert c == {"eval": 150, "audit": 50, "live": 20}
    assert P.total_needed(c) == 220
    assert P.pool_size(c) == 220 + P.SAMPLE_MARGIN


# --------------------------------------------------------------------------- #
# Deterministic sampling + disjoint splits
# --------------------------------------------------------------------------- #
def test_sample_pool_is_deterministic():
    counts = _small_counts()
    pool = [_fake_task(i) for i in range(P.pool_size(counts))]
    a = P.sample_pool(_loader(pool), "math500", counts)
    b = P.sample_pool(_loader(pool), "math500", counts)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert len(a) == P.pool_size(counts)


def test_select_splits_disjoint_and_sized():
    counts = _small_counts()
    pool = [_fake_task(i) for i in range(P.pool_size(counts))]
    tasks = P.sample_pool(_loader(pool), "math500", counts)
    splits = P.select_splits(tasks, counts)
    assert [len(splits[n]) for n in P.SPLIT_ORDER] == [3, 2, 1]
    ids = [t.task_id for n in P.SPLIT_ORDER for t in splits[n]]
    assert len(ids) == len(set(ids))  # no task appears in two splits


def test_select_splits_raises_when_pool_too_small():
    counts = _small_counts()
    tasks = [_fake_task(i) for i in range(P.total_needed(counts) - 1)]
    try:
        P.select_splits(tasks, counts)
        assert False, "expected ValueError for undersized pool"
    except ValueError as exc:
        assert "needs" in str(exc)


# --------------------------------------------------------------------------- #
# Stable ids (no builtin hash())
# --------------------------------------------------------------------------- #
def test_question_id_prefers_existing():
    assert P.question_id("math500", 0, "p", existing="math500-7") == "math500-7"


def test_question_id_fallback_is_stable_and_not_builtin_hash():
    import hashlib

    got = P.question_id("mmlu", 4, "why is the sky blue?")
    # Deterministic and independent of PYTHONHASHSEED — recompute the spec.
    digest = hashlib.sha256("mmlu\x1f4\x1fwhy is the sky blue?".encode()).hexdigest()
    assert got == f"mmlu-{digest[:12]}"
    assert got == P.question_id("mmlu", 4, "why is the sky blue?")


def test_task_type_classification():
    assert P.task_type("math500") == "math"
    assert P.task_type("mmlu") == "knowledge"
    assert P.task_type("livecodebench") == "code"


# --------------------------------------------------------------------------- #
# Integrity hash: deterministic, order-invariant, split- and content-sensitive
# --------------------------------------------------------------------------- #
def _item(qid, split_bench="math500", text="t", answer="a"):
    return {
        "question_id": qid,
        "benchmark": split_bench,
        "task_type": "math",
        "question_text": text,
        "correct_answer": answer,
        "model_answers": {"m": "cached"},  # excluded from the hash
    }


def _splits():
    return {
        "eval": [_item("q1"), _item("q2")],
        "audit": [_item("q3")],
        "live": [_item("q4")],
    }


def test_manifest_hash_is_deterministic_and_order_invariant():
    s = _splits()
    h1 = P.manifest_hash(s)
    shuffled = {"eval": [s["eval"][1], s["eval"][0]], "audit": s["audit"], "live": s["live"]}
    assert P.manifest_hash(shuffled) == h1  # sorting makes input order irrelevant


def test_manifest_hash_ignores_cached_answers():
    s = _splits()
    h1 = P.manifest_hash(s)
    s["eval"][0]["model_answers"] = {"m": "DIFFERENT"}
    assert P.manifest_hash(s) == h1  # cached answers are not part of identity


def test_manifest_hash_pins_split_assignment():
    """Moving a question from eval to audit (same set) must change the hash."""
    s = _splits()
    h1 = P.manifest_hash(s)
    moved = {"eval": [s["eval"][0]], "audit": s["audit"] + [s["eval"][1]], "live": s["live"]}
    assert P.manifest_hash(moved) != h1


def test_manifest_hash_sensitive_to_content():
    s = _splits()
    h1 = P.manifest_hash(s)
    s["audit"][0]["correct_answer"] = "changed"
    assert P.manifest_hash(s) != h1


# --------------------------------------------------------------------------- #
# Public manifest
# --------------------------------------------------------------------------- #
def test_build_manifest_shape_and_determinism():
    import json

    s = _splits()
    m1 = P.build_manifest("math500", s, pool_models=["m"], created_at="2026-01-01T00:00:00Z")
    m2 = P.build_manifest("math500", s, pool_models=["m"], created_at="2026-01-01T00:00:00Z")
    assert m1 == m2
    assert m1["seed"] == P.SEALED_SEED
    assert m1["counts"] == {"eval": 2, "audit": 1, "live": 1}
    assert m1["content_hash"] == P.manifest_hash(s)
    assert m1["question_ids"]["eval"] == ["q1", "q2"]  # sorted
    json.dumps(m1)  # must be JSON-serialisable
