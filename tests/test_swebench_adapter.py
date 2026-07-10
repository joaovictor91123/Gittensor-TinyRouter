"""SWE-bench Verified adapter + patch-task schema (issue #17).

Offline: the loader falls back to a built-in toy patch task when
``datasets``/network are unavailable, so these run with zero network.
"""
from __future__ import annotations

from trinity.adapters import TaskType, get_adapter
from trinity.adapters.swebench import (
    BENCHMARK,
    PatchReference,
    SweBenchAdapter,
    build_patch_prompt,
    normalize_patch,
    score_patch,
)


def test_registered_under_canonical_name():
    adapter = get_adapter(BENCHMARK)
    assert isinstance(adapter, SweBenchAdapter)
    assert adapter.name == "swebench_verified"
    assert adapter.task_type() is TaskType.PATCH


def test_load_tasks_returns_normalized_patch_tasks():
    adapter = get_adapter(BENCHMARK)
    tasks = adapter.load_tasks("test", max_items=5, seed=0)
    assert tasks, "toy fallback should yield at least one task"
    t = tasks[0]
    assert t.benchmark == "swebench_verified"
    # Acceptance: task objects include repo/commit/instance metadata.
    assert t.meta["repo"]
    assert t.meta["base_commit"]
    assert t.meta["instance_id"] == t.task_id
    assert t.meta["task_type"] == TaskType.PATCH.value
    # reference is the structured patch schema.
    ref = PatchReference.from_dict(t.answer)
    assert ref.is_valid()
    assert ref.gold_patch


def test_load_tasks_deterministic_and_capped():
    adapter = get_adapter(BENCHMARK)
    a = adapter.load_tasks("test", max_items=1, seed=3)
    b = adapter.load_tasks("test", max_items=1, seed=3)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert len(a) <= 1


def test_prompt_requests_unified_diff():
    prompt = build_patch_prompt("Something is broken.", "octo/calc", "abc123", hints="try X")
    assert "octo/calc" in prompt
    assert "abc123" in prompt
    assert "Something is broken." in prompt
    assert "try X" in prompt
    assert "diff --git" in prompt  # asks for an applyable patch


def test_patch_reference_roundtrip_and_validation():
    ref = PatchReference(
        repo="a/b",
        base_commit="deadbeef",
        gold_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::x"],
    )
    assert ref.is_valid()
    assert PatchReference.from_dict(ref.to_dict()) == ref
    # missing tests -> not usable
    assert not PatchReference(repo="a/b", base_commit="c", gold_patch="p").is_valid()


def test_normalize_patch_strips_noise():
    a = (
        "```diff\n"
        "diff --git a/f.py b/f.py\n"
        "index 111..222 100644\n"
        "@@ -1,3 +1,3 @@ def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "```"
    )
    b = (
        "diff --git a/f.py b/f.py\n"
        "index 999..aaa 100644\n"          # different blob hashes
        "@@ -10,4 +10,4 @@ other offset\n"  # different line offsets
        "-    return 1\n"
        "+    return 2\n"
    )
    # Index/offset noise and fences are stripped, so the two compare equal.
    assert normalize_patch(a) == normalize_patch(b)


def test_score_output_exact_match_only():
    adapter = get_adapter(BENCHMARK)
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    gold = task.answer["gold_patch"]
    # The gold patch scores 1.0; an unrelated patch scores 0.0.
    assert adapter.score_output(gold, task.answer) == 1.0
    assert adapter.score_output("diff --git a/z b/z\n+nope\n", task.answer) == 0.0
    # No gold patch -> unscoreable -> 0.0 (never a false positive).
    assert score_patch("anything", {"gold_patch": ""}) == 0.0


def test_score_trajectory_scores_final_answer():
    from trinity.types import Trajectory

    adapter = get_adapter(BENCHMARK)
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    good = Trajectory(task=task, final_answer=task.answer["gold_patch"])
    bad = Trajectory(task=task, final_answer="diff --git a/z b/z\n+nope\n")
    assert adapter.score_trajectory(good) == 1.0
    assert adapter.score_trajectory(bad) == 0.0
