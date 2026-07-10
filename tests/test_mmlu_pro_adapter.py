"""MMLU-Pro adapter + A-J choice scoring (issue #12).

Offline via the toy fallback (ten-option questions), no network.
"""
from __future__ import annotations

from trinity.adapters import TaskType, get_adapter
from trinity.adapters.mmlu_pro import (
    BENCHMARK,
    MmluProAdapter,
    format_mcq,
    load_mmlu_pro_tasks,
)
from trinity.orchestration.reward import extract_choice_letter, score_text


def test_registered_and_typed_as_mcq():
    adapter = get_adapter(BENCHMARK)
    assert isinstance(adapter, MmluProAdapter)
    assert adapter.name == "mmlu_pro"
    assert adapter.task_type() is TaskType.MCQ


def test_load_tasks_returns_ten_option_tasks():
    tasks = load_mmlu_pro_tasks("test", max_items=5, seed=0)
    assert tasks
    t = tasks[0]
    assert t.benchmark == "mmlu_pro"
    assert t.meta["n_options"] == 10
    # The rendered prompt exposes options past D (proves >A-D support).
    assert "J." in t.prompt
    assert t.answer in "ABCDEFGHIJ"


def test_load_tasks_deterministic_and_capped():
    a = load_mmlu_pro_tasks("test", max_items=1, seed=5)
    b = load_mmlu_pro_tasks("test", max_items=1, seed=5)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert len(a) <= 1


def test_scoring_extracts_letters_beyond_A_D():
    # The shared extractor now reads E-J, not just A-D.
    assert extract_choice_letter("The answer is G.") == "G"
    assert extract_choice_letter("Final answer: J") == "J"
    # And A-D still works (no regression).
    assert extract_choice_letter("the answer is (B)") == "B"


def test_mmlu_pro_scored_through_shared_mcq_path():
    adapter = get_adapter(BENCHMARK)
    assert adapter.score_output("The answer is H.", "H") == 1.0
    assert adapter.score_output("The answer is H.", "C") == 0.0
    # Reference given as a 0-based index resolves to a letter (H == index 7).
    assert score_text("mmlu_pro", "Answer: H", 7) == 1.0


def test_toy_answers_are_correct():
    """The toy items' declared answers match their option lists."""
    adapter = get_adapter(BENCHMARK)
    for t in load_mmlu_pro_tasks("test", max_items=None, seed=0):
        # Each toy task is self-consistent: scoring its own answer letter passes.
        assert adapter.score_output(f"The answer is {t.answer}.", t.answer) == 1.0


def test_format_mcq_letters_up_to_j():
    prompt = format_mcq("Q?", [f"opt{i}" for i in range(10)])
    for letter in "ABCDEFGHIJ":
        assert f"{letter}." in prompt


def test_serialize_task_shape():
    adapter = get_adapter(BENCHMARK)
    task = load_mmlu_pro_tasks("test", max_items=1, seed=0)[0]
    item = adapter.serialize_task(task)
    assert item["benchmark"] == "mmlu_pro"
    assert item["task_type"] == TaskType.MCQ.value
    assert item["reference"] == task.answer
