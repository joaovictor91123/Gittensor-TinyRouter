"""MMLU-Pro benchmark adapter (issue #12).

MMLU-Pro extends MMLU to **ten** answer options (A-J) and harder, reasoning-heavy
questions. It reuses the shared multiple-choice scoring path — the reward module's
choice extractor and reference normaliser were widened to A-J, and ``mmlu_pro`` is
in ``reward.CHOICE_BENCHMARKS`` — so this adapter only has to load and format the
ten-option questions.

* :func:`load_mmlu_pro_tasks` — lazy/guarded HuggingFace loader
  (``TIGER-Lab/MMLU-Pro``) with an offline toy fallback. Each task's ``answer`` is
  the correct option **letter** (A-J).
* :class:`MmluProAdapter` — task type :data:`TaskType.MCQ`; scoring delegates to
  :func:`trinity.orchestration.reward.score_text`.

This module imports only stdlib + :class:`trinity.types` + the reward scorer
(``datasets`` is lazy), so it has no dataset-module dependency and cannot form an
import cycle.
"""
from __future__ import annotations

from typing import Any

from trinity.orchestration import reward as _reward
from trinity.types import Task

from .base import BenchmarkAdapter, TaskType
from .registry import register_adapter

__all__ = [
    "BENCHMARK",
    "CHOICE_LETTERS",
    "format_mcq",
    "load_mmlu_pro_tasks",
    "MmluProAdapter",
    "register_mmlu_pro_adapter",
]

#: Canonical benchmark name this adapter registers under.
BENCHMARK = "mmlu_pro"

#: HuggingFace dataset id for MMLU-Pro.
_HF_DATASET = "TIGER-Lab/MMLU-Pro"

#: Option letters, up to ten (A-J). Kept in sync with reward._CHOICE_LETTERS.
CHOICE_LETTERS: str = "ABCDEFGHIJ"


def format_mcq(question: str, choices: list[Any]) -> str:
    """Render a (up to ten-option) multiple-choice question with lettered options.

    Asks the model to end with a single answer letter so the shared choice
    extractor can read it reliably.
    """
    lines = [question.strip(), ""]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with the single letter of the correct option.")
    return "\n".join(lines)


def _row_get(row: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        try:
            if k in row and row[k] is not None:
                return row[k]
        except TypeError:
            break
    return default


def _answer_letter(row: Any, n_options: int) -> str | None:
    """Resolve an MMLU-Pro row's answer to a single option letter (A-J).

    MMLU-Pro rows carry both ``answer`` (the letter) and ``answer_index`` (0-based);
    prefer the explicit letter, fall back to the index.
    """
    letter = _row_get(row, "answer")
    if isinstance(letter, str) and len(letter.strip()) == 1:
        up = letter.strip().upper()
        if up in CHOICE_LETTERS[:n_options]:
            return up
    idx = _row_get(row, "answer_index")
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < n_options and idx < len(CHOICE_LETTERS):
        return CHOICE_LETTERS[idx]
    return None


def _hf_mmlu_pro(split: str) -> list[Task] | None:
    """Load MMLU-Pro from HuggingFace, or ``None`` on any failure."""
    try:
        from datasets import load_dataset  # type: ignore import-not-found
    except Exception:
        return None
    try:
        ds = load_dataset(_HF_DATASET, split=split or "test")
    except Exception:
        return None

    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(row, "question", default="")
        options = _row_get(row, "options", "choices", default=None)
        if not question or not options:
            continue
        options = list(options)
        letter = _answer_letter(row, len(options))
        if letter is None:
            continue
        tasks.append(
            Task(
                task_id=str(_row_get(row, "question_id", default=f"mmlu_pro-{i}")),
                benchmark=BENCHMARK,
                prompt=format_mcq(str(question), options),
                answer=letter,
                meta={
                    "source": _HF_DATASET,
                    "category": _row_get(row, "category"),
                    "n_options": len(options),
                    "choices": options,
                },
            )
        )
    return tasks or None


def _toy_mmlu_pro() -> list[Task]:
    """Tiny ten-option questions so smoke tests run with zero network."""
    q1 = [
        "Nitrogen", "Oxygen", "Carbon", "Helium", "Hydrogen",
        "Neon", "Argon", "Iron", "Gold", "Silver",
    ]
    q2 = [
        "2", "3", "5", "7", "11", "13", "17", "19", "23", "29",
    ]
    return [
        Task(
            task_id="mmlu_pro-toy-0",
            benchmark=BENCHMARK,
            prompt=format_mcq("Which element has the chemical symbol 'O'?", q1),
            answer="B",  # Oxygen
            meta={"source": "toy", "n_options": 10, "choices": q1},
        ),
        Task(
            task_id="mmlu_pro-toy-1",
            benchmark=BENCHMARK,
            prompt=format_mcq("Which of these is the 4th smallest prime number?", q2),
            answer="D",  # 7 (2,3,5,7)
            meta={"source": "toy", "n_options": 10, "choices": q2},
        ),
    ]


def load_mmlu_pro_tasks(split: str, max_items: int | None, seed: int = 0) -> list[Task]:
    """Load MMLU-Pro as a deterministic list of :class:`Task`.

    Tries HuggingFace (lazy/guarded); on any failure falls back to the built-in
    toy set. Applies a ``seed``-seeded shuffle and truncates to ``max_items``, so
    repeated calls with identical arguments return identical lists.
    """
    import random

    tasks = _hf_mmlu_pro(split) or _toy_mmlu_pro()
    rng = random.Random(seed)
    tasks = list(tasks)
    rng.shuffle(tasks)
    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks


class MmluProAdapter(BenchmarkAdapter):
    """MMLU-Pro: ten-option multiple choice, scored through the shared MCQ path."""

    name = BENCHMARK

    def load_tasks(self, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
        return load_mmlu_pro_tasks(split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        return _reward.score_text(self.name, output, reference)

    def task_type(self) -> TaskType:
        return TaskType.MCQ

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": TaskType.MCQ.value,
            "meta": dict(task.meta),
        }


def register_mmlu_pro_adapter() -> None:
    """Register the MMLU-Pro adapter (idempotent-friendly)."""
    from .registry import is_registered

    if not is_registered(BENCHMARK):
        register_adapter(BENCHMARK, MmluProAdapter())
