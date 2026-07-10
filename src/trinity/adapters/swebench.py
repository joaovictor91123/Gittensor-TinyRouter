"""SWE-bench Verified benchmark adapter with a structured patch-task schema.

SWE-bench does not fit the text / multiple-choice / code-string answer model the
other benchmarks use: the model is shown a real repository issue and must emit a
**unified diff** that resolves it, and correctness is defined by a test suite
run against the patched repo. This module adds that benchmark behind the
:class:`~trinity.adapters.base.BenchmarkAdapter` interface (issue #17):

* :class:`PatchReference` — the structured ``reference`` for a patch task
  (repo, base commit, gold patch, test patch, FAIL_TO_PASS / PASS_TO_PASS,
  environment setup commit, version), with dict (de)serialization and validation.
* :func:`build_patch_prompt` — the prompt format for a repository-issue input.
* :func:`load_swebench_tasks` — a lazy/guarded HuggingFace loader
  (``princeton-nlp/SWE-bench_Verified``) with an offline toy fallback, returning
  normalized :class:`~trinity.types.Task` objects carrying repo/commit/instance
  metadata in ``task.meta``.
* :class:`SweBenchAdapter` — task type :data:`TaskType.PATCH`.

**Scope note.** Actually *running* the test suite in a sandbox is issue #18; until
then :meth:`SweBenchAdapter.score_output` uses a conservative **exact
normalized-patch match** against the gold patch (whitespace/index-line noise
stripped). That never reports a wrong patch as correct — it only under-credits a
correct-but-different patch — so it is a safe placeholder for the executor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from trinity.types import Task

from .base import BenchmarkAdapter, TaskType
from .registry import register_adapter

__all__ = [
    "BENCHMARK",
    "PatchReference",
    "build_patch_prompt",
    "load_swebench_tasks",
    "normalize_patch",
    "SweBenchAdapter",
    "register_swebench_adapter",
]

#: Canonical benchmark name this adapter registers under.
BENCHMARK = "swebench_verified"

#: HuggingFace dataset id for SWE-bench Verified (500 human-validated instances).
_HF_DATASET = "princeton-nlp/SWE-bench_Verified"


# --------------------------------------------------------------------------- #
# Structured reference schema
# --------------------------------------------------------------------------- #
@dataclass
class PatchReference:
    """The structured ``reference`` for a patch task (stored as ``Task.answer``).

    Captures everything a sandboxed evaluator (#18) needs to grade a candidate
    patch, without pulling the repository itself into the item.
    """

    repo: str
    base_commit: str
    gold_patch: str
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    environment_setup_commit: str | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (the on-disk / hidden-benchmark form)."""
        return {
            "repo": self.repo,
            "base_commit": self.base_commit,
            "gold_patch": self.gold_patch,
            "test_patch": self.test_patch,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "environment_setup_commit": self.environment_setup_commit,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatchReference":
        """Rebuild a :class:`PatchReference` from :meth:`to_dict` output."""
        return cls(
            repo=str(data.get("repo", "")),
            base_commit=str(data.get("base_commit", "")),
            gold_patch=str(data.get("gold_patch", "")),
            test_patch=str(data.get("test_patch", "")),
            fail_to_pass=list(data.get("fail_to_pass", []) or []),
            pass_to_pass=list(data.get("pass_to_pass", []) or []),
            environment_setup_commit=data.get("environment_setup_commit"),
            version=data.get("version"),
        )

    def is_valid(self) -> bool:
        """A reference is usable iff it names a repo, a base commit, and >=1 test."""
        return bool(self.repo and self.base_commit and self.fail_to_pass)


def _as_list(value: Any) -> list[str]:
    """Coerce a SWE-bench FAIL_TO_PASS/PASS_TO_PASS field into a list of strings.

    The HuggingFace rows encode these as either a JSON-encoded string or an
    already-parsed list; normalise both to ``list[str]``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


# --------------------------------------------------------------------------- #
# Prompt format
# --------------------------------------------------------------------------- #
def build_patch_prompt(
    problem_statement: str,
    repo: str,
    base_commit: str,
    hints: str = "",
) -> str:
    """Render the repository-issue prompt shown to a pool model.

    The model is asked for a unified diff so the (future) executor can ``git
    apply`` it directly. The format mirrors SWE-bench's own instructions closely
    enough that patch-capable models produce applyable output.
    """
    parts = [
        f"You are working in the repository `{repo}` at commit `{base_commit}`.",
        "Resolve the following issue by editing the repository's source.",
        "",
        "## Issue",
        problem_statement.strip(),
    ]
    if hints.strip():
        parts += ["", "## Hints", hints.strip()]
    parts += [
        "",
        "## Response format",
        "Return ONLY a unified diff (git patch) that resolves the issue. Begin "
        "each file section with `diff --git` and use `---`/`+++` hunk headers so "
        "the patch applies cleanly with `git apply`. Do not include prose.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Patch normalization + placeholder scoring (real execution is #18)
# --------------------------------------------------------------------------- #
_DIFF_NOISE = re.compile(r"^(index [0-9a-f]+\.\.[0-9a-f]+.*|@@ .* @@.*)$", re.MULTILINE)
_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)


def normalize_patch(patch: str) -> str:
    """Normalise a unified diff for tolerant exact comparison.

    Strips markdown code fences, blob-index lines and hunk ``@@`` headers (which
    carry line offsets that differ harmlessly), collapses trailing whitespace,
    and drops blank lines. This is deliberately conservative: it only removes
    noise that never changes what a patch *does*.
    """
    if not patch:
        return ""
    text = _FENCE.sub("", patch)
    text = _DIFF_NOISE.sub("", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def score_patch(candidate: str, reference: Any) -> float:
    """Placeholder patch scorer: exact normalized match against the gold patch.

    Until the sandboxed test-runner lands (#18), correctness is exact
    normalized-diff equality with the reference solution. This can only
    *under*-credit (a correct-but-different patch scores 0); it never credits a
    wrong patch, so it is safe to report.
    """
    ref = reference if isinstance(reference, dict) else {}
    gold = ref.get("gold_patch", "") if ref else ""
    if not gold:
        return 0.0
    return 1.0 if normalize_patch(candidate) == normalize_patch(gold) else 0.0


# --------------------------------------------------------------------------- #
# Loader (HuggingFace + offline toy fallback)
# --------------------------------------------------------------------------- #
def _row_get(row: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        try:
            if k in row and row[k] is not None:
                return row[k]
        except TypeError:
            break
    return default


def _hf_swebench(split: str) -> list[Task] | None:
    """Load SWE-bench Verified from HuggingFace, or ``None`` on any failure."""
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
        instance_id = str(_row_get(row, "instance_id", default=f"swebench-{i}"))
        problem = _row_get(row, "problem_statement", default="")
        repo = _row_get(row, "repo", default="")
        base_commit = _row_get(row, "base_commit", default="")
        if not problem or not repo:
            continue
        ref = PatchReference(
            repo=str(repo),
            base_commit=str(base_commit),
            gold_patch=str(_row_get(row, "patch", default="")),
            test_patch=str(_row_get(row, "test_patch", default="")),
            fail_to_pass=_as_list(_row_get(row, "FAIL_TO_PASS")),
            pass_to_pass=_as_list(_row_get(row, "PASS_TO_PASS")),
            environment_setup_commit=_row_get(row, "environment_setup_commit"),
            version=_row_get(row, "version"),
        )
        tasks.append(_make_task(instance_id, problem, ref, _row_get(row, "hints_text", default="")))
    return tasks or None


def _make_task(instance_id: str, problem: str, ref: PatchReference, hints: str = "") -> Task:
    """Normalise one SWE-bench instance into a :class:`Task`."""
    return Task(
        task_id=instance_id,
        benchmark=BENCHMARK,
        prompt=build_patch_prompt(problem, ref.repo, ref.base_commit, hints),
        answer=ref.to_dict(),
        meta={
            "source": _HF_DATASET,
            "instance_id": instance_id,
            "repo": ref.repo,
            "base_commit": ref.base_commit,
            "version": ref.version,
            "task_type": TaskType.PATCH.value,
        },
    )


def _toy_swebench() -> list[Task]:
    """Tiny, self-contained patch tasks so smoke tests run with zero network."""
    ref = PatchReference(
        repo="octo/calc",
        base_commit="0" * 40,
        gold_patch=(
            "diff --git a/calc.py b/calc.py\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "-def add(a, b):\n"
            "-    return a - b\n"
            "+def add(a, b):\n"
            "+    return a + b\n"
        ),
        fail_to_pass=["tests/test_calc.py::test_add"],
        pass_to_pass=["tests/test_calc.py::test_sub"],
        version="1.0",
    )
    return [
        _make_task(
            "octo__calc-1",
            "`add(2, 3)` returns -1 instead of 5; the addition helper subtracts.",
            ref,
        )
    ]


def load_swebench_tasks(split: str, max_items: int | None, seed: int = 0) -> list[Task]:
    """Load SWE-bench Verified as a deterministic list of :class:`Task`.

    Tries HuggingFace (lazy/guarded); on any failure falls back to the built-in
    toy set. Applies a ``seed``-seeded shuffle and truncates to ``max_items``, so
    repeated calls with identical arguments return identical lists.
    """
    import random

    tasks = _hf_swebench(split) or _toy_swebench()
    rng = random.Random(seed)
    tasks = list(tasks)
    rng.shuffle(tasks)
    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class SweBenchAdapter(BenchmarkAdapter):
    """SWE-bench Verified: repository-issue in, unified-diff patch out."""

    name = BENCHMARK

    def load_tasks(self, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
        return load_swebench_tasks(split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        return score_patch(output, reference)

    def task_type(self) -> TaskType:
        return TaskType.PATCH

    def score_trajectory(self, traj) -> float:
        # A patch is the final unified diff; there is no per-turn "committed
        # answer" to recover (has_answer has no patch shape), so score the final
        # answer directly against the structured reference.
        return self.score_output(traj.final_answer or "", traj.task.answer)

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": TaskType.PATCH.value,
            "meta": dict(task.meta),
        }


def register_swebench_adapter() -> None:
    """Register the SWE-bench adapter (idempotent-friendly)."""
    from .registry import is_registered

    if not is_registered(BENCHMARK):
        register_adapter(BENCHMARK, SweBenchAdapter())
