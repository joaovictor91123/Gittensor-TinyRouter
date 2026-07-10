"""Verifier verdict parsing (SPEC §4.6).

The Verifier role ends its response with a line of the form::

    VERDICT: ACCEPT
    VERDICT: REVISE

This module extracts that verdict deterministically and the free-text diagnosis
that precedes it. Per SPEC §4.6 the parse is:

- Scan for ``VERDICT:\\s*(ACCEPT|REVISE)\\b`` (case-insensitive). The trailing
  word boundary matters: a longer word that merely *starts* with the token
  (``ACCEPTABLE``, ``ACCEPTED``, ``REVISED``) is NOT a committed verdict.
- Use the **last** match (the model may discuss ACCEPT/REVISE before committing).
- If no match exists, return ``None``. The orchestration layer treats a missing
  verdict as fail-safe REVISE (it never terminates on an unparseable verifier,
  SPEC §0.3.5 / §4.6); parsing stays pure and just reports absence.

Pure / deterministic / no LLM calls.
"""
from __future__ import annotations

import re

__all__ = ["VERDICT_RE", "parse_verdict", "extract_diagnosis"]

# Case-insensitive verdict pattern. ``finditer`` lets us take the LAST occurrence.
#
# Between "VERDICT" and the verdict word we tolerate the colon plus any run of
# whitespace and common Markdown emphasis / separator characters (``*`` bold/
# italic, ``_`` italic, `` ` `` code span, ``~`` strike, ``-`` dash). Models
# routinely format the line as ``**VERDICT:** ACCEPT``, ``VERDICT: **ACCEPT**`` or
# ``VERDICT: `REVISE` `` — a strict ``VERDICT:\s*`` misses all of those, and the
# loop then fail-safes to REVISE, so a correct+complete answer never earns the
# early ACCEPT (which needlessly burns turns, hurting the efficiency term and
# raising latency/cost). The character class matches no letters, so prose like
# "the verdict is ... accept" is still rejected: the verdict word must sit
# immediately after VERDICT + markers.
#
# The trailing ``(?![A-Za-z])`` anchors the token so a longer word that merely
# *starts* with ACCEPT/REVISE ("VERDICT: ACCEPTABLE only if fixed", "ACCEPTED with
# reservations", "REVISED the plan") is NOT read as a committed verdict. A plain
# ``\b`` cannot be used here: an underscore is a word character, so ``\b`` would
# reject the common italic wrapper ``VERDICT: __REVISE__``. The negative lookahead
# blocks only a trailing *letter*, so closing markdown (``**``/``__``/`` ` ``) and
# punctuation are still fine.
VERDICT_RE = re.compile(r"VERDICT\b[\s:*_`~-]*(ACCEPT|REVISE)(?![A-Za-z])", re.IGNORECASE)


def parse_verdict(text: str) -> str | None:
    """Return the verifier's verdict, or ``None`` if absent.

    Matches the last ``VERDICT: ACCEPT`` / ``VERDICT: REVISE`` line
    (case-insensitive) in ``text`` and normalizes it to upper case.

    Args:
        text: the verifier model's raw output ``M_k``.

    Returns:
        ``"ACCEPT"`` or ``"REVISE"`` for the last verdict line found, else
        ``None`` when no verdict line is present.
    """
    if not text:
        return None
    matches = list(VERDICT_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).upper()


def extract_diagnosis(text: str) -> str:
    """Return the diagnosis text that precedes the (last) verdict line.

    The diagnosis ``δ_k`` is everything above the final ``VERDICT:`` line. When
    no verdict line is present, the whole text is treated as the diagnosis.

    Args:
        text: the verifier model's raw output ``M_k``.

    Returns:
        The stripped diagnosis text (possibly empty).
    """
    if not text:
        return ""
    matches = list(VERDICT_RE.finditer(text))
    if not matches:
        return text.strip()
    # Everything before the start of the last verdict match.
    return text[: matches[-1].start()].strip()
