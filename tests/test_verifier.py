"""Offline tests for Verifier verdict parsing (SPEC §4.6). No network, no GPU."""
from __future__ import annotations

from trinity.roles.verifier import extract_diagnosis, parse_verdict


# ---------------------------------------------------------------------------
# Committed verdicts still parse
# ---------------------------------------------------------------------------
def test_clean_verdicts_parse():
    assert parse_verdict("Looks right.\nVERDICT: ACCEPT") == "ACCEPT"
    assert parse_verdict("Step 2 is wrong.\nVERDICT: REVISE") == "REVISE"


def test_verdict_parsing_is_case_and_space_insensitive():
    assert parse_verdict("verdict: accept") == "ACCEPT"
    assert parse_verdict("VERDICT:ACCEPT") == "ACCEPT"
    assert parse_verdict("VERDICT:   REVISE") == "REVISE"


def test_trailing_punctuation_and_newline_still_parse():
    # ``\b`` sits between the token and a non-word char, so these remain verdicts.
    assert parse_verdict("VERDICT: ACCEPT.") == "ACCEPT"
    assert parse_verdict("VERDICT: REVISE!") == "REVISE"
    assert parse_verdict("VERDICT: ACCEPT\n") == "ACCEPT"


# ---------------------------------------------------------------------------
# Regression: a longer word starting with the token is NOT a verdict
# ---------------------------------------------------------------------------
def test_prefix_word_is_not_a_committed_verdict():
    # "ACCEPTABLE only if fixed" is a conditional, i.e. the OPPOSITE of accepting.
    # An unanchored VERDICT:\s*(ACCEPT|REVISE) matched the prefix and returned
    # "ACCEPT", which terminates the trajectory early (session.py, SPEC §0.3.5).
    assert parse_verdict("Bound is loose.\nVERDICT: ACCEPTABLE only if fixed") is None
    assert parse_verdict("VERDICT: ACCEPTED with reservations") is None
    assert parse_verdict("VERDICT: REVISED the plan") is None


def test_missing_verdict_returns_none_for_fail_safe_revise():
    # No verdict at all -> None, so orchestration applies fail-safe REVISE.
    assert parse_verdict("I have no strong opinion.") is None
    assert parse_verdict("") is None


# ---------------------------------------------------------------------------
# Last verdict wins (the model may deliberate before committing)
# ---------------------------------------------------------------------------
def test_last_verdict_wins():
    text = "Maybe VERDICT: REVISE at first.\nOn reflection, VERDICT: ACCEPT"
    assert parse_verdict(text) == "ACCEPT"


def test_prefix_word_does_not_shadow_a_later_real_verdict():
    # The prefix word must be skipped entirely, so the real verdict below wins.
    text = "VERDICT: ACCEPTABLE only if fixed\nAfter the fix: VERDICT: REVISE"
    assert parse_verdict(text) == "REVISE"


# ---------------------------------------------------------------------------
# Markdown / separator formatting around the verdict still parses
# ---------------------------------------------------------------------------
def test_markdown_formatted_verdicts_parse():
    # Models routinely emphasise the verdict line. A strict ``VERDICT:\s*`` missed
    # all of these, so the loop fail-safed to REVISE and never accepted early.
    assert parse_verdict("**VERDICT: ACCEPT**") == "ACCEPT"
    assert parse_verdict("**VERDICT:** ACCEPT") == "ACCEPT"
    assert parse_verdict("VERDICT: **ACCEPT**") == "ACCEPT"
    assert parse_verdict("VERDICT: __REVISE__") == "REVISE"
    assert parse_verdict("VERDICT: `REVISE`") == "REVISE"
    assert parse_verdict("#### VERDICT: ACCEPT") == "ACCEPT"
    assert parse_verdict("VERDICT - ACCEPT") == "ACCEPT"


def test_markdown_does_not_defeat_the_prefix_word_guard():
    # The word-boundary guard must survive the broadened separators.
    assert parse_verdict("VERDICT: **ACCEPTABLE** only if fixed") is None
    assert parse_verdict("VERDICT: `REVISED` the plan") is None


def test_markdown_last_verdict_still_wins():
    text = "First **VERDICT: REVISE**.\nThen, on reflection, VERDICT: **ACCEPT**"
    assert parse_verdict(text) == "ACCEPT"


def test_diagnosis_strips_a_markdown_verdict_line():
    text = "The bound is loose.\nVERDICT: **ACCEPT**"
    assert extract_diagnosis(text) == "The bound is loose."


# ---------------------------------------------------------------------------
# Diagnosis must not be truncated at a false match
# ---------------------------------------------------------------------------
def test_diagnosis_is_everything_above_the_final_verdict():
    text = "The bound is loose.\nVERDICT: ACCEPT"
    assert extract_diagnosis(text) == "The bound is loose."


def test_diagnosis_not_truncated_by_a_prefix_word():
    # With no real verdict line the whole text is the diagnosis; the unanchored
    # regex used to cut it at "ACCEPTABLE", silently dropping the qualifier.
    text = "The bound is loose.\nVERDICT: ACCEPTABLE only if fixed"
    assert extract_diagnosis(text) == text.strip()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("ALL PASS")
