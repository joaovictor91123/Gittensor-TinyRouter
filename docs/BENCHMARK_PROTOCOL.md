# Hidden Benchmark Sampling Protocol

> Frozen protocol for how each hidden benchmark's **eval / audit / live** sets are
> sampled and sealed. Executable form: [`scripts/benchmark_protocol.py`](../scripts/benchmark_protocol.py).
> Builder that follows it: [`scripts/build_benchmark.py`](../scripts/build_benchmark.py).

The goal is that a benchmark rebuild is **deterministic** (same questions, same
splits, every time) and **auditable** (the public hash pins exactly what was
built, down to which question is in which split). Nothing below may change after
a benchmark is first built for the competition — a change silently invalidates
every score computed against the old set.

## 1. Sealed seed

- `SEALED_SEED = 271828182` (first 9 digits of *e*). Arbitrary but **fixed
  forever**. It alone determines which questions are drawn and their order.
- Every random step in the build is seeded from it, so the build has no
  dependence on wall-clock time, process, or `PYTHONHASHSEED`.

## 2. Split policy

- Splits, in carve order: **`eval` → `audit` → `live`** (`SPLIT_ORDER`).
- One task **pool** is sampled, then split into **contiguous, non-overlapping**
  slices in that order: `eval` takes the first `N_eval`, `audit` the next
  `N_audit`, `live` the next `N_live`. Every task lands in exactly one split.
- `live` items are stored **without** cached model answers (they are scored by
  live API calls); `eval` and `audit` items carry cached answers.

## 3. Sample counts

| split | default count | purpose |
|-------|---------------|---------|
| eval  | 150 | the scored set a submission is graded on |
| audit | 50  | held-out overfit check (eval–audit gap gate) |
| live  | 20  | live-API spot check, no cached answers |

- A benchmark may override counts in `_SPLIT_COUNT_OVERRIDES`; an override is a
  visible, reviewable protocol edit.
- The pool is loaded with a fixed **margin** of `SAMPLE_MARGIN = 50` extra tasks
  (`pool_size = eval + audit + live + 50`) so parse/dedupe drops cannot shrink a
  split. The margin is fixed, so the pool size — and thus the selection — is
  deterministic.

## 4. Stable identifiers

- Each item's `question_id` is the task's own stable id (`math500-3`,
  `mmlu-17`, …) when present, else `sha256(benchmark, index, prompt)` truncated.
- The builtin `hash()` is **never** used for ids — its value is randomised per
  process, which would make the audit hash non-reproducible.

## 5. Rebuild / integrity hash

- The public `content_hash` (`hash.txt`) is `manifest_hash(splits)`: a SHA-256
  over every item's identity fields **and its split label**, canonicalised and
  sorted by `(split, question_id)`.
- Hashed fields: `question_id`, `benchmark`, `task_type`, `question_text`,
  `correct_answer`. Cached model answers/scores are **excluded** — they come from
  live API calls and must not perturb the identity hash.
- Because the split label is hashed, a rebuild that keeps the same questions but
  reshuffles them across eval/audit/live produces a **different** hash. (The old
  question-text-only digest did not catch this.)
- The hash is invariant to input ordering (items are sorted first) but sensitive
  to any change in content or placement.

## 6. Public metadata

`build_manifest()` writes `meta.json` — the committable audit record:

- `seed`, `protocol_version`, `split_order`
- per-split `counts`
- `content_hash`
- the full sorted `question_ids` per split
- `pool_models`, `created_at` (the only non-deterministic fields; excluded from
  the hash)

Committing `hash.txt` and `meta.json` lets anyone verify the hidden benchmark has
not changed without revealing the questions themselves.
