"""Entrypoint: evaluate a trained coordinator + baselines on a benchmark.

Reports the relative invariants from SPEC §1.3:
  - TRINITY (trained coordinator, argmax) vs
  - each single model alone (one direct Worker turn) [R1, R2] vs
  - random routing (random agent+role each turn) [R4].

Usage:
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python -m trinity.eval --benchmark math500 \
        --theta experiments/math500/run/best_theta.npy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np
import yaml

from .adapters import get_adapter
from .coordinator import params as P
from .coordinator.policy import CoordinatorPolicy
from .llm.openrouter_client import OpenRouterPool
from .orchestration.session import run_trajectory
from .types import ROLE_ORDER, Role

_REPO = Path(__file__).resolve().parents[2]

# Locked reproducibility seed — committed, never changed.
# Using a custom seed prints a warning: cherry-picking seeds to find a lucky
# eval split undermines result trustworthiness.
REPRODUCIBILITY_SEED: int = 42


class RandomPolicy:
    """Random (agent, role) each turn — the R4 routing baseline (no GPU).

    Draws from the caller-supplied ``rng`` when one is given, so each trajectory
    can own a deterministically-seeded stream. Falling back to a single shared
    ``self.rng`` across trajectories running under ``asyncio.gather`` would make
    the draws depend on network completion order rather than on the seed.
    """

    def __init__(self, n_models: int, seed: int = 0) -> None:
        self.n_models = n_models
        self.rng = random.Random(seed)

    def decide(
        self,
        transcript_text: str,
        *,
        sample: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, Role]:
        """Pick the next (agent index, role) uniformly at random.

        Args:
            transcript_text: Unused — the baseline ignores the transcript.
            sample: Unused — the baseline is always stochastic.
            rng: Per-trajectory RNG. Falls back to the instance RNG when ``None``.

        Returns:
            A ``(agent_idx, role)`` pair.
        """
        r = self.rng if rng is None else rng
        return r.randrange(self.n_models), r.choice(ROLE_ORDER)


def task_rng(seed: int, task_id: str) -> random.Random:
    """Build a per-task RNG whose stream depends only on ``seed`` and ``task_id``.

    Keeps the random-routing baseline invariant to ``asyncio`` scheduling: a task's
    draws never depend on when other concurrent tasks' HTTP calls happen to return.
    Seeding from a string is stable across processes (``random.Random`` hashes str
    seeds with SHA-512, so it does not depend on ``PYTHONHASHSEED``).

    Args:
        seed: The run's base seed.
        task_id: The benchmark item's stable identifier.

    Returns:
        A freshly seeded :class:`random.Random`.
    """
    return random.Random(f"{seed}:{task_id}")


def _mean_scoring_failures_as_zero(
    outcomes: list, score: Callable[[object], float], label: str
) -> float:
    """Average ``score`` over ``outcomes``, counting exhausted-retry failures as 0.0.

    ``outcomes`` is the result of ``asyncio.gather(..., return_exceptions=True)``,
    so a task that exhausted its retries appears as a ``BaseException`` rather than
    aborting the whole evaluation. Such a task produced no answer, so it scores 0.0
    -- the same (slightly pessimistic) convention ``optim.fitness`` already applies
    on the training side.

    Args:
        outcomes: Per-task results, each either a value or a ``BaseException``.
        score: Maps a successful outcome to its scalar score.
        label: Condition name, used in the diagnostic printed on failures.

    Returns:
        The mean score over all tasks.

    Raises:
        RuntimeError: If ``outcomes`` is empty, or if *every* task failed -- a mean
            of 0.0 would then be an artifact of a dead API, not a measurement.
    """
    if not outcomes:
        raise RuntimeError(f"{label}: no tasks to score")

    n_failed = sum(1 for o in outcomes if isinstance(o, BaseException))
    if n_failed == len(outcomes):
        first = next(o for o in outcomes if isinstance(o, BaseException))
        raise RuntimeError(
            f"{label}: all {n_failed} tasks failed after retries; refusing to report "
            f"0.0 as a score. First error: {first!r}"
        )
    if n_failed:
        print(
            f"  [warn] {label}: {n_failed}/{len(outcomes)} tasks failed after "
            f"retries; scored 0.0"
        )

    scores = [0.0 if isinstance(o, BaseException) else float(score(o)) for o in outcomes]
    return float(mean(scores))


async def _score_policy(
    tasks, policy, pool, pool_models, *, adapter, sample, rng_seed: int | None = None, **run_kwargs
) -> float:
    import httpx

    async with httpx.AsyncClient() as cli:
        # return_exceptions=True so one trajectory that exhausts retries (e.g. a
        # persistent timeout) degrades to score 0 instead of discarding the whole
        # evaluation -- including the baselines already computed.
        trajs = await asyncio.gather(
            *[
                run_trajectory(
                    t, policy, pool, pool_models, adapter=adapter, sample=sample, client=cli,
                    rng=None if rng_seed is None else task_rng(rng_seed, t.task_id),
                    **run_kwargs,
                )
                for t in tasks
            ],
            return_exceptions=True,
        )
    # Score through the adapter (not reward.score directly) so the routed path
    # honours the same benchmark contract as the single-model baseline.
    return _mean_scoring_failures_as_zero(trajs, adapter.score_trajectory, "trinity")


async def _score_single_model(tasks, pool, model, adapter, *, max_tokens, reasoning) -> float:
    """Baseline: ask one model directly (one Worker-style turn), score its answer."""
    import httpx

    from .roles.prompts import build_messages

    async with httpx.AsyncClient() as cli:
        async def one(task):
            msgs = build_messages(Role.WORKER, adapter.build_prompt(task), [])
            res = await pool.chat(model, msgs, max_tokens=max_tokens, temperature=0.0,
                                  reasoning=reasoning, client=cli)
            return adapter.score_output(res.text, task.answer)

        scores = await asyncio.gather(*[one(t) for t in tasks], return_exceptions=True)
    return _mean_scoring_failures_as_zero(scores, float, f"single::{model}")


async def evaluate(args) -> dict:
    pool = OpenRouterPool(args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    # Resolve the benchmark to an adapter ONCE; the rest of the evaluator drives
    # the adapter interface and never branches on the benchmark name (#9).
    adapter = get_adapter(args.benchmark)
    tasks = adapter.load_tasks("test", max_items=args.max_items, seed=args.seed)
    print(f"[eval] benchmark={args.benchmark}  {len(tasks)} test tasks  pool={pool_models}")
    run_kwargs = dict(max_turns=args.max_turns, max_tokens=args.max_tokens, reasoning=args.reasoning)

    results: dict[str, float] = {}

    # --- single-model baselines (R1/R2) ---
    for m in pool_models:
        reps = [await _score_single_model(tasks, pool, m, adapter,
                                          max_tokens=args.max_tokens, reasoning=args.reasoning)
                for _ in range(max(1, args.single_reps))]
        s = float(mean(reps))
        results[f"single::{m}"] = s
        if len(reps) > 1:
            sd = (sum((x - s) ** 2 for x in reps) / len(reps)) ** 0.5
            results[f"single_std::{m}"] = sd
            print(f"  single  {m:20s} = {s:.4f} ± {sd:.4f}  (reps={reps})")
        else:
            print(f"  single  {m:20s} = {s:.4f}")

    # --- TRINITY trained coordinator (argmax) ---
    cfg = yaml.safe_load(Path(args.config).read_text())["coordinator"]
    print("[eval] building coordinator on GPU...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cfg["encoder_model"], device=cfg.get("device", "cuda:0"),
        dtype=cfg.get("dtype", "bfloat16"), target_layer=cfg["svf"]["target_layer"],
        svf_matrices=cfg["svf"].get("matrices"), n_models=n_models,
        l2_normalize=cfg["hidden_state"].get("l2_normalize", True),
    )
    theta = np.load(args.theta)
    policy.configure(theta, spec)
    s_trinity = await _score_policy(tasks, policy, pool, pool_models, adapter=adapter,
                                    sample=False, **run_kwargs)
    results["TRINITY"] = s_trinity
    print(f"  TRINITY (trained)        = {s_trinity:.4f}")

    # --- random routing (R4) — multi-seed baseline to remove run-to-run noise ---
    # The paper reports random routing as a single draw per run, but with
    # small eval sets (~120 q's) the variance is large (0.733–0.792 in
    # practice).  Reporting the mean over 100 seeds gives an honest baseline.
    rand_seeds = max(1, args.rand_seeds)
    rand_scores: list[float] = []
    for s in range(rand_seeds):
        seed_s = args.seed * 10000 + s
        rand = RandomPolicy(n_models, seed=seed_s)
        s_r = await _score_policy(tasks, rand, pool, pool_models, adapter=adapter,
                                  sample=False, rng_seed=seed_s, **run_kwargs)
        rand_scores.append(s_r)
    s_rand = float(mean(rand_scores))
    results["random_routing"] = s_rand
    if rand_seeds > 1:
        rand_std = (sum((x - s_rand) ** 2 for x in rand_scores) / rand_seeds) ** 0.5
        results["random_routing_std"] = rand_std
        print(f"  random routing           = {s_rand:.4f} ± {rand_std:.4f}  (n={rand_seeds} seeds)")
    else:
        print(f"  random routing           = {s_rand:.4f}")

    best_single = max(results[k] for k in results if k.startswith("single::"))
    invariants = {
        "R1/R2 TRINITY > best single model": s_trinity > best_single,
        "R4 TRINITY > random routing": s_trinity > s_rand,
        "best_single": best_single,
    }
    out = {"benchmark": args.benchmark, "results": results, "invariants": invariants}
    print("[eval] invariants:", json.dumps(invariants, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate TRINITY + baselines")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--theta", required=True, help="path to trained best_theta.npy")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=100, dest="max_items")
    ap.add_argument("--single-reps", type=int, default=1, dest="single_reps",
                    help="average each single-model baseline over K runs (cuts nondeterminism noise)")
    ap.add_argument("--max-turns", type=int, default=5, dest="max_turns")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--seed", type=int, default=REPRODUCIBILITY_SEED,
                    help=f"random seed (default: {REPRODUCIBILITY_SEED} — locked for reproducibility; "
                         "overriding prints a warning)")
    ap.add_argument("--rand-seeds", type=int, default=100, dest="rand_seeds",
                    help="number of random seeds for the random-routing baseline (default: 100)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.seed != REPRODUCIBILITY_SEED:
        print(
            f"[eval] ⚠ seed={args.seed} differs from locked REPRODUCIBILITY_SEED="
            f"{REPRODUCIBILITY_SEED}. Non-default seeds weaken reproducibility "
            f"and should be justified in any reported results.",
            flush=True,
        )
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
