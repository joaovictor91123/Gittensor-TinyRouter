#!/usr/bin/env python3
"""Evaluate a PR submission against the hidden benchmark.

This is the maintainer's tool. It evaluates a miner's submitted routing head
against the HIDDEN benchmark (stored OUTSIDE the repo — never committed) and
determines whether the head beats the current best accuracy.

All 8 anti-cheat gates run BEFORE any GPU work or API calls. A failing gate
rejects the submission immediately with zero cost to the maintainer.

Usage:
    source ~/.config/trinity/secrets.env
    python scripts/pr_eval.py --pr 42 --benchmark math500 --submission alice/1
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import calendar
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

# ---- Constants ----
_EXPECTED_HEAD_PARAMS = 6144   # 6 × 1024
_EXPECTED_SVF_PARAMS = 7168    # 7 × 1024
_EXPECTED_TOTAL = _EXPECTED_HEAD_PARAMS + _EXPECTED_SVF_PARAMS  # 13312
_MIN_TRAINING_COST = 15.0       # plausible minimum for a full CMA-ES run
_MAX_WEIGHT_MAGNITUDE = 1e6     # weights beyond this are degenerate
_COPY_THRESHOLD = 0.999          # cosine similarity threshold for duplicate detection
_OVERFIT_HARD_REJECT = 0.10      # eval-audit gap above this = reject
_OVERFIT_PENALTY = 0.05          # gap above this = 0.85 penalty
_RATE_LIMIT_WINDOW_DAYS = 7
_RATE_LIMIT_MAX_SUBMISSIONS = 1
_POOL_MODELS = ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"]


# ==========================================================================
# AES decryption (matches build_benchmark.py's encryption)
# ==========================================================================

def _derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, dklen=32)


def _decrypt_json(filepath: Path, password: str) -> dict:
    """Decrypt an AES-256-GCM encrypted JSON benchmark file."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        print("ERROR: cryptography package required. pip install cryptography")
        sys.exit(1)

    ciphertext_b64 = filepath.read_text().strip()
    combined = base64.b64decode(ciphertext_b64)
    salt = combined[:16]
    nonce = combined[16:28]
    ct = combined[28:]
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    plain = aesgcm.decrypt(nonce, ct, None)
    data = json.loads(plain.decode("utf-8"))
    return data["items"]  # extract the items list


# ==========================================================================
# Hidden benchmark loader
# ==========================================================================

def _load_hidden_benchmark(benchmark_name: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """Load encrypted eval, audit, and live question sets for a benchmark."""
    bench_dir = os.environ.get(
        "TINYROUTER_BENCHMARK_DIR",
        str(_REPO.parent / "tinyrouter-benchmark"),
    )
    bench_path = Path(bench_dir) / benchmark_name
    password = os.environ.get("BENCHMARK_PASSWORD", "")

    if not bench_path.exists():
        print(f"[pr_eval] ERROR: Hidden benchmark not found at {bench_path}")
        print(f"  Set TINYROUTER_BENCHMARK_DIR or run scripts/build_benchmark.py first.")
        sys.exit(2)
    if not password:
        print("[pr_eval] ERROR: BENCHMARK_PASSWORD env var not set.")
        sys.exit(2)

    eval_items = []
    audit_items = []
    live_items = []

    for filename, target in [("eval.json", eval_items), ("audit.json", audit_items), ("live.json", live_items)]:
        fp = bench_path / filename
        if fp.exists():
            try:
                items = _decrypt_json(fp, password)
                target.extend(items)
            except Exception as exc:
                print(f"[pr_eval] ERROR: Failed to decrypt {filename}: {exc}")
                sys.exit(2)

    return eval_items, audit_items, live_items


# ==========================================================================
# Load submission
# ==========================================================================

def _load_submission(submission_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """Load head_weights.npy, svf_scales.npy, and receipt.json."""
    hw = submission_dir / "head_weights.npy"
    sv = submission_dir / "svf_scales.npy"
    rc = submission_dir / "receipt.json"

    if not hw.exists() or not sv.exists():
        print(f"[pr_eval] ERROR: Missing submission files in {submission_dir}")
        return None

    head_W = np.load(str(hw))
    svf_scales = np.load(str(sv))
    receipt = json.loads(rc.read_text()) if rc.exists() else {}

    return head_W.astype(np.float32), svf_scales.astype(np.float32), receipt


# ==========================================================================
# Gate 1: Rate Limiting
# ==========================================================================

def _parse_utc_timestamp(ts_str: str) -> Optional[float]:
    """Parse a ``YYYY-MM-DDTHH:MM:SSZ`` UTC stamp to a Unix epoch (seconds).

    Leaderboard timestamps are written in UTC (``time.gmtime`` + trailing ``Z``,
    see :func:`_update_leaderboard`), so they must be read back as UTC. Using
    ``time.mktime`` here would interpret the struct as *local* time and skew the
    epoch by the maintainer host's UTC offset — silently shifting the rate-limit
    window by hours on any non-UTC box. ``calendar.timegm`` is the UTC inverse of
    ``time.gmtime`` and is timezone-independent.

    Returns ``None`` if ``ts_str`` is empty or not in the expected format.
    """
    if not ts_str:
        return None
    try:
        return float(calendar.timegm(time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")))
    except (ValueError, OSError):
        return None


def _check_rate_limit(miner_name: str, benchmark: str, leaderboard: dict) -> Optional[str]:
    """Check if miner has exceeded the submission rate limit.

    Returns None if allowed, or an error string if rate-limited.
    """
    bench_entry = leaderboard.get("benchmarks", {}).get(benchmark, {})
    history = bench_entry.get("history", [])

    cutoff = time.time() - _RATE_LIMIT_WINDOW_DAYS * 86400
    recent = 0
    for entry in history:
        ts = _parse_utc_timestamp(entry.get("timestamp", ""))
        if ts is None:
            continue
        if entry.get("miner") == miner_name and ts > cutoff:
            recent += 1

    if recent >= _RATE_LIMIT_MAX_SUBMISSIONS:
        return f"rate_limited: {recent} submission(s) in the last {_RATE_LIMIT_WINDOW_DAYS} days (max {_RATE_LIMIT_MAX_SUBMISSIONS})"
    return None


# ==========================================================================
# Gate 2: Weight Validation
# ==========================================================================

def _validate_weights(head_W: np.ndarray, svf_scales: np.ndarray) -> Optional[str]:
    """Validate weight array integrity. Returns None if clean, error string if corrupt."""
    # Param count
    if head_W.size + svf_scales.size != _EXPECTED_TOTAL:
        return f"param_count: got {head_W.size + svf_scales.size}, expected {_EXPECTED_TOTAL}"
    if head_W.shape != (6, 1024):
        return f"head_shape: got {head_W.shape}, expected (6, 1024)"

    # NaN / Inf
    if np.any(np.isnan(head_W)) or np.any(np.isnan(svf_scales)):
        return "weights_contain_NaN"
    if np.any(np.isinf(head_W)) or np.any(np.isinf(svf_scales)):
        return "weights_contain_Inf"

    # Extreme magnitudes
    if np.any(np.abs(head_W) > _MAX_WEIGHT_MAGNITUDE):
        return f"head_weights_exceed_max: max_abs={np.max(np.abs(head_W)):.1f}"
    if np.any(np.abs(svf_scales) > _MAX_WEIGHT_MAGNITUDE):
        return f"svf_scales_exceed_max: max_abs={np.max(np.abs(svf_scales)):.1f}"

    # All-zeros head (untrained)
    if np.allclose(head_W, 0.0):
        return "head_weights_all_zeros"

    # SVF scales too close to zero (would zero out the SLM)
    if np.allclose(svf_scales, 0.0):
        return "svf_scales_all_zeros"

    # Head norm implausibly tiny after training (should be > 0.01 after 60 gens)
    head_norm = float(np.linalg.norm(head_W))
    if head_norm < 0.001:
        return f"head_weight_norm_too_small: {head_norm:.6f}"

    return None


# ==========================================================================
# Gate 3: Duplicate Detection
# ==========================================================================

def _check_duplicate(head_vec: np.ndarray, submissions_root: Path,
                      current_miner: str, current_gen: int) -> Optional[str]:
    """Check cosine similarity against ALL previous submissions.

    Scans submissions/*/merged/ and looks for near-identical weight vectors.
    Returns None if unique, error string with the matched miner/gen if duplicate.
    """
    # Check all submission directories
    for sub_dir in sorted(submissions_root.glob("*/*/")):
        parts = sub_dir.relative_to(submissions_root).parts
        if len(parts) < 2:
            continue
        other_miner, other_gen = parts[0], parts[1]
        if other_miner == current_miner and other_gen == str(current_gen):
            continue  # skip self

        hw_path = sub_dir / "head_weights.npy"
        sv_path = sub_dir / "svf_scales.npy"
        if not hw_path.exists() or not sv_path.exists():
            continue

        try:
            other_hw = np.load(str(hw_path)).astype(np.float64).ravel()
            other_sv = np.load(str(sv_path)).astype(np.float64).ravel()
        except (ValueError, OSError):
            continue

        other_vec = np.concatenate([other_hw, other_sv])
        sim = _cosine_similarity(head_vec, other_vec)

        if sim > _COPY_THRESHOLD:
            return f"duplicate_of_{other_miner}_gen_{other_gen}_sim_{sim:.4f}"

    # Also check the current king from leaderboard
    lb = _load_leaderboard()
    for bench_name, bench_entry in lb.get("benchmarks", {}).items():
        king_miner = bench_entry.get("best_miner", "")
        king_gen = bench_entry.get("best_generation", 0)
        if king_miner and king_miner != current_miner:
            king_dir = submissions_root / king_miner / str(king_gen)
            hw_path = king_dir / "head_weights.npy"
            sv_path = king_dir / "svf_scales.npy"
            if hw_path.exists() and sv_path.exists():
                try:
                    king_hw = np.load(str(hw_path)).astype(np.float64).ravel()
                    king_sv = np.load(str(sv_path)).astype(np.float64).ravel()
                    king_vec = np.concatenate([king_hw, king_sv])
                    sim = _cosine_similarity(head_vec, king_vec)
                    if sim > _COPY_THRESHOLD:
                        return f"duplicate_of_king_{king_miner}_gen_{king_gen}_sim_{sim:.4f}"
                except (ValueError, OSError):
                    pass

    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


# ==========================================================================
# Gate 4: Receipt Cross-Validation
# ==========================================================================

def _validate_receipt(receipt: dict) -> Optional[str]:
    """Cross-validate receipt fields for plausibility. Returns None if plausible."""
    # Cost must be non-zero and above minimum
    cost = receipt.get("total_cost_usd", 0.0)
    if cost <= 0.0:
        return "receipt_cost_zero_or_missing"
    if cost < _MIN_TRAINING_COST:
        return f"receipt_cost_too_low: ${cost:.2f} < ${_MIN_TRAINING_COST:.2f} minimum"

    # Fitness history
    history = receipt.get("fitness_history", [])
    if not history or len(history) < 3:
        return "receipt_fitness_history_too_short: need >= 3 entries"

    # Extract fitness values
    values = []
    for entry in history:
        if isinstance(entry, dict):
            v = entry.get("gen_mean_fitness", entry.get("mean_fitness", entry.get("best_fitness")))
        elif isinstance(entry, (int, float)):
            v = entry
        else:
            continue
        if v is not None:
            values.append(float(v))

    if len(values) < 3:
        return "receipt_fitness_history_no_valid_values"

    # First generation should not be perfect
    if values[0] > 0.98:
        return f"receipt_fitness_starts_too_high: {values[0]:.4f}"

    # Should not be a flat line
    if max(values) - min(values) < 0.001:
        return "receipt_fitness_flat_line"

    # Should not be perfectly monotonic (real CMA-ES is noisy)
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    if len(diffs) > 3 and all(d >= 0 for d in diffs):
        return "receipt_fitness_too_perfect: monotonically increasing"

    # Generation count should match history length
    claimed_gens = receipt.get("generations", 0)
    if claimed_gens > 0 and abs(claimed_gens - len(history)) > 5:
        return f"receipt_generations_mismatch: claimed {claimed_gens}, history has {len(history)}"

    # Best fitness is the best CANDIDATE ever evaluated (es.best()); cross-check
    # it against the per-generation PEAK series (gen_max_fitness / max_fitness),
    # NOT the population MEANS in `values`. With m_cma binary-reward tasks per
    # candidate, fitness is granular and the population best sits well above any
    # generation mean, so comparing best_fitness to max(values) rejects honest
    # receipts. The shape checks above stay on the means, where they belong.
    best_fitness = receipt.get("best_fitness", 0.0)
    if best_fitness > 0.0:
        peaks = []
        for entry in history:
            if isinstance(entry, dict):
                p = entry.get("gen_max_fitness", entry.get("max_fitness", entry.get("best_fitness")))
            elif isinstance(entry, (int, float)):
                p = entry
            else:
                continue
            if p is not None:
                peaks.append(float(p))
        if peaks:
            peak_max = max(peaks)
            if abs(best_fitness - peak_max) > 0.1:
                return f"receipt_best_fitness_mismatch: claimed {best_fitness:.4f}, history peak {peak_max:.4f}"

    return None


# ==========================================================================
# Cached evaluation
# ==========================================================================

def _evaluate_cached(head, encoder, items: List[dict], pool_model_names: List[str]) -> float:
    """Evaluate a head on cached benchmark items. Returns accuracy [0, 1]."""
    import torch
    from trinity.orchestration.reward import score_text

    if not items:
        return 0.0

    correct = 0
    for item in items:
        h_np = encoder.encode(item["question_text"])
        h_t = torch.as_tensor(np.asarray(h_np, dtype=np.float32), device=head.weight.device)
        agent_idx, _role, _dbg = head.select(h_t, sample=False)
        model_name = pool_model_names[agent_idx % len(pool_model_names)]
        cached = item.get("model_answers", {}).get(model_name, "")
        if score_text(item.get("benchmark", "math500"), cached, item.get("correct_answer")) > 0.0:
            correct += 1

    return correct / len(items)


# ==========================================================================
# Live evaluation
# ==========================================================================

async def _evaluate_live(
    policy, pool, pool_models, items: List[dict],
    max_turns: int = 5, max_tokens: int = 4096,
) -> Tuple[float, float]:
    """Run full multi-turn eval with real API calls. Returns (accuracy, avg_turns)."""
    import httpx
    from trinity.orchestration.reward import score
    from trinity.orchestration.session import run_trajectory
    from trinity.types import Task

    if not items:
        return 0.0, 0.0

    tasks = [Task(
        task_id=item.get("question_id", f"q{i}"),
        benchmark=item.get("benchmark", "math500"),
        prompt=item["question_text"],
        answer=item.get("correct_answer"),
    ) for i, item in enumerate(items)]

    correct = 0
    total_turns = 0

    async with httpx.AsyncClient() as client:
        trajs = await asyncio.gather(*[
            run_trajectory(t, policy, pool, pool_models, sample=False, client=client,
                           max_turns=max_turns, max_tokens=max_tokens, reasoning="minimal")
            for t in tasks
        ], return_exceptions=True)

    for i, result in enumerate(trajs):
        if isinstance(result, BaseException):
            total_turns += max_turns
            continue
        traj = result
        if score(traj) > 0.0:
            correct += 1
        total_turns += traj.n_turns

    n = len(tasks)
    return correct / n, total_turns / n


# ==========================================================================
# Scoring
# ==========================================================================

def _compute_score(hidden_acc: float, live_acc: float, avg_turns: float,
                   novelty: float) -> float:
    """Composite score: 70% hidden + 15% live + 10% efficiency + 5% novelty."""
    max_turns = 5
    efficiency = max(0.0, (max_turns - avg_turns) / (max_turns - 1)) * live_acc if live_acc > 0 else 0.0
    return 0.70 * hidden_acc + 0.15 * live_acc + 0.10 * efficiency + 0.05 * novelty


# ==========================================================================
# Gate 7: Novelty Computation
# ==========================================================================

def _compute_novelty(head, encoder, eval_items: List[dict],
                      pool_models: List[str]) -> float:
    """Compute novelty by comparing routing decisions against the current king.

    Loads the current best head from the leaderboard, runs both heads on the
    first min(50, len(eval_items)) questions, and returns 1.0 - agreement_rate.

    If there is no king yet, returns 0.5 (neutral).
    """
    lb = _load_leaderboard()
    submissions_root = _REPO / "submissions"

    # Find the current king
    king_miner = None
    king_gen = 0
    for bench_name, bench_entry in lb.get("benchmarks", {}).items():
        km = bench_entry.get("best_miner")
        kg = bench_entry.get("best_generation", 0)
        if km and kg:
            king_miner = km
            king_gen = kg
            break

    if not king_miner:
        return 0.5  # no king yet — neutral novelty

    king_dir = submissions_root / king_miner / str(king_gen)
    hw_path = king_dir / "head_weights.npy"
    if not hw_path.exists():
        return 0.5

    import torch
    from trinity.coordinator.head import LinearHead

    try:
        king_hw = np.load(str(hw_path))
    except (ValueError, OSError):
        return 0.5

    king_head = LinearHead(n_a=6, d_h=1024, n_models=3).to(head.weight.device)
    king_head.load_weight(king_hw)

    # Compare routing decisions on reference questions
    ref_count = min(50, len(eval_items))
    matches = 0
    for item in eval_items[:ref_count]:
        h_np = encoder.encode(item["question_text"])
        h_t = torch.as_tensor(np.asarray(h_np, dtype=np.float32), device=head.weight.device)

        sub_agent, sub_role, _ = head.select(h_t, sample=False)
        king_agent, king_role, _ = king_head.select(h_t, sample=False)

        if sub_agent == king_agent and sub_role == king_role:
            matches += 1

    agreement = matches / ref_count if ref_count > 0 else 0.0
    return 1.0 - agreement


# ==========================================================================
# Leaderboard
# ==========================================================================

def _load_leaderboard() -> dict:
    lb_path = _REPO / "leaderboard.json"
    if lb_path.exists():
        return json.loads(lb_path.read_text())
    return {"benchmarks": {}}


def _update_leaderboard(benchmark: str, miner_name: str, generation: int,
                         pr_number: int, score: float, hidden_acc: float,
                         live_acc: float, avg_turns: float) -> None:
    """Update leaderboard.json with a new winning submission."""
    lb = _load_leaderboard()
    bench_entry = lb.setdefault("benchmarks", {}).setdefault(benchmark, {
        "best_score": 0.0,
        "best_miner": None,
        "best_generation": 0,
        "best_pr": None,
        "baseline_random": None,
        "best_single_model": None,
        "oracle_ceiling": None,
        "history": [],
    })

    bench_entry["best_score"] = round(score, 4)
    bench_entry["best_miner"] = miner_name
    bench_entry["best_generation"] = generation
    bench_entry["best_pr"] = pr_number
    bench_entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    bench_entry["history"].append({
        "miner": miner_name,
        "generation": generation,
        "score": round(score, 4),
        "pr": pr_number,
        "merged": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    lb["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (_REPO / "leaderboard.json").write_text(json.dumps(lb, indent=2) + "\n")
    print(f"[pr_eval] leaderboard.json updated")


# ==========================================================================
# Main
# ==========================================================================

async def evaluate_pr(pr_number: int, benchmark: str,
                       submission_subpath: str) -> dict:
    """Evaluate a PR submission against the hidden benchmark with all 8 gates."""
    print(f"\n{'='*60}")
    print(f"[pr_eval] PR #{pr_number} — {benchmark}")
    print(f"{'='*60}")

    # ---- Parse submission path ----
    sub_dir = _REPO / "submissions" / submission_subpath
    if not sub_dir.exists():
        return _reject("submission_not_found", f"Directory not found: submissions/{submission_subpath}")
    if not (sub_dir / "head_weights.npy").exists():
        return _reject("submission_incomplete", f"No head_weights.npy in submissions/{submission_subpath}")

    parts = sub_dir.relative_to(_REPO / "submissions").parts
    miner_name = parts[0] if len(parts) >= 1 else "unknown"
    generation = int(parts[1]) if len(parts) >= 2 else 0

    print(f"[pr_eval] Miner: {miner_name}  Generation: {generation}")
    print(f"[pr_eval] Path: submissions/{submission_subpath}")

    # ---- Load submission ----
    loaded = _load_submission(sub_dir)
    if loaded is None:
        return _reject("load_failed", "Failed to load submission files")
    head_W, svf_scales, receipt = loaded

    head_vec = np.concatenate([
        np.asarray(head_W, dtype=np.float64).ravel(),
        np.asarray(svf_scales, dtype=np.float64).ravel(),
    ])

    # ══════════════════════════════════════════════════════════════
    # GATE 1: Rate Limiting (before any GPU/API work)
    # ══════════════════════════════════════════════════════════════
    lb = _load_leaderboard()
    err = _check_rate_limit(miner_name, benchmark, lb)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 2: Weight Validation (NaN, Inf, extreme values, untrained)
    # ══════════════════════════════════════════════════════════════
    err = _validate_weights(head_W, svf_scales)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 3: Duplicate Detection (cosine similarity vs all history)
    # ══════════════════════════════════════════════════════════════
    submissions_root = _REPO / "submissions"
    err = _check_duplicate(head_vec, submissions_root, miner_name, generation)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 4: Receipt Cross-Validation
    # ══════════════════════════════════════════════════════════════
    err = _validate_receipt(receipt)
    if err:
        return _reject(err)

    print("[pr_eval] All 4 pre-eval gates passed ✓\n")

    # ══════════════════════════════════════════════════════════════
    # Load benchmark + encoder (GPU work starts here)
    # ══════════════════════════════════════════════════════════════
    eval_items, audit_items, live_items = _load_hidden_benchmark(benchmark)
    print(f"[pr_eval] Loaded encrypted benchmark: {len(eval_items)} eval + "
          f"{len(audit_items)} audit + {len(live_items)} live questions")

    from trinity.coordinator.policy import CoordinatorPolicy
    from trinity.llm.openrouter_client import OpenRouterPool

    cfg = yaml.safe_load((_REPO / "configs" / "trinity.yaml").read_text())
    cc = cfg["coordinator"]

    print("[pr_eval] Loading encoder on GPU...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cc["encoder_model"],
        device=cc.get("device", "cuda:0"),
        dtype=cc.get("dtype", "bfloat16"),
        target_layer=cc["svf"]["target_layer"],
        svf_matrices=cc["svf"].get("matrices"),
        n_models=3, n_roles=3,
        l2_normalize=cc["hidden_state"].get("l2_normalize", True),
    )
    encoder = policy.encoder

    from trinity.coordinator.head import LinearHead
    head = LinearHead(n_a=6, d_h=1024, n_models=3).to(encoder.model.device)
    head.load_weight(head_W)

    # ---- Cached eval ----
    print("[pr_eval] Running cached eval (150 questions)...")
    t0 = time.time()
    hidden_acc = _evaluate_cached(head, encoder, eval_items, _POOL_MODELS)
    print(f"  hidden_acc = {hidden_acc:.4f}  ({time.time() - t0:.1f}s)")

    # ---- Audit eval ----
    print("[pr_eval] Running audit eval (50 questions)...")
    audit_acc = _evaluate_cached(head, encoder, audit_items, _POOL_MODELS) if audit_items else hidden_acc
    print(f"  audit_acc  = {audit_acc:.4f}")

    # ══════════════════════════════════════════════════════════════
    # GATE 5: Overfit Rejection (HARD gate — not just a warning)
    # ══════════════════════════════════════════════════════════════
    gap = hidden_acc - audit_acc
    overfit_penalty = 1.0
    if gap > _OVERFIT_HARD_REJECT:
        return _reject(f"overfit_rejected: eval-audit gap {gap:.4f} > {_OVERFIT_HARD_REJECT}")
    elif gap > _OVERFIT_PENALTY:
        overfit_penalty = 0.85
        print(f"  overfit_penalty = 0.85 (gap {gap:.4f} > {_OVERFIT_PENALTY})")
    else:
        print(f"  eval-audit gap = {gap:.4f} (clean)")

    # ---- Live eval ----
    print("[pr_eval] Running live multi-turn eval (20 questions, real API calls)...")
    pool = OpenRouterPool(str(_REPO / "configs" / "models.yaml"))
    policy.configure(np.concatenate([
        np.asarray(head_W, dtype=np.float64).ravel(),
        np.asarray(svf_scales, dtype=np.float64).ravel(),
    ]), spec)
    live_acc, avg_turns = await _evaluate_live(policy, pool, _POOL_MODELS, live_items)
    print(f"  live_acc   = {live_acc:.4f}  avg_turns = {avg_turns:.2f}")

    # ══════════════════════════════════════════════════════════════
    # GATE 7: Compute actual novelty
    # ══════════════════════════════════════════════════════════════
    novelty = _compute_novelty(head, encoder, eval_items, _POOL_MODELS)
    print(f"  novelty    = {novelty:.4f}")

    # ---- Composite score ----
    score = _compute_score(hidden_acc, live_acc, avg_turns, novelty)
    score *= overfit_penalty
    print(f"  composite  = {score:.4f}")

    # ---- Compare to leaderboard ----
    lb = _load_leaderboard()
    best_score = lb.get("benchmarks", {}).get(benchmark, {}).get("best_score", 0.0)

    print(f"\n  Current best: {best_score:.4f}")
    print(f"  Submission:   {score:.4f}")
    print(f"  Delta:        {score - best_score:+.4f}")

    if score > best_score:
        print(f"\n  *** APPROVED *** — beats current best by {score - best_score:+.4f}")
        _update_leaderboard(benchmark, miner_name, generation, pr_number,
                            score, hidden_acc, live_acc, avg_turns)
        return {
            "approved": True,
            "score": round(score, 4),
            "best_score": round(score, 4),
            "delta": round(score - best_score, 4),
            "message": f"APPROVED: score {score:.4f} beats current best {best_score:.4f} by {score - best_score:+.4f}",
        }
    else:
        # ══════════════════════════════════════════════════════════════
        # GATE 6: Minimize score feedback on rejection
        # Only reveal composite score + delta. NEVER reveal component scores.
        # ══════════════════════════════════════════════════════════════
        print(f"\n  *** REJECTED *** — does not beat current best ({best_score:.4f})")
        print(f"  [internal] hidden={hidden_acc:.4f} audit={audit_acc:.4f} "
              f"live={live_acc:.4f} novelty={novelty:.4f} turns={avg_turns:.2f}")
        return {
            "approved": False,
            "score": round(score, 4),
            "best_score": best_score,
            "delta": round(score - best_score, 4),
            "message": f"REJECTED: score {score:.4f} does not beat current best {best_score:.4f} (delta: {score - best_score:+.4f})",
        }


# ==========================================================================
# Helpers
# ==========================================================================

def _reject(reason: str, detail: str = "") -> dict:
    """Return a rejection result. Never leaks component scores."""
    msg = f"GATE_FAILED: {reason}"
    if detail:
        msg += f" — {detail}"
    print(f"[pr_eval] {msg}")
    return {
        "approved": False,
        "score": 0.0,
        "best_score": None,
        "delta": None,
        "message": msg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a TinyRouter PR submission against the hidden benchmark"
    )
    ap.add_argument("--pr", type=int, required=True, dest="pr_number",
                    help="GitHub PR number")
    ap.add_argument("--benchmark", default="math500",
                    help="Benchmark name (math500 or mmlu)")
    ap.add_argument("--submission", required=True, dest="submission_subpath",
                    help="Path to submission relative to repo root (e.g. alice/1)")
    args = ap.parse_args()

    result = asyncio.run(evaluate_pr(
        pr_number=args.pr_number,
        benchmark=args.benchmark,
        submission_subpath=args.submission_subpath,
    ))

    print(f"\n[pr_eval] RESULT:")
    print(json.dumps(result, indent=2))

    if not result["approved"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
