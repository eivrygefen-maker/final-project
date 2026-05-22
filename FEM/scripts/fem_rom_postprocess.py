"""
ROM post-processing helpers: near-frequency de-duplication and σ-retry limits.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Drop modes closer than this (Hz) before MMR / NPZ packaging.
MODAL_PRUNE_DF_HZ_DEFAULT = 0.5

# Per scheduler shift: primary ST solve + σ-retry workers (fail-fast).
SIGMA_RETRY_MAX_ATTEMPTS_PER_SHIFT_DEFAULT = 5

# Inside one EPS worker: ST LU σ ladder attempts (factorization retries).
ST_SIGMA_LADDER_MAX_DEFAULT = 6


def _mode_merit(row: Dict[str, Any], prefer: Sequence[str]) -> Tuple[float, ...]:
    """Lexicographic merit (higher is better): e.g. wood_participation, then p_frac."""
    parts: List[float] = []
    for key in prefer:
        try:
            parts.append(float(row.get(key) or 0.0))
        except (TypeError, ValueError):
            parts.append(0.0)
    if not parts:
        try:
            parts.append(float(row.get("wood_participation", 0.0) or 0.0))
        except (TypeError, ValueError):
            parts.append(0.0)
    return tuple(parts)


def prune_near_duplicate_modes(
    candidates: Sequence[Dict[str, Any]],
    *,
    df_hz: float = MODAL_PRUNE_DF_HZ_DEFAULT,
    merit_prefer: Sequence[str] = ("wood_participation", "p_frac"),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Cluster modes with pairwise Δf < ``df_hz`` (transitive chains) and keep one per cluster.

    Keeps the mode with highest merit (``wood_participation``, then ``p_frac``).
    """
    tol = max(1.0e-9, float(df_hz))
    pool = sorted(candidates, key=lambda c: float(c.get("hz", 0.0) or 0.0))
    if not pool:
        return [], []

    clusters: List[List[Dict[str, Any]]] = []
    for row in pool:
        f_hz = float(row.get("hz", 0.0) or 0.0)
        if clusters and f_hz - float(clusters[-1][-1].get("hz", 0.0) or 0.0) < tol:
            clusters[-1].append(row)
        else:
            clusters.append([row])

    kept: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []
    for cluster in clusters:
        best = max(cluster, key=lambda r: _mode_merit(r, merit_prefer))
        kept.append(best)
        for row in cluster:
            if row is not best:
                pruned.append(row)
    return kept, pruned


def sigma_retry_max_attempts_per_shift(
    solver_cfg: Optional[Mapping[str, Any]] = None,
) -> int:
    """Max ST solve attempts per scheduler shift (primary + σ-retry workers)."""
    default = int(SIGMA_RETRY_MAX_ATTEMPTS_PER_SHIFT_DEFAULT)
    if not solver_cfg:
        return default
    try:
        v = int(solver_cfg.get("eps_sigma_retry_max_per_shift", default))
    except (TypeError, ValueError):
        return default
    return max(1, min(12, v))


def st_sigma_ladder_max(solver_cfg: Optional[Mapping[str, Any]] = None) -> int:
    """Max σ values tried inside one worker EPS ST setup (LU fail-fast)."""
    default = int(ST_SIGMA_LADDER_MAX_DEFAULT)
    if not solver_cfg:
        return default
    try:
        v = int(solver_cfg.get("eps_st_sigma_ladder_max", default))
    except (TypeError, ValueError):
        return default
    return max(1, min(16, v))
