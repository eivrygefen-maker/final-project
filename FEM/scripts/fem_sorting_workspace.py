#!/usr/bin/env python3
"""
SORTING workspace reset for production FEM sweeps (clean-start runs).

Clears ``temp_results/``, ``temp_modes/`` vectors, and reinitializes ``candidates_log.json``
with pipeline provenance metadata (60–550 Hz staged harvest policy).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fem_harvest_filter import HARVEST_FILTER_POLICY_VERSION
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX

# Production coupled-worker SLEPc quota (must match ``fem_master_dynamic.COUPLED_WORKER_NUM_MODES_CAP``).
PRODUCTION_COUPLED_WORKER_NUM_MODES_CAP = 40
PRODUCTION_SWEEP_HZ_MIN = 60.0
PRODUCTION_SWEEP_HZ_MAX = 550.0


def fresh_candidates_log_payload(
    *,
    sweep_hz_min: float = PRODUCTION_SWEEP_HZ_MIN,
    sweep_hz_max: float = PRODUCTION_SWEEP_HZ_MAX,
    coupled_worker_num_modes_cap: int = PRODUCTION_COUPLED_WORKER_NUM_MODES_CAP,
) -> Dict[str, Any]:
    return {
        "pipeline_meta": {
            "harvest_filter_policy": HARVEST_FILTER_POLICY_VERSION,
            "sweep_hz_min": float(sweep_hz_min),
            "sweep_hz_max": float(sweep_hz_max),
            "coupled_worker_num_modes_cap": int(coupled_worker_num_modes_cap),
            "staged_crossover_hz": 350.0,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "candidates": [],
        "completed_shift_targets": [],
    }


def reset_sorting_workspace(
    sorting_root: Path,
    *,
    reset_log: bool = True,
    clear_temp_results: bool = True,
    clear_temp_modes: bool = True,
    sweep_hz_min: float = PRODUCTION_SWEEP_HZ_MIN,
    sweep_hz_max: float = PRODUCTION_SWEEP_HZ_MAX,
) -> Dict[str, int]:
    """
    Remove scratch artifacts and optionally reinitialize ``candidates_log.json``.

    Returns counts ``{temp_results, temp_modes, candidates_log}``.
    """
    root = sorting_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    counts = {"temp_results": 0, "temp_modes": 0, "candidates_log": 0}

    tr = root / "temp_results"
    if clear_temp_results and tr.is_dir():
        for p in tr.glob("*.json"):
            try:
                p.unlink()
                counts["temp_results"] += 1
            except OSError:
                pass

    tm = root / "temp_modes"
    if clear_temp_modes and tm.is_dir():
        for pat in (
            f"mode_*{MODE_VECTOR_FILE_SUFFIX}",
            "mode_*.npy",
            "mode_w_*",
        ):
            for p in tm.glob(pat):
                try:
                    p.unlink()
                    counts["temp_modes"] += 1
                except OSError:
                    pass

    log_path = root / "candidates_log.json"
    if reset_log:
        log_path.write_text(
            json.dumps(
                fresh_candidates_log_payload(
                    sweep_hz_min=sweep_hz_min,
                    sweep_hz_max=sweep_hz_max,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        counts["candidates_log"] = 1

    (root / "temp_results").mkdir(parents=True, exist_ok=True)
    (root / "temp_modes").mkdir(parents=True, exist_ok=True)
    return counts
