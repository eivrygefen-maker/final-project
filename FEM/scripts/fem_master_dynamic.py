#!/usr/bin/env python3
"""
Master driver for ``fem_worker_single.py``: bounded concurrency, scout-led **spectral
zones**, per-job timeouts, and merge of worker JSON into ``candidates_log.json``.

**Dynamic scheduler (default)**:
  * **Scout seeding**: first tasks at ``f_start+6`` then ``f_start`` Hz (W0 carries
    ``role=scout`` for zone reports).
  * **Zones Z1/Z2/Z3** from merge **efficiency** (``kept_modes / spectral_step_hz``) with thresholds
    12 / 3 modes·Hz⁻¹ plus yield / avg wood when efficiency is neutral, **2-report hysteresis**, and
    adaptive steps 6 / 8 / 18 Hz; Z1 caps ``num_modes`` at 75; Z3 clamps 80–100; absolute SLEPc cap
    ``num_modes`` ≤ 100.
  * **Backfill**: Z3→Z1 transition injects ~2 Hz tasks behind the scout to recover resolution.

**Merge (MMR / veto)**:
  * **Wood V2**: linear floor 0.0005 @ 100 Hz → 0.0003 @ 450 Hz; **isolation relief**: if a
    candidate is >15 Hz from all peers in the same worker batch, floor ×0.6.
  * **Zone-tuned spectral penalty**: Z1 uses tighter ``tau`` (stronger proximity penalty);
    Z3 lowers uniqueness floor for isolation bonus.
  * Sparse CSR overlap, Zone-C HF quota, ``gc.collect()`` as before.

**Legacy**: ``--legacy-static-schedule`` restores fixed ``get_band_params`` stepping.

**Adaptive frequency ceiling (ROM-safe)**: once per run, when a merged shift reaches **435 Hz**,
the master reads the current **spectral zone** (density / yield from the existing scheduler) and
sets the sweep ``hz_max`` to **440 / 470 / 490 Hz** (saturated → sparse interest). Initial default
``--hz-max`` remains **450 Hz** until that single conductor decision. Mode selection (MMR / top-100
export) is unchanged elsewhere; this only bounds how far the shift sweep runs.

**Resume**: ``candidates_log.json`` may list ``completed_shift_targets`` (shift ``target_hz`` values).
All shift identity checks use ``HZ_TOLERANCE`` (1e-4 Hz) via ``hz_shift_key`` / ``hz_shift_quantize``.
If that list is empty but ``candidates`` is not, a **recovery scan** infers shifts from
``vector_path`` (``mode_w_<mHz>_``) or from mode ``hz`` snapped to the band table step, writes
``completed_shift_targets`` when ``merge_lock`` is provided, and advances the cursor by the
scheduler zone step. Scout seeds and the cursor never schedule below ``max(completed_shift_targets)``
(see ``_max_completed_shift_hz``). ``pop_next`` skips shifts already in that set (no worker). Pending
``temp_results/result_*.json`` files are merged once at startup before scheduling.

Resource policy: at most ``--max-workers`` concurrent workers (default **2**); on Linux,
workers use ``taskset`` cores ``{1..N}`` where ``N=max_workers`` with core 0 reserved for
the master/OS, then ``mpiexec --bind-to none -n 1``.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fem_mode_array_utils import (
    MODE_VECTOR_FILE_SUFFIX,
    csr_normalized_overlap,
    load_mode_column_any,
)


def _repo_root() -> Path:
    """Walk parents from this file until a directory named ``final-project`` is found."""
    repo = Path(os.path.abspath(__file__)).resolve()
    while repo.name != "final-project" and repo.parent != repo:
        repo = repo.parent
    if repo.name != "final-project":
        raise RuntimeError(
            "Could not locate a parent directory named 'final-project' starting from "
            f"{Path(__file__).resolve()}"
        )
    return repo


REPO_ROOT = _repo_root()

DEFAULT_MAX_WORKERS = 2
STAGGER_ADDITIONAL_WORKER_SECONDS = 10.0
MIN_SPAWN_GAP_SECONDS = 5.0

# --- Frequency clustering (same scale as legacy gap thinning) ---
MIN_HZ_GAP = 0.4

# --- Shift identity / resume idempotency (float-safe; all "completed?" checks use this) ---
HZ_TOLERANCE = 1e-4


def hz_shift_quantize(hz: float) -> float:
    """Canonical shift frequency on a fixed grid (``HZ_TOLERANCE`` Hz)."""
    return round(float(hz) / HZ_TOLERANCE) * HZ_TOLERANCE


def hz_shift_key(hz: float) -> str:
    """Stable dict/set key for a shift center (tolerant to tiny float noise)."""
    return f"{hz_shift_quantize(hz):.8f}"


def hz_shift_markers_match(a: float, b: float) -> bool:
    """True if two shift centers represent the same bucket (within ``HZ_TOLERANCE``)."""
    return abs(hz_shift_quantize(a) - hz_shift_quantize(b)) <= HZ_TOLERANCE + 1e-12


def hz_any_matches_completed_shift(target: float, completed_values: List[float]) -> bool:
    for x in completed_values:
        if hz_shift_markers_match(target, float(x)):
            return True
    return False


def efficiency_index_sanitized(raw_modes_per_hz: float) -> float:
    """Map raw modes/Hz into ``[3, 15]`` for zone hysteresis (limits optimiser noise)."""
    lo, hi = 3.0, 15.0
    x = math.log1p(max(0.0, float(raw_modes_per_hz)))
    denom = math.log1p(25.0)
    if denom <= 0.0:
        return lo
    return float(lo + (hi - lo) * min(1.0, x / denom))

SWEEP_HZ_MIN = 90.0

WOOD_FLOOR_HZ_ANCHOR_LO = 100.0
WOOD_FLOOR_HZ_ANCHOR_HI = 350.0
WOOD_PARTICIPATION_FLOOR_LO = 0.0008
WOOD_PARTICIPATION_FLOOR_HI = 0.0003

# --- Frequency-dependent wood veto V2 (linear 100 Hz → 450 Hz): 0.0005 → 0.0003 ---
WOOD_V2_LO_HZ = 100.0
WOOD_V2_HI_HZ = 480.0
WOOD_V2_LO = 0.0005
WOOD_V2_HI = 0.0003

# --- Scout zone classification (Worker 1 / leading scout reports) ---
SCOUT_YIELD_DENSE = 0.4
SCOUT_YIELD_SPARSE = 0.15
SCOUT_AVG_WOOD_DENSE = 0.001
ZONE_HYSTERESIS_STREAK = 2

# --- Adaptive stepping (Hz) by spectral zone ---
ZONE1_STEP_HZ = 6.0
ZONE2_STEP_HZ = 8.0
ZONE3_STEP_HZ = 18.0
# Dense (Z1): SLEPc ``nev`` cap — reduced from 100 to cut redundant spectral data (Sim 14).
ZONE1_NUM_MODES_CAP = 75
# Sparse (Z3): flexible ``nev`` band while keeping 18 Hz step for fast traversal.
ZONE3_NUM_MODES_MIN = 80
ZONE3_NUM_MODES_MAX = 100

# --- Adaptive efficiency engine (kept modes / spectral step Hz) ---
EFFICIENCY_HISTORY_MAX = 5
EFFICIENCY_HIGH_THRESHOLD = 12.0
EFFICIENCY_LOW_THRESHOLD = 3.0
SLEPC_NUM_MODES_ABSOLUTE_CEILING = 100

# --- Adaptive sweep ceiling (conductor, once per run at 435 Hz; spectral zone → hz_max) ---
# Spectral zone from ``SpectralScheduler`` / ``on_worker_merge``: 1 = dense (saturated), 2 = normal,
# 3 = sparse (high interest). Maps to user "Conductor" zones and ceilings for ROM-safe sweep span.
CONDUCTOR_TRIGGER_HZ = 450.0
CONDUCTOR_CEILING_SPECTRAL_ZONE_1 = 480.0  # saturated / low spectral interest → stop at sweep max
CONDUCTOR_CEILING_SPECTRAL_ZONE_2 = 480.0
CONDUCTOR_CEILING_SPECTRAL_ZONE_3 = 480.0  # sparse / high interest → full sweep max

# --- Merge-time physical density (numerical duplicate clusters) ---
MERGE_SHIFT_CLUSTER_SPAN_HZ = 1.0
MERGE_SHIFT_CLUSTER_MIN_MODES = 20
WORKER_COL_NORM_MIN = 1e-9
# Strict coupled-mode harvest gate (merge): true FSI vs structural spurious vs σ-locked.
HARVEST_GATE_MIN_WOOD = 0.01
HARVEST_GATE_MIN_P_FRAC_FSI = 0.02
HARVEST_GATE_SIGMA_TOL_HZ = 0.35
HARVEST_GATE_SIGMA_P_FRAC = 1.0e-4
# Incoming worker rows must meet this uniqueness floor (matches worker thin gate).
MERGE_INCOMING_UNIQUENESS_MIN = 0.04

# --- Isolation / MMR-style spectral shaping ---
MIN_UNIQUENESS_FOR_ISOLATION = 0.06
ZONE3_MIN_UNIQUENESS_FOR_ISOLATION = 0.03
ISO_BATCH_GAP_HZ = 15.0
ISOLATION_WOOD_FLOOR_SCALE = 0.6
ZONE1_SPECTRAL_TAU_SCALE = 0.65
ZONE1_MIN_HZ_GAP = 1.2
ISO_DISTANCE_REF_HZ = 120.0
ISO_WOOD_WEIGHT = 1.0
ISO_AIR_DAMPING = 0.25

# --- Spectral penalty / reward vs nearest Hz (existing log ∪ picks this merge) ---
SPECTRAL_CLOSE_HZ = 3.0
SPECTRAL_PENALTY_TAU_HZ = 1.2
SPECTRAL_FAR_HZ = 15.0
SPECTRAL_FAR_CAP_HZ = 40.0
SPECTRAL_REWARD_SLOPE = 0.014

# --- Protected HF (Zone C) ---
ZONE_C_MIN_HZ = 350.0
ZONE_C_MIN_QUOTA = 4

LOGGER = logging.getLogger("fem_master_dynamic")


def _conductor_ceiling_hz_for_spectral_zone(zone: int) -> float:
    z = int(zone)
    if z == 1:
        return CONDUCTOR_CEILING_SPECTRAL_ZONE_1
    if z == 3:
        return CONDUCTOR_CEILING_SPECTRAL_ZONE_3
    return CONDUCTOR_CEILING_SPECTRAL_ZONE_2


def _conductor_user_zone_label(spectral_zone: int) -> Tuple[int, str]:
    """
    User-facing Conductor zone (1/2/3) and label for logging.

    Spectral Z3 (sparse) → Conductor 1 (high interest / sparse spectrum).
    Spectral Z2 → Conductor 2 (normal).
    Spectral Z1 (dense) → Conductor 3 (saturated / low interest).
    """
    sz = int(spectral_zone)
    if sz == 3:
        return (1, "high interest/sparse")
    if sz == 2:
        return (2, "normal")
    return (3, "saturated/low interest")


def hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def result_json_path(sorting_root: Path, hz: float) -> Path:
    return sorting_root / "temp_results" / f"result_{hz_result_tag(hz)}.json"


def _target_hz_from_result_filename(path: Path) -> Optional[float]:
    """Decode ``result_<mHz_tag>.json`` → shift ``target_hz`` (same convention as ``hz_result_tag``)."""
    m = re.match(r"^result_(\d+)\.json$", path.name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) / 1000.0


def get_band_params(current_hz: float) -> Dict[str, Any]:
    hz = float(current_hz)
    if SWEEP_HZ_MIN <= hz < 150.0:
        return {"step_hz": 5, "num_modes": 80, "timeout_minutes": 60, "label": "Dense Band 1"}
    if 150.0 <= hz < 300.0:
        return {"step_hz": 10, "num_modes": 50, "timeout_minutes": 60, "label": "Medium Band"}
    if 300.0 <= hz <= 480.0:
        return {"step_hz": 10, "num_modes": 40, "timeout_minutes": 60, "label": "High Band"}
    if hz > 480.0:
        return {"step_hz": 25, "num_modes": 15, "timeout_minutes": 20, "label": "Dead Zone"}
    raise ValueError(
        f"get_band_params: hz={hz} is outside the supported sweep (expected hz >= {SWEEP_HZ_MIN})."
    )


def _recovery_infer_shift_targets_from_candidates(
    candidates: List[Dict[str, Any]], hz_min: float
) -> Set[float]:
    """
    Infer completed shift centers from legacy logs: ``mode_w_<mHz>_`` in ``vector_path``,
    else snap mode ``hz`` to the nearest ``get_band_params`` step from ``hz_min``.
    """
    inferred: Set[float] = set()
    f0 = float(hz_min)
    for c in candidates:
        vp = str(c.get("vector_path", "") or "")
        m = re.search(r"mode_w_(\d+)_", vp, flags=re.IGNORECASE)
        if m:
            inferred.add(hz_shift_quantize(int(m.group(1)) / 1000.0))
            continue
        try:
            mh = float(c.get("hz", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mh) or mh < f0:
            continue
        hz_band = max(SWEEP_HZ_MIN, mh)
        try:
            step = float(get_band_params(hz_band).get("step_hz", ZONE2_STEP_HZ))
        except ValueError:
            step = ZONE2_STEP_HZ
        step = max(step, 1e-6)
        k = f0 + round((mh - f0) / step) * step
        inferred.add(hz_shift_quantize(float(k)))
    return inferred


def _resume_efficiency_index_from_log(
    candidates: List[Dict[str, Any]], completed: Set[float]
) -> Optional[float]:
    """
    Resume-only density: total log modes per Hz of explored shift span.

    Uses ``len(candidates) / (max(completed) - min(completed))`` when at least two shifts
    exist; otherwise falls back to a nominal ``ZONE2_STEP_HZ`` span (single-shift resume).
    """
    if not candidates or not completed:
        return None
    ts = sorted(float(t) for t in completed)
    if len(ts) >= 2:
        span_hz = float(ts[-1] - ts[0])
        if span_hz < 1e-6:
            span_hz = ZONE2_STEP_HZ
    else:
        span_hz = ZONE2_STEP_HZ
    return float(len(candidates)) / max(span_hz, 1e-9)


def _persist_completed_shift_targets_union(
    log_path: Path, lock: threading.Lock, completed_sorted: List[float]
) -> None:
    """Atomically set ``completed_shift_targets`` while preserving ``candidates``."""
    with lock:
        if log_path.exists():
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"candidates": [], "completed_shift_targets": []}
        else:
            payload = {"candidates": [], "completed_shift_targets": []}
        payload.setdefault("candidates", list(payload.get("candidates") or []))
        payload["completed_shift_targets"] = [float(x) for x in completed_sorted]
        tmp = log_path.with_suffix(log_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(log_path)


BACKFILL_STEP_HZ = 2.0


@dataclass
class MergeContext:
    """Per-merge adaptive manager + veto tuning from spectral zone scheduler."""

    spectral_zone: int = 2
    min_hz_gap: float = MIN_HZ_GAP
    spectral_tau_scale: float = 1.0
    min_uniqueness_iso: float = MIN_UNIQUENESS_FOR_ISOLATION


@dataclass
class MergeStats:
    raw_n: int
    kept_after_veto: int
    kept_after_manager: int
    avg_wood_raw: float
    yield_kept_over_raw: float
    coupled_valid_kept: int = 0


@dataclass
class SpectralScheduler:
    """
    Scout-seeded sweep with **efficiency-based** zone hints (modes per Hz of spectral spacing),
    yield/wood fallback, hysteresis, Zone 3→1 backfill on scout transitions, and **resume**
    from ``candidates_log.json`` (including inferred efficiency and conservative cursor step).
    """

    hz_min: float
    hz_max: float
    max_workers: int = DEFAULT_MAX_WORKERS
    sorting_root: Optional[Path] = None
    merge_lock: Optional[threading.Lock] = field(default=None, repr=False)
    _zone_effective: int = field(default=2, repr=False)
    _pending_zone: Optional[int] = field(default=None, repr=False)
    _pending_streak: int = field(default=0, repr=False)
    _scheduled: Set[str] = field(default_factory=set, repr=False)
    _seed_queue: Deque[Tuple[float, Dict[str, Any], str]] = field(default_factory=deque, repr=False)
    _backfill: Deque[Tuple[float, Dict[str, Any], str]] = field(default_factory=deque, repr=False)
    _cursor_hz: float = field(default=float("nan"), repr=False)
    _seeds_drained: bool = field(default=False, repr=False)
    _efficiency_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=EFFICIENCY_HISTORY_MAX), repr=False
    )
    _max_completed_shift_hz: float = field(default=-math.inf, repr=False)
    _conductor_ceiling_applied: bool = field(default=False, repr=False)
    _low_keep_streak: int = field(default=0, repr=False)
    _coverage_emergency_pending: bool = field(default=False, repr=False)
    _last_kept_after_manager: int = field(default=0, repr=False)
    _last_report_hz: float = field(default=float("nan"), repr=False)

    def __post_init__(self) -> None:
        self._n_modes_in_log = 0
        self._recovery_shift_count = 0
        self._resume_load_state()
        self._bootstrap_seed_queue()
        self._write_coverage_anchor_state()
        self._log_resume_verification_table()

    def _write_coverage_anchor_state(self) -> None:
        if self.sorting_root is None:
            return
        payload = {
            "coverage_emergency_pending": bool(self._coverage_emergency_pending),
            "low_keep_streak": int(self._low_keep_streak),
            "last_kept_after_manager": int(self._last_kept_after_manager),
            "last_report_hz": None if not math.isfinite(self._last_report_hz) else float(self._last_report_hz),
            "rule": "trigger when two consecutive merged shifts keep <=1 candidate each",
        }
        try:
            root = Path(self.sorting_root).resolve()
            p = root / "coverage_anchor_state.json"
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            tmp.replace(p)
        except OSError as exc:
            LOGGER.warning("Could not write coverage_anchor_state.json: %s", exc)

    def _update_coverage_anchor_state(self, report_hz: float, kept_after_manager: int) -> None:
        self._last_report_hz = float(report_hz)
        self._last_kept_after_manager = int(kept_after_manager)
        if int(kept_after_manager) <= 1:
            self._low_keep_streak += 1
        else:
            self._low_keep_streak = 0
            self._coverage_emergency_pending = False
        if self._low_keep_streak >= 2:
            self._coverage_emergency_pending = True
            LOGGER.info(
                "[Coverage Anchor] Emergency coverage mode armed after consecutive low-keep shifts (<=1). "
                "Next tuner selection should relax wood veto and enforce minimum anchors."
            )
        self._write_coverage_anchor_state()

    def _shift_target_done(self, hz: float) -> bool:
        """True if this shift ``target_hz`` was already merged (resume) or consumed."""
        return self._hz_key(hz) in self._scheduled

    def register_completed_shift(self, target_hz: float) -> None:
        """Mark a shift target finished after a successful merge (same-run dedupe for ``pop_next``)."""
        th = float(target_hz)
        self._scheduled.add(self._hz_key(th))
        self._max_completed_shift_hz = max(self._max_completed_shift_hz, th)

    def _log_resume_verification_table(self) -> None:
        """Startup summary: modes in log, recovery-inferred shifts, first scheduled target."""
        n_modes = int(getattr(self, "_n_modes_in_log", 0))
        n_rec = int(getattr(self, "_recovery_shift_count", 0))
        nxt = self._peek_next_task()
        if nxt is not None:
            first_line = f"{float(nxt[0]):.4f}"
        else:
            first_line = "(none — sweep complete or empty band)"
        LOGGER.info(
            "Resume verification —\n"
            "  Total Modes Found in Log:     %d\n"
            "  Recovered Completed Shifts:   %d\n"
            "  First New Target Frequency:   %s Hz",
            n_modes,
            n_rec,
            first_line,
        )

    def _resume_load_state(self) -> None:
        """Load ``candidates_log.json`` + ``completed_shift_targets``; advance cursor past last shift."""
        self._cursor_hz = float(self.hz_min)
        self._n_modes_in_log = 0
        self._recovery_shift_count = 0
        if self.sorting_root is None:
            return
        root = Path(self.sorting_root).resolve()
        self.sorting_root = root
        log_path = root / "candidates_log.json"
        if not log_path.is_file():
            LOGGER.info("Resume: no candidates_log.json at %s (cold start).", log_path)
            return
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Resume: could not parse candidates_log.json: %s", exc)
            return
        cands = list(payload.get("candidates") or [])
        self._n_modes_in_log = len(cands)
        initial_completed: Set[float] = set()
        for x in payload.get("completed_shift_targets") or []:
            try:
                initial_completed.add(hz_shift_quantize(float(x)))
            except (TypeError, ValueError):
                continue
        completed: Set[float] = set(initial_completed)
        if not initial_completed and cands:
            recovered = _recovery_infer_shift_targets_from_candidates(cands, float(self.hz_min))
            recovery_added = recovered - completed
            completed |= recovered
            self._recovery_shift_count = len(recovery_added)
            if recovery_added:
                if self.merge_lock is not None:
                    _persist_completed_shift_targets_union(
                        log_path, self.merge_lock, sorted(completed)
                    )
                    LOGGER.info(
                        "Recovery scan: inferred %d shift target(s) from log modes; "
                        "wrote completed_shift_targets to %s.",
                        len(recovery_added),
                        log_path,
                    )
                else:
                    LOGGER.warning(
                        "Recovery scan inferred %d shift target(s) but merge_lock is unset; "
                        "in-memory resume only (log not patched).",
                        len(recovery_added),
                    )
        for t in completed:
            self._scheduled.add(self._hz_key(float(t)))
        if completed:
            self._max_completed_shift_hz = max(float(t) for t in completed)
        else:
            self._max_completed_shift_hz = -math.inf

        inf_eff = _resume_efficiency_index_from_log(cands, completed)
        if inf_eff is not None and math.isfinite(inf_eff):
            self._efficiency_history.clear()
            self._efficiency_history.append(efficiency_index_sanitized(float(inf_eff)))
            if inf_eff > EFFICIENCY_HIGH_THRESHOLD:
                self._zone_effective = 1
                self._pending_zone = None
                self._pending_streak = 0
                LOGGER.info(
                    "Resume efficiency: inferred %.2f modes/Hz (high); zone=1 conservative cursor step=%.1f Hz.",
                    inf_eff,
                    ZONE1_STEP_HZ,
                )
            elif inf_eff < EFFICIENCY_LOW_THRESHOLD:
                self._zone_effective = 3
                self._pending_zone = None
                self._pending_streak = 0
                LOGGER.info(
                    "Resume efficiency: inferred %.2f modes/Hz (low); zone=3 coarse cursor step=%.1f Hz.",
                    inf_eff,
                    ZONE3_STEP_HZ,
                )

        if completed:
            mx_done = max(completed)
            step0 = float(self._current_step_hz())
            self._cursor_hz = max(float(self.hz_min), mx_done + step0)

    def _bootstrap_seed_queue(self) -> None:
        f0 = float(self.hz_min)
        hi = float(self.hz_max)
        step = float(ZONE1_STEP_HZ)
        n_workers = max(1, int(self.max_workers))
        seed_targets: List[Tuple[float, str, str]] = []
        # Scout (rank 0) starts furthest ahead; remaining workers fill lower offsets.
        scout_hz = max(f0, min(hi, f0 + (n_workers - 1) * step))
        seed_targets.append((scout_hz, "scout W0", "scout"))
        for rank in range(1, n_workers):
            offset_idx = n_workers - 1 - rank
            hz = max(f0, min(hi, f0 + offset_idx * step))
            seed_targets.append((hz, f"W{rank}", ""))

        seen: Set[str] = set()
        seeds: List[Tuple[float, str, str]] = []
        for hz, tag, role in seed_targets:
            key = self._hz_key(hz)
            if key in seen:
                continue
            seen.add(key)
            seeds.append((hz, tag, role))
        mx_done = float(self._max_completed_shift_hz)
        for hz, tag, role in seeds:
            if math.isfinite(mx_done) and float(hz) <= mx_done + HZ_TOLERANCE:
                LOGGER.info(
                    "Resume: skip scout seed %.4f Hz (at or behind max completed shift %.4f Hz).",
                    float(hz),
                    mx_done,
                )
                continue
            if self._shift_target_done(hz):
                continue
            p = self._band_params_with_zone_step(hz, tag)
            self._seed_queue.append((hz, p, role))

    def _hz_key(self, hz: float) -> str:
        return hz_shift_key(hz)

    def _current_step_hz(self) -> float:
        z = self._zone_effective
        if z == 1:
            return ZONE1_STEP_HZ
        if z == 3:
            return ZONE3_STEP_HZ
        return ZONE2_STEP_HZ

    def _band_params_with_zone_step(self, hz: float, label_suffix: str) -> Dict[str, Any]:
        base = dict(get_band_params(hz))
        z = self._zone_effective
        if z == 1:
            base["num_modes"] = min(ZONE1_NUM_MODES_CAP, int(base.get("num_modes", 80)))
        elif z == 3:
            nm = int(base.get("num_modes", 80))
            base["num_modes"] = max(ZONE3_NUM_MODES_MIN, min(ZONE3_NUM_MODES_MAX, nm))
        base["num_modes"] = min(SLEPC_NUM_MODES_ABSOLUTE_CEILING, max(1, int(base.get("num_modes", 80))))
        base["label"] = f"{base.get('label', '')} [{label_suffix}]".strip()
        return base

    @property
    def effective_zone(self) -> int:
        return int(self._zone_effective)

    def _peek_next_task(self) -> Optional[Tuple[float, Dict[str, Any], str]]:
        """Read-only first upcoming task (same priority order as ``pop_next``)."""
        mx_done = float(self._max_completed_shift_hz)
        for hz, p, role in list(self._backfill):
            if math.isfinite(mx_done) and float(hz) <= mx_done + HZ_TOLERANCE:
                continue
            if not self._shift_target_done(hz):
                return (hz, p, role)
        for hz, p, role in list(self._seed_queue):
            if math.isfinite(mx_done) and float(hz) <= mx_done + HZ_TOLERANCE:
                continue
            if not self._shift_target_done(hz):
                return (hz, p, role)
        cur = float(self._cursor_hz)
        step = self._current_step_hz()
        while cur <= self.hz_max + 1e-9:
            hz = float(cur)
            if math.isfinite(mx_done) and hz <= mx_done + HZ_TOLERANCE:
                cur += step
                continue
            if not self._shift_target_done(hz):
                return (hz, self._band_params_with_zone_step(hz, "peek"), "")
            cur += step
        return None

    def log_schedule_snapshot_after_worker(
        self, completed_target_hz: float, stats: Optional[MergeStats]
    ) -> None:
        """Log effective zone and next worker SLEPc quota after a merge (dynamic mode)."""
        nxt = self._peek_next_task()
        y = float(stats.yield_kept_over_raw) if stats is not None else float("nan")
        if nxt is not None:
            nh, np, _ = nxt
            LOGGER.info(
                "Schedule snapshot: effective_zone=%d | merged_shift@%.4f Hz yield=%.4f | "
                "next_solver_batch: target_hz≈%.4f num_modes=%d step_hz=%.1f",
                self.effective_zone,
                completed_target_hz,
                y,
                nh,
                int(np.get("num_modes", 0)),
                float(self._current_step_hz()),
            )
        else:
            LOGGER.info(
                "Schedule snapshot: effective_zone=%d | merged_shift@%.4f Hz yield=%.4f | "
                "next_solver_batch: (queue empty)",
                self.effective_zone,
                completed_target_hz,
                y,
            )

    def try_apply_conductor_ceiling(self, merged_shift_hz: float) -> None:
        """
        Once per sample run: when a completed shift reaches ``CONDUCTOR_TRIGGER_HZ``, set ``hz_max``
        from the current spectral zone (density / participation path already in ``on_worker_merge``).
        """
        if self._conductor_ceiling_applied:
            return
        if float(merged_shift_hz) + 1e-12 < CONDUCTOR_TRIGGER_HZ:
            return
        old = float(self.hz_max)
        sz = int(self._zone_effective)
        new_hi = float(_conductor_ceiling_hz_for_spectral_zone(sz))
        cz, label = _conductor_user_zone_label(sz)
        self.hz_max = new_hi
        self._conductor_ceiling_applied = True
        LOGGER.info(
            "[Conductor] Zone %d (%s) detected at %.0f Hz. Adjusting ceiling from %.0f Hz to %.0f Hz.",
            cz,
            label,
            CONDUCTOR_TRIGGER_HZ,
            old,
            new_hi,
        )

    def merge_context(self) -> MergeContext:
        z = self._zone_effective
        if z == 1:
            return MergeContext(1, ZONE1_MIN_HZ_GAP, ZONE1_SPECTRAL_TAU_SCALE, MIN_UNIQUENESS_FOR_ISOLATION)
        if z == 3:
            return MergeContext(3, MIN_HZ_GAP, 1.0, ZONE3_MIN_UNIQUENESS_FOR_ISOLATION)
        return MergeContext(2, MIN_HZ_GAP, 1.0, MIN_UNIQUENESS_FOR_ISOLATION)

    def pop_next(self) -> Optional[Tuple[float, Dict[str, Any], str]]:
        mx_done = float(self._max_completed_shift_hz)
        while self._backfill:
            hz, p, role = self._backfill.popleft()
            if math.isfinite(mx_done) and float(hz) <= mx_done + HZ_TOLERANCE:
                LOGGER.debug(
                    "Resume skip: backfill %.4f Hz at or behind max completed %.4f Hz.",
                    float(hz),
                    mx_done,
                )
                continue
            if self._shift_target_done(hz):
                LOGGER.debug(
                    "Resume skip: shift %.4f Hz already completed (backfill queue); advancing without worker.",
                    hz,
                )
                continue
            self._scheduled.add(self._hz_key(hz))
            return (hz, p, role)
        while self._seed_queue:
            hz, p, role = self._seed_queue.popleft()
            if math.isfinite(mx_done) and float(hz) <= mx_done + HZ_TOLERANCE:
                LOGGER.debug(
                    "Resume skip: seed %.4f Hz at or behind max completed %.4f Hz.",
                    float(hz),
                    mx_done,
                )
                continue
            if self._shift_target_done(hz):
                LOGGER.debug(
                    "Resume skip: shift %.4f Hz already completed (seed queue); advancing without worker.",
                    hz,
                )
                continue
            self._scheduled.add(self._hz_key(hz))
            return (hz, p, role)
        if not self._seeds_drained:
            self._seeds_drained = True
        step = self._current_step_hz()
        mx_done = float(self._max_completed_shift_hz)
        while self._cursor_hz <= self.hz_max + 1e-9:
            hz = float(self._cursor_hz)
            self._cursor_hz += step
            if math.isfinite(mx_done) and hz <= mx_done + HZ_TOLERANCE:
                LOGGER.debug(
                    "Resume skip: cursor %.4f Hz at or behind max completed %.4f Hz; advancing.",
                    hz,
                    mx_done,
                )
                continue
            key = self._hz_key(hz)
            if key in self._scheduled:
                LOGGER.debug(
                    "Resume skip: shift %.4f Hz already completed (cursor sweep); advancing without worker.",
                    hz,
                )
                continue
            self._scheduled.add(key)
            p = self._band_params_with_zone_step(hz, "dyn")
            return (hz, p, "")
        return None

    def has_pending(self) -> bool:
        if self._backfill or self._seed_queue:
            return True
        return self._cursor_hz <= self.hz_max + 1e-9

    def _classify_raw_zone(self, yield_rate: float, avg_wood: float) -> int:
        if yield_rate > SCOUT_YIELD_DENSE or avg_wood > SCOUT_AVG_WOOD_DENSE:
            return 1
        if yield_rate < SCOUT_YIELD_SPARSE:
            return 3
        return 2

    @staticmethod
    def _efficiency_zone_override(smooth_eff: float) -> Optional[int]:
        """Efficiency-only suggestion: Z1 if dense, Z3 if dead air; else defer to yield/wood."""
        if smooth_eff > EFFICIENCY_HIGH_THRESHOLD:
            return 1
        if smooth_eff < EFFICIENCY_LOW_THRESHOLD:
            return 3
        return None

    def _apply_zone_hysteresis(self, report_hz: float, raw: int, allow_backfill: bool) -> None:
        """Update zone with hysteresis; optional Z3→Z1 backfill when ``allow_backfill`` (scout)."""
        if raw == self._zone_effective:
            self._pending_zone = None
            self._pending_streak = 0
            LOGGER.info(
                "Spectral @ %.4f Hz: stable zone=%d (effective=%d)",
                report_hz,
                raw,
                self._zone_effective,
            )
            return

        if self._pending_zone == raw:
            self._pending_streak += 1
        else:
            self._pending_zone = raw
            self._pending_streak = 1

        if self._pending_streak < ZONE_HYSTERESIS_STREAK:
            LOGGER.info(
                "Spectral @ %.4f Hz: candidate_zone=%d (hysteresis %d/%d, effective=%d)",
                report_hz,
                raw,
                self._pending_streak,
                ZONE_HYSTERESIS_STREAK,
                self._zone_effective,
            )
            return

        prev = self._zone_effective
        LOGGER.info(
            "Spectral @ %.4f Hz: zone %d → %d",
            report_hz,
            prev,
            raw,
        )
        if allow_backfill and prev == 3 and raw == 1:
            self._inject_backfill(report_hz)
        self._zone_effective = raw
        self._pending_zone = None
        self._pending_streak = 0

    def on_worker_merge(
        self, report_hz: float, stats: MergeStats, spectral_step_hz: float, role: str
    ) -> None:
        """
        Efficiency index = ``kept_after_manager / spectral_step_hz`` (modes per Hz of sweep spacing).
        High density forces Z1; low density forces Z3; otherwise yield/wood classification.
        """
        step = max(float(spectral_step_hz), 1e-9)
        instant_raw = float(stats.kept_after_manager) / step
        instant_eff = efficiency_index_sanitized(instant_raw)
        self._efficiency_history.append(instant_eff)
        smooth_eff = (
            float(sum(self._efficiency_history) / len(self._efficiency_history))
            if self._efficiency_history
            else instant_eff
        )

        eff_raw = self._efficiency_zone_override(smooth_eff)
        yield_raw = self._classify_raw_zone(stats.yield_kept_over_raw, stats.avg_wood_raw)
        raw = int(eff_raw) if eff_raw is not None else yield_raw

        pre_zone = int(self._zone_effective)
        self._apply_zone_hysteresis(report_hz, raw, allow_backfill=(role == "scout"))
        post_zone = int(self._zone_effective)
        decision = "Jump" if post_zone != pre_zone else "Stay"
        next_step = float(self._current_step_hz())
        LOGGER.info(
            "[OPTIMIZER] Efficiency: %.2f modes/Hz raw (sanitized smooth=%.2f) | Decision: %s | Next Step: %.1f Hz",
            instant_raw,
            smooth_eff,
            decision,
            next_step,
        )
        self._update_coverage_anchor_state(report_hz, int(stats.kept_after_manager))

    def _inject_backfill(self, scout_hz: float) -> None:
        mx_c = float(self._max_completed_shift_hz)
        lo_scout = max(float(self.hz_min), float(scout_hz) - 28.0)
        lo = lo_scout
        if math.isfinite(mx_c):
            lo = max(lo, mx_c + HZ_TOLERANCE)
        inserted = 0
        h = lo
        while h < scout_hz - 0.5 and inserted < 24:
            hh = round(h, 4)
            if not self._shift_target_done(hh):
                p = self._band_params_with_zone_step(hh, "backfill")
                self._backfill.append((hh, p, ""))
                inserted += 1
            h += BACKFILL_STEP_HZ
        if inserted:
            LOGGER.info(
                "Backfill: injected %d task(s) ≈%.1f–%.1f Hz @ %.1f Hz step (scout=%.2f)",
                inserted,
                lo,
                scout_hz,
                BACKFILL_STEP_HZ,
                scout_hz,
            )


def _wood_floor_for_hz(hz: float) -> float:
    """Linear wood-participation floor from 0.0008 (100 Hz) down to 0.0003 (350 Hz+)."""
    x = float(hz)
    if x <= WOOD_FLOOR_HZ_ANCHOR_LO:
        return WOOD_PARTICIPATION_FLOOR_LO
    if x >= WOOD_FLOOR_HZ_ANCHOR_HI:
        return WOOD_PARTICIPATION_FLOOR_HI
    t = (x - WOOD_FLOOR_HZ_ANCHOR_LO) / (WOOD_FLOOR_HZ_ANCHOR_HI - WOOD_FLOOR_HZ_ANCHOR_LO)
    return WOOD_PARTICIPATION_FLOOR_LO + t * (WOOD_PARTICIPATION_FLOOR_HI - WOOD_PARTICIPATION_FLOOR_LO)


def _wood_floor_for_hz_v2(hz: float) -> float:
    """Linear wood floor 0.0005 @ 100 Hz → 0.0003 @ 450 Hz (merge veto; not tied to zone quota)."""
    x = float(hz)
    if x <= WOOD_V2_LO_HZ:
        return WOOD_V2_LO
    if x >= WOOD_V2_HI_HZ:
        return WOOD_V2_HI
    t = (x - WOOD_V2_LO_HZ) / (WOOD_V2_HI_HZ - WOOD_V2_LO_HZ)
    return WOOD_V2_LO + t * (WOOD_V2_HI - WOOD_V2_LO)


def _peer_min_hz_gap(hz: float, peer_hz: List[float]) -> float:
    if not peer_hz:
        return 1.0e6
    return min(abs(float(hz) - float(ph)) for ph in peer_hz)


def _effective_wood_floor_v2(hz: float, peer_hz: List[float]) -> float:
    """>15 Hz from all peer freqs in this merge batch → floor ×0.6 (40% reduction); applies in merge only."""
    base = _wood_floor_for_hz_v2(hz)
    if _peer_min_hz_gap(hz, peer_hz) > ISO_BATCH_GAP_HZ:
        return base * ISOLATION_WOOD_FLOOR_SCALE
    return base


def _passes_zone_wood_veto(c: Dict[str, Any]) -> bool:
    """Priority-1 gate: wood participation vs zone-specific floor only."""
    try:
        w = float(c.get("wood_participation", 0.0) or 0.0)
        hz = float(c.get("hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(w) or not math.isfinite(hz):
        return False
    return w >= _wood_floor_for_hz(hz)


def _passes_zone_wood_veto_v2(c: Dict[str, Any], peer_hz: List[float]) -> bool:
    """V2 wood gate with frequency-linear floor and batch isolation relief."""
    try:
        w = float(c.get("wood_participation", 0.0) or 0.0)
        hz = float(c.get("hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(w) or not math.isfinite(hz):
        return False
    return w >= _effective_wood_floor_v2(hz, peer_hz)


def _spectral_multiplier(df_nearest: float, tau_scale: float = 1.0) -> float:
    """
    Penalize small Hz gaps (< 3 Hz) exponentially; reward large gaps (> 15 Hz) linearly
    with slope capped as if Δf were at most 40 Hz.
    """
    d = float(df_nearest)
    mult = 1.0
    tau = max(SPECTRAL_PENALTY_TAU_HZ * float(tau_scale), 1e-9)
    if d < SPECTRAL_CLOSE_HZ:
        mult *= math.exp(-max(0.0, (SPECTRAL_CLOSE_HZ - d) / tau))
    if d > SPECTRAL_FAR_HZ:
        span = min(d - SPECTRAL_FAR_HZ, SPECTRAL_FAR_CAP_HZ - SPECTRAL_FAR_HZ)
        if span > 0.0:
            mult *= 1.0 + SPECTRAL_REWARD_SLOPE * span
    return mult


def _isolation_bonus(
    hz: float,
    w: float,
    uniq: float,
    df_nearest: float,
    min_uniqueness_iso: float = MIN_UNIQUENESS_FOR_ISOLATION,
) -> float:
    """Secondary weight: only if wood veto already passed and uniqueness meets floor."""
    if uniq < float(min_uniqueness_iso):
        return 0.0
    d = max(0.0, float(df_nearest))
    iso_shape = math.sqrt(d / max(ISO_DISTANCE_REF_HZ, 1e-9))
    wood_mix = ISO_WOOD_WEIGHT * float(w) + ISO_AIR_DAMPING * max(0.0, 1.0 - float(w))
    return iso_shape * wood_mix


def _max_sparse_overlap(csr_c: sparse.csr_matrix, others: List[sparse.csr_matrix]) -> float:
    if not others:
        return 0.0
    mx = 0.0
    for o in others:
        mx = max(mx, csr_normalized_overlap(csr_c, o))
    return mx


def _row_key(c: Dict[str, Any]) -> Tuple[float, str]:
    return (float(c.get("hz", 0.0) or 0.0), str(c.get("vector_path", "") or ""))


def _is_resolved_path_under_root(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` or a descendant (after resolve). Robust vs ``Path.parents`` identity quirks."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_vector_abs(row: Dict[str, Any], path_root: Path) -> Optional[Path]:
    """
    Resolve ``row['vector_path']`` (relative to ``path_root``) to an existing file.
    Logs the full candidate path whenever resolution fails (no silent skip).
    """
    rel = Path(str(row.get("vector_path", "") or ""))
    root = path_root.resolve()
    if not rel.parts:
        LOGGER.warning(
            "merge: missing vector_path (path_root=%s row hz=%r id=%r)",
            root,
            row.get("hz"),
            row.get("id"),
        )
        return None
    p = (root / rel).resolve()
    if not p.is_file():
        LOGGER.warning(
            "merge: mode vector not found | path_root=%s | vector_path=%s | resolved=%s | parent_exists=%s",
            root,
            rel.as_posix(),
            p,
            p.parent.is_dir(),
        )
        return None
    if not _is_resolved_path_under_root(p, root):
        LOGGER.warning(
            "merge: resolved vector outside path_root (refusing) | path_root=%s | resolved=%s",
            root,
            p,
        )
        return None
    return p


def _load_batch_csr_pairs(
    rows: List[Dict[str, Any]],
    sorting_root: Path,
) -> List[Tuple[Dict[str, Any], sparse.csr_matrix]]:
    out: List[Tuple[Dict[str, Any], sparse.csr_matrix]] = []
    for i, c in enumerate(rows):
        p = _resolve_vector_abs(c, sorting_root)
        if p is None:
            continue
        try:
            mat = load_mode_column_any(p).tocsr().astype(np.float32, copy=False)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable mode vector %s: %s", p, exc)
            continue
        out.append((dict(c), mat))
        if (i + 1) % 32 == 0:
            gc.collect()
    gc.collect()
    return out


def _load_existing_csr_pairs(
    existing: List[Dict[str, Any]],
    sorting_root: Path,
) -> Tuple[List[float], List[sparse.csr_matrix]]:
    hz_list: List[float] = []
    csr_list: List[sparse.csr_matrix] = []
    for i, e in enumerate(existing):
        p = _resolve_vector_abs(e, sorting_root)
        if p is None:
            continue
        try:
            mat = load_mode_column_any(p).tocsr().astype(np.float32, copy=False)
        except Exception as exc:
            LOGGER.warning("Skipping existing log vector %s: %s", p, exc)
            continue
        try:
            hz_list.append(float(e.get("hz", 0.0) or 0.0))
        except (TypeError, ValueError):
            hz_list.append(0.0)
        csr_list.append(mat)
        if (i + 1) % 64 == 0:
            gc.collect()
    gc.collect()
    return hz_list, csr_list


def _df_nearest(hz: float, hz_ref: List[float]) -> float:
    if not hz_ref:
        return 1.0e6
    return min(abs(float(hz) - float(h)) for h in hz_ref)


def _master_mode_score(
    hz: float,
    w: float,
    uniq: float,
    csr_c: sparse.csr_matrix,
    existing_hz: List[float],
    existing_csr: List[sparse.csr_matrix],
    picked_hz: List[float],
    picked_csr: List[sparse.csr_matrix],
    *,
    spectral_tau_scale: float = 1.0,
    min_uniqueness_iso: float = MIN_UNIQUENESS_FOR_ISOLATION,
) -> float:
    hz_ref = list(existing_hz) + list(picked_hz)
    df_n = _df_nearest(hz, hz_ref)
    spec = _spectral_multiplier(df_n, tau_scale=spectral_tau_scale)
    max_ov = _max_sparse_overlap(csr_c, existing_csr + picked_csr)
    shape = float(np.clip(1.0 - max_ov, 0.0, 1.0))
    iso = _isolation_bonus(hz, w, uniq, df_n, min_uniqueness_iso=min_uniqueness_iso)
    return shape * spec + iso


def _cluster_by_min_gap(rows_hz: List[Tuple[Dict[str, Any], sparse.csr_matrix]], gap: float) -> List[List[int]]:
    """Return cluster index lists into ``rows_hz`` sorted by ascending hz."""
    order = sorted(range(len(rows_hz)), key=lambda i: float(rows_hz[i][0].get("hz", 0.0) or 0.0))
    clusters: List[List[int]] = []
    if not order:
        return clusters
    cur = [order[0]]
    anchor = float(rows_hz[order[0]][0].get("hz", 0.0) or 0.0)
    for idx in order[1:]:
        hz = float(rows_hz[idx][0].get("hz", 0.0) or 0.0)
        if hz - anchor <= gap:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
            anchor = hz
    clusters.append(cur)
    return clusters


def _adaptive_manager_select(
    batch: List[Tuple[Dict[str, Any], sparse.csr_matrix]],
    existing: List[Dict[str, Any]],
    sorting_root: Path,
    merge_ctx: MergeContext,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """
    Zone-wood-vetted batch (with CSR) vs log: cluster in Hz, score, pick winners,
    then enforce Zone C quota via sparse-aware swaps.
    Returns (kept_rows, discarded_rows, score_by_row_key).
    """
    if not batch:
        return [], [], {}

    existing_hz, existing_csr = _load_existing_csr_pairs(existing, sorting_root)
    score_map: Dict[str, float] = {}

    clusters = _cluster_by_min_gap(batch, float(merge_ctx.min_hz_gap))
    picked_rows: List[Dict[str, Any]] = []
    picked_csr: List[sparse.csr_matrix] = []
    picked_hz: List[float] = []
    cluster_losers: List[Dict[str, Any]] = []
    chosen_idx: Set[int] = set()

    for cl in clusters:
        best_i: Optional[int] = None
        best_s = -1.0e30
        for i in cl:
            row, csr_i = batch[i]
            try:
                hz = float(row.get("hz", 0.0) or 0.0)
                w = float(row.get("wood_participation", 0.0) or 0.0)
                uniq = float(row.get("uniqueness", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            s = _master_mode_score(
                hz,
                w,
                uniq,
                csr_i,
                existing_hz,
                existing_csr,
                picked_hz,
                picked_csr,
                spectral_tau_scale=float(merge_ctx.spectral_tau_scale),
                min_uniqueness_iso=float(merge_ctx.min_uniqueness_iso),
            )
            rk = str(_row_key(row))
            score_map[rk] = s
            if s > best_s:
                best_s = s
                best_i = i
        if best_i is None:
            for i in cl:
                cluster_losers.append(dict(batch[i][0]))
            continue
        for i in cl:
            if i == best_i:
                continue
            cluster_losers.append(dict(batch[i][0]))
        chosen_idx.add(best_i)
        br, bcsr = batch[best_i]
        picked_rows.append(dict(br))
        picked_csr.append(bcsr)
        picked_hz.append(float(br.get("hz", 0.0) or 0.0))

    gc.collect()

    # --- Zone C minimum quota (350 Hz+): swap out weakest LF winners for best unused HF ---
    def _hz(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("hz", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    swap_drops: List[Dict[str, Any]] = []
    hf_count = sum(1 for r in picked_rows if _hz(r) >= ZONE_C_MIN_HZ)
    need = max(0, ZONE_C_MIN_QUOTA - hf_count)
    if need > 0:
        unused_hf: List[Tuple[int, Dict[str, Any], sparse.csr_matrix, float]] = []
        for i, (row, csr_i) in enumerate(batch):
            if i in chosen_idx:
                continue
            if _hz(row) < ZONE_C_MIN_HZ:
                continue
            try:
                w = float(row.get("wood_participation", 0.0) or 0.0)
                uniq = float(row.get("uniqueness", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            s = _master_mode_score(
                float(row.get("hz", 0.0) or 0.0),
                w,
                uniq,
                csr_i,
                existing_hz,
                existing_csr,
                picked_hz,
                picked_csr,
                spectral_tau_scale=float(merge_ctx.spectral_tau_scale),
                min_uniqueness_iso=float(merge_ctx.min_uniqueness_iso),
            )
            unused_hf.append((i, dict(row), csr_i, s))
        unused_hf.sort(key=lambda t: t[3], reverse=True)

        lf_indices = [j for j, r in enumerate(picked_rows) if _hz(r) < ZONE_C_MIN_HZ]
        lf_scores: List[Tuple[int, float]] = []
        for j in lf_indices:
            r = picked_rows[j]
            rk = str(_row_key(r))
            lf_scores.append((j, float(score_map.get(rk, 0.0))))
        lf_scores.sort(key=lambda t: t[1])

        swaps = 0
        hf_ptr = 0
        lf_ptr = 0
        while swaps < need and hf_ptr < len(unused_hf) and lf_ptr < len(lf_scores):
            j_drop, s_lf = lf_scores[lf_ptr]
            i_add, row_add, csr_add, s_hf = unused_hf[hf_ptr]
            if s_hf <= s_lf + 1e-12:
                hf_ptr += 1
                continue
            old = picked_rows[j_drop]
            swap_drops.append(dict(old))
            picked_rows[j_drop] = row_add
            picked_csr[j_drop] = csr_add
            picked_hz[j_drop] = _hz(row_add)
            rk_old = str(_row_key(old))
            score_map.pop(rk_old, None)
            score_map[str(_row_key(row_add))] = s_hf
            chosen_idx.add(i_add)
            swaps += 1
            hf_ptr += 1
            lf_ptr += 1

    picked_keys = {str(_row_key(r)) for r in picked_rows}
    discarded = [c for c in cluster_losers if str(_row_key(c)) not in picked_keys]
    discarded.extend(swap_drops)

    gc.collect()
    return picked_rows, discarded, score_map


def _unlink_worker_vector(row: Dict[str, Any], path_root: Path) -> None:
    rel = Path(str(row.get("vector_path", "") or ""))
    if not rel.parts:
        return
    root = path_root.resolve()
    p = (root / rel).resolve()
    try:
        if p.is_file() and _is_resolved_path_under_root(p, root):
            p.unlink()
    except OSError as exc:
        LOGGER.warning("Could not remove discarded mode vector %s: %s", p, exc)


def build_task_list(hz_min: float, hz_max: float) -> List[Tuple[float, Dict[str, Any]]]:
    """Build worker target Hz list from ``hz_min`` (inclusive) through ``hz_max`` (inclusive)."""
    lo = float(hz_min)
    hi = float(hz_max)
    if lo < SWEEP_HZ_MIN:
        raise ValueError(
            f"hz_min must be >= {SWEEP_HZ_MIN} (band tables start at {SWEEP_HZ_MIN:g} Hz), got {lo}"
        )
    if hi < lo:
        raise ValueError(f"hz_max ({hi}) must be >= hz_min ({lo})")
    tasks: List[Tuple[float, Dict[str, Any]]] = []
    hz = lo
    while hz <= hi + 1e-9:
        p = dict(get_band_params(hz))
        tasks.append((hz, p))
        hz += float(p["step_hz"])
    return tasks


def _passes_harvest_gate(
    c: Dict[str, Any],
    *,
    st_sigma_hz: float,
    structural_only_run: bool = False,
) -> Tuple[bool, str]:
    """
    Coupled-mode harvest eligibility for merge.

    Returns ``(merge_eligible, tag)`` where ``tag`` is one of:
    ``coupled_fsi``, ``structural_spurious``, ``sigma_locked``, ``low_wood``.
    """
    try:
        wood = float(c.get("wood_participation", 0.0) or 0.0)
    except (TypeError, ValueError):
        wood = 0.0
    try:
        p_frac = float(c.get("p_frac", 0.0) or 0.0)
    except (TypeError, ValueError):
        p_frac = 0.0
    try:
        f_hz = float(c.get("hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        f_hz = 0.0

    sigma = float(st_sigma_hz)
    if abs(f_hz - sigma) < HARVEST_GATE_SIGMA_TOL_HZ and p_frac < HARVEST_GATE_SIGMA_P_FRAC:
        return False, "sigma_locked"

    if wood >= HARVEST_GATE_MIN_WOOD:
        if p_frac >= HARVEST_GATE_MIN_P_FRAC_FSI:
            return True, "coupled_fsi"
        c["harvest_tag"] = "structural_spurious"
        if structural_only_run:
            return True, "structural_spurious"
        return False, "structural_spurious"

    return False, "low_wood"


def _merge_result_into_candidates_log(
    result_path: Path,
    log_path: Path,
    lock: threading.Lock,
    sorting_root: Path,
    *,
    override_root: Optional[Path] = None,
    merge_ctx: Optional[MergeContext] = None,
    force_emergency: bool = False,
) -> Optional[MergeStats]:
    """
    Merge one worker ``result_*.json`` into ``candidates_log.json``.

    ``sorting_root`` is the workspace that holds ``temp_results/`` and ``candidates_log.json``.
    ``override_root`` (LAB / custom layouts): when set, resolve ``vector_path`` and unlink
    worker vectors under this directory instead of ``sorting_root``. Defaults to ``sorting_root``.
    ``merge_ctx`` tunes clustering / spectral MMR penalty and isolation floor (from spectral zone).
    Returns merge statistics for scout feedback (or ``None`` if nothing merged).
    """
    if not result_path.is_file():
        return None
    try:
        incoming = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Skipping unreadable result file %s: %s", result_path, exc)
        return None

    raw = list(incoming.get("candidates") or [])
    if not raw:
        try:
            if not force_emergency:
                result_path.unlink()
        except OSError:
            pass
        return None

    path_root = (override_root if override_root is not None else sorting_root).resolve()
    ghost_drops = 0
    raw_pruned: List[Dict[str, Any]] = []
    for c in raw:
        nz = c.get("column_l2_norm")
        if nz is not None:
            try:
                if (not force_emergency) and float(nz) < WORKER_COL_NORM_MIN:
                    ghost_drops += 1
                    _unlink_worker_vector(c, path_root)
                    continue
            except (TypeError, ValueError):
                pass
        raw_pruned.append(c)
    raw = raw_pruned
    if ghost_drops:
        LOGGER.info(
            "Merge: dropped %d worker candidate(s) with column_l2_norm < %.1e (ghost / null displacement).",
            ghost_drops,
            WORKER_COL_NORM_MIN,
        )
    if not raw:
        try:
            if not force_emergency:
                result_path.unlink()
        except OSError:
            pass
        return None

    uniq_drops = 0
    raw_uq: List[Dict[str, Any]] = []
    for c in raw:
        try:
            ru = c.get("uniqueness", None)
            u = float(ru) if ru is not None else -1.0
        except (TypeError, ValueError):
            u = -1.0
        if (not force_emergency) and u < MERGE_INCOMING_UNIQUENESS_MIN - 1e-15:
            uniq_drops += 1
            _unlink_worker_vector(c, path_root)
            continue
        raw_uq.append(c)
    raw = raw_uq
    if uniq_drops:
        LOGGER.info(
            "Merge: dropped %d incoming candidate(s) with uniqueness < %.2f (master safety net).",
            uniq_drops,
            MERGE_INCOMING_UNIQUENESS_MIN,
        )
    if not raw:
        try:
            if not force_emergency:
                result_path.unlink()
        except OSError:
            pass
        return None

    structural_only_run = bool(incoming.get("structural_only_run", False))
    try:
        st_sigma_hz = float(
            incoming.get("st_sigma_hz", incoming.get("target_hz", 0.0)) or 0.0
        )
    except (TypeError, ValueError):
        st_sigma_hz = 0.0
    if st_sigma_hz <= 0.0:
        tfn_sigma = _target_hz_from_result_filename(result_path)
        if tfn_sigma is not None:
            st_sigma_hz = float(tfn_sigma)

    harvest_gate_drops = 0
    gate_kept: List[Dict[str, Any]] = []
    coupled_valid_pre = 0
    for c in raw:
        eligible, tag = _passes_harvest_gate(
            c,
            st_sigma_hz=st_sigma_hz,
            structural_only_run=structural_only_run,
        )
        if eligible:
            if tag == "coupled_fsi":
                coupled_valid_pre += 1
            gate_kept.append(c)
        else:
            harvest_gate_drops += 1
            if not force_emergency:
                _unlink_worker_vector(c, path_root)
    if harvest_gate_drops:
        LOGGER.info(
            "Harvest gate: dropped %d candidate(s) (σ=%.4f Hz, structural_only=%s); "
            "kept %d (coupled_fsi=%d).",
            harvest_gate_drops,
            st_sigma_hz,
            structural_only_run,
            len(gate_kept),
            coupled_valid_pre,
        )
    raw = gate_kept
    if not raw:
        try:
            if not force_emergency:
                result_path.unlink()
        except OSError:
            pass
        return MergeStats(
            raw_n=0,
            kept_after_veto=0,
            kept_after_manager=0,
            avg_wood_raw=0.0,
            yield_kept_over_raw=0.0,
            coupled_valid_kept=0,
        )

    raw_n = len(raw)
    woods: List[float] = []
    for c in raw:
        try:
            woods.append(float(c.get("wood_participation", 0.0) or 0.0))
        except (TypeError, ValueError):
            woods.append(0.0)
    avg_wood_raw = float(np.mean(woods)) if woods else 0.0

    tgt_early: Optional[float] = None
    try:
        tz = incoming.get("target_hz")
        if tz is not None:
            tgt_early = hz_shift_quantize(float(tz))
    except (TypeError, ValueError):
        tgt_early = None
    if tgt_early is None:
        tfn = _target_hz_from_result_filename(result_path)
        if tfn is not None:
            tgt_early = hz_shift_quantize(float(tfn))
    skip_completed_shift = False
    if tgt_early is not None:
        with lock:
            if log_path.exists():
                try:
                    pl_early = json.loads(log_path.read_text(encoding="utf-8"))
                except Exception:
                    pl_early = {}
            else:
                pl_early = {}
            done_early_list: List[float] = []
            for x in pl_early.get("completed_shift_targets") or []:
                try:
                    done_early_list.append(hz_shift_quantize(float(x)))
                except (TypeError, ValueError):
                    continue
            skip_completed_shift = hz_any_matches_completed_shift(float(tgt_early), done_early_list)
    if skip_completed_shift:
        LOGGER.info(
            "Merge idempotent: shift %.6f Hz already in completed_shift_targets; skipping merge.",
            float(tgt_early),
        )
        if not force_emergency:
            for c in raw:
                _unlink_worker_vector(c, path_root)
            try:
                result_path.unlink()
            except OSError:
                pass
        return MergeStats(
            raw_n=raw_n,
            kept_after_veto=0,
            kept_after_manager=0,
            avg_wood_raw=avg_wood_raw,
            yield_kept_over_raw=0.0,
        )

    # Wood participation absolute floor veto disabled: finer meshes scale L2-normalized
    # tag ratios down; downstream split-quota MMR in dynamic_filter_tuner ranks relatively.

    all_hz: List[float] = []
    for c in raw:
        try:
            all_hz.append(float(c.get("hz", 0.0) or 0.0))
        except (TypeError, ValueError):
            all_hz.append(0.0)
    finite_hz = [float(h) for h in all_hz if math.isfinite(float(h))]
    if (not force_emergency) and len(finite_hz) >= MERGE_SHIFT_CLUSTER_MIN_MODES:
        hz_span = float(max(finite_hz) - min(finite_hz))
        if hz_span < MERGE_SHIFT_CLUSTER_SPAN_HZ - 1e-12:
            LOGGER.warning(
                "Suspect shift (density ceiling): %d modes within %.4f Hz span (< %.1f Hz); "
                "rejecting merge for %s (numerical duplicate cluster / poisoned batch).",
                len(finite_hz),
                hz_span,
                MERGE_SHIFT_CLUSTER_SPAN_HZ,
                result_path.name,
            )
            for c in raw:
                _unlink_worker_vector(c, path_root)
            try:
                result_path.unlink()
            except OSError:
                pass
            return MergeStats(
                raw_n=raw_n,
                kept_after_veto=0,
                kept_after_manager=0,
                avg_wood_raw=avg_wood_raw,
                yield_kept_over_raw=0.0,
            )

    veto_pass: List[Dict[str, Any]] = list(raw)
    failed_veto: List[Dict[str, Any]] = []

    batch_loaded = _load_batch_csr_pairs(veto_pass, path_root)
    if not batch_loaded:
        LOGGER.warning("No loadable sparse vectors after wood veto for %s.", result_path.name)
        if not force_emergency:
            for c in raw:
                _unlink_worker_vector(c, path_root)
            try:
                result_path.unlink()
            except OSError:
                pass
        return MergeStats(
            raw_n=raw_n,
            kept_after_veto=len(veto_pass),
            kept_after_manager=0,
            avg_wood_raw=avg_wood_raw,
            yield_kept_over_raw=0.0,
        )

    n_rows_appended = 0
    with lock:
        if log_path.exists():
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"candidates": [], "completed_shift_targets": []}
        else:
            payload = {"candidates": [], "completed_shift_targets": []}
        payload.setdefault("completed_shift_targets", list(payload.get("completed_shift_targets") or []))
        existing = list(payload.get("candidates") or [])

        if force_emergency:
            thin_kept = [dict(row) for row, _ in batch_loaded]
            thin_discarded: List[Dict[str, Any]] = []
        else:
            # Pass-through: retain all worker modes; MMR selection runs in dynamic_filter_tuner.
            thin_kept = [dict(row) for row, _ in batch_loaded]
            thin_discarded = []

        dropped_total = raw_n - len(thin_kept)
        if dropped_total > 0:
            LOGGER.info(
                "Adaptive merge: raw=%d wood_veto_fail=%d manager_discard=%d kept=%d.",
                raw_n,
                len(failed_veto),
                len(thin_discarded),
                len(thin_kept),
            )

        if not thin_kept:
            LOGGER.warning(
                "No candidates left after adaptive manager for %s; removing result file and worker vectors.",
                result_path.name,
            )
            if not force_emergency:
                for c in raw:
                    _unlink_worker_vector(c, path_root)
                try:
                    result_path.unlink()
                except OSError:
                    pass
            gc.collect()
            return MergeStats(
                raw_n=raw_n,
                kept_after_veto=len(veto_pass),
                kept_after_manager=0,
                avg_wood_raw=avg_wood_raw,
                yield_kept_over_raw=0.0,
            )

        mx = max((int(c.get("id", -1)) for c in existing), default=-1)
        add_i = 0
        for rec in thin_kept:
            row = dict(rec)
            rel_old = Path(str(row.get("vector_path", "")))
            old_abs = (path_root / rel_old).resolve()
            try:
                nh = float(row.get("hz", 0.0) or 0.0)
            except (TypeError, ValueError):
                nh = float("nan")
            row_dupe = False
            if (not force_emergency) and math.isfinite(nh):
                for e in existing:
                    try:
                        eh = float(e.get("hz", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(eh) and math.isfinite(nh) and hz_shift_markers_match(eh, nh):
                        row_dupe = True
                        break
            if row_dupe:
                LOGGER.info(
                    "Merge idempotent: skip new log row (hz≈%.6f Hz matches existing candidate).",
                    nh,
                )
                if old_abs.is_file() and not force_emergency:
                    try:
                        old_abs.unlink()
                    except OSError:
                        pass
                continue
            new_id = int(mx + 1 + add_i)
            add_i += 1
            new_rel = Path("temp_modes") / f"mode_{new_id:06d}{MODE_VECTOR_FILE_SUFFIX}"
            new_abs = (path_root / new_rel).resolve()
            if old_abs.is_file():
                if new_abs != old_abs:
                    if new_abs.exists():
                        new_abs.unlink()
                    old_abs.rename(new_abs)
            row["id"] = new_id
            row["vector_path"] = str(new_rel).replace("\\", "/")
            existing.append(row)
        n_rows_appended = add_i
        payload["candidates"] = existing
        tmp = log_path.with_suffix(log_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(log_path)

    del batch_loaded
    del thin_discarded
    gc.collect()

    if not force_emergency:
        try:
            result_path.unlink()
        except OSError:
            LOGGER.warning("Could not delete merged result file: %s", result_path)

    gc.collect()
    return MergeStats(
        raw_n=raw_n,
        kept_after_veto=len(veto_pass),
        kept_after_manager=n_rows_appended,
        avg_wood_raw=avg_wood_raw,
        yield_kept_over_raw=(n_rows_appended / float(raw_n)) if raw_n > 0 else 0.0,
        coupled_valid_kept=coupled_valid_pre,
    )


def _append_completed_shift_to_log(log_path: Path, lock: threading.Lock, target_hz: float) -> None:
    """Persist ``completed_shift_targets`` (idempotent, ``HZ_TOLERANCE``-aware) for resume across runs."""
    key = hz_shift_quantize(float(target_hz))
    with lock:
        if not log_path.exists():
            payload: Dict[str, Any] = {"candidates": [], "completed_shift_targets": []}
        else:
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"candidates": [], "completed_shift_targets": []}
        arr = [hz_shift_quantize(float(x)) for x in (payload.get("completed_shift_targets") or [])]
        if hz_any_matches_completed_shift(key, arr):
            return
        arr.append(float(key))
        arr.sort()
        payload["completed_shift_targets"] = arr
        tmp = log_path.with_suffix(log_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(log_path)


def flush_pending_worker_shifts(
    sorting_root: Path,
    log_path: Path,
    lock: threading.Lock,
    *,
    force_emergency: bool = False,
) -> int:
    """Merge leftover ``temp_results/result_*.json`` before scheduling (resume-safe)."""
    tr = sorting_root / "temp_results"
    if not tr.is_dir():
        return 0
    paths = sorted(tr.glob("result_*.json"))
    n = 0
    for p in paths:
        th = _target_hz_from_result_filename(p)
        stats = _merge_result_into_candidates_log(
            p, log_path, lock, sorting_root, merge_ctx=MergeContext(), force_emergency=force_emergency
        )
        if th is not None and stats is not None:
            _append_completed_shift_to_log(log_path, lock, float(th))
        n += 1
    if n:
        LOGGER.info("Resume flush: merged %d pending worker result file(s) under temp_results/.", n)
    return n


def _poll_completed(
    running: Dict[subprocess.Popen, Dict[str, Any]],
    log_path: Path,
    sorting_root: Path,
    merge_lock: threading.Lock,
    release_core: Callable[[Optional[int]], None],
    scheduler: Optional[SpectralScheduler] = None,
    *,
    force_emergency: bool = False,
    spawn_worker: Optional[Callable[..., None]] = None,
) -> None:
    for proc, meta in list(running.items()):
        code = proc.poll()
        if code is None:
            continue
        hz = float(meta["hz"])
        rpath = result_json_path(sorting_root, hz)
        try:
            if code == 0:
                LOGGER.info("Worker finished OK for %.4f Hz (exit %s).", hz, code)
                ctx = scheduler.merge_context() if scheduler is not None else None
                stats = _merge_result_into_candidates_log(
                    rpath,
                    log_path,
                    merge_lock,
                    sorting_root,
                    merge_ctx=ctx,
                    force_emergency=force_emergency,
                )
                if stats is not None:
                    _append_completed_shift_to_log(log_path, merge_lock, hz)
                    if scheduler is not None:
                        scheduler.register_completed_shift(hz)
                        sp_step = float(meta.get("spectral_step_hz", ZONE2_STEP_HZ))
                        scheduler.on_worker_merge(
                            hz, stats, sp_step, str(meta.get("role", "") or "")
                        )
                        if not force_emergency:
                            scheduler.try_apply_conductor_ceiling(hz)
                if scheduler is not None:
                    scheduler.log_schedule_snapshot_after_worker(hz, stats)
                if (
                    spawn_worker is not None
                    and stats is not None
                    and int(stats.coupled_valid_kept) == 0
                    and not meta.get("structural_only")
                    and not meta.get("structural_fallback")
                    and not force_emergency
                ):
                    par_fb = dict(meta.get("params") or get_band_params(hz))
                    LOGGER.warning(
                        "Coupled harvest: 0 true FSI modes (wood+p_frac) at %.4f Hz; "
                        "spawning structural-only fallback worker.",
                        hz,
                    )
                    spawn_worker(
                        hz,
                        par_fb,
                        "structural-fallback",
                        structural_only=True,
                        structural_fallback=True,
                    )
            else:
                LOGGER.warning("Worker exited with code %s for %.4f Hz (no merge).", code, hz)
        finally:
            release_core(meta.get("core_id"))  # type: ignore[arg-type]
            del running[proc]


def _enforce_timeouts(
    running: Dict[subprocess.Popen, Dict[str, Any]],
    sorting_root: Path,
    release_core: Callable[[Optional[int]], None],
) -> None:
    now = time.monotonic()
    for proc, meta in list(running.items()):
        if proc.poll() is not None:
            continue
        if now <= float(meta["deadline"]):
            continue
        hz = float(meta["hz"])
        LOGGER.warning(
            "TIMEOUT: killing worker for %.4f Hz (limit %.1f min).",
            hz,
            float(meta["timeout_minutes"]),
        )
        try:
            try:
                proc.kill()
            except OSError as exc:
                LOGGER.warning("kill() failed for %.4f Hz: %s", hz, exc)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                LOGGER.warning("wait() after kill timed out for %.4f Hz", hz)
            rpath = result_json_path(sorting_root, hz)
            if rpath.is_file():
                try:
                    rpath.unlink()
                except OSError:
                    pass
        finally:
            release_core(meta.get("core_id"))  # type: ignore[arg-type]
            del running[proc]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Master scheduler for fem_worker_single.py")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "FEM" / "configs" / "guitar_3d.json",
        help="Passed through to each worker.",
    )
    parser.add_argument(
        "--hz-min",
        type=float,
        default=SWEEP_HZ_MIN,
        help=f"Sweep lower bound (Hz), inclusive. Must be >= {SWEEP_HZ_MIN:g} (default: {SWEEP_HZ_MIN:g}).",
    )
    parser.add_argument(
        "--hz-max",
        type=float,
        default=480.0,
        help=(
            "Sweep upper bound (Hz), inclusive (default: 480). Dynamic scheduler may lower ceiling "
            f"once at {CONDUCTOR_TRIGGER_HZ:.0f} Hz from spectral zone (see [Conductor] log line)."
        ),
    )
    parser.add_argument(
        "--sorting-root",
        type=Path,
        default=None,
        help=(
            "Directory containing temp_modes/, temp_results/, and candidates_log.json "
            "(default: FEM/SORTING under repo root). Use a lab copy to avoid touching the main log."
        ),
    )
    parser.add_argument(
        "--use-mpiexec",
        action="store_true",
        help=(
            "Linux: `taskset -c <1..N> mpiexec --bind-to none -n 1 <python> ...` "
            "where N=--max-workers; `--bind-to none` avoids Open MPI overriding taskset."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            f"Maximum concurrent worker subprocesses (default: {DEFAULT_MAX_WORKERS}). "
            "Also controls scout seed batch width and Linux taskset core leasing."
        ),
    )
    parser.add_argument(
        "--force-emergency",
        action="store_true",
        help=(
            "Coverage-anchor run mode: disable conductor pruning (wood/manager filtering and vector/result cleanup) "
            "and disable adaptive ceiling reduction so full sweep coverage is preserved."
        ),
    )
    parser.add_argument(
        "--legacy-static-schedule",
        action="store_true",
        help=(
            "Use legacy fixed stepping from get_band_params (no scout zones / backfill). "
            "Default: scout-seeded dynamic scheduler with zone hysteresis."
        ),
    )
    args = parser.parse_args()

    worker_script = SCRIPT_DIR / "fem_worker_single.py"
    if not worker_script.is_file():
        LOGGER.error("Worker script not found: %s", worker_script)
        return 1

    sorting_root = (
        args.sorting_root.resolve()
        if args.sorting_root is not None
        else (REPO_ROOT / "FEM" / "SORTING").resolve()
    )
    log_path = sorting_root / "candidates_log.json"
    merge_lock = threading.Lock()

    sorting_root.mkdir(parents=True, exist_ok=True)
    (sorting_root / "temp_results").mkdir(parents=True, exist_ok=True)
    (sorting_root / "temp_modes").mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(
            json.dumps({"candidates": [], "completed_shift_targets": []}, indent=2),
            encoding="utf-8",
        )

    flush_pending_worker_shifts(
        sorting_root, log_path, merge_lock, force_emergency=bool(args.force_emergency)
    )

    LOGGER.info(
        "sorting_root=%s — each worker is spawned with the same --sorting-root so vectors "
        "and temp_results JSON land here (matches merge path resolution).",
        sorting_root,
    )

    max_workers = int(args.max_workers)
    if max_workers < 1:
        LOGGER.error("--max-workers must be >= 1 (got %d).", max_workers)
        return 1
    logical_cores = int(os.cpu_count() or 1)
    max_allowed_workers = max(1, logical_cores - 1)
    if max_workers > max_allowed_workers:
        LOGGER.error(
            "--max-workers=%d exceeds available worker capacity on this host (logical cores=%d, max allowed=%d with one core reserved for OS/master).",
            max_workers,
            logical_cores,
            max_allowed_workers,
        )
        return 1

    use_taskset = sys.platform.startswith("linux")
    scheduler: Optional[SpectralScheduler] = None
    tasks_static: Optional[List[Tuple[float, Dict[str, Any]]]] = None
    if args.legacy_static_schedule:
        try:
            tasks_static = build_task_list(float(args.hz_min), float(args.hz_max))
        except ValueError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        scheduler = SpectralScheduler(
            float(args.hz_min),
            float(args.hz_max),
            max_workers=max_workers,
            sorting_root=sorting_root,
            merge_lock=merge_lock,
        )

    worker_cores = list(range(1, max_workers + 1))
    cores_fmt = "{" + ",".join(str(c) for c in worker_cores) + "}"
    linux_pin_msg = f" Linux: taskset cores leased from {cores_fmt}." if use_taskset else ""
    if scheduler is not None:
        LOGGER.info(
            "Dynamic scout scheduler: scout W0=f_start+(N-1)*%.0f Hz, others fill descending to f_start; "
            "zones Z1 step=%.0f Z2=%.0f Z3=%.0f Hz | Z1 num_modes≤%d Z3∈[%d,%d] | "
            "adaptive ceiling @%.0f Hz (zone→440/470/490) | "
            "merge wood V2 (100–450 Hz) + isolation relief (independent of SLEPc quota) | "
            "max concurrent=%d workers.%s sorting_root=%s | HF quota ≥%.0f Hz.%s",
            ZONE1_STEP_HZ,
            ZONE1_STEP_HZ,
            ZONE2_STEP_HZ,
            ZONE3_STEP_HZ,
            ZONE1_NUM_MODES_CAP,
            ZONE3_NUM_MODES_MIN,
            ZONE3_NUM_MODES_MAX,
            CONDUCTOR_TRIGGER_HZ,
            max_workers,
            linux_pin_msg,
            sorting_root,
            ZONE_C_MIN_HZ,
            " [FORCE-EMERGENCY: pruning disabled + no adaptive ceiling]"
            if bool(args.force_emergency)
            else "",
        )
    else:
        assert tasks_static is not None
        LOGGER.info(
            "Planned %d worker task(s) from %.1f–%.1f Hz (max concurrent=%d workers.%s "
            "sorting_root=%s | legacy static merge (HF quota ≥%.0f Hz).",
            len(tasks_static),
            float(args.hz_min),
            float(args.hz_max),
            max_workers,
            linux_pin_msg,
            sorting_root,
            ZONE_C_MIN_HZ,
        )

    running: Dict[subprocess.Popen, Dict[str, Any]] = {}
    config_path = args.config.resolve()
    static_idx = 0
    last_spawn_mono: List[Optional[float]] = [None]

    _core_lock = threading.Lock()
    _core_free: Deque[int] = deque(worker_cores) if use_taskset else deque()

    def lease_core() -> Optional[int]:
        if not use_taskset:
            return None
        with _core_lock:
            if not _core_free:
                raise RuntimeError(f"No worker CPU cores available in pool {worker_cores}.")
            cid = _core_free.popleft()
            remaining = sorted(_core_free)
        LOGGER.info("Core lease: assigned cpu=%d (pool still free: %s)", cid, remaining)
        return cid

    def release_core(cid: Optional[int]) -> None:
        if cid is None or not use_taskset:
            return
        with _core_lock:
            _core_free.append(int(cid))

    def spawn_worker(
        hz: float,
        params: Dict[str, Any],
        role: str,
        *,
        structural_only: bool = False,
        structural_fallback: bool = False,
    ) -> None:
        timeout_s = float(params["timeout_minutes"]) * 60.0
        if args.use_mpiexec:
            cmd = [
                "mpiexec",
                "--bind-to",
                "none",
                "-n",
                "1",
                sys.executable,
                str(worker_script),
                "--target_hz",
                str(hz),
                "--num_modes",
                str(int(params["num_modes"])),
                "--config",
                str(config_path),
                "--sorting-root",
                str(sorting_root),
            ]
        else:
            cmd = [
                sys.executable,
                str(worker_script),
                "--target_hz",
                str(hz),
                "--num_modes",
                str(int(params["num_modes"])),
                "--config",
                str(config_path),
                "--sorting-root",
                str(sorting_root),
            ]
        if structural_only:
            cmd.append("--structural-only")

        core_id: Optional[int] = None
        if use_taskset:
            core_id = lease_core()
            cmd = ["taskset", "-c", str(core_id)] + cmd

        LOGGER.info(
            "Spawn worker: hz=%.4f (%s) num_modes=%s timeout=%.1f min (taskset_cpu=%s) role=%s",
            hz,
            params.get("label", ""),
            params["num_modes"],
            float(params["timeout_minutes"]),
            core_id if core_id is not None else "n/a",
            role or "-",
        )
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env["OMP_PROC_BIND"] = "false"
        env.pop("OMP_PLACES", None)
        env["OMPI_MCA_hwloc_base_binding_policy"] = "none"
        env.setdefault("I_MPI_PIN", "0")

        LOGGER.info("Worker exec (argv): %s", shlex.join(cmd))
        LOGGER.info(
            "Worker affinity env: OMP_PROC_BIND=%r OMPI_MCA_hwloc_base_binding_policy=%r "
            "PETSC_OPTIONS=%r",
            env.get("OMP_PROC_BIND"),
            env.get("OMPI_MCA_hwloc_base_binding_policy"),
            env.get("PETSC_OPTIONS"),
        )

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=None,
                stderr=None,
                env=env,
            )
        except Exception:
            if core_id is not None:
                release_core(core_id)
            raise

        spectral_step_hz = (
            float(scheduler._current_step_hz())
            if scheduler is not None
            else float(params.get("step_hz", ZONE2_STEP_HZ))
        )
        running[proc] = {
            "hz": hz,
            "deadline": time.monotonic() + timeout_s,
            "timeout_minutes": float(params["timeout_minutes"]),
            "core_id": core_id,
            "role": role,
            "spectral_step_hz": spectral_step_hz,
            "params": dict(params),
            "structural_only": bool(structural_only),
            "structural_fallback": bool(structural_fallback),
        }
        last_spawn_mono[0] = time.monotonic()

    def _has_pending_tasks() -> bool:
        if scheduler is not None:
            return scheduler.has_pending()
        assert tasks_static is not None
        return static_idx < len(tasks_static)

    try:
        while _has_pending_tasks() or running:
            _poll_completed(
                running,
                log_path,
                sorting_root,
                merge_lock,
                release_core,
                scheduler=scheduler,
                force_emergency=bool(args.force_emergency),
                spawn_worker=spawn_worker,
            )
            _enforce_timeouts(running, sorting_root, release_core)
            while len(running) < max_workers and _has_pending_tasks():
                if len(running) == 1:
                    LOGGER.info(
                        "Waiting %.0fs before second concurrent worker (mesh load / I/O stagger)...",
                        STAGGER_ADDITIONAL_WORKER_SECONDS,
                    )
                    time.sleep(STAGGER_ADDITIONAL_WORKER_SECONDS)
                elif len(running) == 2:
                    LOGGER.info(
                        "Waiting %.0fs before third concurrent worker (mesh load / I/O stagger)...",
                        STAGGER_ADDITIONAL_WORKER_SECONDS,
                    )
                    time.sleep(STAGGER_ADDITIONAL_WORKER_SECONDS)

                if last_spawn_mono[0] is not None:
                    gap = time.monotonic() - float(last_spawn_mono[0])
                    if gap < MIN_SPAWN_GAP_SECONDS:
                        wait_s = MIN_SPAWN_GAP_SECONDS - gap
                        LOGGER.info(
                            "Enforcing %.1fs throttle gap (%.2fs since last spawn); sleeping %.2fs...",
                            MIN_SPAWN_GAP_SECONDS,
                            gap,
                            wait_s,
                        )
                        time.sleep(wait_s)

                if scheduler is not None:
                    nxt = scheduler.pop_next()
                    if nxt is None:
                        break
                    hz_n, par_n, role_n = nxt
                    spawn_worker(hz_n, par_n, role_n)
                else:
                    assert tasks_static is not None
                    if static_idx >= len(tasks_static):
                        break
                    hz_s, par_s = tasks_static[static_idx]
                    static_idx += 1
                    spawn_worker(hz_s, par_s, "")
            time.sleep(0.25)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted — terminating running workers.")
        for proc, meta in list(running.items()):
            try:
                proc.kill()
            except OSError:
                pass
            finally:
                release_core(meta.get("core_id"))  # type: ignore[arg-type]
        running.clear()
        return 130

    LOGGER.info("Master sweep complete. Log: %s", log_path)
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
