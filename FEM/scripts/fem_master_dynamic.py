#!/usr/bin/env python3
"""
Master driver for ``fem_worker_single.py``: bounded concurrency, dynamic band
parameters, per-job timeouts, and safe merge of worker JSON into ``candidates_log.json``.

**Conditional Adaptive Manager (merge stage)**:
  * **Zone wood veto (priority 1)**: wood participation must meet a frequency-dependent
    floor (0.0008 at 100 Hz → 0.0003 at 350 Hz+) before any score or reward.
  * **Isolation reward** (secondary): Hz-distance bonus only if the candidate passed the
    wood veto *and* worker uniqueness ≥ ``MIN_UNIQUENESS_FOR_ISOLATION``; wood-weighted
    so unique wood-modes beat isolated air-modes.
  * **Spectral shaping**: exponential penalty when nearest accepted Hz gap < 3 Hz;
    linear reward for gap > 15 Hz, capped at 40 Hz equivalent span.
  * **Zone C quota**: at least ``ZONE_C_MIN_QUOTA`` modes with f ≥ 350 Hz when available,
    trading out lowest-scoring sub-350 picks so HF cannot be saturated away by LF duplicates.
  * **Sparse core**: overlap vs log + intra-merge picks uses ``scipy.sparse`` CSR float32
    (``csr_normalized_overlap``); explicit ``gc.collect()`` after vector loads / selection.

Resource policy (VM-friendly): at most **3** concurrent workers. On Linux, each worker is
launched under ``taskset -c <id>`` with ``<id>`` leased from ``{1, 2, 3}``, then
``mpiexec --bind-to none -n 1`` so Open MPI does not re-bind ranks onto core 0.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
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

MAX_CONCURRENT_WORKERS = 3
STAGGER_ADDITIONAL_WORKER_SECONDS = 10.0
MIN_SPAWN_GAP_SECONDS = 5.0

# --- Frequency clustering (same scale as legacy gap thinning) ---
MIN_HZ_GAP = 0.4

# --- Zone wood floors (linear 100 Hz → 350 Hz): 0.0008 → 0.0003 ---
WOOD_FLOOR_HZ_ANCHOR_LO = 100.0
WOOD_FLOOR_HZ_ANCHOR_HI = 350.0
WOOD_PARTICIPATION_FLOOR_LO = 0.0008
WOOD_PARTICIPATION_FLOOR_HI = 0.0003

# --- Isolation reward (only after wood veto + uniqueness floor) ---
MIN_UNIQUENESS_FOR_ISOLATION = 0.06
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


def hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def result_json_path(sorting_root: Path, hz: float) -> Path:
    return sorting_root / "temp_results" / f"result_{hz_result_tag(hz)}.json"


def get_band_params(current_hz: float) -> Dict[str, Any]:
    hz = float(current_hz)
    if 100.0 <= hz < 150.0:
        return {"step_hz": 5, "num_modes": 80, "timeout_minutes": 60, "label": "Dense Band 1"}
    if 150.0 <= hz < 250.0:
        return {"step_hz": 10, "num_modes": 30, "timeout_minutes": 60, "label": "Medium Band"}
    if 250.0 <= hz < 300.0:
        return {"step_hz": 5, "num_modes": 50, "timeout_minutes": 60, "label": "Dense Band 2"}
    if hz >= 300.0:
        return {"step_hz": 25, "num_modes": 15, "timeout_minutes": 20, "label": "Dead Zone"}
    raise ValueError(f"get_band_params: hz={hz} is outside the supported sweep (expected hz >= 100).")


def _wood_floor_for_hz(hz: float) -> float:
    """Linear wood-participation floor from 0.0008 (100 Hz) down to 0.0003 (350 Hz+)."""
    x = float(hz)
    if x <= WOOD_FLOOR_HZ_ANCHOR_LO:
        return WOOD_PARTICIPATION_FLOOR_LO
    if x >= WOOD_FLOOR_HZ_ANCHOR_HI:
        return WOOD_PARTICIPATION_FLOOR_HI
    t = (x - WOOD_FLOOR_HZ_ANCHOR_LO) / (WOOD_FLOOR_HZ_ANCHOR_HI - WOOD_FLOOR_HZ_ANCHOR_LO)
    return WOOD_PARTICIPATION_FLOOR_LO + t * (WOOD_PARTICIPATION_FLOOR_HI - WOOD_PARTICIPATION_FLOOR_LO)


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


def _spectral_multiplier(df_nearest: float) -> float:
    """
    Penalize small Hz gaps (< 3 Hz) exponentially; reward large gaps (> 15 Hz) linearly
    with slope capped as if Δf were at most 40 Hz.
    """
    d = float(df_nearest)
    mult = 1.0
    if d < SPECTRAL_CLOSE_HZ:
        mult *= math.exp(-max(0.0, (SPECTRAL_CLOSE_HZ - d) / max(SPECTRAL_PENALTY_TAU_HZ, 1e-6)))
    if d > SPECTRAL_FAR_HZ:
        span = min(d - SPECTRAL_FAR_HZ, SPECTRAL_FAR_CAP_HZ - SPECTRAL_FAR_HZ)
        if span > 0.0:
            mult *= 1.0 + SPECTRAL_REWARD_SLOPE * span
    return mult


def _isolation_bonus(hz: float, w: float, uniq: float, df_nearest: float) -> float:
    """Secondary weight: only if wood veto already passed and uniqueness meets floor."""
    if uniq < MIN_UNIQUENESS_FOR_ISOLATION:
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


def _resolve_vector_abs(row: Dict[str, Any], sorting_root: Path) -> Optional[Path]:
    rel = Path(str(row.get("vector_path", "") or ""))
    if not rel.parts:
        return None
    p = (sorting_root / rel).resolve()
    if not p.is_file() or sorting_root not in p.parents:
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
) -> float:
    hz_ref = list(existing_hz) + list(picked_hz)
    df_n = _df_nearest(hz, hz_ref)
    spec = _spectral_multiplier(df_n)
    max_ov = _max_sparse_overlap(csr_c, existing_csr + picked_csr)
    shape = float(np.clip(1.0 - max_ov, 0.0, 1.0))
    iso = _isolation_bonus(hz, w, uniq, df_n)
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

    clusters = _cluster_by_min_gap(batch, MIN_HZ_GAP)
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
            s = _master_mode_score(hz, w, uniq, csr_i, existing_hz, existing_csr, picked_hz, picked_csr)
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


def _unlink_worker_vector(row: Dict[str, Any], sorting_root: Path) -> None:
    rel = Path(str(row.get("vector_path", "") or ""))
    if not rel.parts:
        return
    p = (sorting_root / rel).resolve()
    try:
        if p.is_file() and sorting_root in p.parents:
            p.unlink()
    except OSError as exc:
        LOGGER.warning("Could not remove discarded mode vector %s: %s", p, exc)


def build_task_list(hz_min: float, hz_max: float) -> List[Tuple[float, Dict[str, Any]]]:
    """Build worker target Hz list from ``hz_min`` (inclusive) through ``hz_max`` (inclusive)."""
    lo = float(hz_min)
    hi = float(hz_max)
    if lo < 100.0:
        raise ValueError(f"hz_min must be >= 100.0 (band tables start at 100 Hz), got {lo}")
    if hi < lo:
        raise ValueError(f"hz_max ({hi}) must be >= hz_min ({lo})")
    tasks: List[Tuple[float, Dict[str, Any]]] = []
    hz = lo
    while hz <= hi + 1e-9:
        p = dict(get_band_params(hz))
        tasks.append((hz, p))
        hz += float(p["step_hz"])
    return tasks


def _merge_result_into_candidates_log(
    result_path: Path,
    log_path: Path,
    lock: threading.Lock,
    sorting_root: Path,
) -> None:
    if not result_path.is_file():
        return
    try:
        incoming = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Skipping unreadable result file %s: %s", result_path, exc)
        return

    raw = list(incoming.get("candidates") or [])
    if not raw:
        try:
            result_path.unlink()
        except OSError:
            pass
        return

    raw_n = len(raw)
    veto_pass: List[Dict[str, Any]] = []
    failed_veto: List[Dict[str, Any]] = []
    for c in raw:
        if _passes_zone_wood_veto(c):
            veto_pass.append(c)
        else:
            failed_veto.append(c)

    for c in failed_veto:
        _unlink_worker_vector(c, sorting_root)

    if not veto_pass:
        LOGGER.warning(
            "No candidates passed zone wood veto for %s; removing result file and worker vectors.",
            result_path.name,
        )
        for c in raw:
            _unlink_worker_vector(c, sorting_root)
        try:
            result_path.unlink()
        except OSError:
            pass
        return

    batch_loaded = _load_batch_csr_pairs(veto_pass, sorting_root)
    if not batch_loaded:
        LOGGER.warning("No loadable sparse vectors after wood veto for %s.", result_path.name)
        for c in raw:
            _unlink_worker_vector(c, sorting_root)
        try:
            result_path.unlink()
        except OSError:
            pass
        return

    with lock:
        if log_path.exists():
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"candidates": []}
        else:
            payload = {"candidates": []}
        existing = list(payload.get("candidates") or [])

        thin_kept, thin_discarded, _scores = _adaptive_manager_select(batch_loaded, existing, sorting_root)

        for c in thin_discarded:
            _unlink_worker_vector(c, sorting_root)

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
            for c in raw:
                _unlink_worker_vector(c, sorting_root)
            try:
                result_path.unlink()
            except OSError:
                pass
            gc.collect()
            return

        mx = max((int(c.get("id", -1)) for c in existing), default=-1)
        for i, rec in enumerate(thin_kept):
            new_id = int(mx + 1 + i)
            row = dict(rec)
            rel_old = Path(str(row.get("vector_path", "")))
            old_abs = (sorting_root / rel_old).resolve()
            new_rel = Path("temp_modes") / f"mode_{new_id:06d}{MODE_VECTOR_FILE_SUFFIX}"
            new_abs = (sorting_root / new_rel).resolve()
            if old_abs.is_file():
                if new_abs != old_abs:
                    if new_abs.exists():
                        new_abs.unlink()
                    old_abs.rename(new_abs)
            row["id"] = new_id
            row["vector_path"] = str(new_rel).replace("\\", "/")
            existing.append(row)
        payload["candidates"] = existing
        tmp = log_path.with_suffix(log_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(log_path)

    del batch_loaded
    del thin_discarded
    gc.collect()

    try:
        result_path.unlink()
    except OSError:
        LOGGER.warning("Could not delete merged result file: %s", result_path)

    gc.collect()


def _poll_completed(
    running: Dict[subprocess.Popen, Dict[str, Any]],
    log_path: Path,
    sorting_root: Path,
    merge_lock: threading.Lock,
    release_core: Callable[[Optional[int]], None],
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
                _merge_result_into_candidates_log(rpath, log_path, merge_lock, sorting_root)
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
        default=100.0,
        help="Sweep lower bound (Hz), inclusive. Must be >= 100 (default: 100).",
    )
    parser.add_argument("--hz-max", type=float, default=450.0, help="Sweep upper bound (Hz), inclusive.")
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
            "Linux: `taskset -c <1|2|3> mpiexec --bind-to none -n 1 <python> ...` "
            "(cores leased from {1,2,3}; `--bind-to none` avoids Open MPI overriding taskset)."
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
        log_path.write_text(json.dumps({"candidates": []}, indent=2), encoding="utf-8")

    use_taskset = sys.platform.startswith("linux")
    try:
        tasks = build_task_list(float(args.hz_min), float(args.hz_max))
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1
    linux_pin_msg = " Linux: taskset cores leased from {1,2,3}." if use_taskset else ""
    LOGGER.info(
        "Planned %d worker task(s) from %.1f–%.1f Hz (max concurrent=%d workers.%s "
        "sorting_root=%s | "
        "Merge: conditional adaptive manager (zone wood 0.0008→0.0003, sparse overlap, "
        "spectral shaping, HF quota ≥%.0f Hz).",
        len(tasks),
        float(args.hz_min),
        float(args.hz_max),
        MAX_CONCURRENT_WORKERS,
        linux_pin_msg,
        sorting_root,
        ZONE_C_MIN_HZ,
    )

    running: Dict[subprocess.Popen, Dict[str, Any]] = {}
    config_path = args.config.resolve()
    next_i = 0
    last_spawn_mono: List[Optional[float]] = [None]

    _core_lock = threading.Lock()
    _core_free: Deque[int] = deque([1, 2, 3]) if use_taskset else deque()

    def lease_core() -> Optional[int]:
        if not use_taskset:
            return None
        with _core_lock:
            if not _core_free:
                raise RuntimeError("No worker CPU cores available in pool [1, 2, 3].")
            cid = _core_free.popleft()
            remaining = sorted(_core_free)
        LOGGER.info("Core lease: assigned cpu=%d (pool still free: %s)", cid, remaining)
        return cid

    def release_core(cid: Optional[int]) -> None:
        if cid is None or not use_taskset:
            return
        with _core_lock:
            _core_free.append(int(cid))

    def spawn_index(idx: int) -> None:
        nonlocal next_i
        hz, params = tasks[idx]
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
            ]

        core_id: Optional[int] = None
        if use_taskset:
            core_id = lease_core()
            cmd = ["taskset", "-c", str(core_id)] + cmd

        LOGGER.info(
            "Spawn worker: hz=%.4f (%s) num_modes=%s timeout=%.1f min (taskset_cpu=%s)",
            hz,
            params.get("label", ""),
            params["num_modes"],
            float(params["timeout_minutes"]),
            core_id if core_id is not None else "n/a",
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

        running[proc] = {
            "hz": hz,
            "deadline": time.monotonic() + timeout_s,
            "timeout_minutes": float(params["timeout_minutes"]),
            "core_id": core_id,
        }
        last_spawn_mono[0] = time.monotonic()
        next_i = idx + 1

    try:
        while next_i < len(tasks) or running:
            _poll_completed(running, log_path, sorting_root, merge_lock, release_core)
            _enforce_timeouts(running, sorting_root, release_core)
            while len(running) < MAX_CONCURRENT_WORKERS and next_i < len(tasks):
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

                spawn_index(next_i)
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
