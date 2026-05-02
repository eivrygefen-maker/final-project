#!/usr/bin/env python3
"""
Master driver for ``fem_worker_single.py``: bounded concurrency, dynamic band
parameters, per-job timeouts, and safe merge of worker JSON into ``candidates_log.json``.

Resource policy (VM-friendly): at most **3** concurrent workers. On Linux, each worker is
launched under ``taskset -c <id>`` with ``<id>`` leased from ``{1, 2, 3}``, then
``mpiexec --bind-to none -n 1`` so Open MPI does not re-bind ranks onto core 0.
Before launching the **second** worker (when one is already running), and again before the
**third** (when two are running), the master waits **10 seconds** (mesh load / I/O stagger).
Every spawn is also separated by at least
**5 seconds** from the previous spawn (throttle). Core 0 stays for the master; other CPUs for
OS/UI. Merge uses a **0.4 Hz** frequency-gap thinning pass plus a low wood-participation
floor (diagnostic-friendly; see ``MIN_WOOD_PARTICIPATION``).
"""
from __future__ import annotations

import argparse
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
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


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
# Delay before launching the 2nd / 3rd concurrent worker (mesh load / memory / I/O).
STAGGER_ADDITIONAL_WORKER_SECONDS = 10.0
# Minimum wall time between any two successful spawns (monotonic clock).
MIN_SPAWN_GAP_SECONDS = 5.0
# Merge-time quality gates (worker batches → candidates_log / temp_modes).
# Structural wood-energy floor (fraction of mode energy in tagged wood DOFs). Keep low for
# diagnostic sweeps; raise to suppress air-dominated ghosts once participation is trustworthy.
MIN_WOOD_PARTICIPATION = 0.0005
MIN_UNIQUENESS = 0.0
MIN_HZ_GAP = 0.4
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


def _passes_candidate_gates(c: Dict[str, Any]) -> bool:
    """Wood + uniqueness floors before a mode is treated as a merge candidate."""
    try:
        w = float(c.get("wood_participation", 0.0) or 0.0)
        u = float(c.get("uniqueness", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(w) or not math.isfinite(u):
        return False
    if w < MIN_WOOD_PARTICIPATION:
        return False
    if u < MIN_UNIQUENESS:
        return False
    return True


def _unlink_worker_npy(row: Dict[str, Any], sorting_root: Path) -> None:
    rel = Path(str(row.get("vector_path", "") or ""))
    if not rel.parts:
        return
    p = (sorting_root / rel).resolve()
    try:
        if p.is_file() and sorting_root in p.parents:
            p.unlink()
    except OSError as exc:
        LOGGER.warning("Could not remove discarded mode vector %s: %s", p, exc)


def _thin_frequency_gap_by_uniqueness(
    rows: List[Dict[str, Any]],
    min_gap_hz: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Sort by Hz; cluster consecutive modes whose frequencies lie within ``min_gap_hz`` of the
    cluster anchor; keep the mode with highest ``uniqueness`` per cluster.
    Returns (kept, discarded_from_input_order).
    """
    if not rows:
        return [], []
    ordered = sorted(rows, key=lambda r: float(r.get("hz", 0.0) or 0.0))
    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    i = 0
    n = len(ordered)
    while i < n:
        j = i
        anchor_hz = float(ordered[i].get("hz", 0.0) or 0.0)
        while j + 1 < n and float(ordered[j + 1].get("hz", 0.0) or 0.0) - anchor_hz <= min_gap_hz:
            j += 1
        cluster = ordered[i : j + 1]
        best = max(
            cluster,
            key=lambda r: float(r.get("uniqueness", 0.0) or 0.0),
        )
        kept.append(best)
        for r in cluster:
            if r is not best:
                discarded.append(r)
        i = j + 1
    return kept, discarded


def build_task_list(hz_max: float) -> List[Tuple[float, Dict[str, Any]]]:
    tasks: List[Tuple[float, Dict[str, Any]]] = []
    hz = 100.0
    while hz <= hz_max + 1e-9:
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
    passed_gates: List[Dict[str, Any]] = []
    failed_gates: List[Dict[str, Any]] = []
    for c in raw:
        if _passes_candidate_gates(c):
            passed_gates.append(c)
        else:
            failed_gates.append(c)

    for c in failed_gates:
        _unlink_worker_npy(c, sorting_root)

    thin_kept, thin_discarded = _thin_frequency_gap_by_uniqueness(passed_gates, MIN_HZ_GAP)
    for c in thin_discarded:
        _unlink_worker_npy(c, sorting_root)

    dropped_total = raw_n - len(thin_kept)
    if dropped_total > 0:
        LOGGER.info(
            "Filtered %d redundant/low-quality modes from batch (wood_or_uniq_gate=%d, freq_thin=%d).",
            dropped_total,
            len(failed_gates),
            len(thin_discarded),
        )

    if not thin_kept:
        LOGGER.warning(
            "No candidates left after gates/thinning for %s; removing result file and worker vectors.",
            result_path.name,
        )
        for c in raw:
            _unlink_worker_npy(c, sorting_root)
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
        mx = max((int(c.get("id", -1)) for c in existing), default=-1)
        for i, rec in enumerate(thin_kept):
            new_id = int(mx + 1 + i)
            row = dict(rec)
            rel_old = Path(str(row.get("vector_path", "")))
            old_abs = (sorting_root / rel_old).resolve()
            new_rel = Path("temp_modes") / f"mode_{new_id:06d}.npy"
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
    try:
        result_path.unlink()
    except OSError:
        LOGGER.warning("Could not delete merged result file: %s", result_path)


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
    parser.add_argument("--hz-max", type=float, default=450.0, help="Sweep upper bound (Hz).")
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

    sorting_root = REPO_ROOT / "FEM" / "SORTING"
    log_path = sorting_root / "candidates_log.json"
    merge_lock = threading.Lock()

    sorting_root.mkdir(parents=True, exist_ok=True)
    (sorting_root / "temp_results").mkdir(parents=True, exist_ok=True)
    (sorting_root / "temp_modes").mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(json.dumps({"candidates": []}, indent=2), encoding="utf-8")

    use_taskset = sys.platform.startswith("linux")
    tasks = build_task_list(float(args.hz_max))
    linux_pin_msg = " Linux: taskset cores leased from {1,2,3}." if use_taskset else ""
    LOGGER.info(
        "Planned %d worker task(s) up to %.1f Hz (max concurrent=%d workers.%s "
        "Merge filter: %.1f Hz frequency-gap thinning, wood floor=%.4f.",
        len(tasks),
        float(args.hz_max),
        MAX_CONCURRENT_WORKERS,
        linux_pin_msg,
        MIN_HZ_GAP,
        MIN_WOOD_PARTICIPATION,
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
            # Open MPI may otherwise hwloc-bind ranks (often to core 0), fighting taskset.
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
        # One MPI rank + one BLAS/OpenMP thread per worker → less contention vs extra worker threads.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        # Let taskset + kernel own placement; avoid OpenMP / Open MPI re-pinning to core 0.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
