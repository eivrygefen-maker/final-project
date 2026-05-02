#!/usr/bin/env python3
"""
Master driver for ``fem_worker_single.py``: bounded concurrency, dynamic band
parameters, per-job timeouts, and safe merge of worker JSON into ``candidates_log.json``.

Resource policy (VM-friendly): at most **2** concurrent workers. On Linux, each worker is
launched under ``taskset -c <id>`` with ``<id>`` leased from ``{1, 2}``, then ``mpiexec -n 1``.
Starting the **second** concurrent worker waits **30 seconds** after the first so the OS
can place the first job before the second starts (core 0 for master; other CPUs for OS/UI).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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

MAX_CONCURRENT_WORKERS = 2
# Delay before launching the 2nd concurrent worker (same scheduling burst).
STAGGER_SECOND_WORKER_SECONDS = 30.0
LOGGER = logging.getLogger("fem_master_dynamic")


def hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def result_json_path(sorting_root: Path, hz: float) -> Path:
    return sorting_root / "temp_results" / f"result_{hz_result_tag(hz)}.json"


def get_band_params(current_hz: float) -> Dict[str, Any]:
    hz = float(current_hz)
    if 100.0 <= hz < 150.0:
        return {"step_hz": 5, "num_modes": 40, "timeout_minutes": 60, "label": "Dense Band 1"}
    if 150.0 <= hz < 250.0:
        return {"step_hz": 10, "num_modes": 30, "timeout_minutes": 60, "label": "Medium Band"}
    if 250.0 <= hz < 300.0:
        return {"step_hz": 5, "num_modes": 50, "timeout_minutes": 60, "label": "Dense Band 2"}
    if hz >= 300.0:
        return {"step_hz": 25, "num_modes": 15, "timeout_minutes": 20, "label": "Dead Zone"}
    raise ValueError(f"get_band_params: hz={hz} is outside the supported sweep (expected hz >= 100).")


def build_task_list(hz_max: float) -> List[Tuple[float, Dict[str, Any]]]:
    tasks: List[Tuple[float, Dict[str, Any]]] = []
    hz = 100.0
    while hz <= hz_max + 1e-9:
        p = dict(get_band_params(hz))
        tasks.append((hz, p))
        hz += float(p["step_hz"])
    return tasks


def _merge_result_into_candidates_log(result_path: Path, log_path: Path, lock: threading.Lock) -> None:
    if not result_path.is_file():
        return
    try:
        incoming = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Skipping unreadable result file %s: %s", result_path, exc)
        return

    cands = list(incoming.get("candidates") or [])
    if not cands:
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
        for i, rec in enumerate(cands):
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
                _merge_result_into_candidates_log(rpath, log_path, merge_lock)
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
            "Linux: run each worker as `taskset -c <1|2> mpiexec -n 1 <python> ...` "
            "(cores leased from {1,2}; Open MPI single rank)."
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

    tasks = build_task_list(float(args.hz_max))
    LOGGER.info(
        "Planned %d worker task(s) up to %.1f Hz (max concurrent=%d; leave spare CPUs for OS/master).",
        len(tasks),
        float(args.hz_max),
        MAX_CONCURRENT_WORKERS,
    )

    running: Dict[subprocess.Popen, Dict[str, Any]] = {}
    config_path = args.config.resolve()
    next_i = 0

    use_taskset = sys.platform.startswith("linux")
    _core_lock = threading.Lock()
    _core_free: Deque[int] = deque([1, 2]) if use_taskset else deque()

    def lease_core() -> Optional[int]:
        if not use_taskset:
            return None
        with _core_lock:
            if not _core_free:
                raise RuntimeError("No worker CPU cores available in pool [1, 2].")
            return _core_free.popleft()

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

        log_extra = f" taskset_cpu={core_id}" if core_id is not None else ""
        LOGGER.info(
            "Spawn worker: hz=%.4f (%s) num_modes=%s timeout=%.1f min%s",
            hz,
            params.get("label", ""),
            params["num_modes"],
            float(params["timeout_minutes"]),
            log_extra,
        )
        env = os.environ.copy()
        # One MPI rank + one BLAS/OpenMP thread per worker → less contention vs extra worker threads.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

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
        next_i = idx + 1

    try:
        while next_i < len(tasks) or running:
            _poll_completed(running, log_path, sorting_root, merge_lock, release_core)
            _enforce_timeouts(running, sorting_root, release_core)
            while len(running) < MAX_CONCURRENT_WORKERS and next_i < len(tasks):
                if len(running) == 1:
                    LOGGER.info(
                        "Staggered launch: sleeping %.0f s before starting the second concurrent worker.",
                        STAGGER_SECOND_WORKER_SECONDS,
                    )
                    time.sleep(STAGGER_SECOND_WORKER_SECONDS)
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
