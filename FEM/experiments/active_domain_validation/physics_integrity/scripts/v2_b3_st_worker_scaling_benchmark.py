#!/usr/bin/env python3
"""Dev-only ST multi-target worker scaling benchmark (process-level shards; no FEM rebuild in workers)."""
from __future__ import annotations

import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit
import v2_b3_dev_solver_benchmark as dev_bench
import v2_b3_lmid_overnight_validation as lmid_overnight

B3_ST_WORKER_SCALING_ARG = "--B3-Lmid-ST-multi-target-worker-scaling-benchmark-only"
B3_ST_WORKER_COUNT_ARG = "--B3-ST-worker-count"
B3_ST_TARGETS_HZ_ARG = "--B3-ST-targets-hz"
B3_ST_SCALING_MESH_ARG = "--B3-ST-scaling-mesh-level"
B3_ST_WORKER_SHARD_EXECUTE_ARG = "--B3-ST-worker-shard-execute-only"
B3_ST_WORKER_SHARD_JOB_ARG = "--B3-ST-worker-shard-job-json"

ALLOWED_MESH_LEVELS = frozenset({"L_mid", "L_dev_dense"})
DEFAULT_MESH_LEVEL = "L_mid"
LMID_FULL_TARGETS = list(lmid_overnight.LMID_ST_TARGETS_HZ)
DENSE_DEFAULT_TARGETS = list(lmid_overnight.DENSE_ST_TARGETS_HZ)

_AUDIT_SCRIPT = Path(__file__).resolve().parent / "run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py"


def is_st_worker_scaling_mode(argv: Sequence[str]) -> bool:
    return B3_ST_WORKER_SCALING_ARG in argv


def is_st_worker_shard_execute_mode(argv: Sequence[str]) -> bool:
    return B3_ST_WORKER_SHARD_EXECUTE_ARG in argv


def _parse_arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _parse_hz_list(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty target frequency list")
    return out


def _parse_targets_hz(argv: Sequence[str], *, mesh_level: str) -> List[float]:
    raw = _parse_arg_value(argv, B3_ST_TARGETS_HZ_ARG)
    if raw:
        return _parse_hz_list(raw)
    if mesh_level == "L_dev_dense":
        return list(DENSE_DEFAULT_TARGETS)
    return list(LMID_FULL_TARGETS)


def _parse_worker_count(argv: Sequence[str]) -> int:
    raw = _parse_arg_value(argv, B3_ST_WORKER_COUNT_ARG)
    if raw is None:
        raise ValueError(f"missing required {B3_ST_WORKER_COUNT_ARG}")
    wc = int(raw)
    if wc not in (1, 2, 3):
        raise ValueError(f"{B3_ST_WORKER_COUNT_ARG} must be 1, 2, or 3")
    return wc


def _parse_mesh_level(argv: Sequence[str]) -> str:
    raw = _parse_arg_value(argv, B3_ST_SCALING_MESH_ARG)
    level = str(raw or DEFAULT_MESH_LEVEL).strip()
    if level not in ALLOWED_MESH_LEVELS:
        raise ValueError(f"{B3_ST_SCALING_MESH_ARG} must be one of {sorted(ALLOWED_MESH_LEVELS)}")
    return level


def _struct_active_count_policy(mesh_level: str) -> str:
    return "L_mid_exact" if mesh_level == "L_mid" else "mesh_independent"


def _targets_stem_suffix(targets_hz: List[float]) -> str:
    parts = [f"{float(f):g}".replace(".", "p") for f in targets_hz]
    return "_hz_" + "_".join(parts)


def _out_json_benchmark(mesh_level: str, worker_count: int, targets_hz: List[float]) -> Path:
    stem = (
        f"v2_B3_{mesh_level}_ST_multi_target_worker_scaling_W{worker_count}"
        f"{_targets_stem_suffix(targets_hz)}"
    )
    return audit.CONV_DIAG / f"{stem}.json"


def _sequential_reference_json(mesh_level: str, targets_hz: List[float]) -> Path:
    if mesh_level == "L_mid":
        return lmid_overnight.OUT_JSON_LMID_ST
    return dev_bench._dev_out_json_st_multi_target(mesh_level, targets_hz)


def _checkpoint_dir(mesh_level: str, run_id: str) -> Path:
    d = audit.CONV_DIAG / f"st_worker_scaling_{mesh_level}_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _peak_rss_mb() -> Optional[float]:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return dev_bench._safe_float(float(ru.ru_maxrss) / 1024.0)
    except Exception:
        return None


def _built_metadata_from_built(built: Dict[str, Any], *, mesh_level: str) -> Dict[str, Any]:
    def _arr(key: str) -> List[int]:
        return np.asarray(built[key], dtype=np.int32).ravel().tolist()

    return {
        "mesh_level": mesh_level,
        "struct_active_count_policy": _struct_active_count_policy(mesh_level),
        "active_dimension": int(np.asarray(built["active_local"]).size),
        "n_w": int(built["n_w"]),
        "n_u_b3": int(built["n_u_b3"]),
        "active_local": _arr("active_local"),
        "inactive_local": _arr("inactive_local"),
        "free_rows": _arr("free_rows"),
        "bc_rows": _arr("bc_rows"),
        "u_idx": _arr("u_idx"),
        "p_idx": _arr("p_idx"),
        "A_shape": audit._mat_shape(built["A_active"]),
        "M_shape": audit._mat_shape(built["M_active"]),
    }


def _built_from_metadata(meta: Dict[str, Any], *, A_active: Any, M_active: Any) -> Dict[str, Any]:
    return {
        "A_active": A_active,
        "M_active": M_active,
        "active_local": np.asarray(meta["active_local"], dtype=np.int32),
        "inactive_local": np.asarray(meta["inactive_local"], dtype=np.int32),
        "free_rows": np.asarray(meta["free_rows"], dtype=np.int32),
        "bc_rows": np.asarray(meta["bc_rows"], dtype=np.int32),
        "u_idx": np.asarray(meta["u_idx"], dtype=np.int32),
        "p_idx": np.asarray(meta["p_idx"], dtype=np.int32),
        "n_w": int(meta["n_w"]),
        "n_u_b3": int(meta["n_u_b3"]),
    }


def _export_operators(checkpoint: Path, *, built: Dict[str, Any], mesh_level: str) -> Dict[str, Any]:
    a_path = checkpoint / "A_active.petsc.bin"
    m_path = checkpoint / "M_active.petsc.bin"
    meta_path = checkpoint / "built_metadata.json"
    A_active = built["A_active"]
    M_active = built["M_active"]
    for mat, path in ((A_active, a_path), (M_active, m_path)):
        viewer = PETSc.Viewer().createBinary(str(path), "w", comm=PETSc.COMM_WORLD)
        try:
            mat.view(viewer)
        finally:
            viewer.destroy()
    meta = _built_metadata_from_built(built, mesh_level=mesh_level)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "checkpoint_dir": str(checkpoint.resolve()),
        "A_active_binary": str(a_path.resolve()),
        "M_active_binary": str(m_path.resolve()),
        "built_metadata_json": str(meta_path.resolve()),
        **meta,
    }


def _load_operators(checkpoint: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta_path = checkpoint / "built_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing built metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    a_path = checkpoint / "A_active.petsc.bin"
    m_path = checkpoint / "M_active.petsc.bin"
    if not a_path.is_file() or not m_path.is_file():
        raise FileNotFoundError(f"missing operator binary under {checkpoint}")

    def _load_mat(path: Path) -> Any:
        viewer = PETSc.Viewer().createBinary(str(path), "r", comm=PETSc.COMM_WORLD)
        try:
            mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
            mat.load(viewer)
            audit._petsc_mat_try_assemble(mat)
            return mat
        finally:
            viewer.destroy()

    A_active = _load_mat(a_path)
    M_active = _load_mat(m_path)
    built = _built_from_metadata(meta, A_active=A_active, M_active=M_active)
    return built, meta


def _partition_target_shards(
    targets_hz: List[float],
    worker_count: int,
) -> List[List[Tuple[int, float]]]:
    shards: List[List[Tuple[int, float]]] = [[] for _ in range(worker_count)]
    for ti, hz in enumerate(targets_hz):
        shards[ti % worker_count].append((ti, float(hz)))
    return shards


def _run_st_targets_on_built(
    built: Dict[str, Any],
    target_entries: List[Tuple[int, float]],
    *,
    payload_prefix: str = "B3_ST_scaling",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, float]:
    """Run ST multi-target slice; return accepted modes, per-target fields, setup/solve totals."""
    from slepc4py import SLEPc

    freq_lo = float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    freq_hi = float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    A_active = built["A_active"]
    M_active = built["M_active"]
    per_target: Dict[str, Any] = {}
    all_accepted: List[Dict[str, Any]] = []
    total_setup_s = 0.0
    total_solve_s = 0.0

    for ti, target_hz in target_entries:
        key = f"{payload_prefix}_target_{ti}_"
        per_target[f"{key}target_frequency_hz"] = float(target_hz)
        per_target[f"{key}target_lambda"] = dev_bench._safe_float(audit._b3_hz_to_lambda_sq(float(target_hz)))
        eps = None
        try:
            eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
            cfg: Dict[str, Any] = {}
            dev_bench._dev_configure_coarse_krylovschur_sinvert_eps(
                eps,
                A_active,
                M_active,
                payload=cfg,
                target_lambda=float(audit._b3_hz_to_lambda_sq(float(target_hz))),
                target_hz=float(target_hz),
            )
            t0 = time.perf_counter()
            eps.setUp()
            setup_s = time.perf_counter() - t0
            intro = dev_bench._dev_introspect_st_targeting_after_setup(eps)
            per_target[f"{key}effective_target"] = intro.get("B3_DEV_ST_target_effective")
            per_target[f"{key}effective_shift"] = intro.get("B3_DEV_ST_shift_effective")
            per_target[f"{key}effective_which"] = intro.get("B3_DEV_ST_which_effective_normalized")
            per_target[f"{key}setup_elapsed_seconds"] = dev_bench._safe_float(setup_s)
            t1 = time.perf_counter()
            eps.solve()
            solve_s = time.perf_counter() - t1
            per_target[f"{key}solve_elapsed_seconds"] = dev_bench._safe_float(solve_s)
            per_target[f"{key}solve_pass"] = True
            total_setup_s += setup_s
            total_solve_s += solve_s
            nconv, accepted = dev_bench._dev_collect_accepted_st_modes(
                eps,
                A_active,
                built,
                target_hz=float(target_hz),
                freq_lo=freq_lo,
                freq_hi=freq_hi,
            )
            per_target[f"{key}converged_mode_count"] = int(nconv)
            per_target[f"{key}accepted_mode_count_in_interval"] = int(len(accepted))
            per_target[f"{key}accepted_frequencies"] = [float(m["frequency_hz"]) for m in accepted]
            all_accepted.extend(accepted)
        except Exception as exc:
            per_target[f"{key}solve_pass"] = False
            per_target[f"{key}failure_reason"] = f"{type(exc).__name__}:{exc}"
        finally:
            if eps is not None:
                try:
                    eps.destroy()
                except Exception:
                    pass

    return all_accepted, per_target, total_setup_s, total_solve_s


def _aggregate_union_and_provenance(
    all_accepted: List[Dict[str, Any]],
) -> Tuple[List[float], List[Dict[str, Any]]]:
    union_freqs = dev_bench._dev_deduplicate_frequencies_hz(
        [float(m["frequency_hz"]) for m in all_accepted],
        tol_hz=dev_bench.B3_DEV_ST_MULTI_DEDUP_TOL_HZ,
    )
    provenance = lmid_overnight._st_deduplicated_provenance(
        all_accepted,
        union_freqs,
        tol_hz=dev_bench.B3_DEV_ST_MULTI_DEDUP_TOL_HZ,
    )
    return union_freqs, provenance


def _targets_lists_match(a: List[float], b: List[float], *, rtol: float = 1.0e-9) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= rtol * max(1.0, abs(float(y))) for x, y in zip(a, b))


def _compare_to_sequential_reference(
    union_freqs: List[float],
    *,
    ref_path: Path,
    targets_hz: List[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sequential_reference_json": str(ref_path),
        "sequential_reference_loaded": False,
        "benchmark_targets_hz": list(targets_hz),
    }
    if not ref_path.is_file():
        out["sequential_reference_missing"] = True
        out["frequency_parity_pass"] = None
        out["sequential_comparison_available"] = False
        return out
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_targets = list(
        ref.get("B3_Lmid_ST_multi_target_targets_hz")
        or ref.get("B3_DEV_ST_multi_target_targets_hz")
        or []
    )
    if ref_targets and not _targets_lists_match(list(targets_hz), ref_targets):
        out["sequential_reference_loaded"] = True
        out["sequential_comparison_available"] = False
        out["sequential_comparison_skipped_reason"] = "benchmark_targets_differ_from_sequential_reference_targets"
        out["sequential_reference_targets_hz"] = ref_targets
        out["frequency_parity_pass"] = None
        return out
    ref_freqs = list(
        ref.get("B3_Lmid_ST_multi_target_unique_accepted_frequencies")
        or ref.get("B3_DEV_ST_multi_target_unique_accepted_frequencies")
        or []
    )
    if not ref_freqs:
        ref_freqs = lmid_overnight._accepted_frequencies_from_mode_payload(ref)
    out["sequential_reference_loaded"] = True
    out["sequential_comparison_available"] = True
    out["sequential_unique_accepted_frequency_count"] = len(ref_freqs)
    out["benchmark_unique_accepted_frequency_count"] = len(union_freqs)
    out["same_unique_frequency_count_pass"] = bool(len(ref_freqs) == len(union_freqs))
    matches, missing, extra = dev_bench._dev_compare_frequency_sets(
        union_freqs,
        ref_freqs,
        match_tol_hz=dev_bench.B3_DEV_ST_MULTI_CISS_MATCH_TOL_HZ,
    )
    out["matched_frequency_count_vs_sequential"] = int(matches)
    out["missing_frequencies_vs_sequential"] = missing
    out["extra_frequencies_vs_sequential"] = extra
    out["frequency_parity_pass"] = bool(
        len(missing) == 0 and len(extra) == 0 and len(union_freqs) == len(ref_freqs)
    )
    return out


def _spawn_worker_shard(job_path: Path) -> Tuple[int, str]:
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(_AUDIT_SCRIPT),
        B3_ST_WORKER_SHARD_EXECUTE_ARG,
        B3_ST_WORKER_SHARD_JOB_ARG,
        str(job_path.resolve()),
    ]
    proc = subprocess.run(cmd, cwd=str(_AUDIT_SCRIPT.parent), capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    return int(proc.returncode), log


def run_st_worker_shard_execute(argv: Sequence[str]) -> int:
    if MPI.COMM_WORLD.size != 1:
        print("[B3_ST_scaling] worker shard requires mpiexec -n 1", flush=True)
        return 2
    job_raw = _parse_arg_value(argv, B3_ST_WORKER_SHARD_JOB_ARG)
    if not job_raw:
        print(f"[B3_ST_scaling] missing {B3_ST_WORKER_SHARD_JOB_ARG}", flush=True)
        return 2
    job_path = Path(job_raw)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(job["checkpoint_dir"]))
    worker_id = int(job["worker_id"])
    worker_count = int(job["worker_count"])
    target_entries = [(int(ti), float(hz)) for ti, hz in job["target_entries"]]
    out_shard = Path(str(job["output_shard_json"]))

    mats: List[Any] = []
    built: Optional[Dict[str, Any]] = None
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_ST_worker_shard_execute_only",
        "worker_id": worker_id,
        "worker_count": worker_count,
        "checkpoint_dir": str(checkpoint.resolve()),
        "target_indices": [ti for ti, _ in target_entries],
        "target_frequencies_hz": [hz for _, hz in target_entries],
        "production_promotion": "BLOCKED",
    }
    rc = 2
    try:
        built, meta = _load_operators(checkpoint)
        mats.extend([built["A_active"], built["M_active"]])
        payload["loaded_active_dimension"] = int(meta.get("active_dimension", 0))
        all_accepted, per_target, setup_s, solve_s = _run_st_targets_on_built(
            built,
            target_entries,
            payload_prefix="B3_ST_scaling",
        )
        payload.update(per_target)
        union_freqs, provenance = _aggregate_union_and_provenance(all_accepted)
        payload["B3_ST_scaling_shard_accepted_mode_records"] = all_accepted
        payload["B3_ST_scaling_shard_unique_accepted_frequencies"] = union_freqs
        payload["B3_ST_scaling_shard_unique_accepted_frequency_count"] = len(union_freqs)
        payload["B3_ST_scaling_shard_deduplicated_mode_provenance"] = provenance
        payload["B3_ST_scaling_shard_total_setup_elapsed_seconds"] = dev_bench._safe_float(setup_s)
        payload["B3_ST_scaling_shard_total_solve_elapsed_seconds"] = dev_bench._safe_float(solve_s)
        payload["B3_ST_scaling_shard_peak_rss_mb"] = _peak_rss_mb()
        payload["next_step_verdict"] = "B3_ST_WORKER_SHARD_PASS"
        rc = 0
    except Exception as exc:
        payload["failure_reason"] = f"{type(exc).__name__}:{exc}"
        payload["next_step_verdict"] = "B3_ST_WORKER_SHARD_FAILED"
        rc = 2
    finally:
        for m in mats:
            try:
                m.destroy()
            except Exception:
                pass
        audit._write_json_atomic(out_shard, payload)
    return rc


def run_st_worker_scaling_benchmark(argv: Sequence[str], pre: Dict[str, Any]) -> int:
    if not pre.get("preassembly_contract_pass") or MPI.COMM_WORLD.size != 1:
        return 2
    try:
        mesh_level = _parse_mesh_level(argv)
        worker_count = _parse_worker_count(argv)
        targets_hz = _parse_targets_hz(argv, mesh_level=mesh_level)
    except ValueError as exc:
        print(f"[B3_ST_scaling] {exc}", flush=True)
        return 2

    if mesh_level == "L_mid" and worker_count >= 3:
        print(
            "[B3_ST_scaling] worker_count=3 is blocked on L_mid (RAM risk). "
            "Use L_dev_dense for 3-worker smoke or worker_count<=2 on L_mid.",
            flush=True,
        )
        return 2

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    checkpoint = _checkpoint_dir(mesh_level, run_id)
    out_json = _out_json_benchmark(mesh_level, worker_count, targets_hz)
    ref_json = _sequential_reference_json(mesh_level, targets_hz)

    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_Lmid_ST_multi_target_worker_scaling_benchmark_only",
        "B3_ST_scaling_mesh_level": mesh_level,
        "B3_ST_scaling_worker_count": worker_count,
        "B3_ST_scaling_targets_hz": list(targets_hz),
        "B3_ST_scaling_checkpoint_dir": str(checkpoint.resolve()),
        "B3_ST_scaling_output_json": str(out_json.resolve()),
        "B3_ST_scaling_in_process_parity": bool(worker_count == 1),
        "B3_ST_scaling_solver_name": "KRYLOVSCHUR-ST-SINVERT-MUMPS-MULTI-TARGET-WORKER-SCALING",
        "production_promotion": "BLOCKED",
        "no_automatic_production_promotion": True,
    }
    timer = dev_bench._B3DevTiming(payload)
    mats: List[Any] = []
    seen: set[int] = set()
    verdict = "B3_ST_WORKER_SCALING_BLOCKED"
    rc = 2

    try:
        timer.mark("operator_build_begin")
        built = audit._b3_build_corrected_structural_active_operators(
            mats_to_destroy=mats,
            mat_destroy_seen=seen,
            mesh_level=mesh_level,
            struct_active_count_policy=_struct_active_count_policy(mesh_level),
        )
        timer.mark("operator_build_end")
        payload["B3_ST_scaling_operator_build_elapsed_seconds"] = payload.get(
            "B3_DEV_timing_operator_build_end_elapsed_seconds"
        )

        op_payload: Dict[str, Any] = {}
        if mesh_level == "L_mid":
            contract_pass = lmid_overnight._lmid_operator_contract_pass(op_payload, built=built)
        else:
            dev_bench._dev_record_operator_contract(op_payload, built=built)
            dim_fail = dev_bench._dev_record_active_dimension_target_range(op_payload, mesh_level)
            contract_pass = bool(
                op_payload.get("B3_DEV_operator_contract_pass")
                and op_payload.get("B3_DEV_zero_row_column_cleanup_contract_pass")
                and not dim_fail
            )
        payload.update(op_payload)
        payload["B3_ST_scaling_operator_contract_pass"] = bool(contract_pass)
        if not contract_pass:
            payload["failure_reason"] = "operator_contract_failed"
            verdict = "B3_ST_WORKER_SCALING_OPERATOR_CONTRACT_BLOCKED"
            audit._write_json_atomic(out_json, payload)
            return 2

        export_meta = _export_operators(checkpoint, built=built, mesh_level=mesh_level)
        payload["B3_ST_scaling_operator_export"] = export_meta

        shards = _partition_target_shards(targets_hz, worker_count)
        payload["B3_ST_scaling_target_shards"] = [
            {"worker_id": wi, "target_indices": [ti for ti, _ in ent], "target_frequencies_hz": [hz for _, hz in ent]}
            for wi, ent in enumerate(shards)
        ]

        wall_t0 = time.perf_counter()
        all_accepted: List[Dict[str, Any]] = []
        per_target_merged: Dict[str, Any] = {}
        total_setup_s = 0.0
        total_solve_s = 0.0
        worker_peak_rss: List[Optional[float]] = []
        worker_rcs: List[int] = []

        if worker_count == 1:
            entries = shards[0]
            payload["B3_ST_scaling_execution_path"] = "in_process_single_worker_parity"
            accepted, per_t, setup_s, solve_s = _run_st_targets_on_built(
                built, entries, payload_prefix="B3_ST_scaling"
            )
            all_accepted.extend(accepted)
            per_target_merged.update(per_t)
            total_setup_s += setup_s
            total_solve_s += solve_s
            worker_peak_rss.append(_peak_rss_mb())
            worker_rcs.append(0)
        else:
            payload["B3_ST_scaling_execution_path"] = "subprocess_sharded_workers"
            shard_paths: List[Path] = []
            for wi, entries in enumerate(shards):
                if not entries:
                    worker_rcs.append(0)
                    worker_peak_rss.append(None)
                    continue
                shard_json = checkpoint / f"worker_{wi}_shard.json"
                job_json = checkpoint / f"worker_{wi}_job.json"
                job = {
                    "worker_id": wi,
                    "worker_count": worker_count,
                    "checkpoint_dir": str(checkpoint.resolve()),
                    "target_entries": [[ti, hz] for ti, hz in entries],
                    "output_shard_json": str(shard_json.resolve()),
                }
                job_json.write_text(json.dumps(job, indent=2), encoding="utf-8")
                shard_paths.append(shard_json)
                w_rc, w_log = _spawn_worker_shard(job_json)
                worker_rcs.append(w_rc)
                payload[f"B3_ST_scaling_worker_{wi}_return_code"] = int(w_rc)
                if w_log.strip():
                    payload[f"B3_ST_scaling_worker_{wi}_subprocess_log_tail"] = w_log.strip()[-4096:]
                if w_rc != 0:
                    payload["failure_reason"] = f"worker_{wi}_failed_rc={w_rc}"
                    verdict = "B3_ST_WORKER_SCALING_WORKER_FAILED"
                    audit._write_json_atomic(out_json, payload)
                    return 2
                if not shard_json.is_file():
                    payload["failure_reason"] = f"worker_{wi}_missing_shard_json"
                    verdict = "B3_ST_WORKER_SCALING_WORKER_FAILED"
                    audit._write_json_atomic(out_json, payload)
                    return 2
                shard_data = json.loads(shard_json.read_text(encoding="utf-8"))
                worker_peak_rss.append(shard_data.get("B3_ST_scaling_shard_peak_rss_mb"))
                all_accepted.extend(shard_data.get("B3_ST_scaling_shard_accepted_mode_records") or [])
                for k, v in shard_data.items():
                    if k.startswith("B3_ST_scaling_target_"):
                        per_target_merged[k] = v
                total_setup_s += float(shard_data.get("B3_ST_scaling_shard_total_setup_elapsed_seconds") or 0.0)
                total_solve_s += float(shard_data.get("B3_ST_scaling_shard_total_solve_elapsed_seconds") or 0.0)

        wall_elapsed = time.perf_counter() - wall_t0
        payload["B3_ST_scaling_st_phase_wall_elapsed_seconds"] = dev_bench._safe_float(wall_elapsed)
        payload["B3_ST_scaling_total_setup_elapsed_seconds"] = dev_bench._safe_float(total_setup_s)
        payload["B3_ST_scaling_total_solve_elapsed_seconds"] = dev_bench._safe_float(total_solve_s)
        payload.update(per_target_merged)

        union_freqs, provenance = _aggregate_union_and_provenance(all_accepted)
        payload["B3_ST_scaling_unique_accepted_frequency_count"] = len(union_freqs)
        payload["B3_ST_scaling_unique_accepted_frequencies"] = union_freqs
        payload["B3_ST_scaling_deduplicated_mode_provenance"] = provenance
        payload["B3_ST_scaling_per_target_accepted_mode_records"] = list(all_accepted)
        payload["B3_ST_scaling_worker_return_codes"] = worker_rcs
        rss_vals = [float(x) for x in worker_peak_rss if x is not None and math.isfinite(float(x))]
        payload["B3_ST_scaling_parent_peak_rss_mb"] = _peak_rss_mb()
        payload["B3_ST_scaling_worker_peak_rss_mb"] = worker_peak_rss
        payload["B3_ST_scaling_max_worker_peak_rss_mb"] = (
            dev_bench._safe_float(max(rss_vals)) if rss_vals else None
        )

        payload.update(_compare_to_sequential_reference(union_freqs, ref_path=ref_json, targets_hz=targets_hz))

        timer.finalize()
        payload["B3_ST_scaling_total_wall_elapsed_seconds"] = payload.get("B3_DEV_timing_total_wall_elapsed_seconds")

        parity = payload.get("frequency_parity_pass")
        if parity is True:
            verdict = "B3_ST_WORKER_SCALING_PASS"
            rc = 0
        elif parity is False:
            verdict = "B3_ST_WORKER_SCALING_FREQUENCY_MISMATCH"
            rc = 2
        elif payload.get("sequential_comparison_skipped_reason"):
            verdict = "B3_ST_WORKER_SCALING_COMPLETED_COMPARISON_SKIPPED"
            rc = 0 if union_freqs else 2
        else:
            verdict = "B3_ST_WORKER_SCALING_COMPLETED_NO_REFERENCE"
            rc = 0 if union_freqs else 2

    except Exception as exc:
        payload["failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_ST_WORKER_SCALING_FAILED"
        rc = 2
    finally:
        audit._destroy_mats_deduped(mats)
        payload["next_step_verdict"] = verdict
        audit._write_json_atomic(out_json, payload)
        md_path = out_json.with_suffix(".md")
        md_path.write_text(
            f"# ST worker scaling benchmark\n\n"
            f"- mesh: `{payload.get('B3_ST_scaling_mesh_level')}`\n"
            f"- workers: `{payload.get('B3_ST_scaling_worker_count')}`\n"
            f"- verdict: `{verdict}`\n"
            f"- json: `{out_json}`\n",
            encoding="utf-8",
        )
        print(f"[B3_ST_scaling] written={out_json} verdict={verdict}", flush=True)
    return rc
