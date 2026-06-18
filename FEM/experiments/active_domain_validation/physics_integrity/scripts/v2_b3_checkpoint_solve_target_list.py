#!/usr/bin/env python3
"""Solver-mkl: multi-target checkpoint solve from per-target window JSON (M4 worker chunks)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_target_candidate_audit_lib import (  # noqa: E402
    append_target_candidate_audit_row,
    build_target_candidate_audit_row,
)
from v2_b3_m4_box_raw_modal_discovery_lib import (  # noqa: E402
    box_raw_modal_discovery_enabled,
    resolve_worker_shape_name,
    write_worker_diagnostic_from_solver_targets,
)
from v2_b3_checkpoint_pipeline_lib import (  # noqa: E402
    PIPELINE_SOLVE_MANIFEST,
    fail_with_messages,
    verify_checkpoint_complete,
    verify_checkpoint_matrices,
    verify_mumps_available,
    verify_solver_mkl_stage_environment,
    write_json,
)
from v2_b3_m4_lprod_interfaces import (  # noqa: E402
    acceptance_config_from_chunk_targets,
    build_solver_result_placeholder,
    build_worker_result_placeholder,
    validate_chunk_targets_doc,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_mode_audio_coupling import AUDIO_COUPLING_FIELD_KEYS  # noqa: E402
from v2_b3_rich_modal_lib import load_region_dof_bundle  # noqa: E402

ALLOWED_FACTOR_SOLVERS = ("mkl_pardiso", "mumps")
STAGE_B_PLANNED = "v2_b3_checkpoint_solve_target_list.py"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solver-mkl: checkpoint ST solve for explicit target list with per-target windows.",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--targets-json", required=True, help="m4_worker_chunk_targets_v1 JSON path.")
    parser.add_argument("--factor-solver", choices=ALLOWED_FACTOR_SOLVERS, default="mkl_pardiso")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate targets-json and write placeholder outputs; no solve.",
    )
    parser.add_argument("--nev", type=int, default=12)
    parser.add_argument("--ncv", type=int, default=24)
    parser.add_argument(
        "--skip-mkl-probe",
        action="store_true",
        help="Skip MKL PARDISO availability probe (not recommended).",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _load_chunk_targets(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _run_real_solve(
    *,
    args: argparse.Namespace,
    chunk_targets: Dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
) -> int:
    """M4.4.1b entry: same loop as checkpoint_solver_multi_benchmark with per-target windows."""
    from petsc4py import PETSc  # noqa: F401

    from v2_b3_checkpoint_solver_multi_benchmark import (  # noqa: E402
        build_stable_summary,
        deduplicate_frequencies_hz,
    )
    from v2_b3_operator_checkpoint_portable import load_operators_with_portable_fallback  # noqa: E402
    from v2_b3_st_sinvert_solver_lib import (  # noqa: E402
        built_from_checkpoint_metadata,
        mat_global_nnz_used,
        run_checkpoint_st_target,
        safe_float,
        threading_env_snapshot,
        version_snapshot,
    )
    from v2_b3_petsc_util import mat_shape  # noqa: E402

    factor_solver = str(args.factor_solver).strip().lower()
    targets_hz = [float(t["target_hz"]) for t in (chunk_targets.get("targets") or [])]
    acceptance_cfg = acceptance_config_from_chunk_targets(chunk_targets)
    shape_name = resolve_worker_shape_name(chunk_targets)
    raw_diagnostic = box_raw_modal_discovery_enabled(shape_name=shape_name)
    box_bypass_target_window_acceptance = shape_name == "box"

    meta_path = checkpoint / "built_metadata.json"
    if not meta_path.is_file():
        body = {
            "status": "FAIL",
            "failure_reason": f"missing built metadata: {meta_path}",
            "checkpoint_dir": str(checkpoint),
        }
        write_json_atomic(output_dir / "solver_result.json", body)
        return 2

    built_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mesh_level = str(built_meta.get("mesh_level") or "unknown")
    region_ctx = load_region_dof_bundle(checkpoint, built_meta)
    if not region_ctx.get("structural_indices_available"):
        print(
            "[B3_checkpoint_solve_target_list] warning: region_dof_indices.npz missing; "
            "top/back participation unavailable (air may use p_idx from built_metadata). "
            "Re-run M4 L_prod checkpoint (best_effort region export) if this is a new production run.",
            flush=True,
        )

    result: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_kind": "checkpoint_solve_target_list",
        "checkpoint_dir": str(checkpoint),
        "output_dir": str(output_dir),
        "targets_json": str(Path(args.targets_json).resolve()),
        "chunk_id": chunk_targets.get("chunk_id"),
        "mesh_level": mesh_level,
        "factor_solver": factor_solver,
        "targets_hz": targets_hz,
        "nev": int(args.nev),
        "ncv": int(args.ncv),
        "region_dof_source": region_ctx.get("region_dof_source"),
        "structural_region_participation_status": region_ctx.get(
            "structural_region_participation_status"
        ),
        "box_raw_modal_discovery": raw_diagnostic,
        "shape_name": shape_name,
        "box_bypass_target_window_acceptance": box_bypass_target_window_acceptance,
        **acceptance_cfg.to_result_fields(),
        "versions": version_snapshot(),
        "threading_env": threading_env_snapshot(),
        "targets": [],
        "aggregate": {},
        "status": "FAIL",
        "failure_reason": None,
    }

    mats: List[Any] = []
    t_wall0 = time.perf_counter()
    try:
        A_active, M_active, load_diag = load_operators_with_portable_fallback(checkpoint)
        mats.extend([A_active, M_active])
        built, built_diag = built_from_checkpoint_metadata(
            built_meta,
            A_active=A_active,
            M_active=M_active,
        )
        result["checkpoint_load"] = load_diag
        result["built_metadata_diag"] = built_diag
        result["matrix_contract"] = {
            "A_shape": mat_shape(A_active),
            "M_shape": mat_shape(M_active),
            "A_nnz_used": mat_global_nnz_used(A_active),
            "M_nnz_used": mat_global_nnz_used(M_active),
        }

        per_target_rows: List[Dict[str, Any]] = []
        all_accepted: List[float] = []
        total_setup = 0.0
        total_solve = 0.0
        total_st = 0.0
        succeeded = 0

        chunk_target_rows = list(chunk_targets.get("targets") or [])
        for ti, target_hz in enumerate(targets_hz):
            print(
                f"[B3_checkpoint_solve_target_list] target {ti + 1}/{len(targets_hz)} "
                f"hz={target_hz} factor={factor_solver}",
                flush=True,
            )
            row = run_checkpoint_st_target(
                A_active=A_active,
                M_active=M_active,
                built=built,
                target_hz=float(target_hz),
                factor_solver=factor_solver,
                mesh_level=mesh_level,
                nev=int(args.nev),
                ncv=int(args.ncv),
                target_index=int(ti),
                export_vectors=False,
                acceptance_config=acceptance_cfg,
                region_ctx=region_ctx,
                raw_diagnostic=raw_diagnostic,
                box_bypass_target_window_acceptance=box_bypass_target_window_acceptance,
            )
            per_target_rows.append(row)
            target_meta = chunk_target_rows[ti] if ti < len(chunk_target_rows) else {}
            append_target_candidate_audit_row(
                output_dir,
                build_target_candidate_audit_row(
                    chunk_id=str(chunk_targets.get("chunk_id") or ""),
                    target_row=row,
                    target_meta=target_meta,
                ),
            )
            if row.get("status") == "PASS":
                succeeded += 1
                all_accepted.extend(list(row.get("accepted_frequencies_hz") or []))
                total_setup += float(row.get("setup_elapsed_seconds") or 0.0)
                total_solve += float(row.get("solve_elapsed_seconds") or 0.0)
                total_st += float(row.get("st_total_elapsed_seconds") or 0.0)
            else:
                result["failure_reason"] = row.get("failure_reason")
                break

        unique_accepted = deduplicate_frequencies_hz(all_accepted)
        wall_s = time.perf_counter() - t_wall0
        result["targets"] = per_target_rows
        result["aggregate"] = {
            "targets_attempted": len(targets_hz),
            "targets_succeeded": succeeded,
            "total_setup_seconds": safe_float(total_setup),
            "total_solve_seconds": safe_float(total_solve),
            "total_st_seconds": safe_float(total_st),
            "total_wall_seconds": safe_float(wall_s),
            "unique_accepted_frequencies_hz": unique_accepted,
            "unique_accepted_mode_count": len(unique_accepted),
        }
        if succeeded == len(targets_hz):
            result["status"] = "PASS"
        elif succeeded > 0:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "FAIL"
        result["summary"] = build_stable_summary(result)
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass

    mode_records: List[Dict[str, Any]] = []
    for trow in result.get("targets") or []:
        for m in trow.get("accepted_modes") or []:
            if isinstance(m, dict) and m.get("frequency_hz") is not None:
                row = {
                    "frequency_hz": float(m["frequency_hz"]),
                    "target_frequency_hz": trow.get("target_frequency_hz"),
                    "dominant_region": m.get("dominant_region"),
                    "top_participation": m.get("top_participation"),
                    "back_participation": m.get("back_participation"),
                    "air_participation": m.get("air_participation"),
                    "participation_method": m.get("participation_method"),
                    "participation_status": m.get("participation_status"),
                }
                for key in AUDIO_COUPLING_FIELD_KEYS:
                    if key in m:
                        row[key] = m[key]
                mode_records.append(row)

    if raw_diagnostic:
        diag_count = write_worker_diagnostic_from_solver_targets(
            output_dir=output_dir,
            chunk_targets=chunk_targets,
            solver_targets=list(result.get("targets") or []),
            shape_name=shape_name,
        )
        result["raw_modal_diagnostic_count"] = diag_count
    write_json_atomic(output_dir / "solver_result.json", result)
    worker_body = {
        "schema": "m4_worker_result_v1",
        "chunk_id": chunk_targets.get("chunk_id"),
        "worker_id": None,
        "status": result.get("status"),
        "targets_attempted": len(targets_hz),
        "targets_passed": int((result.get("aggregate") or {}).get("targets_succeeded") or 0),
        "accepted_modes": (result.get("aggregate") or {}).get("unique_accepted_frequencies_hz") or [],
        "accepted_mode_records": mode_records,
        "unique_modes": (result.get("aggregate") or {}).get("unique_accepted_frequencies_hz") or [],
        "timing": {
            "wall_seconds": (result.get("aggregate") or {}).get("total_wall_seconds"),
            "setup_seconds": (result.get("aggregate") or {}).get("total_setup_seconds"),
            "solve_seconds": (result.get("aggregate") or {}).get("total_solve_seconds"),
        },
        "warnings": [],
        "errors": [result["failure_reason"]] if result.get("failure_reason") else [],
        "box_raw_modal_discovery": raw_diagnostic,
        "box_bypass_target_window_acceptance": box_bypass_target_window_acceptance,
        "solver_result_json": str(output_dir / "solver_result.json"),
        "generated_utc": result.get("generated_utc"),
    }
    write_json_atomic(output_dir / "worker_result.json", worker_body)
    (output_dir / "log.txt").write_text(
        f"status={result.get('status')} targets={len(targets_hz)} succeeded={worker_body['targets_passed']}\n",
        encoding="utf-8",
    )
    write_json(output_dir / PIPELINE_SOLVE_MANIFEST, {
        "generated_utc": result.get("generated_utc"),
        "stage": "solver_mkl_target_list",
        "status": result.get("status"),
        "solver_result_json": str(output_dir / "solver_result.json"),
    })
    return 0 if result.get("status") == "PASS" else 2


def run_solve_target_list(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    targets_path = Path(args.targets_json).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not targets_path.is_file():
        print(f"error: missing --targets-json: {targets_path}", file=sys.stderr)
        return 2

    try:
        chunk_targets = _load_chunk_targets(targets_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: invalid targets-json: {exc}", file=sys.stderr)
        return 2

    val_errors = validate_chunk_targets_doc(chunk_targets)
    if val_errors:
        print("error: chunk_targets validation failed:", file=sys.stderr)
        for e in val_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    factor_solver = str(args.factor_solver).strip().lower()
    cmd_preview = (
        f"python {STAGE_B_PLANNED} "
        f'--checkpoint-dir "{checkpoint}" '
        f'--targets-json "{targets_path}" '
        f"--factor-solver {factor_solver} "
        f'--output-dir "{output_dir}"'
    )

    if args.dry_run:
        solver_body = build_solver_result_placeholder(
            chunk_targets=chunk_targets,
            checkpoint_dir=checkpoint,
            factor_solver=factor_solver,
        )
        worker_body = build_worker_result_placeholder(
            chunk_id=str(chunk_targets.get("chunk_id")),
            worker_id=None,
            chunk_targets=chunk_targets,
            output_dir=output_dir,
        )
        write_json_atomic(output_dir / "solver_result.json", solver_body)
        write_json_atomic(output_dir / "worker_result.json", worker_body)
        (output_dir / "log.txt").write_text(
            "M4.4.1a dry-run — no solve executed.\n"
            f"command_preview={cmd_preview}\n",
            encoding="utf-8",
        )
        print("will_execute=false")
        print(f"chunk_id={chunk_targets.get('chunk_id')}")
        print(f"target_count={len(chunk_targets.get('targets') or [])}")
        print(f"command_preview={cmd_preview}")
        return 0

    require_mkl = factor_solver == "mkl_pardiso" and not args.skip_mkl_probe
    ok, messages = verify_solver_mkl_stage_environment(require_mkl_pardiso=require_mkl)
    if not ok:
        fail_with_messages("B3_checkpoint_solve_target_list", messages)

    if factor_solver == "mumps":
        mumps_ok, mumps_err = verify_mumps_available()
        if not mumps_ok:
            fail_with_messages("B3_checkpoint_solve_target_list", [f"mumps unavailable: {mumps_err}"])

    ckpt_ok, ckpt_errors, _ = verify_checkpoint_complete(
        checkpoint,
        require_csr=False,
        require_export_manifest=True,
    )
    if not ckpt_ok:
        fail_with_messages("B3_checkpoint_solve_target_list", ckpt_errors)

    mat_ok, mat_errors, _ = verify_checkpoint_matrices(checkpoint)
    if not mat_ok:
        fail_with_messages("B3_checkpoint_solve_target_list", mat_errors)

    return _run_real_solve(
        args=args,
        chunk_targets=chunk_targets,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_solve_target_list(argv)


if __name__ == "__main__":
    raise SystemExit(main())
