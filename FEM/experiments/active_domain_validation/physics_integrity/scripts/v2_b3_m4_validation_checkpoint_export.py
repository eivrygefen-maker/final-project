#!/usr/bin/env python3
"""Validation-only Stage A export: assemble operators from sample-specific mesh."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit
import v2_b3_dev_solver_benchmark as dev_bench
import v2_b3_checkpoint_export as prod_export
from v2_b3_block_compose_backend import CLI_BACKEND_ARG, apply_compose_backend_from_argv
from v2_b3_checkpoint_pipeline_lib import (
    PIPELINE_EXPORT_MANIFEST,
    fail_with_messages,
    verify_checkpoint_complete,
    verify_checkpoint_matrices,
    write_json,
)
from v2_b3_st_worker_scaling_benchmark import (
    _export_operators,
    _st_scaling_operator_contract_pass,
    _struct_active_count_policy,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation Stage A: operator build on sample-specific mesh (not production).",
    )
    parser.add_argument("--mesh-level", default="L_prod")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--core-config", required=True)
    parser.add_argument(
        "--use-sample-operator-mesh",
        required=True,
        help="Absolute path to sample_XXX.msh used for A/M assembly.",
    )
    parser.add_argument("--B3-block-compose-backend", dest="compose_backend", default="csr_bulk")
    parser.add_argument("--B3-synthesis-region-dofs", dest="synthesis_region_dofs", default="best_effort")
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _verify_operator_mesh_provenance(
    *,
    requested_mesh: Path,
    built: Dict[str, Any],
    built_meta_path: Path,
) -> None:
    requested = requested_mesh.expanduser().resolve()
    used = Path(str(built.get("operator_mesh_file_used") or "")).expanduser().resolve()
    if not used.is_file():
        raise RuntimeError(f"operator_mesh_file_used missing from built payload: {used}")
    if used != requested:
        raise RuntimeError(
            f"operator mesh provenance mismatch: requested={requested} used={used}"
        )
    meta = json.loads(built_meta_path.read_text(encoding="utf-8"))
    meta["operator_mesh_file_used"] = str(used)
    meta["validation_sample_operator_mesh"] = True
    write_json_atomic(built_meta_path, meta)


def run_validation_checkpoint_export(argv: Optional[List[str]] = None) -> int:
    ok, messages = prod_export.verify_production_stage_environment()
    if not ok:
        fail_with_messages("B3_validation_checkpoint_export", messages)

    args = _parse_args(argv)
    operator_mesh = Path(args.use_sample_operator_mesh).expanduser().resolve()
    if not operator_mesh.is_file():
        fail_with_messages(
            "B3_validation_checkpoint_export",
            [f"--use-sample-operator-mesh not found: {operator_mesh}"],
        )

    core_config_path, core_config_provenance = prod_export._resolve_core_config_provenance(args.core_config)
    mesh_level = str(args.mesh_level)
    checkpoint = Path(args.output_dir).expanduser().resolve()
    checkpoint.mkdir(parents=True, exist_ok=True)

    apply_compose_backend_from_argv([f"{CLI_BACKEND_ARG}={args.compose_backend}"], mesh_level=mesh_level)

    pre = audit._precheck_allow_b3_jd_first_bounded_execution()
    if not pre.get("preassembly_contract_pass"):
        write_json(
            checkpoint / PIPELINE_EXPORT_MANIFEST,
            {
                "stage": "validation_checkpoint_export",
                "status": "FAIL",
                "failure_reason": "preassembly_contract_pass=False",
                "operator_mesh_file": str(operator_mesh),
            },
        )
        return 2

    mats: List[Any] = []
    seen: Set[int] = set()
    t0 = time.perf_counter()
    try:
        built = audit._b3_build_corrected_structural_active_operators(
            mats_to_destroy=mats,
            mat_destroy_seen=seen,
            mesh_level=mesh_level,
            struct_active_count_policy=_struct_active_count_policy(mesh_level),
            operator_build_profile=None,
            core_config_path=core_config_path,
            core_config_provenance=core_config_provenance,
            operator_mesh_file=operator_mesh,
        )
        op_payload: Dict[str, Any] = {}
        if not _st_scaling_operator_contract_pass(op_payload, built=built, mesh_level=mesh_level):
            write_json(
                checkpoint / PIPELINE_EXPORT_MANIFEST,
                {
                    "stage": "validation_checkpoint_export",
                    "status": "FAIL",
                    "failure_reason": f"operator_contract_failed:{op_payload.get('failure_reason')}",
                    "operator_mesh_file": str(operator_mesh),
                },
            )
            return 2

        export_meta = _export_operators(checkpoint, built=built, mesh_level=mesh_level)
        export_pass, missing, export_detail = verify_checkpoint_complete(checkpoint, require_csr=True)
        mat_ok, mat_errors, mat_detail = verify_checkpoint_matrices(checkpoint)
        built_meta_path = checkpoint / "built_metadata.json"
        _verify_operator_mesh_provenance(
            requested_mesh=operator_mesh,
            built=built,
            built_meta_path=built_meta_path,
        )

        built_meta = json.loads(built_meta_path.read_text(encoding="utf-8"))
        synthesis_export: Dict[str, Any] = {}
        try:
            from v2_b3_synthesis_export import write_stage_a_synthesis_artifacts

            synthesis_export = write_stage_a_synthesis_artifacts(
                checkpoint,
                built=built,
                built_meta=built_meta,
                mesh_level=mesh_level,
                compose_backend=args.compose_backend,
                region_dofs_mode=str(args.synthesis_region_dofs),
                core_config_provenance=core_config_provenance,
                core_config_path=core_config_path,
                python_executable=sys.executable,
            )
        except Exception as exc:
            synthesis_export = {"error": f"{type(exc).__name__}:{exc}"}

        elapsed = time.perf_counter() - t0
        manifest = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": "validation_checkpoint_export",
            "status": "PASS" if export_pass and mat_ok else "FAIL",
            "validation_only": True,
            "operator_mesh_file": str(operator_mesh),
            "operator_mesh_file_used": built.get("operator_mesh_file_used"),
            "mesh_level": mesh_level,
            "checkpoint_dir": str(checkpoint.resolve()),
            "operator_build_elapsed_seconds": dev_bench._safe_float(elapsed),
            "built_metadata_summary": {
                "active_dimension": built_meta.get("active_dimension"),
                "n_w": built_meta.get("n_w"),
                "A_shape": built_meta.get("A_shape"),
                "M_shape": built_meta.get("M_shape"),
            },
            "export_pass": bool(export_pass),
            "matrix_verify_pass": bool(mat_ok),
            "portable_csr_export": export_meta.get("portable_csr_export"),
            "synthesis_export": synthesis_export,
            **prod_export._core_config_manifest_block(core_config_provenance),
        }
        write_json(checkpoint / PIPELINE_EXPORT_MANIFEST, manifest)
        if manifest["status"] != "PASS":
            return 2
        print(
            f"[B3_validation_checkpoint_export] PASS checkpoint={checkpoint} "
            f"mesh={operator_mesh.name} active_dim={built_meta.get('active_dimension')}",
            flush=True,
        )
        return 0
    except Exception as exc:
        write_json(
            checkpoint / PIPELINE_EXPORT_MANIFEST,
            {
                "stage": "validation_checkpoint_export",
                "status": "FAIL",
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "operator_mesh_file": str(operator_mesh),
            },
        )
        raise


def main() -> int:
    return run_validation_checkpoint_export()


if __name__ == "__main__":
    raise SystemExit(main())
