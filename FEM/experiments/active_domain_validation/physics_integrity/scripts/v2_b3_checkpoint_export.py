#!/usr/bin/env python3
"""Production-stage operator checkpoint export (build + PETSc binary + CSR fallback)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
DEFAULT_CORE_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit
import v2_b3_dev_solver_benchmark as dev_bench
from v2_b3_block_compose_backend import CLI_BACKEND_ARG, apply_compose_backend_from_argv
from v2_b3_checkpoint_pipeline_lib import (
    B3_EXPORT_RICH_MODAL_DATA_ARG,
    B3_SYNTHESIS_REGION_DOFS_ARG,
    B3_SYNTHESIS_REGION_DOFS_ENV,
    PIPELINE_EXPORT_MANIFEST,
    default_checkpoint_dir,
    ensure_rich_modal_export_allowed,
    fail_with_messages,
    resolve_synthesis_region_dofs_mode,
    rich_modal_export_manifest_block,
    verify_checkpoint_complete,
    verify_checkpoint_matrices,
    verify_production_stage_environment,
    write_json,
)
from v2_b3_st_worker_scaling_benchmark import (
    _export_operators,
    _st_scaling_operator_contract_pass,
    _struct_active_count_policy,
)

ALLOWED_MESH_LEVELS = frozenset({"L_mid", "L_dev_dense", "L_prod"})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_core_config_provenance(core_config_arg: Optional[str]) -> Tuple[Optional[Path], Dict[str, Any]]:
    canonical = DEFAULT_CORE_CONFIG.resolve()
    canonical_body = json.loads(canonical.read_text(encoding="utf-8"))
    canonical_mats = canonical_body.get("materials") or {}
    canonical_fp = {
        "top_density": (canonical_mats.get("top") or {}).get("density"),
        "back_density": (canonical_mats.get("back") or {}).get("density"),
    }

    def _build_prov(path: Path, mode: str) -> Dict[str, Any]:
        body = json.loads(path.read_text(encoding="utf-8"))
        mats = body.get("materials") or {}
        return {
            "core_config_mode": mode,
            "core_config_path": str(path),
            "core_config_sha256": _sha256_file(path),
            "canonical_core_config_path": str(canonical),
            "material_fingerprint": {
                "top_density": (mats.get("top") or {}).get("density"),
                "back_density": (mats.get("back") or {}).get("density"),
            },
            "canonical_material_fingerprint": canonical_fp,
        }

    if not core_config_arg:
        return None, _build_prov(canonical, mode="default")
    override = Path(core_config_arg).expanduser().resolve()
    if not override.is_file():
        raise FileNotFoundError(f"--core-config not found: {override}")
    return override, _build_prov(override, mode="override")


def _core_config_manifest_block(prov: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "core_config_provenance": prov,
        "core_config_mode": prov.get("core_config_mode"),
        "core_config_path": prov.get("core_config_path"),
        "core_config_sha256": prov.get("core_config_sha256"),
        "canonical_core_config_path": prov.get("canonical_core_config_path"),
        "core_config_material_fingerprint": prov.get("material_fingerprint"),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-stage checkpoint export (no ST/EPS solve).",
    )
    parser.add_argument("--mesh-level", choices=sorted(ALLOWED_MESH_LEVELS), default="L_prod")
    parser.add_argument(
        "--output-dir",
        help="Checkpoint output directory. Default: v2_mesh_convergence/diagnostics/st_worker_scaling_<mesh>_<utc>",
    )
    parser.add_argument(
        CLI_BACKEND_ARG,
        dest="compose_backend",
        default="csr_bulk",
        help="Block compose backend for operator build (default: csr_bulk).",
    )
    parser.add_argument(
        B3_EXPORT_RICH_MODAL_DATA_ARG,
        dest="export_rich_modal_data",
        action="store_true",
        default=False,
        help="Opt-in rich modal export (active eigenvectors under rich_modal/).",
    )
    parser.add_argument(
        B3_SYNTHESIS_REGION_DOFS_ARG,
        dest="synthesis_region_dofs",
        choices=("off", "best_effort"),
        default=None,
        metavar="MODE",
        help=(
            "Stage A region DOF locate: off (default, no dolfinx locate) or best_effort "
            f"(isolated subprocess). Env: {B3_SYNTHESIS_REGION_DOFS_ENV}=off|best_effort"
        ),
    )
    parser.add_argument(
        "--core-config",
        dest="core_config",
        default=None,
        help=(
            "Optional resolved core config JSON (e.g. pipeline_runs/config_overlays/.../resolved_core_config.json). "
            f"Default: {DEFAULT_CORE_CONFIG.name}"
        ),
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def run_checkpoint_export(argv: Optional[List[str]] = None) -> int:
    ok, messages = verify_production_stage_environment()
    if not ok:
        fail_with_messages("B3_checkpoint_export", messages)

    args = _parse_args(argv)
    rich_modal_requested = bool(args.export_rich_modal_data)
    ensure_rich_modal_export_allowed(requested=rich_modal_requested, context="B3_checkpoint_export")
    try:
        region_dofs_mode = resolve_synthesis_region_dofs_mode(args.synthesis_region_dofs)
    except ValueError as exc:
        fail_with_messages("B3_checkpoint_export", [str(exc)])
    try:
        core_config_path, core_config_provenance = _resolve_core_config_provenance(args.core_config)
    except FileNotFoundError as exc:
        fail_with_messages("B3_checkpoint_export", [str(exc)])
    mesh_level = str(args.mesh_level)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    checkpoint = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_checkpoint_dir(mesh_level, run_id=run_id)
    )
    checkpoint.mkdir(parents=True, exist_ok=True)

    argv_for_backend = [f"{CLI_BACKEND_ARG}={args.compose_backend}"]
    apply_compose_backend_from_argv(argv_for_backend, mesh_level=mesh_level)

    pre = audit._precheck_allow_b3_jd_first_bounded_execution()
    if not pre.get("preassembly_contract_pass"):
        manifest = {
            "stage": "production_checkpoint_export",
            "status": "FAIL",
            "failure_reason": "preassembly_contract_pass=False",
            "checkpoint_dir": str(checkpoint),
            "precheck": pre,
            **_core_config_manifest_block(core_config_provenance),
        }
        write_json(checkpoint / PIPELINE_EXPORT_MANIFEST, manifest)
        print(f"[B3_checkpoint_export] FAIL precheck -> {checkpoint / PIPELINE_EXPORT_MANIFEST}", flush=True)
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
        )
        op_payload: Dict[str, Any] = {}
        contract_pass = _st_scaling_operator_contract_pass(op_payload, built=built, mesh_level=mesh_level)
        if not contract_pass:
            manifest = {
                "stage": "production_checkpoint_export",
                "status": "FAIL",
                "failure_reason": f"operator_contract_failed:{op_payload.get('failure_reason')}",
                "checkpoint_dir": str(checkpoint),
                "operator_contract": op_payload,
                **_core_config_manifest_block(core_config_provenance),
            }
            write_json(checkpoint / PIPELINE_EXPORT_MANIFEST, manifest)
            print(f"[B3_checkpoint_export] FAIL operator contract", flush=True)
            return 2

        export_meta = _export_operators(checkpoint, built=built, mesh_level=mesh_level)
        export_pass, missing, export_detail = verify_checkpoint_complete(checkpoint, require_csr=True)
        mat_ok, mat_errors, mat_detail = verify_checkpoint_matrices(checkpoint)

        built_meta = json.loads((checkpoint / "built_metadata.json").read_text(encoding="utf-8"))
        synthesis_export: Dict[str, Any] = {}
        synthesis_warnings: List[str] = []
        try:
            from v2_b3_synthesis_export import write_stage_a_synthesis_artifacts

            synthesis_export = write_stage_a_synthesis_artifacts(
                checkpoint,
                built=built,
                built_meta=built_meta,
                mesh_level=mesh_level,
                compose_backend=args.compose_backend,
                region_dofs_mode=region_dofs_mode,
                core_config_provenance=core_config_provenance,
            )
            warn = synthesis_export.pop("warning", None)
            if warn:
                synthesis_warnings.append(str(warn))
        except Exception as exc:
            synthesis_warnings.append(f"synthesis_artifacts_exception:{type(exc).__name__}:{exc}")
            try:
                from v2_b3_synthesis_export import write_stage_a_synthesis_artifacts

                synthesis_export = write_stage_a_synthesis_artifacts(
                    checkpoint,
                    built=built,
                    built_meta=built_meta,
                    mesh_level=mesh_level,
                    compose_backend=args.compose_backend,
                    region_dofs_mode="off",
                    core_config_provenance=core_config_provenance,
                )
                synthesis_export["recovery"] = "rewrote_synthesis_metadata_with_region_dofs_off"
                warn = synthesis_export.pop("warning", None)
                if warn:
                    synthesis_warnings.append(str(warn))
            except Exception as exc2:
                synthesis_export = {
                    "synthesis_metadata_json": False,
                    "region_dof_indices_status": "deferred_to_stage_c",
                    "region_dof_indices_file": None,
                    "region_dof_indices_mode": "off",
                    "region_dof_indices_error": f"{type(exc).__name__}:{exc}; recovery:{type(exc2).__name__}:{exc2}",
                }
                synthesis_warnings.append(
                    f"synthesis_metadata_write_failed:{type(exc2).__name__}:{exc2}"
                )
        elapsed = time.perf_counter() - t0
        manifest: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": "production_checkpoint_export",
            "status": "PASS" if export_pass and mat_ok else "FAIL",
            "mesh_level": mesh_level,
            "checkpoint_dir": str(checkpoint.resolve()),
            "compose_backend": args.compose_backend,
            "synthesis_region_dofs_mode": region_dofs_mode,
            "operator_build_elapsed_seconds": dev_bench._safe_float(elapsed),
            "export_pass": bool(export_pass),
            "export_missing_files": missing,
            "export_detail": export_detail,
            "matrix_verify_pass": bool(mat_ok),
            "matrix_verify_errors": mat_errors,
            "matrix_verify_detail": mat_detail,
            "built_metadata_summary": {
                "active_dimension": built_meta.get("active_dimension"),
                "mesh_level": built_meta.get("mesh_level"),
                "A_shape": built_meta.get("A_shape"),
                "M_shape": built_meta.get("M_shape"),
            },
            "portable_csr_export": export_meta.get("portable_csr_export"),
            "production_promotion": "BLOCKED",
            "no_automatic_production_promotion": True,
            "next_stage": "solver-mkl checkpoint solve",
            "synthesis_export": synthesis_export,
            "rich_modal_export": rich_modal_export_manifest_block(requested=rich_modal_requested),
            **_core_config_manifest_block(core_config_provenance),
        }
        if synthesis_warnings:
            manifest["warnings"] = synthesis_warnings
        write_json(checkpoint / PIPELINE_EXPORT_MANIFEST, manifest)
        if manifest["status"] != "PASS":
            print(f"[B3_checkpoint_export] FAIL export/matrix verify -> {checkpoint}", flush=True)
            return 2
        print(
            f"[B3_checkpoint_export] PASS checkpoint={checkpoint} "
            f"build_s={elapsed:.1f} active_dim={built_meta.get('active_dimension')}",
            flush=True,
        )
        return 0
    except Exception as exc:
        manifest = {
            "stage": "production_checkpoint_export",
            "status": "FAIL",
            "failure_reason": f"{type(exc).__name__}:{exc}",
            "checkpoint_dir": str(checkpoint),
            **_core_config_manifest_block(core_config_provenance),
        }
        write_json(checkpoint / PIPELINE_EXPORT_MANIFEST, manifest)
        print(f"[B3_checkpoint_export] FAIL {exc}", flush=True)
        return 2
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    return run_checkpoint_export(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
