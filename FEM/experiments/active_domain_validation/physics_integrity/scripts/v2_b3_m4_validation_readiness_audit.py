#!/usr/bin/env python3
"""Read-only validation readiness audit before first L_rom_prod solve."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    evaluate_legacy_reference_compatibility,
    load_durable_target_plan,
)
from v2_b3_m4_mesh_profile_provenance_lib import (  # noqa: E402
    create_external_validation_input_package,
    material_fingerprint,
    reconstruct_target_plan_from_durable,
)
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    count_forbidden_heavy_artifacts,
    verify_post_compaction_contract,
)
from v2_b3_m4_production_contracts import (  # noqa: E402
    evaluate_post_cleanup_region_dof_evidence,
    evaluate_production_region_dof_gate,
)
from v2_b3_m4_production_freeze import production_freeze_complete  # noqa: E402
from v2_b3_m4_sample_cleanup_barrier import (  # noqa: E402
    BARRIER_MANIFEST_REL,
    FAILURE_REPORT_REL,
    load_cleanup_barrier_manifest,
    require_cleanup_barrier_passed_for_validation,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

AGG_PASS = "AGGREGATION_PASS"
GUITARS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars")
KNOWN_REFERENCE_RUN_SUFFIXES = (
    "m4prod2_strict_clean4",
    "m4prod2_strict_val",
    "m4prod2",
    "m45dry1",
    "m4prod1",
)


def _is_completed_production_run(run_root: Path) -> Tuple[bool, Dict[str, Any]]:
    meta: Dict[str, Any] = {"run_root": str(run_root)}
    if not run_root.is_dir():
        meta["reason"] = "missing_run_root"
        return False, meta
    manifest_path = run_root / "pipeline_run_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        meta["pipeline_mode"] = manifest.get("mode")
        meta["terminal_status"] = manifest.get("terminal_status")
        if manifest.get("will_execute") is False and str(manifest.get("mode") or "").endswith("dry_run"):
            meta["reason"] = "dry_run_only"
            return False, meta
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if not agg_path.is_file():
        meta["reason"] = "missing_aggregation_result"
        return False, meta
    agg = load_json(agg_path)
    meta["aggregation_status"] = agg.get("status")
    if str(agg.get("status")) != AGG_PASS or not agg.get("final_aggregation_ready"):
        meta["reason"] = "aggregation_not_pass"
        return False, meta
    if not production_freeze_complete(run_root):
        meta["reason"] = "freeze_incomplete"
        return False, meta
    meta["reason"] = "completed_production_candidate"
    return True, meta


def discover_sample_runs(repo_root: Path, sample_id: str) -> List[Path]:
    base = repo_root / GUITARS_REL / sample_id / "runs"
    if not base.is_dir():
        return []
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def select_reference_run(repo_root: Path, sample_id: str, *, explicit_run_id: Optional[str]) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    if explicit_run_id:
        run_root = repo_root / GUITARS_REL / sample_id / "runs" / explicit_run_id
        ok, meta = _is_completed_production_run(run_root)
        candidates.append({"run_id": explicit_run_id, "completed": ok, **meta})
        return (run_root if ok else None), candidates

    for run_root in discover_sample_runs(repo_root, sample_id):
        ok, meta = _is_completed_production_run(run_root)
        candidates.append({"run_id": run_root.name, "completed": ok, **meta})
        if ok:
            return run_root, candidates
    return None, candidates


def audit_legacy_reference(
    *,
    repo_root: Path,
    run_root: Path,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "LEGACY_REFERENCE_READY": False,
        "checks": {},
        "errors": [],
    }
    if run_root is None or not run_root.is_dir():
        report["errors"].append("completed_reference_run_not_found")
        return report

    sample_id = run_root.parent.parent.name
    run_id = run_root.name
    report["sample_id"] = sample_id
    report["run_id"] = run_id
    report["run_root"] = str(run_root)

    ok, legacy_meta, legacy_errors = evaluate_legacy_reference_compatibility(
        run_root=run_root, repo_root=repo_root,
    )
    report["legacy_compatibility"] = legacy_meta
    report["errors"].extend(legacy_errors)

    barrier_ok, barrier_meta, barrier_errors = require_cleanup_barrier_passed_for_validation(
        repo_root=repo_root, run_root=run_root, label="legacy_reference",
    )
    report["cleanup_barrier"] = barrier_meta
    report["errors"].extend(barrier_errors)

    forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(run_root)
    report["checks"]["forbidden_heavy_artifact_count"] = forbidden_count
    if forbidden_count != 0:
        report["errors"].append(f"forbidden_heavy_artifacts:{forbidden_paths}")

    barrier = load_cleanup_barrier_manifest(run_root)
    report["checks"]["cleanup_barrier_status"] = (barrier or {}).get("status")
    if (run_root / FAILURE_REPORT_REL).is_file():
        report["errors"].append("cleanup_failure_report_present")

    compaction = verify_post_compaction_contract(run_root)
    report["compaction_verify"] = compaction
    if not compaction.get("pass"):
        report["errors"].extend(compaction.get("errors") or [])

    checkpoint_present = (run_root / "lprod" / "checkpoint").is_dir()
    cleanup_completed = str((barrier or {}).get("status") or "") == "completed"
    report["checks"]["lprod_checkpoint_present"] = checkpoint_present
    report["checks"]["cleanup_barrier_status"] = (barrier or {}).get("status")

    if checkpoint_present:
        region_ok, region_errors = evaluate_production_region_dof_gate(run_root, repo_root=repo_root)
        report["region_dof_gate"] = {
            "pass": region_ok,
            "evidence_mode": "live_checkpoint",
            "errors": region_errors,
        }
    elif cleanup_completed:
        region_ok, region_errors, region_meta = evaluate_post_cleanup_region_dof_evidence(run_root)
        report["region_dof_gate"] = {
            "pass": region_ok,
            "evidence_mode": "durable_post_cleanup",
            "errors": region_errors,
            **region_meta,
        }
    else:
        region_ok, region_errors = evaluate_production_region_dof_gate(run_root, repo_root=repo_root)
        report["region_dof_gate"] = {
            "pass": region_ok,
            "evidence_mode": "live_checkpoint_missing",
            "errors": region_errors,
        }
    if not region_ok:
        report["errors"].extend(region_errors)

    report["checks"]["reference_controls_m"] = legacy_meta.get("reference_controls_m")
    report["checks"]["reference_controls_sources"] = legacy_meta.get("reference_controls_sources")

    report["LEGACY_REFERENCE_READY"] = (
        ok
        and barrier_ok
        and forbidden_count == 0
        and compaction.get("pass")
        and region_ok
    )
    return report


def audit_target_plan(
    *,
    repo_root: Path,
    run_root: Optional[Path],
    sample_id: str,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "TARGET_PLAN_READY": False,
        "TARGET_PLAN_UNAVAILABLE": False,
    }
    if run_root is None:
        report["TARGET_PLAN_UNAVAILABLE"] = True
        report["errors"] = ["completed_reference_run_not_found"]
        return report

    plan, method, errors, sources = reconstruct_target_plan_from_durable(run_root)
    report["reconstruction_method"] = method
    report["source_artifacts"] = sources
    if plan is None or "TARGET_PLAN_UNAVAILABLE" in errors:
        report["TARGET_PLAN_UNAVAILABLE"] = True
        report["errors"] = errors or ["TARGET_PLAN_UNAVAILABLE"]
        return report

    durable_body, durable_sha, durable_errs = load_durable_target_plan(run_root)
    if durable_body and durable_sha:
        report["in_run_durable_target_plan_sha256"] = durable_sha
        report["in_run_durable_target_count"] = len(durable_body.get("targets_hz") or [])

    geom_fp = None
    mat_fp = None
    sample_in_path = run_root / "sample" / "sample_input.json"
    if sample_in_path.is_file():
        sample_in = load_json(sample_in_path)
        geom = extract_geometry_dict(sample_in)
        if geom:
            geom_fp = geometry_fingerprint(geom)
        mat_fp = material_fingerprint(sample_in)

    pkg_root, pkg_report = create_external_validation_input_package(
        repo_root=repo_root,
        reference_run_root=run_root,
        sample_id=sample_id,
        run_id=run_root.name,
        geometry_fingerprint=geom_fp,
        material_fp=mat_fp,
    )
    report.update(pkg_report)
    report["TARGET_PLAN_READY"] = bool(pkg_report.get("TARGET_PLAN_READY"))
    report["external_package_root"] = str(pkg_root) if pkg_root else None
    return report


def audit_readiness(
    *,
    repo_root: Path,
    sample_id: str = "sample_002",
    reference_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_root, candidates = select_reference_run(repo_root, sample_id, explicit_run_id=reference_run_id)
    legacy = audit_legacy_reference(repo_root=repo_root, run_root=run_root) if run_root else {
        "LEGACY_REFERENCE_READY": False,
        "errors": ["completed_reference_run_not_found"],
        "candidate_runs": candidates,
    }
    if run_root is None:
        legacy["candidate_runs"] = candidates
    target = audit_target_plan(repo_root=repo_root, run_root=run_root, sample_id=sample_id)

    ready = (
        legacy.get("LEGACY_REFERENCE_READY") is True
        and target.get("TARGET_PLAN_READY") is True
    )
    return {
        "schema": "m4_validation_readiness_audit_v1",
        "sample_id": sample_id,
        "reference_run_id": run_root.name if run_root else None,
        "candidate_runs": candidates,
        "LEGACY_REFERENCE_READY": legacy.get("LEGACY_REFERENCE_READY"),
        "TARGET_PLAN_READY": target.get("TARGET_PLAN_READY"),
        "TARGET_PLAN_UNAVAILABLE": target.get("TARGET_PLAN_UNAVAILABLE", False),
        "FINAL_STATUS": "READY_FOR_FIRST_L_ROM_PROD_SOLVE" if ready else "BLOCKED",
        "legacy_reference_audit": legacy,
        "target_plan_audit": target,
    }


def build_rom_command(
    *,
    target_plan_package: Path,
    run_id_suffix: str = "rom_prod_001",
    sample_id: str = "sample_002",
) -> str:
    run_id = f"{sample_id}_{run_id_suffix}"
    run_root_rel = (
        "FEM/experiments/active_domain_validation/physics_integrity/"
        f"pipeline_runs/guitars/{sample_id}/runs/{run_id}"
    )
    target_plan_rel = (
        "FEM/experiments/active_domain_validation/physics_integrity/"
        "pipeline_runs/validation_inputs/"
        "sample_sample_002_reference_0661505c893237ee/"
        "target_plan.json"
    )

    return "\n".join(
        [
            "cd ~/final-project",
            "source .venv/bin/activate",
            f"tmux new-session -d -s {run_id} \\",
            "  'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \\",
            "   python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \\",
            f"     --force-sample {sample_id} \\",
            f"     --run-id-suffix {run_id_suffix} \\",
            "     --mesh-profile rom \\",
            "     --dataset-version m4_geometry_corrected_rommesh_v1 \\",
            "     --workers 3 \\",
            f"     --target-plan-file {target_plan_rel} \\",
            "     --execute \\",
            "     --compact-after-sample; \\",
            f"   echo exit_code=\\$? run_id={run_id} run_root={run_root_rel}'",
            f"tmux attach -t {run_id}",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit validation readiness for first L_rom_prod solve.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--sample-id", default="sample_002")
    parser.add_argument("--reference-run-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or detect_repo_root(SCRIPT_DIR)).resolve()
    report = audit_readiness(
        repo_root=repo_root,
        sample_id=str(args.sample_id),
        reference_run_id=args.reference_run_id,
    )
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))
    if report.get("FINAL_STATUS") == "READY_FOR_FIRST_L_ROM_PROD_SOLVE":
        pkg = (report.get("target_plan_audit") or {}).get("package_root")
        if pkg:
            print("\nROM_COMMAND (not executed):")
            print(build_rom_command(target_plan_package=Path(pkg)))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
