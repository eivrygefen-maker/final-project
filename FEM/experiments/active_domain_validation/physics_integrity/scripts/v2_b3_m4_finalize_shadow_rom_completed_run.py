#!/usr/bin/env python3
"""Finalization-only recovery for a shadow-ROM completed run (no FEM, no ROM repredict)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_finalize_completed_run import (  # noqa: E402
    DEFAULT_LHS_REL,
    finalize_completed_run,
    is_run_already_finalized,
    resolve_run_root,
)
from v2_b3_m4_official_rom_dataset_lib import load_official_dataset_registry  # noqa: E402
from v2_b3_m4_production_freeze import read_production_acceptance_status  # noqa: E402
from v2_b3_m4_rom_shadow_pipeline_lib import (  # noqa: E402
    DEFAULT_RETRAIN_EVERY_N_NEW_SAMPLES,
    RetrainPolicy,
    RomShadowIntegrityError,
    attempt_register_and_retrain_after_cleanup,
    diagnose_shadow_rom_stages,
    ensure_durable_rom_comparison,
    print_shadow_rom_stages,
    prune_rom_directory_to_durable,
    verify_rom_prediction_summary,
)
from v2_b3_m4_shared_export import detect_shared_root  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _is_registered_in_official_dataset(
    repo_root: Path, *, run_id: str, shape_name: str = "classic"
) -> bool:
    return any(str(r.get("run_id") or "") == run_id for r in load_official_dataset_registry(repo_root, shape_name))


def finalize_shadow_rom_completed_run(
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
    lhs_path: Path,
    shared_root: Path,
    batch_id: Optional[str] = None,
    reconcile_bookkeeping: bool = True,
    workers_requested: int = 3,
    rom_retrain_every_n: int = DEFAULT_RETRAIN_EVERY_N_NEW_SAMPLES,
    diagnose_only: bool = False,
) -> Dict[str, Any]:
    run_root = resolve_run_root(repo_root, sample_id, run_id)
    if not run_root.is_dir():
        raise FileNotFoundError(f"run_root missing: {run_root}")

    shadow_stages = diagnose_shadow_rom_stages(run_root)
    report: Dict[str, Any] = {
        "schema": "m4_finalize_shadow_rom_completed_run_v1",
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": rel(run_root, repo_root=repo_root),
        "shadow_stages": shadow_stages,
        "fem_stages_executed": False,
        "rom_prediction_rerun": False,
        "rom_compare_rerun": False,
    }

    if diagnose_only:
        ok_pred, pred_meta = verify_rom_prediction_summary(run_root)
        report["rom_prediction_verified"] = ok_pred
        report["rom_prediction_meta"] = pred_meta if ok_pred else {"error": pred_meta}
        report["already_finalized"], report["already_finalized_evidence"] = is_run_already_finalized(run_root)
        report["already_registered"] = _is_registered_in_official_dataset(repo_root, run_id=run_id)
        print_shadow_rom_stages(shadow_stages)
        return report

    ok_pred, pred_meta = verify_rom_prediction_summary(run_root)
    if not ok_pred:
        raise RomShadowIntegrityError(f"invalid_rom_prediction_summary:{pred_meta}")
    report["rom_prediction_verified"] = True

    context: Optional[Dict[str, Any]] = None
    frozen_path = run_root / "rom" / "rom_prediction_frozen_internal.json"
    if frozen_path.is_file():
        try:
            frozen = load_json(frozen_path)
            context = dict(frozen.get("context") or {})
        except (OSError, ValueError, json.JSONDecodeError):
            context = None
    if not context and (run_root / "sample" / "sample_input.json").is_file():
        si = load_json(run_root / "sample" / "sample_input.json")
        context = {
            "sample_id": sample_id,
            "run_id": run_id,
            "shape_name": str(si.get("shape_name") or "classic"),
            "lhs_row_index": si.get("lhs_row_index"),
            "parameters": dict(si.get("parameters") or {}),
        }

    cmp_result = ensure_durable_rom_comparison(
        repo_root=repo_root,
        run_root=run_root,
        context=context,
        reuse_existing=True,
    )
    report["rom_comparison"] = cmp_result
    report["rom_compare_rerun"] = not bool(cmp_result.get("reused"))
    shadow_stages = diagnose_shadow_rom_stages(run_root)
    report["shadow_stages"] = shadow_stages

    already_finalized, already_evidence = is_run_already_finalized(run_root)
    report["already_finalized"] = already_finalized
    report["already_finalized_evidence"] = already_evidence

    if already_finalized and _is_registered_in_official_dataset(repo_root, run_id=run_id):
        report["outcome"] = "ALREADY_FINALIZED_AND_REGISTERED"
        prune_rom_directory_to_durable(run_root)
        shadow_stages = diagnose_shadow_rom_stages(run_root)
        report["shadow_stages"] = shadow_stages
        print_shadow_rom_stages(shadow_stages)
        return report

    if not already_finalized:
        fin_report = finalize_completed_run(
            repo_root=repo_root,
            sample_id=sample_id,
            run_id=run_id,
            lhs_path=lhs_path,
            shared_root=shared_root,
            batch_id=batch_id,
            reconcile_bookkeeping=reconcile_bookkeeping,
            workers_requested=workers_requested,
        )
        report["finalization"] = fin_report
        report["outcome"] = fin_report.get("outcome")
        if str(fin_report.get("outcome") or "") not in ("pass", "ALREADY_FINALIZED"):
            raise RuntimeError(f"finalization_failed:{fin_report.get('outcome')}")
    else:
        report["finalization"] = {"outcome": "ALREADY_FINALIZED", "skipped": True}

    acceptance = read_production_acceptance_status(run_root)
    reg = attempt_register_and_retrain_after_cleanup(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        shape_name=str((context or {}).get("shape_name") or "classic"),
        production_acceptance_pass=bool(acceptance.get("production_acceptance_pass")),
        policy=RetrainPolicy(retrain_every_n_new_samples=int(rom_retrain_every_n)),
    )
    report["rom_dataset_registration"] = reg
    shadow_stages = reg.get("shadow_stages") or diagnose_shadow_rom_stages(run_root)
    report["shadow_stages"] = shadow_stages
    removed = prune_rom_directory_to_durable(run_root)
    report["rom_pruned_files"] = removed

    if reg.get("registered"):
        report["outcome"] = "pass"
    elif already_finalized:
        report["outcome"] = "finalized_registration_blocked"
    else:
        report["outcome"] = "pass_registration_blocked"

    print_shadow_rom_stages(shadow_stages)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a numerically completed shadow-ROM run: verify ROM artifacts, "
            "export/compact/cleanup/reconcile, register dataset, optional retrain (no FEM)."
        )
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--batch-id", help="Optional batch id for bookkeeping reconcile.")
    parser.add_argument("--shared-root", type=Path, help="Shared export root (default: /media/sf_gmar).")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rom-retrain-every-n", type=int, default=DEFAULT_RETRAIN_EVERY_N_NEW_SAMPLES)
    parser.add_argument("--no-reconcile", action="store_true")
    parser.add_argument("--diagnose", action="store_true", help="Read-only diagnostic.")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    shared_root = detect_shared_root(args.shared_root)
    if not args.diagnose and shared_root is None:
        print("error: shared root not found", file=sys.stderr)
        return 2

    try:
        report = finalize_shadow_rom_completed_run(
            repo_root=repo_root,
            sample_id=str(args.sample_id),
            run_id=str(args.run_id),
            lhs_path=lhs_path,
            shared_root=shared_root or repo_root,
            batch_id=args.batch_id,
            reconcile_bookkeeping=not bool(args.no_reconcile),
            workers_requested=int(args.workers),
            rom_retrain_every_n=int(args.rom_retrain_every_n),
            diagnose_only=bool(args.diagnose),
        )
    except (FileNotFoundError, RomShadowIntegrityError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report_path = args.report_path
    if report_path is None:
        report_path = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/generated"
            / f"finalize_shadow_{args.sample_id}_{args.run_id}.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    print(f"sample_id={args.sample_id}")
    print(f"run_id={args.run_id}")
    print(f"outcome={report.get('outcome')}")
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print("fem_stages_executed=false")
    print("rom_prediction_rerun=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
