#!/usr/bin/env python3
"""Standalone aperture-mask diagnostic + build for existing validation checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_aperture_pressure_mask import diagnose_aperture_pressure_mask  # noqa: E402
from v2_b3_m4_lhs_pool_bridge import load_lhs_pool  # noqa: E402
from v2_b3_m4_validation_lib import (  # noqa: E402
    MASK_ARTIFACT_NAMES,
    run_aperture_mask_stage,
    validation_run_id,
    validation_run_root,
    validation_tree_ready,
    verify_mask_artifacts,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel  # noqa: E402

DEFAULT_SAMPLES = ("sample_001", "sample_034")
DEFAULT_LHS = "ROM/classic/lhs_pool.json"


def _format_report(sample_id: str, diag: Mapping[str, Any], mask: Optional[Mapping[str, Any]]) -> str:
    sh = diag.get("soundhole_geometry") or {}
    lines = [
        f"=== {sample_id} ({validation_run_id(sample_id)}) ===",
        f"soundhole centre (m): {sh.get('center_m')}",
        f"soundhole radius (m): {sh.get('radius_m')}",
        f"soundhole source: {sh.get('source')}",
        f"air pressure DOF count: {diag.get('air_pressure_dof_count')}",
        f"min distance any pressure DOF to centre (m): {diag.get('min_distance_any_air_pressure_dof_to_center_m')}",
        f"facet tag counts: {diag.get('facet_tag_counts')}",
    ]
    if mask:
        lines.extend(
            [
                "--- mask build ---",
                f"MASK_STAGE_STATUS: {mask.get('MASK_STAGE_STATUS')}",
                f"selection method: {mask.get('mask_method')}",
                f"p_idx_aperture_count: {mask.get('p_idx_aperture_count')}",
                f"mic_output_method: {mask.get('mic_output_method')}",
            ]
        )
    return "\n".join(lines)


def _resolve_samples(args: argparse.Namespace) -> List[str]:
    if args.sample:
        return [str(args.sample).strip()]
    return [s.strip() for s in str(args.samples).split(",") if s.strip()]


def audit_sample(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    build_mask: bool,
    reuse_validation_checkpoint: bool,
) -> Dict[str, Any]:
    val_root = validation_run_root(repo_root, sample_id)
    mesh = val_root / "lprod" / "mesh" / "L_prod" / f"{sample_id}.msh"
    ckpt = val_root / "lprod" / "checkpoint" / "built_metadata.json"
    core_cfg = val_root / "lprod" / "resolved_core_config.json"
    val_dir = val_root / "validation"

    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "validation_run_id": validation_run_id(sample_id),
        "validation_run_root": rel(val_root, repo_root=repo_root),
        "mesh_present": mesh.is_file(),
        "checkpoint_present": ckpt.is_file(),
        "validation_tree_ready": validation_tree_ready(val_root),
        "reuse_validation_checkpoint": bool(reuse_validation_checkpoint),
    }

    if reuse_validation_checkpoint and not validation_tree_ready(val_root):
        out["status"] = "FAIL"
        out["mask_build_status"] = "NOT_RUN"
        out["mask_build_error"] = "validation_tree_not_ready_for_reuse"
        return out
    if not mesh.is_file() or not ckpt.is_file():
        out["status"] = "FAIL"
        out["mask_build_status"] = "NOT_RUN"
        out["mask_build_error"] = "missing_validation_tree"
        return out

    built = load_json(ckpt)
    out["active_dimension"] = built.get("active_dimension")
    out["n_w"] = built.get("n_w")
    out["n_p_air"] = len(built.get("p_idx") or [])

    if not build_mask:
        from v2_b3_m4_lprod_interfaces import extract_geometry_dict  # noqa: WPS433
        from v2_b3_m4_lhs_pool_bridge import lhs_entry_index  # noqa: WPS433

        idx = lhs_entry_index(pool, sample_id)
        entry = (pool.get("entries") or [])[idx] if idx is not None else {}
        geom = extract_geometry_dict(entry)
        diag = diagnose_aperture_pressure_mask(
            mesh,
            geometry=geom,
            built_meta=built,
            core_config_path=core_cfg if core_cfg.is_file() else None,
        )
        out["diagnostic"] = diag
        out["text_report"] = _format_report(sample_id, diag, None)
        out["status"] = "diagnostic_only"
        out["mask_build_status"] = "NOT_RUN"
        return out

    try:
        mask_summary = run_aperture_mask_stage(
            repo_root=repo_root,
            val_root=val_root,
            generated_mesh=mesh,
            pool=pool,
            sample_id=sample_id,
            core_config_path=core_cfg if core_cfg.is_file() else None,
            write_diagnostics=True,
        )
    except Exception as exc:  # noqa: BLE001
        out["status"] = "FAIL"
        out["mask_build_status"] = "FAIL"
        out["mask_build_error"] = f"{type(exc).__name__}:{exc}"
        fail_json = val_dir / "aperture_mask_failure.json"
        if fail_json.is_file():
            try:
                out["aperture_mask_failure"] = json.loads(fail_json.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        ok, missing, present = verify_mask_artifacts(val_dir)
        out["mask_artifacts_present"] = present
        out["mask_artifacts_missing"] = missing
        out["text_report"] = f"MASK_STAGE_STATUS=FAIL sample={sample_id} error={exc}"
        return out

    ok, missing, present = verify_mask_artifacts(val_dir)
    out.update(mask_summary.get("mask_stage") or {})
    out["mask_build_status"] = "PASS"
    out["p_idx_aperture_count"] = mask_summary.get("p_idx_aperture_count")
    out["mask_method"] = mask_summary.get("mask_method")
    out["mic_output_method"] = mask_summary.get("mic_output_method")
    out["mask_artifacts_present"] = present
    out["mask_artifacts_missing"] = missing
    out["required_artifact_paths"] = {name: str((val_dir / name).resolve()) for name in MASK_ARTIFACT_NAMES}
    diag_path = val_dir / "aperture_mask_diagnostic.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}
    out["text_report"] = _format_report(sample_id, diag, out)
    out["status"] = "PASS" if ok and int(out.get("p_idx_aperture_count") or 0) > 0 else "FAIL"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose/build aperture pressure mask on existing validation checkpoints."
    )
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument("--sample", default=None, help="Single sample id, e.g. sample_001")
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--build-mask", action="store_true", help="Build mask artifacts (no eigensolver).")
    parser.add_argument(
        "--reuse-validation-checkpoint",
        action="store_true",
        help="Require existing validation checkpoint; do not copy production mesh/config.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    samples = _resolve_samples(args)

    report = {
        "schema": "aperture_pressure_mask_diagnostic_v2",
        "samples": samples,
        "build_mask": bool(args.build_mask),
        "reuse_validation_checkpoint": bool(args.reuse_validation_checkpoint),
        "results": [
            audit_sample(
                repo_root=repo_root,
                pool=pool,
                sample_id=sid,
                build_mask=bool(args.build_mask),
                reuse_validation_checkpoint=bool(args.reuse_validation_checkpoint),
            )
            for sid in samples
        ],
    }
    for row in report["results"]:
        print(row.get("text_report") or json.dumps(row, indent=2))
        print()

    if args.json_out:
        out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"json_out={out}")

    any_fail = any(r.get("status") not in ("PASS", "diagnostic_only") for r in report["results"])
    return 2 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
