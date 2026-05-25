#!/usr/bin/env python3
"""
Phase-2 production-parameter validation (length, width, wood species).

Preserves phase-1 radius/depth/thickness artifacts. Resumes completed work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_common import (
    DIAG_DIR,
    PRODUCTION_SUMMARY_JSON,
    VALIDATION_MESH,
    capture_branch_with_locator_then_coupled,
    capture_branch_with_retries,
    load_baseline_structural_mac_catalog,
    load_phase1_preserved_results,
    load_phase2_reusable_rows,
    load_production_manifest,
    load_saved_mesh_gates,
    production_sample_by_id,
    row_from_existing_solve_artifacts,
    row_from_solve,
    sample_has_completed_solve,
    write_phase2_incremental,
)
from v2_sensitivity_gates import run_mesh_gates
from v2_sensitivity_mesh import build_sample_mesh, sample_geometry, sample_mesh_path


def _is_geometry_sample(sample: Dict[str, Any]) -> bool:
    return bool(sample.get("requires_remesh") or sample.get("acoustic_geometry_test"))


def _is_material_sample(sample: Dict[str, Any]) -> bool:
    return bool(sample.get("materials")) and not sample.get("requires_remesh")


def _resolve_mesh(sample: Dict[str, Any]) -> Path:
    if sample.get("reuse_baseline_mesh"):
        return VALIDATION_MESH
    if sample.get("requires_remesh"):
        return sample_mesh_path(str(sample["id"]))
    return VALIDATION_MESH


def _process_geometry_sample(sample: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    geom = sample_geometry(sample)
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "phase": sample.get("phase", 2),
        "varied_parameter_or_material": (sample.get("expected_direction") or {}).get("parameter"),
        "geometry_values": {
            k: geom[k]
            for k in ("length", "width", "depth", "hole_radius", "top_thickness")
        },
        "requires_remesh": True,
    }
    mesh_path = _resolve_mesh(sample)
    case_dir = DIAG_DIR.parent / "samples" / sample_id
    if not mesh_path.is_file():
        try:
            mesh_path = build_sample_mesh(sample)
            row["mesh_built"] = True
        except Exception as exc:
            return {
                **row,
                "status": "mesh_build_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    row["mesh_file"] = str(mesh_path)
    saved_gates = load_saved_mesh_gates(sample_id, hole_radius_m=float(geom["hole_radius"]))
    if saved_gates and saved_gates.get("combined_mesh_gate_pass"):
        row["mesh_gates"] = saved_gates
    else:
        gates_dir = case_dir / "diagnostics" / "gates"
        gates = run_mesh_gates(
            mesh_path,
            hole_radius_m=float(geom["hole_radius"]),
            gates_dir=gates_dir,
        )
        row["mesh_gates"] = gates
        from v2_sensitivity_common import write_json

        write_json(gates_dir / "mesh_gates_summary.json", gates)
        if not gates.get("combined_mesh_gate_pass"):
            return {
                **row,
                "status": "mesh_gate_failed",
                "locator_status": "skipped_mesh_gate_failed",
                "error": "combined_mesh_gate_pass=False",
            }
    solve, attempts, locator_meta = capture_branch_with_locator_then_coupled(
        sample, mesh_path, manifest
    )
    row = row_from_solve(
        sample,
        solve,
        mesh_gates=row["mesh_gates"],
        attempts_log=attempts,
        locator_meta=locator_meta,
    )
    row["expected_direction_or_interpretation"] = (sample.get("expected_direction") or {}).get(
        "interpretation"
    )
    return row


def _refresh_material_row(sample: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    row = row_from_existing_solve_artifacts(sample, manifest)
    if row is None:
        return {
            "sample_id": str(sample["id"]),
            "status": "missing_artifacts",
            "error": "material sample solve artifacts missing",
        }
    row["resume_action"] = "material_mac_refresh_no_resolve"
    return row


def _resume_sample(
    sample: Dict[str, Any],
    manifest: Dict[str, Any],
    reusable: Dict[str, Dict[str, Any]],
    *,
    geometry_only: bool,
    material_mac_only: bool,
) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    if _is_material_sample(sample):
        if geometry_only:
            return {"sample_id": sample_id, "status": "skipped", "resume_action": "geometry_only_skip"}
        if sample_has_completed_solve(sample_id):
            row = _refresh_material_row(sample, manifest)
            row["resume_action"] = "material_mac_refresh"
            return row
        return {
            "sample_id": sample_id,
            "status": "missing_artifacts",
            "error": "material sample not solved",
        }

    if material_mac_only:
        return {"sample_id": sample_id, "status": "skipped", "resume_action": "material_mac_only_skip"}

    if sample_has_completed_solve(sample_id):
        row = row_from_existing_solve_artifacts(sample, manifest)
        if row is not None:
            row["resume_action"] = "geometry_rebuilt_from_solve_artifacts"
            return row

    if sample_id in reusable and reusable[sample_id].get("status") == "ok":
        row = dict(reusable[sample_id])
        row["resume_action"] = "reused_summary_row"
        return row

    return _process_geometry_sample(sample, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 production-parameter validation stage")
    parser.add_argument("--sample-id", type=str, default="")
    parser.add_argument("--skip-optional", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Process length/width samples only (locator + coupled)",
    )
    parser.add_argument(
        "--material-mac-only",
        action="store_true",
        help="Refresh material rows/MAC from artifacts; no material re-solve",
    )
    parser.add_argument(
        "--capture-baseline-mac",
        action="store_true",
        help="Capture baseline structural MAC reference if missing (post-only replay)",
    )
    args = parser.parse_args()

    if args.capture_baseline_mac:
        import capture_baseline_structural_mac_reference as cap_mod

        cat = load_baseline_structural_mac_catalog()
        if not cat.get("ready"):
            print("[v2_production] capturing baseline structural MAC reference...", flush=True)
            rc = cap_mod.main()
            if rc != 0:
                print(
                    "[v2_production] baseline MAC reference capture failed; "
                    "material structural MAC will remain unavailable",
                    file=sys.stderr,
                )

    manifest = load_production_manifest()
    phase2_ids: List[str] = list(manifest.get("phase2_sample_ids") or [])
    samples = [production_sample_by_id(manifest, sid) for sid in phase2_ids]
    if args.skip_optional:
        samples = [s for s in samples if not s.get("optional")]
    if args.sample_id:
        samples = [s for s in samples if str(s["id"]) == args.sample_id]
    if args.geometry_only:
        samples = [s for s in samples if _is_geometry_sample(s)]
    if args.material_mac_only:
        samples = [s for s in samples if _is_material_sample(s)]

    resume = (args.resume or args.report_only or args.material_mac_only) and not args.force_rerun
    reusable = load_phase2_reusable_rows(phase2_ids) if resume else {}

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_phase1_preserved_results(manifest)
    if PRODUCTION_SUMMARY_JSON.is_file():
        import json

        prior = json.loads(PRODUCTION_SUMMARY_JSON.read_text(encoding="utf-8"))
        for sid in phase2_ids:
            if sid in (prior.get("samples") or {}) and sid not in results:
                results[sid] = dict(prior["samples"][sid])

    reuse_log: Dict[str, str] = {}

    for sample in samples:
        sid = str(sample["id"])
        print(
            f"[v2_production] sample={sid} resume={resume} "
            f"geometry_only={args.geometry_only} material_mac_only={args.material_mac_only}",
            flush=True,
        )
        if resume:
            row = _resume_sample(
                sample,
                manifest,
                reusable,
                geometry_only=args.geometry_only,
                material_mac_only=args.material_mac_only,
            )
            reuse_log[sid] = str(row.get("resume_action", "unknown"))
        elif _is_material_sample(sample):
            row = _refresh_material_row(sample, manifest)
            reuse_log[sid] = row.get("resume_action", "material_refresh")
        else:
            row = _process_geometry_sample(sample, manifest)
            reuse_log[sid] = "full_geometry_run"
        results[sid] = row
        write_phase2_incremental(results, manifest, phase2_ids=phase2_ids)

    radius_trend = results.pop("_radius_trend_evaluation", None)
    from v2_sensitivity_common import _phase2_staged_promotion, write_json

    promotion = _phase2_staged_promotion(results, phase2_ids)
    summary = {
        "suite": manifest.get("suite"),
        "phase": manifest.get("phase"),
        "frozen_formulation": manifest.get("frozen_formulation"),
        "coupled_baseline": manifest.get("coupled_baseline"),
        "preserve_phase1_sample_ids": manifest.get("preserve_phase1_sample_ids"),
        "phase2_sample_ids": phase2_ids,
        "radius_trend_evaluation": radius_trend,
        "resume_log": reuse_log,
        "baseline_structural_mac_ready": bool(load_baseline_structural_mac_catalog().get("ready")),
        "geometry_samples_reused": {
            sid: reuse_log.get(sid)
            for sid in phase2_ids
            if sid.startswith(("length_", "width_"))
        },
        "samples": {k: v for k, v in results.items() if not str(k).startswith("_")},
        **promotion,
    }
    write_json(PRODUCTION_SUMMARY_JSON, summary)

    check_ids = [
        sid
        for sid in phase2_ids
        if not args.geometry_only or sid.startswith(("length_", "width_"))
    ]
    if args.material_mac_only:
        check_ids = [s for s in phase2_ids if s.startswith("material_")]
    failed = [sid for sid in check_ids if (results.get(sid) or {}).get("status") != "ok"]
    if failed:
        print(f"[v2_production] failed or incomplete: {failed}", file=sys.stderr)
        print(f"[v2_production] resume_log={reuse_log}", file=sys.stderr)
        return 1
    print(f"[v2_production] wrote {PRODUCTION_SUMMARY_JSON}")
    print(f"[v2_production] resume_log={reuse_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
