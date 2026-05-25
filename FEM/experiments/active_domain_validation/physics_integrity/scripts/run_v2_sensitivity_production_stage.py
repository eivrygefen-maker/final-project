#!/usr/bin/env python3
"""
Phase-2 production-parameter validation (length, width, wood species).

Preserves phase-1 radius/depth/thickness artifacts. Does not rerun them.
Does not modify coupled_physical_core_v2 formulation.
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
    load_phase1_preserved_results,
    load_production_manifest,
    production_sample_by_id,
    row_from_solve,
    write_json,
    write_validation_status,
)
from v2_sensitivity_gates import run_mesh_gates
from v2_sensitivity_mesh import build_sample_mesh, sample_geometry, sample_mesh_path


def _resolve_mesh(sample: Dict[str, Any]) -> Path:
    if sample.get("reuse_baseline_mesh"):
        return VALIDATION_MESH
    if sample.get("requires_remesh"):
        return sample_mesh_path(str(sample["id"]))
    return VALIDATION_MESH


def _process_sample(sample: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    geom = sample_geometry(sample)
    mats = sample.get("materials") or {}
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "phase": sample.get("phase", 2),
        "varied_parameter_or_material": (sample.get("expected_direction") or {}).get("parameter"),
        "geometry_values": {
            k: geom[k]
            for k in ("length", "width", "depth", "hole_radius", "top_thickness")
        },
        "material_assignment": dict(mats) if mats else None,
        "requires_remesh": bool(sample.get("requires_remesh")),
        "reuse_baseline_mesh": bool(sample.get("reuse_baseline_mesh")),
    }

    mesh_path = _resolve_mesh(sample)
    case_dir = DIAG_DIR.parent / "samples" / sample_id
    if sample.get("requires_remesh") and not mesh_path.is_file():
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
    if not mesh_path.is_file():
        return {**row, "status": "failed", "error": f"mesh missing: {mesh_path}"}

    if sample.get("reuse_baseline_mesh"):
        row["mesh_gates"] = {
            "combined_mesh_gate_pass": True,
            "reused_baseline_validation_mesh": True,
            "mesh_file": str(VALIDATION_MESH),
        }
        row["remesh_required"] = False
    else:
        gates_dir = case_dir / "diagnostics" / "gates"
        gates = run_mesh_gates(
            mesh_path,
            hole_radius_m=float(geom["hole_radius"]),
            gates_dir=gates_dir,
        )
        row["mesh_gates"] = gates
        row["remesh_required"] = True
        if not gates.get("combined_mesh_gate_pass"):
            return {
                **row,
                "status": "mesh_gate_failed",
                "error": "combined_mesh_gate_pass=False",
            }

    use_locator = bool(sample.get("acoustic_geometry_test") or sample.get("requires_remesh"))
    if use_locator:
        solve, attempts, locator_meta = capture_branch_with_locator_then_coupled(
            sample, mesh_path, manifest
        )
        row = row_from_solve(
            sample, solve, mesh_gates=row["mesh_gates"], attempts_log=attempts, locator_meta=locator_meta
        )
    else:
        solve, attempts = capture_branch_with_retries(sample, mesh_path, manifest)
        row = row_from_solve(
            sample, solve, mesh_gates=row["mesh_gates"], attempts_log=attempts
        )
    row["expected_direction_or_interpretation"] = (sample.get("expected_direction") or {}).get(
        "interpretation"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 production-parameter validation stage")
    parser.add_argument("--sample-id", type=str, default="", help="Run one phase-2 sample")
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip samples marked optional in manifest",
    )
    args = parser.parse_args()

    manifest = load_production_manifest()
    phase2_ids: List[str] = list(manifest.get("phase2_sample_ids") or [])
    samples = [production_sample_by_id(manifest, sid) for sid in phase2_ids]
    if args.skip_optional:
        samples = [s for s in samples if not s.get("optional")]
    if args.sample_id:
        samples = [s for s in samples if str(s["id"]) == args.sample_id]

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_phase1_preserved_results(manifest)

    for sample in samples:
        sid = str(sample["id"])
        print(f"[v2_production] sample={sid}", flush=True)
        results[sid] = _process_sample(sample, manifest)
        write_json(
            DIAG_DIR / "v2_production_validation_summary.partial.json",
            {"samples": {k: v for k, v in results.items() if not k.startswith("_")}},
        )

    radius_trend = results.pop("_radius_trend_evaluation", None)
    summary = {
        "suite": manifest.get("suite"),
        "phase": manifest.get("phase"),
        "frozen_formulation": manifest.get("frozen_formulation"),
        "coupled_baseline": manifest.get("coupled_baseline"),
        "preserve_phase1_sample_ids": manifest.get("preserve_phase1_sample_ids"),
        "phase2_sample_ids": phase2_ids,
        "radius_trend_evaluation": radius_trend,
        "samples": {k: v for k, v in results.items() if not k.startswith("_")},
        "note": (
            "Phase-2 only: length/width remesh + wood species on baseline mesh. "
            "Phase-1 radius/depth preserved; not re-solved."
        ),
    }
    write_json(PRODUCTION_SUMMARY_JSON, summary)
    write_validation_status(
        {k: v for k, v in results.items() if k in (manifest.get("preserve_phase1_sample_ids") or [])},
        {k: v for k, v in results.items() if k in phase2_ids},
        production_manifest=manifest,
    )
    failed = [
        sid
        for sid in phase2_ids
        if (results.get(sid) or {}).get("status") != "ok"
    ]
    if failed:
        print(f"[v2_production] failed samples: {failed}", file=sys.stderr)
        return 1
    print(f"[v2_production] wrote {PRODUCTION_SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
