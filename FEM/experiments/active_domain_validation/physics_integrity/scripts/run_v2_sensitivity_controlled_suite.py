#!/usr/bin/env python3
"""
Controlled non-random v2 sensitivity suite (depth, top thickness, E_L).

Preserves passed radius-pilot artifacts. Does not re-solve hole_radius samples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_common import (
    DIAG_DIR,
    SENS_ROOT,
    VALIDATION_MESH,
    capture_branch_with_retries,
    evaluate_expected_direction,
    load_manifest,
    load_pilot_preserved_results,
    row_from_solve,
    sample_by_id,
    write_json,
    write_suite_summary,
)
from v2_sensitivity_gates import run_mesh_gates
from v2_sensitivity_mesh import build_sample_mesh, sample_geometry, sample_mesh_path


def _resolve_mesh(sample: Dict[str, Any]) -> Path:
    if sample.get("reuse_baseline_mesh"):
        return VALIDATION_MESH
    if sample.get("requires_remesh"):
        return sample_mesh_path(str(sample["id"]))
    return VALIDATION_MESH


def _process_controlled_sample(sample: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    case_dir = SENS_ROOT / "samples" / sample_id
    geom = sample_geometry(sample)
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "varied_parameters": {
            k: geom[k]
            for k in ("hole_radius", "depth", "top_thickness")
            if k in geom
        },
    }
    mo = sample.get("materials_override") or {}
    if (mo.get("top") or {}).get("E_L_scale") is not None:
        row["varied_parameters"]["top_E_L_scale"] = float(mo["top"]["E_L_scale"])

    mesh_path = _resolve_mesh(sample)
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

    gates_dir = case_dir / "diagnostics" / "gates"
    if sample.get("reuse_baseline_mesh"):
        row["mesh_gates"] = {
            "combined_mesh_gate_pass": True,
            "reused_baseline_validation_mesh": True,
            "mesh_file": str(VALIDATION_MESH),
        }
    else:
        gates = run_mesh_gates(
            mesh_path,
            hole_radius_m=float(geom["hole_radius"]),
            gates_dir=gates_dir,
        )
        row["mesh_gates"] = gates
        if not gates.get("combined_mesh_gate_pass"):
            return {
                **row,
                "status": "mesh_gate_failed",
                "error": "combined_mesh_gate_pass=False",
            }

    solve, attempts = capture_branch_with_retries(sample, mesh_path, manifest)
    row = row_from_solve(sample, solve, mesh_gates=row["mesh_gates"], attempts_log=attempts)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 controlled sensitivity suite")
    parser.add_argument("--sample-id", type=str, default="", help="Run one controlled sample")
    args = parser.parse_args()

    manifest = load_manifest()
    controlled_ids = list(manifest.get("controlled_sample_ids") or [])
    samples = [sample_by_id(manifest, sid) for sid in controlled_ids]
    if args.sample_id:
        samples = [s for s in samples if str(s["id"]) == args.sample_id]

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_pilot_preserved_results(manifest)

    for sample in samples:
        sid = str(sample["id"])
        print(f"[v2_sensitivity][controlled] sample={sid}", flush=True)
        results[sid] = _process_controlled_sample(sample, manifest)
        write_json(
            DIAG_DIR / "v2_sensitivity_validation_summary.partial.json",
            {"samples": {k: v for k, v in results.items() if not k.startswith("_")}},
        )

    for sid, row in list(results.items()):
        if sid.startswith("_"):
            continue
        sample = sample_by_id(manifest, sid) if sid in controlled_ids else None
        if sample and row.get("status") in ("ok", "acoustic_branch_not_captured"):
            row["expected_direction_evaluation"] = evaluate_expected_direction(
                sample, row, peer_results=results
            )

    write_suite_summary(
        {k: v for k, v in results.items() if not k.startswith("_")},
        manifest,
        controlled_suite=True,
        pilot_mode=False,
    )
    print(f"[v2_sensitivity][controlled] wrote {DIAG_DIR / 'v2_sensitivity_validation_summary.json'}")

    failed = [
        sid
        for sid in controlled_ids
        if (results.get(sid) or {}).get("status") != "ok"
    ]
    return 0 if not failed else 4


if __name__ == "__main__":
    raise SystemExit(main())
