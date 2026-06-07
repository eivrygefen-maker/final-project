#!/usr/bin/env python3
"""Task 1 diagnostics for empty p_idx_aperture root-cause analysis (validation-only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_aperture_pressure_mask import (  # noqa: E402
    _estimate_cell_length_m,
    _load_pressure_layout,
    _soundhole_facet_geometry,
    build_aperture_pressure_mask,
    diagnose_aperture_pressure_mask,
)
from v2_b3_m4_lhs_pool_bridge import load_lhs_pool  # noqa: E402
from v2_b3_m4_lprod_interfaces import extract_geometry_dict  # noqa: E402
from v2_b3_m4_validation_lib import (  # noqa: E402
    validation_run_id,
    validation_run_root,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json  # noqa: E402

DEFAULT_SAMPLES = ("sample_001", "sample_034")
DEFAULT_LHS = "ROM/classic/lhs_pool.json"


def _format_report(sample_id: str, diag: Mapping[str, Any], mask: Optional[Mapping[str, Any]]) -> str:
    sh = diag.get("soundhole_geometry") or {}
    lines = [
        f"=== {sample_id} ({validation_run_id(sample_id)}) ===",
        f"soundhole centre (m): {sh.get('center_m')}",
        f"soundhole radius (m): {sh.get('radius_m')}",
        f"soundhole source: {sh.get('source')}",
        f"operator mesh bbox min (m): {diag.get('mesh_bbox_min')}",
        f"operator mesh bbox max (m): {diag.get('mesh_bbox_max')}",
        f"air pressure DOF coord shape: {diag.get('air_pressure_coord_shape')}",
        f"air pressure DOF coord bbox min (m): {diag.get('air_pressure_coord_bbox_min')}",
        f"air pressure DOF coord bbox max (m): {diag.get('air_pressure_coord_bbox_max')}",
        f"min distance any pressure DOF to centre (m): {diag.get('min_distance_any_air_pressure_dof_to_center_m')}",
        "counts within radii:",
    ]
    for label in ("0.5r", "1.0r", "1.25r", "1.5r", "2.0r"):
        counts = diag.get("counts_within_radius") or {}
        lines.append(f"  {label}: {counts.get(label)}")
    lines.append(f"estimated local cell length (m): {diag.get('estimated_cell_length_m')}")
    slab = diag.get("thin_aperture_slab_counts") or {}
    lines.append("counts in thin aperture cylinder/slab:")
    for k, v in slab.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"facet tag counts: {diag.get('facet_tag_counts')}")
    lines.append(f"n_soundhole facets (tag 2): {sh.get('n_soundhole_facets')}")
    lines.append(f"legacy vertex-indexing bug count at 1r: {diag.get('legacy_vertex_indexing_bug_count_at_1r')}")
    lines.append(f"mapping contract: {diag.get('mapping_contract')}")
    lines.append(f"n_u_b3: {diag.get('n_u_b3')}  p_idx_len: {diag.get('p_idx_len')}  active_local_len: {diag.get('active_local_len')}")
    lines.append("root cause notes:")
    for note in diag.get("root_cause_notes") or []:
        lines.append(f"  - {note}")
    if mask:
        lines.extend(
            [
                "--- rebuilt mask ---",
                f"selection method: {mask.get('mask_method')}",
                f"mic_output_method: {mask.get('mic_output_method')}",
                f"p_idx_aperture_count: {mask.get('n_p_aperture_dofs')}",
                f"coordinate bbox min: {mask.get('coordinate_bbox_min')}",
                f"coordinate bbox max: {mask.get('coordinate_bbox_max')}",
                f"distance stats (m): {mask.get('distance_to_center_stats_m')}",
                f"selection meta: {mask.get('selection_meta')}",
            ]
        )
    return "\n".join(lines)


def _infer_root_causes(diag: Mapping[str, Any]) -> List[str]:
    causes: List[str] = []
    counts = diag.get("counts_within_radius") or {}
    sh = diag.get("soundhole_geometry") or {}
    facet_counts = diag.get("facet_tag_counts") or {}
    legacy = int(diag.get("legacy_vertex_indexing_bug_count_at_1r") or 0)
    correct_1r = int(counts.get("1.0r") or 0)

    if legacy == 0 and correct_1r > 0:
        causes.append("1_wrong_vertex_indexing: previous mask used mesh vertex coords indexed by pressure DOF ids")
    if legacy > 0 and correct_1r == 0:
        causes.append("2_pressure_dofs_not_on_boundary: cell-based pressure coords may miss facet plane")
    if int(facet_counts.get("2") or 0) == 0:
        causes.append("4_aperture_facet_tag_missing: no tag-2 soundhole facets on imported mesh")
    min_dist = diag.get("min_distance_any_air_pressure_dof_to_center_m")
    radius = float(sh.get("radius_m") or 0.0)
    if min_dist is not None and radius > 0 and float(min_dist) > 2.0 * radius:
        causes.append("1_wrong_soundhole_centre_or_axis: nearest air DOF far from expected aperture")
    if correct_1r > 0:
        causes.append("5_W_to_active_mapping_or_bc_elimination: interior DOFs exist but prior mask failed W/active map")
    if not causes:
        causes.append("7_probe_too_small_or_bc_eliminated: check facet-adjacent + nearfield slab expansion")
    return causes


def audit_sample(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    build_mask: bool,
    load_dolfinx: bool,
) -> Dict[str, Any]:
    val_root = validation_run_root(repo_root, sample_id)
    mesh = val_root / "lprod" / "mesh" / "L_prod" / f"{sample_id}.msh"
    ckpt = val_root / "lprod" / "checkpoint" / "built_metadata.json"
    core_cfg = val_root / "lprod" / "resolved_core_config.json"
    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "validation_run_id": validation_run_id(sample_id),
        "validation_run_root": str(val_root),
        "mesh_present": mesh.is_file(),
        "checkpoint_present": ckpt.is_file(),
    }
    if not mesh.is_file() or not ckpt.is_file():
        out["status"] = "missing_validation_tree"
        return out

    idx = None
    try:
        from v2_b3_m4_lhs_pool_bridge import lhs_entry_index  # noqa: WPS433

        idx = lhs_entry_index(pool, sample_id)
    except Exception:
        pass
    entry = (pool.get("entries") or [])[idx] if idx is not None else {}
    geom = extract_geometry_dict(entry)
    built = load_json(ckpt)

    if not load_dolfinx:
        out["status"] = "dolfinx_required_for_pressure_layout"
        out["note"] = "Run on VM with --dolfinx"
        return out

    diag = diagnose_aperture_pressure_mask(
        mesh,
        geometry=geom,
        built_meta=built,
        core_config_path=core_cfg if core_cfg.is_file() else None,
    )
    diag["inferred_root_causes"] = _infer_root_causes(diag)
    out["diagnostic"] = diag
    out["text_report"] = _format_report(sample_id, diag, None)

    if build_mask:
        mask = build_aperture_pressure_mask(
            mesh,
            geometry=geom,
            built_meta=built,
            core_config_path=core_cfg if core_cfg.is_file() else None,
        )
        out["mask"] = {
            k: v
            for k, v in mask.items()
            if k not in ("p_idx_aperture", "selected_coordinates", "p_active_indices")
        }
        out["p_idx_aperture"] = mask.get("p_idx_aperture", []).tolist() if hasattr(mask.get("p_idx_aperture"), "tolist") else list(mask.get("p_idx_aperture") or [])
        out["p_active_indices"] = list(mask.get("p_active_indices") or [])
        out["text_report"] = _format_report(sample_id, diag, mask)
        out["status"] = "ok" if int(mask.get("n_p_aperture_dofs") or 0) > 0 else "empty_mask"
    else:
        out["status"] = "diagnostic_only"

    val_dir = val_root / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "aperture_mask_diagnostic.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose aperture pressure mask (Task 1).")
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--build-mask", action="store_true", help="Also rebuild non-empty mask after diagnostics.")
    parser.add_argument("--dolfinx", action="store_true", default=True)
    parser.add_argument("--no-dolfinx", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    samples = [s.strip() for s in str(args.samples).split(",") if s.strip()]
    load_dolfinx = bool(args.dolfinx) and not bool(args.no_dolfinx)

    report = {
        "schema": "aperture_pressure_mask_diagnostic_v1",
        "samples": samples,
        "results": [
            audit_sample(
                repo_root=repo_root,
                pool=pool,
                sample_id=sid,
                build_mask=bool(args.build_mask),
                load_dolfinx=load_dolfinx,
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

    any_fail = any(r.get("status") not in ("ok", "diagnostic_only") for r in report["results"])
    return 2 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
