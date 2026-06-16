#!/usr/bin/env python3
"""Build sample-specific L_scout_coarse FOM mesh for M4 scout (geometry from run tree)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = SCRIPT_DIR / "configs" / "v2_mesh_convergence_build"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lprod_interfaces import extract_geometry_dict, extract_run_metadata  # noqa: E402
from v2_mesh_convergence_common import load_manifest, mesh_path, write_json  # noqa: E402
from v2_mesh_convergence_mesh import build_level_mesh, effective_controls_from_level_def  # noqa: E402

LEVEL_ID = "L_scout_coarse"
REQUIRED_VOLUME_TAGS = (1, 2, 3, 10)
REQUIRED_FACET_TAGS = (1, 2, 3, 4, 5)


def _tag_count(tag_map: dict, tag: int) -> int:
    return int(tag_map.get(str(tag), tag_map.get(tag, 0)) or 0)


def _load_sample_input(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "sample" / "sample_input.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def build_scout_mesh_for_case(
    *,
    sample_id: str,
    geometry: Dict[str, float],
    shape_name: str,
    geometry_shape_type: Optional[str] = None,
    gmsh_shape_type: Optional[str] = None,
    lhs_path: str = "",
) -> Dict[str, Any]:
    manifest = load_manifest()
    level_def = (manifest.get("mesh_levels") or {}).get(LEVEL_ID)
    if not level_def:
        raise RuntimeError(f"missing mesh_levels.{LEVEL_ID} in v2_mesh_convergence_manifest.json")

    repo_root = Path(__file__).resolve().parents[5]
    fem_scripts = repo_root / "FEM" / "scripts"
    if str(fem_scripts) not in sys.path:
        sys.path.insert(0, str(fem_scripts))
    from m4_shape_registry import resolve_geometry_shape_type, resolve_shape_config  # noqa: WPS433

    cfg = resolve_shape_config(shape_name)
    geom_shape_type = str(
        geometry_shape_type
        or resolve_geometry_shape_type(parameters={"geometry.shape_type": cfg.geometry_shape_type})
        or cfg.geometry_shape_type
    )
    gmsh_type = str(gmsh_shape_type or cfg.gmsh_shape_type)

    case = {
        "id": sample_id,
        "geometry": dict(geometry),
        "shape_name": shape_name,
        "geometry_shape_type": geom_shape_type,
        "gmsh_shape_type": gmsh_type,
        "lhs_path": lhs_path,
    }
    audit = build_level_mesh(case, LEVEL_ID, level_def, config_dir=CONFIG_DIR)
    out_msh = mesh_path(LEVEL_ID, sample_id)

    if audit.get("build_failed"):
        return {
            "status": "FAIL",
            "mesh_path": str(out_msh.resolve()),
            "build_failed": True,
            "audit": audit,
        }

    vol = audit.get("volume_tag_counts") or {}
    tri = audit.get("triangle_tag_counts") or {}
    vol_ok = all(_tag_count(vol, t) > 0 for t in REQUIRED_VOLUME_TAGS)
    facet_ok = all(_tag_count(tri, t) > 0 for t in REQUIRED_FACET_TAGS)
    effective = effective_controls_from_level_def(level_def)

    summary = {
        "B3_scout_mesh_level": LEVEL_ID,
        "sample_id": sample_id,
        "mesh_path": str(out_msh.resolve()),
        "effective_controls_m": effective,
        "n_nodes": audit.get("n_nodes"),
        "n_tetrahedra": audit.get("n_tetrahedra"),
        "volume_tags_ok": bool(vol_ok),
        "facet_tags_ok": bool(facet_ok),
        "geometry": geometry,
        "shape_name": shape_name,
        "geometry_shape_type": geom_shape_type,
        "gmsh_shape_type": gmsh_type,
        "sample_specific_geometry": True,
    }
    write_json(out_msh.parent / f"{sample_id}_mesh_build_summary.json", summary)

    ok = out_msh.is_file() and vol_ok and facet_ok
    return {
        "status": "PASS" if ok else "FAIL",
        "mesh_path": str(out_msh.resolve()),
        "summary": summary,
        "audit": audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build L_scout_coarse mesh for one M4 sample geometry.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="M4 run tree; reads sample/sample_input.json for geometry/material.",
    )
    parser.add_argument(
        "--geometry-json",
        type=Path,
        help="Optional JSON object with geometry keys (overrides run-dir sample).",
    )
    args = parser.parse_args()
    sample_id = str(args.sample_id)

    geometry: Dict[str, float] = {}
    shape_name = "classic"
    geometry_shape_type: Optional[str] = None
    gmsh_shape_type: Optional[str] = None
    lhs_path = ""
    if args.geometry_json and args.geometry_json.is_file():
        raw = json.loads(args.geometry_json.read_text(encoding="utf-8"))
        geometry = extract_geometry_dict(raw)
        meta = extract_run_metadata(raw)
        shape_name = str(meta.get("shape_name") or raw.get("shape_name") or shape_name)
        geometry_shape_type = str(raw.get("geometry_shape_type") or meta.get("geometry_shape_type") or "")
        gmsh_shape_type = str(raw.get("gmsh_shape_type") or meta.get("gmsh_shape_type") or "")
        lhs_path = str(meta.get("lhs_path") or raw.get("lhs_path") or "")
    elif args.run_dir:
        sample = _load_sample_input(args.run_dir.expanduser().resolve())
        geometry = extract_geometry_dict(sample)
        meta = extract_run_metadata(sample)
        shape_name = str(meta.get("shape_name") or sample.get("shape_name") or shape_name)
        lhs_path = str(meta.get("lhs_path") or sample.get("lhs_path") or "")
        repo_root = Path(__file__).resolve().parents[5]
        fem_scripts = repo_root / "FEM" / "scripts"
        if str(fem_scripts) not in sys.path:
            sys.path.insert(0, str(fem_scripts))
        from m4_shape_registry import resolve_geometry_shape_type, resolve_shape_config  # noqa: WPS433

        geometry_shape_type = resolve_geometry_shape_type(sample_input=sample)
        gmsh_shape_type = resolve_shape_config(shape_name).gmsh_shape_type
    else:
        print("error: provide --run-dir or --geometry-json", file=sys.stderr)
        return 2

    if not geometry:
        print("error: empty geometry for L_scout_coarse mesh build", file=sys.stderr)
        return 2

    result = build_scout_mesh_for_case(
        sample_id=sample_id,
        geometry=geometry,
        shape_name=shape_name,
        geometry_shape_type=geometry_shape_type or None,
        gmsh_shape_type=gmsh_shape_type or None,
        lhs_path=lhs_path,
    )
    print(
        f"[B3_scout_mesh] shape_name={shape_name} geometry.shape_type={geometry_shape_type} "
        f"gmsh_shape_type={gmsh_shape_type or result.get('summary', {}).get('gmsh_shape_type')}",
        flush=True,
    )
    print(f"[B3_scout_mesh] status={result.get('status')}", flush=True)
    print(f"[B3_scout_mesh] mesh_path={result.get('mesh_path')}", flush=True)
    if result.get("summary"):
        s = result["summary"]
        print(f"[B3_scout_mesh] n_nodes={s.get('n_nodes')} n_tetra={s.get('n_tetrahedra')}", flush=True)
    if result.get("build_failed"):
        audit = result.get("audit") or {}
        print(f"[B3_scout_mesh] build_failed exit={audit.get('build_exit_code')}", file=sys.stderr)
        return 2
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
