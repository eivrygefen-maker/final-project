#!/usr/bin/env python3
"""
Write ``FEM/configs/guitar_3d.json`` for a fixed classical Torres FOM case.

Design Studio reference (7-parameter ROM basis):
  Classical, L=0.48 m, W=0.325 m, D=0.10 m, top thickness=3 mm,
  soundhole radius=47 mm, spruce top, rosewood back/sides.

After this script, run the full coupled FOM pipeline::

    python run_fom_case.py
    python FEM/scripts/mesh_sync.py --config FEM/configs/guitar_3d.json
    mpiexec -n 1 python FEM/scripts/fem_main_3d.py

Output: ``FEM/outputs/fem_3d_output.json``
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
FEM_GEOMETRY = REPO_ROOT / "FEM" / "geometry"
BASE_TEMPLATE = REPO_ROOT / "FEM" / "SORTING" / "pipeline_merged_configs" / "sample_001.json"
OUTPUT_CONFIG = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"
SOUNDHOLE_FROM_NECK_RATIO = 0.5

# Fixed case matching the Design Studio screenshot / poster demo.
FOM_CASE: Dict[str, Any] = {
    "geometry.shape_type": "Classical",
    "geometry.length": 0.48,
    "geometry.width": 0.325,
    "geometry.depth": 0.10,
    "geometry.top_thickness": 0.003,
    "geometry.hole_radius": 0.047,
    "geometry.mesh_mode": "fom",
    "materials.top.wood_id": "spruce",
    "materials.back.wood_id": "rosewood",
}


def _ensure_import_path() -> None:
    for folder in (FEM_SCRIPTS, FEM_GEOMETRY):
        path = str(folder.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _finalize_geometry_bouts(geom: Dict[str, Any]) -> None:
    """Match Design Studio / ``build_geometry_state`` bout scaling for Gmsh morphing."""
    from generate_reference_models import get_luthier_gui_defaults  # noqa: WPS433

    shape = str(geom.get("shape_type", "Classical"))
    defs = get_luthier_gui_defaults(shape)
    w = float(geom["width"])
    ref_w = float(defs["width"])
    w_scale = w / ref_w if ref_w > 0.0 else 1.0
    length = float(geom["length"])
    geom["lower_bout"] = w
    geom["upper_bout"] = float(defs["upper_bout"]) * w_scale
    geom["waist"] = float(defs["waist"]) * w_scale
    geom["soundhole_x"] = 0.5 * length - SOUNDHOLE_FROM_NECK_RATIO * length
    geom["soundhole_y"] = 0.0
    geom["soundhole_from_neck_ratio"] = SOUNDHOLE_FROM_NECK_RATIO
    geom.setdefault("bridge_x", float(defs["bridge_x"]))


def build_guitar_config() -> Dict[str, Any]:
    """Merge the fixed LHS parameters into the production solver template."""
    if not BASE_TEMPLATE.is_file():
        raise FileNotFoundError(f"FOM base template not found: {BASE_TEMPLATE}")

    _ensure_import_path()
    from wood_library import apply_lhs_parameters_to_config, apply_wood_ids_to_config  # noqa: WPS433

    cfg = copy.deepcopy(json.loads(BASE_TEMPLATE.read_text(encoding="utf-8")))
    apply_lhs_parameters_to_config(cfg, dict(FOM_CASE))
    apply_wood_ids_to_config(cfg, top_wood_id="spruce", back_wood_id="rosewood")
    geom = cfg.get("geometry")
    if isinstance(geom, dict):
        _finalize_geometry_bouts(geom)
        geom["mesh_mode"] = "fom"

    solver = cfg.setdefault("solver", {})
    solver["mesh_file"] = "FEM/mesh/guitar_3d.msh"
    solver["couple_fluid"] = True
    solver["structural_only_diagnosis"] = False
    solver.setdefault("num_modes", 50)

    return cfg


def write_guitar_config(path: Path | None = None) -> Path:
    """Write merged config JSON and return the output path."""
    out = (path or OUTPUT_CONFIG).resolve()
    cfg = build_guitar_config()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return out


def main() -> int:
    try:
        out = write_guitar_config()
    except Exception as exc:
        print(f"[run_fom_case] FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"[run_fom_case] Wrote {out}")
    print("[run_fom_case] Case:")
    for key, val in FOM_CASE.items():
        print(f"  {key} = {val}")
    print()
    print("Next steps:")
    print("  python FEM/scripts/mesh_sync.py --config FEM/configs/guitar_3d.json")
    print("  mpiexec -n 1 python FEM/scripts/fem_main_3d.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
