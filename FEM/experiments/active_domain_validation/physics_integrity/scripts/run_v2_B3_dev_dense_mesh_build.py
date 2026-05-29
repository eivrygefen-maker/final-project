#!/usr/bin/env python3
"""Build isolated L_dev_dense mesh for B3 dev solver validation (does not touch L_mid)."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "configs" / "v2_mesh_convergence_build"

from v2_mesh_convergence_common import CONV_MESH, case_by_id, load_manifest, mesh_path, write_json
from v2_mesh_convergence_mesh import build_level_mesh

CASE_ID = "baseline_coupled_v2"
LEVEL_ID = "L_dev_dense"
REQUIRED_VOLUME_TAGS = [1, 2, 3, 10]
REQUIRED_FACET_TAGS = [1, 2, 3, 4, 5]


def _tag_count(tag_map: dict, tag: int) -> int:
    return int(tag_map.get(str(tag), tag_map.get(tag, 0)) or 0)


def main() -> int:
    manifest = load_manifest()
    case = case_by_id(manifest, CASE_ID)
    level_def = (manifest.get("mesh_levels") or {}).get(LEVEL_ID)
    if not level_def:
        print(f"[B3_DEV_mesh] missing mesh_levels.{LEVEL_ID} in manifest", file=sys.stderr)
        return 2

    audit = build_level_mesh(case, LEVEL_ID, level_def, config_dir=CONFIG_DIR)
    out_msh = mesh_path(LEVEL_ID, CASE_ID)
    if audit.get("build_failed"):
        print(f"[B3_DEV_mesh] build_failed exit={audit.get('build_exit_code')} log={audit.get('log')}", flush=True)
        return 2

    vol = audit.get("volume_tag_counts") or {}
    tri = audit.get("triangle_tag_counts") or {}
    vol_ok = all(_tag_count(vol, t) > 0 for t in REQUIRED_VOLUME_TAGS)
    facet_ok = all(_tag_count(tri, t) > 0 for t in REQUIRED_FACET_TAGS)
    lc_scale = float(level_def.get("lc_scale", audit.get("lc_scale") or 1.2))

    summary = {
        "B3_DEV_mesh_variant": LEVEL_ID,
        "B3_DEV_mesh_is_solver_smoke_test_only": True,
        "B3_DEV_mesh_not_authorized_for_final_physics_validation": True,
        "B3_DEV_mesh_path": str(out_msh.resolve()),
        "B3_DEV_mesh_lc_scale": lc_scale,
        "B3_DEV_mesh_node_count": audit.get("n_nodes"),
        "B3_DEV_mesh_element_count": audit.get("n_tetrahedra"),
        "volume_tags_ok": bool(vol_ok),
        "facet_tags_ok": bool(facet_ok),
        "soundhole_facets_tag2": _tag_count(tri, 2),
        "volume_tag_counts": vol,
        "triangle_tag_counts": tri,
        "target_active_dimension_min": level_def.get("target_active_dimension_min"),
        "target_active_dimension_max": level_def.get("target_active_dimension_max"),
        "effective_controls_m": audit.get("effective_controls_m"),
    }
    write_json(CONV_MESH / LEVEL_ID / "baseline_coupled_v2_mesh_build_summary.json", summary)

    print(f"[B3_DEV_mesh] mesh_variant={LEVEL_ID}", flush=True)
    print(f"[B3_DEV_mesh] mesh_path={out_msh.resolve()}", flush=True)
    print(f"[B3_DEV_mesh] B3_DEV_mesh_lc_scale={lc_scale}", flush=True)
    print(f"[B3_DEV_mesh] n_nodes={audit.get('n_nodes')}", flush=True)
    print(f"[B3_DEV_mesh] n_tetrahedra={audit.get('n_tetrahedra')}", flush=True)
    print(f"[B3_DEV_mesh] volume_tags_ok={vol_ok} facet_tags_ok={facet_ok}", flush=True)
    print(f"[B3_DEV_mesh] soundhole_facets_tag2={_tag_count(tri, 2)}", flush=True)
    return 0 if (vol_ok and facet_ok and out_msh.is_file()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
