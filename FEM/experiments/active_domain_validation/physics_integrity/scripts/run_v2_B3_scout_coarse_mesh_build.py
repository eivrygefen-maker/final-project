#!/usr/bin/env python3
"""Build isolated L_scout_coarse FOM mesh for B3 modal-density scouting (no L_prod/L_mid touch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "configs" / "v2_mesh_convergence_build"

from v2_mesh_convergence_common import CONV_MESH, case_by_id, load_manifest, mesh_path, write_json
from v2_mesh_convergence_mesh import build_level_mesh, effective_controls_from_level_def

CASE_ID = "baseline_coupled_v2"
LEVEL_ID = "L_scout_coarse"
REQUIRED_VOLUME_TAGS = [1, 2, 3, 10]
REQUIRED_FACET_TAGS = [1, 2, 3, 4, 5]


def main() -> int:
    manifest = load_manifest()
    case = case_by_id(manifest, CASE_ID)
    level_def = (manifest.get("mesh_levels") or {}).get(LEVEL_ID)
    if not level_def:
        print(f"[B3_scout_mesh] missing mesh_levels.{LEVEL_ID} in manifest", file=sys.stderr)
        return 2

    audit = build_level_mesh(case, LEVEL_ID, level_def, config_dir=CONFIG_DIR)
    out_msh = mesh_path(LEVEL_ID, CASE_ID)
    if audit.get("build_failed"):
        print(
            f"[B3_scout_mesh] build_failed exit={audit.get('build_exit_code')} log={audit.get('log')}",
            flush=True,
        )
        return 2

    vol = audit.get("volume_tag_counts") or {}
    tri = audit.get("triangle_tag_counts") or {}

    def _tag_count(tag_map: dict, tag: int) -> int:
        return int(tag_map.get(str(tag), tag_map.get(tag, 0)) or 0)

    vol_ok = all(_tag_count(vol, t) > 0 for t in REQUIRED_VOLUME_TAGS)
    facet_ok = all(_tag_count(tri, t) > 0 for t in REQUIRED_FACET_TAGS)
    effective = effective_controls_from_level_def(level_def)

    summary = {
        "B3_scout_mesh_level": LEVEL_ID,
        "purpose": level_def.get("purpose"),
        "production_physics": level_def.get("production_physics"),
        "final_results": level_def.get("final_results"),
        "modal_density_scout_only": level_def.get("modal_density_scout_only"),
        "not_authorized_for_final_physics_validation": level_def.get(
            "not_authorized_for_final_physics_validation"
        ),
        "mesh_path": str(out_msh.resolve()),
        "lc_scale": float(level_def.get("lc_scale", audit.get("lc_scale") or 1.0)),
        "explicit_controls_m": level_def.get("explicit_controls_m"),
        "effective_controls_m": effective,
        "n_nodes": audit.get("n_nodes"),
        "n_tetrahedra": audit.get("n_tetrahedra"),
        "volume_tags_ok": bool(vol_ok),
        "facet_tags_ok": bool(facet_ok),
        "soundhole_facets_tag2": _tag_count(tri, 2),
        "required_volume_tags_present": bool(vol_ok),
        "required_facet_tags_present": bool(facet_ok),
    }
    write_json(CONV_MESH / LEVEL_ID / "baseline_coupled_v2_mesh_build_summary.json", summary)

    print(f"[B3_scout_mesh] mesh_level={LEVEL_ID}", flush=True)
    print(f"[B3_scout_mesh] mesh_path={out_msh.resolve()}", flush=True)
    print(f"[B3_scout_mesh] effective_controls_m={json.dumps(effective)}", flush=True)
    print(f"[B3_scout_mesh] n_nodes={audit.get('n_nodes')}", flush=True)
    print(f"[B3_scout_mesh] n_tetrahedra={audit.get('n_tetrahedra')}", flush=True)
    print(f"[B3_scout_mesh] volume_tags_ok={vol_ok} facet_tags_ok={facet_ok}", flush=True)
    return 0 if (vol_ok and facet_ok and out_msh.is_file()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
