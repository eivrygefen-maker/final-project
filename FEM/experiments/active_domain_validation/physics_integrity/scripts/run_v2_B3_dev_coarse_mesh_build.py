#!/usr/bin/env python3
"""Build isolated L_dev_coarse mesh for B3 dev solver smoke tests (does not touch L_mid)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
CONFIG_DIR = SCRIPT_DIR.parent / "configs" / "v2_mesh_convergence_build"

from v2_mesh_convergence_common import CONV_MESH, case_by_id, load_manifest, mesh_path, write_json
from v2_mesh_convergence_mesh import build_level_mesh

CASE_ID = "baseline_coupled_v2"
LEVEL_ID = "L_dev_coarse"
REQUIRED_VOLUME_TAGS = [1, 2, 3, 10]
REQUIRED_FACET_TAGS = [1, 2, 3, 4, 5]


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
    def _tag_count(tag_map: dict, tag: int) -> int:
        return int(tag_map.get(str(tag), tag_map.get(tag, 0)) or 0)

    vol_ok = all(_tag_count(vol, t) > 0 for t in REQUIRED_VOLUME_TAGS)
    facet_ok = all(_tag_count(tri, t) > 0 for t in REQUIRED_FACET_TAGS)

    summary = {
        "mesh_variant": LEVEL_ID,
        "mesh_path": str(out_msh.resolve()),
        "n_nodes": audit.get("n_nodes"),
        "n_tetrahedra": audit.get("n_tetrahedra"),
        "volume_tag_counts": vol,
        "triangle_tag_counts": tri,
        "required_volume_tags_present": bool(vol_ok),
        "required_facet_tags_present": bool(facet_ok),
        "lc_scale": audit.get("lc_scale"),
        "effective_controls_m": audit.get("effective_controls_m"),
        "solver_smoke_test_only": True,
        "not_authorized_for_final_physics_validation": True,
    }
    write_json(CONV_MESH / LEVEL_ID / "baseline_coupled_v2_mesh_build_summary.json", summary)

    print(f"[B3_DEV_mesh] mesh_variant={LEVEL_ID}", flush=True)
    print(f"[B3_DEV_mesh] mesh_path={out_msh.resolve()}", flush=True)
    print(f"[B3_DEV_mesh] n_nodes={audit.get('n_nodes')}", flush=True)
    print(f"[B3_DEV_mesh] n_tetrahedra={audit.get('n_tetrahedra')}", flush=True)
    print(f"[B3_DEV_mesh] volume_tags_ok={vol_ok} facet_tags_ok={facet_ok}", flush=True)
    return 0 if (vol_ok and facet_ok and out_msh.is_file()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
