#!/usr/bin/env python3
"""Print validation mesh size + soundhole-air gate metrics (post audit)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh"
AUDIT_JSON = (
    EXPERIMENT_ROOT
    / "physics_integrity"
    / "diagnostics"
    / "soundhole_air_audit"
    / "soundhole_air_adjacency_report.json"
)


def main() -> int:
    if not MESH_PATH.is_file():
        print(f"[gate] Mesh missing: {MESH_PATH}", file=sys.stderr)
        return 1
    if not AUDIT_JSON.is_file():
        print(f"[gate] Audit JSON missing: {AUDIT_JSON}", file=sys.stderr)
        return 1

    import meshio

    m = meshio.read(str(MESH_PATH))
    n_nodes = int(len(m.points))
    n_tets = sum(int(len(b.data)) for b in m.cells if b.type == "tetra")

    summary = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    fc = summary["facet_counts"]
    dof = summary["pressure_dof_audit"]

    print("[validation_soundhole_gate]")
    print(f"  n_nodes={n_nodes} n_tetrahedra={n_tets}")
    print(f"  tag2_total={fc['tag2_total']}")
    print(f"  tag2_adjacent_to_air_10={fc['tag2_adjacent_to_air_10']}")
    print(f"  wood_only={fc['tag2_adjacent_wood_only']}")
    print(
        f"  soundhole_p_in_air_subgraph={dof['n_p_soundhole_in_air_subgraph']}"
    )
    ok = (
        fc["tag2_adjacent_to_air_10"] > 0
        and dof["n_p_soundhole_in_air_subgraph"] > 0
    )
    print(f"  gate_pass={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
