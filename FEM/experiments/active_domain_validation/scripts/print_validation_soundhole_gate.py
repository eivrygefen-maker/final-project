#!/usr/bin/env python3
"""Print validation mesh size + soundhole adjacency + aperture geometry gates."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh"
ADJACENCY_JSON = (
    EXPERIMENT_ROOT
    / "physics_integrity"
    / "diagnostics"
    / "soundhole_air_audit"
    / "soundhole_air_adjacency_report.json"
)
APERTURE_JSON = (
    EXPERIMENT_ROOT
    / "physics_integrity"
    / "diagnostics"
    / "soundhole_aperture_audit"
    / "soundhole_aperture_geometry_report.json"
)
EXPECTED_AREA = math.pi * 0.047 * 0.047


def main() -> int:
    if not MESH_PATH.is_file():
        print(f"[gate] Mesh missing: {MESH_PATH}", file=sys.stderr)
        return 1
    if not ADJACENCY_JSON.is_file():
        print(f"[gate] Adjacency audit missing: {ADJACENCY_JSON}", file=sys.stderr)
        return 1
    if not APERTURE_JSON.is_file():
        print(f"[gate] Aperture audit missing: {APERTURE_JSON}", file=sys.stderr)
        return 1

    import meshio

    m = meshio.read(str(MESH_PATH))
    n_nodes = int(len(m.points))
    n_tets = sum(int(len(b.data)) for b in m.cells if b.type == "tetra")

    adj = json.loads(ADJACENCY_JSON.read_text(encoding="utf-8"))
    fc = adj["facet_counts"]
    dof = adj["pressure_dof_audit"]
    ap = json.loads(APERTURE_JSON.read_text(encoding="utf-8"))

    print("[validation_soundhole_gate]")
    print(f"  n_nodes={n_nodes} n_tetrahedra={n_tets}")
    print(f"  tag2_total={fc['tag2_total']}")
    print(f"  tag2_adjacent_to_air_10={fc['tag2_adjacent_to_air_10']}")
    print(f"  wood_only={fc['tag2_adjacent_wood_only']}")
    print(f"  soundhole_p_in_air_subgraph={dof['n_p_soundhole_in_air_subgraph']}")
    print(f"  tag2_area_m2={ap['tag2_total_area_m2']:.8f} expected={EXPECTED_AREA:.8f}")
    print(f"  area_ratio={ap['area_ratio_vs_pi_r2']:.4f}")
    print(f"  radial_max_m={ap['radial_max_m']:.6f}")
    print(f"  horizontal_area_fraction={ap['horizontal_area_fraction']:.6f}")
    print(f"  aperture_gate_pass={ap['gate_pass']}")

    ok = (
        fc["tag2_adjacent_to_air_10"] > 0
        and fc["tag2_adjacent_wood_only"] == 0
        and dof["n_p_soundhole_in_air_subgraph"] > 0
        and ap["gate_pass"]
    )
    print(f"  combined_gate_pass={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
