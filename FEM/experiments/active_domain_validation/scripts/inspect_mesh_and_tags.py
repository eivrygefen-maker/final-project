#!/usr/bin/env python3
"""Audit validation mesh size, tags, and checksum (experiment-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_with_meshio(msh_path: Path) -> dict:
    import meshio
    import numpy as np

    m = meshio.read(str(msh_path))
    n_nodes = int(len(m.points))
    n_tets = 0
    tri_counts: dict = {}
    vol_counts: dict = {}
    phys = m.cell_data_dict.get("gmsh:physical", {})
    for block in m.cells:
        if block.type == "tetra":
            n_tets += int(len(block.data))
        if block.type == "triangle" and isinstance(phys, dict):
            arr = np.asarray(phys.get("triangle", []), dtype=np.int32).ravel()
            if arr.size == block.data.shape[0]:
                u, c = np.unique(arr, return_counts=True)
                for t, n in zip(u, c):
                    tri_counts[int(t)] = tri_counts.get(int(t), 0) + int(n)
    if isinstance(phys, dict):
        for key, raw in phys.items():
            if key == "triangle":
                continue
            arr = np.asarray(raw, dtype=np.int32).ravel()
            if arr.size:
                u, c = np.unique(arr, return_counts=True)
                for t, n in zip(u, c):
                    vol_counts[int(t)] = int(n)

    return {
        "mesh_file": str(msh_path.resolve()),
        "sha256": _sha256_file(msh_path),
        "n_nodes": n_nodes,
        "n_tetrahedra": n_tets,
        "triangle_tag_counts": tri_counts,
        "volume_tag_counts": vol_counts,
        "soundhole_facets_tag2": int(tri_counts.get(2, 0)),
        "within_node_target_20k_80k": 20000 <= n_nodes <= 80000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        type=Path,
        default=EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPERIMENT_ROOT / "mesh" / "mesh_audit.json",
    )
    args = parser.parse_args()
    msh = args.mesh.resolve()
    if not msh.is_file():
        print(f"[audit] Mesh not found: {msh}", file=sys.stderr)
        return 1
    try:
        audit = audit_with_meshio(msh)
    except ImportError as exc:
        print(f"[audit] meshio required: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["within_node_target_20k_80k"]:
        print(
            f"[audit][WARN] Node count {audit['n_nodes']} outside 20k–80k target; "
            "coarsen FEM_VALIDATION_MESH profile before long solves.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
