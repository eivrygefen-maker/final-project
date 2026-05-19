"""Audit ``FEM/mesh/guitar_3d.msh`` for Gmsh physical tags (facet tags 1, 3, 4 required)."""
from __future__ import annotations

import sys
from pathlib import Path

import meshio
import numpy as np

REQUIRED_FACET_TAGS = (1, 3, 4)


def audit_msh(msh_path: Path) -> dict[str, dict[int, int]]:
    mesh = meshio.read(str(msh_path))
    phys = mesh.cell_data_dict.get("gmsh:physical")
    if not phys:
        raise RuntimeError("No gmsh:physical cell_data in mesh (mesh is physically unlabelled).")

    counts_by_type: dict[str, dict[int, int]] = {}
    if isinstance(phys, dict):
        for cell_type, tags in phys.items():
            arr = np.asarray(tags, dtype=np.int32).ravel()
            if arr.size == 0:
                continue
            uniq, cnt = np.unique(arr, return_counts=True)
            counts_by_type[str(cell_type)] = {int(t): int(c) for t, c in zip(uniq, cnt)}
    else:
        for i, cell_block in enumerate(mesh.cells):
            if i >= len(phys):
                continue
            arr = np.asarray(phys[i], dtype=np.int32).ravel()
            if arr.size == 0:
                continue
            uniq, cnt = np.unique(arr, return_counts=True)
            block_counts = {int(t): int(c) for t, c in zip(uniq, cnt)}
            prev = counts_by_type.get(cell_block.type, {})
            for tag, n in block_counts.items():
                prev[tag] = prev.get(tag, 0) + n
            counts_by_type[cell_block.type] = prev

    return counts_by_type


def main() -> int:
    msh_path = Path("FEM/mesh/guitar_3d.msh")
    if len(sys.argv) > 1:
        msh_path = Path(sys.argv[1])

    print(f"--- Physical tags in {msh_path} ---")
    try:
        by_type = audit_msh(msh_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    if not by_type:
        print("No gmsh:physical tags found in any cell block.")
        return 1

    for cell_type, tag_counts in sorted(by_type.items()):
        print(f"Cell type: {cell_type}")
        for tag in sorted(tag_counts):
            print(f"  Tag {tag}: {tag_counts[tag]} elements")

    tri = by_type.get("triangle", {})
    missing = [t for t in REQUIRED_FACET_TAGS if t not in tri or tri[t] <= 0]
    if missing:
        print(f"FAIL: triangle facets missing required tags: {missing}")
        return 1

    print(f"OK: facet tags {REQUIRED_FACET_TAGS} present on triangles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
