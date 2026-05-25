#!/usr/bin/env python3
"""
Geometric acceptance audit for validation soundhole facet tag 2 (circular aperture).

Experiment-only: no eigen solve, no mesh modification.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]

HOLE_R_M = 0.047
EXPECTED_AREA_M2 = math.pi * HOLE_R_M * HOLE_R_M
AREA_REL_TOL = 0.15
def _radial_max_limit_m(hole_radius_m: float) -> float:
    """Match build_3d_guitar validation aperture radial extent (parameterized by r)."""
    return max(0.050, float(hole_radius_m) * 1.08 + 1.0e-4)
Z_SPAN_MAX_M = 0.012
HORIZONTAL_NZ_MIN = 0.85
HORIZONTAL_FRAC_MIN = 0.90

# Classical validation hole centre (matches build_3d_guitar / adjacency audit).
EXPECTED_LENGTH_M = 0.48
EXPECTED_WIDTH_M = 0.325
EXPECTED_SOUNDHOLE_FROM_NECK = 0.5


def _expected_hole_center_xy() -> Tuple[float, float]:
    L = EXPECTED_LENGTH_M
    hx = 0.5 * L - EXPECTED_SOUNDHOLE_FROM_NECK * L
    return float(hx), 0.0


def _triangle_area_and_normal(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> Tuple[float, np.ndarray]:
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    area = 0.5 * float(np.linalg.norm(cross))
    n = cross / (2.0 * area) if area > 0.0 else np.zeros(3)
    return area, n


def _audit_meshio(mesh_path: Path, *, hole_radius_m: float = HOLE_R_M) -> Dict[str, Any]:
    expected_area = math.pi * float(hole_radius_m) ** 2
    radial_max_limit = _radial_max_limit_m(hole_radius_m)
    import meshio

    hx, hy = _expected_hole_center_xy()
    m = meshio.read(str(mesh_path))
    phys = m.cell_data_dict.get("gmsh:physical", {})
    tri_blocks = [b.data for b in m.cells if b.type == "triangle"]
    if not tri_blocks:
        raise RuntimeError("Mesh has no triangle elements")
    tris = np.vstack(tri_blocks)
    tags = np.asarray(phys.get("triangle", []), dtype=np.int32).ravel()
    if tags.size != tris.shape[0]:
        raise RuntimeError("Triangle physical tags missing or wrong length")

    mask = tags == 2
    if not np.any(mask):
        raise RuntimeError("No facet tag 2 triangles in mesh")

    pts = np.asarray(m.points, dtype=np.float64)
    total_area = 0.0
    horiz_area = 0.0
    r_max = 0.0
    z_vals: List[float] = []
    centroids: List[np.ndarray] = []

    for tri in tris[mask]:
        p0, p1, p2 = pts[int(tri[0])], pts[int(tri[1])], pts[int(tri[2])]
        area, n = _triangle_area_and_normal(p0, p1, p2)
        total_area += area
        if abs(float(n[2])) >= HORIZONTAL_NZ_MIN:
            horiz_area += area
        for p in (p0, p1, p2):
            r_max = max(r_max, float(math.hypot(float(p[0]) - hx, float(p[1]) - hy)))
            z_vals.append(float(p[2]))
        centroids.append((p0 + p1 + p2) / 3.0)

    centroids_arr = np.vstack(centroids)
    z_span = float(max(z_vals) - min(z_vals)) if z_vals else float("inf")
    horiz_frac = float(horiz_area / total_area) if total_area > 0.0 else 0.0
    area_ratio = float(total_area / expected_area) if expected_area > 0 else 0.0

    xmin, xmax = float(centroids_arr[:, 0].min()), float(centroids_arr[:, 0].max())
    ymin, ymax = float(centroids_arr[:, 1].min()), float(centroids_arr[:, 1].max())
    zmin, zmax = float(centroids_arr[:, 2].min()), float(centroids_arr[:, 2].max())

    checks = {
        "area_within_15pct": bool(
            (1.0 - AREA_REL_TOL) * expected_area
            <= total_area
            <= (1.0 + AREA_REL_TOL) * expected_area
        ),
        "radial_max_within_limit": bool(r_max <= radial_max_limit),
        "z_span_planar": bool(z_span <= Z_SPAN_MAX_M),
        "horizontal_fraction_ok": bool(horiz_frac >= HORIZONTAL_FRAC_MIN),
    }
    gate_pass = all(checks.values())

    return {
        "mesh_file": str(mesh_path.resolve()),
        "expected_hole_radius_m": float(hole_radius_m),
        "expected_aperture_area_m2": float(expected_area),
        "radial_max_limit_m": float(radial_max_limit),
        "expected_hole_center_xy_m": [hx, hy],
        "tag2_triangle_count": int(mask.sum()),
        "tag2_total_area_m2": float(total_area),
        "area_ratio_vs_pi_r2": float(area_ratio),
        "radial_max_m": float(r_max),
        "z_extent_m": [zmin, zmax],
        "z_span_m": float(z_span),
        "horizontal_area_fraction": float(horiz_frac),
        "centroid_bbox_m": {
            "x": [xmin, xmax],
            "y": [ymin, ymax],
            "z": [zmin, zmax],
        },
        "acceptance_checks": checks,
        "gate_pass": gate_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Soundhole tag-2 aperture geometry audit")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PHYSICS_ROOT / "diagnostics" / "soundhole_aperture_audit",
    )
    parser.add_argument(
        "--hole-radius",
        type=float,
        default=HOLE_R_M,
        help="Expected soundhole radius (m) for pi*r^2 area gate",
    )
    args = parser.parse_args()
    mesh_path = args.mesh.resolve()
    if not mesh_path.is_file():
        print(f"[aperture_audit] Mesh not found: {mesh_path}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = _audit_meshio(mesh_path, hole_radius_m=float(args.hole_radius))
    except Exception as exc:
        print(f"[aperture_audit] failed: {exc}", file=sys.stderr)
        return 2

    json_path = args.out_dir / "soundhole_aperture_geometry_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[soundhole_aperture_audit] Done.")
    print(f"  tag2_total_area_m2={summary['tag2_total_area_m2']:.8f}")
    print(f"  expected_area_m2={summary['expected_aperture_area_m2']:.8f}")
    print(f"  area_ratio={summary['area_ratio_vs_pi_r2']:.4f}")
    print(f"  radial_max_m={summary['radial_max_m']:.6f}")
    print(f"  z_span_m={summary['z_span_m']:.6f}")
    print(f"  horizontal_area_fraction={summary['horizontal_area_fraction']:.6f}")
    print(f"  gate_pass={summary['gate_pass']}")
    print(f"  report={json_path}")
    if not summary["gate_pass"]:
        failed = [k for k, v in summary["acceptance_checks"].items() if not v]
        print(f"  FAILED checks: {failed}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
