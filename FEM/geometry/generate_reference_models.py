#!/usr/bin/env python3
"""
Write STEP reference bodies into FEM/geometry/models/.

Classical / dreadnought silhouettes use fixed perimeter point templates (no polar math).
Tail end is a 3-point B-spline arc (rounded), not a flat vertical cut.

    python3 FEM/geometry/generate_reference_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import gmsh

MODELS_DIR = Path(__file__).resolve().parent / "models"

Point2 = Tuple[float, float]

# Nominal length L = 0.50 m: neck x = +0.25, tail x = −0.25.  All widths are half-widths (m).

# --- Classical (refined): narrow waist 0.11, lower bout 0.17, moderate shoulders ---
CLASSICAL_NECK_TOP: Point2 = (0.25, 0.05)
CLASSICAL_NECK_BOTTOM: Point2 = (0.25, -0.05)
CLASSICAL_WAIST: Point2 = (0.0, 0.11)
CLASSICAL_LOWER_BOUT: float = 0.17
CLASSICAL_TAIL_ARC: Tuple[Point2, Point2, Point2] = (
    (-0.25, CLASSICAL_LOWER_BOUT),
    (-0.265, 0.0),
    (-0.25, -CLASSICAL_LOWER_BOUT),
)

CLASSICAL_UPPER_BODY: Tuple[Point2, ...] = (
    (0.23, 0.082),
    (0.20, 0.108),
    (0.17, 0.128),
    (0.14, 0.134),  # shoulder (refined, narrower)
    (0.11, 0.132),
    (0.08, 0.124),
    (0.05, 0.118),
    (0.02, 0.114),
    CLASSICAL_WAIST,
    (-0.02, 0.116),
    (-0.05, 0.124),
    (-0.08, 0.136),
    (-0.11, 0.150),
    (-0.14, 0.162),
    (-0.17, 0.168),
    (-0.20, 0.170),
    (-0.23, 0.169),
)

# --- Acoustic / dreadnought (beefy): waist 0.13, lower bout 0.20, wide shoulders ---
ACOUSTIC_NECK_TOP: Point2 = (0.25, 0.052)
ACOUSTIC_NECK_BOTTOM: Point2 = (0.25, -0.052)
ACOUSTIC_WAIST: Point2 = (0.0, 0.13)
ACOUSTIC_LOWER_BOUT: float = 0.20
ACOUSTIC_TAIL_ARC: Tuple[Point2, Point2, Point2] = (
    (-0.25, ACOUSTIC_LOWER_BOUT),
    (-0.270, 0.0),
    (-0.25, -ACOUSTIC_LOWER_BOUT),
)

ACOUSTIC_UPPER_BODY: Tuple[Point2, ...] = (
    (0.23, 0.102),
    (0.20, 0.132),
    (0.17, 0.152),
    (0.14, 0.158),  # shoulder (significantly wider than classical)
    (0.11, 0.156),
    (0.08, 0.148),
    (0.05, 0.140),
    (0.02, 0.134),
    ACOUSTIC_WAIST,
    (-0.02, 0.136),
    (-0.05, 0.146),
    (-0.08, 0.160),
    (-0.11, 0.178),
    (-0.14, 0.192),
    (-0.17, 0.198),
    (-0.20, 0.199),
    (-0.23, 0.199),
)


def classical_guitar_perimeter(length: float = 0.50) -> List[Point2]:
    """Closed perimeter: flat neck, +y body, rounded 3-point tail arc, −y body."""
    return _closed_template_perimeter(
        length,
        neck_top=CLASSICAL_NECK_TOP,
        neck_bottom=CLASSICAL_NECK_BOTTOM,
        upper_body=CLASSICAL_UPPER_BODY,
        tail_arc=CLASSICAL_TAIL_ARC,
    )


def dreadnought_guitar_perimeter(length: float = 0.50) -> List[Point2]:
    return _closed_template_perimeter(
        length,
        neck_top=ACOUSTIC_NECK_TOP,
        neck_bottom=ACOUSTIC_NECK_BOTTOM,
        upper_body=ACOUSTIC_UPPER_BODY,
        tail_arc=ACOUSTIC_TAIL_ARC,
    )


def _closed_template_perimeter(
    length: float,
    *,
    neck_top: Point2,
    neck_bottom: Point2,
    upper_body: Sequence[Point2],
    tail_arc: Tuple[Point2, Point2, Point2],
) -> List[Point2]:
    scale = float(length) / 0.50

    def _s(pt: Point2) -> Point2:
        return (float(pt[0]) * scale, float(pt[1]) * scale)

    upper = [_s(neck_top)] + [_s(p) for p in upper_body]
    tail = [_s(p) for p in tail_arc]
    lower_return = [
        (sx, -sy) for x, y in reversed(upper_body) for sx, sy in [_s((x, y))]
    ]
    return upper + tail + lower_return + [_s(neck_bottom)]


def _volume_tags_from_extrude(out) -> List[int]:
    tags: List[int] = []

    def _walk(obj) -> None:
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2:
                try:
                    if int(obj[0]) == 3:
                        tags.append(int(obj[1]))
                        return
                except (TypeError, ValueError):
                    pass
            for child in obj:
                _walk(child)

    _walk(out)
    return tags


def _build_closed_bspline_solid(
    occ,
    perimeter: Sequence[Point2],
    *,
    depth: float,
    n_upper_body: int,
) -> None:
    """
    Single closed wire from the full perimeter (no multi-segment B-spline stitch).

    ``[neck_top, *upper_body, tail_arc×3, *lower_mirror, neck_bottom, neck_top]``
    """
    ring = list(perimeter)
    expected = 2 + 2 * int(n_upper_body) + 3
    if len(ring) != expected:
        raise RuntimeError(
            f"Perimeter length {len(ring)} != expected {expected} for template."
        )

    neck = (float(ring[0][0]), float(ring[0][1]))
    if ring[-1] != ring[0]:
        ring.append(ring[0])
    ring[-1] = ring[0]
    verts = ring[:-1]

    lc = max(
        max(abs(p[0]) for p in verts),
        max(abs(p[1]) for p in verts),
        float(depth),
    ) / 60.0

    p_tags = [int(occ.addPoint(float(x), float(y), 0.0, lc)) for x, y in verts]
    n_pts = len(p_tags)

    loop_tag: int
    add_polygon = getattr(occ, "addPolygon", None)
    if add_polygon is not None:
        try:
            loop_tag = int(add_polygon(p_tags))
            occ.synchronize()
            print(f"[diag] profile loop via addPolygon ({n_pts} pts)")
        except Exception:
            loop_tag = 0
    else:
        loop_tag = 0

    if not loop_tag:
        for i in range(n_pts):
            occ.addLine(p_tags[i], p_tags[(i + 1) % n_pts])
        occ.synchronize()
        try:
            occ.removeAllDuplicates()
        except Exception:
            pass
        occ.synchronize()
        curve_tags = [int(t) for dim, t in occ.getEntities(1) if int(dim) == 1]
        if len(curve_tags) < 3:
            raise RuntimeError(
                f"OCC found {len(curve_tags)} curves after perimeter lines (need >= 3)."
            )
        loop_tag = int(occ.addCurveLoop(curve_tags))
        print(
            f"[diag] profile loop via {len(curve_tags)} line segments "
            f"(ring_pts={n_pts}, closure={neck})"
        )

    surf_tag = int(occ.addPlaneSurface([loop_tag]))
    try:
        ext = occ.extrude([(2, surf_tag)], 0, 0, float(depth), [3])
    except (TypeError, ValueError):
        ext = occ.extrude([(2, surf_tag)], 0, 0, float(depth))
    vol_tags = _volume_tags_from_extrude(ext)
    if not vol_tags:
        raise RuntimeError("Extrude produced no volume.")
    occ.translate([(3, vol_tags[0])], 0, 0, -float(depth) / 2.0)


def _write_step(name: str, builder) -> None:
    gmsh.model.add(name)
    builder(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    out = MODELS_DIR / name
    gmsh.write(str(out))
    print(f"Wrote {out}")
    gmsh.model.remove()


def _classic(occ) -> None:
    ring = classical_guitar_perimeter(0.50)
    print(
        f"[diag] classical: n={len(ring)} waist={CLASSICAL_WAIST[1]} "
        f"lower={CLASSICAL_LOWER_BOUT} shoulder≈{CLASSICAL_UPPER_BODY[3][1]}"
    )
    _build_closed_bspline_solid(
        occ, ring, depth=0.10, n_upper_body=len(CLASSICAL_UPPER_BODY)
    )


def _acoustic(occ) -> None:
    ring = dreadnought_guitar_perimeter(0.50)
    print(
        f"[diag] acoustic: n={len(ring)} waist={ACOUSTIC_WAIST[1]} "
        f"lower={ACOUSTIC_LOWER_BOUT} shoulder≈{ACOUSTIC_UPPER_BODY[3][1]}"
    )
    _build_closed_bspline_solid(
        occ, ring, depth=0.10, n_upper_body=len(ACOUSTIC_UPPER_BODY)
    )


def _box(occ) -> None:
    lx, ly, lz = 0.48, 0.37, 0.10
    occ.addBox(-lx / 2.0, -ly / 2.0, -lz / 2.0, lx, ly, lz)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gmsh.initialize([sys.argv[0], "-nopopup"])
    gmsh.option.setNumber("Geometry.Tolerance", 1.0e-4)
    try:
        _write_step("classic.step", _classic)
        _write_step("acoustic.step", _acoustic)
        _write_step("box.step", _box)
    finally:
        gmsh.finalize()
    print(f"Reference models ready in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
