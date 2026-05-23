#!/usr/bin/env python3
"""
Write STEP reference bodies into FEM/geometry/models/.

Fixed Torres / dreadnought point templates (half-widths in metres, L = 0.50 m).
Five-point tail arcs for a smooth semi-circular endpin region.

    python3 FEM/geometry/generate_reference_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import gmsh

MODELS_DIR = Path(__file__).resolve().parent / "models"

Point2 = Tuple[float, float]

# --- Classical (Torres): pinched waist 0.09 half, lower 0.17, deep tail bulge x = -0.28 ---
CLASSICAL_NECK_TOP: Point2 = (0.25, 0.05)
CLASSICAL_NECK_BOTTOM: Point2 = (0.25, -0.05)
CLASSICAL_WAIST: Point2 = (0.0, 0.09)
CLASSICAL_LOWER_HALF: float = 0.17
CLASSICAL_SHOULDER_HALF: float = 0.134

CLASSICAL_TAIL_ARC: Tuple[Point2, ...] = (
    (-0.25, CLASSICAL_LOWER_HALF),
    (-0.265, CLASSICAL_LOWER_HALF * 0.94),
    (-0.28, 0.0),
    (-0.265, -CLASSICAL_LOWER_HALF * 0.94),
    (-0.25, -CLASSICAL_LOWER_HALF),
)

CLASSICAL_UPPER_BODY: Tuple[Point2, ...] = (
    (0.23, 0.080),
    (0.20, 0.102),
    (0.17, 0.122),
    (0.14, CLASSICAL_SHOULDER_HALF),
    (0.11, 0.128),
    (0.08, 0.118),
    (0.05, 0.108),
    (0.02, 0.098),
    CLASSICAL_WAIST,
    (-0.02, 0.100),
    (-0.05, 0.112),
    (-0.08, 0.128),
    (-0.11, 0.148),
    (-0.14, 0.162),
    (-0.17, 0.168),
    (-0.20, 0.169),
    (-0.23, 0.169),
)

# --- Dreadnought: shallow waist 0.14 half, lower 0.21, broad shoulders 0.17, tail x = -0.27 ---
ACOUSTIC_NECK_TOP: Point2 = (0.25, 0.052)
ACOUSTIC_NECK_BOTTOM: Point2 = (0.25, -0.052)
ACOUSTIC_WAIST: Point2 = (0.0, 0.14)
ACOUSTIC_LOWER_HALF: float = 0.21
ACOUSTIC_SHOULDER_HALF: float = 0.17

ACOUSTIC_TAIL_ARC: Tuple[Point2, ...] = (
    (-0.25, ACOUSTIC_LOWER_HALF),
    (-0.262, ACOUSTIC_LOWER_HALF * 0.95),
    (-0.27, 0.0),
    (-0.262, -ACOUSTIC_LOWER_HALF * 0.95),
    (-0.25, -ACOUSTIC_LOWER_HALF),
)

ACOUSTIC_UPPER_BODY: Tuple[Point2, ...] = (
    (0.23, 0.108),
    (0.20, 0.138),
    (0.17, 0.158),
    (0.14, ACOUSTIC_SHOULDER_HALF),
    (0.11, 0.168),
    (0.08, 0.162),
    (0.05, 0.152),
    (0.02, 0.146),
    ACOUSTIC_WAIST,
    (-0.02, 0.148),
    (-0.05, 0.158),
    (-0.08, 0.172),
    (-0.11, 0.188),
    (-0.14, 0.202),
    (-0.17, 0.208),
    (-0.20, 0.209),
    (-0.23, 0.209),
)


def classical_guitar_perimeter(length: float = 0.50) -> List[Point2]:
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
    tail_arc: Sequence[Point2],
) -> List[Point2]:
    """Scale only length (x); preserve authored half-widths (y) for distinct silhouettes."""
    sx = float(length) / 0.50

    def _s(pt: Point2) -> Point2:
        return (float(pt[0]) * sx, float(pt[1]))

    upper = [_s(neck_top)] + [_s(p) for p in upper_body]
    tail = [_s(p) for p in tail_arc]
    lower_return = [
        (px, -py) for x, y in reversed(upper_body) for px, py in [_s((x, y))]
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
    n_tail_arc: int,
) -> None:
    """Single closed polyline: neck → body → 5-point tail → return → neck."""
    ring = list(perimeter)
    expected = 2 + 2 * int(n_upper_body) + int(n_tail_arc)
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
            f"[diag] profile loop: {len(curve_tags)} segments, "
            f"{n_pts} pts, closure={neck}"
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
        f"lower={CLASSICAL_LOWER_HALF} shoulder={CLASSICAL_SHOULDER_HALF} "
        f"tail_x={CLASSICAL_TAIL_ARC[2][0]}"
    )
    _build_closed_bspline_solid(
        occ,
        ring,
        depth=0.10,
        n_upper_body=len(CLASSICAL_UPPER_BODY),
        n_tail_arc=len(CLASSICAL_TAIL_ARC),
    )


def _acoustic(occ) -> None:
    ring = dreadnought_guitar_perimeter(0.50)
    print(
        f"[diag] acoustic: n={len(ring)} waist={ACOUSTIC_WAIST[1]} "
        f"lower={ACOUSTIC_LOWER_HALF} shoulder={ACOUSTIC_SHOULDER_HALF} "
        f"tail_x={ACOUSTIC_TAIL_ARC[2][0]}"
    )
    _build_closed_bspline_solid(
        occ,
        ring,
        depth=0.10,
        n_upper_body=len(ACOUSTIC_UPPER_BODY),
        n_tail_arc=len(ACOUSTIC_TAIL_ARC),
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
