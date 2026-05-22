#!/usr/bin/env python3
"""
Write STEP reference bodies into FEM/geometry/models/.

Classical / dreadnought silhouettes use **fixed perimeter point templates** (no polar
or Gaussian envelope math). Box stays rectangular.

    python3 FEM/geometry/generate_reference_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import gmsh

MODELS_DIR = Path(__file__).resolve().parent / "models"

Point2 = Tuple[float, float]

# ---------------------------------------------------------------------------
# Classical guitar template (L = 0.50 m nominal, neck at x = +0.25, tail at −0.25)
# Frame: +x neck, −x tail, ±y lateral. Flat neck & flat tail = vertical lines at ±L/2.
# ---------------------------------------------------------------------------

CLASSICAL_NECK_TOP: Point2 = (0.25, 0.05)
CLASSICAL_NECK_BOTTOM: Point2 = (0.25, -0.05)
CLASSICAL_TAIL_TOP: Point2 = (-0.25, 0.18)
CLASSICAL_TAIL_BOTTOM: Point2 = (-0.25, -0.18)
CLASSICAL_WAIST: Point2 = (0.0, 0.12)

# +y stations neck → tail (interior only; neck/tail flats are separate line segments).
CLASSICAL_UPPER_BODY: Tuple[Point2, ...] = (
    (0.23, 0.085),
    (0.20, 0.112),
    (0.17, 0.132),
    (0.14, 0.138),  # flat shoulder
    (0.11, 0.136),
    (0.08, 0.128),
    (0.05, 0.123),
    (0.02, 0.121),
    CLASSICAL_WAIST,
    (-0.02, 0.123),
    (-0.05, 0.131),
    (-0.08, 0.143),
    (-0.11, 0.158),
    (-0.14, 0.170),
    (-0.17, 0.176),
    (-0.20, 0.178),
    (-0.23, 0.179),
)

# Dreadnought: broader shoulders & lower bout, gentler waist (L = 0.50 m).
DREADNOUGHT_NECK_TOP: Point2 = (0.25, 0.052)
DREADNOUGHT_NECK_BOTTOM: Point2 = (0.25, -0.052)
DREADNOUGHT_TAIL_TOP: Point2 = (-0.25, 0.195)
DREADNOUGHT_TAIL_BOTTOM: Point2 = (-0.25, -0.195)
DREADNOUGHT_WAIST: Point2 = (0.0, 0.125)

DREADNOUGHT_UPPER_BODY: Tuple[Point2, ...] = (
    (0.23, 0.098),
    (0.20, 0.128),
    (0.17, 0.148),
    (0.14, 0.152),
    (0.11, 0.150),
    (0.08, 0.144),
    (0.05, 0.138),
    (0.02, 0.132),
    DREADNOUGHT_WAIST,
    (-0.02, 0.134),
    (-0.05, 0.142),
    (-0.08, 0.155),
    (-0.11, 0.172),
    (-0.14, 0.186),
    (-0.17, 0.192),
    (-0.20, 0.194),
    (-0.23, 0.195),
)


def classical_guitar_perimeter(length: float = 0.50) -> List[Point2]:
    """
    Closed perimeter (CCW): flat neck line → upper B-spline chain → flat tail → lower chain.

    Points are authored for ``length=0.5`` m and scaled uniformly when ``length`` differs.
    """
    return _closed_template_perimeter(
        length,
        neck_top=CLASSICAL_NECK_TOP,
        neck_bottom=CLASSICAL_NECK_BOTTOM,
        tail_top=CLASSICAL_TAIL_TOP,
        tail_bottom=CLASSICAL_TAIL_BOTTOM,
        upper_body=CLASSICAL_UPPER_BODY,
    )


def dreadnought_guitar_perimeter(length: float = 0.50) -> List[Point2]:
    """Dreadnought template (same perimeter walk as classical)."""
    return _closed_template_perimeter(
        length,
        neck_top=DREADNOUGHT_NECK_TOP,
        neck_bottom=DREADNOUGHT_NECK_BOTTOM,
        tail_top=DREADNOUGHT_TAIL_TOP,
        tail_bottom=DREADNOUGHT_TAIL_BOTTOM,
        upper_body=DREADNOUGHT_UPPER_BODY,
    )


def _closed_template_perimeter(
    length: float,
    *,
    neck_top: Point2,
    neck_bottom: Point2,
    tail_top: Point2,
    tail_bottom: Point2,
    upper_body: Sequence[Point2],
) -> List[Point2]:
    scale = float(length) / 0.50

    def _s(pt: Point2) -> Point2:
        return (float(pt[0]) * scale, float(pt[1]) * scale)

    upper = [_s(neck_top)] + [_s(p) for p in upper_body] + [_s(tail_top)]
    lower_return = [_s(tail_bottom)] + [
        (sx, -sy) for x, y in reversed(upper_body) for sx, sy in [_s((x, y))]
    ] + [_s(neck_bottom)]
    return upper + lower_return


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
    Extrude a closed profile to a solid.

    Perimeter layout:
    ``[neck_top, *upper_body, tail_top, tail_bottom, *lower_mirror, neck_bottom]``

    Flat neck & tail = ``addLine``; bouts = ``addBSpline`` (no polar / atan2 math).
    """
    ring = list(perimeter)
    n = len(ring)
    expected = 2 * int(n_upper_body) + 4
    if n != expected:
        raise RuntimeError(f"Perimeter length {n} != expected {expected} for template.")

    lc = max(
        max(abs(p[0]) for p in ring),
        max(abs(p[1]) for p in ring),
        float(depth),
    ) / 60.0

    p_tags = [int(occ.addPoint(float(x), float(y), 0.0, lc)) for x, y in ring]

    i_neck_top = 0
    i_tail_top = 1 + int(n_upper_body)
    i_tail_bot = i_tail_top + 1
    i_neck_bot = n - 1

    c_neck = int(occ.addLine(p_tags[i_neck_bot], p_tags[i_neck_top]))
    c_upper = int(occ.addBSpline(p_tags[i_neck_top : i_tail_top + 1]))
    c_tail = int(occ.addLine(p_tags[i_tail_top], p_tags[i_tail_bot]))
    c_lower = int(occ.addBSpline(p_tags[i_tail_bot : i_neck_bot + 1]))
    loop_tag = int(occ.addCurveLoop([c_neck, c_upper, c_tail, c_lower]))

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
    print(f"[diag] classical template: n_perimeter={len(ring)}")
    _build_closed_bspline_solid(
        occ, ring, depth=0.10, n_upper_body=len(CLASSICAL_UPPER_BODY)
    )


def _acoustic(occ) -> None:
    ring = dreadnought_guitar_perimeter(0.50)
    print(f"[diag] dreadnought template: n_perimeter={len(ring)}")
    _build_closed_bspline_solid(
        occ, ring, depth=0.10, n_upper_body=len(DREADNOUGHT_UPPER_BODY)
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
