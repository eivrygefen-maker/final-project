#!/usr/bin/env python3
"""
Write STEP reference bodies into FEM/geometry/models/.

Torres (classical) and Martin D-28 (acoustic) luthier blueprints: closed
``occ.addBSpline`` through explicit control points (C2-smooth outline).

    python3 FEM/geometry/generate_reference_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

MODELS_DIR = Path(__file__).resolve().parent / "models"

Point2 = Tuple[float, float]

# ---------------------------------------------------------------------------
# Luthier blueprints — edit control points only (half-profile + tail tip).
# ---------------------------------------------------------------------------

CLASSICAL_TOP_HALF: Tuple[Point2, ...] = (
    (0.22, 0.045),  # neck junction
    (0.18, 0.110),  # upper shoulder
    (0.12, 0.140),  # upper bout max
    (0.06, 0.130),  # curve to waist
    (0.00, 0.115),  # waist (narrow)
    (-0.06, 0.135),  # curve to lower bout
    (-0.12, 0.175),  # lower bout swell
    (-0.18, 0.185),  # lower bout max
    (-0.24, 0.140),  # tail curve start
    (-0.26, 0.070),  # tail corner restriction
)
CLASSICAL_TAIL_TIP: Point2 = (-0.27, 0.0)

ACOUSTIC_TOP_HALF: Tuple[Point2, ...] = (
    (0.22, 0.050),  # neck junction
    (0.18, 0.135),  # broad shoulder
    (0.12, 0.155),  # upper bout max
    (0.06, 0.148),  # curve to waist
    (0.00, 0.140),  # waist (wide)
    (-0.06, 0.165),  # curve to lower bout
    (-0.12, 0.195),  # lower bout swell
    (-0.19, 0.205),  # lower bout max
    (-0.25, 0.160),  # tail curve start
    (-0.27, 0.080),  # tail corner restriction
)
ACOUSTIC_TAIL_TIP: Point2 = (-0.275, 0.0)

# Nominal span / widths derived from blueprints (build_3d_guitar morphing).
NOMINAL_LENGTH_CLASSICAL: float = 0.49
NOMINAL_LENGTH_ACOUSTIC: float = 0.495

REFERENCE_NOMINAL_WIDTHS: dict[str, Tuple[float, float, float]] = {
    "classical": (0.28, 0.23, 0.37),  # upper, waist, lower (full width)
    "acoustic": (0.31, 0.28, 0.41),
}

# GUI / config defaults when user picks a shape (blueprint frame, +x = neck).
LUTHIER_GUI_DEFAULTS: Dict[str, Dict[str, float]] = {
    "Classical": {
        "length": 0.49,
        "width": 0.37,
        "depth": 0.095,
        "soundhole_x": 0.09,
        "soundhole_radius": 0.042,
        "bridge_x": -0.12,
        "upper_bout": 0.28,
        "waist": 0.23,
        "lower_bout": 0.37,
    },
    "Dreadnought": {
        "length": 0.495,
        "width": 0.41,
        "depth": 0.115,
        "soundhole_x": 0.075,
        "soundhole_radius": 0.050,
        "bridge_x": -0.10,
        "upper_bout": 0.31,
        "waist": 0.28,
        "lower_bout": 0.41,
    },
    "Acoustic": {
        "length": 0.495,
        "width": 0.41,
        "depth": 0.115,
        "soundhole_x": 0.075,
        "soundhole_radius": 0.050,
        "bridge_x": -0.10,
        "upper_bout": 0.31,
        "waist": 0.28,
        "lower_bout": 0.41,
    },
    "Box": {
        "length": 0.48,
        "width": 0.37,
        "depth": 0.10,
        "soundhole_x": 0.08,
        "soundhole_radius": 0.04,
        "bridge_x": -0.12,
        "upper_bout": 0.28,
        "waist": 0.24,
        "lower_bout": 0.37,
    },
}


def get_luthier_gui_defaults(shape_type: str) -> Dict[str, float]:
    """Return blueprint-derived defaults for Streamlit sliders / JSON config."""
    st = str(shape_type).strip()
    if st in LUTHIER_GUI_DEFAULTS:
        return dict(LUTHIER_GUI_DEFAULTS[st])
    if "dread" in st.lower() or "acoustic" in st.lower() or "martin" in st.lower():
        return dict(LUTHIER_GUI_DEFAULTS["Dreadnought"])
    if "box" in st.lower() or "rect" in st.lower():
        return dict(LUTHIER_GUI_DEFAULTS["Box"])
    return dict(LUTHIER_GUI_DEFAULTS["Classical"])


def luthier_closed_loop(
    top_half: Sequence[Point2],
    tail_tip: Point2,
) -> Tuple[Point2, ...]:
    """
    Build a closed CCW control polygon: top half → tail tip → mirrored bottom → neck close.

    The last point duplicates the first so ``addBSpline`` forms a periodic closed curve.
    """
    half = [(float(x), float(y)) for x, y in top_half]
    tip = (float(tail_tip[0]), float(tail_tip[1]))
    bottom = [(float(x), -float(y)) for x, y in reversed(half)]
    loop: List[Point2] = half + [tip] + bottom
    if loop[0] != loop[-1]:
        loop.append(loop[0])
    return tuple(loop)


CLASSICAL_LOOP = luthier_closed_loop(CLASSICAL_TOP_HALF, CLASSICAL_TAIL_TIP)
ACOUSTIC_LOOP = luthier_closed_loop(ACOUSTIC_TOP_HALF, ACOUSTIC_TAIL_TIP)


def _nominal_length_for_loop(loop: Sequence[Point2]) -> float:
    xs = [float(x) for x, _ in loop]
    return max(xs) - min(xs)


NOMINAL_LENGTH: float = _nominal_length_for_loop(CLASSICAL_LOOP)


def _scale_loop(loop: Sequence[Point2], length: float) -> List[Point2]:
    ref = _nominal_length_for_loop(loop)
    sx = float(length) / max(ref, 1.0e-9)
    return [(float(x) * sx, float(y) * sx) for x, y in loop]


def classical_guitar_perimeter(length: float = NOMINAL_LENGTH_CLASSICAL) -> List[Point2]:
    return _scale_loop(CLASSICAL_LOOP, length)


def dreadnought_guitar_perimeter(length: float = NOMINAL_LENGTH_ACOUSTIC) -> List[Point2]:
    return _scale_loop(ACOUSTIC_LOOP, length)


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


def _occ_closed_bspline_loop(occ, loop: Sequence[Point2], lc: float) -> int:
    """Single closed ``addBSpline`` wire → curve loop tag."""
    verts = [(float(x), float(y)) for x, y in loop]
    if len(verts) < 4:
        raise ValueError("B-spline loop needs at least 4 control points")
    # Drop duplicate closing vertex; repeat first tag for periodic spline.
    if verts[-1] == verts[0]:
        verts = verts[:-1]
    pt_tags = [int(occ.addPoint(x, y, 0.0, lc)) for x, y in verts]
    curve_tag = int(occ.addBSpline(pt_tags + [pt_tags[0]]))
    return int(occ.addCurveLoop([curve_tag]))


def _build_profile_solid(
    occ,
    loop: Sequence[Point2],
    *,
    depth: float,
    label: str,
) -> None:
    max_extent = max(max(abs(float(x)), abs(float(y))) for x, y in loop)
    lc = max(max_extent, float(depth), 0.1) / 50.0

    loop_tag = _occ_closed_bspline_loop(occ, loop, lc)
    surf_tag = int(occ.addPlaneSurface([loop_tag]))

    xs = [float(x) for x, _ in loop]
    neck_x = max(xs)
    tip_x = min(xs)
    print(
        f"[diag] {label}: luthier B-spline, {len(loop)} control pts, "
        f"neck_x={neck_x:.3f} tip_x={tip_x:.3f} span≈{neck_x - tip_x:.3f}m"
    )

    try:
        ext = occ.extrude([(2, surf_tag)], 0, 0, float(depth), [3])
    except (TypeError, ValueError):
        ext = occ.extrude([(2, surf_tag)], 0, 0, float(depth))
    vol_tags = _volume_tags_from_extrude(ext)
    if not vol_tags:
        raise RuntimeError("Extrude produced no volume.")
    occ.translate([(3, vol_tags[0])], 0, 0, -float(depth) / 2.0)


def _write_step(name: str, builder) -> None:
    import gmsh

    gmsh.model.add(name)
    builder(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    out = MODELS_DIR / name
    gmsh.write(str(out))
    print(f"Wrote {out}")
    gmsh.model.remove()


def _classic(occ) -> None:
    _build_profile_solid(
        occ, CLASSICAL_LOOP, depth=0.095, label="classical (Torres)"
    )


def _acoustic(occ) -> None:
    _build_profile_solid(
        occ, ACOUSTIC_LOOP, depth=0.115, label="acoustic (Martin D-28)"
    )


def _box(occ) -> None:
    d = get_luthier_gui_defaults("Box")
    lx, ly, lz = float(d["length"]), float(d["width"]), float(d["depth"])
    occ.addBox(-lx / 2.0, -ly / 2.0, -lz / 2.0, lx, ly, lz)


def main() -> int:
    import gmsh

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
