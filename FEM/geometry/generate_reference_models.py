#!/usr/bin/env python3
"""
Write STEP reference bodies into FEM/geometry/models/.

Classical and dreadnought bodies use closed B-spline figure-8 outlines; box stays rectangular.

    python3 FEM/geometry/generate_reference_models.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import gmsh

MODELS_DIR = Path(__file__).resolve().parent / "models"

Point2 = Tuple[float, float]


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def _half_width_classical(x_norm: float, *, upper: float, waist: float, lower: float) -> float:
    """Half-width along body (x_norm: 0=neck, 1=tail). Torres-style figure-8."""
    u = max(0.0, min(1.0, float(x_norm)))
    y_upper = (upper / 2.0) * (0.55 + 0.45 * math.exp(-((u - 0.14) / 0.11) ** 2))
    y_waist = (waist / 2.0) * (1.0 - 0.12 * math.exp(-((u - 0.40) / 0.10) ** 2))
    y_lower = (lower / 2.0) * math.exp(-((u - 0.58) / 0.14) ** 2)
    y = y_upper
    if u > 0.22:
        y = y_upper + (y_waist - y_upper) * _smoothstep((u - 0.22) / 0.18)
    if u > 0.40:
        y = y_waist + (y_lower - y_waist) * _smoothstep((u - 0.40) / 0.22)
    if u > 0.72:
        y = y_lower * (1.0 - _smoothstep((u - 0.72) / 0.28))
    neck = math.sin(0.5 * math.pi * min(u / 0.10, 1.0)) ** 0.9 if u < 0.10 else 1.0
    tail = math.sin(0.5 * math.pi * min((1.0 - u) / 0.10, 1.0)) ** 1.0 if u > 0.90 else 1.0
    return max(0.0, y * neck * tail)


def _half_width_dreadnought(x_norm: float, *, upper: float, waist: float, lower: float) -> float:
    """Dreadnought: broader shoulders/lower bout, shallower waist."""
    u = max(0.0, min(1.0, float(x_norm)))
    y_upper = (upper / 2.0) * (0.62 + 0.38 * math.exp(-((u - 0.12) / 0.13) ** 2))
    y_waist = (waist / 2.0) * (1.0 - 0.06 * math.exp(-((u - 0.36) / 0.12) ** 2))
    y_lower = (lower / 2.0) * math.exp(-((u - 0.54) / 0.16) ** 2)
    y = y_upper
    if u > 0.18:
        y = y_upper + (y_waist - y_upper) * _smoothstep((u - 0.18) / 0.16)
    if u > 0.36:
        y = y_waist + (y_lower - y_waist) * _smoothstep((u - 0.36) / 0.24)
    if u > 0.70:
        y = y_lower * (1.0 - _smoothstep((u - 0.70) / 0.30))
    neck = math.sin(0.5 * math.pi * min(u / 0.09, 1.0)) ** 0.85 if u < 0.09 else 1.0
    tail = math.sin(0.5 * math.pi * min((1.0 - u) / 0.09, 1.0)) ** 0.95 if u > 0.91 else 1.0
    return max(0.0, y * neck * tail)


def _half_profile(
    length: float,
    *,
    upper_bout: float,
    waist: float,
    lower_bout: float,
    kind: str,
    n_stations: int = 32,
) -> List[Point2]:
    """+y half-profile: neck (+L/2, 0) → tail (−L/2, 0)."""
    L = float(length)
    width_fn = _half_width_dreadnought if kind == "dreadnought" else _half_width_classical
    pts: List[Point2] = []
    for i in range(n_stations):
        x_norm = i / float(n_stations - 1)
        x = 0.5 * L - x_norm * L
        y = width_fn(x_norm, upper=upper_bout, waist=waist, lower=lower_bout)
        pts.append((x, y))
    pts[0] = (0.5 * L, 0.0)
    pts[-1] = (-0.5 * L, 0.0)
    if len(pts) >= 2 and pts[1][1] < 1.0e-5:
        pts[1] = (pts[1][0], 1.0e-4)
    if len(pts) >= 3 and pts[-2][1] < 1.0e-5:
        pts[-2] = (pts[-2][0], 1.0e-4)
    return pts


def _closed_perimeter(half: Sequence[Point2]) -> List[Point2]:
    """Perimeter walk: upper (+y) then lower (−y), neck → tail → neck."""
    upper = list(half)
    lower = [(x, -y) for x, y in reversed(upper)]
    return upper + lower[1:-1]


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


def _extrude_figure8_solid(
    occ,
    *,
    length: float,
    upper_bout: float,
    waist: float,
    lower_bout: float,
    depth: float,
    kind: str,
) -> None:
    """BSpline figure-8 outline in xy, extruded along +z, centred on z=0."""
    half = _half_profile(
        length,
        upper_bout=upper_bout,
        waist=waist,
        lower_bout=lower_bout,
        kind=kind,
    )
    ring = _closed_perimeter(half)
    lc = max(float(length), float(depth)) / 80.0

    p_tags = [int(occ.addPoint(float(x), float(y), 0.0, lc)) for x, y in ring]
    try:
        bsp = int(occ.addBSpline(p_tags))
        loop_tag = int(occ.addCurveLoop([bsp, int(occ.addLine(p_tags[-1], p_tags[0]))]))
    except Exception:
        line_tags = [
            int(occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]))
            for i in range(len(p_tags))
        ]
        loop_tag = int(occ.addCurveLoop(line_tags))

    surf_tag = int(occ.addPlaneSurface([loop_tag]))
    try:
        ext = occ.extrude([(2, surf_tag)], 0, 0, float(depth), [3])
    except (TypeError, ValueError):
        ext = occ.extrude([(2, surf_tag)], 0, 0, float(depth))
    vol_tags = _volume_tags_from_extrude(ext)
    if not vol_tags:
        raise RuntimeError(f"Extrude failed for {kind} reference (depth={depth}).")
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
    """Nominal 0.50 × 0.36 × 0.10 m classical (Torres proportions)."""
    _extrude_figure8_solid(
        occ,
        length=0.50,
        upper_bout=0.277,
        waist=0.241,
        lower_bout=0.365,
        depth=0.10,
        kind="classical",
    )


def _acoustic(occ) -> None:
    """Nominal dreadnought: 0.50 × 0.40 × 0.10 m, broader bouts."""
    _extrude_figure8_solid(
        occ,
        length=0.50,
        upper_bout=0.305,
        waist=0.238,
        lower_bout=0.400,
        depth=0.10,
        kind="dreadnought",
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
