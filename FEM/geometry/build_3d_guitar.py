import gmsh
import sys
import json
import os
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Normalized half-profile templates: x_norm 0=neck → 1=tail, y_norm 0=centerline → 1=max half-width.
# Pre-calculated from stable analytical Torres / Martin D-28 proportions (64 points, no splines).


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def _bout_width_scale(x_norm: float, upper_bout: float, waist: float, lower_bout: float) -> float:
    """Smooth upper → waist → lower bout multipliers along normalized length."""
    xn = max(0.0, min(1.0, float(x_norm)))
    u, w, l = float(upper_bout), float(waist), float(lower_bout)
    if xn < 0.36:
        return u + (w - u) * _smoothstep(xn / 0.36)
    if xn < 0.58:
        return w + (l - w) * _smoothstep((xn - 0.36) / 0.22)
    return l * (1.0 - _smoothstep((xn - 0.58) / 0.42))


def _classical_y_norm(x_norm: float) -> float:
    """Torres/Hauser-style half-width envelope (deterministic, non-inflecting)."""
    u = max(0.0, min(1.0, float(x_norm)))
    lower = math.exp(-((u - 0.56) / 0.11) ** 2)
    waist = 1.0 - 0.28 * math.exp(-((u - 0.40) / 0.09) ** 2)
    upper = 0.62 + 0.38 * math.exp(-((u - 0.14) / 0.16) ** 2)
    neck = math.sin(0.5 * math.pi * u / 0.12) ** 0.9 if u < 0.12 else 1.0
    tail = math.sin(0.5 * math.pi * (1.0 - u) / 0.12) ** 1.1 if u > 0.88 else 1.0
    return max(0.0, lower * waist * upper * neck * tail)


def _dreadnought_y_norm(x_norm: float) -> float:
    """Martin D-28-style half-width envelope (broader shoulders, shallower waist)."""
    u = max(0.0, min(1.0, float(x_norm)))
    lower = math.exp(-((u - 0.54) / 0.13) ** 2)
    waist = 1.0 - 0.12 * math.exp(-((u - 0.38) / 0.12) ** 2)
    upper = 0.72 + 0.28 * math.exp(-((u - 0.12) / 0.14) ** 2)
    neck = math.sin(0.5 * math.pi * u / 0.10) ** 0.85 if u < 0.10 else 1.0
    tail = math.sin(0.5 * math.pi * (1.0 - u) / 0.10) ** 1.0 if u > 0.90 else 1.0
    return max(0.0, lower * waist * upper * neck * tail)


def _build_normalized_template(kind: str, n: int = 64) -> Tuple[Tuple[float, float], ...]:
    yn_fn = _dreadnought_y_norm if kind == "dreadnought" else _classical_y_norm
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        xn = i / float(n - 1)
        pts.append((xn, yn_fn(xn)))
    ymax = max(y for _, y in pts) or 1.0
    normed = [(xn, (y / ymax if ymax > 1.0e-9 else y)) for xn, y in pts]
    normed[0] = (0.0, 0.0)
    normed[-1] = (1.0, 0.0)
    return tuple(normed)


CLASSICAL_TEMPLATE_2D: Tuple[Tuple[float, float], ...] = _build_normalized_template("classical")
DREADNOUGHT_TEMPLATE_2D: Tuple[Tuple[float, float], ...] = _build_normalized_template("dreadnought")


def _template_for_shape(shape_type: str) -> Tuple[Tuple[float, float], ...]:
    if "dread" in str(shape_type).strip().lower():
        return DREADNOUGHT_TEMPLATE_2D
    return CLASSICAL_TEMPLATE_2D


def warp_template_half_profile(
    template: Sequence[Tuple[float, float]],
    *,
    length: float,
    upper_bout: float,
    waist: float,
    lower_bout: float,
    wall_offset: float = 0.0,
) -> List[Tuple[float, float]]:
    """Map normalized template → physical half-profile (neck +L/2, tail -L/2, +y side)."""
    L = float(length)
    out: List[Tuple[float, float]] = []
    for x_norm, y_norm in template:
        x = L * (0.5 - float(x_norm))
        y = float(y_norm) * _bout_width_scale(x_norm, upper_bout, waist, lower_bout)
        if wall_offset > 0.0:
            y = max(0.0, y - wall_offset)
            if x_norm < 0.06:
                x += wall_offset
            elif x_norm > 0.94:
                x -= wall_offset
        out.append((x, y))
    return out


def _subsample_chain_monotonic(
    coords: Sequence[Tuple[float, float]], n_max: int = 32
) -> List[Tuple[float, float]]:
    """Subsample while preserving template order (strictly increasing indices)."""
    pts = [(float(x), float(y)) for x, y in coords]
    n = len(pts)
    if n <= n_max:
        return pts
    raw_idx = [int(round(k * (n - 1) / float(n_max - 1))) for k in range(n_max)]
    idx: List[int] = []
    for j in raw_idx:
        if not idx or j > idx[-1]:
            idx.append(j)
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [pts[i] for i in idx]


def _full_ring_points_from_half(
    half: Sequence[Tuple[float, float]], n_side_max: int = 40
) -> List[Tuple[float, float]]:
    """
    Strict centerline closure: neck/tail snapped to y=0, lower leg mirrors side[1:-1] only.

    Avoids floating-point asymmetry at the centerline that breaks OCC wire assembly.
    """
    side = [(float(x), float(y)) for x, y in _subsample_chain_monotonic(half, n_side_max)]
    if len(side) < 3:
        raise RuntimeError("Half profile too short for closed perimeter.")

    # Force exact centerline anchors before mirroring (warp may leave y ≈ 1e-16, not 0).
    side[0] = (side[0][0], 0.0)
    side[-1] = (side[-1][0], 0.0)

    lower_half = [(pt[0], -pt[1]) for pt in reversed(side[1:-1])]
    full_ring_points = list(side) + lower_half
    if len(full_ring_points) < 4:
        raise RuntimeError("Closed profile ring collapsed (too few vertices).")
    return full_ring_points


def _filter_ring_min_spacing(
    points: Sequence[Tuple[float, float]], min_dist: float = 1e-4
) -> List[Tuple[float, float]]:
    """
    Drop vertices closer than min_dist (default 0.1 mm) to the previous kept point.

    Also drops the last vertex if it lies within min_dist of the first (closure gap).
    """
    pts: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts.pop()
    clean: List[Tuple[float, float]] = []
    for pt in pts:
        if not clean:
            clean.append(pt)
            continue
        if math.hypot(pt[0] - clean[-1][0], pt[1] - clean[-1][1]) > min_dist:
            clean.append(pt)
    if len(clean) >= 2:
        gap = math.hypot(clean[-1][0] - clean[0][0], clean[-1][1] - clean[0][1])
        if gap <= min_dist:
            clean.pop()
    if len(clean) < 4:
        raise RuntimeError(
            f"Ring spacing filter left {len(clean)} vertices (need >= 4, min_dist={min_dist} m)."
        )
    return clean


def _occ_polygon_contour_from_point_tags(p_tags: List[int]) -> int:
    """
    OCC closed polygon wire when ``addPolygon`` is unavailable (spacing already filtered).
    """
    occ = gmsh.model.occ
    n = len(p_tags)
    if n < 4:
        raise RuntimeError(f"OCC polygon needs >= 4 point tags (got {n}).")
    l_tags = [
        int(occ.addLine(p_tags[i], p_tags[(i + 1) % n])) for i in range(n)
    ]
    occ.synchronize()
    add_wire = getattr(occ, "addWire", None)
    if add_wire is not None:
        try:
            return int(add_wire(l_tags, tag=-1, checkClosed=True))
        except TypeError:
            return int(add_wire(l_tags))
    return int(occ.addCurveLoop(l_tags))


def _volume_tags_from_extrude(out) -> List[int]:
    """Parse ``geo/occ.extrude`` return value into 3D volume tags (ints only)."""
    tags: List[int] = []

    def _walk(obj) -> None:
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2:
                try:
                    dim = int(obj[0])
                    tag = int(obj[1])
                    if dim == 3:
                        tags.append(tag)
                        return
                except (TypeError, ValueError):
                    pass
            for child in obj:
                _walk(child)

    _walk(out)
    return tags


def _geo_closed_polyline_surface(
    ring_points: Sequence[Tuple[float, float]], lc: float
) -> int:
    """Sketch preview profile: built-in geo kernel (stable closed polygon loops)."""
    geo = gmsh.model.geo
    p_tags = [int(geo.addPoint(float(x), float(y), 0.0, lc)) for x, y in ring_points]
    n_pts = len(p_tags)
    if n_pts < 4:
        raise RuntimeError(f"Geo profile needs >= 4 points (got {n_pts}).")
    l_tags = [
        int(geo.addLine(p_tags[i], p_tags[(i + 1) % n_pts])) for i in range(n_pts)
    ]
    loop_tag = int(geo.addCurveLoop(l_tags))
    surface_tag = int(geo.addPlaneSurface([loop_tag]))
    geo.synchronize()
    return surface_tag


def _geo_extrude_shell_volume(surface_tag: int, dz: float, z_shift: float) -> int:
    """Extrude a geo 2D profile surface to a 3D volume."""
    geo = gmsh.model.geo
    ext = geo.extrude([(2, int(surface_tag))], 0, 0, float(dz))
    vol_tags = _volume_tags_from_extrude(ext)
    if not vol_tags:
        raise RuntimeError(f"geo.extrude produced no volume (dz={dz}).")
    geo.translate([(3, vol_tags[0])], 0, 0, float(z_shift))
    geo.synchronize()
    return vol_tags[0]


def _geo_box_volume(L: float, W: float, D: float) -> int:
    """Solid box via geo rectangle + extrude (sketch preview)."""
    geo = gmsh.model.geo
    surf = int(geo.addRectangle(-L / 2.0, -W / 2.0, 0.0, L, W))
    ext = geo.extrude([(2, surf)], 0, 0, float(D))
    vol_tags = _volume_tags_from_extrude(ext)
    if not vol_tags:
        raise RuntimeError("geo box extrude produced no volume.")
    geo.translate([(3, vol_tags[0])], 0, 0, -D / 2.0)
    geo.synchronize()
    return vol_tags[0]


def _occ_extrude_shell_volume(surface_tag: int, dz: float, z_shift: float) -> int:
    """Extrude an OCC 2D profile surface to a 3D volume."""
    occ = gmsh.model.occ
    try:
        ext = occ.extrude([(2, int(surface_tag))], 0, 0, float(dz), [3])
    except (TypeError, ValueError):
        ext = occ.extrude([(2, int(surface_tag))], 0, 0, float(dz))
    vol_tags = _volume_tags_from_extrude(ext)
    if not vol_tags:
        raise RuntimeError(f"occ.extrude produced no volume (dz={dz}).")
    occ.translate([(3, vol_tags[0])], 0, 0, float(z_shift))
    occ.synchronize()
    return vol_tags[0]


def _stabilize_half_profile(half: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Nudge the first outer points off the centerline so OCC lines are non-degenerate."""
    out = [(float(x), float(y)) for x, y in half]
    if len(out) >= 2 and out[1][1] < 1.0e-5:
        out[1] = (out[1][0], 1.0e-4)
    if len(out) >= 3 and out[-2][1] < 1.0e-5:
        out[-2] = (out[-2][0], 1.0e-4)
    return out


def audit_enabled() -> bool:
    return os.environ.get("FEM_RENDER_AUDIT", "0").strip().lower() in ("1", "true", "yes", "on")

def _script_flags() -> tuple[bool, bool]:
    """Script-only CLI flags (must not be passed to ``gmsh.initialize``)."""
    argv = sys.argv[1:]
    return ("--preview" in argv, "-nopopup" in argv)


def _gmsh_initialize_argv() -> list[str]:
    """Argv for Gmsh: drop script-only tokens so Gmsh does not warn on unknown options."""
    skip_next = False
    out = [sys.argv[0]]
    for i, arg in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--preview", "-nopopup"):
            continue
        if arg == "--config":
            skip_next = True
            continue
        out.append(arg)
    return out


def _resolve_config_path(fem_dir: Path) -> Path:
    """CLI ``--config PATH`` overrides default ``configs/guitar_3d.json``."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            p = Path(sys.argv[i + 1]).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            return p
    return fem_dir / "configs" / "guitar_3d.json"


def create_guitar_mesh():
    # 1. Setup paths
    geometry_dir = Path(__file__).resolve().parent
    fem_dir = geometry_dir.parent
    config_path = _resolve_config_path(fem_dir)
    mesh_dir = fem_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Mesh pipeline mode (mutually exclusive; set by gui/app.py subprocess env):
    #   FEM_ALLOW_PREVIEW=1  → sketch wireframe (coarse lc)
    #   FEM_ALLOW_DISPLAY=1 → PyVista display shell only (uniform lc, no air)
    #   FEM_ALLOW_FOM=1     → full FSI volume mesh for FEM (never shown in PyVista)
    preview_cli, _nopopup = _script_flags()
    is_display = os.environ.get("FEM_ALLOW_DISPLAY", "0") == "1"
    is_preview = (os.environ.get("FEM_ALLOW_PREVIEW", "0") == "1" or preview_cli) and not is_display
    is_fom = os.environ.get("FEM_ALLOW_FOM", "0") == "1"
    shell_only = is_preview or is_display

    if is_display:
        out_file = mesh_dir / "display_mesh.msh"
    elif is_preview:
        out_file = mesh_dir / "preview_mesh.msh"
    elif is_fom:
        out_file = mesh_dir / "guitar_3d.msh"
    else:
        raise RuntimeError(
            "Set exactly one mesh mode env var: FEM_ALLOW_PREVIEW, FEM_ALLOW_DISPLAY, or FEM_ALLOW_FOM."
        )
    # -------------------------------------------

    # 2. Load geometry data
    print(f"[diag] mesh build config: {config_path}")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        top_mat = config.get("materials", {}).get("top", {})
        back_mat = config.get("materials", {}).get("back", {})
        if top_mat:
            print(
                f"[diag] FEM materials.top: rho={top_mat.get('density')} kg/m^3, E_L={top_mat.get('E_L')} Pa, "
                f"name={top_mat.get('name', '')!r}"
            )
        if back_mat:
            print(
                f"[diag] FEM materials.back: rho={back_mat.get('density')} kg/m^3, E_L={back_mat.get('E_L')} Pa, "
                f"name={back_mat.get('name', '')!r}"
            )
        p = config["geometry"]
        L, W, D = p["length"], p["width"], p["depth"]
        # CAD wall offset uses top-plate thickness (back is thicker in FEM shell forms only).
        t = float(p.get("top_thickness", p.get("thickness", 0.003)))
        hr = min(float(p["hole_radius"]), 0.08)
        shape_type = str(p.get("shape_type", "Classical")).strip()
        upper_bout = float(p.get("upper_bout", W * 0.75))
        lower_bout = float(p.get("lower_bout", W))
        waist = float(p.get("waist", W * 0.65))
        hole_from_neck_ratio = float(p.get("soundhole_from_neck_ratio", 0.5))
    else:
        L, W, D, t, hr, shape_type = 0.48, 0.37, 0.1, 0.003, 0.04, "Classical"
        upper_bout, lower_bout, waist = W * 0.75, W, W * 0.65
        hole_from_neck_ratio = 0.5

    def _is_box_shape(st: str) -> bool:
        return str(st).strip().lower() == "box"

    # Feasible wall thickness for hollow-shell booleans (avoids zero-thickness / PLC mesh failures).
    t = max(0.001, min(float(t), max(0.001, 0.45 * float(D))))
    inner_depth = max(1.0e-4, float(D) - 2.0 * t)

    mode = "display" if is_display else ("sketch" if is_preview else "fom")
    print(f"[diag] shape_type={shape_type!r} mesh_mode={mode}")

    # Soundhole centre: measured from neck (+x) toward bridge; higher ratio → lower bout.
    hole_from_neck_ratio = float(max(0.05, min(0.95, hole_from_neck_ratio)))
    hole_x = 0.5 * L - hole_from_neck_ratio * L
    hole_y = 0.0

    # Engineering: high-density shell (6 mm) + 1 mm thickness edges.
    # Preview sketch: uniform wood_surface_lc (6 mm) on solid volume — no size gradients. Display: 4 mm.
    wood_surface_size = 0.006   # 6 mm engineering baseline (user target lc)
    wood_thickness_size = 0.001  # 1 mm on short ~through-thickness edges (>=2–3 elements across ~3 mm wood)
    thickness_curve_len_max = 0.005  # curves shorter than this (m) are treated as thickness direction
    # Thickness-edge Threshold: smooth 1 mm → 6.5 mm over ~8 mm band from short edges
    thickness_threshold_dist_min = 0.0005
    thickness_threshold_dist_max = 0.008
    # Air Threshold field (distance from wood shell): near-field at soundhole band, coarser far cap for ~500 Hz
    air_threshold_dist_min = 0.015
    air_threshold_dist_max = 0.25
    air_threshold_size_min = 0.008   # 8 mm near wood (Helmholtz / hole region; bridges 6.5 mm shell)
    air_threshold_size_max = 0.080   # 80 mm far field

    # Sketch: coarse. Display: uniform 4 mm shell (no adaptive fields). FOM: full graded FSI mesh.
    if is_display:
        mesh_size = 0.004
        mesh_size_min = 0.004
        mesh_size_max = 0.004
    elif is_preview:
        # Real-time sketch: uniform 6 mm skin (zero gradient — avoids spline-cap skew / core dump).
        mesh_size = wood_surface_size
        mesh_size_min = wood_surface_size
        mesh_size_max = wood_surface_size
    else:
        mesh_size = wood_surface_size
        mesh_size_min = wood_thickness_size
        mesh_size_max = air_threshold_size_max

    print(
        "DEBUG: Golden mesh — wood: 6.5 mm faces / 1 mm thickness edges; "
        "air: 8 mm near wood → 80 mm far (Dist 1.5–25 cm)."
    )
    print(
        f"Building geometry with Thickness: {t*1000:.1f}mm (inner_depth={inner_depth*1000:.1f}mm), "
        f"wood_surface_lc={wood_surface_size*1000:.1f}mm, wood_thickness_lc={wood_thickness_size*1000:.1f}mm"
    )
    if shell_only:
        print(f"[diag] shell mesh target lc={mesh_size*1000:.2f}mm (mode={mode})")
    print(
        f"[diag] mesh_mode={mode} preview={is_preview} display={is_display} fom={is_fom} "
        f"FEM_ALLOW_PREVIEW={os.environ.get('FEM_ALLOW_PREVIEW', '0')} "
        f"FEM_ALLOW_DISPLAY={os.environ.get('FEM_ALLOW_DISPLAY', '0')} "
        f"FEM_ALLOW_FOM={os.environ.get('FEM_ALLOW_FOM', '0')}"
    )

    # Body frame: x = +L/2 at neck, x = -L/2 at tail; y = lateral; z = up.
    hr = max(1.0e-4, float(hr))
    hole_x = float(max(-0.5 * L + hr, min(0.5 * L - hr, hole_x)))
    hole_y = 0.0
    print(
        f"[diag] soundhole centre (m): x={hole_x:.4f} y={hole_y:.4f} r={hr:.4f} "
        f"(from_neck_ratio={hole_from_neck_ratio:.3f})"
    )

    gmsh.initialize(_gmsh_initialize_argv())
    if audit_enabled():
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 5)
        print("[AUDIT] Gmsh verbosity elevated (General.Verbosity=5)")
    gmsh.model.add("Guitar3D_Performance_Optimized")
    occ = gmsh.model.occ
    geo = gmsh.model.geo
    sketch_use_geo = bool(is_preview)
    gmsh.option.setNumber("Geometry.Tolerance", 1.0e-4)
    if not sketch_use_geo:
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    if sketch_use_geo:
        print("[diag] CAD kernel fork: sketch → gmsh.model.geo (solid preview, no OCC booleans)")
    else:
        print(f"[diag] CAD kernel fork: {mode} → gmsh.model.occ (hollow/FSI booleans)")

    def create_template_profile_surface(
        *,
        length: float,
        upper_bout: float,
        waist: float,
        lower_bout: float,
        template: Sequence[Tuple[float, float]],
        wall_offset: float = 0.0,
        point_lc: Optional[float] = None,
    ) -> int:
        """Normalized template → closed ring → geo (sketch) or OCC (display/FOM) surface."""
        lc = float(point_lc if point_lc is not None else mesh_size)
        half = _stabilize_half_profile(
            warp_template_half_profile(
                template,
                length=length,
                upper_bout=upper_bout,
                waist=waist,
                lower_bout=lower_bout,
                wall_offset=wall_offset,
            )
        )
        full_ring_points = _full_ring_points_from_half(half, n_side_max=40)
        clean_ring_points = _filter_ring_min_spacing(full_ring_points, min_dist=1e-4)

        if sketch_use_geo:
            # Perimeter order from _full_ring_points_from_half (neck→tail→return); no polar sort
            # (atan2 breaks non-convex guitar waists).
            surface_tag = _geo_closed_polyline_surface(clean_ring_points, lc)
            print(
                f"[diag] template profile (geo sketch): ring_in={len(full_ring_points)} "
                f"perimeter_pts={len(clean_ring_points)} surface={surface_tag}"
            )
            return surface_tag

        p_tags: List[int] = [
            int(gmsh.model.occ.addPoint(pt[0], pt[1], 0.0, lc)) for pt in clean_ring_points
        ]
        add_polygon = getattr(gmsh.model.occ, "addPolygon", None)
        polygon_mode = "addPolygon"
        try:
            if add_polygon is not None:
                polygon_loop_tag = int(add_polygon(p_tags))
            else:
                polygon_mode = "addWire(fallback)"
                polygon_loop_tag = _occ_polygon_contour_from_point_tags(p_tags)
            surface_tag = int(gmsh.model.occ.addPlaneSurface([polygon_loop_tag]))
        except Exception as exc:
            raise RuntimeError(
                f"OCC polygon profile failed ({polygon_mode}): {exc}"
            ) from exc
        gmsh.model.occ.synchronize()
        print(
            f"[diag] template profile (OCC {polygon_mode}): ring_in={len(full_ring_points)} "
            f"ring_out={len(clean_ring_points)} pts={len(p_tags)} surface={surface_tag}"
        )
        return surface_tag

    def as_dimtags(result):
        if isinstance(result, tuple):
            if result and isinstance(result[0], list):
                return result[0]
            return list(result)
        return result

    def get_boundary_tags(dimtags, dim):
        bnds = gmsh.model.getBoundary(dimtags, oriented=False, recursive=False)
        return {tag for bdim, tag in bnds if bdim == dim}

    def get_point_tags_from_surfaces(surface_tags):
        if not surface_tags:
            return set()
        bnds = gmsh.model.getBoundary([(2, s) for s in surface_tags], oriented=False, recursive=True)
        return {tag for bdim, tag in bnds if bdim == 0}

    def get_point_tags_from_surface(surface_tag):
        bnds = gmsh.model.getBoundary([(2, surface_tag)], oriented=False, recursive=True)
        return {tag for bdim, tag in bnds if bdim == 0}

    def _entity_center_of_mass(dim: int, entity_tag: int) -> Tuple[float, float, float]:
        try:
            com = occ.getCenterOfMass(int(dim), int(entity_tag))
            return (float(com[0]), float(com[1]), float(com[2]))
        except Exception:
            bb = gmsh.model.getBoundingBox(int(dim), int(entity_tag))
            return (
                0.5 * (bb[0] + bb[3]),
                0.5 * (bb[1] + bb[4]),
                0.5 * (bb[2] + bb[5]),
            )

    def get_surface_center_z(surf_tag):
        return _entity_center_of_mass(2, surf_tag)[2]

    def get_surface_center(surf_tag):
        return _entity_center_of_mass(2, surf_tag)

    def get_volume_center_z(vol_tag):
        return _entity_center_of_mass(3, vol_tag)[2]

    def _curve_length_m(ctag):
        """Physical length (m) of OCC curve tag."""
        try:
            mass = gmsh.model.occ.getMass(1, int(ctag))
            if isinstance(mass, (int, float)):
                return float(mass)
            if isinstance(mass, (list, tuple)) and len(mass) >= 1:
                return float(mass[0])
        except Exception:
            pass
        try:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, int(ctag))
            return float(
                math.sqrt(
                    (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
                )
            )
        except Exception:
            return float("inf")

    def _wood_boundary_curve_tags(vol_tags):
        seen = set()
        out = []
        for v in vol_tags:
            bnd = gmsh.model.getBoundary([(3, int(v))], oriented=False, recursive=True)
            for dim, tag in bnd:
                if dim == 1 and tag not in seen:
                    seen.add(tag)
                    out.append(int(tag))
        return out

    def get_surface_normal_signed_z(surf_tag):
        """Signed nz at surface midpoint (+Z = outward top). None if unavailable."""
        try:
            uv_min, uv_max = gmsh.model.getParametrizationBounds(2, surf_tag)
            u_mid = 0.5 * (uv_min[0] + uv_max[0])
            v_mid = 0.5 * (uv_min[1] + uv_max[1])
            n = gmsh.model.getNormal(surf_tag, [u_mid, v_mid])
            if n and len(n) >= 3:
                return float(n[2])
        except Exception:
            return None
        return None

    def get_surface_normal_z(surf_tag):
        """Return |nz| at parametric midpoint (soundhole span filter)."""
        nz = get_surface_normal_signed_z(surf_tag)
        return abs(nz) if nz is not None else None

    def get_surface_normal_vec(surf_tag):
        try:
            uv_min, uv_max = gmsh.model.getParametrizationBounds(2, int(surf_tag))
            u_mid = 0.5 * (uv_min[0] + uv_max[0])
            v_mid = 0.5 * (uv_min[1] + uv_max[1])
            n = gmsh.model.getNormal(int(surf_tag), [u_mid, v_mid])
            if n and len(n) >= 3:
                return (float(n[0]), float(n[1]), float(n[2]))
        except Exception:
            pass
        return None

    def _is_exterior_boundary_facet(surf_tag: int, vol_tag: int) -> bool:
        """Keep outer mould faces; drop interior cavity lining from hollow-shell boundary."""
        c = get_surface_center(surf_tag)
        vcom = _entity_center_of_mass(3, int(vol_tag))
        vc = (c[0] - vcom[0], c[1] - vcom[1], c[2] - vcom[2])
        n = get_surface_normal_vec(surf_tag)
        if n is None:
            return True
        dot = n[0] * vc[0] + n[1] * vc[1] + n[2] * vc[2]
        return dot > -5.0e-7

    def _wood_volumes_adjacent_to_surface(surf_tag: int, wood_vol_set: set) -> set:
        """Wood volume tags sharing a surface (Z-partition internal faces touch two)."""
        found: set = set()
        try:
            up, down = gmsh.model.getAdjacencies(2, int(surf_tag))
            for lst in (up, down):
                for dim, etag in lst:
                    if int(dim) == 3 and int(etag) in wood_vol_set:
                        found.add(int(etag))
        except Exception:
            pass
        return found

    def _drop_wood_partition_interfaces(surfaces: list, wood_vol_set: set) -> list:
        """Remove horizontal Z-slab cuts from facet groups (they look like a box in the UI)."""
        if not surfaces or not wood_vol_set:
            return list(surfaces)
        kept = []
        dropped = 0
        for s in surfaces:
            if len(_wood_volumes_adjacent_to_surface(int(s), wood_vol_set)) >= 2:
                dropped += 1
                continue
            kept.append(int(s))
        if dropped:
            print(f"[diag] dropped {dropped} wood/wood partition interface facets")
        return kept

    # Sketch: solid outer volume only (20 mm lc cannot mesh a 3 mm hollow gap).
    # Display / FOM: hollow shell via outer − inner boolean.
    solid_sketch = bool(is_preview)

    # Build guitar solid (sketch → geo solid; display/FOM → OCC hollow + booleans).
    if sketch_use_geo:
        if _is_box_shape(shape_type):
            vol_out_id = _geo_box_volume(L, W, D)
        else:
            profile_template = _template_for_shape(shape_type)
            profile_lc = mesh_size
            print(
                f"[diag] template engine: shape={shape_type!r} n_pts={len(profile_template)} "
                f"upper={upper_bout:.4f} waist={waist:.4f} lower={lower_bout:.4f} lc={profile_lc*1000:.1f}mm"
            )
            surf_out = create_template_profile_surface(
                length=L,
                upper_bout=upper_bout,
                waist=waist,
                lower_bout=lower_bout,
                template=profile_template,
                wall_offset=0.0,
                point_lc=profile_lc,
            )
            vol_out_id = _geo_extrude_shell_volume(surf_out, D, -D / 2.0)
        vol_in_id = vol_out_id
        print(
            f"[diag] sketch geo solid: volume={vol_out_id} "
            "(no OCC inner shell / soundhole cut)"
        )
    elif _is_box_shape(shape_type):
        vol_out_id = occ.addBox(-L/2, -W/2, -D/2, L, W, D)
        vol_in_id = occ.addBox(
            -L/2 + t, -W/2 + t, -D/2 + t, L - 2 * t, W - 2 * t, D - 2 * t
        )
    else:
        profile_template = _template_for_shape(shape_type)
        profile_lc = mesh_size
        print(
            f"[diag] template engine: shape={shape_type!r} n_pts={len(profile_template)} "
            f"upper={upper_bout:.4f} waist={waist:.4f} lower={lower_bout:.4f} lc={profile_lc*1000:.1f}mm"
        )
        surf_out = create_template_profile_surface(
            length=L,
            upper_bout=upper_bout,
            waist=waist,
            lower_bout=lower_bout,
            template=profile_template,
            wall_offset=0.0,
            point_lc=profile_lc,
        )
        vol_out_id = _occ_extrude_shell_volume(surf_out, D, -D / 2.0)
        surf_in = create_template_profile_surface(
            length=L,
            upper_bout=upper_bout,
            waist=waist,
            lower_bout=lower_bout,
            template=profile_template,
            wall_offset=t,
            point_lc=profile_lc,
        )
        vol_in_id = _occ_extrude_shell_volume(surf_in, inner_depth, -D / 2.0 + t)
        print(f"[diag] OCC hollow shell volumes: outer={vol_out_id} inner={vol_in_id}")

    hole_cyl: Optional[int] = None
    if not sketch_use_geo:
        if _is_box_shape(shape_type):
            hole_x, hole_y = 0.0, 0.0
        z_hole_lo = (D / 2.0) - t - 0.001
        z_hole_hi = (D / 2.0) + 0.001
        hole_cyl = int(
            occ.addCylinder(hole_x, hole_y, z_hole_lo, 0, 0, z_hole_hi - z_hole_lo, hr)
        )

    def _audit_boolean(stage: str, op, *args, **kwargs):
        """Log OCC boolean stage (B-rep / NURBS — always before gmsh.model.mesh.generate)."""
        if not audit_enabled():
            return op(*args, **kwargs)
        t0 = time.perf_counter()
        print(
            f"[AUDIT] Soundhole/wood boolean stage={stage!r} "
            "backend=OpenCASCADE_BRep (pre-discretization)"
        )
        try:
            out = op(*args, **kwargs)
            dt = time.perf_counter() - t0
            print(f"[AUDIT] stage={stage!r} completed in {dt:.3f}s")
            return out
        except Exception as exc:
            print(f"[AUDIT][ERROR] stage={stage!r} failed: {exc}")
            raise

    if shell_only:
        # Sketch + display: wood only (no acoustic air volume).
        if solid_sketch:
            print(
                "[diag] sketch CAD: solid outer volume only "
                "(geo kernel — no occ.cut hollow)"
            )
            wood_dimtags = [(3, int(vol_out_id))]
            if sketch_use_geo:
                geo.synchronize()
            else:
                occ.synchronize()
        else:
            print(f"[diag] {mode} CAD: hollow wood shell (outer − inner cut)")
            wood_shell = _audit_boolean(
                "display_hollow_shell",
                occ.cut,
                [(3, vol_out_id)],
                [(3, vol_in_id)],
                removeObject=True,
                removeTool=True,
            )
            wood_dimtags = [dt for dt in as_dimtags(wood_shell) if dt[0] == 3]
        if is_display:
            wood_hole_cut = _audit_boolean(
                "display_soundhole_cut",
                occ.cut,
                wood_dimtags,
                [(3, hole_cyl)],
                removeObject=True,
                removeTool=True,
            )
            wood_dimtags = [dt for dt in as_dimtags(wood_hole_cut) if dt[0] == 3]
        else:
            print("[diag] preview CAD: soundhole OCC cut skipped (GUI paints hole)")
        try:
            occ.removeAllDuplicates()
        except Exception:
            pass
        occ.synchronize()
        wood_vols = [int(tag) for _, tag in wood_dimtags]
        air_vols: list = []
        air_dimtags: list = []
    else:
        # FSI engineering mesh: internal air cavity + shared interface with wood.
        air_cut = _audit_boolean(
            "engineering_air_hole",
            occ.cut,
            [(3, vol_in_id)],
            [(3, hole_cyl)],
            removeObject=True,
            removeTool=False,
        )
        air_dimtags = [dt for dt in as_dimtags(air_cut) if dt[0] == 3]

        wood_cut = _audit_boolean(
            "engineering_wood_hollow",
            occ.cut,
            [(3, vol_out_id)],
            air_dimtags,
            removeObject=True,
            removeTool=False,
        )
        wood_dimtags = [dt for dt in as_dimtags(wood_cut) if dt[0] == 3]
        wood_hole_cut = _audit_boolean(
            "engineering_soundhole_cut",
            occ.cut,
            wood_dimtags,
            [(3, hole_cyl)],
            removeObject=True,
            removeTool=True,
        )
        wood_dimtags = [dt for dt in as_dimtags(wood_hole_cut) if dt[0] == 3]

        if not wood_dimtags or not air_dimtags:
            raise RuntimeError(
                f"FSI fragment skipped: wood_vols={len(wood_dimtags)} air_vols={len(air_dimtags)} "
                "(boolean hollow/soundhole may have failed)."
            )
        frags, _ = occ.fragment(wood_dimtags, air_dimtags, removeObject=True, removeTool=True)
        try:
            occ.removeAllDuplicates()
        except Exception:
            pass
        occ.synchronize()

        resulting_vols = [dt for dt in frags if dt[0] == 3]
        air_candidate_set = set(tag for _, tag in air_dimtags)
        air_vols = [tag for _, tag in resulting_vols if tag in air_candidate_set]
        wood_vols = [tag for _, tag in resulting_vols if tag not in air_candidate_set]

        if len(air_vols) != 1:
            raise RuntimeError(f"Expected exactly 1 internal air volume, found {len(air_vols)}")

    if not wood_vols:
        raise RuntimeError("No wood shell volumes found after boolean operations")

    # Z-partition with axis-aligned boxes is for FSI volume tagging only (creates boxy
    # facet artifacts in UI preview). Skip in preview — classify from exterior skin only.
    if is_fom and len(wood_vols) < 3:
        wood_bbox = [float("inf"), float("inf"), float("inf"), -float("inf"), -float("inf"), -float("inf")]
        for v in wood_vols:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, int(v))
            wood_bbox[0] = min(wood_bbox[0], xmin)
            wood_bbox[1] = min(wood_bbox[1], ymin)
            wood_bbox[2] = min(wood_bbox[2], zmin)
            wood_bbox[3] = max(wood_bbox[3], xmax)
            wood_bbox[4] = max(wood_bbox[4], ymax)
            wood_bbox[5] = max(wood_bbox[5], zmax)

        xmin, ymin, zmin, xmax, ymax, zmax = wood_bbox
        dz = max(1e-6, zmax - zmin)
        dx = max(1e-6, xmax - xmin)
        dy = max(1e-6, ymax - ymin)
        z1 = zmin + dz / 3.0
        z2 = zmin + 2.0 * dz / 3.0
        margin = 0.05 * max(dx, dy, dz)

        b_back = occ.addBox(xmin - margin, ymin - margin, zmin - margin, dx + 2 * margin, dy + 2 * margin, (z1 - zmin) + margin)
        b_mid = occ.addBox(xmin - margin, ymin - margin, z1, dx + 2 * margin, dy + 2 * margin, max(1e-6, z2 - z1))
        b_top = occ.addBox(xmin - margin, ymin - margin, z2, dx + 2 * margin, dy + 2 * margin, (zmax - z2) + margin)
        occ.synchronize()

        split_dimtags, _ = occ.fragment(
            [(3, int(v)) for v in wood_vols],
            [(3, b_back), (3, b_mid), (3, b_top)],
            removeObject=True,
            removeTool=True,
        )
        occ.synchronize()
        wood_vols = sorted([tag for dim, tag in split_dimtags if dim == 3])
        if not wood_vols:
            raise RuntimeError("Wood partitioning produced no volumes.")
        print(f"[diag] Wood volumes after Z-partition: {wood_vols}")

    if is_fom:
        # Final geometry unification: re-fragment wood+air for shared FSI interfaces.
        try:
            air_ref_com = (
                _entity_center_of_mass(3, int(air_vols[0])) if air_vols else (0.0, 0.0, 0.0)
            )
        except Exception:
            air_ref_com = (0.0, 0.0, 0.0)
        try:
            all_split, _ = occ.fragment(
                [(3, int(v)) for v in wood_vols],
                [(3, int(v)) for v in air_vols],
                removeObject=True,
                removeTool=True,
            )
            try:
                occ.removeAllDuplicates()
            except Exception:
                pass
            occ.synchronize()

            final_vols = sorted([tag for dim, tag in all_split if dim == 3])
            if final_vols:

                def _dist2_to_air(vtag):
                    cx, cy, cz = _entity_center_of_mass(3, int(vtag))
                    dx = cx - air_ref_com[0]
                    dy = cy - air_ref_com[1]
                    dz = cz - air_ref_com[2]
                    return dx * dx + dy * dy + dz * dz

                air_pick = min(final_vols, key=_dist2_to_air)
                air_vols = [int(air_pick)]
                wood_vols = [int(v) for v in final_vols if int(v) != int(air_pick)]
                print(f"[diag] Final unified volumes: wood={wood_vols}, air={air_vols}")
        except Exception as exc:
            print(f"[diag][warn] final wood/air unification skipped: {exc}")

    # Strict one-cell-one-tag policy for 3D physical groups.
    # Sort wood volumes by center-of-mass Z:
    # - highest Z -> tag 1 (Top volume)
    # - lowest Z  -> tag 2 (Back volume)
    # - remaining -> tag 3 (Ribs volume)
    wood_by_z = sorted([(v, get_volume_center_z(v)) for v in wood_vols], key=lambda row: row[1])
    top_vols = []
    back_vols = []
    rib_vols = []
    if len(wood_by_z) == 1:
        top_vols = [wood_by_z[0][0]]
    elif len(wood_by_z) >= 2:
        back_vols = [wood_by_z[0][0]]
        top_vols = [wood_by_z[-1][0]]
        rib_vols = [v for v, _z in wood_by_z[1:-1]]

    assigned_wood = set(top_vols) | set(back_vols) | set(rib_vols)
    if len(assigned_wood) != len(top_vols) + len(back_vols) + len(rib_vols):
        raise RuntimeError("Wood volume assignment overlap detected across tags 1/2/3.")
    if assigned_wood != set(wood_vols):
        missing = sorted(list(set(wood_vols) - assigned_wood))
        raise RuntimeError(f"Wood volume assignment incomplete. Unassigned volumes: {missing}")
    if air_vols and assigned_wood.intersection(set(air_vols)):
        overlap = sorted(list(assigned_wood.intersection(set(air_vols))))
        raise RuntimeError(f"Wood/Air volume overlap detected: {overlap}")

    # Surface identification via direct boundaries (fragmentation-safe, no interface tracing).
    wood_boundary_surfs = get_boundary_tags([(3, tag) for tag in wood_vols], 2)

    if shell_only and wood_vols:
        all_b = [int(s) for s in wood_boundary_surfs]
        if solid_sketch:
            wood_boundary_surfs = all_b
            print(
                f"[diag] sketch solid exterior: all {len(all_b)} boundary facets "
                "(no cavity-skin filter)"
            )
        else:
            primary_vol = int(wood_vols[0])
            exterior = [s for s in all_b if _is_exterior_boundary_facet(s, primary_vol)]
            if len(exterior) >= max(12, len(all_b) // 4):
                wood_boundary_surfs = exterior
            print(
                f"[diag] display exterior shell: kept {len(wood_boundary_surfs)} / {len(all_b)} "
                "boundary facets (outer skin only)"
            )

    air_boundary_surfs = (
        get_boundary_tags([(3, tag) for tag in air_vols], 2) if air_vols else []
    )

    def _classify_shell_facets_for_preview(shell_surfs: list) -> tuple:
        """
        Top plate = flat upward faces in the top Z band only.
        Back plate = flat downward faces in the bottom Z band.
        All remaining exterior facets = ribs/sides (back & sides wood in FEM).
        """
        if not shell_surfs:
            return [], [], []
        zs = [(int(s), get_surface_center_z(s)) for s in shell_surfs]
        zmax = max(z for _, z in zs)
        zmin = min(z for _, z in zs)
        dz = max(zmax - zmin, 1e-6)
        top_z = zmax - 0.08 * dz
        back_z = zmin + 0.08 * dz

        top: list = []
        back: list = []
        ribs: list = []
        for sid, z in zs:
            nz = get_surface_normal_signed_z(sid)
            if nz is not None and nz > 0.72 and z >= top_z:
                top.append(sid)
            elif nz is not None and nz < -0.72 and z <= back_z:
                back.append(sid)
            else:
                ribs.append(sid)

        if not top:
            top = [int(max(shell_surfs, key=lambda s: get_surface_center_z(s)))]
        if not back:
            back = [int(min(shell_surfs, key=lambda s: get_surface_center_z(s)))]
        top_set = set(top)
        back_set = set(back)
        ribs = [int(s) for s in shell_surfs if int(s) not in top_set and int(s) not in back_set]
        print(
            f"[diag] preview facet classify: top={len(top)} back={len(back)} ribs={len(ribs)} "
            f"(shell n={len(shell_surfs)})"
        )
        return top, back, ribs

    if shell_only:
        top_plate_surfs, back_plate_surfs, rib_surfs = _classify_shell_facets_for_preview(
            wood_boundary_surfs
        )
    else:
        wood_vol_set = {int(v) for v in wood_vols}
        top_plate_surfs = (
            sorted(list(get_boundary_tags([(3, tag) for tag in top_vols], 2))) if top_vols else []
        )
        top_plate_surfs = _drop_wood_partition_interfaces(top_plate_surfs, wood_vol_set)
        if not top_plate_surfs and wood_boundary_surfs:
            highest = max(list(wood_boundary_surfs), key=lambda s: get_surface_center_z(s))
            top_plate_surfs = [highest]
            print("[diag][warn] top_plate_surfs fallback: using highest wood boundary surface.")

        back_plate_surfs = (
            sorted(list(get_boundary_tags([(3, int(v)) for v in back_vols], 2))) if back_vols else []
        )
        back_plate_surfs = _drop_wood_partition_interfaces(back_plate_surfs, wood_vol_set)
        rib_surfs = (
            sorted(list(get_boundary_tags([(3, int(v)) for v in rib_vols], 2))) if rib_vols else []
        )
        rib_surfs = _drop_wood_partition_interfaces(rib_surfs, wood_vol_set)
        if not back_plate_surfs and not rib_surfs:
            legacy_body = sorted(list(set(wood_boundary_surfs) - set(top_plate_surfs)))
            if legacy_body:
                rib_surfs = legacy_body
                print("[diag][warn] back/rib volume boundaries empty; legacy body shell → ribs tag 4.")

    def _xy_dist_point_to_rect(px, py, xmin, xmax, ymin, ymax):
        if px < xmin:
            qx = xmin
        elif px > xmax:
            qx = xmax
        else:
            qx = px
        if py < ymin:
            qy = ymin
        elif py > ymax:
            qy = ymax
        else:
            qy = py
        return math.hypot(px - qx, py - qy)

    def _select_soundhole_surfaces(shell_tags, z_plane, z_tol, hx, hy, hole_r):
        """
        Surfaces on the top opening after the soundhole boolean (tag 2 for FEM).

        No radius clamp — large cylinders clip at the spline boundary via OCC.
        Span filter scales with body size so edge-bite holes stay selectable.
        """
        scored = []
        body_scale = max(float(W), float(L), 1.0e-3)
        span_limit = max(6.0 * hole_r, 0.35 * body_scale)
        for s in sorted(shell_tags):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, int(s))
            if zmax < z_plane - z_tol or zmin > z_plane + z_tol:
                continue
            dxy = _xy_dist_point_to_rect(hx, hy, xmin, xmax, ymin, ymax)
            if dxy > hole_r * 1.25:
                continue
            span_xy = max(xmax - xmin, ymax - ymin)
            if span_xy > span_limit:
                continue
            nz = get_surface_normal_z(s)
            nz_abs = float(nz) if nz is not None else 0.0
            scored.append((int(s), nz_abs))
        scored.sort(key=lambda row: -row[1])
        return [row[0] for row in scored]

    # Soundhole: z = D/2 plane + hole disk on exterior shell.
    z_top_outer = D / 2.0
    z_tol = max(1.0e-4, t, 0.25 * hr)
    if is_preview:
        all_shell_surfs = sorted(wood_boundary_surfs)
    else:
        all_shell_surfs = sorted(set(wood_boundary_surfs) | set(air_boundary_surfs))
    soundhole_surfs = _select_soundhole_surfaces(
        all_shell_surfs, z_top_outer, z_tol, hole_x, hole_y, hr
    )
    if not soundhole_surfs and air_boundary_surfs:
        soundhole_surfs = _select_soundhole_surfaces(
            sorted(air_boundary_surfs), z_top_outer, z_tol, hole_x, hole_y, hr
        )
    # Do not assign whole top plate or raw air shell to tag 2 (causes jumps / wrong BCs).

    # Do not double-tag hole annulus; keep top / back / ribs disjoint.
    _sh_set = set(soundhole_surfs)
    _top_set = set(top_plate_surfs)
    if _sh_set:
        top_plate_surfs = [s for s in top_plate_surfs if s not in _sh_set]
        back_plate_surfs = [s for s in back_plate_surfs if s not in _sh_set and s not in _top_set]
        rib_surfs = [s for s in rib_surfs if s not in _sh_set and s not in _top_set]
        if shell_only and not rib_surfs:
            rib_surfs = [
                int(s)
                for s in wood_boundary_surfs
                if int(s) not in _top_set
                and int(s) not in set(int(x) for x in back_plate_surfs)
                and int(s) not in _sh_set
            ]
    _back_set = set(back_plate_surfs)
    rib_surfs = [s for s in rib_surfs if s not in _back_set]
    if is_fom and not back_plate_surfs and back_vols:
        b_back = get_boundary_tags([(3, int(v)) for v in back_vols], 2)
        if b_back:
            back_plate_surfs = [min(b_back, key=lambda s: get_surface_center_z(s))]
            print("[diag][warn] back_plate_surfs fallback: lowest-Z back volume face.")

    # Geometry fallback when Z-partition yields only top+back volumes (no rib volume):
    # assign remaining exterior wood shell to back (lowest Z / outward -Z) and ribs (rest).
    _claimed = set(soundhole_surfs) | set(top_plate_surfs) | set(back_plate_surfs) | set(rib_surfs)
    _pool = sorted(set(wood_boundary_surfs) - _claimed)
    if not back_plate_surfs and _pool:
        back_plate_surfs = [
            min(
                _pool,
                key=lambda s: (
                    get_surface_center_z(s),
                    -(get_surface_normal_z(s) or 0.0),
                ),
            )
        ]
        _claimed |= set(back_plate_surfs)
        _pool = sorted(set(wood_boundary_surfs) - _claimed)
        print("[diag][warn] back_plate_surfs geometry fallback from wood shell pool.")
    if not rib_surfs:
        rib_surfs = sorted(set(wood_boundary_surfs) - _claimed)
        if rib_surfs:
            print(f"[diag][warn] rib_surfs geometry fallback: {len(rib_surfs)} shell surfaces → tag 4.")

    # Optional neck patch (tag 5); ribs (tag 4) are clamped in FEM, not wood_fix.
    wood_fix_surfs: list = []
    if is_fom:
        fix_candidates = []
        for s in rib_surfs + back_plate_surfs:
            cx, cy, cz = get_surface_center(s)
            fix_candidates.append((s, cx, abs(cy), cz))
        fix_candidates.sort(key=lambda row: (-row[1], row[2]))
        n_fix = min(2, len(fix_candidates))
        wood_fix_surfs = [row[0] for row in fix_candidates[:n_fix]]
        if wood_fix_surfs:
            _fix_set = set(wood_fix_surfs)
            rib_surfs = [s for s in rib_surfs if s not in _fix_set]
            back_plate_surfs = [s for s in back_plate_surfs if s not in _fix_set]

    # Define physical groups using fixed tag protocol.
    # IMPORTANT: 2D tags (facets) and 3D tags (volumes) are both created.
    # This ensures structural-only volume assembly can target wood cells (1/2/3).
    if not top_plate_surfs and wood_boundary_surfs:
        top_plate_surfs = [max(list(wood_boundary_surfs), key=lambda s: get_surface_center_z(s))]
    if not soundhole_surfs:
        soundhole_surfs = _select_soundhole_surfaces(
            all_shell_surfs, z_top_outer, z_tol * 2.5, hole_x, hole_y, hr
        )
    if not soundhole_surfs:
        print(
            "[diag][warn] Soundhole tag 2 empty after boolean clip — "
            "edge-bite hole may share top-plate facets only (FEM uses top opening)."
        )

    def _add_surface_physical_group(surfaces, tag: int, name: str, required: bool = True) -> int:
        if not surfaces:
            if required:
                raise RuntimeError(
                    f"Facet physical group {tag} ({name}) has no surfaces; "
                    "FEM cannot apply shell BCs/material regions."
                )
            print(f"[diag][warn] Facet physical group {tag} ({name}) skipped (empty).")
            return -1
        pg = gmsh.model.addPhysicalGroup(2, surfaces, tag=tag)
        gmsh.model.setPhysicalName(2, pg, name)
        return int(pg)

    # 2D facet protocol (must match fem_main_3d WOOD_SURFACE_TAGS): 1=Top, 2=Soundhole, 3=Back, 4=Ribs, 5=wood_fix
    if shell_only:
        _hole_pre = set(int(s) for s in soundhole_surfs)
        _top_set_pre = set(int(s) for s in top_plate_surfs)
        _back_set_pre = set(int(s) for s in back_plate_surfs)
        if not back_plate_surfs:
            pool_b = [
                int(s)
                for s in wood_boundary_surfs
                if int(s) not in _top_set_pre and int(s) not in _hole_pre
            ]
            if pool_b:
                back_plate_surfs = [min(pool_b, key=lambda s: get_surface_center_z(s))]
                _back_set_pre = set(back_plate_surfs)
        if not rib_surfs:
            rib_surfs = [
                int(s)
                for s in wood_boundary_surfs
                if int(s) not in _top_set_pre
                and int(s) not in _back_set_pre
                and int(s) not in _hole_pre
            ]
        if not rib_surfs and wood_boundary_surfs:
            side_scored = sorted(
                [
                    (
                        int(s),
                        abs(get_surface_normal_signed_z(int(s)) or 0.0),
                    )
                    for s in wood_boundary_surfs
                    if int(s) not in _top_set_pre and int(s) not in _hole_pre
                ],
                key=lambda row: row[1],
            )
            rib_surfs = [row[0] for row in side_scored[: max(4, len(side_scored) // 3)]]
        if not rib_surfs and wood_boundary_surfs:
            rib_surfs = sorted(
                set(int(s) for s in wood_boundary_surfs)
                - set(int(s) for s in top_plate_surfs)
                - set(int(s) for s in back_plate_surfs)
                - set(int(s) for s in soundhole_surfs)
            )
            if rib_surfs:
                print(f"[diag][warn] rib_surfs fallback from solid shell: {len(rib_surfs)} facets → tag 4.")

    req_back = is_fom
    req_ribs = True
    _add_surface_physical_group(top_plate_surfs, 1, "Top_Plate", required=True)
    _add_surface_physical_group(soundhole_surfs, 2, "Soundhole", required=False)
    _add_surface_physical_group(back_plate_surfs, 3, "Back_Plate", required=req_back)
    _add_surface_physical_group(rib_surfs, 4, "Ribs_Sides", required=req_ribs)
    if wood_fix_surfs:
        _add_surface_physical_group(wood_fix_surfs, 5, "wood_fix", required=False)
    else:
        print("[diag][warn] wood_fix surfaces empty; tag 5 not created.")
    if top_vols:
        pg_top_v = gmsh.model.addPhysicalGroup(3, top_vols, tag=1)
        gmsh.model.setPhysicalName(3, pg_top_v, "Top_Plate_Volume")
    else:
        print("[diag][warn] Physical Volume 1 (Top_Plate_Volume) is empty.")
    if back_vols:
        pg_back_v = gmsh.model.addPhysicalGroup(3, back_vols, tag=2)
        gmsh.model.setPhysicalName(3, pg_back_v, "Back_Plate_Volume")
    else:
        print("[diag][warn] Physical Volume 2 (Back_Plate_Volume) is empty.")
    if rib_vols:
        pg_rib_v = gmsh.model.addPhysicalGroup(3, rib_vols, tag=3)
        gmsh.model.setPhysicalName(3, pg_rib_v, "Ribs_Sides_Volume")
    else:
        print("[diag][warn] Physical Volume 3 (Ribs_Sides_Volume) is empty.")

    if is_fom:
        pg_air = gmsh.model.addPhysicalGroup(3, air_vols, tag=10)
        gmsh.model.setPhysicalName(3, pg_air, "Air_Internal")

    print(
        f"[diag] facet groups: top={len(top_plate_surfs)}, back={len(back_plate_surfs)}, "
        f"ribs={len(rib_surfs)}, wood_fix={len(wood_fix_surfs)}, mode={mode}"
    )
    if is_fom:
        air_group_entities = gmsh.model.getEntitiesForPhysicalGroup(3, 10)
        print(f"[diag] Physical Group 10 (Air_Internal) volume entities: {list(air_group_entities)}")
        if len(air_group_entities) == 0:
            raise RuntimeError("Tag 10 was created but has no 3D volume entities.")
    else:
        print("[diag] preview mesh: no Air_Internal (tag 10) — outer boundary is wood skin only")
    v1 = gmsh.model.getEntitiesForPhysicalGroup(3, 1) if top_vols else []
    v2 = gmsh.model.getEntitiesForPhysicalGroup(3, 2) if back_vols else []
    v3 = gmsh.model.getEntitiesForPhysicalGroup(3, 3) if rib_vols else []
    print(f"[diag] Physical Group 1 (Top_Plate_Volume) entities: {list(v1)}")
    print(f"[diag] Physical Volume 2 (Back_Plate_Volume) entities: {list(v2)}")
    print(f"[diag] Physical Group 3 (Ribs_Sides_Volume) entities: {list(v3)}")
    sh2: list = []
    try:
        sh2 = list(gmsh.model.getEntitiesForPhysicalGroup(2, 2))
        print(f"[diag] Physical Surface Group 2 (Soundhole) CAD surface tags: {sh2}")
    except Exception:
        print(
            "[diag][warn] Physical Surface 2 (Soundhole) not created — "
            "edge-clipped hole uses top-plate opening only."
        )
    if len(sh2) == 0:
        print("[diag][warn] Tag 2 (Soundhole) has no 2D entities — acceptable for edge-clipped holes.")

    # Configure balanced meshing options.
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    # Keep characteristic length controls enabled so lc/global sizing is effective.
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    # Explicitly enable point/boundary-based sizing controls (requested).
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    print(
        f"[diag] Gmsh size controls (meters): target_lc={mesh_size:.6f}, "
        f"MeshSizeMin={mesh_size_min:.6f}, MeshSizeMax={mesh_size_max:.6f}"
    )
    
    if shell_only:
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 8 if is_display else 12)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 36)
    
    gmsh.model.mesh.setOrder(1)
    mesh_resolution_factor = 1.0
    print(f"[diag] mesh_resolution_factor={mesh_resolution_factor}")

    if is_display:
        shell_surf_tags = sorted(
            set(int(s) for s in top_plate_surfs + back_plate_surfs + rib_surfs + soundhole_surfs)
        )
        for s in shell_surf_tags:
            try:
                gmsh.model.mesh.setSize(2, int(s), mesh_size)
            except Exception:
                pass
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        print(
            f"[diag] display shell sizing: uniform_lc={mesh_size*1000:.1f}mm, "
            f"n_surfaces={len(shell_surf_tags)} (no FSI fields)"
        )
    elif is_preview:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        shell_surf_tags = sorted(
            set(int(s) for s in top_plate_surfs + back_plate_surfs + rib_surfs + soundhole_surfs)
        )
        for s in shell_surf_tags:
            try:
                gmsh.model.mesh.setSize(2, int(s), mesh_size)
            except Exception:
                pass
        print(
            f"[diag] preview shell sizing: uniform_lc={mesh_size*1000:.1f}mm (wood_surface_lc), "
            f"min=max={wood_surface_size*1000:.1f}mm, n_surfaces={len(shell_surf_tags)} "
            "(no curvature gradient — solid sketch volume)"
        )

    # Deep-probe overrides (display/FOM): uniform shell sizing must not fight background fields.
    if not is_preview:
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

    # Wood: 6.5 mm baseline on shell (Restrict), 1 mm in a band around short thickness edges
    # (mesh.setSize + Threshold from EdgesList) so P1 has multiple elements across ~3 mm wood.
    # Air: distance from wood shell -> Threshold (8 mm near field, 80 mm far, smooth 1.5–25 cm band).
    if is_fom:
        wood_surface_tags = top_plate_surfs + back_plate_surfs + rib_surfs
        if not top_plate_surfs:
            raise RuntimeError("top_plate_surfs is empty; cannot apply wood refinement field.")
        for surf in wood_surface_tags:
            pts = get_point_tags_from_surface(surf)
            print(f"[DEBUG] Refining surface {surf}: found {len(pts)} boundary points")

        if wood_surface_tags:
            print(f"[DEBUG] All Wood Surfaces to refine: {wood_surface_tags}")
            # Geometry constraints: large plate faces + wood boundary curves (short = thickness).
            for s in list(dict.fromkeys(top_plate_surfs + back_plate_surfs)):
                try:
                    gmsh.model.mesh.setSize(2, int(s), wood_surface_size)
                except Exception:
                    pass
            wood_curve_tags = _wood_boundary_curve_tags(wood_vols)
            thickness_edge_tags = []
            for ctag in wood_curve_tags:
                Lc = _curve_length_m(ctag)
                try:
                    if Lc < thickness_curve_len_max:
                        gmsh.model.mesh.setSize(1, ctag, wood_thickness_size)
                        thickness_edge_tags.append(ctag)
                    else:
                        gmsh.model.mesh.setSize(1, ctag, wood_surface_size)
                except Exception:
                    pass
            print(
                f"[diag] Wood boundary curves: n={len(wood_curve_tags)}, "
                f"thickness_edges (L<{thickness_curve_len_max*1000:.1f}mm): {len(thickness_edge_tags)}"
            )

            # Field 1: distance from wood surfaces (shell used for air gradient).
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "FacesList", wood_surface_tags)

            # Wood baseline: coarse characteristic length on wood domain (top/sides/back shell).
            fine_const = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(fine_const, "F", str(wood_surface_size))
            fine_restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(fine_restrict, "InField", fine_const)
            gmsh.model.mesh.field.setNumbers(fine_restrict, "FacesList", wood_surface_tags)
            gmsh.model.mesh.field.setNumbers(fine_restrict, "VolumesList", wood_vols)

            # Thickness: refine from short-edge curve set (works with background mesh when Distance is used).
            thick_thresh = None
            if thickness_edge_tags:
                dist_thick = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_thick, "EdgesList", thickness_edge_tags)
                thick_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(thick_thresh, "InField", dist_thick)
                gmsh.model.mesh.field.setNumber(thick_thresh, "DistMin", thickness_threshold_dist_min)
                gmsh.model.mesh.field.setNumber(thick_thresh, "DistMax", thickness_threshold_dist_max)
                gmsh.model.mesh.field.setNumber(thick_thresh, "SizeMin", wood_thickness_size)
                gmsh.model.mesh.field.setNumber(thick_thresh, "SizeMax", wood_surface_size)

            # Air: distance-based characteristic length (Gmsh Threshold = linear between distances).
            # I < DistMin -> SizeMin; I > DistMax -> SizeMax; else linear in I.
            air_grad = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(air_grad, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(air_grad, "DistMin", air_threshold_dist_min)
            gmsh.model.mesh.field.setNumber(air_grad, "DistMax", air_threshold_dist_max)
            gmsh.model.mesh.field.setNumber(air_grad, "SizeMin", air_threshold_size_min)
            gmsh.model.mesh.field.setNumber(air_grad, "SizeMax", air_threshold_size_max)

            combine_list = [air_grad, fine_restrict]
            if thick_thresh is not None:
                combine_list.append(thick_thresh)
            # Soundhole annulus: extra refinement (engineering only) for circular opening.
            hole_lc_target = max(wood_thickness_size, min(0.002, hr / 6.0))
            if soundhole_surfs:
                for s in soundhole_surfs:
                    try:
                        gmsh.model.mesh.setSize(2, int(s), hole_lc_target)
                    except Exception:
                        pass
                dist_hole = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_hole, "FacesList", [int(s) for s in soundhole_surfs])
                hole_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(hole_thresh, "InField", dist_hole)
                gmsh.model.mesh.field.setNumber(hole_thresh, "DistMin", 0.0005)
                gmsh.model.mesh.field.setNumber(hole_thresh, "DistMax", max(0.015, 2.5 * hr))
                gmsh.model.mesh.field.setNumber(hole_thresh, "SizeMin", hole_lc_target)
                gmsh.model.mesh.field.setNumber(hole_thresh, "SizeMax", air_threshold_size_min)
                combine_list.append(hole_thresh)
                print(
                    f"[diag] engineering soundhole refine: lc={hole_lc_target*1000:.2f}mm "
                    f"over d=0.5–{max(0.015, 2.5 * hr)*1000:.1f}mm, n_faces={len(soundhole_surfs)}"
                )

            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", combine_list)
            print(
                "[diag] Golden mesh fields: "
                f"wood restrict lc={wood_surface_size*1000:.1f}mm; "
                f"thickness Threshold {wood_thickness_size*1000:.1f}–{wood_surface_size*1000:.1f}mm "
                f"over d={thickness_threshold_dist_min*1000:.1f}–{thickness_threshold_dist_max*1000:.1f}mm; "
                f"air Threshold {air_threshold_size_min*1000:.0f}–{air_threshold_size_max*1000:.0f}mm "
                f"over d={air_threshold_dist_min*100:.0f}–{air_threshold_dist_max*100:.0f}cm; "
                f"n_wood_surfaces={len(wood_surface_tags)}."
            )
            # Keep this as the very last field instruction before mesh generation.
            try:
                gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
            except Exception as exc:
                raise RuntimeError(f"Failed to set triple-tier Min background field: {exc}")

    try:
        print(
            f"Generating optimized mesh (target={mesh_size*1000:.2f}mm, "
            f"min={mesh_size_min*1000:.2f}mm, max={mesh_size_max*1000:.2f}mm, "
            f"wall={t*1000:.2f}mm | raw_meters: lc={mesh_size:.6f}, "
            f"min={mesh_size_min:.6f}, max={mesh_size_max:.6f})..."
        )
        if shell_only:
            gmsh.model.mesh.generate(2)
        else:
            gmsh.model.mesh.generate(3)

        def _audit_soundhole_boundary_mesh():
            """Edge-length stats on curves bounding soundhole tag-2 surfaces."""
            if not audit_enabled():
                return
            edge_lens: List[float] = []
            n_tri = 0
            n_line = 0
            try:
                sh_entities = gmsh.model.getEntitiesForPhysicalGroup(2, 2)
            except Exception as exc:
                print(f"[AUDIT][warn] Soundhole physical group 2 unavailable: {exc}")
                return
            curve_tags: set = set()
            for ent in sh_entities:
                dim_e = int(ent[0]) if isinstance(ent, (list, tuple)) else 2
                tag_e = int(ent[1]) if isinstance(ent, (list, tuple)) else int(ent)
                bnd = gmsh.model.getBoundary([(dim_e, tag_e)], oriented=False, recursive=True)
                for bdim, btag in bnd:
                    if int(bdim) == 1:
                        curve_tags.add(int(btag))
                etypes, etags, _nodes = gmsh.model.mesh.getElements(dim_e, tag_e)
                for et, arr in zip(etypes, etags):
                    if et == 2:
                        n_tri += int(len(arr))
                    elif et == 1:
                        n_line += int(len(arr))
            for ctag in sorted(curve_tags):
                try:
                    etypes, etags, node_tags = gmsh.model.mesh.getElements(1, ctag)
                except Exception:
                    continue
                for et, e_arr, n_arr in zip(etypes, etags, node_tags):
                    if et != 1:
                        continue
                    coords = gmsh.model.mesh.getNode(int(n_arr[0]))[1]
                    coords2 = gmsh.model.mesh.getNode(int(n_arr[1]))[1]
                    edge_lens.append(
                        math.dist(
                            (coords[0], coords[1], coords[2]),
                            (coords2[0], coords2[1], coords2[2]),
                        )
                    )
            if edge_lens:
                print(
                    "[AUDIT] Soundhole boundary mesh: "
                    f"n_curves={len(curve_tags)} n_line_elems={n_line} n_tri_facets={n_tri} "
                    f"edge_len_min={min(edge_lens):.6e} max={max(edge_lens):.6e} "
                    f"mean={sum(edge_lens)/len(edge_lens):.6e} (m)"
                )
            else:
                print(
                    f"[AUDIT][warn] Soundhole boundary: no 1D edge lengths "
                    f"(tag2_surfaces={len(list(sh_entities))}, n_tri={n_tri})"
                )

        _audit_soundhole_boundary_mesh()

        def _count_mesh_elements_for_physical(dim: int, phys_tag: int) -> int:
            n_elem = 0
            try:
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, int(phys_tag))
            except Exception:
                return 0
            for ent in entities:
                if isinstance(ent, (list, tuple)) and len(ent) >= 2:
                    dim_e, tag_e = int(ent[0]), int(ent[1])
                else:
                    dim_e, tag_e = dim, int(ent)
                _types, elem_tags, _nodes = gmsh.model.mesh.getElements(dim_e, tag_e)
                for arr in elem_tags:
                    n_elem += int(len(arr))
            return n_elem

        audit_tags = [1, 2, 3, 4]
        if shell_only:
            audit_tags = [1]
            if back_plate_surfs:
                audit_tags.append(3)
            if rib_surfs:
                audit_tags.append(4)
        facet_audit = {t: _count_mesh_elements_for_physical(2, t) for t in audit_tags}
        print(f"[diag] post-generate facet element counts by physical tag: {facet_audit}")
        req_checks = [(1, "Top"), (3, "Back"), (4, "Ribs")]
        if shell_only:
            req_checks = [(1, "Top")]
            if back_plate_surfs:
                req_checks.append((3, "Back"))
            if rib_surfs:
                req_checks.append((4, "Ribs"))
        for req_tag, label in req_checks:
            if facet_audit.get(req_tag, 0) <= 0:
                raise RuntimeError(
                    f"Mesh has 0 triangle elements on facet physical tag {req_tag} ({label}). "
                    "Check shell surface classification and addPhysicalGroup assignments."
                )

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        # Count mesh facets (2D elements) on Soundhole physical surfaces for downstream BC sanity.
        n_soundhole_mesh_facets = 0
        try:
            for ent in gmsh.model.getEntitiesForPhysicalGroup(2, 2):
                if isinstance(ent, (list, tuple)) and len(ent) >= 2:
                    dim_e, tag_e = int(ent[0]), int(ent[1])
                else:
                    dim_e, tag_e = 2, int(ent)
                _types, elem_tags, _nodes = gmsh.model.mesh.getElements(dim_e, tag_e)
                for arr in elem_tags:
                    n_soundhole_mesh_facets += int(len(arr))
        except Exception as _exc:
            print(f"[diag][warn] Soundhole mesh facet count failed: {_exc}")
        print(f"PRINT: Found {n_soundhole_mesh_facets} facets for Soundhole")
        gmsh.write(str(out_file))
        print(f"SUCCESS: Optimized mesh saved to {out_file}")
    except Exception as e:
        print(f"Mesh generation failed: {e}", file=sys.stderr)
        gmsh.finalize()
        raise SystemExit(1) from e

    gmsh.finalize()

if __name__ == "__main__":
    try:
        create_guitar_mesh()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc