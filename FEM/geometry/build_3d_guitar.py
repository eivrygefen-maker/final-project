import gmsh
import sys
import json
import os
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# CAD reference injection: closed STEP/B-rep bodies in FEM/geometry/models/
_MODELS_DIR = Path(__file__).resolve().parent / "models"


def _reference_step_filename(shape_type: str) -> str:
    """Map simulator ``shape_type`` → ``FEM/geometry/models/*.step``."""
    st = str(shape_type).strip().lower()
    if st in ("box", "rect", "rectangular"):
        return "box.step"
    if st in ("acoustic", "dreadnought", "dread", "martin"):
        return "acoustic.step"
    if st in ("classical", "classic", "torres", "hauser"):
        return "classic.step"
    if "box" in st or "rect" in st:
        return "box.step"
    if "dread" in st or "acoustic" in st or "martin" in st:
        return "acoustic.step"
    return "classic.step"


def _load_reference_model(shape_type: str) -> List[Tuple[int, int]]:
    """Import a closed reference solid from ``FEM/geometry/models/*.step``."""
    path = _MODELS_DIR / _reference_step_filename(shape_type)
    if not path.is_file():
        raise FileNotFoundError(
            f"CAD reference not found: {path}\n"
            "Run: python3 FEM/geometry/generate_reference_models.py\n"
            "See FEM/geometry/models/README.md"
        )
    occ = gmsh.model.occ
    imported = occ.importShapes(str(path))
    occ.synchronize()
    vol_dimtags = [(int(d), int(t)) for d, t in imported if int(d) == 3]
    if not vol_dimtags:
        vol_dimtags = [(int(d), int(t)) for d, t in occ.getEntities(3)]
    if not vol_dimtags:
        raise RuntimeError(f"importShapes({path.name}) produced no 3D volumes.")
    print(
        f"[diag] CAD reference injection: {path.name} "
        f"volumes={[t for _, t in vol_dimtags]}"
    )
    return vol_dimtags


def _bbox_dimtags(dimtags: Sequence[Tuple[int, int]]) -> Tuple[float, float, float, float, float, float]:
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for dim, tag in dimtags:
        bb = gmsh.model.getBoundingBox(int(dim), int(tag))
        xmin = min(xmin, float(bb[0]))
        ymin = min(ymin, float(bb[1]))
        zmin = min(zmin, float(bb[2]))
        xmax = max(xmax, float(bb[3]))
        ymax = max(ymax, float(bb[4]))
        zmax = max(zmax, float(bb[5]))
    return xmin, ymin, zmin, xmax, ymax, zmax


def _reference_shape_family(shape_type: str) -> str:
    st = str(shape_type).strip().lower()
    if "dread" in st or "acoustic" in st or "martin" in st:
        return "acoustic"
    return "classical"


def _reference_nominal_lateral(shape_type: str) -> Tuple[float, float, float]:
    """Nominal full widths (m) baked into STEP templates — classical vs acoustic differ."""
    try:
        from FEM.geometry.generate_reference_models import REFERENCE_NOMINAL_WIDTHS
    except ImportError:
        from generate_reference_models import REFERENCE_NOMINAL_WIDTHS

    key = _reference_shape_family(shape_type)
    return REFERENCE_NOMINAL_WIDTHS[key]  # upper, waist, lower


def _scale_reference_to_target(
    vol_dimtags: Sequence[Tuple[int, int]],
    *,
    length: float,
    width: float,
    depth: float,
    upper_bout: float,
    waist: float,
    lower_bout: float,
    shape_type: str,
) -> None:
    """
    Morph reference STEP to simulator size without erasing classical vs dreadnought ratios.

    - ``sx`` from target length only (not from global ``width``).
    - ``sy`` from bout parameters vs shape-specific STEP nominals (never ``W / ly``).
    - ``sz`` from target depth.
    """
    occ = gmsh.model.occ
    dimtags = [(int(d), int(t)) for d, t in vol_dimtags]
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox_dimtags(dimtags)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    lx = max(float(xmax - xmin), 1.0e-9)
    lz = max(float(zmax - zmin), 1.0e-9)

    L, D = float(length), float(depth)
    nom_shoulder, nom_waist, nom_lower = _reference_nominal_lateral(shape_type)

    def _full_width(val: float, nominal: float) -> float:
        v = float(val)
        if v <= 0.0:
            return nominal
        return v if v > 0.22 else 2.0 * v

    tgt_lower = _full_width(lower_bout, nom_lower)
    tgt_waist = _full_width(waist, nom_waist)
    tgt_upper = _full_width(upper_bout, nom_shoulder)

    sx = L / lx
    sz = D / lz
    sy_lower = tgt_lower / nom_lower
    sy_waist = tgt_waist / nom_waist
    sy_upper = tgt_upper / nom_shoulder
    # Min scale avoids lateral “ballooning” past any target bout.
    sy = min(sy_lower, sy_waist, sy_upper)

    occ.dilate(dimtags, cx, cy, cz, sx, sy, sz)
    occ.synchronize()

    xmin, ymin, zmin, xmax, ymax, zmax = _bbox_dimtags(dimtags)
    mx = 0.5 * (xmin + xmax)
    my = 0.5 * (ymin + ymax)
    mz = 0.5 * (zmin + zmax)
    occ.translate(dimtags, -mx, -my, -mz)
    occ.synchronize()

    xmin, ymin, zmin, xmax, ymax, zmax = _bbox_dimtags(dimtags)
    dx = 0.5 * L - float(xmax)
    occ.translate(dimtags, dx, 0.0, 0.0)
    occ.synchronize()
    print(
        f"[diag] reference morph ({_reference_shape_family(shape_type)}): "
        f"L={L:.4f} D={D:.4f} scale=({sx:.4f},{sy:.4f},{sz:.4f}) "
        f"(sy from bouts, not W={width:.4f}) neck_x=+{0.5*L:.4f} "
        f"targets lower/waist/upper={tgt_lower:.3f}/{tgt_waist:.3f}/{tgt_upper:.3f}"
    )


def _hollow_inner_volume(
    outer_vol: int,
    *,
    wall_t: float,
    inner_depth: float,
    full_z_gap: bool = False,
) -> int:
    """Shrink a copy of the outer reference solid for ``outer − inner`` shell cuts."""
    occ = gmsh.model.occ
    outer_dt = [(3, int(outer_vol))]
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox_dimtags(outer_dt)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    lx = max(float(xmax - xmin), 1.0e-9)
    ly = max(float(ymax - ymin), 1.0e-9)
    lz = max(float(zmax - zmin), 1.0e-9)
    sx = max(0.02, (lx - 2.0 * wall_t) / lx)
    sy = max(0.02, (ly - 2.0 * wall_t) / ly)
    if full_z_gap:
        sz = max(0.02, (lz - 2.0 * wall_t) / lz)
    else:
        sz = max(0.02, float(inner_depth) / lz)

    copied = occ.copy(outer_dt)
    inner_dimtags = [(int(d), int(t)) for d, t in copied]
    occ.dilate(inner_dimtags, cx, cy, cz, sx, sy, sz)
    occ.synchronize()
    inner_tags = [t for d, t in inner_dimtags if int(d) == 3]
    if not inner_tags:
        raise RuntimeError("Hollow inner copy produced no volume.")
    print(
        f"[diag] reference hollow tool: outer={outer_vol} inner={inner_tags[0]} "
        f"shrink=({sx:.4f},{sy:.4f},{sz:.4f})"
    )
    return int(inner_tags[0])


def _vol_center_z(vol_tag: int) -> float:
    try:
        com = gmsh.model.occ.getCenterOfMass(3, int(vol_tag))
        return float(com[2])
    except Exception:
        bb = gmsh.model.getBoundingBox(3, int(vol_tag))
        return 0.5 * (float(bb[2]) + float(bb[5]))


def _remove_occ_volumes(vol_tags: Sequence[int], occ) -> None:
    """Delete auxiliary OCC volumes (e.g. Boolean tool slabs) from the model."""
    for tg in vol_tags:
        try:
            occ.remove([(3, int(tg))], recursive=True)
        except Exception:
            pass


def _fragment_display_shell_seams(occ, wood_dimtags: list) -> list:
    """Fragment wood shell so plate–rib interfaces share edges (faces + seam curves imprinted)."""
    vols = [(3, int(tag)) for dim, tag in wood_dimtags if int(dim) == 3]
    if not vols:
        return wood_dimtags
    try:
        occ.synchronize()
        iface_surfs: list = []
        iface_curves: list = []
        for _dim, vtag in vols:
            try:
                bnd = gmsh.model.getBoundary(
                    [(3, int(vtag))], oriented=False, recursive=False
                )
            except Exception:
                continue
            for bdim, stag in bnd:
                if int(bdim) != 2:
                    continue
                stag = int(stag)
                iface_surfs.append((2, stag))
                try:
                    cbnd = gmsh.model.getBoundary(
                        [(2, stag)], oriented=False, recursive=False
                    )
                except Exception:
                    continue
                for cdim, ctag in cbnd:
                    if int(cdim) == 1:
                        iface_curves.append((1, int(ctag)))
        iface_surfs = sorted(set(iface_surfs), key=lambda x: x[1])
        iface_curves = sorted(set(iface_curves), key=lambda x: x[1])
        imprint_tags = list(vols) + iface_surfs + iface_curves
        if len(imprint_tags) > len(vols):
            try:
                occ.imprint(imprint_tags)
                occ.synchronize()
            except Exception as exc:
                print(f"[diag] display shell seam imprint skipped: {exc}")

        frags, _ = occ.fragment(vols, [], removeObject=True, removeTool=False)
        occ.synchronize()
        try:
            occ.removeAllDuplicates()
            gmsh.model.occ.healShapes()
            occ.synchronize()
        except Exception:
            pass
        unified = sorted([int(tag) for dim, tag in frags if dim == 3])
        if unified:
            print(
                f"[diag] display shell seam fragment: {len(vols)} volume(s) → "
                f"{len(unified)} (imprinted {len(iface_surfs)} faces, "
                f"{len(iface_curves)} curves; shared interface edges)"
            )
            return [(3, int(v)) for v in unified]
    except Exception as exc:
        print(f"[diag] display shell seam fragment skipped: {exc}")
    return wood_dimtags


def _heal_display_shell_cad(occ, wood_dimtags: list) -> list:
    """Merge duplicate B-rep faces after booleans (reduces PyVista seam ripple / z-fight)."""
    try:
        occ.removeAllDuplicates()
        occ.synchronize()
    except Exception:
        pass
    try:
        gmsh.model.occ.healShapes()
        occ.synchronize()
    except Exception as exc:
        print(f"[diag] occ.healShapes skipped: {exc}")
    vol_tags = [int(tag) for dim, tag in wood_dimtags if int(dim) == 3]
    if len(vol_tags) > 1:
        frags, _ = occ.fragment(
            [(3, int(v)) for v in vol_tags],
            [],
            removeObject=True,
            removeTool=False,
        )
        occ.synchronize()
        unified = sorted([int(tag) for dim, tag in frags if dim == 3])
        if unified:
            print(
                f"[diag] healed shell fragment: {len(vol_tags)} volumes → "
                f"{len(unified)} (shared interface nodes)"
            )
            return [(3, int(v)) for v in unified]
    return wood_dimtags


def _split_and_fragment_wood_shell_for_conforming_interfaces(
    wood_vols: List[int],
    t: float,
    occ,
) -> List[int]:
    """
    Keep monolithic hollow shell for display/preview.

    Box-based ``occ.fragment`` plate splitting left rectangular tool slabs in the
    model (blocky overlay hiding the soundhole). Interface conformity is handled
    via mesh Distance/Threshold fields on plate–rim curves instead.
    """
    if len(wood_vols) > 1:
        frags, _ = occ.fragment(
            [(3, int(v)) for v in wood_vols],
            [],
            removeObject=True,
            removeTool=False,
        )
        gmsh.model.occ.synchronize()
        unified = sorted([int(tag) for dim, tag in frags if dim == 3])
        print(
            f"[diag] conforming fragment: {len(wood_vols)} volumes → "
            f"{len(unified)} fragmented (shared interface nodes)"
        )
        return unified if unified else wood_vols

    print(
        "[diag] display shell: skipping box-split fragment (avoids OCC tool slabs); "
        "using monolithic hollow shell + soundhole cut + interface mesh fields"
    )
    return wood_vols


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
    env_cfg = os.environ.get("FEM_MESH_CONFIG", "").strip()
    if env_cfg:
        p = Path(env_cfg).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
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
    is_validation = os.environ.get("FEM_VALIDATION_MESH", "0") == "1"
    is_fom = os.environ.get("FEM_ALLOW_FOM", "0") == "1" or is_validation
    is_preview = (
        (os.environ.get("FEM_ALLOW_PREVIEW", "0") == "1" or preview_cli)
        and not is_display
        and not is_fom
    )
    shell_only = is_preview or is_display
    use_air_opening_geom = is_validation or os.environ.get(
        "FEM_SOUNDHOLE_TAG_AIR_OPENING", "0"
    ) == "1"

    if is_display:
        out_file = mesh_dir / "display_mesh.msh"
    elif is_preview:
        out_file = mesh_dir / "preview_mesh.msh"
    elif is_validation:
        out_env = os.environ.get("FEM_MESH_OUT", "").strip()
        if not out_env:
            raise RuntimeError("FEM_VALIDATION_MESH=1 requires FEM_MESH_OUT=<path to .msh>")
        out_file = Path(out_env)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[diag] validation mesh output: {out_file}")
    elif is_fom:
        out_file = mesh_dir / "guitar_3d.msh"
    else:
        raise RuntimeError(
            "Set exactly one mesh mode env var: FEM_ALLOW_PREVIEW, FEM_ALLOW_DISPLAY, "
            "FEM_ALLOW_FOM, or FEM_VALIDATION_MESH."
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
        hr_raw = float(p["hole_radius"])
        hr_cap = 0.25 * min(float(L), float(W))
        hr = min(hr_raw, hr_cap, 0.08)
        shape_type = str(p.get("shape_type", "Classical")).strip()
        upper_bout = float(p.get("upper_bout", W * 0.75))
        lower_bout = float(p.get("lower_bout", W))
        waist = float(p.get("waist", W * 0.65))
        hole_from_neck_ratio = float(p.get("soundhole_from_neck_ratio", 0.5))
        soundhole_x_cfg = p.get("soundhole_x")
    else:
        L, W, D, t, hr, shape_type = 0.48, 0.37, 0.1, 0.003, 0.04, "Classical"
        upper_bout, lower_bout, waist = W * 0.75, W, W * 0.65
        hole_from_neck_ratio = 0.5
        soundhole_x_cfg = None

    def _is_box_shape(st: str) -> bool:
        return str(st).strip().lower() == "box"

    # Feasible wall thickness for hollow-shell booleans (avoids zero-thickness / PLC mesh failures).
    t = max(0.001, min(float(t), max(0.001, 0.45 * float(D))))
    inner_depth = max(1.0e-4, float(D) - 2.0 * t)

    mode = "display" if is_display else ("sketch" if is_preview else "fom")
    print(f"[diag] shape_type={shape_type!r} mesh_mode={mode}")

    # Soundhole centre: luthier blueprint x (neck at +x) or legacy ratio from neck.
    hole_from_neck_ratio = float(max(0.05, min(0.95, hole_from_neck_ratio)))
    if soundhole_x_cfg is not None:
        try:
            from FEM.geometry.generate_reference_models import (
                NOMINAL_LENGTH_ACOUSTIC,
                NOMINAL_LENGTH_CLASSICAL,
            )
        except ImportError:
            from generate_reference_models import (  # type: ignore
                NOMINAL_LENGTH_ACOUSTIC,
                NOMINAL_LENGTH_CLASSICAL,
            )
        fam = _reference_shape_family(shape_type)
        nom_l = (
            NOMINAL_LENGTH_ACOUSTIC
            if fam == "acoustic"
            else NOMINAL_LENGTH_CLASSICAL
        )
        hole_x = float(soundhole_x_cfg) * (float(L) / float(nom_l))
    else:
        hole_x = 0.5 * L - hole_from_neck_ratio * L
    hole_y = 0.0

    # Display validation shell: uniform aesthetic lc only (decoupled from FOM refinement).
    # Preview sketch: coarse global + local zones. FOM: graded wood/air fields.
    DISPLAY_GLOBAL_LC_M = 0.012   # 12 mm coarse display shell (PyVista validation only)
    DISPLAY_SEAM_LC_M = 0.002     # 2 mm nodes at top/back plate perimeters only
    DISPLAY_SEAM_BAND_M = 0.001   # 1 mm Distance band (node alignment at plate seams)
    PREVIEW_GLOBAL_LC_M = 0.012   # 12 mm coarse preview shell
    LOCAL_REFINE_LC_M = 0.001     # 1.0 mm local zones (preview only)
    wood_surface_size = (
        0.014
        if is_validation
        else (0.007 if is_fom else (DISPLAY_GLOBAL_LC_M if is_display else PREVIEW_GLOBAL_LC_M))
    )
    wood_thickness_size = (
        0.003 if is_validation else (0.001 if is_fom else LOCAL_REFINE_LC_M)
    )
    thickness_curve_len_max = 0.005  # curves shorter than this (m) are treated as thickness direction
    thickness_threshold_dist_min = 0.0005
    thickness_threshold_dist_max = 0.008
    soundhole_threshold_dist_min = thickness_threshold_dist_min
    soundhole_threshold_dist_max = 0.012  # 12 mm band from soundhole faces
    if is_validation:
        air_threshold_dist_min = 0.010
        air_threshold_dist_max = 0.12
        air_threshold_size_min = 0.009
        air_threshold_size_max = 0.040
        print(
            "[diag] FEM_VALIDATION_MESH profile: wood_surface=14mm, air_min=9mm, "
            "air_max=40mm, soundhole band=12mm",
            flush=True,
        )
    else:
        air_threshold_dist_min = 0.015
        air_threshold_dist_max = 0.25
        air_threshold_size_min = 0.004 if is_fom else 0.003
        air_threshold_size_max = 0.050

    # Display: uniform 12 mm (visualization only). Preview/FOM: graded fields unchanged.
    if is_display:
        mesh_size = DISPLAY_GLOBAL_LC_M
        mesh_size_min = DISPLAY_GLOBAL_LC_M
        mesh_size_max = DISPLAY_GLOBAL_LC_M
    elif is_preview:
        mesh_size = PREVIEW_GLOBAL_LC_M
        mesh_size_min = LOCAL_REFINE_LC_M
        mesh_size_max = 0.015
    else:
        mesh_size = wood_surface_size
        mesh_size_min = wood_thickness_size
        mesh_size_max = air_threshold_size_max

    print(
        "DEBUG: Display mesh — 12 mm shell + 3 mm seam band; preview uses local zones; "
        "FOM mesh uses graded wood/air fields."
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
    gmsh.option.setNumber("Geometry.Tolerance", 1.0e-4)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    print(
        f"[diag] CAD kernel: gmsh.model.occ reference injection "
        f"(mode={mode}, sketch_solid={is_preview})"
    )

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

    def _occ_volume_mass_m3(vol_tag: int) -> float:
        try:
            mass = gmsh.model.occ.getMass(3, int(vol_tag))
            if isinstance(mass, (int, float)):
                return float(mass)
            if isinstance(mass, (list, tuple)) and len(mass) >= 1:
                return float(mass[0])
        except Exception:
            pass
        return float("nan")

    def _classify_point_in_volume(
        vol_tag: int, x: float, y: float, z: float
    ) -> dict:
        """
        Classify a Cartesian point against a live OCC volume (post-sync).

        Gmsh expects ``gmsh.model.isInside(dim, tag, [x,y,z], parametric=False)``
        and returns the count of input points inside (0 or 1 for a single probe).
        The legacy ``occ.isInside(x,y,z)`` positional form is unreliable here.
        """
        coord = [float(x), float(y), float(z)]
        out: dict = {
            "volume_id": int(vol_tag),
            "point_m": coord,
            "inside": False,
            "method": None,
            "raw": None,
        }
        try:
            n_in = int(gmsh.model.isInside(3, int(vol_tag), coord, False))
            out["method"] = "gmsh.model.isInside"
            out["raw"] = int(n_in)
            out["inside"] = n_in >= 1
            return out
        except Exception as exc_model:
            out["gmsh_model_isInside_error"] = str(exc_model)
        try:
            n_in = int(
                gmsh.model.occ.isInside(3, int(vol_tag), coord, False)
            )
            out["method"] = "gmsh.model.occ.isInside(coord)"
            out["raw"] = int(n_in)
            out["inside"] = n_in >= 1
            return out
        except Exception as exc_occ_list:
            out["gmsh_occ_isInside_coord_error"] = str(exc_occ_list)
        try:
            r = gmsh.model.occ.isInside(
                3, int(vol_tag), float(x), float(y), float(z)
            )
            out["method"] = "gmsh.model.occ.isInside(x,y,z)"
            out["raw"] = r
            out["inside"] = int(r) == 1
            return out
        except Exception as exc_occ_xyz:
            out["gmsh_occ_isInside_xyz_error"] = str(exc_occ_xyz)
        bb = gmsh.model.getBoundingBox(3, int(vol_tag))
        inset = 0.002
        out["method"] = "bbox_inset_fallback"
        out["inside"] = bool(
            float(bb[0]) + inset <= float(x) <= float(bb[3]) - inset
            and float(bb[1]) + inset <= float(y) <= float(bb[4]) - inset
            and float(bb[2]) + inset <= float(z) <= float(bb[5]) - inset
        )
        out["raw"] = out["inside"]
        return out

    def _occ_point_inside_volume(vol_tag: int, x: float, y: float, z: float) -> bool:
        return bool(
            _classify_point_in_volume(int(vol_tag), float(x), float(y), float(z)).get(
                "inside"
            )
        )

    def _write_validation_cad_volume_audit(
        *,
        audit_stem: Path,
        air_vols: list,
        wood_vols: list,
        hole_x: float,
        hole_y: float,
        hole_r: float,
        depth_m: float,
        shell_t: float,
        inner_tool_bb: Optional[list],
    ) -> dict:
        """Pre-mesh validation audit: wood/air volume classification and cavity probes."""
        air_set = {int(v) for v in air_vols}
        wood_set = {int(v) for v in wood_vols}
        all_vol_tags = sorted(int(t) for d, t in gmsh.model.getEntities(3) if int(d) == 3)

        def _union_bbox_live_volumes(vol_tags: list) -> list:
            """Union bbox of live OCC volumes (imported ids are invalid after Booleans)."""
            if not vol_tags:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            xmin = ymin = zmin = float("inf")
            xmax = ymax = zmax = float("-inf")
            for v in vol_tags:
                vx0, vy0, vz0, vx1, vy1, vz1 = gmsh.model.getBoundingBox(3, int(v))
                xmin = min(xmin, float(vx0))
                ymin = min(ymin, float(vy0))
                zmin = min(zmin, float(vz0))
                xmax = max(xmax, float(vx1))
                ymax = max(ymax, float(vy1))
                zmax = max(zmax, float(vz1))
            return [xmin, ymin, zmin, xmax, ymax, zmax]
        vol_records: list = []
        for v in all_vol_tags:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, int(v))
            cx, cy, cz = _entity_center_of_mass(3, int(v))
            if int(v) in air_set:
                classification = "air"
            elif int(v) in wood_set:
                classification = "wood"
            else:
                classification = "unassigned"
            vol_records.append(
                {
                    "entity_id": int(v),
                    "classification": classification,
                    "bbox_m": [
                        float(xmin),
                        float(ymin),
                        float(zmin),
                        float(xmax),
                        float(ymax),
                        float(zmax),
                    ],
                    "center_of_mass_m": [float(cx), float(cy), float(cz)],
                    "volume_m3": float(_occ_volume_mass_m3(int(v))),
                    "z_min_m": float(zmin),
                    "z_max_m": float(zmax),
                }
            )

        outer_bb = _union_bbox_live_volumes(all_vol_tags)
        ox0 = 0.5 * (float(outer_bb[0]) + float(outer_bb[3]))
        oy0 = 0.5 * (float(outer_bb[1]) + float(outer_bb[4]))
        oz0 = 0.5 * (float(outer_bb[2]) + float(outer_bb[5]))
        z_outer_min = float(outer_bb[2])
        z_outer_max = float(outer_bb[5])
        z_inner_top = (
            float(inner_tool_bb[5]) if inner_tool_bb is not None else z_outer_max - shell_t
        )
        z_inner_back = (
            float(inner_tool_bb[2]) if inner_tool_bb is not None else z_outer_min + shell_t
        )
        expected_inner_z_span = max(1.0e-6, z_inner_top - z_inner_back)

        air_bb = (
            list(gmsh.model.getBoundingBox(3, int(air_vols[0])))
            if air_vols
            else [0.0] * 6
        )
        air_boundary = (
            sorted(get_boundary_tags([(3, int(air_vols[0]))], 2)) if air_vols else []
        )
        air_bnd_z = []
        for s in air_boundary:
            _cx, _cy, cz = get_surface_center(int(s))
            air_bnd_z.append(float(cz))
        air_bnd_z_range = (
            [min(air_bnd_z), max(air_bnd_z)] if air_bnd_z else [float("nan"), float("nan")]
        )

        air_com = (
            _entity_center_of_mass(3, int(air_vols[0])) if air_vols else (ox0, oy0, oz0)
        )
        air_bbox_center = (
            (
                0.5 * (float(air_bb[0]) + float(air_bb[3])),
                0.5 * (float(air_bb[1]) + float(air_bb[4])),
                0.5 * (float(air_bb[2]) + float(air_bb[5])),
            )
            if air_vols
            else (ox0, oy0, oz0)
        )
        membership_calibration: dict = {}
        if air_vols:
            av = int(air_vols[0])
            membership_calibration["air_volume_id"] = av
            membership_calibration["air_center_of_mass_m"] = [
                float(air_com[0]),
                float(air_com[1]),
                float(air_com[2]),
            ]
            membership_calibration["air_com_test"] = _classify_point_in_volume(
                av, float(air_com[0]), float(air_com[1]), float(air_com[2])
            )
            membership_calibration["air_bbox_center_test"] = _classify_point_in_volume(
                av,
                float(air_bbox_center[0]),
                float(air_bbox_center[1]),
                float(air_bbox_center[2]),
            )
        if wood_vols:
            wv = int(wood_vols[0])
            wcom = _entity_center_of_mass(3, wv)
            membership_calibration["wood_sample_volume_id"] = wv
            membership_calibration["wood_center_of_mass_m"] = [
                float(wcom[0]),
                float(wcom[1]),
                float(wcom[2]),
            ]
            membership_calibration["wood_com_test"] = _classify_point_in_volume(
                wv, float(wcom[0]), float(wcom[1]), float(wcom[2])
            )
        membership_trusted = bool(
            air_vols
            and membership_calibration.get("air_com_test", {}).get("method")
            == "gmsh.model.isInside"
            and membership_calibration.get("air_com_test", {}).get("inside")
        )

        probe_specs = [
            ("air_volume_com", float(air_com[0]), float(air_com[1]), float(air_com[2])),
            (
                "model_union_bbox_center",
                float(ox0),
                float(oy0),
                float(oz0),
            ),
            (
                "below_top_lid_at_hole",
                float(hole_x),
                float(hole_y),
                float(z_inner_top) - 0.002,
            ),
            (
                "above_back_inner",
                float(ox0),
                float(oy0),
                float(z_inner_back) + 0.002,
            ),
            (
                "cavity_mid_off_hole",
                float(air_bbox_center[0]) + 0.05,
                float(air_bbox_center[1]),
                float(air_bbox_center[2]),
            ),
            (
                "cavity_lower_quarter",
                float(air_bbox_center[0]),
                float(air_bbox_center[1]),
                float(z_inner_back) + 0.25 * expected_inner_z_span,
            ),
            (
                "near_soundhole_top_opening",
                float(hole_x),
                float(hole_y),
                float(z_inner_top) - 0.0005,
            ),
        ]
        probe_records: list = []
        for label, px, py, pz in probe_specs:
            air_tests = [
                _classify_point_in_volume(int(v), px, py, pz) for v in air_vols
            ]
            wood_tests = [
                _classify_point_in_volume(int(v), px, py, pz) for v in wood_vols
            ]
            inside_air = [int(t["volume_id"]) for t in air_tests if t.get("inside")]
            inside_wood = [int(t["volume_id"]) for t in wood_tests if t.get("inside")]
            inside_any = sorted(set(inside_air) | set(inside_wood))
            probe_records.append(
                {
                    "label": label,
                    "point_m": [float(px), float(py), float(pz)],
                    "inside_air_volume_ids": inside_air,
                    "inside_wood_volume_ids": inside_wood,
                    "inside_any_volume_ids": inside_any,
                    "classified_as_air": bool(inside_air),
                    "air_tests": air_tests,
                    "wood_tests": wood_tests,
                }
            )

        air_z_span = float(air_bb[5]) - float(air_bb[2]) if air_vols else 0.0
        fill_ratio = (
            float(air_z_span / expected_inner_z_span)
            if expected_inner_z_span > 0.0
            else 0.0
        )
        top_lid_probe = next(
            (p for p in probe_records if p["label"] == "below_top_lid_at_hole"), None
        )
        air_com_probe = next(
            (p for p in probe_records if p["label"] == "air_volume_com"), None
        )
        opening_probe = next(
            (p for p in probe_records if p["label"] == "near_soundhole_top_opening"),
            None,
        )

        checks = {
            "air_volume_present": bool(air_vols),
            "air_z_span_fill_ratio_ge_0_85": bool(fill_ratio >= 0.85),
            "membership_api_trusted": bool(membership_trusted),
            "air_com_inside_via_isInside": bool(
                membership_calibration.get("air_com_test", {}).get("inside")
            ),
            "air_volume_com_in_air": bool(
                air_com_probe and air_com_probe["classified_as_air"]
            ),
            "below_top_lid_in_air": bool(
                top_lid_probe and top_lid_probe["classified_as_air"]
            ),
            "soundhole_top_opening_in_air": bool(
                opening_probe and opening_probe["classified_as_air"]
            ),
            "air_bbox_reaches_inner_top": bool(
                air_vols and float(air_bb[5]) >= float(z_inner_top) - 0.004
            ),
            "air_boundary_near_inner_top": bool(
                air_bnd_z and max(air_bnd_z) >= float(z_inner_top) - 0.006
            ),
        }
        if membership_trusted:
            cavity_pass = all(checks.values())
        else:
            cavity_pass = False

        if not membership_trusted:
            primary = "MEMBERSHIP_INCONCLUSIVE"
            narrative = (
                "gmsh.model.isInside could not be trusted on the tagged air volume "
                "(air center-of-mass must register inside). Use the exported BREP/GEO "
                "visualization and calibration block in the JSON report before changing "
                "geometry."
            )
        elif cavity_pass:
            primary = "OK"
            narrative = (
                "Tagged air volume spans the inner cavity and soundhole channel; "
                "gmsh.model.isInside confirms air COM and cavity/soundhole probes "
                "lie inside air tag 10."
            )
        elif air_z_span < 0.5 * expected_inner_z_span:
            primary = "AIR_MIDDLE_SLICE"
            narrative = (
                "Tagged air volume Z extent is much smaller than the inner tool / shell "
                "cavity height — likely only a central Z-slab is classified as air (e.g. "
                "after wood Z-partition + final wood/air re-fragment). A disk imprint at "
                f"z≈{z_inner_top:.4f} cannot create an aperture on that surface family."
            )
        elif not checks["below_top_lid_in_air"]:
            primary = "AIR_MISSING_TOP"
            narrative = (
                "Points below the inner top lid / soundhole are not inside the tagged air "
                "volume; the cavity mouth region is wood or void."
            )
        else:
            primary = "AIR_INCOMPLETE"
            narrative = (
                "Tagged air volume does not fully occupy the expected interior cavity; "
                "see probe table and per-volume bounding boxes."
            )

        air_zmax = float(air_bb[5]) if air_vols else float("nan")
        if air_vols and air_zmax < float(z_inner_top) - 0.004:
            why_no_boundary = (
                "Air-boundary facets end below the inner top plane: "
                f"air z_max={air_zmax:.4f} m < inner top z={float(z_inner_top):.4f} m, "
                "so no air exterior surface exists at the intended soundhole disk plane."
            )
        elif air_vols and air_zmax >= float(z_inner_top) - 0.004:
            why_no_boundary = (
                "Air bbox reaches the inner top plane "
                f"(air z_max={air_zmax:.4f} m >= inner top z={float(z_inner_top):.4f} m). "
                "If aperture imprint still fails, inspect channel connectivity at the "
                "hole axis in the exported BREP."
            )
        else:
            why_no_boundary = "No air volume tagged."

        report = {
            "audit_type": "validation_cad_volume",
            "geometry": {
                "depth_m": float(depth_m),
                "shell_t_m": float(shell_t),
                "hole_radius_m": float(hole_r),
                "hole_center_xy_m": [float(hole_x), float(hole_y)],
            },
            "live_volume_ids": [int(v) for v in all_vol_tags],
            "outer_body_bbox_m": [float(x) for x in outer_bb],
            "outer_body_bbox_source": "union_of_gmsh.model.getEntities(3)_after_sync",
            "inner_tool_bbox_m": [float(x) for x in inner_tool_bb]
            if inner_tool_bb is not None
            else None,
            "expected_inner_z_span_m": float(expected_inner_z_span),
            "z_inner_back_m": float(z_inner_back),
            "z_inner_top_m": float(z_inner_top),
            "tagged_air_volume_ids": [int(v) for v in air_vols],
            "tagged_wood_volume_ids": [int(v) for v in wood_vols],
            "tagged_air_bbox_m": [float(x) for x in air_bb],
            "tagged_air_z_span_m": float(air_z_span),
            "inner_cavity_fill_ratio": float(fill_ratio),
            "air_boundary_surface_count": len(air_boundary),
            "air_boundary_centroid_z_range_m": air_bnd_z_range,
            "all_volumes": vol_records,
            "membership_calibration": membership_calibration,
            "membership_trusted": bool(membership_trusted),
            "cavity_probes": probe_records,
            "acceptance_checks": checks,
            "cavity_verification_pass": bool(cavity_pass),
            "recommended_next_step": (
                "A_proceed_to_aperture_surface_and_mesh_gates"
                if cavity_pass
                else (
                    "C_manual_brep_check_required"
                    if not membership_trusted
                    else "B_minimal_geometry_correction_at_soundhole_channel"
                )
            ),
            "diagnosis": {
                "primary": primary,
                "narrative": narrative,
                "why_no_boundary_near_z_inner_top": why_no_boundary,
            },
            "visualization": {
                "soundhole_axis_origin_m": [float(hole_x), float(hole_y), float(z_inner_back)],
                "soundhole_axis_end_m": [float(hole_x), float(hole_y), float(air_zmax)],
                "aperture_plane_z_m": float(z_inner_top),
                "hole_radius_m": float(hole_r),
            },
        }

        json_path = audit_stem.with_suffix(".json")
        md_path = audit_stem.with_suffix(".md")
        brep_path = audit_stem.with_suffix(".brep")
        geo_path = audit_stem.with_suffix(".geo")
        try:
            gmsh.write(str(brep_path))
            report["visualization"]["brep_file"] = str(brep_path)
        except Exception as exc:
            report["visualization"]["brep_error"] = str(exc)
        geo_lines = [
            "// Validation CAD volume audit visualization helper",
            f"// Air volume OCC id(s): {air_vols}",
            f"// Wood volume OCC id(s): {wood_vols}",
            f"// Soundhole axis: ({hole_x:.6f}, {hole_y:.6f}, {z_inner_back:.6f}) -> "
            f"({hole_x:.6f}, {hole_y:.6f}, {air_zmax:.6f})",
            f"// Aperture plane z = {z_inner_top:.6f} m, radius = {hole_r:.6f} m",
            "//",
            f"Merge \"{brep_path.name}\";",
            "",
            f"z_aperture = {float(z_inner_top):.6f};",
            f"x_hole = {float(hole_x):.6f};",
            f"y_hole = {float(hole_y):.6f};",
            f"r_hole = {float(hole_r):.6f};",
            "Point(1) = {x_hole, y_hole, z_aperture};",
            "Point(2) = {x_hole + r_hole, y_hole, z_aperture};",
            "Point(3) = {x_hole, y_hole + r_hole, z_aperture};",
            "Point(4) = {x_hole - r_hole, y_hole, z_aperture};",
            "Point(5) = {x_hole, y_hole - r_hole, z_aperture};",
            "Line(1) = {1, 2};",
            "Line(2) = {1, 3};",
            "Line(3) = {1, 4};",
            "Line(4) = {1, 5};",
            "// Open BREP: gmsh validation_cad_volume_audit.brep",
            "// Then enable Mesh.Volume / explore entity tags 3:{air_vols} and wood.",
        ]
        geo_path.write_text("\n".join(geo_lines) + "\n", encoding="utf-8")
        report["visualization"]["geo_helper"] = str(geo_path)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = [
            "# Validation CAD volume audit",
            "",
            f"**Primary:** `{primary}` — {narrative}",
            "",
            "## Tagged air (physical vol 10 precursor)",
            "",
            f"- Air volume ids: `{air_vols}`",
            f"- Air bbox z: `[{air_bb[2]:.4f}, {air_bb[5]:.4f}]` span={air_z_span:.4f} m",
            f"- Inner tool z: `[{z_inner_back:.4f}, {z_inner_top:.4f}]` "
            f"expected span={expected_inner_z_span:.4f} m",
            f"- Fill ratio (air span / inner tool span): **{fill_ratio:.3f}**",
            f"- Air-boundary centroid z range: `{air_bnd_z_range}`",
            "",
            report["diagnosis"]["why_no_boundary_near_z_inner_top"],
            "",
            "## Membership calibration",
            "",
            f"- Trusted `gmsh.model.isInside`: **{membership_trusted}**",
            f"- Recommended next step: `{report.get('recommended_next_step')}`",
            "",
            "## Visualization",
            "",
            f"- BREP: `{report.get('visualization', {}).get('brep_file', 'n/a')}`",
            f"- GEO helper (aperture plane + hole radius markers): "
            f"`{report.get('visualization', {}).get('geo_helper', 'n/a')}`",
            f"- Soundhole axis: `{report.get('visualization', {}).get('soundhole_axis_origin_m')}` "
            f"→ `{report.get('visualization', {}).get('soundhole_axis_end_m')}`",
            f"- Aperture plane z = `{report.get('visualization', {}).get('aperture_plane_z_m')}` m",
            "",
            "Gmsh GUI: open the `.brep`, use **Tools → Visibility** to hide/show volume tags; "
            "air is the tagged cavity/channel solid, wood volumes are the shell partition.",
            "",
            "## Acceptance checks",
            "",
        ]
        for k, v in checks.items():
            lines.append(f"- `{k}`: **{v}**")
        lines.extend(
            [
                "",
                "## Cavity probe points (gmsh.model.isInside per volume)",
                "",
            ]
        )
        for pr in probe_records:
            lines.append(
                f"- **{pr['label']}** `{pr['point_m']}` → air={pr['inside_air_volume_ids']} "
                f"wood={pr['inside_wood_volume_ids']}"
            )
        lines.extend(["", "## All 3D volumes", "", "| id | class | z_min | z_max | volume m³ |", "|---:|---|---:|---:|---:|"])
        for rec in vol_records:
            lines.append(
                f"| {rec['entity_id']} | {rec['classification']} | "
                f"{rec['z_min_m']:.4f} | {rec['z_max_m']:.4f} | {rec['volume_m3']:.6e} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[CAD-VOL-AUDIT] wrote {json_path}", flush=True)
        print(f"[CAD-VOL-AUDIT] wrote {md_path}", flush=True)
        print(
            f"[CAD-VOL-AUDIT] air_z_span={air_z_span:.4f} fill_ratio={fill_ratio:.3f} "
            f"pass={cavity_pass} primary={primary}",
            flush=True,
        )
        return report

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
    inner_tool_bb: Optional[list] = None

    # CAD reference injection: sketch = import + scale only; display/FOM = hollow + soundhole.
    vol_dimtags = _load_reference_model(shape_type)
    _scale_reference_to_target(
        vol_dimtags,
        length=L,
        width=W,
        depth=D,
        upper_bout=upper_bout,
        waist=waist,
        lower_bout=lower_bout,
        shape_type=shape_type,
    )
    vol_tags = [int(t) for d, t in vol_dimtags if int(d) == 3]
    if not vol_tags:
        vol_tags = [int(t) for d, t in occ.getEntities(3)]
    vol_out_id = int(vol_tags[0])

    if solid_sketch:
        vol_in_id = vol_out_id
        print(
            f"[diag] sketch: reference solid volume={vol_out_id} "
            "(no hollow / soundhole booleans)"
        )
    else:
        vol_in_id = _hollow_inner_volume(
            vol_out_id,
            wall_t=t,
            inner_depth=inner_depth,
            full_z_gap=is_validation,
        )
        inner_tool_bb = list(gmsh.model.getBoundingBox(3, int(vol_in_id)))
        print(
            f"[diag] reference shell tools: outer={vol_out_id} inner={vol_in_id} "
            f"inner_bbox_z=[{inner_tool_bb[2]:.4f},{inner_tool_bb[5]:.4f}]"
        )

    # Soundhole cutter: short band at outer top (production / display). Validation FSI uses a
    # full-height cylinder through the inner air pocket (created after vol_in_id is known).
    hole_cyl: Optional[int] = None
    if not solid_sketch and (shell_only or not use_air_opening_geom):
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
                "[diag] sketch CAD: reference solid only "
                "(no occ.cut hollow / soundhole)"
            )
            wood_dimtags = [(3, int(vol_out_id))]
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
        if is_display or is_preview:
            wood_hole_cut = _audit_boolean(
                "display_soundhole_cut" if is_display else "preview_soundhole_cut",
                occ.cut,
                wood_dimtags,
                [(3, hole_cyl)],
                removeObject=True,
                removeTool=True,
            )
            wood_dimtags = [dt for dt in as_dimtags(wood_hole_cut) if dt[0] == 3]
            print(
                f"[diag] {mode} CAD: soundhole OCC cut applied "
                f"(cylinder r={hr:.4f} m, cap=0.25*min(L,W)={0.25 * min(float(L), float(W)):.4f})"
            )
        try:
            occ.removeAllDuplicates()
        except Exception:
            pass
        occ.synchronize()
        if is_display or is_preview:
            wood_dimtags = _heal_display_shell_cad(occ, wood_dimtags)
        if is_display:
            wood_dimtags = _fragment_display_shell_seams(occ, wood_dimtags)
        wood_vols = [int(tag) for _, tag in wood_dimtags]
        if not solid_sketch and not (is_display or is_preview):
            wood_vols = _split_and_fragment_wood_shell_for_conforming_interfaces(
                wood_vols, t, occ
            )
        air_vols: list = []
        air_dimtags: list = []
    else:
        # FSI engineering mesh: internal air cavity + shared interface with wood.
        if use_air_opening_geom:
            # Validation: fuse full cavity with a through-plate soundhole air channel (r=hr).
            bb_in = (
                list(inner_tool_bb)
                if inner_tool_bb is not None
                else list(gmsh.model.getBoundingBox(3, int(vol_in_id)))
            )
            z_ch_lo = float(bb_in[2]) + 0.001
            z_ch_hi = float(bb_in[5]) + float(t) + 0.002
            hole_cyl = int(
                occ.addCylinder(
                    hole_x,
                    hole_y,
                    z_ch_lo,
                    0,
                    0,
                    z_ch_hi - z_ch_lo,
                    hr,
                )
            )
            occ.synchronize()
            print(
                "[diag] validation soundhole air-channel: fuse cavity+channel "
                f"z=[{z_ch_lo:.4f},{z_ch_hi:.4f}] r={hr:.4f} m "
                "(inner cavity through top plate to exterior)"
            )
            air_fused = _audit_boolean(
                "validation_air_cavity_channel_fuse",
                occ.fuse,
                [(3, int(vol_in_id))],
                [(3, int(hole_cyl))],
                removeObject=True,
                removeTool=False,
            )
            air_dimtags = [dt for dt in as_dimtags(air_fused) if dt[0] == 3]
            if not air_dimtags:
                raise RuntimeError(
                    "validation air cavity+channel fuse produced no volume"
                )
        else:
            if hole_cyl is None:
                z_hole_lo = (D / 2.0) - t - 0.001
                z_hole_hi = (D / 2.0) + 0.001
                hole_cyl = int(
                    occ.addCylinder(
                        hole_x, hole_y, z_hole_lo, 0, 0, z_hole_hi - z_hole_lo, hr
                    )
                )
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

    if is_fom and not is_validation:
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

    if is_validation and not shell_only:
        vol_audit_stem = out_file.parent / "validation_cad_volume_audit"
        vol_audit = _write_validation_cad_volume_audit(
            audit_stem=vol_audit_stem,
            air_vols=air_vols,
            wood_vols=wood_vols,
            hole_x=float(hole_x),
            hole_y=float(hole_y),
            hole_r=float(hr),
            depth_m=float(D),
            shell_t=float(t),
            inner_tool_bb=inner_tool_bb,
        )
        if not vol_audit.get("cavity_verification_pass"):
            diag = vol_audit.get("diagnosis", {})
            raise RuntimeError(
                "FEM_VALIDATION_MESH: air cavity verification FAILED before meshing. "
                f"{diag.get('primary', '?')}: {diag.get('narrative', '')} "
                f"See {vol_audit_stem.with_suffix('.json')} and "
                f"{vol_audit_stem.with_suffix('.md')}"
            )
        if os.environ.get("FEM_VALIDATION_CAD_AUDIT_ONLY", "0") == "1":
            print(
                "[diag] FEM_VALIDATION_CAD_AUDIT_ONLY=1: cavity audit PASS; "
                "stopping before mesh generation (no .msh written)."
            )
            gmsh.finalize()
            return

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

    def _surface_area_m(surf_tag: int) -> float:
        try:
            mass = gmsh.model.occ.getMass(2, int(surf_tag))
            if isinstance(mass, (int, float)):
                return float(mass)
            if isinstance(mass, (list, tuple)) and len(mass) >= 1:
                return float(mass[0])
        except Exception:
            pass
        try:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, int(surf_tag))
            return float(
                max(xmax - xmin, 1.0e-30)
                * max(ymax - ymin, 1.0e-30)
            )
        except Exception:
            return float("nan")

    def _cad_surface_record(
        surf_tag: int,
        *,
        hx: float,
        hy: float,
        hole_r: float,
        z_top_outer: float,
        z_inner_lid: float,
        picker_z_lo: float,
        picker_z_hi: float,
        picker_r_cap: float,
    ) -> dict:
        cx, cy, cz = get_surface_center(surf_tag)
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, int(surf_tag))
        rxy = math.hypot(float(cx) - float(hx), float(cy) - float(hy))
        in_disk_relaxed = rxy <= float(hole_r) * 1.5
        in_disk_picker = rxy <= float(picker_r_cap)
        z_ok_picker = float(picker_z_lo) <= float(cz) <= float(picker_z_hi)
        nz = get_surface_normal_signed_z(surf_tag)
        return {
            "surface_id": int(surf_tag),
            "bbox_m": [float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax)],
            "center_m": [float(cx), float(cy), float(cz)],
            "area_m2": float(_surface_area_m(surf_tag)),
            "dist_xy_from_hole_center_m": float(rxy),
            "z_top_outer_m": float(z_top_outer),
            "z_inner_lid_m": float(z_inner_lid),
            "z_minus_top_outer_m": float(cz) - float(z_top_outer),
            "z_minus_inner_lid_m": float(cz) - float(z_inner_lid),
            "in_soundhole_disk_relaxed": bool(in_disk_relaxed),
            "in_soundhole_disk_picker_r": bool(in_disk_picker),
            "in_picker_z_band": bool(z_ok_picker),
            "picker_would_select": bool(in_disk_picker and z_ok_picker),
            "normal_signed_z": float(nz) if nz is not None else None,
        }

    def _diagnose_cad_soundhole_audit(
        air_records: list,
        near_disk_records: list,
    ) -> dict:
        n_air = len(air_records)
        n_air_disk = sum(1 for r in air_records if r.get("in_soundhole_disk_relaxed"))
        n_air_picker = sum(1 for r in air_records if r.get("picker_would_select"))
        n_wood_near = sum(
            1
            for r in near_disk_records
            if r.get("membership") == "wood_boundary_only"
        )
        n_air_near = sum(
            1 for r in near_disk_records if r.get("membership") == "air_boundary_only"
        )
        codes: List[str] = []
        if n_air == 0:
            codes.append("C")
        if n_air_disk == 0 and n_wood_near > 0:
            codes.append("C")
        if n_air_disk > 0 and n_air_picker == 0:
            codes.append("A")
        if n_air_disk > 0 and n_air_picker > 0:
            codes.append("OK")
        if n_air_disk > 0 and n_air_near > 10 and n_air_picker == 0:
            codes.append("B")
        if not codes:
            codes.append("D")
        primary = codes[0]
        if "OK" in codes:
            primary = "OK"
        elif "A" in codes and "C" not in codes:
            primary = "A"
        elif "C" in codes:
            primary = "C"
        elif "B" in codes:
            primary = "B"
        narratives = {
            "A": (
                "Air-boundary surfaces exist near the soundhole disk but current z/r "
                "picker bands exclude them (tolerance/plane mismatch)."
            ),
            "B": (
                "Air-boundary opening geometry is fragmented into many small surfaces; "
                "picker or grouping may not match assumed single opening patch."
            ),
            "C": (
                "No air-boundary surface lies in the expected soundhole disk/opening; "
                "cavity may be sealed at the hole or opening exists only on wood exterior."
            ),
            "D": "Ambiguous CAD topology; inspect full air_boundary enumeration.",
            "OK": "Air-boundary opening surfaces found in picker band.",
        }
        z_vals = [r["center_m"][2] for r in air_records if r.get("center_m")]
        return {
            "primary_hypothesis": primary,
            "hypothesis_codes": codes,
            "narrative": narratives.get(primary, ""),
            "counts": {
                "n_air_boundary_total": n_air,
                "n_air_in_disk_relaxed": n_air_disk,
                "n_air_picker_would_select": n_air_picker,
                "n_near_disk_air_only": n_air_near,
                "n_near_disk_wood_only": n_wood_near,
            },
            "air_boundary_z_range_m": (
                [min(z_vals), max(z_vals)] if z_vals else None
            ),
        }

    def _write_cad_soundhole_air_boundary_audit(
        *,
        audit_stem: Path,
        air_boundary_surfs: list,
        wood_boundary_surfs: list,
        hx: float,
        hy: float,
        hole_r: float,
        depth_m: float,
        shell_t: float,
        hole_x: float,
        hole_y: float,
    ) -> Path:
        """Pre-mesh CAD audit (validation / FEM_SOUNDHOLE_TAG_AIR_OPENING only)."""
        z_top_outer = float(depth_m) / 2.0
        z_inner_lid = z_top_outer - float(shell_t)
        z_tol = max(1.0e-4, float(shell_t), 0.25 * float(hole_r))
        picker_z_lo = z_top_outer - float(shell_t) - 0.004
        picker_z_hi = z_top_outer + 0.003
        picker_r_cap = float(hole_r) * 1.5
        z_near_lo = z_top_outer - float(shell_t) - 0.015
        z_near_hi = z_top_outer + 0.008

        air_set = {int(s) for s in air_boundary_surfs}
        wood_set = {int(s) for s in wood_boundary_surfs}
        all_boundary = sorted(air_set | wood_set)

        air_records = [
            _cad_surface_record(
                int(s),
                hx=hx,
                hy=hy,
                hole_r=hole_r,
                z_top_outer=z_top_outer,
                z_inner_lid=z_inner_lid,
                picker_z_lo=picker_z_lo,
                picker_z_hi=picker_z_hi,
                picker_r_cap=picker_r_cap,
            )
            for s in sorted(air_set)
        ]

        near_disk_records: List[dict] = []
        for s in all_boundary:
            rec = _cad_surface_record(
                int(s),
                hx=hx,
                hy=hy,
                hole_r=hole_r,
                z_top_outer=z_top_outer,
                z_inner_lid=z_inner_lid,
                picker_z_lo=picker_z_lo,
                picker_z_hi=picker_z_hi,
                picker_r_cap=picker_r_cap,
            )
            cz = rec["center_m"][2]
            if not rec["in_soundhole_disk_relaxed"]:
                continue
            if float(cz) < z_near_lo or float(cz) > z_near_hi:
                continue
            in_air = int(s) in air_set
            in_wood = int(s) in wood_set
            if in_air and in_wood:
                membership = "air_and_wood_boundary"
            elif in_air:
                membership = "air_boundary_only"
            elif in_wood:
                membership = "wood_boundary_only"
            else:
                membership = "neither"
            near_disk_records.append({**rec, "membership": membership})

        diagnosis = _diagnose_cad_soundhole_audit(air_records, near_disk_records)
        payload = {
            "expected_soundhole": {
                "center_m": [float(hole_x), float(hole_y), float(z_top_outer)],
                "radius_m": float(hole_r),
                "depth_m": float(depth_m),
                "shell_t_m": float(shell_t),
                "z_top_outer_m": z_top_outer,
                "z_inner_lid_m": z_inner_lid,
                "picker_z_band_m": [picker_z_lo, picker_z_hi],
                "picker_r_cap_m": picker_r_cap,
            },
            "air_boundary_surfaces": air_records,
            "near_soundhole_disk_surfaces": near_disk_records,
            "diagnosis": diagnosis,
        }

        audit_stem.parent.mkdir(parents=True, exist_ok=True)
        json_path = audit_stem.with_suffix(".json")
        md_path = audit_stem.with_suffix(".md")
        log_path = audit_stem.with_suffix(".log")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        lines = [
            "# CAD soundhole ↔ air-boundary audit (pre-mesh)",
            "",
            f"Expected hole centre (x,y,z_outer): `{hole_x:.4f}, {hole_y:.4f}, {z_top_outer:.4f}` m",
            f"Hole radius: `{hole_r:.4f}` m | shell t: `{shell_t:.4f}` m",
            "",
            "## Summary",
            "",
            f"- Air boundary surfaces (total): **{len(air_records)}**",
            f"- Air in soundhole disk (relaxed r): **{diagnosis['counts']['n_air_in_disk_relaxed']}**",
            f"- Air passing current picker z/r: **{diagnosis['counts']['n_air_picker_would_select']}**",
            f"- Near disk, wood-only: **{diagnosis['counts']['n_near_disk_wood_only']}**",
            f"- Near disk, air-only: **{diagnosis['counts']['n_near_disk_air_only']}**",
            f"- Air boundary z range (centroids): `{diagnosis.get('air_boundary_z_range_m')}`",
            "",
            f"**Primary hypothesis: {diagnosis['primary_hypothesis']}** — {diagnosis['narrative']}",
            "",
            f"Full JSON: `{json_path}`",
            "",
            "## Air boundary surfaces (all)",
            "",
            "| id | area | cx | cy | cz | r_xy | z−z_outer | z−z_inner | in_disk | picker_ok |",
            "|----|------|----|----|-----|------|-----------|-----------|---------|-----------|",
        ]
        for r in air_records:
            c = r["center_m"]
            lines.append(
                f"| {r['surface_id']} | {r['area_m2']:.4e} | {c[0]:.4f} | {c[1]:.4f} | {c[2]:.4f} | "
                f"{r['dist_xy_from_hole_center_m']:.4f} | {r['z_minus_top_outer_m']:.4f} | "
                f"{r['z_minus_inner_lid_m']:.4f} | {r['in_soundhole_disk_relaxed']} | "
                f"{r['picker_would_select']} |"
            )
        lines.append("")
        lines.append("## Surfaces near soundhole disk (wood ∪ air boundaries)")
        lines.append("")
        for r in near_disk_records[:40]:
            c = r["center_m"]
            lines.append(
                f"- id={r['surface_id']} membership={r['membership']} "
                f"r_xy={r['dist_xy_from_hole_center_m']:.4f} z={c[2]:.4f} area={r['area_m2']:.4e}"
            )
        if len(near_disk_records) > 40:
            lines.append(f"- ... ({len(near_disk_records) - 40} more in JSON)")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        log_lines = [
            "[CAD-AUDIT] soundhole air-boundary enumeration (pre-mesh)",
            f"[CAD-AUDIT] n_air_boundary={len(air_records)} n_near_disk={len(near_disk_records)}",
            f"[CAD-AUDIT] diagnosis={diagnosis['primary_hypothesis']}: {diagnosis['narrative']}",
            f"[CAD-AUDIT] wrote {json_path}",
            f"[CAD-AUDIT] wrote {md_path}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        for ln in log_lines:
            print(ln, flush=True)
        return json_path

    def _imprint_validation_soundhole_aperture_disk(
        air_vol_tag: int,
        *,
        hx: float,
        hy: float,
        hole_r: float,
    ) -> Tuple[float, int]:
        """
        Imprint a planar disk on the air volume top boundary so the soundhole mouth is a
        dedicated CAD surface (validation / opt-in only).
        """
        bb = gmsh.model.getBoundingBox(3, int(air_vol_tag))
        z_plane = float(bb[5]) - 5.0e-6
        disk_tag = int(
            occ.addDisk(float(hx), float(hy), z_plane, float(hole_r), float(hole_r))
        )
        occ.synchronize()
        out, _map = _audit_boolean(
            "validation_soundhole_aperture_imprint",
            occ.fragment,
            [(3, int(air_vol_tag))],
            [(2, disk_tag)],
            removeObject=False,
            removeTool=True,
        )
        occ.synchronize()
        air_out = int(air_vol_tag)
        for dim, tag in out:
            if int(dim) == 3:
                air_out = int(tag)
                break
        print(
            "[diag] validation soundhole aperture disk imprint: "
            f"z_plane={z_plane:.6f} r={hole_r:.4f} air_vol={air_out}"
        )
        return z_plane, air_out

    def _select_validation_soundhole_aperture_surfaces(
        air_boundary_surfs: list,
        *,
        hx: float,
        hy: float,
        hole_r: float,
        z_plane: float,
    ) -> list:
        """
        Select only the circular external air-side aperture (not cavity walls or lids).

        Requires prior disk imprint; matches area, span, centroid, +Z normal, and z band.
        """
        if not air_boundary_surfs:
            return []
        expected = math.pi * float(hole_r) * float(hole_r)
        a_lo = 0.85 * expected
        a_hi = 1.15 * expected
        z_tol = 0.008
        r_cent_max = 1.05 * float(hole_r)
        span_max = 2.05 * float(hole_r)
        scored: list = []
        for s in sorted(int(x) for x in air_boundary_surfs):
            area = _surface_area_m(int(s))
            if not (a_lo <= area <= a_hi):
                continue
            cx, cy, cz = get_surface_center(int(s))
            if math.hypot(float(cx) - float(hx), float(cy) - float(hy)) > r_cent_max:
                continue
            if abs(float(cz) - float(z_plane)) > z_tol:
                continue
            xmin, ymin, _zmin, xmax, ymax, _zmax = gmsh.model.getBoundingBox(2, int(s))
            span_xy = max(float(xmax) - float(xmin), float(ymax) - float(ymin))
            if span_xy > span_max:
                continue
            nz = get_surface_normal_signed_z(int(s))
            if nz is None or float(nz) < 0.85:
                continue
            scored.append((int(s), float(area), abs(float(area) - expected)))
        if not scored:
            return []
        scored.sort(key=lambda row: row[2])
        best_area = scored[0][1]
        return sorted(
            {
                row[0]
                for row in scored
                if abs(row[1] - best_area) <= 0.05 * expected
            }
        )

    def _validate_validation_soundhole_aperture_cad(
        soundhole_surfs: list,
        *,
        hx: float,
        hy: float,
        hole_r: float,
        z_plane: float,
    ) -> dict:
        """Pre-mesh CAD acceptance for validation soundhole tag 2."""
        expected = math.pi * float(hole_r) * float(hole_r)
        if not soundhole_surfs:
            raise RuntimeError(
                "FEM_VALIDATION_MESH: no dedicated soundhole aperture surface after "
                "disk imprint (tag 2 would be empty)."
            )
        total_area = sum(_surface_area_m(int(s)) for s in soundhole_surfs)
        if not (0.85 * expected <= total_area <= 1.15 * expected):
            raise RuntimeError(
                "FEM_VALIDATION_MESH: soundhole aperture CAD area "
                f"{total_area:.8f} m² outside ±15% of πr²={expected:.8f} m² "
                f"(surfaces={soundhole_surfs})."
            )
        r_max = 0.0
        z_vals: list = []
        horiz_ok = 0
        for s in soundhole_surfs:
            cx, cy, cz = get_surface_center(int(s))
            r_max = max(r_max, math.hypot(float(cx) - float(hx), float(cy) - float(hy)))
            z_vals.append(float(cz))
            nz = get_surface_normal_signed_z(int(s))
            if nz is not None and float(nz) >= 0.85:
                horiz_ok += 1
        if r_max > 0.050:
            raise RuntimeError(
                f"FEM_VALIDATION_MESH: soundhole aperture radial extent {r_max:.6f} m "
                f"> 0.050 m (r={hole_r:.4f} m)."
            )
        z_span = max(z_vals) - min(z_vals) if z_vals else float("inf")
        if z_span > 0.012:
            raise RuntimeError(
                f"FEM_VALIDATION_MESH: soundhole aperture z-span {z_span:.6f} m "
                "(expected planar disk near external opening)."
            )
        if horiz_ok < len(soundhole_surfs):
            raise RuntimeError(
                "FEM_VALIDATION_MESH: soundhole aperture surface(s) not horizontal (+Z)."
            )
        return {
            "cad_surface_tags": [int(s) for s in soundhole_surfs],
            "total_area_m2": float(total_area),
            "expected_area_m2": float(expected),
            "radial_max_m": float(r_max),
            "z_plane_m": float(z_plane),
            "z_span_m": float(z_span),
            "n_surfaces": len(soundhole_surfs),
        }

    def _select_soundhole_disk_centroid_fallback(
        shell_tags, z_plane, z_tol, hx, hy, hole_r
    ):
        """
        FOM fallback when edge-bite booleans leave no dedicated hole rim surfaces.

        Tag facets whose centroid lies in the soundhole disk near the top opening
        (air inner lid + top annulus), rejecting whole-plate-sized patches.
        """
        r_lim = float(hole_r) * 1.45
        z_slack = max(float(z_tol) * 4.0, 2.0e-3)
        max_span = max(8.0 * float(hole_r), 0.12)
        out: list = []
        for s in shell_tags:
            try:
                cx, cy, cz = get_surface_center(s)
            except Exception:
                continue
            if abs(float(cz) - float(z_plane)) > z_slack:
                continue
            if math.hypot(float(cx) - float(hx), float(cy) - float(hy)) > r_lim:
                continue
            xmin, ymin, _zmin, xmax, ymax, _zmax = gmsh.model.getBoundingBox(2, int(s))
            span_xy = max(float(xmax) - float(xmin), float(ymax) - float(ymin))
            if span_xy > max_span:
                continue
            out.append(int(s))
        return sorted(set(out))

    # Soundhole: air-cavity mouth (validation/opt-in) or legacy exterior-shell picker (production FOM).
    z_top_outer = D / 2.0
    z_tol = max(1.0e-4, t, 0.25 * hr)
    use_air_opening_tag = (
        is_validation or os.environ.get("FEM_SOUNDHOLE_TAG_AIR_OPENING", "0") == "1"
    )
    soundhole_surfs: list = []
    if use_air_opening_tag and not shell_only:
        if not air_boundary_surfs:
            raise RuntimeError(
                "FEM_VALIDATION_MESH / FEM_SOUNDHOLE_TAG_AIR_OPENING: no air volume "
                "boundary surfaces after fragment — cannot define acoustic soundhole tag 2."
            )
        audit_stem = (
            out_file.parent / "soundhole_cad_air_boundary_audit"
            if is_validation
            else mesh_dir / "soundhole_cad_air_boundary_audit"
        )
        audit_json = _write_cad_soundhole_air_boundary_audit(
            audit_stem=audit_stem,
            air_boundary_surfs=air_boundary_surfs,
            wood_boundary_surfs=wood_boundary_surfs,
            hx=float(hole_x),
            hy=float(hole_y),
            hole_r=float(hr),
            depth_m=float(D),
            shell_t=float(t),
            hole_x=float(hole_x),
            hole_y=float(hole_y),
        )
        z_aperture_plane, air_vol_tag = _imprint_validation_soundhole_aperture_disk(
            int(air_vols[0]),
            hx=float(hole_x),
            hy=float(hole_y),
            hole_r=float(hr),
        )
        air_vols[0] = int(air_vol_tag)
        air_boundary_surfs = get_boundary_tags([(3, int(air_vols[0]))], 2)
        soundhole_surfs = _select_validation_soundhole_aperture_surfaces(
            sorted(air_boundary_surfs),
            hx=float(hole_x),
            hy=float(hole_y),
            hole_r=float(hr),
            z_plane=float(z_aperture_plane),
        )
        cad_gate = _validate_validation_soundhole_aperture_cad(
            soundhole_surfs,
            hx=float(hole_x),
            hy=float(hole_y),
            hole_r=float(hr),
            z_plane=float(z_aperture_plane),
        )
        print(
            f"[diag] soundhole tag 2: validation aperture surfaces n={len(soundhole_surfs)} "
            f"CAD tags={cad_gate['cad_surface_tags']} area={cad_gate['total_area_m2']:.6f} m² "
            f"(disk imprint; not broad air-boundary picker)"
        )
        if is_validation:
            gate_path = out_file.parent / "soundhole_aperture_cad_gate.json"
            gate_path.write_text(json.dumps(cad_gate, indent=2), encoding="utf-8")
            print(f"[diag] wrote validation CAD aperture gate: {gate_path}")
        if not soundhole_surfs:
            raise RuntimeError(
                "FEM_VALIDATION_MESH: dedicated soundhole aperture surface not found after "
                "disk imprint. Legacy wood-top / broad air-boundary tagging is disabled. "
                f"Inspect CAD audit: {audit_json} and {audit_stem.with_suffix('.md')}"
            )
    elif not shell_only:
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
        if not soundhole_surfs and is_fom:
            disk_pool = sorted(set(air_boundary_surfs) | set(int(s) for s in top_plate_surfs))
            soundhole_surfs = _select_soundhole_disk_centroid_fallback(
                disk_pool, z_top_outer, z_tol, hole_x, hole_y, hr
            )
            if soundhole_surfs:
                print(
                    f"[diag] soundhole tag 2 FOM disk-centroid fallback: {len(soundhole_surfs)} facets"
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
    if not soundhole_surfs and not (use_air_opening_tag and not shell_only):
        soundhole_surfs = _select_soundhole_surfaces(
            all_shell_surfs, z_top_outer, z_tol * 2.5, hole_x, hole_y, hr
        )
    if not soundhole_surfs and not (use_air_opening_tag and not shell_only):
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
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 24 if is_display else 12)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 48)
    
    gmsh.model.mesh.setOrder(1)
    mesh_resolution_factor = 1.0
    print(
        f"[diag] mesh_resolution_factor={mesh_resolution_factor}, "
        "element_order=1"
    )

    def _soundhole_boundary_curve_tags():
        curve_tags: set = set()
        for st in soundhole_surfs:
            try:
                bnd = gmsh.model.getBoundary([(2, int(st))], oriented=False, recursive=False)
            except Exception:
                continue
            for bdim, btag in bnd:
                if int(bdim) == 1:
                    curve_tags.add(int(btag))
        return sorted(curve_tags)

    def _plate_interface_curve_tags() -> List[int]:
        """Curves on top/back plate perimeters (plate–side interfaces)."""
        curve_tags: set = set()
        for st in list(dict.fromkeys(top_plate_surfs + back_plate_surfs)):
            try:
                bnd = gmsh.model.getBoundary([(2, int(st))], oriented=False, recursive=False)
            except Exception:
                continue
            for bdim, btag in bnd:
                if int(bdim) == 1:
                    curve_tags.add(int(btag))
        return sorted(curve_tags)

    def _interface_refinement_curve_tags() -> List[int]:
        """Plate perimeters + rib↔plate contact lines (boundary refinement zone)."""
        curve_tags: set = set(_plate_interface_curve_tags())
        top_set = set(int(s) for s in top_plate_surfs)
        back_set = set(int(s) for s in back_plate_surfs)
        for rs in rib_surfs:
            try:
                bnd = gmsh.model.getBoundary([(2, int(rs))], oriented=False, recursive=False)
            except Exception:
                continue
            for bdim, ctag in bnd:
                if int(bdim) != 1:
                    continue
                ctag = int(ctag)
                try:
                    up, _down = gmsh.model.getAdjacencies(1, ctag)
                except Exception:
                    up, _down = [], []
                adj_faces = set(int(t) for t in up if int(t) > 0)
                if adj_faces & top_set or adj_faces & back_set:
                    curve_tags.add(ctag)
        return sorted(curve_tags)

    def _apply_display_shell_coarse_with_seam_band(
        global_lc: float,
        seam_lc: float,
        seam_band_m: float,
    ) -> None:
        """Display: 12 mm global + narrow Distance band on top/back plate perimeters only."""
        field_ids: List[int] = []
        coarse = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(coarse, "VIn", float(global_lc))
        gmsh.model.mesh.field.setNumber(coarse, "VOut", float(global_lc))
        field_ids.append(coarse)

        plate_seam_curves = _plate_interface_curve_tags()
        if plate_seam_curves:
            dist_iface = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_iface, "CurvesList", plate_seam_curves)
            gmsh.model.mesh.field.setNumber(dist_iface, "Sampling", 100)
            thresh_iface = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_iface, "InField", dist_iface)
            gmsh.model.mesh.field.setNumber(thresh_iface, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(thresh_iface, "DistMax", float(seam_band_m))
            gmsh.model.mesh.field.setNumber(thresh_iface, "SizeMin", float(seam_lc))
            gmsh.model.mesh.field.setNumber(thresh_iface, "SizeMax", float(global_lc))
            field_ids.append(thresh_iface)

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(global_lc))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(global_lc))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        print(
            f"[diag] display shell: global_lc={global_lc * 1000:.1f}mm, "
            f"seam_lc={seam_lc * 1000:.1f}mm in {seam_band_m * 1000:.1f}mm DistMax band "
            f"({len(plate_seam_curves)} top/back plate seam curves)"
        )

    def _apply_shell_coarse_global_local_fields(
        global_lc: float,
        fine_lc: float,
    ) -> None:
        """Preview shell with Distance/Threshold refinement at hole and plate edges only."""
        field_ids: List[int] = []
        coarse = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(coarse, "VIn", float(global_lc))
        gmsh.model.mesh.field.setNumber(coarse, "VOut", float(global_lc))
        field_ids.append(coarse)

        if soundhole_surfs:
            dist_hole = gmsh.model.mesh.field.add("Distance")
            hole_faces = [int(s) for s in soundhole_surfs]
            gmsh.model.mesh.field.setNumbers(dist_hole, "FacesList", hole_faces)
            hole_curves = _soundhole_boundary_curve_tags()
            if hole_curves:
                gmsh.model.mesh.field.setNumbers(dist_hole, "CurvesList", hole_curves)
            thresh_hole = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_hole, "InField", dist_hole)
            gmsh.model.mesh.field.setNumber(thresh_hole, "DistMin", 0.0002)
            gmsh.model.mesh.field.setNumber(thresh_hole, "DistMax", max(0.008, 2.0 * hr))
            gmsh.model.mesh.field.setNumber(thresh_hole, "SizeMin", float(fine_lc))
            gmsh.model.mesh.field.setNumber(thresh_hole, "SizeMax", float(global_lc))
            field_ids.append(thresh_hole)

        interface_curves = _interface_refinement_curve_tags()
        if interface_curves:
            dist_iface = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_iface, "CurvesList", interface_curves)
            thresh_iface = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_iface, "InField", dist_iface)
            gmsh.model.mesh.field.setNumber(thresh_iface, "DistMin", 0.001)
            gmsh.model.mesh.field.setNumber(thresh_iface, "DistMax", 0.008)
            gmsh.model.mesh.field.setNumber(thresh_iface, "SizeMin", float(fine_lc))
            gmsh.model.mesh.field.setNumber(thresh_iface, "SizeMax", float(global_lc))
            field_ids.append(thresh_iface)

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        print(
            f"[diag] shell mesh: global_lc={global_lc * 1000:.1f}mm, "
            f"local_lc={fine_lc * 1000:.2f}mm at soundhole ({len(soundhole_surfs)} faces) "
            f"and interface refinement zone ({len(interface_curves)} curves)"
        )

    if is_display:
        _apply_display_shell_coarse_with_seam_band(
            DISPLAY_GLOBAL_LC_M, DISPLAY_SEAM_LC_M, DISPLAY_SEAM_BAND_M
        )
    elif is_preview:
        _apply_shell_coarse_global_local_fields(PREVIEW_GLOBAL_LC_M, LOCAL_REFINE_LC_M)

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
            # Soundhole: gradual Threshold (thickness lc → wood surface lc) over a short band only.
            hole_lc_target = wood_thickness_size
            soundhole_gradual_refine = False
            if soundhole_surfs:
                soundhole_gradual_refine = True
                for s in soundhole_surfs:
                    try:
                        gmsh.model.mesh.setSize(2, int(s), hole_lc_target)
                    except Exception:
                        pass
                dist_hole = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_hole, "FacesList", [int(s) for s in soundhole_surfs])
                hole_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(hole_thresh, "InField", dist_hole)
                gmsh.model.mesh.field.setNumber(hole_thresh, "DistMin", soundhole_threshold_dist_min)
                gmsh.model.mesh.field.setNumber(hole_thresh, "DistMax", soundhole_threshold_dist_max)
                gmsh.model.mesh.field.setNumber(hole_thresh, "SizeMin", hole_lc_target)
                gmsh.model.mesh.field.setNumber(hole_thresh, "SizeMax", wood_surface_size)
                combine_list.append(hole_thresh)

            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", combine_list)
            print("[diag] FOM mesh targets:")
            print(f"  wood_surface_lc={wood_surface_size * 1000:.2f} mm")
            print(f"  wood_through_thickness_lc={wood_thickness_size * 1000:.2f} mm")
            print(f"  air_min_lc={air_threshold_size_min * 1000:.2f} mm")
            if soundhole_surfs:
                print(
                    f"  soundhole_local_lc={hole_lc_target * 1000:.2f} mm "
                    f"(gradual Threshold {hole_lc_target * 1000:.1f}–{wood_surface_size * 1000:.1f} mm "
                    f"over d={soundhole_threshold_dist_min * 1000:.1f}–{soundhole_threshold_dist_max * 1000:.1f} mm, "
                    f"n_faces={len(soundhole_surfs)})"
                )
            else:
                print("  soundhole_local_lc=inactive (no soundhole surfaces tagged)")
            print(f"  soundhole_gradual_refine_active={soundhole_gradual_refine}")
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

        def _validation_gate_soundhole_aperture_post_mesh() -> None:
            """Fail validation mesh build if tag-2 is not a single circular aperture."""
            if not (is_validation and use_air_opening_tag and not shell_only):
                return
            expected = math.pi * float(hr) * float(hr)
            a_lo, a_hi = 0.85 * expected, 1.15 * expected
            try:
                entities = gmsh.model.getEntitiesForPhysicalGroup(2, 2)
            except Exception as exc:
                raise RuntimeError(
                    f"FEM_VALIDATION_MESH post-mesh gate: soundhole group 2 missing: {exc}"
                ) from exc
            surf_tags: list = []
            cad_area = 0.0
            r_max = 0.0
            z_vals: list = []
            horiz = 0
            for ent in entities:
                if isinstance(ent, (list, tuple)) and len(ent) >= 2:
                    dim_e, tag_e = int(ent[0]), int(ent[1])
                else:
                    dim_e, tag_e = 2, int(ent)
                if dim_e != 2:
                    continue
                surf_tags.append(int(tag_e))
                cad_area += _surface_area_m(int(tag_e))
                cx, cy, cz = get_surface_center(int(tag_e))
                r_max = max(
                    r_max,
                    math.hypot(float(cx) - float(hole_x), float(cy) - float(hole_y)),
                )
                z_vals.append(float(cz))
                nz = get_surface_normal_signed_z(int(tag_e))
                if nz is not None and float(nz) >= 0.85:
                    horiz += 1
            def _triangle_area_m(c0, c1, c2):
                ax, ay, az = (
                    float(c1[0]) - float(c0[0]),
                    float(c1[1]) - float(c0[1]),
                    float(c1[2]) - float(c0[2]),
                )
                bx, by, bz = (
                    float(c2[0]) - float(c0[0]),
                    float(c2[1]) - float(c0[1]),
                    float(c2[2]) - float(c0[2]),
                )
                cx = ay * bz - az * by
                cy = az * bx - ax * bz
                cz = ax * by - ay * bx
                return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

            mesh_area = 0.0
            for tag_e in surf_tags:
                _types, elem_tags, node_tags = gmsh.model.mesh.getElements(2, int(tag_e))
                for etype, et_arr, nt_arr in zip(_types, elem_tags, node_tags):
                    if int(etype) != 2:
                        continue
                    nodes = [int(n) for n in nt_arr]
                    for i in range(0, len(nodes), 3):
                        coords = [
                            gmsh.model.mesh.getNode(int(nodes[i + k]))[1] for k in range(3)
                        ]
                        mesh_area += _triangle_area_m(coords[0], coords[1], coords[2])
                        for c in coords:
                            r_max = max(
                                r_max,
                                math.hypot(
                                    float(c[0]) - float(hole_x),
                                    float(c[1]) - float(hole_y),
                                ),
                            )
                            z_vals.append(float(c[2]))
            z_span = max(z_vals) - min(z_vals) if z_vals else float("inf")
            failures: list = []
            if not surf_tags:
                failures.append("physical group 2 has no surfaces")
            if not (a_lo <= cad_area <= a_hi):
                failures.append(
                    f"CAD area {cad_area:.8f} m² not in [{a_lo:.8f},{a_hi:.8f}]"
                )
            if mesh_area > 0.0 and not (a_lo <= mesh_area <= a_hi):
                failures.append(
                    f"mesh triangle area {mesh_area:.8f} m² not in [{a_lo:.8f},{a_hi:.8f}]"
                )
            if r_max > 0.050:
                failures.append(f"radial max {r_max:.6f} m > 0.050 m")
            if z_span > 0.012:
                failures.append(f"z-span {z_span:.6f} m > 0.012 m (not planar)")
            if surf_tags and horiz < len(surf_tags):
                failures.append("not all aperture surfaces horizontal (+Z)")
            if failures:
                raise RuntimeError(
                    "FEM_VALIDATION_MESH post-mesh soundhole aperture gate FAILED: "
                    + "; ".join(failures)
                    + f" (surfaces={surf_tags}, n_cad={len(surf_tags)})."
                )
            gate = {
                "cad_surface_tags": surf_tags,
                "cad_area_m2": float(cad_area),
                "mesh_triangle_area_m2": float(mesh_area),
                "expected_area_m2": float(expected),
                "radial_max_m": float(r_max),
                "z_span_m": float(z_span),
                "gate_pass": True,
            }
            gate_path = out_file.parent / "soundhole_aperture_mesh_gate.json"
            gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
            print(
                "[diag] VALIDATION soundhole aperture post-mesh gate PASS: "
                f"cad_area={cad_area:.8f} m² mesh_area={mesh_area:.8f} m² "
                f"r_max={r_max:.6f} m z_span={z_span:.6f} m surfaces={surf_tags}"
            )

        if is_validation and use_air_opening_tag and not shell_only:
            _validation_gate_soundhole_aperture_post_mesh()

        if is_fom and not shell_only:
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            n_nodes = int(len(node_tags))
            n_tets = 0
            for _dim, vol_tag in gmsh.model.getEntities(3):
                etypes, elem_tags, _ = gmsh.model.mesh.getElements(3, int(vol_tag))
                for etype, tags in zip(etypes, elem_tags):
                    if int(etype) in (4, 11):
                        n_tets += int(len(tags))
            print(f"[diag] FOM mesh size: nodes={n_nodes}, tetrahedra={n_tets}")

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