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
            vol_out_id, wall_t=t, inner_depth=inner_depth
        )
        print(
            f"[diag] reference shell tools: outer={vol_out_id} inner={vol_in_id}"
        )

    hole_cyl: Optional[int] = None
    if not solid_sketch:
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