import gmsh
import sys
import json
import os
import math
from pathlib import Path

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

    # Preview: env gate (Streamlit) + optional ``--preview`` script flag; never a Gmsh CLI option.
    preview_cli, _nopopup = _script_flags()
    is_preview = (os.environ.get("FEM_ALLOW_PREVIEW", "0") == "1") or preview_cli
    
    if is_preview:
        out_file = mesh_dir / "preview_mesh.msh"
    else:
        out_file = mesh_dir / "guitar_3d.msh"
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
        hr = p["hole_radius"]
        shape_type = p.get('shape_type', 'Classical')
        hole_y = float(p.get("soundhole_y", 0.0))
    else:
        L, W, D, t, hr, shape_type = 0.48, 0.37, 0.1, 0.003, 0.04, 'Classical'
        hole_y = 0.0

    # --- Golden mesh: graded wood faces (6.5 mm), dense through-thickness (1 mm); graded air ---
    wood_surface_size = 0.0065   # 6.5 mm on large top/back plate surfaces and long perimeter curves
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

    # Preview: identical CAD/booleans; only mesh sizing is coarser (not shape creation).
    if is_preview:
        mesh_size = 0.014
        mesh_size_min = 0.006
        mesh_size_max = 0.028
    else:
        mesh_size = wood_surface_size
        mesh_size_min = wood_thickness_size
        mesh_size_max = air_threshold_size_max

    print(
        "DEBUG: Golden mesh — wood: 6.5 mm faces / 1 mm thickness edges; "
        "air: 8 mm near wood → 80 mm far (Dist 1.5–25 cm)."
    )
    print(
        f"Building geometry with Thickness: {t*1000:.1f}mm, "
        f"wood_surface_lc={wood_surface_size*1000:.1f}mm, wood_thickness_lc={wood_thickness_size*1000:.1f}mm"
    )
    print(f"[diag] preview_mode={is_preview}, FEM_ALLOW_PREVIEW={os.environ.get('FEM_ALLOW_PREVIEW', '0')}")

    shy = (L / 2) + (L * 0.02)
    # No artificial radius cap — OCC cut/fragment clips overflow at plate edges.
    hr = max(1.0e-4, float(hr))

    gmsh.initialize(_gmsh_initialize_argv())
    gmsh.model.add("Guitar3D_Performance_Optimized")
    occ = gmsh.model.occ

    def create_guitar_profile(l, w, is_dreadnought=False, offset=0, point_lc=None):
        lc = point_lc if point_lc is not None else mesh_size
        top_x = 0.50 * l - offset if offset > 0 else 0.50 * l
        p_top_center = occ.addPoint(top_x, 0, 0, lc)
        x_facs = [0.50, 0.44, 0.25, 0.05, -0.10, -0.25, -0.40, -0.48, -0.50] if is_dreadnought else [0.50, 0.44, 0.25, 0.10, 0.00, -0.15, -0.35, -0.45, -0.50]
        y_facs = [0.08, 0.20, 0.38, 0.35, 0.34, 0.45, 0.50, 0.25, 0.00] if is_dreadnought else [0.08, 0.18, 0.36, 0.30, 0.28, 0.45, 0.50, 0.30, 0.00]
        
        pts = []
        for i, (x_f, y_f) in enumerate(zip(x_facs, y_facs)):
            x = x_f * l
            y = y_f * w
            if offset > 0:
                y = max(0, y - offset)
                if x_f == 0.5: x -= offset
                elif x_f == -0.5: x += offset
            pts.append(occ.addPoint(x, max(0, y), 0, lc))
            
        l_top = occ.addLine(p_top_center, pts[0]); c_body = occ.addSpline(pts)
        m_l = occ.copy([(1, l_top)]); occ.mirror(m_l, 0, 1, 0, 0)
        m_c = occ.copy([(1, c_body)]); occ.mirror(m_c, 0, 1, 0, 0)
        loop = occ.addCurveLoop([l_top, c_body, -m_c[0][1], -m_l[0][1]])
        return occ.addPlaneSurface([loop])

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

    def get_surface_center_z(surf_tag):
        com = occ.getCenterOfMass(2, surf_tag)
        return com[2]

    def get_surface_center(surf_tag):
        return occ.getCenterOfMass(2, surf_tag)

    def get_volume_center_z(vol_tag):
        com = occ.getCenterOfMass(3, vol_tag)
        return com[2]

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

    def get_surface_normal_z(surf_tag):
        """
        Return |nz| at parametric midpoint of a surface.
        Falls back to None if the CAD kernel/API cannot provide a normal.
        """
        try:
            uv_min, uv_max = gmsh.model.getParametrizationBounds(2, surf_tag)
            u_mid = 0.5 * (uv_min[0] + uv_max[0])
            v_mid = 0.5 * (uv_min[1] + uv_max[1])
            n = gmsh.model.getNormal(surf_tag, [u_mid, v_mid])
            if n and len(n) >= 3:
                return abs(float(n[2]))
        except Exception:
            return None
        return None

    # Build guitar solid and internal air domain (updated step 1).
    if "Box" in shape_type:
        vol_out_id = occ.addBox(-L/2, -W/2, -D/2, L, W, D)
        vol_in_id = occ.addBox(-L/2+t, -W/2+t, -D/2+t, L-2*t, W-2*t, D-2*t)
    else:
        is_dread = "Dreadnought" in shape_type
        # Fine spline control points even in preview (silhouette fidelity).
        profile_lc = wood_surface_size
        surf_out = create_guitar_profile(L, W, is_dread, 0, point_lc=profile_lc)
        # OCC extrude: numElements may be ignored by OpenCASCADE; thickness resolution is enforced via
        # curve/field sizing below. When supported, [3] requests prism stacks along the extrusion axis.
        try:
            v_out = occ.extrude([(2, surf_out)], 0, 0, D, [3])
        except (TypeError, ValueError):
            v_out = occ.extrude([(2, surf_out)], 0, 0, D)
        occ.translate([v for v in v_out if v[0]==3], 0, 0, -D/2)
        vol_out_id = [v[1] for v in v_out if v[0] == 3][0]
        
        surf_in = create_guitar_profile(L, W, is_dread, t, point_lc=profile_lc)
        try:
            v_in = occ.extrude([(2, surf_in)], 0, 0, D - 2*t, [3])
        except (TypeError, ValueError):
            v_in = occ.extrude([(2, surf_in)], 0, 0, D - 2*t)
        occ.translate([v for v in v_in if v[0]==3], 0, 0, -D/2 + t)
        vol_in_id = [v[1] for v in v_in if v[0] == 3][0]

    # Create soundhole cylinder (axis +z; center (hole_x, hole_y) from geometry config).
    hole_x = shy - L/2 if "Box" not in shape_type else 0
    z_inner_top = (D / 2) - t
    hole_cyl = occ.addCylinder(hole_x, hole_y, z_inner_top, 0, 0, 2 * t, hr)

    if is_preview:
        # UI sketch: hollow wood shell only — no acoustic air volume or outer air box.
        print("[diag] preview CAD: wood shell + soundhole cut (air domain disabled)")
        wood_shell = occ.cut([(3, vol_out_id)], [(3, vol_in_id)], removeObject=True, removeTool=True)
        wood_dimtags = [dt for dt in as_dimtags(wood_shell) if dt[0] == 3]
        wood_hole_cut = occ.cut(wood_dimtags, [(3, hole_cyl)], removeObject=True, removeTool=True)
        wood_dimtags = [dt for dt in as_dimtags(wood_hole_cut) if dt[0] == 3]
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
        air_cut = occ.cut([(3, vol_in_id)], [(3, hole_cyl)], removeObject=True, removeTool=False)
        air_dimtags = [dt for dt in as_dimtags(air_cut) if dt[0] == 3]

        wood_cut = occ.cut([(3, vol_out_id)], air_dimtags, removeObject=True, removeTool=False)
        wood_dimtags = [dt for dt in as_dimtags(wood_cut) if dt[0] == 3]
        wood_hole_cut = occ.cut(wood_dimtags, [(3, hole_cyl)], removeObject=True, removeTool=True)
        wood_dimtags = [dt for dt in as_dimtags(wood_hole_cut) if dt[0] == 3]

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

    # If booleans leave wood as a single connected solid, explicitly partition it
    # into multiple disjoint 3D volumes along Z so Top/Back/Ribs volume groups
    # are all non-empty and mutually exclusive.
    if len(wood_vols) < 3:
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

    if not is_preview:
        # Final geometry unification: re-fragment wood+air for shared FSI interfaces.
        try:
            air_ref_com = occ.getCenterOfMass(3, int(air_vols[0])) if air_vols else (0.0, 0.0, 0.0)
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
                    cx, cy, cz = occ.getCenterOfMass(3, int(vtag))
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
    air_boundary_surfs = (
        get_boundary_tags([(3, tag) for tag in air_vols], 2) if air_vols else []
    )
    top_plate_surfs = sorted(list(get_boundary_tags([(3, tag) for tag in top_vols], 2))) if top_vols else []
    if not top_plate_surfs and wood_boundary_surfs:
        highest = max(list(wood_boundary_surfs), key=lambda s: get_surface_center_z(s))
        top_plate_surfs = [highest]
        print("[diag][warn] top_plate_surfs fallback: using highest wood boundary surface.")

    back_plate_surfs = (
        sorted(list(get_boundary_tags([(3, int(v)) for v in back_vols], 2))) if back_vols else []
    )
    rib_surfs = (
        sorted(list(get_boundary_tags([(3, int(v)) for v in rib_vols], 2))) if rib_vols else []
    )
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
    _back_set = set(back_plate_surfs)
    rib_surfs = [s for s in rib_surfs if s not in _back_set]
    if not back_plate_surfs and back_vols:
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
    _add_surface_physical_group(top_plate_surfs, 1, "Top_Plate", required=True)
    _add_surface_physical_group(soundhole_surfs, 2, "Soundhole", required=False)
    _add_surface_physical_group(back_plate_surfs, 3, "Back_Plate", required=True)
    _add_surface_physical_group(rib_surfs, 4, "Ribs_Sides", required=True)
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

    if not is_preview:
        pg_air = gmsh.model.addPhysicalGroup(3, air_vols, tag=10)
        gmsh.model.setPhysicalName(3, pg_air, "Air_Internal")

    print(
        f"[diag] facet groups: top={len(top_plate_surfs)}, back={len(back_plate_surfs)}, "
        f"ribs={len(rib_surfs)}, wood_fix={len(wood_fix_surfs)}, preview={is_preview}"
    )
    if not is_preview:
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
    
    # Improve circular feature fidelity using curvature-aware mesh sizing.
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    # Enforce at least 36 points per full circle (~10 degrees per point).
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 36) 
    # ---------------------------------------------
    
    gmsh.model.mesh.setOrder(1)
    mesh_resolution_factor = 1.0
    print(f"[diag] mesh_resolution_factor={mesh_resolution_factor}")

    if is_preview:
        preview_hole_lc = max(0.006, min(mesh_size, 0.45 * hr))
        for s in soundhole_surfs:
            try:
                gmsh.model.mesh.setSize(2, int(s), preview_hole_lc)
            except Exception:
                pass
        print(
            f"[diag] preview local sizing: global_lc={mesh_size*1000:.1f}mm, "
            f"soundhole_lc={preview_hole_lc*1000:.1f}mm, hole_r={hr*1000:.1f}mm"
        )

    # Deep-probe overrides: force background field dominance.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

    # Wood: 6.5 mm baseline on shell (Restrict), 1 mm in a band around short thickness edges
    # (mesh.setSize + Threshold from EdgesList) so P1 has multiple elements across ~3 mm wood.
    # Air: distance from wood shell -> Threshold (8 mm near field, 80 mm far, smooth 1.5–25 cm band).
    if not is_preview:
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
        gmsh.model.mesh.generate(3)

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

        facet_audit = {t: _count_mesh_elements_for_physical(2, t) for t in (1, 2, 3, 4)}
        print(f"[diag] post-generate facet element counts by physical tag: {facet_audit}")
        for req_tag, label in ((1, "Top"), (3, "Back"), (4, "Ribs")):
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
        print(f"Mesh generation failed: {e}")
    
    gmsh.finalize()

if __name__ == "__main__":
    create_guitar_mesh()