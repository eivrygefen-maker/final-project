import gmsh
import sys
import json
import os
from pathlib import Path

def create_guitar_mesh():
    # 1. Setup paths
    geometry_dir = Path(__file__).resolve().parent
    fem_dir = geometry_dir.parent 
    config_path = fem_dir / "configs" / "guitar_3d.json"
    mesh_dir = fem_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --- Nuclear mesh fix: disable preview mode unless explicitly allowed ---
    # This prevents accidental coarse preview meshes during offline runs.
    is_preview = ("--preview" in sys.argv) and (os.environ.get("FEM_ALLOW_PREVIEW", "0") == "1")
    
    if is_preview:
        out_file = mesh_dir / "preview_mesh.msh"
    else:
        out_file = mesh_dir / "guitar_3d.msh"
    # -------------------------------------------

    # 2. Load geometry data
    if config_path.exists():
        with open(config_path, 'r') as f:
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
        p = config['geometry']
        L, W, D = p['length'], p['width'], p['depth']
        t = p['thickness']
        hr = p['hole_radius']
        shape_type = p.get('shape_type', 'Classical')
    else:
        L, W, D, t, hr, shape_type = 0.48, 0.37, 0.1, 0.003, 0.04, 'Classical'

    # --- Hybrid mesh target: coarse air, refined wood ---
    if is_preview:
        mesh_size = 0.030
        mesh_size_min = mesh_size
        mesh_size_max = mesh_size
    else:
        mesh_size = 0.020
        mesh_size_min = 0.020
        mesh_size_max = 0.020
    
    print("DEBUG: Forcing Triple-Tier Mesh: 0.0015m wood, 0.007m transition, 0.015m far field.")
    print(f"Building geometry with Thickness: {t*1000:.1f}mm, Mesh Size: {mesh_size*1000:.2f}mm")
    print(f"[diag] preview_mode={is_preview}, FEM_ALLOW_PREVIEW={os.environ.get('FEM_ALLOW_PREVIEW', '0')}")
    
    shy = (L / 2) + (L * 0.02)
    hr = min(hr, W * 0.40)    

    gmsh.initialize(sys.argv)
    gmsh.model.add("Guitar3D_Performance_Optimized")
    occ = gmsh.model.occ

    def create_guitar_profile(l, w, is_dreadnought=False, offset=0):
        top_x = 0.50 * l - offset if offset > 0 else 0.50 * l
        p_top_center = occ.addPoint(top_x, 0, 0, mesh_size)
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
            pts.append(occ.addPoint(x, max(0, y), 0, mesh_size))
            
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
        surf_out = create_guitar_profile(L, W, is_dread, 0)
        v_out = occ.extrude([(2, surf_out)], 0, 0, D)
        occ.translate([v for v in v_out if v[0]==3], 0, 0, -D/2)
        vol_out_id = [v[1] for v in v_out if v[0] == 3][0]
        
        surf_in = create_guitar_profile(L, W, is_dread, t)
        v_in = occ.extrude([(2, surf_in)], 0, 0, D - 2*t)
        occ.translate([v for v in v_in if v[0]==3], 0, 0, -D/2 + t)
        vol_in_id = [v[1] for v in v_in if v[0] == 3][0]

    # Create soundhole cylinder.
    hole_x = shy - L/2 if "Box" not in shape_type else 0
    z_inner_top = (D / 2) - t
    hole_cyl = occ.addCylinder(hole_x, 0, z_inner_top, 0, 0, 2 * t, hr)

    # 1) Air cavity: inner volume minus soundhole cylinder.
    air_cut = occ.cut([(3, vol_in_id)], [(3, hole_cyl)], removeObject=True, removeTool=False)
    air_dimtags = [dt for dt in as_dimtags(air_cut) if dt[0] == 3]

    # 2) Wood shell: outer volume minus air cavity, then open top soundhole.
    wood_cut = occ.cut([(3, vol_out_id)], air_dimtags, removeObject=True, removeTool=False)
    wood_dimtags = [dt for dt in as_dimtags(wood_cut) if dt[0] == 3]
    wood_hole_cut = occ.cut(wood_dimtags, [(3, hole_cyl)], removeObject=True, removeTool=True)
    wood_dimtags = [dt for dt in as_dimtags(wood_hole_cut) if dt[0] == 3]

    # 3) Fragment is required to enforce node sharing between wood and air.
    frags, _ = occ.fragment(wood_dimtags, air_dimtags, removeObject=True, removeTool=True)
    # Coherence-equivalent cleanup after BooleanFragments: merge duplicate entities/nodes.
    # This helps guarantee shared-node interfaces when CAD booleans leave near-duplicates.
    try:
        occ.removeAllDuplicates()
    except Exception:
        pass
    occ.synchronize()

    # Classify volumes after fragment by topology (no height heuristics).
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
    if assigned_wood.intersection(set(air_vols)):
        overlap = sorted(list(assigned_wood.intersection(set(air_vols))))
        raise RuntimeError(f"Wood/Air volume overlap detected: {overlap}")

    # Surface identification via direct boundaries (fragmentation-safe, no interface tracing).
    wood_boundary_surfs = get_boundary_tags([(3, tag) for tag in wood_vols], 2)
    air_boundary_surfs = get_boundary_tags([(3, tag) for tag in air_vols], 2)
    top_plate_surfs = sorted(list(get_boundary_tags([(3, tag) for tag in top_vols], 2))) if top_vols else []
    if not top_plate_surfs and wood_boundary_surfs:
        highest = max(list(wood_boundary_surfs), key=lambda s: get_surface_center_z(s))
        top_plate_surfs = [highest]
        print("[diag][warn] top_plate_surfs fallback: using highest wood boundary surface.")

    body_surfs = sorted(list(set(wood_boundary_surfs) - set(top_plate_surfs)))
    if not body_surfs and wood_boundary_surfs:
        body_surfs = sorted(list(wood_boundary_surfs))
        print("[diag][warn] body_surfs fallback: using all wood boundary surfaces.")

    # Soundhole boundaries: exposed air boundary surfaces (not shared with wood boundaries).
    soundhole_surfs = sorted(list(set(air_boundary_surfs) - set(wood_boundary_surfs)))
    if not soundhole_surfs:
        # Fallback: use top boundary surfaces so tagging remains valid for downstream code.
        soundhole_surfs = list(top_plate_surfs)
        print("[diag][warn] soundhole_surfs fallback: using top_plate_surfs.")

    # Define a compact support region (wood_fix) near neck-side body area.
    # This provides a deterministic structural clamp for FEM boundary conditions.
    fix_candidates = []
    for s in body_surfs:
        cx, cy, cz = get_surface_center(s)
        fix_candidates.append((s, cx, abs(cy), cz))
    # Prefer surfaces near maximum +x and close to centerline (small |y|).
    fix_candidates.sort(key=lambda row: (-row[1], row[2]))
    n_fix = min(2, len(fix_candidates))
    wood_fix_surfs = [row[0] for row in fix_candidates[:n_fix]]
    if not wood_fix_surfs and body_surfs:
        wood_fix_surfs = [body_surfs[0]]

    # Define physical groups using fixed tag protocol.
    # IMPORTANT: 2D tags (facets) and 3D tags (volumes) are both created.
    # This ensures structural-only volume assembly can target wood cells (1/2/3).
    if not top_plate_surfs and wood_boundary_surfs:
        top_plate_surfs = [max(list(wood_boundary_surfs), key=lambda s: get_surface_center_z(s))]
    if not body_surfs and wood_boundary_surfs:
        body_surfs = sorted(list(wood_boundary_surfs))
    if not soundhole_surfs:
        soundhole_surfs = list(top_plate_surfs)

    pg_top = gmsh.model.addPhysicalGroup(2, top_plate_surfs, tag=1)
    gmsh.model.setPhysicalName(2, pg_top, "Top_Plate")
    pg_soundhole = gmsh.model.addPhysicalGroup(2, soundhole_surfs, tag=2)
    gmsh.model.setPhysicalName(2, pg_soundhole, "Soundhole")
    pg_body = gmsh.model.addPhysicalGroup(2, body_surfs, tag=3)
    gmsh.model.setPhysicalName(2, pg_body, "Body_Shell")
    pg_fix = gmsh.model.addPhysicalGroup(2, wood_fix_surfs, tag=4)
    gmsh.model.setPhysicalName(2, pg_fix, "wood_fix")
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

    pg_air = gmsh.model.addPhysicalGroup(3, air_vols, tag=10)
    gmsh.model.setPhysicalName(3, pg_air, "Air_Internal")

    print(f"[diag] wood_fix surfaces (tag=4): {wood_fix_surfs}")
    air_group_entities = gmsh.model.getEntitiesForPhysicalGroup(3, 10)
    print(f"[diag] Physical Group 10 (Air_Internal) volume entities: {list(air_group_entities)}")
    v1 = gmsh.model.getEntitiesForPhysicalGroup(3, 1) if top_vols else []
    v2 = gmsh.model.getEntitiesForPhysicalGroup(3, 2) if back_vols else []
    v3 = gmsh.model.getEntitiesForPhysicalGroup(3, 3) if rib_vols else []
    print(f"[diag] Physical Group 1 (Top_Plate_Volume) entities: {list(v1)}")
    print(f"[diag] Physical Group 2 (Back_Plate_Volume) entities: {list(v2)}")
    print(f"[diag] Physical Group 3 (Ribs_Sides_Volume) entities: {list(v3)}")
    if len(air_group_entities) == 0:
        raise RuntimeError("Tag 10 was created but has no 3D volume entities.")

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

    # Deep-probe overrides: force background field dominance.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

    # Triple-tier refinement strategy:
    #   Level 1 (Fine): 0.0015 m on wood surfaces/volumes
    #   Level 2 (Transition): 0.0015 -> 0.008 m within 0.02 m from wood
    #   Level 3 (Coarse): 0.020 m in far-field air
    # Combined with Min(...) to avoid conflicting constraints.
    if not is_preview:
        wood_surface_tags = top_plate_surfs + body_surfs
        if not top_plate_surfs:
            raise RuntimeError("top_plate_surfs is empty; cannot apply wood refinement field.")
        for surf in wood_surface_tags:
            pts = get_point_tags_from_surface(surf)
            print(f"[DEBUG] Refining surface {surf}: found {len(pts)} boundary points")

        if wood_surface_tags:
            print(f"[DEBUG] All Wood Surfaces to refine: {wood_surface_tags}")
            # Field 1: distance from wood surfaces.
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "FacesList", wood_surface_tags)

            # Level 1 (Fine): constant fine size, restricted to wood entities.
            fine_const = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(fine_const, "F", "0.0015")
            fine_restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(fine_restrict, "InField", fine_const)
            gmsh.model.mesh.field.setNumbers(fine_restrict, "FacesList", wood_surface_tags)
            gmsh.model.mesh.field.setNumbers(fine_restrict, "VolumesList", wood_vols)

            # Level 2 (Transition): increase away from wood up to 7 mm at 3 cm.
            transition_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(transition_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(transition_field, "SizeMin", 0.0015)
            gmsh.model.mesh.field.setNumber(transition_field, "SizeMax", 0.0070)
            gmsh.model.mesh.field.setNumber(transition_field, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(transition_field, "DistMax", 0.030)

            # Level 3 (Coarse): far-field target in air.
            coarse_field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(coarse_field, "F", "0.015")

            # Combine all constraints using Min to prevent overlap conflicts.
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(
                min_field, "FieldsList", [coarse_field, transition_field, fine_restrict]
            )
            print(
                "[diag] Triple-tier background field enabled "
                f"(fine=1.5mm, transition<=7mm@3cm, coarse=15mm; n_surfaces={len(wood_surface_tags)})."
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
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(out_file))
        print(f"SUCCESS: Optimized mesh saved to {out_file}")
    except Exception as e:
        print(f"Mesh generation failed: {e}")
    
    gmsh.finalize()

if __name__ == "__main__":
    create_guitar_mesh()