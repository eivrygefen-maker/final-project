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
        p = config['geometry']
        L, W, D = p['length'], p['width'], p['depth']
        t = p['thickness']
        hr = p['hole_radius']
        shape_type = p.get('shape_type', 'Classical')
    else:
        L, W, D, t, hr, shape_type = 0.48, 0.37, 0.1, 0.003, 0.04, 'Classical'

    # --- Locked mesh target: 12mm global with local wood refinement ---
    if is_preview:
        mesh_size = 0.030
        mesh_size_min = mesh_size
        mesh_size_max = mesh_size
    else:
        mesh_size = 0.012
        mesh_size_min = 0.001
        mesh_size_max = 0.012
    
    print("DEBUG: Forcing Mesh Size to 0.012m (12mm global).")
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

    # Surface classification by boundary relations:
    # interface = shared wood/air boundary; soundhole = external air boundary not in interface.
    wood_boundary_surfs = get_boundary_tags([(3, tag) for tag in wood_vols], 2)
    air_boundary_surfs = get_boundary_tags([(3, tag) for tag in air_vols], 2)

    interface_surfs = sorted(list(wood_boundary_surfs.intersection(air_boundary_surfs)))
    soundhole_surfs = sorted(list(air_boundary_surfs.difference(interface_surfs)))

    if len(soundhole_surfs) != 1:
        raise RuntimeError(f"Expected exactly 1 soundhole opening surface, found {len(soundhole_surfs)}")

    # Build adjacency graph from shared curves among interface surfaces.
    iface_set = set(interface_surfs)
    surf_to_curves = {
        s: get_boundary_tags([(2, s)], 1)
        for s in interface_surfs
    }

    curve_to_surfs = {}
    for s, curves in surf_to_curves.items():
        for c in curves:
            curve_to_surfs.setdefault(c, set()).add(s)

    neighbors = {s: set() for s in interface_surfs}
    for curve, surfs in curve_to_surfs.items():
        if len(surfs) < 2:
            continue
        surfs_list = list(surfs)
        for i in range(len(surfs_list)):
            for j in range(i + 1, len(surfs_list)):
                a, b = surfs_list[i], surfs_list[j]
                neighbors[a].add(b)
                neighbors[b].add(a)

    soundhole_curves = get_boundary_tags([(2, soundhole_surfs[0])], 1)
    top_seeds = [s for s, curves in surf_to_curves.items() if curves.intersection(soundhole_curves)]
    if not top_seeds:
        raise RuntimeError("Could not identify Top_Plate surfaces from soundhole topology")

    # Flood-fill: all interface surfaces connected to soundhole belong to Top_Plate.
    top_plate_set = set()
    stack = list(top_seeds)
    while stack:
        cur = stack.pop()
        if cur in top_plate_set:
            continue
        top_plate_set.add(cur)
        stack.extend(neighbors[cur] - top_plate_set)

    top_plate_surfs = sorted(list(top_plate_set))
    body_surfs = sorted(list(iface_set - top_plate_set))

    # Fallback robustification:
    # In some fragmented topologies, flood-fill can accidentally absorb all interface surfaces.
    # If so, split by surface orientation (normal ~ Z) and top elevation; keep topology as primary.
    if not body_surfs:
        z_values = {s: get_surface_center_z(s) for s in interface_surfs}
        z_top = max(z_values.values()) if z_values else (D / 2.0)
        z_tol = max(1e-4, 0.35 * t)

        fallback_top = set()
        for s in interface_surfs:
            nz_abs = get_surface_normal_z(s)
            zc = z_values[s]

            # Top candidates: near top elevation and mostly horizontal.
            # If normal is unavailable, fallback to elevation-only.
            if (nz_abs is not None and nz_abs >= 0.6 and zc >= (z_top - z_tol)) or \
               (nz_abs is None and zc >= (z_top - z_tol)):
                fallback_top.add(s)

        # Guarantee at least one top surface if heuristics were too strict.
        if not fallback_top and interface_surfs:
            highest = max(interface_surfs, key=lambda s: z_values[s])
            fallback_top.add(highest)

        top_plate_surfs = sorted(list(fallback_top))
        body_surfs = sorted(list(iface_set - fallback_top))

    if not body_surfs:
        raise RuntimeError("Body_Shell classification failed after topology+normal fallback")

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
    gmsh.model.addPhysicalGroup(2, top_plate_surfs, tag=1, name="Top_Plate")
    gmsh.model.addPhysicalGroup(2, soundhole_surfs, tag=2, name="Soundhole")
    gmsh.model.addPhysicalGroup(2, body_surfs, tag=3, name="Body_Shell")
    gmsh.model.addPhysicalGroup(2, wood_fix_surfs, tag=4, name="wood_fix")
    gmsh.model.addPhysicalGroup(3, air_vols, tag=10, name="Air_Internal")
    print(f"[diag] wood_fix surfaces (tag=4): {wood_fix_surfs}")
    air_group_entities = gmsh.model.getEntitiesForPhysicalGroup(3, 10)
    print(f"[diag] Physical Group 10 (Air_Internal) volume entities: {list(air_group_entities)}")
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

    # Local refinement on wood plate/shell surfaces via Distance+Threshold fields.
    if not is_preview:
        wood_surface_tags = top_plate_surfs + body_surfs
        if not top_plate_surfs:
            raise RuntimeError("top_plate_surfs is empty; cannot apply wood refinement field.")
        for surf in wood_surface_tags:
            pts = get_point_tags_from_surface(surf)
            print(f"[DEBUG] Refining surface {surf}: found {len(pts)} boundary points")

        if wood_surface_tags:
            print(f"[DEBUG] All Wood Surfaces to refine: {wood_surface_tags}")
            # Field 1: distance from all wood surfaces.
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "FacesList", wood_surface_tags)

            # Field 2: threshold over distance field.
            thresh_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", 0.0015)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", 0.012)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.001)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 0.015)
            field_type = gmsh.model.mesh.field.getType(thresh_field)
            print(f"[DEBUG] Field {thresh_field} created as type: {field_type}")
            print(
                f"[diag] Distance+Threshold background field enabled on wood surfaces "
                f"(n_surfaces={len(wood_surface_tags)})."
            )
            # Keep this as the very last field instruction before mesh generation.
            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

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