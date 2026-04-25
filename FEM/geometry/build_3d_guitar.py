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
    out_file = mesh_dir / "guitar_3d.msh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load geometry data
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        p = config['geometry']
        L, W, D, t, hr = p['length'], p['width'], p['depth'], p['thickness'], p['hole_radius']
        shape_type = p.get('shape_type', 'Classical')
        vis_mode = p.get('vis_mode', 'Mesh + Solid (Wood)')
        exploded = p.get('exploded_view', False)
        
        m_top_name = config.get('materials', {}).get('top', {}).get('name', 'Spruce')
        m_back_name = config.get('materials', {}).get('back', {}).get('name', 'Mahogany')
    else:
        L, W, D, t, hr, shape_type, vis_mode, exploded = 0.48, 0.37, 0.1, 0.005, 0.04, 'Classical', 'Mesh + Solid (Wood)', False
        m_top_name, m_back_name = 'Spruce', 'Mahogany'

    t = max(0.002, min(t, D / 4.0))
    hr = min(hr, W * 0.15)    

    gmsh.initialize(sys.argv)
    gmsh.model.add("Guitar3D")
    occ = gmsh.model.occ

    # --- Geometry Creation (הלוגיקה המלאה שלך) ---
    def create_guitar_profile(l, w, is_dreadnought=False, offset=0):
        top_x = 0.50 * l - offset if offset > 0 else 0.50 * l
        p_top_center = occ.addPoint(top_x, 0, 0)
        x_facs = [0.50, 0.44, 0.25, 0.05, -0.10, -0.25, -0.40, -0.48, -0.50] if is_dreadnought else [0.50, 0.44, 0.25, 0.10, 0.00, -0.15, -0.35, -0.45, -0.50]
        y_facs = [0.08, 0.20, 0.38, 0.35, 0.34, 0.45, 0.50, 0.25, 0.00] if is_dreadnought else [0.08, 0.18, 0.36, 0.30, 0.28, 0.45, 0.50, 0.30, 0.00]
        pts = []
        for x_f, y_f in zip(x_facs, y_facs):
            x, y = x_f * l, y_f * w
            if offset > 0:
                if y_f > 0: y -= offset
                if x_f == 0.5: x -= offset
                elif x_f == -0.5: x += offset
            pts.append(occ.addPoint(x, max(0, y), 0))
        l_top = occ.addLine(p_top_center, pts[0]); c_body = occ.addSpline(pts)
        m_l = occ.copy([(1, l_top)]); occ.mirror(m_l, 0, 1, 0, 0)
        m_c = occ.copy([(1, c_body)]); occ.mirror(m_c, 0, 1, 0, 0)
        loop = occ.addCurveLoop([l_top, c_body, -m_c[0][1], -m_l[0][1]])
        return occ.addPlaneSurface([loop])

    if "Box" in shape_type:
        outer = occ.addBox(-L/2, -W/2, -D/2, L, W, D)
        inner = occ.addBox(-L/2+t, -W/2+t, -D/2+t, L-2*t, W-2*t, D-2*t)
        shell = occ.cut([(3, outer)], [(3, inner)])
        shell_id = shell[0][0][1]
    else:
        is_dread = "Dreadnought" in shape_type
        v_out = occ.extrude([(2, create_guitar_profile(L, W, is_dread, 0))], 0, 0, D)
        occ.translate([v for v in v_out if v[0]==3], 0, 0, -D/2)
        v_in = occ.extrude([(2, create_guitar_profile(L, W, is_dread, t))], 0, 0, D - 2*t)
        occ.translate([v for v in v_in if v[0]==3], 0, 0, -D/2 + t)
        shell = occ.cut([v for v in v_out if v[0]==3], [v for v in v_in if v[0]==3])
        shell_id = shell[0][0][1]

    hole = occ.addCylinder(L*0.02 if "Box" not in shape_type else 0, 0, D/2-2*t, 0, 0, 4*t, hr)
    guitar_raw = occ.cut([(3, shell_id)], [(3, hole)])
    raw_id = guitar_raw[0][0][1]

    # Fragmenting
    cutter = occ.addBox(-L, -W, D/2 - t, 2*L, 2*W, t + 0.001)
    res, res_map = occ.fragment([(3, raw_id)], [(3, cutter)])
    occ.synchronize()

    # Identification
    to_remove = []; top_vol = None; back_vol_list = []
    for dim, tag in gmsh.model.getEntities(3):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if (xmax - xmin) > L*1.5: to_remove.append((dim, tag)); continue
        if zmax > (D/2 - 0.01) and (zmax - zmin) < (t + 0.01):
            top_vol = tag
            if exploded: occ.translate([(3, tag)], 0, 0, 0.04)
        else: back_vol_list.append(tag)

    if to_remove: occ.remove(to_remove, recursive=True); occ.synchronize()

    # 5. Physical Groups
    if top_vol: gmsh.model.addPhysicalGroup(3, [top_vol], name=f"Top_{m_top_name}")
    if back_vol_list: gmsh.model.addPhysicalGroup(3, back_vol_list, name=f"Back_{m_back_name}")

    # 6. Mesh Generation
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.008)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02) 
    try:
        gmsh.model.mesh.generate(3)
        gmsh.write(str(out_file))
    except: pass
    
    # 7. Visualization (SAFE MODE - מונע קריסה)
    def safe_option(func, *args):
        try:
            func(*args)
        except:
            pass # אם הפקודה לא נתמכת בגרסה הזו, פשוט מדלגים

    # צביעת משטחים לפתרון הבעיה הסגולה
    def color_surfaces(vol_tag, r, g, b):
        try:
            boundary = gmsh.model.getBoundary([(3, vol_tag)], combined=True, oriented=False)
            surfs = [b[1] for b in boundary if b[0] == 2]
            gmsh.model.setColor([(2, s) for s in surfs], r, g, b, 255)
        except: pass

    # יישום צבעים
    if top_vol: color_surfaces(top_vol, 222, 184, 135) # חום בהיר
    for v in back_vol_list: color_surfaces(v, 101, 56, 24) # חום כהה

    # הגדרות תצוגה
    safe_option(gmsh.option.setNumber, "Mesh.ColorCarousel", 0)
    
    if "Solid Wood" in vis_mode:
        safe_option(gmsh.option.setNumber, "Mesh.SurfaceFaces", 1)
        safe_set_lines = lambda: gmsh.option.setNumber("Mesh.Lines", 0)
        safe_option(safe_set_lines)
    else: # Mesh + Solid
        safe_option(gmsh.option.setNumber, "Mesh.SurfaceFaces", 1)
        safe_option(gmsh.option.setNumber, "Mesh.Lines", 1)
        try: gmsh.option.setColor("Mesh.Lines", 50, 25, 10, 255)
        except: pass

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    
    gmsh.finalize()

if __name__ == "__main__":
    create_guitar_mesh()