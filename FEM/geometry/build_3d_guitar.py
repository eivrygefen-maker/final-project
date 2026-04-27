import gmsh
import sys
import json
from pathlib import Path

def create_guitar_mesh():
    # 1. Setup paths
    geometry_dir = Path(__file__).resolve().parent
    fem_dir = geometry_dir.parent 
    config_path = fem_dir / "configs" / "guitar_3d.json"
    mesh_dir = fem_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --- התיקון שלנו: זיהוי מצב תצוגה מקדימה ---
    is_preview = "--preview" in sys.argv
    
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

    # --- התיקון שלנו: שינוי צפיפות הרשת בהתאם למצב ---
    if is_preview:
        mesh_size = 0.030  # רשת של 30 מ"מ במקום 80, כדי שהמנוע לא יקרוס בעובי דק!
    else:
        mesh_size = t * 1.4
    
    print(f"Building geometry with Thickness: {t*1000:.1f}mm, Mesh Size: {mesh_size*1000:.2f}mm")
    
    shy = (L / 2) + (L * 0.02)
    hr = min(hr, W * 0.40)    

    gmsh.initialize(sys.argv)
    gmsh.model.add("Guitar3D_Performance_Optimized")
    occ = gmsh.model.occ

    def create_guitar_profile(l, w, is_dreadnought=False, offset=0):
        top_x = 0.50 * l - offset if offset > 0 else 0.50 * l
        p_top_center = occ.addPoint(top_x, 0, 0)
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
            pts.append(occ.addPoint(x, max(0, y), 0))
            
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

    # --- בניית הגיאומטריה והאוויר (שלב 1 המעודכן) ---
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

    # יצירת הצילינדר של חור התהודה
    hole_x = shy - L/2 if "Box" not in shape_type else 0
    z_inner_top = (D / 2) - t
    hole_cyl = occ.addCylinder(hole_x, 0, z_inner_top, 0, 0, 2 * t, hr)

    # 1) חלל האוויר: חלל פנימי פחות חור התהודה
    air_cut = occ.cut([(3, vol_in_id)], [(3, hole_cyl)], removeObject=True, removeTool=False)
    air_dimtags = [dt for dt in as_dimtags(air_cut) if dt[0] == 3]

    # 2) מעטפת העץ: נפח חיצוני פחות חלל פנימי, ואז פתיחת החור בלוח העליון
    wood_cut = occ.cut([(3, vol_out_id)], air_dimtags, removeObject=True, removeTool=False)
    wood_dimtags = [dt for dt in as_dimtags(wood_cut) if dt[0] == 3]
    wood_hole_cut = occ.cut(wood_dimtags, [(3, hole_cyl)], removeObject=True, removeTool=True)
    wood_dimtags = [dt for dt in as_dimtags(wood_hole_cut) if dt[0] == 3]

    # 3) Fragment חובה לשיתוף צמתים בין העץ והאוויר
    frags, _ = occ.fragment(wood_dimtags, air_dimtags, removeObject=True, removeTool=True)
    occ.synchronize()

    # סיווג נפחים אחרי ה-fragment לפי טופולוגיה (ללא ניחוש גבהים)
    resulting_vols = [dt for dt in frags if dt[0] == 3]
    air_candidate_set = set(tag for _, tag in air_dimtags)
    air_vols = [tag for _, tag in resulting_vols if tag in air_candidate_set]
    wood_vols = [tag for _, tag in resulting_vols if tag not in air_candidate_set]

    if len(air_vols) != 1:
        raise RuntimeError(f"Expected exactly 1 internal air volume, found {len(air_vols)}")

    if not wood_vols:
        raise RuntimeError("No wood shell volumes found after boolean operations")

    # זיהוי משטחים לפי קשרי גבול:
    # interface = גבול משותף עץ/אוויר, soundhole = גבול אוויר חיצוני שאינו interface
    wood_boundary_surfs = get_boundary_tags([(3, tag) for tag in wood_vols], 2)
    air_boundary_surfs = get_boundary_tags([(3, tag) for tag in air_vols], 2)

    interface_surfs = sorted(list(wood_boundary_surfs.intersection(air_boundary_surfs)))
    soundhole_surfs = sorted(list(air_boundary_surfs.difference(interface_surfs)))

    if len(soundhole_surfs) != 1:
        raise RuntimeError(f"Expected exactly 1 soundhole opening surface, found {len(soundhole_surfs)}")

    # בניית גרף שכנויות על בסיס קווים משותפים בין משטחי ה-interface
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

    # flood-fill: כל משטחי ה-interface המחוברים ל-soundhole הם Top_Plate
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

    if not body_surfs:
        raise RuntimeError("Body_Shell classification failed: no remaining interface surfaces")

    # הגדרת הקבוצות הפיזיקליות לפי תגים קבועים
    gmsh.model.addPhysicalGroup(2, top_plate_surfs, tag=1, name="Top_Plate")
    gmsh.model.addPhysicalGroup(2, soundhole_surfs, tag=2, name="Soundhole")
    gmsh.model.addPhysicalGroup(2, body_surfs, tag=3, name="Body_Shell")
    gmsh.model.addPhysicalGroup(3, air_vols, tag=10, name="Air_Internal")

    # הגדרת רשת האלמנטים המאוזנת
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    
    # --- התיקון שלנו: עקמומיות מושלמת למעגלים ---
    # מפעיל אלגוריתם שמתאים את גודל הרשת לפי הקימור של הצורה
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    # מכריח את המערכת להשתמש במינימום 36 נקודות למעגל שלם (כל 10 מעלות = נקודה)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 36) 
    # ---------------------------------------------
    
    gmsh.model.mesh.setOrder(1)

    try:
        print(f"Generating optimized mesh (Density: {mesh_size*1000}mm, Wall: {t*1000}mm)...")
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(out_file))
        print(f"SUCCESS: Optimized mesh saved to {out_file}")
    except Exception as e:
        print(f"Mesh generation failed: {e}")
    
    gmsh.finalize()

if __name__ == "__main__":
    create_guitar_mesh()