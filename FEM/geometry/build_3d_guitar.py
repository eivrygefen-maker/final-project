import gmsh
import sys
import json
import os
import numpy as np
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
        hr = p['hole_radius']
        shape_type = p.get('shape_type', 'Classical')
        vis_mode = p.get('vis_mode', 'Mesh + Solid (Wood)')
    else:
        L, W, D, hr, shape_type, vis_mode = 0.48, 0.37, 0.1, 0.04, 'Classical', 'Mesh + Solid (Wood)'

    # --- התיקון שלנו: שינוי צפיפות הרשת בהתאם למצב ---
    t = config['geometry']['thickness']
    
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
    hole_cyl = occ.addCylinder(hole_x, 0, D/2 - 2*t, 0, 0, 4*t, hr)
    
    # שימוש ב-Fragment במקום Cut כדי להשאיר את האוויר בתוך העץ [cite: 141, 150]
    # זה יוצר "שיתוף נקודות" (shared nodes) הכרחי לצימוד האקוסטי [cite: 128]
    occ.fragment([(3, vol_out_id)], [(3, vol_in_id), (3, hole_cyl)])
    occ.synchronize()

    # --- שלב 1 המלוטש: אופטימיזציה ויזואלית וחישובית ---
    
    # 1. זיהוי נפח האוויר - דילול משמעותי לקיצור זמן ה-JSON
    air_vols = []
    for dim, tag in gmsh.model.getEntities(3):
        com = occ.getCenterOfMass(dim, tag)
        if np.linalg.norm(com) < min(L, W)/3:
            air_vols.append(tag)
            # רשת גסה מאוד לאוויר (פי 6 מהעץ) - זה יגרום ל-JSON לרוץ בשניות
            gmsh.model.mesh.setSize([(3, tag)], mesh_size * 6) 
    

    # --- תיקון: הפרדה ל-Top ו-Body והוספת נפח אוויר ---
    top_plate_surfs = []
    body_surfs = []
    soundhole_surfs = []
    
    for dim, tag in gmsh.model.getEntities(2):
        com = occ.getCenterOfMass(dim, tag)
        dist_from_hole = np.linalg.norm(com[:2] - np.array([hole_x, 0]))
        is_at_top = np.isclose(com[2], D/2, atol=1e-3)
        
        # זיהוי אזור החור
        if dist_from_hole < hr * 0.95:
            if is_at_top:
                soundhole_surfs.append(tag)
            else:
                continue # "פקק" פנימי - לא מוסיפים לקבוצה
        # הפרדה בין הלוח העליון לשאר הגוף
        elif is_at_top:
            top_plate_surfs.append(tag)
        else:
            body_surfs.append(tag)

    # הגדרת הקבוצות הפיזיקליות לפי תגים קבועים
    if top_plate_surfs:
        gmsh.model.addPhysicalGroup(2, top_plate_surfs, tag=1, name="Top_Plate")
    if soundhole_surfs:
        gmsh.model.addPhysicalGroup(2, soundhole_surfs, tag=2, name="Soundhole_Air")
    if body_surfs:
        gmsh.model.addPhysicalGroup(2, body_surfs, tag=3, name="Body_Shell")
    if air_vols:
        # הוספת נפח האוויר (שלב 2 של הסולבר)
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