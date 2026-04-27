import numpy as np
import json
import os
from pathlib import Path

# --- SfePy Imports ---
from sfepy.discrete import Problem, Variables, Conditions 
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import FieldVariable, Material, Integral, Equation, Equations
from sfepy.discrete.conditions import EssentialBC
from sfepy.terms import Term

# ייבוא Eigensolver מ-SciPy ליציבות מקסימלית [cite: 1878, 2217]
from scipy.sparse.linalg import eigsh

def get_stiffness_matrix_3d(props):
    """חישוב מטריצת הקשיחות האורתוטרופית (C) מתוך נתוני החומר[cite: 1878]."""
    EL, ET, ER = props['E_L'], props['E_T'], props['E_R']
    vLT, vLR, vTR = props['nu_LT'], props['nu_LR'], props['nu_TR']
    GLT, GLR, GTR = props['G_LT'], props['G_LR'], props['G_TR']
    vTL = vLT * (ET / EL)
    vRL = vLR * (ER / EL)
    vRT = vTR * (ER / ET)
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1.0/EL, 1.0/ET, 1.0/ER
    S[0, 1] = S[1, 0] = -vTL / ET
    S[0, 2] = S[2, 0] = -vRL / ER
    S[1, 2] = S[2, 1] = -vRT / ER
    S[3, 3], S[4, 4], S[5, 5] = 1.0/GLT, 1.0/GLR, 1.0/GTR
    return np.linalg.inv(S)

def solve_3d_eigenmodes(mesh_file, config, num_modes=15):
    mesh = Mesh.from_file(mesh_file)
    domain = FEDomain('domain', mesh)
    # 1. הגדרת אזורים (Regions) - פיצול בין עץ לאוויר
    # 'Wood' משתמש ב-Surface (group 1) ו-'Air' משתמש ב-Volume (group 10)
    omega_wood = domain.create_region('Wood', 'vertices of group 1', 'facet')
    omega_air = domain.create_region('Air', 'vertices of group 10', 'volume')
    
    # 2. הגדרת שדות (Fields)
    # שדה תזוזה וקטורי (3 רכיבים) לעץ
    field_u = Field.from_args('fu', np.float64, 3, omega_wood, approx_order=1)
    # שדה לחץ סקלרי (רכיב 1) לאוויר
    field_p = Field.from_args('fp', np.float64, 1, omega_air, approx_order=1)
    
    # 3. הגדרת משתנים (Variables)
    # משתני העץ (תזוזה ופונקציית בדיקה)
    u = FieldVariable('u', 'unknown', field_u)
    v = FieldVariable('v', 'test', field_u, primary_var_name='u')
    
    # משתני האוויר (לחץ ופונקציית בדיקה)
    p = FieldVariable('p', 'unknown', field_p)
    phi = FieldVariable('phi', 'test', field_p, primary_var_name='p')
    
    # חומרים משתנים במרחב לפי Z [cite: 1937, 1939, 1960]
    C_top = get_stiffness_matrix_3d(config['materials']['top'])
    rho_top = config['materials']['top']['density']
    C_back = get_stiffness_matrix_3d(config['materials']['back'])
    rho_back = config['materials']['back']['density']
    
    def guitar_material(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return {} 
        n_pts = coors.shape[0]
        C_out = np.zeros((n_pts, 6, 6), dtype=np.float64)
        rho_out = np.zeros((n_pts, 1, 1), dtype=np.float64)
        
        z_thresh = (config['geometry']['depth'] / 2.0) - (config['geometry']['thickness'] * 1.2)
        is_top = coors[:, 2] > z_thresh
        
        C_out[is_top] = C_top
        rho_out[is_top] = rho_top
        C_out[~is_top] = C_back
        rho_out[~is_top] = rho_back
        return {'C': C_out, 'rho': rho_out}
        
    mat = Material('m', function=guitar_material)
    integral = Integral('i', order=2)
    
    # הגדרת המשוואות
    eq1 = Equation('stiff', Term.new('dw_lin_elastic(m.C, v, u)', integral, omega, m=mat, v=v, u=u))
    eq2 = Equation('mass', Term.new('dw_volume_dot(m.rho, v, u)', integral, omega, m=mat, v=v, u=u))
    pb = Problem('guitar', equations=Equations([eq1, eq2]))
    
    # תנאי שפה (Fixed Ribs) [cite: 1977, 2155]
    def select_ribs(coors, domain=None):
        # חישוב הגבול כך שייעצר 2 מילימטר מתחת לתחילת הלוח העליון/תחתון
        depth = config['geometry']['depth']
        thick = config['geometry']['thickness']
        
        # הגבול החדש: חצי גובה פחות עובי העץ, ופחות מרווח ביטחון קטן
        z_limit = (depth / 2.0) - thick - 0.002
        
        return np.where((coors[:, 2] < z_limit) & (coors[:, 2] > -z_limit))[0]

    fixed_ribs_region = domain.create_region('Fixed_Ribs', 'vertices by select_ribs', 'facet', functions={'select_ribs': select_ribs})
    pb.set_bcs(Conditions([EssentialBC('Fixed_BC', fixed_ribs_region, {'u.all': 0.0})]))

    # --- Setup סופי עבור 2025.3 (עקיפת נעילת משתנים) [cite: 2156, 2207, 2231] ---
    pb.time_update() 
    pb.update_materials()
    variables = pb.get_variables()
    variables.init_state()
    # הזרקת וקטור מצב מלא למניעת שגיאת "Variable Locked" [cite: 2206, 2231]
    variables.set_state(np.zeros(u.n_dof, dtype=np.float64))
    variables.apply_ebc()

    print("Assembling Global Matrices manually...")
    # יצירת המטריצות הגלובליות מתוך ה-Equations [cite: 2265, 2278]
    mtx_k = pb.equations.create_matrix_graph()
    res_k = eq1.evaluate(mode='weak', dw_mode='matrix', asm_obj=mtx_k)
    if isinstance(res_k, tuple): mtx_k = res_k[0] # לכידה מה-Tuple 
    
    mtx_m = pb.equations.create_matrix_graph()
    res_m = eq2.evaluate(mode='weak', dw_mode='matrix', asm_obj=mtx_m)
    if isinstance(res_m, tuple): mtx_m = res_m[0] # לכידה מה-Tuple 

    print(f"Solving for first {num_modes} vibrational modes using eigsh...")
    # פתרון הבעיה הכללית: K * x = lambda * M * x [cite: 1878, 1968, 2097]
    vals, vecs = eigsh(mtx_k, k=num_modes, M=mtx_m, sigma=1.0, tol=1e-5)
    
    freqs = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)
    return sorted(freqs.tolist())

def run_fem_3d_simulation(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    mesh_file = config['solver']['mesh_file']
    try:
        # שימוש ב-15 מודים לביצועים 
        real_modes = solve_3d_eigenmodes(mesh_file, config, config['solver']['num_modes'])
    except Exception as e:
        print(f"FEM Solver encountered a fatal error: {e}")
        import traceback
        traceback.print_exc()
        return

    output_data = {
        "modes_hz": real_modes,
        "amplitudes": [1.0] * len(real_modes),
        "dimensions": config['geometry']
    }
    
    output_path = Path("FEM/outputs/fem_3d_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"SUCCESS: 3D frequencies saved to {output_path}")
    return output_path

if __name__ == "__main__":
    test_config = Path(__file__).resolve().parents[1] / "configs" / "guitar_3d.json"
    run_fem_3d_simulation(str(test_config))