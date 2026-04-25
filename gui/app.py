import streamlit as st
import json
import os
import subprocess
from pathlib import Path
import wave
import pyvista as pv
from stpyvista import stpyvista
import numpy as np

# --- Constants & Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "FEM" / "configs" / "guitar_3d.json"
GEOMETRY_SCRIPT = BASE_DIR / "FEM" / "geometry" / "build_3d_guitar.py"
FEM_SCRIPT = BASE_DIR / "FEM" / "scripts" / "fem_main_3d.py"
STK_BINARY = BASE_DIR / "cpp" / "guitar_stk"
WAV_OUTPUT = BASE_DIR / "audio" / "guitar_sound.wav"
MESH_FILE = BASE_DIR / "FEM" / "mesh" / "guitar_3d.msh"

# הגדרות סביבה קריטיות ל-Linux/VM
os.environ["QT_QPA_PLATFORM"] = "offscreen"
pv.OFF_SCREEN = True

# --- Initialization ---
if 'cam_pos' not in st.session_state:
    st.session_state.cam_pos = None
if 'fem_ready' not in st.session_state:
    st.session_state.fem_ready = False

# --- Material Library (צבעים ריאליסטיים ומופרדים) ---
WOOD_LIBRARY = {
    "Sitka Spruce": {
        "density": 450.0, "E_L": 11.0e9, "E_T": 1.0e9, "E_R": 0.7e9,
        "nu_LT": 0.37, "nu_LR": 0.37, "nu_TR": 0.4,
        "G_LT": 0.75e9, "G_LR": 0.75e9, "G_TR": 0.05e9,
        "q_min": 60, "q_max": 80,
        "color": "#FCE6C9"  # קרם שמנת (עץ אשוח)
    },
    "Honduran Mahogany": {
        "density": 530.0, "E_L": 10.8e9, "E_T": 0.9e9, "E_R": 0.7e9,
        "nu_LT": 0.35, "nu_LR": 0.35, "nu_TR": 0.4,
        "G_LT": 0.5e9, "G_LR": 0.5e9, "G_TR": 0.05e9,
        "q_min": 45, "q_max": 60,
        "color": "#93441A"  # חום אדמדם חם
    },
    "Indian Rosewood": {
        "density": 830.0, "E_L": 11.5e9, "E_T": 1.3e9, "E_R": 0.7e9,
        "nu_LT": 0.33, "nu_LR": 0.33, "nu_TR": 0.4,
        "G_LT": 0.95e9, "G_LR": 0.95e9, "G_TR": 0.05e9,
        "q_min": 90, "q_max": 100,
        "color": "#3D2B1F"  # חום שוקולד כהה עמוק
    }
}

# --- Note Map ---
NOTES_DICT = {
    "E2": 82.41, "A2": 110.00, "D3": 146.83, "G3": 196.00, "B3": 246.94, "E4": 329.63
}

def save_cfg():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "geometry": {
            "shape_type": shape_type, "length": L, "width": W, "depth": D, 
            "thickness": thick, "hole_radius": hr, "vis_mode": vis_mode, "exploded_view": exploded
        },
        "materials": {
            "top": {**WOOD_LIBRARY[top_wood], "name": top_wood},
            "back": {**WOOD_LIBRARY[back_wood], "name": back_wood},
            "air": {"density": 1.204, "speed_of_sound": 343.0}
        },
        "solver": {"num_modes": 30, "mesh_file": str(MESH_FILE)}
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=4)

st.set_page_config(page_title="3D Guitar Simulator", layout="wide")
st.title("🎸 3D Multi-Material Guitar Simulator")

# ==========================================
# --- Sidebar ---
# ==========================================
st.sidebar.header("1. Material & Shape")
shape_type = st.sidebar.selectbox("Guitar Shape", ["Classical", "Dreadnought", "Box"])

if shape_type == "Classical":
    L_min, L_max, L_def = 0.35, 0.60, 0.48
    W_min, W_max, W_def = 0.20, 0.45, 0.37
    D_min, D_max, D_def = 0.08, 0.15, 0.10
elif shape_type == "Dreadnought":
    L_min, L_max, L_def = 0.45, 0.70, 0.51
    W_min, W_max, W_def = 0.30, 0.55, 0.40
    D_min, D_max, D_def = 0.10, 0.20, 0.12
else: 
    L_min, L_max, L_def = 0.10, 1.00, 0.40
    W_min, W_max, W_def = 0.10, 0.80, 0.30
    D_min, D_max, D_def = 0.01, 0.50, 0.10

col_top, col_back = st.sidebar.columns(2)
with col_top:
    top_wood = st.selectbox("Front (Top)", list(WOOD_LIBRARY.keys()), index=0)
with col_back:
    back_wood = st.selectbox("Back & Sides", list(WOOD_LIBRARY.keys()), index=2) # Default Rosewood

exploded = st.sidebar.checkbox("Exploded View", value=False)
hr = st.sidebar.slider("Soundhole Radius (m)", 0.01, 0.12, 0.04)

st.sidebar.header("2. Geometry")
L = st.sidebar.slider("Length (m)", L_min, L_max, L_def, key=f"L_{shape_type}")
W = st.sidebar.slider("Width (m)", W_min, W_max, W_def, key=f"W_{shape_type}")
D = st.sidebar.slider("Depth (m)", D_min, D_max, D_def, key=f"D_{shape_type}")
thick = st.sidebar.slider("Thickness (m)", 0.001, 0.05, 0.003, format="%.3f")
vis_mode = st.sidebar.selectbox("Style", ["Mesh + Solid", "Solid Wood", "Wireframe"])

# ==========================================
# --- Actions ---
# ==========================================
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Update Config", use_container_width=True):
        save_cfg(); st.success("Settings Updated")
with c2:
    if st.button("Preview 3D Mesh", use_container_width=True):
        save_cfg()
        subprocess.run(["python3", str(GEOMETRY_SCRIPT), "-nopopup"])
        st.info("Mesh updated. Syncing view...")
with c3:
    if st.button("Run & Generate Sound", type="primary", use_container_width=True):
        save_cfg()
        with st.spinner("FEM Solving..."):
            subprocess.run(["python3", str(FEM_SCRIPT)])
            st.session_state.fem_ready = True
        top_q = WOOD_LIBRARY[top_wood]
        stk_cmd = [str(STK_BINARY), "--fem_json", str(BASE_DIR / "FEM" / "outputs" / "fem_3d_output.json"),
                   "--note_hz", "110.0", "--out", str(WAV_OUTPUT), "--q", str((top_q["q_min"]+top_q["q_max"])/2)]
        subprocess.run(stk_cmd)
        st.audio(str(WAV_OUTPUT))

st.divider()

# ==========================================
# --- Sidebar Footer & Engineering Tools ---
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Engineering Tools")

# כפתור לפתיחת Gmsh המלא (GUI)
if st.sidebar.button("🔍 Open full Model in Gmsh", use_container_width=True):
    save_cfg()
    with st.spinner("Opening Gmsh GUI..."):
        # הרצה ללא flag ה-nopopup כדי שהחלון ייפתח
        subprocess.run(["python3", str(GEOMETRY_SCRIPT)])

st.sidebar.caption("Use this to verify physical groups and mesh alignment.")

# ==========================================
# --- 3D Preview (Multi-Material & Locking) ---
# ==========================================
st.subheader("3. 3D Design Preview")

if MESH_FILE.exists():
    try:
        pv.set_jupyter_backend('static') 
        plotter = pv.Plotter(window_size=[900, 500])
        plotter.background_color = "#fdfdfd"
        
        # 1. טעינת הרשת המלאה
        vol_mesh = pv.read(str(MESH_FILE))
        
        # 2. התיקון: חילוץ המעטפת החיצונית בלבד (הופך ל-PolyData)
        surface_mesh = vol_mesh.extract_surface()
        
        # 3. סידור כיווני האור (העלמת החצי השחור)
        surface_mesh.compute_normals(inplace=True, auto_orient_normals=True)
        
        # --- לוגיקת פיצול חומרים (Top vs Body) ---
        centers = surface_mesh.cell_centers().points
        z_max = np.max(centers[:, 2])
        is_top = centers[:, 2] > (z_max - thick * 1.5)
        
        # צביעת הלוח הקדמי (Front)
        plotter.add_mesh(surface_mesh.extract_cells(is_top), 
                        color=WOOD_LIBRARY[top_wood]["color"], 
                        show_edges=("Mesh" in vis_mode), 
                        edge_color="#3c1e0a", 
                        smooth_shading=True,
                        ambient=0.3) 
        
        # צביעת הגוף (Back/Sides)
        plotter.add_mesh(surface_mesh.extract_cells(~is_top), 
                        color=WOOD_LIBRARY[back_wood]["color"], 
                        show_edges=("Mesh" in vis_mode), 
                        edge_color="#1a0a00", 
                        smooth_shading=True,
                        ambient=0.3)

        plotter.enable_eye_dome_lighting()

        # --- נעילת מצלמה ---
        if st.session_state.cam_pos:
            try:
                plotter.camera_position = st.session_state.cam_pos
            except:
                plotter.view_isometric()
        else:
            plotter.view_isometric()

        # תצוגה ושמירה של הזווית החדשה
        new_cam_pos = stpyvista(plotter, key="guitar_3d_preview")
        if new_cam_pos:
            st.session_state.cam_pos = new_cam_pos
        
        plotter.close()

    except Exception as e:
        st.warning("Visualizer syncing... Click 'Preview 3D Mesh' again.")
        st.expander("Debug").write(e)
else:
    st.info("Click 'Preview 3D Mesh' to generate the model.")

st.caption(f"Visual: Top ({top_wood}) | Body ({back_wood}) | Camera Persistence: ON")