import streamlit as st
import json
import os
import subprocess
import sys  # Needed to use the current virtualenv Python executable
from pathlib import Path
import wave
import numpy as np
import pyvista as pv
from stpyvista import stpyvista
from scipy.interpolate import CubicSpline

# --- Constants & Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "FEM" / "configs" / "guitar_3d.json"
GEOMETRY_SCRIPT = BASE_DIR / "FEM" / "geometry" / "build_3d_guitar.py"
FEM_SCRIPT = BASE_DIR / "FEM" / "scripts" / "fem_main_3d.py"
STK_BINARY = BASE_DIR / "cpp" / "guitar_stk"
WAV_OUTPUT = BASE_DIR / "audio" / "guitar_sound.wav"
MESH_FILE = BASE_DIR / "FEM" / "mesh" / "guitar_3d.msh"
ROM_CLASSIC_SNAPSHOTS = BASE_DIR / "ROM" / "classic" / "snapshots"

# Allow in-process import of FEM solver for live Streamlit status updates.
sys.path.append(str(BASE_DIR / "FEM" / "scripts"))
import fem_main_3d
from wood_library import (
    BACK_WOOD_IDS,
    TOP_WOOD_IDS,
    material_block_for_id,
    plot_color_for_wood,
    wood_display_name,
)

# Critical environment settings for Linux/VM
os.environ["QT_QPA_PLATFORM"] = "offscreen"
pv.OFF_SCREEN = True

# --- Initialization ---
if 'fem_ready' not in st.session_state:
    st.session_state.fem_ready = False

# Load saved geometry for Live Preview comparison
saved_geom = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, 'r') as f:
            saved_geom = json.load(f).get("geometry", {})
    except:
        pass

# Physical tags in Gmsh mesh (must match FEM geometry protocol).
TAG_TOP_PLATE = 1
TAG_SOUNDHOLE = 2
TAG_BACK_SIDES = 3

NOTES_DICT = {
    "E2": 82.41, "A2": 110.00, "D3": 146.83, "G3": 196.00, "B3": 246.94, "E4": 329.63
}

def add_silence_to_wav(path, duration=0.3):
    try:
        with wave.open(str(path), 'rb') as reader:
            params = reader.getparams()
            frames = reader.readframes(params.nframes)
            silence = b'\x00' * int(duration * params.framerate * params.sampwidth * params.nchannels)
            with wave.open(str(path), 'wb') as writer:
                writer.setparams(params)
                writer.writeframes(silence + frames)
    except Exception as e:
        pass

def save_cfg():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    data["geometry"] = {
        "design_mode": design_mode,
        "shape_type": shape_type,
        "length": L,
        "width": W,
        "depth": D,
        "thickness": thick,
        "hole_radius": hr,
        "lower_bout": lower_bout,
        "upper_bout": upper_bout,
        "waist": waist,
        "soundhole_y": soundhole_y,
        "vis_mode": st.session_state.get("vis_mode_ui", "Mesh + Solid"),
        "exploded_view": exploded,
    }
    data["materials"] = {
        "top": {**material_block_for_id(top_wood_id), "name": wood_display_name(top_wood_id), "wood_id": top_wood_id},
        "back": {**material_block_for_id(back_wood_id), "name": wood_display_name(back_wood_id), "wood_id": back_wood_id},
        "air": {"density": 1.204, "speed_of_sound": 343.0},
    }
    # Merge solver so GUI updates do not strip FEM keys (adaptive_mode_sifter, sifter_*, shifts, etc.).
    solver = dict(data.get("solver") or {})
    solver["num_modes"] = int(solver.get("num_modes", 50))
    solver["mesh_file"] = str(MESH_FILE)
    data["solver"] = solver
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def generate_live_preview_mesh(L, W, D, shape_type, design_mode, upper_bout, waist, lower_bout):
    save_cfg() 
    
    preview_file = BASE_DIR / "FEM" / "mesh" / "preview_mesh.msh"
    
    # 1) Delete the old preview file so any new file is guaranteed fresh.
    if preview_file.exists():
        preview_file.unlink()
        
    # 2) Run Gmsh
    py_exe = sys.executable
    result = subprocess.run([py_exe, str(GEOMETRY_SCRIPT), "-nopopup", "--preview"], capture_output=True, text=True)
    
    # 3) If Gmsh fails or does not create a file, stop and show the error.
    if not preview_file.exists():
        st.error(f"Gmsh Failed to update preview!\nError Log:\n{result.stderr}")
        return None
        
    # 4) The file was created successfully, load it for rendering.
    mesh = pv.read(str(preview_file))
    surface_mesh = mesh.extract_surface()
    
    # Keep corners sharp and waist contours smooth using split vertices.
    surface_mesh.compute_normals(
        cell_normals=False, 
        point_normals=True, 
        feature_angle=30, 
        split_vertices=True, 
        inplace=True, 
        auto_orient_normals=True
    )
    
    return surface_mesh

st.set_page_config(page_title="3D Guitar Simulator", layout="wide")
st.title("Guitar Simulator")

# --- Sidebar ---
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

def _wood_option_label(wood_id: str) -> str:
    """Sidebar label: explicit wood_id plus common name."""
    return f"{wood_id} — {wood_display_name(wood_id)}"

col_top, col_back = st.sidebar.columns(2)
with col_top:
    top_wood_id = st.selectbox(
        "Top Wood",
        TOP_WOOD_IDS,
        index=0,
        format_func=_wood_option_label,
        help="spruce, cedar",
    )
with col_back:
    back_wood_id = st.selectbox(
        "Back/Sides Wood",
        BACK_WOOD_IDS,
        index=0,
        format_func=_wood_option_label,
        help="rosewood, mahogany, maple",
    )
top_plot_color = plot_color_for_wood(top_wood_id)
back_plot_color = plot_color_for_wood(back_wood_id)

exploded = st.sidebar.checkbox("Exploded View", value=False)
hr = st.sidebar.slider("Soundhole Radius (m)", 0.035, 0.055, 0.04)

st.sidebar.header("2. Geometry")
design_mode = st.sidebar.radio("Design Mode", ["Basic (Geometric)", "Professional (Luthier)"])

if design_mode == "Basic (Geometric)":
    L = st.sidebar.slider("Length (m)", L_min, L_max, L_def, key=f"L_{shape_type}")
    W = st.sidebar.slider("Width (m)", W_min, W_max, W_def, key=f"W_{shape_type}")
    D = st.sidebar.slider("Depth (m)", D_min, D_max, D_def, key=f"D_{shape_type}")
    lower_bout, upper_bout, waist, soundhole_y = W, W * 0.75, W * 0.65, L * 0.4
else:
    L = st.sidebar.slider("Total Body Length (m)", L_min, L_max, L_def, key=f"L_prof_{shape_type}")
    D = st.sidebar.slider("Depth (m)", D_min, D_max, D_def, key=f"D_prof_{shape_type}")
    lower_bout = st.sidebar.slider("Lower Bout Width (m)", 0.20, 0.60, W_def, key="lb")
    upper_bout = st.sidebar.slider("Upper Bout Width (m)", 0.15, 0.50, W_def * 0.75, key="ub")
    waist = st.sidebar.slider("Waist Width (m)", 0.10, 0.40, W_def * 0.65, key="wst")
    soundhole_y = st.sidebar.slider("Soundhole Position Y (m)", 0.10, 0.60, L_def * 0.4, key="sh_y")
    W = lower_bout

# Dynamic slider for top plate thickness (3mm to 6mm).
thick = st.sidebar.slider(
    "Top Plate Thickness (m)", 
    min_value=0.0030, 
    max_value=0.0060, 
    value=0.0030, 
    step=0.0005, 
    format="%.4f", 
    help="Standard acoustic guitar soundboards typically range from 3mm to 6mm."
)

st.sidebar.header("3. Synthesis Settings")
pitch_mode = st.sidebar.radio("Input Mode", ["Musical Notes", "Manual Hz"])
if pitch_mode == "Musical Notes":
    note_name = st.sidebar.selectbox("Select Note", list(NOTES_DICT.keys()), index=1)
    note_hz = NOTES_DICT[note_name]
else:
    note_hz = st.sidebar.number_input("Frequency (Hz)", value=110.0)

q_mode = st.sidebar.radio("Q Estimation", ["Mean (Stable)", "Random (Realistic)"])
mix_val = 0.98
gain_val = 400

st.sidebar.header("4. Sifter Settings")
_solver_cfg: dict = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as _sf:
            _solver_cfg = (json.load(_sf).get("solver") or {})
    except Exception:
        _solver_cfg = {}
sifter_break_hz = st.sidebar.number_input(
    "Break frequency (Hz)",
    min_value=50.0,
    max_value=400.0,
    value=float(_solver_cfg.get("sifter_adaptive_break_hz", 200.0)),
    step=1.0,
    help="Adaptive threshold switch: batch center and per-mode wood gate use f < vs ≥ this value.",
)
sifter_high_uniq = st.sidebar.number_input(
    "High-band uniqueness min",
    min_value=0.01,
    max_value=0.5,
    value=float(_solver_cfg.get("sifter_uniqueness_min_high_band", 0.06)),
    step=0.01,
    format="%.2f",
)
sifter_high_dup_hz = st.sidebar.number_input(
    "High-band frequency gap / dup (Hz)",
    min_value=0.1,
    max_value=5.0,
    value=float(_solver_cfg.get("sifter_dup_hz_high_band", 0.8)),
    step=0.1,
    format="%.1f",
)
if st.sidebar.button("Save sifter settings to guitar_3d.json", use_container_width=True):
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
        except Exception:
            cfg_data = {}
    else:
        cfg_data = {}
    sol = dict(cfg_data.get("solver") or {})
    sol["sifter_adaptive_break_hz"] = float(sifter_break_hz)
    sol["sifter_uniqueness_min_high_band"] = float(sifter_high_uniq)
    sol["sifter_dup_hz_high_band"] = float(sifter_high_dup_hz)
    cfg_data["solver"] = sol
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=4)
    st.sidebar.success("Sifter settings saved.")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Engineering Tools")
if st.sidebar.button("🔍 Open full Model in Gmsh", use_container_width=True):
    save_cfg()
    with st.spinner("Opening Gmsh GUI..."):
        # Use sys.executable to ensure execution inside the active virtualenv.
        subprocess.run([sys.executable, str(GEOMETRY_SCRIPT)])

# --- Check Sync State ---
current_geom = {
    "design_mode": design_mode, "shape_type": shape_type, "length": L, 
    "width": W, "depth": D, "thickness": thick, "hole_radius": hr, 
    "lower_bout": lower_bout, "upper_bout": upper_bout, "waist": waist, 
    "soundhole_y": soundhole_y
}

is_synced = True
for k, v in current_geom.items():
    if type(v) is float:
        if abs(v - saved_geom.get(k, 0)) > 1e-4:
            is_synced = False
            break
    else:
        if v != saved_geom.get(k):
            is_synced = False
            break

if not MESH_FILE.exists(): is_synced = False

# ==========================================
# --- Process Control ---
# ==========================================
st.markdown("### Process Control")
c1, c2 = st.columns(2)

# Track whether to show a one-time success banner after rerun.
if "show_success_msg" not in st.session_state:
    st.session_state.show_success_msg = False

with c1:
    if st.button("1. Apply Design", use_container_width=True, type="primary"):
        save_cfg()
        
        if MESH_FILE.exists(): os.remove(MESH_FILE)
        out_json = Path("FEM/outputs/fem_3d_output.json")
        if out_json.exists(): os.remove(out_json)
        
        py_exe = sys.executable
        
        with st.spinner("Building 3D Model..."):
            subprocess.run([py_exe, str(GEOMETRY_SCRIPT), "-nopopup"], capture_output=True, text=True)

        # Run FEM in-process with live UI status updates.
        try:
            with st.status("Running FEM Simulation...", expanded=True) as fem_status:
                def fem_status_callback(message: str):
                    fem_status.update(label=message, state="running", expanded=True)

                fem_status_callback("Trigger received from UI. Starting FEM backend...")
                fem_main_3d.run_fem_3d_simulation(str(CONFIG_PATH), status_callback=fem_status_callback)
                fem_status.update(label="FEM simulation completed successfully.", state="complete", expanded=False)
            st.session_state.fem_ready = True
        except Exception as e:
            st.session_state.fem_ready = False
            st.error(f"FEM simulation failed: {e}")
        
        # Trigger a full rerun so the new model appears immediately.
        st.session_state.show_success_msg = True  # Show one-time success message
        st.rerun()  # Force full page refresh and re-render

# Show one-time success message after rerun.
if st.session_state.show_success_msg:
    st.success("Design applied! Explore your 3D model below.")
    st.session_state.show_success_msg = False  # Reset so it does not persist forever

with c2:
    if st.button("2. Generate Sound", use_container_width=True, disabled=not st.session_state.fem_ready):
        save_cfg()
        
        with st.spinner("Synthesizing Audio..."):
            out_json = Path("FEM/outputs/fem_3d_output.json")
            
            if not out_json.exists():
                st.error("Audio synthesis failed. Try tweaking the dimensions.")
            else:
                top_q = material_block_for_id(top_wood_id)
                stk_cmd = [
                    str(STK_BINARY), "--fem_json", str(out_json),
                    "--note_hz", str(note_hz), "--dur", "3.0", "--mix", str(mix_val),
                    "--wet_gain", str(gain_val), "--out", str(WAV_OUTPUT),
                    "--rad_k", "0.06", "--amp", "0.3", "--seed", "123"
                ]
                # ... (rest of the standard STK command)
                if q_mode == "Random (Realistic)":
                    stk_cmd.extend(["--q_min", str(top_q["q_min"]), "--q_max", str(top_q["q_max"]), "--q_mode", "random"])
                else:
                    stk_cmd.extend(["--q", str((top_q["q_min"] + top_q["q_max"])/2)])
                    
                subprocess.run(stk_cmd)
                add_silence_to_wav(WAV_OUTPUT, 0.3)
            
        st.success("Audio Generated!")
        st.audio(str(WAV_OUTPUT))

st.divider()

# ==========================================
# --- ROM / Classic snapshot modes ---
# ==========================================
st.subheader("4. ROM / Classic snapshot analysis")
if ROM_CLASSIC_SNAPSHOTS.is_dir():
    npz_list = sorted(ROM_CLASSIC_SNAPSHOTS.glob("snapshot_*.npz"))
else:
    npz_list = []
if not npz_list:
    st.info("No `snapshot_*.npz` files under `ROM/classic/snapshots/`. Run the offline ROM pipeline to generate snapshots.")
else:
    snap_choice = st.selectbox("Snapshot file", npz_list, format_func=lambda p: p.name)
    try:
        with np.load(snap_choice, allow_pickle=True) as snap:
            if "freqs_hz" in snap:
                freqs = np.asarray(snap["freqs_hz"], dtype=np.float64).reshape(-1)
            elif "eigenfrequencies_hz" in snap:
                freqs = np.asarray(snap["eigenfrequencies_hz"], dtype=np.float64).reshape(-1)
            elif "frequencies_hz" in snap:
                freqs = np.asarray(snap["frequencies_hz"], dtype=np.float64).reshape(-1)
            else:
                freqs = np.array([])
                st.error("No frequency array found (expected `freqs_hz` first).")

            uniq_arr = snap["uniqueness_scores"] if "uniqueness_scores" in snap else None
            part_arr = snap["participation_ratios"] if "participation_ratios" in snap else None
            if uniq_arr is not None:
                uniq_arr = np.asarray(uniq_arr, dtype=np.float64).reshape(-1)
            if part_arr is not None:
                part_arr = np.asarray(part_arr, dtype=np.float64).reshape(-1)

        if freqs.size:
            order = np.argsort(freqs)
            f_sorted = freqs[order]
            u_sorted = uniq_arr[order] if uniq_arr is not None and uniq_arr.size == freqs.size else None
            p_sorted = part_arr[order] if part_arr is not None and part_arr.size == freqs.size else None
            if uniq_arr is not None and uniq_arr.size != freqs.size:
                st.warning("`uniqueness_scores` length does not match `freqs_hz` — showing N/A for uniqueness.")
            if part_arr is not None and part_arr.size != freqs.size:
                st.warning("`participation_ratios` length does not match `freqs_hz` — showing N/A for wood participation.")
            if uniq_arr is None:
                st.caption("This snapshot has no `uniqueness_scores` (older run). Re-export with the current solver for telemetry.")
            if part_arr is None:
                st.caption("This snapshot has no `participation_ratios` (older run). Re-export with the current solver for telemetry.")

            table_rows = []
            prev_f = None
            for i in range(f_sorted.size):
                gap = ""
                if prev_f is not None:
                    gap = f"{f_sorted[i] - prev_f:.4f}"
                u_cell = "N/A"
                if u_sorted is not None:
                    _u = float(u_sorted[i])
                    u_cell = f"{_u:.4f}" if np.isfinite(_u) else "N/A"
                p_cell = "N/A"
                if p_sorted is not None:
                    _p = float(p_sorted[i])
                    p_cell = f"{_p:.4f}" if np.isfinite(_p) else "N/A"
                table_rows.append(
                    {
                        "Mode": i + 1,
                        "Frequency (Hz)": float(f_sorted[i]),
                        "Δ from prev (Hz)": gap if gap else "—",
                        "Uniqueness": u_cell,
                        "Wood participation": p_cell,
                    }
                )
                prev_f = float(f_sorted[i])
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Failed to load snapshot: {e}")

st.divider()

# ==========================================
# --- 3D Preview ---
# ==========================================
st.subheader("5. 3D Design Preview")

col_style, col_cam = st.columns(2)
with col_style:
    vis_mode = st.selectbox("Visual Style", ["Mesh + Solid", "Solid Wood"], key="vis_mode_ui")
with col_cam:
    cam_preset = st.selectbox("Camera Anchor (Auto-Reset)", ["Standing Angled (3D)", "Standing Upright (Front)", "Laying Flat (Top View)", "Laying on Side (Profile)"])

try:
    pv.set_jupyter_backend('static') 
    plotter = pv.Plotter(window_size=[900, 500])
    plotter.background_color = "#f4f4f9"
    show_edges_flag = ("Mesh" in vis_mode)

    def render_mesh_by_protocol(mesh, show_edges, top_color: str, back_color: str):
        """
        Render mesh by physical tags protocol:
        1=Top_Plate, 2=Soundhole, 3=Back/Sides, 10=Air_Internal(hidden).
        Returns True if tag-based rendering succeeded, else False (caller should fallback).
        """
        tags = mesh.cell_data.get('gmsh:physical')
        if tags is None:
            return False

        # Efficient mask creation once, then extract only when needed.
        is_top = (tags == TAG_TOP_PLATE)
        is_hole = (tags == TAG_SOUNDHOLE)
        is_body = (tags == TAG_BACK_SIDES)

        if not (is_top.any() or is_body.any() or is_hole.any()):
            return False

        rendered_any = False

        if is_top.any():
            top_mesh = mesh.extract_cells(is_top).extract_surface()
            if top_mesh.n_cells > 0:
                plotter.add_mesh(
                    top_mesh,
                    color=top_color,
                    show_edges=show_edges,
                    edge_color="#2b1a10"
                )
                rendered_any = True

        if is_body.any():
            body_mesh = mesh.extract_cells(is_body).extract_surface()
            if body_mesh.n_cells > 0:
                plotter.add_mesh(
                    body_mesh,
                    color=back_color,
                    show_edges=show_edges,
                    edge_color="#2b1a10"
                )
                rendered_any = True

        # Tag 2: represent as dark semi-transparent surface (or leave as hole by not drawing it).
        if is_hole.any():
            hole_mesh = mesh.extract_cells(is_hole).extract_surface()
            if hole_mesh.n_cells > 0:
                plotter.add_mesh(hole_mesh, color="#111111", opacity=0.45, show_edges=False)
                rendered_any = True

        # Tag 10 (Air_Internal) is intentionally not rendered to avoid black block.
        return rendered_any

    if is_synced:
        st.success("✅ Viewing High-Fidelity Engineering Mesh (Gmsh)")
        try:
            vol_mesh = pv.read(str(MESH_FILE))

            # Preferred path: tag-based rendering by protocol.
            if not render_mesh_by_protocol(vol_mesh, show_edges_flag, top_plot_color, back_plot_color):
                st.warning("Physical tags missing/empty - displaying full surface fallback.")
                plotter.add_mesh(
                    vol_mesh.extract_surface(),
                    color=back_plot_color,
                    show_edges=show_edges_flag,
                    edge_color="#2b1a10"
                )

        except Exception as e:
            st.error(f"Visualization Error: {e}")
    else:
        st.warning("⚠️ Live Preview Mode (Fast CAD)")
        preview_mesh = generate_live_preview_mesh(L, W, D, shape_type, design_mode, upper_bout, waist, lower_bout)
        
        if preview_mesh is not None:
            if not render_mesh_by_protocol(preview_mesh, show_edges=False, top_color=top_plot_color, back_color=back_plot_color):
                st.warning("Preview tags missing/empty - displaying full surface fallback.")
                plotter.add_mesh(preview_mesh, color=back_plot_color, show_edges=False)
        else:
            st.warning("Preview mesh is unavailable.")
            
    plotter.enable_anti_aliasing("ssaa") 
    if cam_preset == "Standing Angled (3D)": plotter.camera_position = [(0.0, -0.4, 1.1), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    elif cam_preset == "Standing Upright (Front)": plotter.camera_position = [(0.0, 0.0, 1.2), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    elif cam_preset == "Laying Flat (Top View)": plotter.camera_position = [(0.0, 0.0, 1.2), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    elif cam_preset == "Laying on Side (Profile)": plotter.camera_position = [(0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]

    stpyvista(plotter, key="guitar_3d_preview")
    plotter.close()

except Exception as e:
    st.warning("Visualizer syncing...")

if is_synced:
    st.caption(
        f"Visual: top={top_wood_id} ({top_plot_color}) | back/sides={back_wood_id} ({back_plot_color}) | CAD Mode: ON"
    )
else: st.caption("Visual: Live Shape Prototype (Solid Block)")