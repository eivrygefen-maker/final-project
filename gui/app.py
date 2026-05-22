import streamlit as st
import json
import os
import subprocess
import sys  # Needed to use the current virtualenv Python executable
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
FEM_FOM_JSON = BASE_DIR / "FEM" / "outputs" / "fem_3d_output.json"
ROM_STK_JSON = BASE_DIR / "FEM" / "outputs" / "rom_stk_body.json"
DEFAULT_ROM_SHAPE = "classic"
ROM_CLASSIC_ROOT = BASE_DIR / "ROM" / "classic"
ROM_CLASSIC_SNAPSHOTS = ROM_CLASSIC_ROOT / "snapshots"
ROM_REDUCED_BASIS = ROM_CLASSIC_ROOT / "reduced_basis.npz"
PACKAGED_ROM_NPZ = BASE_DIR / "FEM" / "SORTING" / "final_guitar_rom.npz"
SELECTED_MODES_CSV = BASE_DIR / "FEM" / "SORTING" / "selected_modes.csv"
CANDIDATES_LOG = BASE_DIR / "FEM" / "SORTING" / "candidates_log.json"
SHAPES_CONFIG = BASE_DIR / "FEM" / "configs" / "rom_shapes.json"

# Allow in-process import of FEM solver for live Streamlit status updates.
sys.path.append(str(BASE_DIR / "FEM" / "scripts"))
sys.path.append(str(BASE_DIR / "FEM" / "rom"))
import fem_main_3d
from fem_rom_postprocess import DOMINANT_TAG_BACK, DOMINANT_TAG_TOP, dominant_tag_for_row
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
if "fem_ready" not in st.session_state:
    st.session_state.fem_ready = False
if "developer_fom_mode" not in st.session_state:
    st.session_state.developer_fom_mode = False
if "last_engine" not in st.session_state:
    st.session_state.last_engine = ""
if "rom_last_result" not in st.session_state:
    st.session_state.rom_last_result = {}

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

DOMINANT_COLOR = {DOMINANT_TAG_TOP: "#2e7d32", DOMINANT_TAG_BACK: "#1565c0"}


def _try_import_rom_manager():
    try:
        from rom_manager import ROMManager  # noqa: WPS433

        return ROMManager, None
    except Exception as exc:
        return None, exc


def _gui_lhs_params(
    *,
    shape_type: str,
    length: float,
    width: float,
    depth: float,
    thickness: float,
    hole_radius: float,
    lower_bout: float,
    upper_bout: float,
    waist: float,
    soundhole_y: float,
    top_wood: str,
    back_wood: str,
) -> Dict[str, Any]:
    """Flat LHS keys from sidebar geometry and unified material selection."""
    return {
        "geometry.shape_type": shape_type,
        "geometry.length": float(length),
        "geometry.width": float(width),
        "geometry.depth": float(depth),
        "geometry.thickness": float(thickness),
        "geometry.hole_radius": float(hole_radius),
        "geometry.lower_bout": float(lower_bout),
        "geometry.upper_bout": float(upper_bout),
        "geometry.waist": float(waist),
        "geometry.soundhole_y": float(soundhole_y),
        "materials.top.wood_id": str(top_wood),
        "materials.back.wood_id": str(back_wood),
    }


def _write_rom_stk_body_json(freqs_hz: Sequence[float], out_path: Path) -> Path:
    """STK body input: ROM ``solve_online`` frequencies only (no plate filtering)."""
    freqs = [float(f) for f in freqs_hz if float(f) > 0.0]
    if not freqs:
        raise ValueError("ROM solve returned no positive frequencies for STK.")
    n = len(freqs)
    weights = [1.0 / (1.0 + 0.25 * i) for i in range(n)]
    wmax = max(weights)
    if wmax > 0.0:
        weights = [w / wmax for w in weights]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "rom_online_body",
        "modes_hz": freqs,
        "mode_weights": weights,
        "num_modes": n,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _run_stk_synthesis(
    *,
    body_json: Path,
    top_wood_id: str,
    note_hz: float,
    q_mode: str,
    mix_val: float = 0.98,
    gain_val: float = 400.0,
) -> None:
    if not body_json.is_file():
        raise FileNotFoundError(f"STK body JSON missing: {body_json}")
    if not STK_BINARY.is_file():
        raise FileNotFoundError(f"STK binary missing: {STK_BINARY}")
    top_q = material_block_for_id(top_wood_id)
    stk_cmd = [
        str(STK_BINARY),
        "--fem_json",
        str(body_json),
        "--note_hz",
        str(note_hz),
        "--dur",
        "3.0",
        "--mix",
        str(mix_val),
        "--wet_gain",
        str(gain_val),
        "--out",
        str(WAV_OUTPUT),
        "--rad_k",
        "0.06",
        "--amp",
        "0.3",
        "--seed",
        "123",
        "--modes",
        "0",
        "--skip",
        "0",
    ]
    if q_mode == "Random (Realistic)":
        stk_cmd.extend(
            ["--q_min", str(top_q["q_min"]), "--q_max", str(top_q["q_max"]), "--q_mode", "random"]
        )
    else:
        stk_cmd.extend(["--q", str((top_q["q_min"] + top_q["q_max"]) / 2.0)])
    subprocess.run(stk_cmd, check=False)
    add_silence_to_wav(WAV_OUTPUT, 0.3)


def _build_engineering_mesh(py_exe: str) -> None:
    if MESH_FILE.exists():
        os.remove(MESH_FILE)
    subprocess.run([py_exe, str(GEOMETRY_SCRIPT), "-nopopup"], capture_output=True, text=True)
    if not MESH_FILE.exists():
        raise RuntimeError("Gmsh did not produce guitar_3d.msh")


# ---------------------------------------------------------------------------
# DEV_ONLY_FOM_BRANCH — remove this block when FOM diagnostics retire.
# ---------------------------------------------------------------------------
def _execute_fom_engine(config_path: Path, status_callback) -> None:
    """Full-order coupled solve; writes ``fem_3d_output.json`` for STK."""
    fem_main_3d.run_fem_3d_simulation(str(config_path), status_callback=status_callback)


# ---------------------------------------------------------------------------
# END DEV_ONLY_FOM_BRANCH
# ---------------------------------------------------------------------------


def _execute_rom_engine(
    *,
    rom_shape: str,
    lhs_params: Dict[str, Any],
    nev: int = 0,
) -> Dict[str, Any]:
    """Reduced-basis online solve; frequencies drive STK (not ``dominant_tag``)."""
    ROMManagerCls, import_err = _try_import_rom_manager()
    if ROMManagerCls is None:
        raise RuntimeError(f"ROMManager unavailable: {import_err}")
    manager = ROMManagerCls(shapes_config_path=SHAPES_CONFIG)
    result = manager.solve_online(rom_shape, params=lhs_params, nev=int(nev))
    freqs = list(result.get("freqs_hz") or [])
    _write_rom_stk_body_json(freqs, ROM_STK_JSON)
    result["stk_body_json"] = str(ROM_STK_JSON)
    return result


def _on_generate_3d_guitar_model(
    *,
    fom_mode: bool,
    lhs_params: Dict[str, Any],
    rom_shape: str,
    note_hz: float,
    top_wood_id: str,
    q_mode: str,
) -> None:
    """Single production entry: mesh build → ROM or FOM engine → STK audio."""
    save_cfg()
    py_exe = sys.executable

    if FEM_FOM_JSON.exists():
        FEM_FOM_JSON.unlink()
    if ROM_STK_JSON.exists():
        ROM_STK_JSON.unlink()

    with st.spinner("Building 3D mesh..."):
        _build_engineering_mesh(py_exe)

    if fom_mode:
        with st.status("FOM engine (developer diagnostics)...", expanded=True) as fem_status:

            def _cb(msg: str) -> None:
                fem_status.update(label=msg, state="running", expanded=True)

            try:
                _cb("Starting full-order FEM solve...")
                _execute_fom_engine(CONFIG_PATH, _cb)
                fem_status.update(label="FOM simulation complete.", state="complete", expanded=False)
            except Exception as exc:
                fem_status.update(label="FOM simulation failed.", state="error", expanded=True)
                raise exc
        st.session_state.last_engine = "fom"
        stk_json = FEM_FOM_JSON
    else:
        with st.spinner("ROM engine (reduced basis online solve)..."):
            rom_result = _execute_rom_engine(
                rom_shape=rom_shape,
                lhs_params=lhs_params,
                nev=0,
            )
            st.session_state.rom_last_result = rom_result
        st.session_state.last_engine = "rom"
        stk_json = ROM_STK_JSON

    with st.spinner("Synthesizing audio (STK)..."):
        _run_stk_synthesis(
            body_json=stk_json,
            top_wood_id=top_wood_id,
            note_hz=note_hz,
            q_mode=q_mode,
        )

    st.session_state.fem_ready = True
    st.session_state.stk_body_json = str(stk_json)


def _modal_rows_from_packaged_rom(npz_path: Path) -> List[Dict[str, Any]]:
    with np.load(npz_path, allow_pickle=True) as z:
        if "frequencies" in z.files:
            freqs = np.asarray(z["frequencies"], dtype=np.float64).reshape(-1)
        elif "freqs_hz" in z.files:
            freqs = np.asarray(z["freqs_hz"], dtype=np.float64).reshape(-1)
        else:
            raise KeyError("NPZ missing frequencies / freqs_hz")
        n = int(freqs.size)
        tag1 = (
            np.asarray(z["tag1_ratio"], dtype=np.float64).reshape(-1)
            if "tag1_ratio" in z.files
            else np.full(n, np.nan)
        )
        tag3 = (
            np.asarray(z["tag3_ratio"], dtype=np.float64).reshape(-1)
            if "tag3_ratio" in z.files
            else np.full(n, np.nan)
        )
        pfrac = (
            np.asarray(z["p_frac"], dtype=np.float64).reshape(-1)
            if "p_frac" in z.files
            else np.full(n, np.nan)
        )
        dom = (
            [str(x).strip() for x in np.asarray(z["dominant_tag"]).reshape(-1)]
            if "dominant_tag" in z.files
            else None
        )
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {
            "id": i + 1,
            "hz": float(freqs[i]),
            "tag1_ratio": float(tag1[i]) if i < tag1.size else float("nan"),
            "tag3_ratio": float(tag3[i]) if i < tag3.size else float("nan"),
            "p_frac": float(pfrac[i]) if i < pfrac.size else float("nan"),
        }
        if dom is not None and i < len(dom) and dom[i]:
            row["dominant_tag"] = dom[i]
        else:
            row["dominant_tag"] = dominant_tag_for_row(row)
        rows.append(row)
    return rows


def _modal_rows_from_candidates_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = list(data.get("candidates", [])) if isinstance(data, dict) else list(data)
    rows: List[Dict[str, Any]] = []
    for c in raw:
        try:
            row = {
                "id": int(c.get("id")),
                "hz": float(c.get("hz")),
                "tag1_ratio": float(c.get("tag1_ratio", 0.0) or 0.0),
                "tag3_ratio": float(c.get("tag3_ratio", 0.0) or 0.0),
                "wood_participation": float(c.get("wood_participation", 0.0) or 0.0),
                "uniqueness": float(c.get("uniqueness", 0.0) or 0.0),
            }
            if c.get("p_frac") is not None:
                row["p_frac"] = float(c.get("p_frac"))
            row["dominant_tag"] = str(c.get("dominant_tag") or dominant_tag_for_row(row))
            rows.append(row)
        except (TypeError, ValueError, KeyError):
            continue
    return rows


def _render_modal_audit_plot(rows: List[Dict[str, Any]], *, title: str) -> None:
    if not rows:
        st.warning("No modal rows to plot.")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = sorted(rows, key=lambda r: float(r["hz"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    for ax, ykey, ylabel in (
        (axes[0], "tag1_ratio", "Top plate energy ratio (tag 1)"),
        (axes[1], "p_frac", "Pressure fraction (log scale)"),
    ):
        for tag in (DOMINANT_TAG_TOP, DOMINANT_TAG_BACK):
            subset = [r for r in all_rows if r.get("dominant_tag") == tag]
            if not subset:
                continue
            xs = [float(r["hz"]) for r in subset]
            ys = [float(r.get(ykey, 0.0) or 0.0) for r in subset]
            if ykey == "p_frac":
                ys = [max(y, 1.0e-20) for y in ys]
            ax.scatter(
                xs,
                ys,
                c=DOMINANT_COLOR[tag],
                marker="o" if tag == DOMINANT_TAG_TOP else "^",
                s=48,
                alpha=0.9,
                label=tag,
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        if ykey == "p_frac":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    top_n = sum(1 for r in all_rows if r.get("dominant_tag") == DOMINANT_TAG_TOP)
    fig.suptitle(f"{title} | modes={len(all_rows)} Top={top_n} Back={len(all_rows) - top_n}")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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
st.sidebar.header("1. Shape & materials")
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

wood_id = st.sidebar.selectbox(
    "Soundboard wood",
    TOP_WOOD_IDS,
    index=0,
    format_func=_wood_option_label,
    help="Top plate material (spruce, cedar).",
)
body_wood_id = st.sidebar.selectbox(
    "Back & sides wood",
    BACK_WOOD_IDS,
    index=0,
    format_func=_wood_option_label,
    help="Back plate and rim material.",
)
top_wood_id = wood_id
back_wood_id = body_wood_id
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

st.sidebar.header("3. Sound")
pitch_mode = st.sidebar.radio("Input Mode", ["Musical Notes", "Manual Hz"])
if pitch_mode == "Musical Notes":
    note_name = st.sidebar.selectbox("Select Note", list(NOTES_DICT.keys()), index=1)
    note_hz = NOTES_DICT[note_name]
else:
    note_hz = st.sidebar.number_input("Frequency (Hz)", value=110.0)

q_mode = st.sidebar.radio("Q Estimation", ["Mean (Stable)", "Random (Realistic)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Engineering tools")
if st.sidebar.button("Open full model in Gmsh", use_container_width=True):
    save_cfg()
    with st.spinner("Opening Gmsh GUI..."):
        subprocess.run([sys.executable, str(GEOMETRY_SCRIPT)])

with st.sidebar.expander("Advanced Diagnostics", expanded=False):
    developer_fom_mode = st.checkbox(
        "Enable Developer FOM Mode",
        value=bool(st.session_state.developer_fom_mode),
        help="When enabled, runs the legacy full-order FEM path and STK from fem_3d_output.json.",
    )
    st.session_state.developer_fom_mode = bool(developer_fom_mode)
    if developer_fom_mode:
        st.caption("Diagnostics: FOM + fem_3d_output.json. Default users stay on ROM.")

lhs_params = _gui_lhs_params(
    shape_type=shape_type,
    length=L,
    width=W,
    depth=D,
    thickness=thick,
    hole_radius=hr,
    lower_bout=lower_bout,
    upper_bout=upper_bout,
    waist=waist,
    soundhole_y=soundhole_y,
    top_wood=top_wood_id,
    back_wood=back_wood_id,
)

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
# --- Process Control (dual engine routing) ---
# ==========================================
st.markdown("### Generate 3D Guitar Model")
fom_mode = bool(st.session_state.developer_fom_mode)
engine_label = "FOM (developer)" if fom_mode else "ROM (production)"
st.caption(f"Active engine: **{engine_label}** — mesh, modal synthesis, and STK in one step.")

if "show_success_msg" not in st.session_state:
    st.session_state.show_success_msg = False

if st.button("Generate 3D Guitar Model", use_container_width=True, type="primary"):
    try:
        _on_generate_3d_guitar_model(
            fom_mode=fom_mode,
            lhs_params=lhs_params,
            rom_shape=DEFAULT_ROM_SHAPE,
            note_hz=float(note_hz),
            top_wood_id=top_wood_id,
            q_mode=q_mode,
        )
        st.session_state.show_success_msg = True
        st.rerun()
    except Exception as exc:
        st.session_state.fem_ready = False
        st.error(f"Generate 3D Guitar Model failed: {exc}")

if st.session_state.show_success_msg:
    eng = str(st.session_state.get("last_engine", "")).upper() or "—"
    st.success(f"Model ready ({eng} engine). Mesh and audio updated below.")
    st.session_state.show_success_msg = False

if st.session_state.fem_ready and WAV_OUTPUT.is_file():
    st.audio(str(WAV_OUTPUT))
    stk_src = st.session_state.get("stk_body_json", "")
    if stk_src:
        st.caption(f"STK body modes: `{stk_src}`")

if st.session_state.get("last_engine") == "rom" and st.session_state.get("rom_last_result"):
    with st.expander("Last ROM solve (diagnostics)", expanded=False):
        st.json(st.session_state.rom_last_result)

st.divider()

# ==========================================
# --- ROM: unified modal basis & audit ---
# ==========================================
st.subheader("4. ROM — unified modal basis & audit")
st.caption(
    "Coupled ROM uses every mode in the packaged NPZ / reduced basis (30–40 typical). "
    "`dominant_tag` (Top/Back) is diagnostic only and is never used to filter modes or audio."
)

rom_tab_audit, rom_tab_online, rom_tab_table = st.tabs(
    ["Modal pool audit", "Online ROM solve", "Mode table"]
)

with rom_tab_audit:
    audit_sources: List[Tuple[str, Path]] = []
    if PACKAGED_ROM_NPZ.is_file():
        audit_sources.append(("Packaged ROM (pipeline)", PACKAGED_ROM_NPZ))
    if CANDIDATES_LOG.is_file():
        audit_sources.append(("Harvest candidates log", CANDIDATES_LOG))
    if ROM_CLASSIC_SNAPSHOTS.is_dir():
        for p in sorted(ROM_CLASSIC_SNAPSHOTS.glob("snapshot_*.npz")):
            audit_sources.append((f"Snapshot {p.name}", p))
    if not audit_sources:
        st.info("Run the FEM pipeline (Steps A–C) to create `final_guitar_rom.npz` or snapshot NPZ files.")
    else:
        labels = [x[0] for x in audit_sources]
        pick = st.selectbox("Audit data source", range(len(labels)), format_func=lambda i: labels[i])
        src_path = audit_sources[int(pick)][1]
        try:
            if src_path == CANDIDATES_LOG:
                audit_rows = _modal_rows_from_candidates_json(src_path)
            else:
                audit_rows = _modal_rows_from_packaged_rom(src_path)
            top_n = sum(1 for r in audit_rows if r.get("dominant_tag") == DOMINANT_TAG_TOP)
            st.metric("Modes in pool", len(audit_rows))
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Top-dominant", top_n)
            col_m2.metric("Back-dominant", len(audit_rows) - top_n)
            col_m3.metric("Source", src_path.name)
            _render_modal_audit_plot(
                audit_rows,
                title=f"Unified frequency axis — {labels[int(pick)]}",
            )
        except Exception as exc:
            st.error(f"Audit plot failed: {exc}")

with rom_tab_online:
    ROMManagerCls, rom_import_err = _try_import_rom_manager()
    if ROMManagerCls is None:
        st.warning(f"ROMManager unavailable (MPI/PETSc required for online solve): {rom_import_err}")
    else:
        shape_names = ["classic"]
        if SHAPES_CONFIG.is_file():
            try:
                shape_names = sorted(json.loads(SHAPES_CONFIG.read_text(encoding="utf-8")).get("shapes", {}).keys())
            except Exception:
                pass
        rom_shape = st.selectbox("ROM shape", shape_names, index=0)
        nev_rom = st.number_input("Eigenvalues to return (0 = all basis modes)", min_value=0, max_value=128, value=5)
        basis_path = BASE_DIR / "ROM" / rom_shape / "reduced_basis.npz"
        if basis_path.is_file():
            try:
                V, bmeta = ROMManagerCls.read_reduced_basis_npz(basis_path)
                st.success(
                    f"Reduced basis loaded: DOF={bmeta['num_dof']} × modes={bmeta['num_basis_modes']} "
                    f"(selected_rank={bmeta['selected_rank']})"
                )
            except Exception as exc:
                st.error(f"Could not read reduced basis: {exc}")
        else:
            st.info(f"No `{basis_path.name}` yet. Run `build_basis` after pipeline snapshots exist.")

        if st.button("Run ROMManager.solve_online()", type="primary", use_container_width=True):
            with st.spinner("Projecting operators onto reduced basis..."):
                try:
                    manager = ROMManagerCls(shapes_config_path=SHAPES_CONFIG)
                    result = manager.solve_online(
                        rom_shape,
                        params=lhs_params,
                        nev=int(nev_rom),
                    )
                    st.json(result)
                    if result.get("freqs_hz"):
                        st.line_chart(
                            {
                                "ROM frequency (Hz)": result["freqs_hz"],
                            }
                        )
                except Exception as exc:
                    st.error(f"Online ROM solve failed: {exc}")

with rom_tab_table:
    table_path: Optional[Path] = None
    if PACKAGED_ROM_NPZ.is_file():
        table_path = PACKAGED_ROM_NPZ
    elif ROM_CLASSIC_SNAPSHOTS.is_dir():
        snaps = sorted(ROM_CLASSIC_SNAPSHOTS.glob("snapshot_*.npz"))
        if snaps:
            table_path = snaps[-1]
    if table_path is None:
        st.info("No packaged ROM NPZ available for the mode table.")
    else:
        try:
            trows = _modal_rows_from_packaged_rom(table_path)
            prev_f = None
            table_out = []
            for i, r in enumerate(sorted(trows, key=lambda x: float(x["hz"]))):
                gap = "—"
                if prev_f is not None:
                    gap = f"{float(r['hz']) - prev_f:.4f}"
                pf = r.get("p_frac")
                table_out.append(
                    {
                        "Mode": i + 1,
                        "Frequency (Hz)": float(r["hz"]),
                        "Δ from prev (Hz)": gap,
                        "Dominant tag": str(r.get("dominant_tag", "")),
                        "tag1_ratio": float(r.get("tag1_ratio", float("nan"))),
                        "tag3_ratio": float(r.get("tag3_ratio", float("nan"))),
                        "p_frac": float(pf) if pf is not None and np.isfinite(float(pf)) else None,
                    }
                )
                prev_f = float(r["hz"])
            st.dataframe(table_out, use_container_width=True, hide_index=True)
            st.caption(f"Loaded from `{table_path.relative_to(BASE_DIR).as_posix()}`")
        except Exception as exc:
            st.error(f"Mode table failed: {exc}")

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