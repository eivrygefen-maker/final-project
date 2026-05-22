"""
3D Guitar Simulator — Streamlit UI.

Workflow
--------
Sketch mode  : slider change → fast Gmsh preview → PyVista wireframe (spatial colors).
Save Changes : engineering mesh → detailed model on screen → ROM/FOM acoustics → physics ready.
Generate Sound : STK synthesis from modal frequencies (E2–E4 note range).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "FEM" / "configs" / "guitar_3d.json"
GEOMETRY_SCRIPT = BASE_DIR / "FEM" / "geometry" / "build_3d_guitar.py"
STK_BINARY = BASE_DIR / "cpp" / "guitar_stk"
WAV_OUTPUT = BASE_DIR / "audio" / "guitar_sound.wav"
MESH_FILE = BASE_DIR / "FEM" / "mesh" / "guitar_3d.msh"
PREVIEW_MESH_FILE = BASE_DIR / "FEM" / "mesh" / "preview_mesh.msh"
FEM_FOM_JSON = BASE_DIR / "FEM" / "outputs" / "fem_3d_output.json"
ROM_STK_JSON = BASE_DIR / "FEM" / "outputs" / "rom_stk_body.json"
SHAPES_CONFIG = BASE_DIR / "FEM" / "configs" / "rom_shapes.json"
ROM_CLASSIC_SNAPSHOTS = BASE_DIR / "ROM" / "classic" / "snapshots"
ROM_REDUCED_BASIS = BASE_DIR / "ROM" / "classic" / "reduced_basis.npz"
PACKAGED_ROM_NPZ = BASE_DIR / "FEM" / "SORTING" / "final_guitar_rom.npz"
SELECTED_MODES_CSV = BASE_DIR / "FEM" / "SORTING" / "selected_modes.csv"
FALLBACK_MODES_CSV = BASE_DIR / "FEM" / "configs" / "archive" / "selected_modes_SIM1.csv"

DEFAULT_ROM_SHAPE = "classic"
SHAPE_OPTIONS = ("Classical", "Dreadnought", "Box")
HOLE_RADIUS_MAX_M = 0.12
SOUNDHOLE_FROM_NECK_RATIO = 0.43
TOP_Z_BAND_FRAC = 0.10
HOLE_VIS_COLOR = "#0c0c0c"
SHELL_VIS_TAGS = frozenset({1, 2, 3, 4})

NOTES: Dict[str, float] = {
    "E2": 82.41,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63,
}

sys.path.insert(0, str(BASE_DIR / "FEM" / "scripts"))
sys.path.insert(0, str(BASE_DIR / "FEM" / "rom"))

import fem_main_3d  # noqa: E402
from wood_library import (  # noqa: E402
    ALL_WOOD_IDS,
    material_block_for_id,
    plot_color_for_wood,
    wood_display_name,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pv.OFF_SCREEN = True

st.set_page_config(page_title="Guitar Simulator", layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults = {
        "physics_ready": False,
        "physics_geom_fp": "",
        "live_preview_fp": "",
        "stk_body_json": "",
        "acoustics_pending": False,
        "developer_fom_mode": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


def _load_saved_geometry() -> Dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f).get("geometry", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def soundhole_center_x(body_length: float) -> float:
    L = float(body_length)
    return 0.5 * L - SOUNDHOLE_FROM_NECK_RATIO * L


def shape_limits(shape: str) -> Tuple[float, float, float, float, float, float, float, float, float]:
    if shape == "Classical":
        return 0.35, 0.60, 0.48, 0.20, 0.45, 0.37, 0.08, 0.15, 0.10
    if shape == "Dreadnought":
        return 0.45, 0.70, 0.51, 0.30, 0.55, 0.40, 0.10, 0.20, 0.12
    return 0.10, 1.00, 0.40, 0.10, 0.80, 0.30, 0.01, 0.50, 0.10


def build_geometry_state(
    *,
    shape_type: str,
    length: float,
    width: float,
    depth: float,
    thickness: float,
    hole_radius: float,
) -> Dict[str, Any]:
    W = float(width)
    return {
        "shape_type": shape_type,
        "length": float(length),
        "width": W,
        "depth": float(depth),
        "thickness": float(thickness),
        "hole_radius": min(float(hole_radius), HOLE_RADIUS_MAX_M),
        "lower_bout": W,
        "upper_bout": W * 0.75,
        "waist": W * 0.65,
        "soundhole_y": 0.0,
        "soundhole_x": soundhole_center_x(length),
        "soundhole_from_neck_ratio": SOUNDHOLE_FROM_NECK_RATIO,
    }


def geometry_fingerprint(geom: Dict[str, Any], top_wood: str, back_wood: str) -> str:
    payload = {**geom, "top_wood_id": top_wood, "back_wood_id": back_wood}
    return json.dumps(payload, sort_keys=True, default=str)


def lhs_params_from_ui(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
) -> Dict[str, Any]:
    return {
        "geometry.shape_type": geom["shape_type"],
        "geometry.length": geom["length"],
        "geometry.width": geom["width"],
        "geometry.depth": geom["depth"],
        "geometry.thickness": geom["thickness"],
        "geometry.hole_radius": geom["hole_radius"],
        "geometry.lower_bout": geom["lower_bout"],
        "geometry.upper_bout": geom["upper_bout"],
        "geometry.waist": geom["waist"],
        "geometry.soundhole_y": geom["soundhole_y"],
        "materials.top.wood_id": top_wood,
        "materials.back.wood_id": back_wood,
    }


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def save_config(geom: Dict[str, Any], *, top_wood: str, back_wood: str) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if CONFIG_PATH.is_file():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["geometry"] = {
        **geom,
        "top_wood_id": top_wood,
        "back_wood_id": back_wood,
    }
    data["materials"] = {
        "top": {**material_block_for_id(top_wood), "name": wood_display_name(top_wood), "wood_id": top_wood},
        "back": {**material_block_for_id(back_wood), "name": wood_display_name(back_wood), "wood_id": back_wood},
        "air": {"density": 1.204, "speed_of_sound": 343.0},
    }
    solver = dict(data.get("solver") or {})
    solver["mesh_file"] = str(MESH_FILE)
    solver.setdefault("num_modes", 50)
    data["solver"] = solver
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Gmsh mesh build
# ---------------------------------------------------------------------------
def gmsh_argv() -> List[str]:
    return [
        sys.executable,
        str(GEOMETRY_SCRIPT),
        "--config",
        str(CONFIG_PATH.resolve()),
        "-nopopup",
    ]


def run_gmsh_preview() -> None:
    save_config(st.session_state["_geom"], top_wood=st.session_state["_top_wood"], back_wood=st.session_state["_back_wood"])
    if PREVIEW_MESH_FILE.is_file():
        PREVIEW_MESH_FILE.unlink()
    env = {**os.environ, "FEM_ALLOW_PREVIEW": "1"}
    result = subprocess.run(gmsh_argv(), capture_output=True, text=True, env=env, cwd=str(BASE_DIR))
    if not PREVIEW_MESH_FILE.is_file():
        tail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"Sketch mesh failed.\n{tail}")


def run_gmsh_engineering() -> None:
    save_config(st.session_state["_geom"], top_wood=st.session_state["_top_wood"], back_wood=st.session_state["_back_wood"])
    if MESH_FILE.is_file():
        MESH_FILE.unlink()
    env = {k: v for k, v in os.environ.items() if k != "FEM_ALLOW_PREVIEW"}
    result = subprocess.run(gmsh_argv(), capture_output=True, text=True, env=env, cwd=str(BASE_DIR))
    if not MESH_FILE.is_file():
        tail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"Engineering mesh failed.\n{tail}")


# ---------------------------------------------------------------------------
# Mesh I/O & spatial coloring (decoupled from Gmsh physical tags)
# ---------------------------------------------------------------------------
def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = str(hex_color).strip().lstrip("#")
    if len(h) != 6:
        return (0.7, 0.5, 0.35)
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def load_surface_mesh(msh_path: Path) -> Optional[pv.PolyData]:
    try:
        import meshio
    except ImportError:
        return None
    try:
        msh = meshio.read(str(msh_path))
        tri = msh.get_cells_type("triangle")
        if tri is None or len(tri) == 0:
            return None
        points = np.asarray(msh.points, dtype=np.float64)
        tri = np.asarray(tri, dtype=np.int64)
        phys = msh.cell_data_dict.get("gmsh:physical")
        if phys is not None and "triangle" in phys:
            tags = np.asarray(phys["triangle"], dtype=np.int32).ravel()
            tri = tri[np.isin(tags, list(SHELL_VIS_TAGS))]
        if tri.shape[0] < 3:
            return None
        faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri]).ravel()
        poly = pv.PolyData(points, faces)
        poly.compute_normals(
            cell_normals=False,
            point_normals=True,
            feature_angle=30,
            split_vertices=True,
            inplace=True,
            auto_orient_normals=True,
        )
        return poly
    except Exception:
        return None


def apply_spatial_colormap(
    mesh: pv.PolyData,
    *,
    top_color: str,
    back_color: str,
    body_length: float,
    hole_radius: float,
) -> pv.PolyData:
    centers = mesh.cell_centers().points
    z = centers[:, 2]
    x = centers[:, 0]
    y = centers[:, 1]
    zmax, zmin = float(np.max(z)), float(np.min(z))
    dz = max(zmax - zmin, 1.0e-9)
    top_z = zmax - TOP_Z_BAND_FRAC * dz
    hole_x = soundhole_center_x(body_length)
    hr2 = float(hole_radius) ** 2

    top_rgb = np.array(hex_to_rgb01(top_color), dtype=np.float32)
    back_rgb = np.array(hex_to_rgb01(back_color), dtype=np.float32)
    hole_rgb = np.array(hex_to_rgb01(HOLE_VIS_COLOR), dtype=np.float32)

    is_top = z >= top_z
    in_hole = is_top & ((x - hole_x) ** 2 + y**2 <= hr2 * 1.05)

    colors = np.tile(back_rgb, (mesh.n_cells, 1))
    colors[is_top & ~in_hole] = top_rgb
    colors[in_hole] = hole_rgb

    out = mesh.copy(deep=True)
    out.cell_data["rgb"] = colors
    return out


def render_guitar(
    mesh: Optional[pv.PolyData],
    *,
    top_color: str,
    back_color: str,
    body_length: float,
    hole_radius: float,
    sketch_mode: bool,
    plot_key: str,
) -> None:
    plotter = pv.Plotter(window_size=[1100, 620], lighting="three lights")
    plotter.background_color = "#f4f4f9"
    if mesh is not None and mesh.n_cells > 0:
        colored = apply_spatial_colormap(
            mesh,
            top_color=top_color,
            back_color=back_color,
            body_length=body_length,
            hole_radius=hole_radius,
        )
        plotter.add_mesh(
            colored,
            scalars="rgb",
            rgb=True,
            preference="cell",
            show_edges=sketch_mode,
            edge_color="#3d2817" if sketch_mode else "#5c4033",
            line_width=0.8 if sketch_mode else 0.4,
            smooth_shading=not sketch_mode,
            opacity=1.0,
        )
    else:
        plotter.add_text("Preview unavailable", position="upper_left", font_size=12)
    plotter.camera_position = [(0.0, -0.45, 1.05), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.enable_anti_aliasing("ssaa")
    stpyvista(plotter, key=plot_key)
    plotter.close()


def get_display_mesh(geom_fp: str, *, sketch_mode: bool) -> Tuple[Optional[pv.PolyData], bool]:
    """Return (mesh, is_sketch). Engineering mesh wins when fingerprint matches."""
    eng_match = (
        st.session_state.physics_geom_fp == geom_fp
        and MESH_FILE.is_file()
        and not sketch_mode
    )
    if eng_match:
        return load_surface_mesh(MESH_FILE), False

    if st.session_state.live_preview_fp == geom_fp and PREVIEW_MESH_FILE.is_file():
        cached = load_surface_mesh(PREVIEW_MESH_FILE)
        if cached is not None:
            return cached, True

    with st.spinner("Building sketch mesh…"):
        run_gmsh_preview()
    st.session_state.live_preview_fp = geom_fp
    return load_surface_mesh(PREVIEW_MESH_FILE), True


# ---------------------------------------------------------------------------
# Acoustics (ROM / FOM)
# ---------------------------------------------------------------------------
def try_import_rom():
    try:
        from rom_manager import ROMManager  # noqa: WPS433

        return ROMManager, None
    except Exception as exc:
        return None, exc


def load_freqs_npz(path: Path) -> List[float]:
    with np.load(path, allow_pickle=True) as z:
        if "frequencies" in z.files:
            arr = np.asarray(z["frequencies"], dtype=np.float64).ravel()
        elif "freqs_hz" in z.files:
            arr = np.asarray(z["freqs_hz"], dtype=np.float64).ravel()
        else:
            raise KeyError("missing frequencies")
    return [float(f) for f in arr if f > 0]


def load_freqs_csv(path: Path) -> List[float]:
    import csv

    out: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hz = row.get("hz") or row.get("frequency_hz")
            if hz:
                try:
                    v = float(hz)
                    if v > 0:
                        out.append(v)
                except ValueError:
                    pass
    return sorted(out)


def rom_frequency_fallback() -> Tuple[List[float], str]:
    for path, label in (
        (PACKAGED_ROM_NPZ, "packaged_rom"),
        (SELECTED_MODES_CSV, "selected_csv"),
        (FALLBACK_MODES_CSV, "archive_csv"),
    ):
        if not path.is_file():
            continue
        try:
            freqs = load_freqs_npz(path) if path.suffix == ".npz" else load_freqs_csv(path)
            if freqs:
                return freqs, label
        except Exception:
            continue
    return [], ""


def write_stk_body_json(freqs_hz: Sequence[float], out_path: Path) -> Path:
    freqs = [float(f) for f in freqs_hz if float(f) > 0.0]
    if not freqs:
        raise ValueError("No modal frequencies for STK.")
    weights = [1.0 / (1.0 + 0.25 * i) for i in range(len(freqs))]
    wmax = max(weights)
    weights = [w / wmax for w in weights]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "analysis": "rom_online_body",
                "modes_hz": freqs,
                "mode_weights": weights,
                "num_modes": len(freqs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def run_rom_acoustics(lhs_params: Dict[str, Any]) -> Path:
    ROMManager, err = try_import_rom()
    if ROMManager is None:
        raise RuntimeError(f"ROM unavailable: {err}")

    if not ROM_REDUCED_BASIS.is_file() and ROM_CLASSIC_SNAPSHOTS.is_dir():
        snaps = list(ROM_CLASSIC_SNAPSHOTS.glob("snapshot_*.npz"))
        if snaps:
            ROMManager(shapes_config_path=SHAPES_CONFIG).build_basis(DEFAULT_ROM_SHAPE)

    manager = ROMManager(shapes_config_path=SHAPES_CONFIG)
    try:
        result = manager.solve_online(DEFAULT_ROM_SHAPE, params=lhs_params, nev=0)
        freqs = list(result.get("freqs_hz") or [])
        write_stk_body_json(freqs, ROM_STK_JSON)
        return ROM_STK_JSON
    except RuntimeError as exc:
        if "Reduced basis missing" not in str(exc):
            raise
        freqs, _ = rom_frequency_fallback()
        if not freqs:
            raise RuntimeError(
                f"{exc}\nRun: python FEM/scripts/rom_pipeline.py collect classic && "
                "python FEM/scripts/rom_pipeline.py build-basis classic"
            ) from exc
        write_stk_body_json(freqs, ROM_STK_JSON)
        st.warning("Using archived mode frequencies (ROM basis not on this machine).")
        return ROM_STK_JSON


def run_fom_acoustics() -> Path:
    def _cb(msg: str) -> None:
        st.write(msg)

    if FEM_FOM_JSON.is_file():
        FEM_FOM_JSON.unlink()
    fem_main_3d.run_fem_3d_simulation(str(CONFIG_PATH), status_callback=_cb)
    return FEM_FOM_JSON


def run_acoustics(lhs_params: Dict[str, Any]) -> Path:
    if ROM_STK_JSON.is_file():
        ROM_STK_JSON.unlink()
    if st.session_state.developer_fom_mode:
        with st.spinner("Running full-order acoustics…"):
            return run_fom_acoustics()
    with st.spinner("Computing acoustics…"):
        return run_rom_acoustics(lhs_params)


# ---------------------------------------------------------------------------
# STK audio
# ---------------------------------------------------------------------------
def append_silence_wav(path: Path, seconds: float) -> None:
    try:
        with wave.open(str(path), "rb") as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
        pad = b"\x00" * int(seconds * params.framerate * params.sampwidth * params.nchannels)
        with wave.open(str(path), "wb") as w:
            w.setparams(params)
            w.writeframes(pad + frames)
    except Exception:
        pass


def run_stk(*, body_json: Path, top_wood: str, note_hz: float, q_mode: str) -> None:
    if not body_json.is_file():
        raise FileNotFoundError("Acoustics data missing — save changes first.")
    if not STK_BINARY.is_file():
        raise FileNotFoundError(f"STK binary not found: {STK_BINARY}")
    top_q = material_block_for_id(top_wood)
    cmd = [
        str(STK_BINARY),
        "--fem_json",
        str(body_json),
        "--note_hz",
        str(note_hz),
        "--dur",
        "3.0",
        "--mix",
        "0.98",
        "--wet_gain",
        "400",
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
        cmd += ["--q_min", str(top_q["q_min"]), "--q_max", str(top_q["q_max"]), "--q_mode", "random"]
    else:
        cmd += ["--q", str((top_q["q_min"] + top_q["q_max"]) / 2.0)]
    subprocess.run(cmd, check=False)
    append_silence_wav(WAV_OUTPUT, 0.3)


# ---------------------------------------------------------------------------
# State machine actions
# ---------------------------------------------------------------------------
def invalidate_physics() -> None:
    st.session_state.physics_ready = False
    st.session_state.acoustics_pending = False
    st.session_state.stk_body_json = ""


def on_save_changes(geom_fp: str, lhs_params: Dict[str, Any]) -> None:
    """Phase 1: engineering mesh only; acoustics runs on next rerun after plot."""
    invalidate_physics()
    with st.spinner("Building detailed model…"):
        run_gmsh_engineering()
    st.session_state.physics_geom_fp = geom_fp
    st.session_state.acoustics_pending = True
    st.rerun()


def on_generate_sound(*, top_wood: str, note_hz: float, q_mode: str) -> None:
    body = Path(st.session_state.stk_body_json)
    with st.spinner("Synthesizing sound…"):
        run_stk(body_json=body, top_wood=top_wood, note_hz=note_hz, q_mode=q_mode)
    if WAV_OUTPUT.is_file():
        st.audio(WAV_OUTPUT.read_bytes(), format="audio/wav")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    saved = _load_saved_geometry()
    saved_shape = str(saved.get("shape_type", "Classical"))
    if saved_shape not in SHAPE_OPTIONS:
        saved_shape = "Classical"

    st.markdown(
        """
        <style>
        h1 { letter-spacing: 0.06em; font-weight: 700; }
        div[data-testid="stpyvista"] canvas { opacity: 1 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("GUITAR SIMULATOR")

    col_ctrl, col_vis = st.columns([0.22, 0.78], gap="large")

    with col_ctrl:
        st.subheader("Parameters")
        shape = st.selectbox("Shape", SHAPE_OPTIONS, index=SHAPE_OPTIONS.index(saved_shape))
        L_lo, L_hi, L_def, W_lo, W_hi, W_def, D_lo, D_hi, D_def = shape_limits(shape)

        def _wood_idx(key: str, default: str) -> int:
            wid = str(saved.get(key, default)).strip().lower()
            return ALL_WOOD_IDS.index(wid) if wid in ALL_WOOD_IDS else 0

        top_wood = st.selectbox(
            "Top wood",
            ALL_WOOD_IDS,
            index=_wood_idx("top_wood_id", "spruce"),
            format_func=lambda w: f"{w} — {wood_display_name(w)}",
        )
        back_wood = st.selectbox(
            "Back & sides wood",
            ALL_WOOD_IDS,
            index=_wood_idx("back_wood_id", "rosewood"),
            format_func=lambda w: f"{w} — {wood_display_name(w)}",
        )

        length = st.slider("Length (m)", L_lo, L_hi, float(saved.get("length", L_def)))
        width = st.slider("Width (m)", W_lo, W_hi, float(saved.get("width", W_def)))
        depth = st.slider("Depth (m)", D_lo, D_hi, float(saved.get("depth", D_def)))
        thickness = st.slider("Top thickness (m)", 0.003, 0.006, float(saved.get("thickness", 0.003)), 0.0005)
        hr_def = min(float(saved.get("hole_radius", 0.04)), HOLE_RADIUS_MAX_M)
        hole_r = st.slider("Hole radius (m)", 0.02, HOLE_RADIUS_MAX_M, hr_def, 0.0005, format="%.4f")

        st.markdown("**Sound**")
        note = st.selectbox("Note (E2–E4)", list(NOTES.keys()), index=1)
        note_hz = NOTES[note]
        q_mode = st.radio("Q", ["Mean (Stable)", "Random (Realistic)"], horizontal=True)

        with st.expander("Advanced"):
            st.session_state.developer_fom_mode = st.checkbox(
                "FOM developer mode",
                value=bool(st.session_state.developer_fom_mode),
            )

    geom = build_geometry_state(
        shape_type=shape,
        length=length,
        width=width,
        depth=depth,
        thickness=thickness,
        hole_radius=hole_r,
    )
    geom_fp = geometry_fingerprint(geom, top_wood, back_wood)
    lhs_params = lhs_params_from_ui(geom, top_wood=top_wood, back_wood=back_wood)

    st.session_state["_geom"] = geom
    st.session_state["_top_wood"] = top_wood
    st.session_state["_back_wood"] = back_wood

    if geom_fp != st.session_state.physics_geom_fp:
        invalidate_physics()
        st.session_state.live_preview_fp = ""

    sketch_mode = geom_fp != st.session_state.physics_geom_fp or not MESH_FILE.is_file()
    top_color = plot_color_for_wood(top_wood)
    back_color = plot_color_for_wood(back_wood)

    with col_vis:
        st.subheader("PREVIEW")

        if sketch_mode:
            st.caption("Sketch mode — adjust sliders, then Save Changes for the detailed model.")
        elif st.session_state.physics_ready:
            st.caption("Detailed model — acoustics ready.")
        elif st.session_state.acoustics_pending:
            st.caption("Detailed model — computing acoustics…")
        else:
            st.caption("Detailed model saved.")

        mesh, is_sketch = get_display_mesh(geom_fp, sketch_mode=sketch_mode)
        try:
            pv.set_jupyter_backend("static")
            render_guitar(
                mesh,
                top_color=top_color,
                back_color=back_color,
                body_length=geom["length"],
                hole_radius=geom["hole_radius"],
                sketch_mode=is_sketch,
                plot_key=f"v_{geom_fp[:16]}",
            )
        except Exception as exc:
            st.warning(f"Render error: {exc}")

        btn_save, btn_sound = st.columns(2)
        with btn_save:
            if st.button("SAVE CHANGES", type="primary", use_container_width=True):
                try:
                    on_save_changes(geom_fp, lhs_params)
                except Exception as exc:
                    st.session_state.physics_geom_fp = ""
                    invalidate_physics()
                    st.error(f"Save failed: {exc}")

        with btn_sound:
            if st.button(
                "GENERATE SOUND",
                use_container_width=True,
                disabled=not st.session_state.physics_ready,
            ):
                try:
                    on_generate_sound(top_wood=top_wood, note_hz=note_hz, q_mode=q_mode)
                except Exception as exc:
                    st.error(f"Sound failed: {exc}")

        if WAV_OUTPUT.is_file() and st.session_state.physics_ready:
            st.audio(WAV_OUTPUT.read_bytes(), format="audio/wav")

        # Phase 2 acoustics: only after detailed mesh is on screen
        if (
            st.session_state.acoustics_pending
            and st.session_state.physics_geom_fp == geom_fp
            and MESH_FILE.is_file()
        ):
            try:
                stk_path = run_acoustics(lhs_params)
                st.session_state.stk_body_json = str(stk_path)
                st.session_state.physics_ready = True
                st.session_state.acoustics_pending = False
                st.rerun()
            except Exception as exc:
                st.session_state.acoustics_pending = False
                invalidate_physics()
                st.error(f"Acoustics failed: {exc}")


main()
