"""
3D Guitar Simulator — strict display vs physics decoupling.

Pipelines
---------
Sketch (sliders)  : preview_mesh.msh — coarse wireframe, instant feedback.
Save Changes      : display_mesh.msh — uniform wood shell for PyVista ONLY.
Acoustics ROM     : reduced_basis.npz + ROMManager.solve_online (no FOM Gmsh).
Acoustics FOM     : guitar_3d.msh — full FSI volume mesh, FEM only (never PyVista).
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

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "FEM" / "configs" / "guitar_3d.json"
GEOMETRY_SCRIPT = BASE_DIR / "FEM" / "geometry" / "build_3d_guitar.py"
STK_BINARY = BASE_DIR / "cpp" / "guitar_stk"
WAV_OUTPUT = BASE_DIR / "audio" / "guitar_sound.wav"

# Visual meshes (PyVista only)
PREVIEW_MESH_FILE = BASE_DIR / "FEM" / "mesh" / "preview_mesh.msh"
DISPLAY_MESH_FILE = BASE_DIR / "FEM" / "mesh" / "display_mesh.msh"

# Physics mesh (FOM FEM only — never loaded into PyVista)
MESH_FILE = BASE_DIR / "FEM" / "mesh" / "guitar_3d.msh"
FEM_FOM_JSON = BASE_DIR / "FEM" / "outputs" / "fem_3d_output.json"
ROM_STK_JSON = BASE_DIR / "FEM" / "outputs" / "rom_stk_body.json"

SHAPES_CONFIG = BASE_DIR / "FEM" / "configs" / "rom_shapes.json"
ROM_ROOT = BASE_DIR / "ROM"
PACKAGED_ROM_NPZ = BASE_DIR / "FEM" / "SORTING" / "final_guitar_rom.npz"
SELECTED_MODES_CSV = BASE_DIR / "FEM" / "SORTING" / "selected_modes.csv"
FALLBACK_MODES_CSV = BASE_DIR / "FEM" / "configs" / "archive" / "selected_modes_SIM1.csv"

SHAPE_OPTIONS = ("Classical", "Dreadnought", "Box")
ROM_NAMESPACE = {"Classical": "classic", "Dreadnought": "dreadnought", "Box": "classic"}
HOLE_RADIUS_MAX_M = 0.08
SOUNDHOLE_FROM_NECK_RATIO = 0.5
TOP_Z_BAND_FRAC = 0.10
HOLE_VIS_COLOR = "#0c0c0c"
SHELL_VIS_TAGS = frozenset({1, 2, 3, 4})
DEFAULT_STK_NOTE_HZ = 110.0

FIXTURE_PRESETS = (
    "Standing Angled (3D)",
    "Standing Upright (Front)",
    "Laying Flat (Top View)",
    "Laying on Side (Profile)",
)
CAMERA_BY_FIXTURE: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = {
    "Standing Angled (3D)": ((0.0, -0.45, 1.05), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "Standing Upright (Front)": ((0.0, 0.0, 1.2), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "Laying Flat (Top View)": ((0.0, 0.0, 1.2), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "Laying on Side (Profile)": ((0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
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


def _init_session() -> None:
    for key, val in {
        "physics_ready": False,
        "saved_geom_fp": "",
        "live_preview_fp": "",
        "stk_body_json": "",
        "acoustics_pending": False,
        "developer_fom_mode": False,
        "show_display_mesh": False,
        "fixture_fp": "",
        "_clamp_ribs": False,
        "_pin_neck_fix": True,
        "_fixture_preset": "Standing Angled (3D)",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


def _load_saved_config() -> Dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def soundhole_center_x(body_length: float) -> float:
    return 0.5 * float(body_length) - SOUNDHOLE_FROM_NECK_RATIO * float(body_length)


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


def geometry_fingerprint(
    geom: Dict[str, Any],
    top_wood: str,
    back_wood: str,
    *,
    clamp_ribs: bool,
    pin_neck_fix: bool,
    fixture_preset: str,
) -> str:
    payload = {
        **geom,
        "top_wood_id": top_wood,
        "back_wood_id": back_wood,
        "clamp_ribs": clamp_ribs,
        "pin_neck_fix": pin_neck_fix,
        "fixture_preset": fixture_preset,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def lhs_params_from_ui(geom: Dict[str, Any], *, top_wood: str, back_wood: str) -> Dict[str, Any]:
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


def rom_namespace(shape_type: str) -> str:
    return ROM_NAMESPACE.get(str(shape_type).strip(), "classic")


def rom_basis_path(shape_type: str) -> Path:
    return ROM_ROOT / rom_namespace(shape_type) / "reduced_basis.npz"


def save_config(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
    clamp_ribs: bool,
    pin_neck_fix: bool,
    fixture_preset: str,
) -> None:
    data = _load_saved_config()
    data["geometry"] = {
        **geom,
        "top_wood_id": top_wood,
        "back_wood_id": back_wood,
        "fixture_preset": fixture_preset,
    }
    data["materials"] = {
        "top": {**material_block_for_id(top_wood), "name": wood_display_name(top_wood), "wood_id": top_wood},
        "back": {**material_block_for_id(back_wood), "name": wood_display_name(back_wood), "wood_id": back_wood},
        "air": {"density": 1.204, "speed_of_sound": 343.0},
    }
    solver = dict(data.get("solver") or {})
    solver["mesh_file"] = str(MESH_FILE)
    solver.setdefault("num_modes", 50)
    solver["clamp_ribs"] = bool(clamp_ribs)
    solver["eps_pin_fix_tag5"] = bool(pin_neck_fix)
    data["solver"] = solver
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _gmsh_cmd() -> List[str]:
    return [sys.executable, str(GEOMETRY_SCRIPT), "--config", str(CONFIG_PATH.resolve()), "-nopopup"]


def _run_gmsh(env: Dict[str, str], out_path: Path, label: str) -> None:
    save_config(
        st.session_state["_geom"],
        top_wood=st.session_state["_top_wood"],
        back_wood=st.session_state["_back_wood"],
        clamp_ribs=st.session_state["_clamp_ribs"],
        pin_neck_fix=st.session_state["_pin_neck_fix"],
        fixture_preset=st.session_state["_fixture_preset"],
    )
    if out_path.is_file():
        out_path.unlink()
    clean = {k: v for k, v in os.environ.items() if k not in ("FEM_ALLOW_PREVIEW", "FEM_ALLOW_DISPLAY", "FEM_ALLOW_FOM")}
    result = subprocess.run(_gmsh_cmd(), capture_output=True, text=True, env={**clean, **env}, cwd=str(BASE_DIR))
    if not out_path.is_file():
        tail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"{label} failed.\n{tail}")


def run_gmsh_sketch() -> None:
    _run_gmsh({"FEM_ALLOW_PREVIEW": "1"}, PREVIEW_MESH_FILE, "Sketch mesh")


def run_gmsh_display() -> None:
    _run_gmsh({"FEM_ALLOW_DISPLAY": "1"}, DISPLAY_MESH_FILE, "Display mesh")


def run_gmsh_fom() -> None:
    _run_gmsh({"FEM_ALLOW_FOM": "1"}, MESH_FILE, "FOM physics mesh")


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
    z, x, y = centers[:, 2], centers[:, 0], centers[:, 1]
    zmax, zmin = float(np.max(z)), float(np.min(z))
    top_z = zmax - TOP_Z_BAND_FRAC * max(zmax - zmin, 1e-9)
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
    fixture_preset: str,
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
            line_width=0.8 if sketch_mode else 0.35,
            smooth_shading=not sketch_mode,
        )
    else:
        plotter.add_text("Preview unavailable", position="upper_left", font_size=12)
    plotter.camera_position = CAMERA_BY_FIXTURE.get(fixture_preset, CAMERA_BY_FIXTURE["Standing Angled (3D)"])
    plotter.enable_anti_aliasing("ssaa")
    stpyvista(plotter, key=plot_key)
    plotter.close()


def display_mesh_active(geom_fp: str) -> bool:
    return (
        bool(st.session_state.show_display_mesh)
        and st.session_state.saved_geom_fp == geom_fp
        and DISPLAY_MESH_FILE.is_file()
    )


def get_view_mesh(geom_fp: str) -> Tuple[Optional[pv.PolyData], bool, str]:
    """PyVista loads display_mesh.msh or sketch preview only — never guitar_3d.msh."""
    if display_mesh_active(geom_fp):
        mesh = load_surface_mesh(DISPLAY_MESH_FILE)
        if mesh is not None:
            return mesh, False, "display_mesh.msh"
    if st.session_state.live_preview_fp == geom_fp and PREVIEW_MESH_FILE.is_file():
        cached = load_surface_mesh(PREVIEW_MESH_FILE)
        if cached is not None:
            return cached, True, "preview_mesh.msh"
    try:
        with st.spinner("Building sketch preview (low-poly)…"):
            run_gmsh_sketch()
    except Exception as exc:
        if PREVIEW_MESH_FILE.is_file():
            PREVIEW_MESH_FILE.unlink()
        raise RuntimeError(
            f"Sketch mesh failed ({exc}). Try adjusting depth/thickness or shape, then move a slider again."
        ) from exc
    st.session_state.live_preview_fp = geom_fp
    return load_surface_mesh(PREVIEW_MESH_FILE), True, "preview_mesh.msh"


def invalidate_saved_state() -> None:
    st.session_state.physics_ready = False
    st.session_state.acoustics_pending = False
    st.session_state.stk_body_json = ""
    st.session_state.show_display_mesh = False


def try_import_rom():
    try:
        from rom_manager import ROMManager  # noqa: WPS433

        return ROMManager, None
    except Exception as exc:
        return None, exc


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
            {"analysis": "rom_online_body", "modes_hz": freqs, "mode_weights": weights, "num_modes": len(freqs)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def run_rom_acoustics(lhs_params: Dict[str, Any], shape_type: str) -> Path:
    """Branch A: ROM only — no FOM Gmsh volume mesh."""
    ns = rom_namespace(shape_type)
    basis = rom_basis_path(shape_type)
    if not basis.is_file():
        raise RuntimeError(
            f"Reduced basis missing: {basis}\n"
            f"Run: python FEM/scripts/rom_pipeline.py collect {ns} --pool-size 8\n"
            f"     python FEM/scripts/rom_pipeline.py build-basis {ns}"
        )
    ROMManager, err = try_import_rom()
    if ROMManager is None:
        raise RuntimeError(f"ROM unavailable: {err}")
    manager = ROMManager(shapes_config_path=SHAPES_CONFIG)
    result = manager.solve_online(ns, params=lhs_params, nev=0)
    freqs = list(result.get("freqs_hz") or [])
    if not freqs:
        raise RuntimeError("ROM solve_online returned no frequencies.")
    write_stk_body_json(freqs, ROM_STK_JSON)
    return ROM_STK_JSON


def run_fom_acoustics() -> Path:
    """Branch B: build FOM mesh then full FEM — mesh never shown in PyVista."""
    with st.spinner("Building FOM physics mesh (FSI volume, not shown)…"):
        run_gmsh_fom()

    def _cb(msg: str) -> None:
        st.write(msg)

    if FEM_FOM_JSON.is_file():
        FEM_FOM_JSON.unlink()
    fem_main_3d.run_fem_3d_simulation(str(CONFIG_PATH), status_callback=_cb)
    return FEM_FOM_JSON


def run_acoustics(lhs_params: Dict[str, Any], shape_type: str) -> Path:
    if ROM_STK_JSON.is_file():
        ROM_STK_JSON.unlink()
    if st.session_state.developer_fom_mode:
        return run_fom_acoustics()
    return run_rom_acoustics(lhs_params, shape_type)


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


def run_stk(*, body_json: Path, top_wood: str) -> None:
    top_q = material_block_for_id(top_wood)
    q_mean = (float(top_q["q_min"]) + float(top_q["q_max"])) / 2.0
    subprocess.run(
        [
            str(STK_BINARY),
            "--fem_json",
            str(body_json),
            "--note_hz",
            str(DEFAULT_STK_NOTE_HZ),
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
            "--q",
            str(q_mean),
        ],
        check=False,
    )
    append_silence_wav(WAV_OUTPUT, 0.3)


def on_save_changes(geom_fp: str) -> None:
    invalidate_saved_state()
    st.session_state.live_preview_fp = ""
    with st.spinner("Building display mesh (wood shell, uniform 4 mm)…"):
        run_gmsh_display()
    st.session_state.saved_geom_fp = geom_fp
    st.session_state.show_display_mesh = True
    st.session_state.acoustics_pending = True
    st.rerun()


def main() -> None:
    saved = _load_saved_config().get("geometry", {}) or {}
    saved_solver = _load_saved_config().get("solver", {}) or {}
    saved_shape = str(saved.get("shape_type", "Classical"))
    if saved_shape not in SHAPE_OPTIONS:
        saved_shape = "Classical"
    saved_fixture = str(saved.get("fixture_preset", "Standing Angled (3D)"))
    if saved_fixture not in FIXTURE_PRESETS:
        saved_fixture = "Standing Angled (3D)"

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

    # Declare button row first so it renders directly under the title (filled after controls).
    row_btn = st.columns(2)
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

        with st.expander("Advanced"):
            st.session_state.developer_fom_mode = st.checkbox(
                "FOM developer mode (builds hidden FSI mesh + full FEM)",
                value=bool(st.session_state.developer_fom_mode),
            )

    clamp_ribs = bool(st.session_state.get("_clamp_ribs", saved_solver.get("clamp_ribs", False)))
    pin_neck = bool(st.session_state.get("_pin_neck_fix", saved_solver.get("eps_pin_fix_tag5", True)))
    fixture_preset = str(st.session_state.get("_fixture_preset", saved_fixture))

    geom = build_geometry_state(
        shape_type=shape,
        length=length,
        width=width,
        depth=depth,
        thickness=thickness,
        hole_radius=hole_r,
    )
    geom_fp = geometry_fingerprint(
        geom, top_wood, back_wood, clamp_ribs=clamp_ribs, pin_neck_fix=pin_neck, fixture_preset=fixture_preset
    )
    lhs_params = lhs_params_from_ui(geom, top_wood=top_wood, back_wood=back_wood)

    st.session_state["_geom"] = geom
    st.session_state["_top_wood"] = top_wood
    st.session_state["_back_wood"] = back_wood
    st.session_state["_clamp_ribs"] = clamp_ribs
    st.session_state["_pin_neck_fix"] = pin_neck
    st.session_state["_fixture_preset"] = fixture_preset

    with col_vis:
        st.subheader("Guitar fixtures")
        fixture_preset = st.selectbox("Orientation preset", FIXTURE_PRESETS, index=FIXTURE_PRESETS.index(fixture_preset))
        st.session_state["_fixture_preset"] = fixture_preset
        bc1, bc2 = st.columns(2)
        with bc1:
            clamp_ribs = st.checkbox("Clamp ribs (tag 4)", value=clamp_ribs)
        with bc2:
            pin_neck = st.checkbox("Pin neck (tag 5)", value=pin_neck)
        st.session_state["_clamp_ribs"] = clamp_ribs
        st.session_state["_pin_neck_fix"] = pin_neck

        new_fp = geometry_fingerprint(
            geom, top_wood, back_wood, clamp_ribs=clamp_ribs, pin_neck_fix=pin_neck, fixture_preset=fixture_preset
        )
        if new_fp != st.session_state.saved_geom_fp:
            invalidate_saved_state()
            st.session_state.live_preview_fp = ""
        geom_fp = new_fp

        st.subheader("PREVIEW")
        show_display = display_mesh_active(geom_fp)
        if show_display:
            dm = load_surface_mesh(DISPLAY_MESH_FILE)
            n = f" — {dm.n_cells:,} triangles" if dm is not None else ""
            if st.session_state.physics_ready:
                st.caption(f"Display mesh (`display_mesh.msh`, lc=4 mm){n} — ROM/FOM acoustics ready.")
            elif st.session_state.acoustics_pending:
                st.caption(f"Display mesh (`display_mesh.msh`){n} — computing acoustics…")
            else:
                st.caption(f"Display mesh (`display_mesh.msh`){n}.")
        else:
            st.caption("Sketch mode (`preview_mesh.msh`, lc≈30 mm) — Save Changes for display + acoustics.")

        mesh, is_sketch, mesh_src = get_view_mesh(geom_fp)
        try:
            pv.set_jupyter_backend("static")
            render_guitar(
                mesh,
                top_color=plot_color_for_wood(top_wood),
                back_color=plot_color_for_wood(back_wood),
                body_length=geom["length"],
                hole_radius=geom["hole_radius"],
                sketch_mode=is_sketch,
                plot_key=f"{'sketch' if is_sketch else 'display'}_{mesh_src}_{geom_fp[:10]}",
                fixture_preset=fixture_preset,
            )
        except Exception as exc:
            st.warning(f"Render error: {exc}")

        if (
            st.session_state.acoustics_pending
            and display_mesh_active(geom_fp)
        ):
            with st.spinner("Computing acoustics…"):
                try:
                    stk_path = run_acoustics(lhs_params, shape)
                    st.session_state.stk_body_json = str(stk_path)
                    st.session_state.physics_ready = True
                    st.session_state.acoustics_pending = False
                    st.rerun()
                except Exception as exc:
                    st.session_state.acoustics_pending = False
                    invalidate_saved_state()
                    st.error(f"Acoustics failed: {exc}")

        if WAV_OUTPUT.is_file() and st.session_state.physics_ready:
            st.audio(WAV_OUTPUT.read_bytes(), format="audio/wav")

    with row_btn[0]:
        if st.button("SAVE CHANGES", type="primary", use_container_width=True):
            try:
                on_save_changes(geom_fp)
            except Exception as exc:
                st.session_state.saved_geom_fp = ""
                invalidate_saved_state()
                st.error(f"Save failed: {exc}")

    with row_btn[1]:
        if st.button(
            "GENERATE SOUND",
            use_container_width=True,
            disabled=not st.session_state.physics_ready,
        ):
            try:
                with st.spinner("Synthesizing sound…"):
                    run_stk(body_json=Path(st.session_state.stk_body_json), top_wood=top_wood)
            except Exception as exc:
                st.error(f"Sound failed: {exc}")


main()
