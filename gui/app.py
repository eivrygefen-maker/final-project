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
PREVIEW_MESH_FILE = BASE_DIR / "FEM" / "mesh" / "preview_mesh.msh"
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
    ALL_WOOD_IDS,
    material_block_for_id,
    plot_color_for_wood,
    wood_display_name,
)

# Critical environment settings for Linux/VM
os.environ["QT_QPA_PLATFORM"] = "offscreen"
pv.OFF_SCREEN = True

st.set_page_config(page_title="3D Guitar Simulator", layout="wide", initial_sidebar_state="collapsed")

# --- Session state (two-button pipeline) ---
if "physics_ready" not in st.session_state:
    st.session_state.physics_ready = False
if "physics_geom_fp" not in st.session_state:
    st.session_state.physics_geom_fp = ""
if "developer_fom_mode" not in st.session_state:
    st.session_state.developer_fom_mode = False
if "last_engine" not in st.session_state:
    st.session_state.last_engine = ""
if "rom_last_result" not in st.session_state:
    st.session_state.rom_last_result = {}
if "live_preview_fp" not in st.session_state:
    st.session_state.live_preview_fp = ""
if "stk_body_json" not in st.session_state:
    st.session_state.stk_body_json = ""
if "show_physics_success" not in st.session_state:
    st.session_state.show_physics_success = False
# Bust stale preview meshes when preview CAD schema changes.
if st.session_state.get("preview_cad_schema", 0) < 5:
    st.session_state.preview_cad_schema = 5
    st.session_state.live_preview_fp = ""
    if PREVIEW_MESH_FILE.is_file():
        PREVIEW_MESH_FILE.unlink(missing_ok=True)

# Load saved geometry for Live Preview comparison
saved_geom = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, 'r') as f:
            saved_geom = json.load(f).get("geometry", {})
    except:
        pass

# 2D facet physical tags from build_3d_guitar.py (must match FEM geometry protocol).
TAG_TOP_PLATE = 1
TAG_SOUNDHOLE = 2
TAG_BACK_PLATE = 3
TAG_RIBS = 4
TAG_AIR = 10  # volume / cavity — never rendered in live preview
# Live preview: wood shell facets only (drops air box walls, soundhole tag, fix patches).
SHELL_VIS_TAGS = frozenset({TAG_TOP_PLATE, TAG_BACK_PLATE, TAG_RIBS})

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


def _run_physics_compute(
    *,
    fom_mode: bool,
    lhs_params: Dict[str, Any],
    rom_shape: str,
    geom_fp: str,
) -> Path:
    """Button 1: engineering mesh + ROM/FOM (no STK). Returns body JSON path for audio."""
    py_exe = sys.executable

    if FEM_FOM_JSON.exists():
        FEM_FOM_JSON.unlink()
    if ROM_STK_JSON.exists():
        ROM_STK_JSON.unlink()

    with st.spinner("Building high-fidelity engineering mesh…"):
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
        with st.spinner("ROM online solve (reduced basis)…"):
            rom_result = _execute_rom_engine(
                rom_shape=rom_shape,
                lhs_params=lhs_params,
                nev=0,
            )
            st.session_state.rom_last_result = rom_result
        st.session_state.last_engine = "rom"
        stk_json = ROM_STK_JSON

    st.session_state.physics_ready = True
    st.session_state.physics_geom_fp = geom_fp
    st.session_state.stk_body_json = str(stk_json)
    return stk_json


def _run_stk_audio(
    *,
    body_json: Path,
    top_wood_id: str,
    note_hz: float,
    q_mode: str,
) -> None:
    """Button 2: C++ STK synthesis only (requires prior physics compute)."""
    with st.spinner("Synthesizing audio (STK)…"):
        _run_stk_synthesis(
            body_json=body_json,
            top_wood_id=top_wood_id,
            note_hz=note_hz,
            q_mode=q_mode,
        )


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

def _geometry_state_dict(
    *,
    design_mode: str,
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
    exploded: bool,
) -> Dict[str, Any]:
    return {
        "design_mode": design_mode,
        "shape_type": shape_type,
        "length": float(length),
        "width": float(width),
        "depth": float(depth),
        "thickness": float(thickness),
        "hole_radius": float(hole_radius),
        "lower_bout": float(lower_bout),
        "upper_bout": float(upper_bout),
        "waist": float(waist),
        "soundhole_y": float(soundhole_y),
        "exploded_view": bool(exploded),
    }


def _geometry_fingerprint(geom: Dict[str, Any]) -> str:
    return json.dumps(geom, sort_keys=True, default=str)


def save_cfg_from_state(
    *,
    geom: Dict[str, Any],
    top_wood_id: str,
    back_wood_id: str,
    vis_mode: str = "Mesh + Solid",
) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    g = dict(geom)
    g["vis_mode"] = vis_mode
    g["top_wood_id"] = top_wood_id
    g["back_wood_id"] = back_wood_id
    data["geometry"] = g
    data["materials"] = {
        "top": {
            **material_block_for_id(top_wood_id),
            "name": wood_display_name(top_wood_id),
            "wood_id": top_wood_id,
        },
        "back": {
            **material_block_for_id(back_wood_id),
            "name": wood_display_name(back_wood_id),
            "wood_id": back_wood_id,
        },
        "air": {"density": 1.204, "speed_of_sound": 343.0},
    }
    solver = dict(data.get("solver") or {})
    solver["num_modes"] = int(solver.get("num_modes", 50))
    solver["mesh_file"] = str(MESH_FILE)
    data["solver"] = solver
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _load_guitar_surface_from_msh(msh_path: Path) -> Optional[Any]:
    """
    Load guitar shell triangles (facet tags 1=top, 3=back, 4=ribs).

    Preview meshes contain wood skin only (no air volume in CAD). Engineering meshes
    may include tag-10 volume tets but facet whitelist still selects the wood shell.
    """
    try:
        import meshio
    except ImportError:
        return None

    try:
        msh = meshio.read(str(msh_path))
        phys = msh.cell_data_dict.get("gmsh:physical")
        tri = msh.get_cells_type("triangle")
        if not phys or tri is None or len(tri) == 0:
            return None
        tri_tags = phys.get("triangle")
        if tri_tags is None:
            return None

        points = np.asarray(msh.points, dtype=np.float64)
        tags = np.asarray(tri_tags, dtype=np.int32).ravel()
        tri = np.asarray(tri, dtype=np.int64)

        # Strict whitelist: top, back, ribs only.
        keep = np.isin(tags, list(SHELL_VIS_TAGS))
        if not np.any(keep):
            return None
        tri = tri[keep]
        tags = tags[keep]

        faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri]).ravel()
        poly = pv.PolyData(points, faces)
        poly.cell_data["gmsh:physical"] = tags
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


def _build_live_preview_surface(
    *,
    geom: Dict[str, Any],
    top_wood_id: str,
    back_wood_id: str,
    vis_mode: str,
) -> Optional[Any]:
    """
    Fast Gmsh preview (--preview). Runs only when the geometry fingerprint changes.

    Called on every Streamlit rerun; skipped when sliders are unchanged.
    """
    fp = _geometry_fingerprint(geom)
    if st.session_state.live_preview_fp == fp and PREVIEW_MESH_FILE.is_file():
        cached = _load_guitar_surface_from_msh(PREVIEW_MESH_FILE)
        if cached is not None:
            return cached

    save_cfg_from_state(
        geom=geom,
        top_wood_id=top_wood_id,
        back_wood_id=back_wood_id,
        vis_mode=vis_mode,
    )
    if PREVIEW_MESH_FILE.exists():
        PREVIEW_MESH_FILE.unlink()

    py_exe = sys.executable
    preview_env = {**os.environ, "FEM_ALLOW_PREVIEW": "1"}
    result = subprocess.run(
        [py_exe, str(GEOMETRY_SCRIPT), "-nopopup"],
        capture_output=True,
        text=True,
        env=preview_env,
    )
    if not PREVIEW_MESH_FILE.is_file():
        st.session_state.live_preview_fp = ""
        err_tail = (result.stderr or "").strip() or (result.stdout or "").strip() or "unknown error"
        st.error(f"Gmsh preview failed.\n{err_tail}")
        return None

    st.session_state.live_preview_fp = fp
    st.session_state.physics_ready = False
    return _load_guitar_surface_from_msh(PREVIEW_MESH_FILE)


def _render_pyvista_guitar(
    surface_mesh,
    *,
    top_color: str,
    back_color: str,
    show_edges: bool,
    cam_preset: str,
    plot_key: str,
    sketch_mode: bool = False,
) -> None:
    plotter = pv.Plotter(window_size=[1100, 620])
    plotter.background_color = "#f4f4f9"

    def _add_part(mesh, mask, color: str, *, opacity: float = 1.0, edges: bool = False) -> bool:
        if not np.any(mask):
            return False
        part = mesh.extract_cells(mask)
        if part.n_cells <= 0:
            return False
        if sketch_mode:
            plotter.add_mesh(
                part,
                style="wireframe",
                color=color,
                line_width=2.0,
                opacity=1.0,
                lighting=False,
            )
        else:
            plotter.add_mesh(
                part,
                color=color,
                opacity=opacity,
                show_edges=edges,
                edge_color="#2b1a10",
                smooth_shading=True,
                lighting=True,
            )
        return True

    def render_mesh_by_protocol(mesh, show_edges_flag: bool, top_c: str, back_c: str) -> bool:
        tags = mesh.cell_data.get("gmsh:physical")
        if tags is None:
            return False
        tags = np.asarray(tags).ravel()
        # Wood shell only (loader keeps tags 1, 3, 4). Opening is the top-plate boolean cut.
        is_top = tags == TAG_TOP_PLATE
        is_back_wood = (tags == TAG_BACK_PLATE) | (tags == TAG_RIBS)
        if not (is_top.any() or is_back_wood.any()):
            return False
        rendered = False
        rendered |= _add_part(mesh, is_top, top_c, edges=show_edges_flag)
        rendered |= _add_part(mesh, is_back_wood, back_c, edges=show_edges_flag)
        return rendered

    if surface_mesh is not None:
        for key in list(surface_mesh.cell_data.keys()):
            if key != "gmsh:physical":
                try:
                    del surface_mesh.cell_data[key]
                except Exception:
                    pass
        if not render_mesh_by_protocol(surface_mesh, show_edges, top_color, back_color):
            plotter.add_mesh(
                surface_mesh,
                color=back_color,
                show_edges=show_edges,
                smooth_shading=True,
                scalar_bar_args=None,
            )
    else:
        plotter.add_text("Preview unavailable", position="upper_left", font_size=12)

    plotter.enable_anti_aliasing("ssaa")
    if cam_preset == "Standing Angled (3D)":
        plotter.camera_position = [(0.0, -0.4, 1.1), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    elif cam_preset == "Standing Upright (Front)":
        plotter.camera_position = [(0.0, 0.0, 1.2), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    elif cam_preset == "Laying Flat (Top View)":
        plotter.camera_position = [(0.0, 0.0, 1.2), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    elif cam_preset == "Laying on Side (Profile)":
        plotter.camera_position = [(0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]

    stpyvista(plotter, key=plot_key)
    plotter.close()


def _invalidate_physics_state() -> None:
    st.session_state.physics_ready = False


def _on_button_save(
    *,
    geom_state: Dict[str, Any],
    geom_fp: str,
    top_wood_id: str,
    back_wood_id: str,
    vis_mode: str,
    fom_mode: bool,
    lhs_params: Dict[str, Any],
) -> None:
    """Button 1: lock design, build engineering mesh, run ROM/FOM."""
    save_cfg_from_state(
        geom=geom_state,
        top_wood_id=top_wood_id,
        back_wood_id=back_wood_id,
        vis_mode=vis_mode,
    )
    _run_physics_compute(
        fom_mode=fom_mode,
        lhs_params=lhs_params,
        rom_shape=DEFAULT_ROM_SHAPE,
        geom_fp=geom_fp,
    )
    st.session_state.show_physics_success = True


def _on_button_audio(
    *,
    top_wood_id: str,
    note_hz: float,
    q_mode: str,
) -> None:
    """Button 2: STK synthesis (requires physics_ready)."""
    stk_path = Path(st.session_state.stk_body_json)
    if not stk_path.is_file():
        raise FileNotFoundError("Physics body JSON missing — run Save & Compute Physics first.")
    _run_stk_audio(
        body_json=stk_path,
        top_wood_id=top_wood_id,
        note_hz=note_hz,
        q_mode=q_mode,
    )


def _resolve_display_mesh(
    *,
    geom_state: Dict[str, Any],
    geom_fp: str,
    top_wood_id: str,
    back_wood_id: str,
    vis_mode: str,
    physics_ready: bool,
) -> Tuple[Optional[Any], bool, str]:
    """Return (mesh, sketch_mode, plot_key_suffix)."""
    if physics_ready and MESH_FILE.is_file():
        mesh = _load_guitar_surface_from_msh(MESH_FILE)
        return mesh, False, f"eng_{st.session_state.physics_geom_fp[:12]}"

    if st.session_state.live_preview_fp != geom_fp:
        with st.spinner("Sketch mesh…"):
            mesh = _build_live_preview_surface(
                geom=geom_state,
                top_wood_id=top_wood_id,
                back_wood_id=back_wood_id,
                vis_mode=vis_mode,
            )
    else:
        mesh = _build_live_preview_surface(
            geom=geom_state,
            top_wood_id=top_wood_id,
            back_wood_id=back_wood_id,
            vis_mode=vis_mode,
        )
    return mesh, True, geom_fp[:12]


st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlockBorderWrapper"]) {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Guitar Simulator")
st.caption("Move sliders for a live wireframe sketch. Save physics, then generate audio.")

def _shape_limits(shape: str) -> Tuple[float, float, float, float, float, float, float, float, float]:
    if shape == "Classical":
        return 0.35, 0.60, 0.48, 0.20, 0.45, 0.37, 0.08, 0.15, 0.10
    if shape == "Dreadnought":
        return 0.45, 0.70, 0.51, 0.30, 0.55, 0.40, 0.10, 0.20, 0.12
    return 0.10, 1.00, 0.40, 0.10, 0.80, 0.30, 0.01, 0.50, 0.10


def _wood_option_label(wood_id: str) -> str:
    return f"{wood_id} — {wood_display_name(wood_id)}"

def _wood_index(wood_id: str, options: List[str]) -> int:
    key = str(wood_id).strip().lower().replace(" ", "_")
    return options.index(key) if key in options else 0


_saved_top = str(saved_geom.get("top_wood_id") or saved_geom.get("wood_id") or "spruce")
_saved_back = str(saved_geom.get("back_wood_id") or saved_geom.get("body_wood_id") or "rosewood")

col_controls, col_visual = st.columns([0.22, 0.78], gap="large")

with col_controls:
    st.subheader("Design controls")

    shape_type = st.selectbox("Guitar shape", ["Classical", "Dreadnought", "Box"])
    L_min, L_max, L_def, W_min, W_max, W_def, D_min, D_max, D_def = _shape_limits(shape_type)

    top_wood_id = st.selectbox(
        "Soundboard wood",
        ALL_WOOD_IDS,
        index=_wood_index(_saved_top, ALL_WOOD_IDS),
        format_func=_wood_option_label,
    )
    back_wood_id = st.selectbox(
        "Back & sides wood",
        ALL_WOOD_IDS,
        index=_wood_index(_saved_back, ALL_WOOD_IDS),
        format_func=_wood_option_label,
    )

    design_mode = st.radio("Design mode", ["Basic (Geometric)", "Professional (Luthier)"])
    if design_mode == "Basic (Geometric)":
        L = st.slider("Length (m)", L_min, L_max, L_def, key=f"L_{shape_type}")
        W = st.slider("Width (m)", W_min, W_max, W_def, key=f"W_{shape_type}")
        D = st.slider("Depth (m)", D_min, D_max, D_def, key=f"D_{shape_type}")
        lower_bout, upper_bout, waist, soundhole_y = W, W * 0.75, W * 0.65, L * 0.4
    else:
        L = st.slider("Total body length (m)", L_min, L_max, L_def, key=f"L_prof_{shape_type}")
        D = st.slider("Depth (m)", D_min, D_max, D_def, key=f"D_prof_{shape_type}")
        lower_bout = st.slider("Lower bout (m)", 0.20, 0.60, W_def, key="lb")
        upper_bout = st.slider("Upper bout (m)", 0.15, 0.50, W_def * 0.75, key="ub")
        waist = st.slider("Waist (m)", 0.10, 0.40, W_def * 0.65, key="wst")
        soundhole_y = st.slider("Soundhole Y (m)", 0.10, 0.60, L_def * 0.4, key="sh_y")
        W = lower_bout

    _hr_default = float(saved_geom.get("hole_radius", 0.04))
    _hr_lo, _hr_hi = 0.020, max(0.055, min(0.18, float(W) * 0.55))
    hr = st.slider(
        "Soundhole radius (m)",
        min_value=_hr_lo,
        max_value=_hr_hi,
        value=min(max(_hr_default, _hr_lo), _hr_hi),
        step=0.0005,
        format="%.4f",
    )
    thick = st.slider(
        "Top plate thickness (m)",
        min_value=0.0030,
        max_value=0.0060,
        value=0.0030,
        step=0.0005,
        format="%.4f",
    )
    exploded = st.checkbox("Exploded view", value=False)

    st.markdown("**Sound (for STK)**")
    pitch_mode = st.radio("Pitch input", ["Musical notes", "Manual Hz"], horizontal=True)
    if pitch_mode == "Musical notes":
        note_hz = NOTES_DICT[st.selectbox("Note", list(NOTES_DICT.keys()), index=1)]
    else:
        note_hz = st.number_input("Frequency (Hz)", value=110.0)
    q_mode = st.radio("Q estimation", ["Mean (Stable)", "Random (Realistic)"], horizontal=True)

    with st.expander("Advanced diagnostics", expanded=False):
        st.session_state.developer_fom_mode = st.checkbox(
            "FOM developer mode",
            value=bool(st.session_state.developer_fom_mode),
            help="Full-order FEM instead of ROM.",
        )

# --- Derived design state (rerun on every widget change) ---
top_plot_color = plot_color_for_wood(top_wood_id)
back_plot_color = plot_color_for_wood(back_wood_id)
geom_state = _geometry_state_dict(
    design_mode=design_mode,
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
    exploded=exploded,
)
geom_fp = _geometry_fingerprint(
    {**geom_state, "top_wood_id": top_wood_id, "back_wood_id": back_wood_id}
)
if geom_fp != st.session_state.physics_geom_fp:
    _invalidate_physics_state()

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

with col_visual:
    st.subheader("3D preview")
    physics_ready = bool(st.session_state.get("physics_ready", False))
    fom_mode = bool(st.session_state.developer_fom_mode)
    vis_mode = st.selectbox("Visual style", ["Mesh + Solid", "Solid Wood"], key="vis_mode_ui")
    cam_preset = st.selectbox(
        "Camera",
        [
            "Standing Angled (3D)",
            "Standing Upright (Front)",
            "Laying Flat (Top View)",
            "Laying on Side (Profile)",
        ],
    )

    btn_save, btn_audio = st.columns(2)
    with btn_save:
        if st.button(
            "Save Changes & Compute Physics",
            use_container_width=True,
            type="primary",
        ):
            try:
                _on_button_save(
                    geom_state=geom_state,
                    geom_fp=geom_fp,
                    top_wood_id=top_wood_id,
                    back_wood_id=back_wood_id,
                    vis_mode=vis_mode,
                    fom_mode=fom_mode,
                    lhs_params=lhs_params,
                )
                st.rerun()
            except Exception as exc:
                _invalidate_physics_state()
                st.error(f"Physics failed: {exc}")

    with btn_audio:
        if st.button(
            "Generate Audio (STK)",
            use_container_width=True,
            disabled=not physics_ready,
            help="Enabled after physics compute finishes.",
        ):
            try:
                _on_button_audio(
                    top_wood_id=top_wood_id,
                    note_hz=float(note_hz),
                    q_mode=q_mode,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Audio failed: {exc}")

    if physics_ready and MESH_FILE.is_file():
        st.success("Solid engineering model — audio unlocked")
    else:
        st.info("Wireframe sketch — move a slider to refresh")

    display_mesh, sketch_mode, plot_suffix = _resolve_display_mesh(
        geom_state=geom_state,
        geom_fp=geom_fp,
        top_wood_id=top_wood_id,
        back_wood_id=back_wood_id,
        vis_mode=vis_mode,
        physics_ready=physics_ready,
    )
    try:
        pv.set_jupyter_backend("static")
        _render_pyvista_guitar(
            display_mesh,
            top_color=top_plot_color,
            back_color=back_plot_color,
            show_edges=("Mesh" in vis_mode) or sketch_mode,
            cam_preset=cam_preset,
            plot_key=f"view_{plot_suffix}",
            sketch_mode=sketch_mode,
        )
    except Exception as exc:
        st.warning(f"3D render issue: {exc}")

    st.caption(
        f"Soundboard {top_wood_id} ({top_plot_color}) · "
        f"back/sides {back_wood_id} ({back_plot_color})"
    )

    if st.session_state.show_physics_success:
        eng = str(st.session_state.get("last_engine", "")).upper() or "ROM"
        st.success(f"Physics saved ({eng}). You can generate audio.")
        st.session_state.show_physics_success = False

    if WAV_OUTPUT.is_file():
        st.audio(str(WAV_OUTPUT))

    if st.session_state.get("last_engine") == "rom" and st.session_state.get("rom_last_result"):
        with st.expander("ROM solve details"):
            st.json(st.session_state.rom_last_result)

