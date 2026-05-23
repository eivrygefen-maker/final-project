"""
3D Guitar Simulator — ROM-aligned Streamlit UI.

Sliders match the 7-parameter LHS basis (L, W, D, top thickness, hole radius, woods).
PyVista shows ``display_mesh.msh`` from ``build_3d_guitar.py`` (same config as FEM/ROM).
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
ROM_HOLE_RADIUS_BOUNDS = (0.035, 0.055)

FAST_PREVIEW_HEIGHT = 720
STUDIO_HANDSHAKE_ACTIONS = frozenset({"ready", "_handshake", "handshake", "_ready_ping"})
STUDIO_ROM_PAYLOAD_KEYS = (
    "shape_type",
    "length",
    "width",
    "depth",
    "top_thickness",
    "hole_radius",
    "top_wood_id",
    "back_wood_id",
    "gui_mode",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE_DIR / "FEM" / "geometry"))
from generate_reference_models import (  # noqa: E402
    ACOUSTIC_LOOP,
    CLASSICAL_LOOP,
    NOMINAL_LENGTH_ACOUSTIC,
    NOMINAL_LENGTH_CLASSICAL,
    get_luthier_gui_defaults,
)
from components.fast_preview import fast_preview  # noqa: E402
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
    TOP_THICKNESS_MAX_M,
    TOP_THICKNESS_MIN_M,
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
        "_rom_mesh_fp": "",
        "_pending_fom_run": False,
        "_fast_preview_geom": None,
        "_studio_event_id": "",
        "_studio_templates_sent": False,
        "_studio_component_ready": False,
        "_studio_handshake_id": "",
        "_studio_iframe_payload_fp": "",
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


def rom_lwd_bounds(shape_type: str) -> Dict[str, Tuple[float, float]]:
    """Match ``ROMManager._shape_length_width_depth_bounds`` (LHS training box)."""
    stl = str(shape_type).strip().lower()
    if "dreadnought" in stl or "acoustic" in stl:
        return {"length": (0.45, 0.70), "width": (0.30, 0.55), "depth": (0.10, 0.20)}
    if "box" in stl:
        return {"length": (0.10, 1.00), "width": (0.10, 0.80), "depth": (0.01, 0.50)}
    return {"length": (0.35, 0.60), "width": (0.20, 0.45), "depth": (0.08, 0.15)}


def rom_defaults(shape_type: str) -> Dict[str, float]:
    """Mid-range ROM defaults for a shape (slider starting points)."""
    b = rom_lwd_bounds(shape_type)
    defs = get_luthier_gui_defaults(shape_type)
    return {
        "length": 0.5 * (b["length"][0] + b["length"][1]),
        "width": 0.5 * (b["width"][0] + b["width"][1]),
        "depth": 0.5 * (b["depth"][0] + b["depth"][1]),
        "top_thickness": 0.5 * (TOP_THICKNESS_MIN_M + TOP_THICKNESS_MAX_M),
        "hole_radius": 0.5 * (ROM_HOLE_RADIUS_BOUNDS[0] + ROM_HOLE_RADIUS_BOUNDS[1]),
    }


def build_geometry_state(
    *,
    shape_type: str,
    length: float,
    width: float,
    depth: float,
    top_thickness: float,
    hole_radius: float,
) -> Dict[str, Any]:
    """ROM-facing geometry; bout widths derived from W for STEP morphing only."""
    defs = get_luthier_gui_defaults(shape_type)
    w = float(width)
    w_scale = w / float(defs["width"]) if float(defs["width"]) > 0.0 else 1.0
    l_val = float(length)
    return {
        "shape_type": shape_type,
        "length": l_val,
        "width": w,
        "depth": float(depth),
        "thickness": float(top_thickness),
        "top_thickness": float(top_thickness),
        "hole_radius": min(float(hole_radius), HOLE_RADIUS_MAX_M),
        "lower_bout": w,
        "upper_bout": float(defs["upper_bout"]) * w_scale,
        "waist": float(defs["waist"]) * w_scale,
        "soundhole_y": 0.0,
        "soundhole_x": soundhole_center_x(l_val),
        "soundhole_from_neck_ratio": SOUNDHOLE_FROM_NECK_RATIO,
        "bridge_x": float(defs["bridge_x"]),
    }


def _loop_to_xy(loop: Sequence[Tuple[float, float]]) -> List[List[float]]:
    pts = [(float(x), float(y)) for x, y in loop]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return [[x, y] for x, y in pts]


# Keys sent to / returned from the Design Studio (7D ROM + presentation metadata).
STUDIO_ROM_KEYS = (
    "shape_type",
    "length",
    "width",
    "depth",
    "top_thickness",
    "hole_radius",
    "top_wood_id",
    "back_wood_id",
)
STUDIO_META_KEYS = (
    "gui_mode",
    "bounds",
    "bounds_by_shape",
    "top_thickness_bounds",
    "hole_radius_bounds",
    "wood_ids",
    "wood_colors",
    "templates",
    "soundhole_from_neck_ratio",
)


def sanitize_studio_payload(data: Optional[Dict[str, Any]], shape_type: str = "Classical") -> Dict[str, Any]:
    """Strip legacy bout/waist keys; keep only ROM 7D + component metadata."""
    src = dict(data or {})
    shape = str(src.get("shape_type", shape_type)).strip()
    if shape not in SHAPE_OPTIONS:
        shape = shape_type if shape_type in SHAPE_OPTIONS else "Classical"
    rom_def = rom_defaults(shape)
    width = src.get("width")
    if width is None:
        width = src.get("lower_bout_width") or src.get("lower_bout")
    if width is None:
        width = rom_def["width"]
    top_t = src.get("top_thickness", src.get("thickness", rom_def["top_thickness"]))
    hole = src.get("hole_radius", src.get("soundhole_radius", rom_def["hole_radius"]))
    top_wood = str(src.get("top_wood_id", "spruce")).lower()
    back_wood = str(src.get("back_wood_id", "rosewood")).lower()
    if top_wood not in ALL_WOOD_IDS:
        top_wood = "spruce"
    if back_wood not in ALL_WOOD_IDS:
        back_wood = "rosewood"
    bounds = rom_lwd_bounds(shape)
    out: Dict[str, Any] = {
        "shape_type": shape,
        "length": float(src.get("length", rom_def["length"])),
        "width": float(width),
        "depth": float(src.get("depth", rom_def["depth"])),
        "top_thickness": float(top_t),
        "hole_radius": float(hole),
        "top_wood_id": top_wood,
        "back_wood_id": back_wood,
        "gui_mode": str(src.get("gui_mode", "user")),
        "bounds": src.get("bounds") or {
            "length": list(bounds["length"]),
            "width": list(bounds["width"]),
            "depth": list(bounds["depth"]),
        },
        "bounds_by_shape": src.get("bounds_by_shape")
        or {s: {k: list(v) for k, v in rom_lwd_bounds(s).items()} for s in SHAPE_OPTIONS},
        "top_thickness_bounds": list(src.get("top_thickness_bounds") or [TOP_THICKNESS_MIN_M, TOP_THICKNESS_MAX_M]),
        "hole_radius_bounds": list(src.get("hole_radius_bounds") or list(ROM_HOLE_RADIUS_BOUNDS)),
        "wood_ids": list(src.get("wood_ids") or ALL_WOOD_IDS),
        "wood_colors": dict(src.get("wood_colors") or {w: plot_color_for_wood(w) for w in ALL_WOOD_IDS}),
        "templates": dict(src.get("templates") or export_luthier_templates()),
        "soundhole_from_neck_ratio": float(src.get("soundhole_from_neck_ratio", SOUNDHOLE_FROM_NECK_RATIO)),
    }
    return out


def export_luthier_templates() -> Dict[str, Dict[str, Any]]:
    """Closed-loop control points for the Design Studio (matches Python STEP blueprints)."""
    cdef = get_luthier_gui_defaults("Classical")
    adef = get_luthier_gui_defaults("Dreadnought")
    return {
        "Classical": {
            "loop": _loop_to_xy(CLASSICAL_LOOP),
            "nominal_length": float(NOMINAL_LENGTH_CLASSICAL),
            "nominal_width": float(cdef["width"]),
        },
        "Dreadnought": {
            "loop": _loop_to_xy(ACOUSTIC_LOOP),
            "nominal_length": float(NOMINAL_LENGTH_ACOUSTIC),
            "nominal_width": float(adef["width"]),
        },
    }


def studio_initial_from_saved(
    saved_geom: Dict[str, Any],
    shape_type: str,
    *,
    developer_fom_mode: bool,
) -> Dict[str, Any]:
    """Payload for the Three.js Design Studio component (7D ROM + metadata only)."""
    cfg = _load_saved_config()
    mats = cfg.get("materials") or {}
    merged = dict(saved_geom)
    merged.setdefault("top_wood_id", (mats.get("top") or {}).get("wood_id", "spruce"))
    merged.setdefault("back_wood_id", (mats.get("back") or {}).get("wood_id", "rosewood"))
    payload = sanitize_studio_payload(merged, shape_type)
    payload["gui_mode"] = "admin" if developer_fom_mode else "user"
    return payload


def studio_payload_for_component(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Omit heavy template geometry after first iframe load (cached client-side)."""
    out = dict(payload)
    if st.session_state.get("_studio_templates_sent"):
        out.pop("templates", None)
    else:
        st.session_state._studio_templates_sent = True
    return out


def studio_iframe_payload_fp(payload: Dict[str, Any]) -> str:
    """Fingerprint of props that should trigger a full iframe re-render."""
    subset = {k: payload.get(k) for k in STUDIO_ROM_PAYLOAD_KEYS}
    subset["has_templates"] = "templates" in payload
    return json.dumps(subset, sort_keys=True, default=str)


def studio_payload_for_iframe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stable component ``initial`` dict — avoids remount churn when only PyVista/session
    state changes. After the first successful mount, omit templates and skip redundant
    metadata when the ROM fingerprint is unchanged.
    """
    out = studio_payload_for_component(payload)
    fp = studio_iframe_payload_fp(out)
    last_fp = str(st.session_state.get("_studio_iframe_payload_fp", ""))
    ready = bool(st.session_state.get("_studio_component_ready"))
    if ready and last_fp and fp == last_fp:
        return {k: out[k] for k in STUDIO_ROM_PAYLOAD_KEYS if k in out}
    st.session_state._studio_iframe_payload_fp = fp
    return out


def is_studio_handshake_event(event: Dict[str, Any]) -> bool:
    """True for mount/ping payloads from the iframe (no geometry side effects)."""
    action = str(event.get("action") or "").strip().lower()
    if action in STUDIO_HANDSHAKE_ACTIONS:
        return True
    evt_type = str(event.get("type") or "").strip().lower()
    return evt_type in {"ready", "handshake", "component_ready"}


def record_studio_handshake(event: Dict[str, Any]) -> None:
    """Mark the Design Studio iframe as mounted; dedupe repeated ready pings."""
    token = event.get("_ts")
    if token is None:
        token = event.get("_handshake_ts")
    eid = f"handshake:{token if token is not None else 'mount'}"
    if eid == st.session_state.get("_studio_handshake_id"):
        return
    st.session_state._studio_handshake_id = eid
    st.session_state._studio_component_ready = True


def render_fast_preview_studio(initial: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Mount the Design Studio iframe with fixed height and a safe fallback."""
    try:
        with st.container():
            out = fast_preview(
                initial=initial,
                key="fast_preview_geom",
                height=FAST_PREVIEW_HEIGHT,
            )
            st.session_state._studio_component_ready = True
            return out
    except FileNotFoundError as exc:
        st.error(f"Design Studio assets missing: {exc}")
    except Exception as exc:
        st.warning(f"Design Studio failed to mount: {exc}")
    return None


def studio_event_id(event: Dict[str, Any]) -> str:
    """Unique id for a component button event (dedupes stale return values across reruns)."""
    clean = sanitize_studio_payload(event)
    return json.dumps(
        {
            "action": str(event.get("action", "")).strip().lower(),
            "_ts": event.get("_ts"),
            "shape_type": clean.get("shape_type"),
            "length": clean.get("length"),
            "width": clean.get("width"),
            "depth": clean.get("depth"),
            "top_thickness": clean.get("top_thickness"),
            "hole_radius": clean.get("hole_radius"),
            "top_wood_id": clean.get("top_wood_id"),
            "back_wood_id": clean.get("back_wood_id"),
        },
        sort_keys=True,
        default=str,
    )


def geom_from_studio_event(event: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    """Parse component event into geometry dict and wood IDs (7D ROM only)."""
    clean = sanitize_studio_payload(event)
    shape = str(clean["shape_type"])
    top_wood = str(clean["top_wood_id"])
    back_wood = str(clean["back_wood_id"])
    geom = build_geometry_state(
        shape_type=shape,
        length=float(clean["length"]),
        width=float(clean["width"]),
        depth=float(clean["depth"]),
        top_thickness=float(clean["top_thickness"]),
        hole_radius=float(clean["hole_radius"]),
    )
    return geom, top_wood, back_wood


def process_fast_preview_event(
    event: Optional[Dict[str, Any]],
    *,
    clamp_ribs: bool,
    pin_neck: bool,
    fixture_preset: str,
) -> None:
    """Handle Save & Sync / Run ROM / Run FEM from the Design Studio."""
    if not event or not isinstance(event, dict):
        return

    if is_studio_handshake_event(event):
        record_studio_handshake(event)
        return

    action = str(event.get("action") or "").strip().lower()
    if not action:
        return
    if event.get("_ts") is None:
        return

    eid = studio_event_id(event)
    if eid == st.session_state.get("_studio_event_id"):
        return
    st.session_state._studio_event_id = eid

    clean = sanitize_studio_payload(event)
    geom, top_wood, back_wood = geom_from_studio_event(clean)
    st.session_state._fast_preview_geom = clean
    st.session_state._geom = geom
    st.session_state._top_wood = top_wood
    st.session_state._back_wood = back_wood

    geom_fp = geometry_fingerprint(
        geom,
        top_wood,
        back_wood,
        clamp_ribs=clamp_ribs,
        pin_neck_fix=pin_neck,
        fixture_preset=fixture_preset,
    )

    if action in ("save_sync", "run_rom", "run_fem"):
        regenerate_display_mesh(
            geom,
            top_wood=top_wood,
            back_wood=back_wood,
            clamp_ribs=clamp_ribs,
            pin_neck_fix=pin_neck,
            fixture_preset=fixture_preset,
            geom_fp=geom_fp,
        )
        invalidate_physics_state()

    if action == "run_rom":
        st.session_state.developer_fom_mode = False
        st.session_state.acoustics_pending = True
    elif action == "run_fem":
        st.session_state.developer_fom_mode = True
        st.session_state._pending_fom_run = True

    # Do not call st.rerun() here — setComponentValue already triggers one rerun.


def rom_mesh_fingerprint(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
) -> str:
    """Fingerprint of the 7 ROM/LHS parameters (triggers mesh rebuild when changed)."""
    return json.dumps(
        {
            "shape_type": geom["shape_type"],
            "length": geom["length"],
            "width": geom["width"],
            "depth": geom["depth"],
            "top_thickness": geom["top_thickness"],
            "hole_radius": geom["hole_radius"],
            "top_wood_id": top_wood,
            "back_wood_id": back_wood,
        },
        sort_keys=True,
    )


def regenerate_display_mesh(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
    clamp_ribs: bool,
    pin_neck_fix: bool,
    fixture_preset: str,
    geom_fp: str,
) -> None:
    """Write ``guitar_3d.json``, run ``build_3d_guitar.py``, refresh PyVista mesh."""
    save_config(
        geom,
        top_wood=top_wood,
        back_wood=back_wood,
        clamp_ribs=clamp_ribs,
        pin_neck_fix=pin_neck_fix,
        fixture_preset=fixture_preset,
    )
    run_gmsh_display()
    st.session_state._rom_mesh_fp = rom_mesh_fingerprint(geom, top_wood=top_wood, back_wood=back_wood)
    st.session_state.saved_geom_fp = geom_fp
    st.session_state.show_display_mesh = True
    st.session_state.live_preview_fp = ""


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
    """Seven-parameter dict consumed by ``ROMManager.solve_online`` / LHS pool."""
    return {
        "geometry.shape_type": geom["shape_type"],
        "geometry.length": geom["length"],
        "geometry.width": geom["width"],
        "geometry.depth": geom["depth"],
        "geometry.top_thickness": geom["top_thickness"],
        "geometry.hole_radius": geom["hole_radius"],
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
    """PyVista shows ``display_mesh.msh`` from ``build_3d_guitar.py`` (never FOM volume mesh)."""
    if display_mesh_active(geom_fp):
        mesh = load_surface_mesh(DISPLAY_MESH_FILE)
        if mesh is not None:
            return mesh, False, "display_mesh.msh"
    mesh = load_surface_mesh(DISPLAY_MESH_FILE)
    if mesh is not None:
        return mesh, False, "display_mesh.msh"
    return None, False, ""


def invalidate_physics_state() -> None:
    """Clear ROM/FEM acoustics results when geometry or materials change."""
    st.session_state.physics_ready = False
    st.session_state.acoustics_pending = False
    st.session_state.stk_body_json = ""


def invalidate_saved_state() -> None:
    invalidate_physics_state()


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


def _render_interactive_panel(
    saved: Dict[str, Any],
    saved_solver: Dict[str, Any],
    saved_shape: str,
    saved_fixture: str,
) -> None:
    """Controls + Design Studio + PyVista (isolated via ``st.fragment`` when available)."""
    row_btn = st.columns(2)
    col_ctrl, col_vis = st.columns([0.18, 0.82], gap="large")

    clamp_ribs = bool(st.session_state.get("_clamp_ribs", saved_solver.get("clamp_ribs", False)))
    pin_neck = bool(st.session_state.get("_pin_neck_fix", saved_solver.get("eps_pin_fix_tag5", True)))
    fixture_preset = str(st.session_state.get("_fixture_preset", saved_fixture))

    with col_ctrl:
        st.subheader("Session")
        mode_ix = 1 if st.session_state.developer_fom_mode else 0
        mode = st.radio(
            "Mode",
            ("User (ROM)", "Admin (FEM)"),
            index=mode_ix,
            horizontal=False,
            help="Mirrors Design Studio actions; Admin enables Run Full FEM in the studio.",
        )
        st.session_state.developer_fom_mode = mode.startswith("Admin")

        st.subheader("Fixtures")
        fixture_preset = st.selectbox(
            "Orientation", FIXTURE_PRESETS, index=FIXTURE_PRESETS.index(fixture_preset)
        )
        st.session_state["_fixture_preset"] = fixture_preset
        clamp_ribs = st.checkbox("Clamp ribs (tag 4)", value=clamp_ribs)
        pin_neck = st.checkbox("Pin neck (tag 5)", value=pin_neck)
        st.session_state["_clamp_ribs"] = clamp_ribs
        st.session_state["_pin_neck_fix"] = pin_neck

    fp_seed = studio_initial_from_saved(
        saved, saved_shape, developer_fom_mode=st.session_state.developer_fom_mode
    )
    fp_raw = st.session_state.get("_fast_preview_geom")
    if isinstance(fp_raw, dict) and fp_raw:
        fp_seed = sanitize_studio_payload({**fp_seed, **fp_raw}, saved_shape)
    fp_seed["gui_mode"] = "admin" if st.session_state.developer_fom_mode else "user"
    fp_for_component = studio_payload_for_iframe(fp_seed)

    with col_vis:
        st.subheader("Design Studio")
        studio_event = render_fast_preview_studio(fp_for_component)
        process_fast_preview_event(
            studio_event,
            clamp_ribs=clamp_ribs,
            pin_neck=pin_neck,
            fixture_preset=fixture_preset,
        )

        fp_live = sanitize_studio_payload(
            st.session_state.get("_fast_preview_geom") or fp_seed,
            saved_shape,
        )
        shape = str(fp_live.get("shape_type", saved_shape))
        if shape not in SHAPE_OPTIONS:
            shape = saved_shape
        geom, top_wood, back_wood = geom_from_studio_event(fp_live)
        lhs_params = lhs_params_from_ui(geom, top_wood=top_wood, back_wood=back_wood)
        st.session_state["_geom"] = geom
        st.session_state["_top_wood"] = top_wood
        st.session_state["_back_wood"] = back_wood

        geom_fp = geometry_fingerprint(
            geom,
            top_wood,
            back_wood,
            clamp_ribs=clamp_ribs,
            pin_neck_fix=pin_neck,
            fixture_preset=fixture_preset,
        )

        if st.session_state.get("_pending_fom_run") and display_mesh_active(geom_fp):
            st.session_state._pending_fom_run = False
            with st.spinner("Running full FEM…"):
                try:
                    stk_path = run_fom_acoustics()
                    st.session_state.stk_body_json = str(stk_path)
                    st.session_state.physics_ready = True
                    st.success("Full FEM complete.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Full FEM failed: {exc}")

        st.subheader("High-fidelity 3D (PyVista)")
        mesh, is_sketch, mesh_src = get_view_mesh(geom_fp)
        if mesh is None and not DISPLAY_MESH_FILE.is_file():
            st.info("Click **Save & Sync** in the Design Studio to build `display_mesh.msh` via Gmsh.")
        else:
            dm = load_surface_mesh(DISPLAY_MESH_FILE)
            n_cells = f"{dm.n_cells:,} triangles" if dm is not None else "—"
            st.caption(f"`display_mesh.msh` (lc=4 mm) · {n_cells} · same config as ROM/FEM")
            try:
                pv.set_jupyter_backend("static")
                render_guitar(
                    mesh,
                    top_color=plot_color_for_wood(top_wood),
                    back_color=plot_color_for_wood(back_wood),
                    body_length=geom["length"],
                    hole_radius=geom["hole_radius"],
                    sketch_mode=False,
                    plot_key=f"display_{mesh_src}_{geom_fp[:12]}",
                    fixture_preset=fixture_preset,
                )
            except Exception as exc:
                st.warning(f"Render error: {exc}")

        if st.session_state.acoustics_pending and display_mesh_active(geom_fp):
            with st.spinner("Computing acoustics…"):
                try:
                    stk_path = run_acoustics(lhs_params, shape)
                    st.session_state.stk_body_json = str(stk_path)
                    st.session_state.physics_ready = True
                    st.session_state.acoustics_pending = False
                    st.rerun()
                except Exception as exc:
                    st.session_state.acoustics_pending = False
                    st.error(f"Acoustics failed: {exc}")

        if WAV_OUTPUT.is_file() and st.session_state.physics_ready:
            st.audio(WAV_OUTPUT.read_bytes(), format="audio/wav")

    with row_btn[0]:
        if st.button("Regenerate Gmsh mesh", use_container_width=True):
            try:
                with st.spinner("Rebuilding display mesh…"):
                    regenerate_display_mesh(
                        geom,
                        top_wood=top_wood,
                        back_wood=back_wood,
                        clamp_ribs=clamp_ribs,
                        pin_neck_fix=pin_neck,
                        fixture_preset=fixture_preset,
                        geom_fp=geom_fp,
                    )
                invalidate_physics_state()
                st.success("Mesh updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Rebuild failed: {exc}")

    with row_btn[1]:
        if st.button(
            "Generate sound",
            use_container_width=True,
            disabled=not st.session_state.physics_ready,
        ):
            try:
                with st.spinner("Synthesizing sound…"):
                    run_stk(body_json=Path(st.session_state.stk_body_json), top_wood=top_wood)
            except Exception as exc:
                st.error(f"Sound failed: {exc}")


if hasattr(st, "fragment"):
    _render_interactive_panel = st.fragment(_render_interactive_panel)


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
        /* Keep custom component iframe at declared height (avoids 0-height timeout races). */
        iframe[title="components.fast_preview.fast_preview"] {
            min-height: {FAST_PREVIEW_HEIGHT}px !important;
            height: {FAST_PREVIEW_HEIGHT}px !important;
            display: block !important;
        }
        div[data-testid="stCustomComponentV1"] {
            min-height: {FAST_PREVIEW_HEIGHT}px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("GUITAR SIMULATOR")
    st.caption(
        "Design Studio: instant Three.js preview (ROM sliders). "
        "PyVista below: high-fidelity ``display_mesh.msh`` from ``build_3d_guitar.py`` after **Save & Sync**."
    )

    _render_interactive_panel(saved, saved_solver, saved_shape, saved_fixture)


main()
