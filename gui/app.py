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
import streamlit as st

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
NOTE_CACHE_DIR = BASE_DIR / "audio" / "note_cache"

SHAPES_CONFIG = BASE_DIR / "FEM" / "configs" / "rom_shapes.json"
ROM_ROOT = BASE_DIR / "ROM"
CLASSIC_PIPELINE_PROFILE: Dict[str, str] = {
    "shape_name": "classic",
    "mesh_profile": "rom",
    "mesh_level_id": "L_rom_prod",
    "dataset_version": "m4_geometry_corrected_rommesh_v1",
    "run_id_suffix": "rom_shadow_v1",
}
M4_ROM_MANIFEST = ROM_ROOT / "classic" / "rom_model_manifest.json"
M4_ROM_SURROGATE_JSON = ROM_ROOT / "classic" / "m4_modal_surrogate.json"
M4_ROM_SURROGATE_NPZ = ROM_ROOT / "classic" / "m4_modal_surrogate.npz"
M4_ROM_DATASET_REGISTRY = ROM_ROOT / "classic" / "official_rom_dataset.jsonl"
M4_LHS_POOL = ROM_ROOT / "classic" / "lhs_pool.json"
M4_PRODUCTION_PIPELINE = (
    BASE_DIR
    / "FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py"
)
PACKAGED_ROM_NPZ = BASE_DIR / "FEM" / "SORTING" / "final_guitar_rom.npz"
SELECTED_MODES_CSV = BASE_DIR / "FEM" / "SORTING" / "selected_modes.csv"
FALLBACK_MODES_CSV = BASE_DIR / "FEM" / "configs" / "archive" / "selected_modes_SIM1.csv"

SHAPE_OPTIONS = ("Classical", "Dreadnought", "Box")
ROM_NAMESPACE = {"Classical": "classic", "Dreadnought": "dreadnought", "Box": "classic"}
HOLE_RADIUS_MAX_M = 0.08
SOUNDHOLE_FROM_NECK_RATIO = 0.5
ROM_HOLE_RADIUS_BOUNDS = (0.035, 0.055)

FAST_PREVIEW_HEIGHT = 1080
FAST_PREVIEW_WIDTH = 1680
ROM_ONLINE_MODES = 15
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
DEFAULT_FIXTURE_PRESET = "Standing Upright (Front)"
DEFAULT_CLAMP_RIBS = True
DEFAULT_PIN_NECK_FIX = True
CAMERA_BY_FIXTURE: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = {
    "Standing Angled (3D)": ((0.0, -0.45, 1.05), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "Standing Upright (Front)": ((0.22, 0.0, 1.12), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
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

_pyvista_module: Any = None
_stpyvista_callable: Any = None
_stpyvista_import_error: Optional[str] = None


def _import_pyvista() -> Any:
    """Lazy PyVista import — avoids loading VTK at app import time."""
    global _pyvista_module
    if _pyvista_module is not None:
        return _pyvista_module
    try:
        import pyvista as pv_mod
    except ImportError as exc:
        raise RuntimeError(f"pyvista not installed: {exc}") from exc
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pv_mod.OFF_SCREEN = True
    _pyvista_module = pv_mod
    return _pyvista_module


def _import_stpyvista() -> Any:
    """Lazy stpyvista import — defers Streamlit custom-component registration."""
    global _stpyvista_callable, _stpyvista_import_error
    if _stpyvista_callable is not None:
        return _stpyvista_callable
    if _stpyvista_import_error is not None:
        raise RuntimeError(_stpyvista_import_error)
    try:
        from stpyvista import stpyvista as stpyvista_fn
    except Exception as exc:
        _stpyvista_import_error = str(exc)
        raise RuntimeError(str(exc)) from exc
    _stpyvista_callable = stpyvista_fn
    return _stpyvista_callable


def _streamlit_script_active() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _streamlit_script_active():
    st.set_page_config(page_title="Guitar Simulator", layout="wide", initial_sidebar_state="collapsed")


def _init_session() -> None:
    for key, val in {
        "physics_ready": False,
        "saved_geom_fp": "",
        "live_preview_fp": "",
        "stk_body_json": "",
        "acoustics_pending": False,
        "rom_body_ready": False,
        "rom_body_pending": False,
        "rom_body_fingerprint": "",
        "rom_body_error": "",
        "sound_stale": True,
        "developer_fom_mode": False,
        "stk_debug_mode_alias": "",
        "show_display_mesh": False,
        "fixture_fp": "",
        "_clamp_ribs": DEFAULT_CLAMP_RIBS,
        "_pin_neck_fix": DEFAULT_PIN_NECK_FIX,
        "_fixture_preset": DEFAULT_FIXTURE_PRESET,
        "_rom_mesh_fp": "",
        "_pending_fom_run": False,
        "_fast_preview_geom": None,
        "_studio_event_id": "",
        "_studio_templates_sent": False,
        "_studio_component_ready": False,
        "_studio_handshake_id": "",
        "_studio_iframe_payload_fp": "",
        "_studio_iframe_initial": None,
        "_studio_param_change_fp": "",
        "note_cache_ready_fp": "",
        "note_cache_building": False,
        "stk_parameter_hash": "",
        "stk_job_status": "not_started",
        "stk_preview_cache_ready": False,
        "stk_preview_cache_path": "",
        "stk_note_count": 0,
        "active_stk_cache_path": "",
        "active_stk_parameter_hash": "",
        "active_stk_guitar_id": "",
        "active_stk_player_fp": "",
        "active_stk_player_payload": {},
        "active_stk_player_validation": {},
        "stk_generate_intent_hash": "",
        "stk_render_requested": False,
        "stk_show_diagnostics": False,
        "_fast_preview_paths_verified": False,
        "show_mesh_overlay": False,
        "mesh_is_dirty": True,
        "_mesh_overlay_rom_fp": "",
        "_rom_online_fp": "",
        "rom_frequencies_hz": [],
        "rom_mode_shapes": {},
        "rom_basis_missing": False,
        "rom_solver_error": "",
        "rom_solve_elapsed_s": 0.0,
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


def _round_studio_dim(v: Any, ndigits: int = 5) -> float:
    """Round ROM slider dimensions so Python↔iframe sync does not jitter on float noise."""
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return 0.0


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
    length_v = _round_studio_dim(src.get("length", rom_def["length"]))
    width_v = _round_studio_dim(width)
    hole_v = _round_studio_dim(hole)
    hole_cap = 0.25 * min(float(length_v), float(width_v))
    if hole_cap > 1e-5:
        hole_v = min(float(hole_v), hole_cap - 1e-5)
    out: Dict[str, Any] = {
        "shape_type": shape,
        "length": length_v,
        "width": width_v,
        "depth": _round_studio_dim(src.get("depth", rom_def["depth"])),
        "top_thickness": _round_studio_dim(top_t),
        "hole_radius": _round_studio_dim(hole_v),
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
        "wood_colors": dict(src.get("wood_colors") or wood_colors_for_studio()),
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


# Orthotropic plate surface colors (Design Studio / Three.js — tuned for realistic shading).
STUDIO_WOOD_HEX: Dict[str, str] = {
    "spruce": "#C19A6B",
    "cedar": "#5D4037",
    "maple": "#A67C52",
    "mahogany": "#795548",
    "rosewood": "#3F2A20",
}


def wood_colors_for_studio() -> Dict[str, str]:
    """Hex surface colors passed to the Design Studio iframe (orthotropic palette)."""
    return {wid: STUDIO_WOOD_HEX.get(wid, plot_color_for_wood(wid)) for wid in ALL_WOOD_IDS}


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
    """Payload for the Design Studio iframe (templates omitted after first send)."""
    out = studio_payload_for_component(payload)
    out["wood_colors"] = dict(out.get("wood_colors") or wood_colors_for_studio())
    return out


def stable_studio_iframe_initial(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Return cached iframe props unless ROM-relevant fields changed (avoids iframe reload)."""
    prepared = studio_payload_for_iframe(candidate)
    fp = studio_iframe_payload_fp(prepared)
    cached_fp = st.session_state.get("_studio_iframe_payload_fp", "")
    cached = st.session_state.get("_studio_iframe_initial")
    if fp == cached_fp and isinstance(cached, dict):
        return cached
    st.session_state._studio_iframe_payload_fp = fp
    st.session_state._studio_iframe_initial = prepared
    return prepared


def is_studio_handshake_event(event: Dict[str, Any]) -> bool:
    """True for mount/ping payloads from the iframe (no geometry side effects)."""
    if str(event.get("status") or "").strip().lower() == "ready":
        return True
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


def inject_user_flow_css() -> None:
    """Step headings, prominent Generate Sound button, full-width layout."""
    st.markdown(
        """
        <style>
        .user-step-heading {
            font-size: 3.6rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            line-height: 1.08;
            margin: 1.25rem 0 0.65rem 0;
            color: #ffffff;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
        }
        .user-gen-sound-heading {
            font-size: 3.2rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            line-height: 1.1;
            margin: 1.5rem 0 0.75rem 0;
            text-align: center;
            color: #ffffff;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
        }
        .user-step-copy {
            font-size: 1.0rem;
            color: #cbd5e1;
            margin: 0 0 1.1rem 0;
            line-height: 1.5;
        }
        .user-step-block {
            margin-bottom: 1.75rem;
            padding-bottom: 0.25rem;
        }
        .gen-sound-spacer {
            margin: 1.75rem 0 0.5rem 0;
        }
        .gen-sound-block {
            margin: 0.5rem 0 2rem 0;
        }
        section.main button[data-testid="baseButton-primary"] {
            width: min(420px, 100%);
            min-height: 3.25rem;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em;
            border-radius: 0.65rem !important;
            background: linear-gradient(180deg, #22c55e 0%, #16a34a 100%) !important;
            color: #ffffff !important;
            border: 1px solid #15803d !important;
            box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35);
        }
        section.main button[data-testid="baseButton-primary"]:hover {
            background: linear-gradient(180deg, #16a34a 0%, #15803d 100%) !important;
            border-color: #166534 !important;
        }
        section.main button[data-testid="baseButton-primary"]:disabled {
            background: #9ca3af !important;
            border-color: #9ca3af !important;
            box-shadow: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_step_heading(step_num: int, title: str, explanation: str) -> None:
    st.markdown(
        f'<div class="user-step-block">'
        f'<p class="user-step-heading">STEP {step_num}: {title}</p>'
        f'<p class="user-step-copy">{explanation}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def inject_studio_viewport_css() -> None:
    """Force studio iframe visible; hide false-positive Streamlit component timeout banner."""
    h = FAST_PREVIEW_HEIGHT
    st.markdown(
        f"""
        <style>
        div[data-testid="column"]:has(div[data-testid="stCustomComponentV1"]) {{
            position: relative !important;
            min-height: {h}px !important;
            overflow: visible !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }}
        div[data-testid="stCustomComponentV1"] {{
            height: {h}px !important;
            min-height: {h}px !important;
            width: 100% !important;
            max-width: min(100%, 1200px) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 10 !important;
            overflow: visible !important;
        }}
        div[data-testid="stCustomComponentV1"] iframe {{
            height: {h}px !important;
            min-height: {h}px !important;
            width: 100% !important;
            max-width: 100% !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
            visibility: visible !important;
            opacity: 1 !important;
            border: none !important;
        }}
        iframe[src*="fast_preview"] {{
            height: {h}px !important;
            min-height: {h}px !important;
        }}
        div[data-testid="column"]:has(div[data-testid="stCustomComponentV1"]) [data-testid="stAlert"],
        div[data-testid="column"]:has(div[data-testid="stCustomComponentV1"]) div[role="alert"] {{
            display: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def verify_fast_preview_component_paths() -> bool:
    """Print absolute paths for fast_preview static assets before iframe mount."""
    from components.fast_preview import component_dir

    comp_dir = os.path.abspath(component_dir())
    index_path = os.path.join(comp_dir, "index.html")
    bridge_path = os.path.join(comp_dir, "streamlit_bridge.js")
    checks = (
        ("component_dir (declare_component path)", comp_dir, os.path.isdir(comp_dir)),
        ("index.html", index_path, os.path.isfile(index_path)),
        ("streamlit_bridge.js", bridge_path, os.path.isfile(bridge_path)),
    )
    all_ok = True
    for label, path, ok in checks:
        abs_path = os.path.abspath(path)
        print(f"DEBUG fast_preview: {label} -> {abs_path} exists={ok}", flush=True)
        all_ok = all_ok and ok
    print(f"DEBUG fast_preview: all assets OK={all_ok}", flush=True)
    return all_ok


def mount_design_studio_iframe(initial: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Mount Design Studio; ``initial`` should come from ``stable_studio_iframe_initial``."""
    if not st.session_state.get("_fast_preview_paths_verified"):
        verify_fast_preview_component_paths()
        st.session_state._fast_preview_paths_verified = True
    return fast_preview(
        initial=initial,
        key="fast_preview_geom",
        height=FAST_PREVIEW_HEIGHT,
    )


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

    if action == "param_change":
        invalidate_rom_and_audio_state()
        st.session_state.mesh_is_dirty = True
        st.session_state.show_mesh_overlay = False
        st.session_state._mesh_overlay_rom_fp = ""
        return

    geom_fp = geometry_fingerprint(
        geom,
        top_wood,
        back_wood,
        clamp_ribs=clamp_ribs,
        pin_neck_fix=pin_neck,
        fixture_preset=fixture_preset,
    )

    if action in ("save_sync", "run_rom", "run_fem"):
        st.session_state.mesh_is_dirty = True
        st.session_state.show_mesh_overlay = False
        regenerate_display_mesh(
            geom,
            top_wood=top_wood,
            back_wood=back_wood,
            clamp_ribs=clamp_ribs,
            pin_neck_fix=pin_neck,
            fixture_preset=fixture_preset,
            geom_fp=geom_fp,
        )
        invalidate_rom_and_audio_state()
        st.session_state.mesh_is_dirty = False
        st.session_state.show_mesh_overlay = True
        st.session_state._mesh_overlay_rom_fp = rom_mesh_fingerprint(
            geom, top_wood=top_wood, back_wood=back_wood
        )

    if action == "save_sync":
        if not st.session_state.developer_fom_mode:
            st.session_state.rom_body_pending = True
    elif action == "run_rom":
        st.session_state.developer_fom_mode = False
        st.session_state.rom_body_pending = True
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
    with st.status("Initializing Gmsh…", expanded=True) as gmsh_status:
        gmsh_status.update(label="Writing guitar configuration…", state="running")
        save_config(
            geom,
            top_wood=top_wood,
            back_wood=back_wood,
            clamp_ribs=clamp_ribs,
            pin_neck_fix=pin_neck_fix,
            fixture_preset=fixture_preset,
        )
        gmsh_status.update(
            label="Generating display mesh (may take several minutes)…",
            state="running",
        )
        run_gmsh_display()
        gmsh_status.update(label="Gmsh display mesh complete.", state="complete")
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


def m4_rom_manifest_path(shape_type: str) -> Path:
    return ROM_ROOT / rom_namespace(shape_type) / "rom_model_manifest.json"


def m4_rom_available(shape_type: str) -> bool:
    ns = rom_namespace(shape_type)
    root = ROM_ROOT / ns
    return (
        (root / "rom_model_manifest.json").is_file()
        and (root / "m4_modal_surrogate.json").is_file()
        and (root / "m4_modal_surrogate.npz").is_file()
    )


def m4_parameters_from_ui(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
) -> Dict[str, Any]:
    """Parameter dict for M4 modal surrogate (8-D LHS feature encoding)."""
    top_t = float(geom.get("top_thickness") or geom.get("thickness") or 0.003)
    back_t = float(geom.get("back_thickness") or top_t * 1.1)
    return {
        "geometry.length": float(geom["length"]),
        "geometry.width": float(geom["width"]),
        "geometry.depth": float(geom["depth"]),
        "geometry.top_thickness": top_t,
        "geometry.hole_radius": float(geom["hole_radius"]),
        "geometry.back_thickness": back_t,
        "top_wood_id": str(top_wood),
        "back_wood_id": str(back_wood),
    }


def load_m4_rom_manifest(shape_type: str) -> Dict[str, Any]:
    path = m4_rom_manifest_path(shape_type)
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def predict_m4_modal_frequencies(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
    shape_type: str,
    nev: int = ROM_ONLINE_MODES,
) -> Tuple[List[float], Dict[str, Any]]:
    """Online prediction via ROM/classic/m4_modal_surrogate (no FEM)."""
    m4_scripts = BASE_DIR / "FEM/experiments/active_domain_validation/physics_integrity/scripts"
    if str(m4_scripts) not in sys.path:
        sys.path.insert(0, str(m4_scripts))
    from v2_b3_m4_modal_surrogate_lib import load_surrogate_model, predict_modal_catalog  # noqa: WPS433

    ns = rom_namespace(shape_type)
    model = load_surrogate_model(BASE_DIR, ns)
    prediction = predict_modal_catalog(
        model,
        m4_parameters_from_ui(geom, top_wood=top_wood, back_wood=back_wood),
        nev=int(nev),
    )
    raw_freqs = [float(f) for f in (prediction.get("frequencies_hz") or [])]
    if int(nev) > 0:
        freqs = raw_freqs[: int(nev)]
    else:
        freqs = raw_freqs
    return freqs, prediction


def render_classic_pipeline_status(*, shape_type: str) -> None:
    """Sidebar: compact ROM status + optional debug expander (classic guitar)."""
    if rom_namespace(shape_type) != "classic":
        st.sidebar.caption(f"Shape namespace: `{rom_namespace(shape_type)}` (legacy ROM paths)")
        return
    prof = CLASSIC_PIPELINE_PROFILE
    manifest = load_m4_rom_manifest(shape_type)
    if m4_rom_available(shape_type):
        count = int(manifest.get("training_sample_count") or manifest.get("total_training_count") or 0)
        st.sidebar.caption(f"M4 ROM: ready, {count} training samples")
        st.sidebar.caption(
            f"profile: {prof['shape_name']} / {prof['mesh_profile']} / {prof['mesh_level_id']}"
        )
    else:
        st.sidebar.warning("M4 ROM: not found")
    with st.sidebar.expander("Debug / pipeline details"):
        st.code(
            "\n".join(
                (
                    f"shape_name = {prof['shape_name']}",
                    f"mesh_profile = {prof['mesh_profile']}",
                    f"mesh_level_id = {prof['mesh_level_id']}",
                    f"dataset_version = {prof['dataset_version']}",
                    f"run_id_suffix = {prof['run_id_suffix']}",
                )
            ),
            language=None,
        )
        if m4_rom_available(shape_type):
            st.caption(f"Manifest: `{M4_ROM_MANIFEST.relative_to(BASE_DIR)}`")
            st.caption(f"Surrogate: `{M4_ROM_SURROGATE_JSON.relative_to(BASE_DIR)}`")
        else:
            st.caption("Build: `build_m4_rom_from_completed_fom.py --official-rom-mesh-only`")
        if M4_ROM_DATASET_REGISTRY.is_file():
            n_reg = sum(
                1 for line in M4_ROM_DATASET_REGISTRY.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            st.caption(f"Official dataset registry: {n_reg} entries")
        if M4_LHS_POOL.is_file():
            st.caption(f"LHS pool: `{M4_LHS_POOL.relative_to(BASE_DIR)}`")


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
        combined = "\n".join(
            line for line in (result.stdout or "").splitlines() + (result.stderr or "").splitlines() if line.strip()
        )
        fatal = next((ln for ln in combined.splitlines() if "FATAL:" in ln or "RuntimeError:" in ln), "")
        tail = combined[-4000:] if combined else "unknown error"
        detail = fatal or tail
        raise RuntimeError(
            f"{label} failed.\n{detail}\n\n"
            "Check geometry parameters (length, width, depth, top thickness, hole radius) "
            "and ensure Gmsh/OpenCASCADE is available."
        )


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


def load_surface_mesh(msh_path: Path) -> Optional[Any]:
    try:
        import meshio
    except ImportError:
        return None
    try:
        pv = _import_pyvista()
    except RuntimeError:
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
        cleaned = poly.clean(tolerance=1e-8, absolute=True)
        return cleaned
    except Exception:
        return None


def apply_spatial_colormap(
    mesh: Any,
    *,
    top_color: str,
    back_color: str,
    body_length: float,
    hole_radius: float,
) -> Any:
    centers = mesh.cell_centers().points
    z = centers[:, 2]
    zmax, zmin = float(np.max(z)), float(np.min(z))
    top_z = zmax - TOP_Z_BAND_FRAC * max(zmax - zmin, 1e-9)
    top_rgb = np.array(hex_to_rgb01(top_color), dtype=np.float32)
    back_rgb = np.array(hex_to_rgb01(back_color), dtype=np.float32)
    is_top = z >= top_z
    colors = np.tile(back_rgb, (mesh.n_cells, 1))
    colors[is_top] = top_rgb
    out = mesh.copy(deep=True)
    out.cell_data["rgb"] = colors
    return out


def render_guitar(
    mesh: Optional[Any],
    *,
    top_color: str,
    back_color: str,
    body_length: float,
    hole_radius: float,
    sketch_mode: bool,
    plot_key: str,
    fixture_preset: str,
    show_mesh_edges: bool = False,
) -> None:
    try:
        pv = _import_pyvista()
        stpyvista = _import_stpyvista()
    except RuntimeError as exc:
        st.warning(
            "3D mesh viewer unavailable (PyVista/stpyvista). "
            f"The display mesh file is still saved on disk. Detail: {exc}"
        )
        return
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
        edges_on = sketch_mode or show_mesh_edges
        plotter.add_mesh(
            colored,
            scalars="rgb",
            rgb=True,
            preference="cell",
            show_edges=edges_on,
            edge_color="#3d2817" if sketch_mode else "#2c3e50",
            line_width=0.8 if sketch_mode else 1.0,
            smooth_shading=not sketch_mode,
        )
    else:
        plotter.add_text("Preview unavailable", position="upper_left", font_size=12)
    plotter.camera_position = CAMERA_BY_FIXTURE.get(fixture_preset, CAMERA_BY_FIXTURE[DEFAULT_FIXTURE_PRESET])
    plotter.enable_anti_aliasing("ssaa")
    stpyvista(plotter, key=plot_key)
    plotter.close()


def render_validation_mesh_viewport(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
    geom_fp: str,
    fixture_preset: str,
) -> None:
    """Gmsh ``display_mesh.msh`` — rendered below the Design Studio iframe when in sync."""
    st.markdown(
        """
        <style>
        div.gmsh-validation-block {
            max-width: 1120px;
            margin: 1.25rem auto 2rem auto;
            padding: 0 0.5rem;
            width: 100%;
        }
        div.gmsh-validation-block [data-testid="stCaptionContainer"] {
            text-align: center;
        }
        div.gmsh-validation-block div[data-testid="stpyvista"] {
            margin-left: auto !important;
            margin-right: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="gmsh-validation-block">', unsafe_allow_html=True)
        st.caption(
            "Compiled ``display_mesh.msh`` (12 mm shell, 3 mm seam band) from **Save & Sync** "
            "or **Regenerate Gmsh mesh**. Engineering FOM mesh uses separate local refinement."
        )
        mesh, _, mesh_src = get_view_mesh(geom_fp)
        if mesh is None and not DISPLAY_MESH_FILE.is_file():
            st.info(
                "Mesh file missing. Click **Regenerate Gmsh mesh** or **Save & Sync** "
                "in the Design Studio."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return
        dm = load_surface_mesh(DISPLAY_MESH_FILE)
        n_cells = f"{dm.n_cells:,} triangles" if dm is not None else "—"
        st.caption(f"`display_mesh.msh` · {n_cells} · source: {mesh_src or 'display_mesh.msh'}")
        try:
            try:
                _import_pyvista().set_jupyter_backend("static")
            except RuntimeError:
                pass
            render_guitar(
                mesh,
                top_color=plot_color_for_wood(top_wood),
                back_color=plot_color_for_wood(back_wood),
                body_length=geom["length"],
                hole_radius=geom["hole_radius"],
                sketch_mode=False,
                plot_key=f"display_{mesh_src}_{geom_fp[:12]}",
                fixture_preset=fixture_preset,
                show_mesh_edges=True,
            )
        except Exception as exc:
            st.warning(f"Render error: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)


def display_mesh_active(geom_fp: str) -> bool:
    return (
        bool(st.session_state.show_display_mesh)
        and st.session_state.saved_geom_fp == geom_fp
        and DISPLAY_MESH_FILE.is_file()
    )


def get_view_mesh(geom_fp: str) -> Tuple[Optional[Any], bool, str]:
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
    invalidate_rom_and_audio_state()


def invalidate_rom_and_audio_state() -> None:
    """Mark ROM body response and synthesized audio stale (design changed)."""
    st.session_state.physics_ready = False
    st.session_state.acoustics_pending = False
    st.session_state.rom_body_ready = False
    st.session_state.rom_body_pending = False
    st.session_state.rom_body_fingerprint = ""
    st.session_state.rom_body_error = ""
    st.session_state.stk_body_json = ""
    st.session_state.sound_stale = True
    st.session_state.note_cache_ready_fp = ""
    st.session_state.note_cache_building = False
    st.session_state["stk_parameter_hash"] = ""
    st.session_state["stk_job_status"] = "not_started"
    st.session_state["stk_preview_cache_ready"] = False
    st.session_state["stk_preview_cache_path"] = ""
    st.session_state["stk_note_count"] = 0
    st.session_state["active_stk_cache_path"] = ""
    st.session_state["active_stk_parameter_hash"] = ""
    st.session_state["active_stk_guitar_id"] = ""
    st.session_state["active_stk_player_fp"] = ""
    st.session_state["active_stk_player_payload"] = {}
    st.session_state["active_stk_player_validation"] = {}
    st.session_state["stk_generate_intent_hash"] = ""
    st.session_state["stk_render_requested"] = False
    try:
        from stk_app_audio_service import mark_stk_job_stale  # noqa: WPS433

        mark_stk_job_stale()
    except Exception:
        pass


def invalidate_saved_state() -> None:
    invalidate_rom_and_audio_state()


def display_mesh_file_ready() -> bool:
    return DISPLAY_MESH_FILE.is_file() and DISPLAY_MESH_FILE.stat().st_size > 0


def rom_body_response_ready(current_rom_fp: str) -> bool:
    """True when M4 ROM body JSON matches the current design fingerprint."""
    if st.session_state.get("developer_fom_mode"):
        path = str(st.session_state.get("stk_body_json") or "")
        return bool(st.session_state.get("physics_ready")) and bool(path) and Path(path).is_file()
    if not st.session_state.get("rom_body_ready"):
        return False
    if st.session_state.get("rom_body_fingerprint") != current_rom_fp:
        return False
    body_json = str(st.session_state.get("stk_body_json") or "")
    if not body_json or not Path(body_json).is_file():
        return False
    return ROM_STK_JSON.is_file() and ROM_STK_JSON.stat().st_size > 0


def mark_rom_body_stale_if_design_changed(current_rom_fp: str) -> None:
    """Invalidate ROM/audio when sliders move without a fresh Save & Sync."""
    if not st.session_state.get("rom_body_ready"):
        return
    if st.session_state.get("rom_body_fingerprint") != current_rom_fp:
        st.session_state.rom_body_ready = False
        st.session_state.physics_ready = False
        st.session_state.sound_stale = True
        st.session_state.stk_body_json = ""
        st.session_state.note_cache_ready_fp = ""
        try:
            from stk_app_audio_service import mark_stk_job_stale  # noqa: WPS433

            mark_stk_job_stale()
        except Exception:
            pass


def complete_rom_body_response(
    lhs_params: Dict[str, Any],
    shape_type: str,
    *,
    rom_fp: str,
) -> Path:
    """Run M4 ROM prediction and write ``rom_stk_body.json`` (no FEM)."""
    stk_path = run_rom_acoustics(lhs_params, shape_type)
    st.session_state.stk_body_json = str(stk_path)
    st.session_state.rom_body_ready = True
    st.session_state.physics_ready = True
    st.session_state.rom_body_fingerprint = rom_fp
    st.session_state.rom_body_error = ""
    st.session_state.rom_body_pending = False
    st.session_state.sound_stale = True
    try:
        from stk_final_v1_precompute_cache import trigger_stk_precompute_if_ready  # noqa: WPS433

        trigger_stk_precompute_if_ready(
            repo_root=BASE_DIR,
            modal_json=stk_path,
            lhs_params=lhs_params,
            geometry_config=CONFIG_PATH,
            stk_mode_alias=website_stk_mode_alias(),
            developer_debug=bool(st.session_state.get("developer_fom_mode")),
        )
    except Exception:
        pass
    return stk_path


def schedule_stk_note_library_after_rom(
    lhs_params: Dict[str, Any],
    *,
    rom_fp: str,
) -> None:
    """Start background STK note-library build (Generate-only; not called after ROM)."""
    try:
        from stk_app_audio_service import (  # noqa: WPS433
            compute_parameter_hash,
            schedule_stk_after_rom,
        )

        job = schedule_stk_after_rom(
            rom_fp=rom_fp,
            lhs_params=lhs_params,
            repo_root=BASE_DIR,
        )
        st.session_state["stk_parameter_hash"] = compute_parameter_hash(rom_fp, lhs_params)
        st.session_state["stk_job_status"] = str(job.get("status") or "running")
    except Exception as exc:
        st.session_state["stk_job_status"] = "failed"
        st.session_state.rom_body_error = str(exc)


def website_stk_mode_alias() -> str:
    """Pipeline STK alias for Generate (default: frozen final v1)."""
    from stk_pipeline_defaults import (  # noqa: WPS433
        DEFAULT_WEBSITE_STK_MODE,
        resolve_pipeline_stk_mode_alias,
    )

    override = str(st.session_state.get("stk_debug_mode_alias") or "").strip()
    if st.session_state.get("developer_fom_mode") and override:
        return resolve_pipeline_stk_mode_alias(
            override=override,
            developer_debug=True,
        )
    return DEFAULT_WEBSITE_STK_MODE


def website_stk_user_label() -> str:
    from stk_pipeline_defaults import user_label_for_stk_mode  # noqa: WPS433

    return user_label_for_stk_mode(website_stk_mode_alias())


def render_stk_debug_controls() -> None:
    """Developer-only STK mode override (de-thump safety variant)."""
    if not st.session_state.get("developer_fom_mode"):
        return
    from stk_pipeline_defaults import (  # noqa: WPS433
        DEFAULT_WEBSITE_STK_MODE,
        debug_stk_mode_options,
        user_label_for_stk_mode,
    )

    labels = {alias: label for alias, label in debug_stk_mode_options()}
    options = [DEFAULT_WEBSITE_STK_MODE] + [
        alias for alias in labels if alias != DEFAULT_WEBSITE_STK_MODE
    ]
    current = website_stk_mode_alias()
    if current not in options:
        current = DEFAULT_WEBSITE_STK_MODE
    with st.sidebar.expander("STK synthesis (debug)"):
        picked = st.selectbox(
            "Body transfer mode",
            options=options,
            index=options.index(current),
            format_func=lambda alias: user_label_for_stk_mode(alias),
            key="stk_debug_mode_select",
        )
        st.session_state.stk_debug_mode_alias = picked


def render_rom_body_status(*, current_rom_fp: str) -> None:
    """Sidebar/control status for ROM body response readiness."""
    if st.session_state.get("rom_body_pending"):
        st.caption("ROM body response: preparing…")
        return
    if rom_body_response_ready(current_rom_fp):
        freqs = [float(f) for f in (st.session_state.get("rom_frequencies_hz") or []) if float(f) > 0]
        if not freqs and ROM_STK_JSON.is_file():
            try:
                doc = json.loads(ROM_STK_JSON.read_text(encoding="utf-8"))
                freqs = [float(f) for f in (doc.get("modes_hz") or []) if float(f) > 0]
            except (OSError, ValueError, json.JSONDecodeError):
                freqs = []
        n_modes = len(freqs)
        if n_modes:
            st.caption(
                f"ROM body response: **ready** · modes: {n_modes} · "
                f"range: {min(freqs):.0f}–{max(freqs):.0f} Hz"
            )
        else:
            st.caption("ROM body response: **ready**")
        return
    err = str(st.session_state.get("rom_body_error") or "").strip()
    if err:
        st.warning(f"ROM body response failed: {err}")
    elif st.session_state.get("rom_body_ready"):
        st.caption("ROM body response: stale — Save & Sync again")
    else:
        st.caption("ROM body response: not ready — Save & Sync to prepare sound")


def try_import_rom():
    try:
        from rom_manager import ROMManager  # noqa: WPS433

        return ROMManager, None
    except Exception as exc:
        return None, exc


@st.cache_resource(show_spinner=False)
def get_rom_manager():
    """Singleton ROM manager for online reduced-basis solves (MPI-aware)."""
    ROMManager, err = try_import_rom()
    if ROMManager is None:
        return None, str(err)
    try:
        return ROMManager(shapes_config_path=SHAPES_CONFIG), None
    except Exception as exc:
        return None, str(exc)


def update_rom_online_prediction(
    geom: Dict[str, Any],
    *,
    top_wood: str,
    back_wood: str,
    shape_type: str,
) -> None:
    """Project 7D ROM parameters through ``ROMManager.solve_online``; cache modal frequencies."""
    fp = rom_mesh_fingerprint(geom, top_wood=top_wood, back_wood=back_wood)
    if fp == st.session_state.get("_rom_online_fp"):
        return
    st.session_state._rom_online_fp = fp

    ns = rom_namespace(shape_type)
    if m4_rom_available(shape_type):
        try:
            freqs, prediction = predict_m4_modal_frequencies(
                geom,
                top_wood=top_wood,
                back_wood=back_wood,
                shape_type=shape_type,
            )
            st.session_state.rom_basis_missing = False
            st.session_state.rom_backend = "m4_modal_surrogate"
            st.session_state.rom_frequencies_hz = freqs
            st.session_state.rom_mode_shapes = dict(prediction)
            st.session_state.rom_solve_elapsed_s = float(prediction.get("runtime_s") or 0.0)
            st.session_state.rom_solver_error = ""
            return
        except Exception as exc:
            st.session_state.rom_backend = "m4_modal_surrogate"
            st.session_state.rom_solver_error = f"M4 surrogate: {exc}"

    basis = rom_basis_path(shape_type)
    if not basis.is_file():
        st.session_state.rom_basis_missing = True
        st.session_state.rom_frequencies_hz = []
        st.session_state.rom_mode_shapes = {}
        if not st.session_state.get("rom_solver_error"):
            st.session_state.rom_solver_error = ""
        return

    st.session_state.rom_basis_missing = False
    st.session_state.rom_backend = "legacy_reduced_basis"
    manager, err = get_rom_manager()
    if manager is None:
        st.session_state.rom_solver_error = err or "ROMManager unavailable"
        st.session_state.rom_frequencies_hz = []
        st.session_state.rom_mode_shapes = {}
        return

    lhs_params = lhs_params_from_ui(geom, top_wood=top_wood, back_wood=back_wood)
    try:
        result = manager.solve_online(ns, params=lhs_params, nev=ROM_ONLINE_MODES)
        freqs = [float(f) for f in (result.get("freqs_hz") or [])[:ROM_ONLINE_MODES]]
        st.session_state.rom_frequencies_hz = freqs
        st.session_state.rom_mode_shapes = dict(result)
        st.session_state.rom_solve_elapsed_s = float(result.get("elapsed_s") or 0.0)
        st.session_state.rom_solver_error = ""
    except Exception as exc:
        st.session_state.rom_solver_error = str(exc)
        st.session_state.rom_frequencies_hz = []
        st.session_state.rom_mode_shapes = {}


def render_rom_metrics_dashboard(shape_type: str) -> None:
    """Natural-frequency table fed by the online ROM solver."""
    st.subheader("ROM modes")
    ns = rom_namespace(shape_type)
    basis = rom_basis_path(shape_type)

    if not m4_rom_available(shape_type) and (
        st.session_state.get("rom_basis_missing") or not basis.is_file()
    ):
        st.info(
            f"ROM prediction unavailable for **{shape_type}**. "
            f"Expected `ROM/{ns}/m4_modal_surrogate.{{json,npz}}`."
        )
        return

    err = str(st.session_state.get("rom_solver_error") or "").strip()
    if err:
        st.warning(f"ROM prediction unavailable: {err}")
        return

    freqs: List[float] = list(st.session_state.get("rom_frequencies_hz") or [])
    if not freqs:
        st.caption("Move a slider to refresh modal frequencies.")
        return

    display_n = min(len(freqs), ROM_ONLINE_MODES)
    elapsed = float(st.session_state.get("rom_solve_elapsed_s") or 0.0)
    backend = str(st.session_state.get("rom_backend") or "rom")
    st.caption(f"{backend} · {display_n} modes · {elapsed * 1000.0:.0f} ms")

    rows = [
        {"Mode": i + 1, "Frequency (Hz)": f"{float(freqs[i]):.1f} Hz"}
        for i in range(display_n)
    ]
    st.table(rows)


def write_stk_body_json(
    freqs_hz: Sequence[float],
    out_path: Path,
    *,
    prediction: Optional[Dict[str, Any]] = None,
) -> Path:
    freqs = [float(f) for f in freqs_hz if float(f) > 0.0]
    if not freqs:
        raise ValueError("No modal frequencies for STK.")
    weights = [1.0 / (1.0 + 0.25 * i) for i in range(len(freqs))]
    wmax = max(weights)
    weights = [w / wmax for w in weights]
    doc: Dict[str, Any] = {
        "analysis": "rom_online_body",
        "modes_hz": freqs,
        "mode_weights": weights,
        "num_modes": len(freqs),
        "full_modal_band_hz": [60.0, 550.0],
    }
    predicted_modes = list((prediction or {}).get("predicted_modes") or [])
    if predicted_modes:
        doc["predicted_modes"] = predicted_modes
        doc["frequencies_hz"] = freqs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out_path


def run_rom_acoustics(lhs_params: Dict[str, Any], shape_type: str) -> Path:
    """Branch A: ROM only — no FOM Gmsh volume mesh."""
    geom = st.session_state.get("_geom") or {}
    top_wood = str(st.session_state.get("_top_wood") or "spruce")
    back_wood = str(st.session_state.get("_back_wood") or "rosewood")
    if m4_rom_available(shape_type):
        freqs, prediction = predict_m4_modal_frequencies(
            geom,
            top_wood=top_wood,
            back_wood=back_wood,
            shape_type=shape_type,
            nev=0,
        )
        if not freqs:
            raise RuntimeError("M4 modal surrogate returned no frequencies.")
        st.session_state.rom_frequencies_hz = [float(f) for f in freqs[:ROM_ONLINE_MODES]]
        st.session_state.rom_mode_shapes = dict(prediction)
        st.session_state.rom_backend = "m4_modal_surrogate"
        write_stk_body_json(freqs, ROM_STK_JSON, prediction=prediction)
        return ROM_STK_JSON

    ns = rom_namespace(shape_type)
    basis = rom_basis_path(shape_type)
    if not basis.is_file():
        raise RuntimeError(
            f"No ROM model for {shape_type}: missing M4 surrogate and legacy basis {basis}"
        )
    manager, err = get_rom_manager()
    if manager is None:
        raise RuntimeError(f"ROM unavailable: {err}")
    result = manager.solve_online(ns, params=lhs_params, nev=0)
    freqs = list(result.get("freqs_hz") or [])
    if not freqs:
        raise RuntimeError("ROM solve_online returned no frequencies.")
    st.session_state.rom_frequencies_hz = [float(f) for f in freqs[:ROM_ONLINE_MODES]]
    st.session_state.rom_mode_shapes = dict(result)
    st.session_state.rom_backend = "legacy_reduced_basis"
    write_stk_body_json(freqs, ROM_STK_JSON, prediction=result if isinstance(result, dict) else None)
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


def run_stk(
    *,
    body_json: Path,
    top_wood: str,
    note_hz: Optional[float] = None,
    note_name: str = "A2",
    lhs_params: Optional[Dict[str, Any]] = None,
    stk_mode_alias: Optional[str] = None,
    precompute_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stage-1 Python body-response synth (frozen STK final v1 default)."""
    from body_response_synth import (  # noqa: WPS433
        DEFAULT_DURATION_S,
        DEFAULT_SAMPLE_RATE,
        load_modal_data_from_path,
        synthesize_note_with_body_response,
    )
    from stk_pipeline_defaults import (  # noqa: WPS433
        WEBSITE_SAMPLE_ID,
        enrich_sample_parameters_for_note,
        resolve_pipeline_stk_mode_alias,
    )

    pitch_hz = float(note_hz if note_hz is not None else DEFAULT_STK_NOTE_HZ)
    meta_path = WAV_OUTPUT.with_name(f"{WAV_OUTPUT.stem}_metadata.json")
    modal_data = load_modal_data_from_path(body_json)
    alias = resolve_pipeline_stk_mode_alias(override=stk_mode_alias)
    if precompute_bundle:
        base_params = dict(precompute_bundle.get("sample_parameters") or {})
        if precompute_bundle.get("model_alias"):
            alias = str(precompute_bundle["model_alias"])
    elif lhs_params:
        from stk_pipeline_defaults import lhs_params_to_sample_parameters  # noqa: WPS433

        base_params = lhs_params_to_sample_parameters(lhs_params)
    else:
        base_params = {"top_wood_id": top_wood}
    note_params = enrich_sample_parameters_for_note(
        base_parameters=base_params,
        modal_data=modal_data,
        frequency_hz=pitch_hz,
        stk_mode_alias=alias,
        sample_id=WEBSITE_SAMPLE_ID,
        repo_root=BASE_DIR,
    )
    metadata = synthesize_note_with_body_response(
        frequency_hz=pitch_hz,
        note_name=note_name,
        duration_s=DEFAULT_DURATION_S,
        sample_rate=DEFAULT_SAMPLE_RATE,
        modal_data=modal_data,
        output_wav=WAV_OUTPUT,
        output_metadata_json=meta_path,
        diagnostic_mode=alias,
        sample_parameters=note_params,
        repo_root=BASE_DIR,
        sample_id=WEBSITE_SAMPLE_ID,
    )
    append_silence_wav(WAV_OUTPUT, 0.3)
    return metadata


def _render_main_studio(
    saved: Dict[str, Any],
    saved_solver: Dict[str, Any],
    saved_shape: str,
    saved_fixture: str,
) -> None:
    """Minimal layout: Design Studio iframe always mounted; Gmsh overlay on save only."""
    clamp_ribs = DEFAULT_CLAMP_RIBS
    pin_neck = DEFAULT_PIN_NECK_FIX
    fixture_preset = DEFAULT_FIXTURE_PRESET
    st.session_state["_clamp_ribs"] = clamp_ribs
    st.session_state["_pin_neck_fix"] = pin_neck
    st.session_state["_fixture_preset"] = fixture_preset

    fp_seed = studio_initial_from_saved(
        saved, saved_shape, developer_fom_mode=st.session_state.developer_fom_mode
    )
    fp_raw = st.session_state.get("_fast_preview_geom")
    if isinstance(fp_raw, dict) and fp_raw:
        fp_seed = sanitize_studio_payload({**fp_seed, **fp_raw}, saved_shape)

    fp_seed["gui_mode"] = "admin" if st.session_state.developer_fom_mode else "user"

    _render_step_heading(
        1,
        "DESIGN YOUR GUITAR",
        "Adjust the guitar shape and materials in the studio below. When ready, click <strong>Save &amp; Sync</strong>.",
    )

    iframe_initial = stable_studio_iframe_initial(fp_seed)
    studio_event = mount_design_studio_iframe(iframe_initial)

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
    geom, top_wood, back_wood = geom_from_studio_event(fp_live)
    shape = str(fp_live.get("shape_type", saved_shape))
    if shape not in SHAPE_OPTIONS:
        shape = saved_shape
    rom_fp = rom_mesh_fingerprint(geom, top_wood=top_wood, back_wood=back_wood)
    geom_fp = geometry_fingerprint(
        geom,
        top_wood,
        back_wood,
        clamp_ribs=clamp_ribs,
        pin_neck_fix=pin_neck,
        fixture_preset=fixture_preset,
    )

    pinned_mesh_fp = str(st.session_state.get("_mesh_overlay_rom_fp", ""))
    if pinned_mesh_fp and rom_fp != pinned_mesh_fp:
        st.session_state.mesh_is_dirty = True
        st.session_state.show_mesh_overlay = False
        mark_rom_body_stale_if_design_changed(rom_fp)
    elif not pinned_mesh_fp:
        st.session_state.mesh_is_dirty = True

    mark_rom_body_stale_if_design_changed(rom_fp)

    lhs_params = lhs_params_from_ui(geom, top_wood=top_wood, back_wood=back_wood)
    st.session_state["_geom"] = geom
    st.session_state["_top_wood"] = top_wood
    st.session_state["_back_wood"] = back_wood

    stk_job: Dict[str, Any] = {}
    if rom_body_response_ready(rom_fp):
        try:
            from stk_app_audio_service import (  # noqa: WPS433
                compute_parameter_hash,
                refresh_stk_background_job_status,
            )

            _stk_ph = compute_parameter_hash(rom_fp, lhs_params)
            stk_job = refresh_stk_background_job_status(_stk_ph)
            st.session_state["stk_parameter_hash"] = _stk_ph
            st.session_state["stk_job_status"] = str(stk_job.get("status") or "not_started")
            st.session_state["stk_preview_cache_ready"] = bool(stk_job.get("preview_cache_ready"))
            st.session_state["stk_preview_cache_path"] = str(stk_job.get("preview_cache_path") or "")
            st.session_state["stk_note_count"] = int(
                stk_job.get("actual_wav_count") or stk_job.get("wav_count") or stk_job.get("note_count") or 0
            )
        except Exception:
            st.session_state["stk_job_status"] = "failed"
            st.session_state["stk_preview_cache_ready"] = False
    elif st.session_state.get("rom_body_pending"):
        st.session_state["stk_job_status"] = "waiting_for_rom"
    elif not st.session_state.get("rom_body_ready"):
        st.session_state["stk_job_status"] = "waiting_for_rom"

    update_rom_online_prediction(geom, top_wood=top_wood, back_wood=back_wood, shape_type=shape)

    _render_step_heading(
        2,
        "VIEW THE FULL MODEL OF YOUR GUITAR",
        "After saving, inspect the full generated guitar model. This is the detailed mesh view of your design.",
    )

    if st.session_state.get("show_mesh_overlay") and not st.session_state.get("mesh_is_dirty", True):
        render_validation_mesh_viewport(
            geom,
            top_wood=top_wood,
            back_wood=back_wood,
            geom_fp=geom_fp,
            fixture_preset=fixture_preset,
        )
    elif st.session_state.get("mesh_is_dirty"):
        st.caption(
            "The full model appears here after you click **Save & Sync** in the Design Studio."
        )

    st.markdown('<div class="gen-sound-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="user-gen-sound-heading">GENERATE SOUND</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gen-sound-block">', unsafe_allow_html=True)
    _stk_status = str(st.session_state.get("stk_job_status") or "not_started")
    if not rom_body_response_ready(rom_fp):
        st.caption("Save & Sync first to simulate your guitar and prepare sound.")
    elif not st.session_state.get("stk_render_requested") and _stk_status in (
        "not_started",
        "stale",
    ):
        st.caption("Click **Generate Sound** to build your guitar audio.")
    else:
        from stk_app_audio_service import user_facing_stk_status  # noqa: WPS433

        st.caption(user_facing_stk_status(_stk_status))
    _gc1, _gc2, _gc3 = st.columns([1, 2, 1])
    with _gc2:
        gen_sound = st.button(
            "Generate Sound",
            type="primary",
            use_container_width=True,
            disabled=not rom_body_response_ready(rom_fp),
            key="btn_gen_sound",
            help="Save this guitar when sound is ready and open the interactive player.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if gen_sound:
        if not rom_body_response_ready(rom_fp):
            st.warning("Save & Sync first to prepare your guitar.")
        else:
            from stk_app_ui import request_generate_guitar  # noqa: WPS433

            try:
                result = request_generate_guitar(
                    repo_root=BASE_DIR,
                    rom_fp=rom_fp,
                    lhs_params=lhs_params,
                    geom=geom,
                    top_wood=top_wood,
                    back_wood=back_wood,
                    rom_physical_summary_path=str(st.session_state.get("stk_body_json") or ""),
                )
                if result.get("action") in ("stk_started", "stk_running"):
                    st.info(
                        "Building guitar sound… This may take a few minutes. "
                        "Click **Generate Sound** again when ready to load the player."
                    )
                elif result.get("action") == "loaded_existing":
                    st.info("This guitar is already saved — loaded from comparison stack.")
                else:
                    st.success(
                        f"Saved **{result['entry'].get('display_name')}** — play notes on the guitar below."
                    )
            except RuntimeError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Could not save guitar: {exc}")

    from app_stk_config import load_app_stk_config  # noqa: WPS433

    _stk_cfg = load_app_stk_config(BASE_DIR)
    if (
        st.session_state.get("stk_render_requested")
        and _stk_status in ("running", "partial_ready")
    ):
        _refresh_s = int(_stk_cfg.get("auto_refresh_interval_s") or 12)
        st.markdown(
            f'<meta http-equiv="refresh" content="{_refresh_s}">',
            unsafe_allow_html=True,
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

    if (
        st.session_state.get("rom_body_pending")
        and not st.session_state.developer_fom_mode
        and display_mesh_active(geom_fp)
        and display_mesh_file_ready()
    ):
        with st.spinner("Preparing ROM body response…"):
            try:
                complete_rom_body_response(lhs_params, shape, rom_fp=rom_fp)
                # STK auto-background after Save/ROM is intentionally disabled.
                # Current stable flow: STK note cache generation starts only from Generate.
                st.rerun()
            except Exception as exc:
                st.session_state.rom_body_pending = False
                st.session_state.rom_body_error = str(exc)
                st.warning(f"ROM body response failed: {exc}")

    _render_step_heading(
        3,
        "LISTEN TO YOUR GUITAR",
        "After sound is ready, click <strong>Generate Sound</strong> to save your guitar and play notes "
        "on the interactive fretboard below.",
    )

    from components.guitar_player import guitar_player  # noqa: WPS433
    from stk_app_ui import (  # noqa: WPS433
        apply_stk_activation_to_session,
        load_stack_guitar_for_player,
        render_saved_guitars_row,
        render_stk_diagnostics_panel,
    )

    loaded_guitar_id = render_saved_guitars_row(base_key="stk_saved_row")
    if loaded_guitar_id:
        try:
            activation = load_stack_guitar_for_player(loaded_guitar_id)
            apply_stk_activation_to_session(activation)
        except Exception as exc:
            st.warning(f"Could not load saved guitar: {exc}")

    active_payload = st.session_state.get("active_stk_player_payload") or {}
    player_validation = st.session_state.get("active_stk_player_validation") or {}
    if active_payload.get("status") == "ready" and active_payload.get("positions"):
        if player_validation and not player_validation.get("ok"):
            st.warning(
                "Guitar player cache is incomplete: "
                + ", ".join(player_validation.get("errors") or ["unknown error"])
            )
            player_payload = {"status": "hidden", "positions": [], "fingerprint": ""}
            player_key = "guitar_player_invalid"
        else:
            player_payload = active_payload
            player_key = f"guitar_player_{st.session_state.get('active_stk_player_fp') or 'stk'}"
    elif _stk_status in ("running", "partial_ready", "not_started", "waiting_for_rom"):
        player_payload = {"status": "building", "positions": [], "fingerprint": ""}
        player_key = "guitar_player_building"
    else:
        player_payload = {"status": "hidden", "positions": [], "fingerprint": ""}
        player_key = "guitar_player_idle"

    guitar_player(player=player_payload, key=player_key, height=560)

    with st.expander("Advanced STK diagnostics", expanded=False):
        if st.checkbox("Show developer diagnostics", key="stk_show_diagnostics"):
            render_stk_diagnostics_panel(
                repo_root=BASE_DIR,
                base_key="stk_diag",
                rom_fp=rom_fp,
                lhs_params=lhs_params,
                rom_ready=rom_body_response_ready(rom_fp),
                rom_pending=bool(st.session_state.get("rom_body_pending")),
                rom_error=str(st.session_state.get("rom_body_error") or ""),
            )


def main() -> None:
    saved = _load_saved_config().get("geometry", {}) or {}
    saved_solver = _load_saved_config().get("solver", {}) or {}
    saved_shape = str(saved.get("shape_type", "Classical"))
    if saved_shape not in SHAPE_OPTIONS:
        saved_shape = "Classical"
    saved_fixture = str(saved.get("fixture_preset", DEFAULT_FIXTURE_PRESET))
    if saved_fixture not in FIXTURE_PRESETS:
        saved_fixture = DEFAULT_FIXTURE_PRESET
    st.title("Guitar Simulator")
    st.caption("Design your guitar, view the full model, then generate sound and play notes interactively.")
    inject_user_flow_css()
    inject_studio_viewport_css()

    _rom_mgr, _rom_init_err = get_rom_manager()
    if _rom_init_err and not m4_rom_available(saved_shape):
        st.sidebar.warning(f"Legacy ROM engine offline: {_rom_init_err}")
    render_classic_pipeline_status(shape_type=saved_shape)
    render_stk_debug_controls()

    _render_main_studio(saved, saved_solver, saved_shape, saved_fixture)


if _streamlit_script_active():
    main()
