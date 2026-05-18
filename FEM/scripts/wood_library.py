"""
Discrete wood species library for 3D guitar FEM (top / back assignment).

Top woods: Sitka Spruce, Western Red Cedar.
Back woods: Indian Rosewood, Honduran Mahogany, Maple.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

TOP_WOOD_IDS: List[str] = ["spruce", "cedar"]
BACK_WOOD_IDS: List[str] = ["rosewood", "mahogany", "maple"]

# PyVista / GUI surface colors (physical tag 1 = top, tag 3 = back/sides).
WOOD_PLOT_COLORS: Dict[str, str] = {
    "spruce": "#FFF8DC",    # pale ivory
    "cedar": "#A0522D",     # warm reddish-brown (sienna)
    "maple": "#FFDEAD",     # blonde cream (navajowhite)
    "mahogany": "#8B4513",  # saddle brown
    "rosewood": "#3E1F12",  # dark chocolate brown
}

# 3D shell model uses isotropic reduction from orthotropic sheet constants (E_L, nu_LT, rho).
WOOD_SPECS: Dict[str, Dict[str, Any]] = {
    "spruce": {
        "wood_id": "spruce",
        "role": "top",
        "name": "Sitka Spruce",
        "density": 450.0,
        "E_L": 11.0e9,
        "E_T": 1.0e9,
        "E_R": 0.7e9,
        "nu_LT": 0.37,
        "nu_LR": 0.37,
        "nu_TR": 0.4,
        "G_LT": 0.75e9,
        "G_LR": 0.75e9,
        "G_TR": 0.05e9,
        "q_min": 60,
        "q_max": 80,
        "color": WOOD_PLOT_COLORS["spruce"],
    },
    "cedar": {
        "wood_id": "cedar",
        "role": "top",
        "name": "Western Red Cedar",
        "density": 390.0,
        "E_L": 9.0e9,
        "E_T": 0.85e9,
        "E_R": 0.6e9,
        "nu_LT": 0.35,
        "nu_LR": 0.35,
        "nu_TR": 0.4,
        "G_LT": 0.65e9,
        "G_LR": 0.65e9,
        "G_TR": 0.05e9,
        "q_min": 55,
        "q_max": 75,
        "color": WOOD_PLOT_COLORS["cedar"],
    },
    "rosewood": {
        "wood_id": "rosewood",
        "role": "back",
        "name": "Indian Rosewood",
        "density": 830.0,
        "E_L": 11.5e9,
        "E_T": 1.3e9,
        "E_R": 0.7e9,
        "nu_LT": 0.33,
        "nu_LR": 0.33,
        "nu_TR": 0.4,
        "G_LT": 0.95e9,
        "G_LR": 0.95e9,
        "G_TR": 0.05e9,
        "q_min": 90,
        "q_max": 100,
        "color": WOOD_PLOT_COLORS["rosewood"],
    },
    "mahogany": {
        "wood_id": "mahogany",
        "role": "back",
        "name": "Honduran Mahogany",
        "density": 540.0,
        "E_L": 10.5e9,
        "E_T": 1.2e9,
        "E_R": 0.7e9,
        "nu_LT": 0.35,
        "nu_LR": 0.35,
        "nu_TR": 0.4,
        "G_LT": 0.8e9,
        "G_LR": 0.8e9,
        "G_TR": 0.05e9,
        "q_min": 45,
        "q_max": 60,
        "color": WOOD_PLOT_COLORS["mahogany"],
    },
    "maple": {
        "wood_id": "maple",
        "role": "back",
        "name": "Maple",
        "density": 650.0,
        "E_L": 1.1e10,
        "E_T": 1.1e9,
        "E_R": 0.75e9,
        "nu_LT": 0.32,
        "nu_LR": 0.32,
        "nu_TR": 0.4,
        "G_LT": 0.9e9,
        "G_LR": 0.9e9,
        "G_TR": 0.05e9,
        "q_min": 70,
        "q_max": 90,
        "color": WOOD_PLOT_COLORS["maple"],
    },
}


def _normalize_id(wood_id: str) -> str:
    return str(wood_id).strip().lower().replace(" ", "_")


def plot_color_for_wood(wood_id: str) -> str:
    """Hex color for PyVista tag-based rendering (GUI preview)."""
    key = _normalize_id(wood_id)
    if key not in WOOD_PLOT_COLORS:
        raise KeyError(f"Unknown wood_id {wood_id!r}; known: {sorted(WOOD_PLOT_COLORS)}")
    return WOOD_PLOT_COLORS[key]


def wood_display_name(wood_id: str) -> str:
    key = _normalize_id(wood_id)
    if key not in WOOD_SPECS:
        raise KeyError(f"Unknown wood_id {wood_id!r}")
    return str(WOOD_SPECS[key]["name"])


def material_block_for_id(wood_id: str) -> Dict[str, Any]:
    key = _normalize_id(wood_id)
    if key not in WOOD_SPECS:
        raise KeyError(f"Unknown wood_id {wood_id!r}; known: {sorted(WOOD_SPECS)}")
    spec = WOOD_SPECS[key]
    out = deepcopy(spec)
    out.pop("role", None)
    out.pop("wood_id", None)
    return out


def apply_wood_ids_to_config(
    config: Dict[str, Any],
    *,
    top_wood_id: Optional[str] = None,
    back_wood_id: Optional[str] = None,
) -> None:
    """Set ``materials.top`` / ``materials.back`` from discrete wood IDs (in-place)."""
    if top_wood_id is not None:
        tid = _normalize_id(top_wood_id)
        if tid not in TOP_WOOD_IDS:
            raise ValueError(f"top_wood_id must be one of {TOP_WOOD_IDS}, got {top_wood_id!r}")
        config.setdefault("materials", {})["top"] = material_block_for_id(tid)
    if back_wood_id is not None:
        bid = _normalize_id(back_wood_id)
        if bid not in BACK_WOOD_IDS:
            raise ValueError(f"back_wood_id must be one of {BACK_WOOD_IDS}, got {back_wood_id!r}")
        config.setdefault("materials", {})["back"] = material_block_for_id(bid)


def materialize_lhs_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy LHS flat parameters and resolve ``top_wood_id`` / ``back_wood_id`` into
    dotted keys for nested config merge (without embedding full material dicts in pool JSON).
    """
    out = dict(parameters)
    tid = out.pop("top_wood_id", None) or out.pop("materials.top_wood_id", None)
    bid = out.pop("back_wood_id", None) or out.pop("materials.back_wood_id", None)
    if tid is not None:
        out["materials.top_wood_id"] = _normalize_id(str(tid))
    if bid is not None:
        out["materials.back_wood_id"] = _normalize_id(str(bid))
    return out


def apply_lhs_parameters_to_config(config: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """Apply flat/dotted LHS parameters plus discrete wood IDs to a FEM config dict."""
    flat = materialize_lhs_parameters(parameters)
    tid = flat.pop("materials.top_wood_id", None)
    bid = flat.pop("materials.back_wood_id", None)
    apply_wood_ids_to_config(config, top_wood_id=tid, back_wood_id=bid)
    for key, val in flat.items():
        if not isinstance(key, str) or "." not in key:
            continue
        cur: Dict[str, Any] = config
        parts = [p for p in key.split(".") if p]
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = val


def woods_ortho_json_export() -> Dict[str, Any]:
    """Legacy 2D plate library slice (rosewood, mahogany, spruce) for ``woods_ortho.json``."""
    keys = ("rosewood_indian", "mahogany_honduran", "spruce_sitka")
    mapping = {"rosewood_indian": "rosewood", "mahogany_honduran": "mahogany", "spruce_sitka": "spruce"}
    out: Dict[str, Any] = {}
    for jkey, wid in mapping.items():
        s = WOOD_SPECS[wid]
        out[jkey] = {
            "rho": float(s["density"]),
            "E_L": float(s["E_L"]),
            "E_T": float(s["E_T"]),
            "E_R": float(s["E_R"]),
            "G_LT": float(s["G_LT"]),
            "G_LR": float(s["G_LR"]),
            "G_TR": float(s["G_TR"]),
            "nu_LT": float(s["nu_LT"]),
            "nu_LR": float(s["nu_LR"]),
            "nu_TR": float(s["nu_TR"]),
            "q_min": int(s["q_min"]),
            "q_max": int(s["q_max"]),
            "color": s["color"],
        }
    out["cedar_western"] = {
        "rho": 390.0,
        "E_L": 9.0e9,
        "E_T": 0.85e9,
        "E_R": 0.6e9,
        "G_LT": 0.65e9,
        "G_LR": 0.65e9,
        "G_TR": 5.0e7,
        "nu_LT": 0.35,
        "nu_LR": 0.35,
        "nu_TR": 0.40,
        "q_min": 55,
        "q_max": 75,
        "color": WOOD_PLOT_COLORS["cedar"],
    }
    out["maple_hard"] = {
        "rho": 650.0,
        "E_L": 1.1e10,
        "E_T": 1.1e9,
        "E_R": 0.75e9,
        "G_LT": 0.9e9,
        "G_LR": 0.9e9,
        "G_TR": 5.0e7,
        "nu_LT": 0.32,
        "nu_LR": 0.32,
        "nu_TR": 0.40,
        "q_min": 70,
        "q_max": 90,
        "color": WOOD_PLOT_COLORS["maple"],
    }
    return out
