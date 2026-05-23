"""Streamlit custom component: ROM-aligned 3D Design Studio (Three.js)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent

_fast_preview = components.declare_component(
    "fast_preview",
    path=str(_COMPONENT_DIR),
)


def fast_preview(
    *,
    initial: Optional[Dict[str, Any]] = None,
    key: Optional[str] = None,
    height: int = 680,
) -> Optional[Dict[str, Any]]:
    """
    3D Design Studio — instant Three.js preview; returns a dict on user actions.

    Actions: ``save_sync``, ``run_rom``, ``run_fem`` (see ``index.html``).
    """
    return _fast_preview(initial=initial or {}, key=key, default=None, height=height)
