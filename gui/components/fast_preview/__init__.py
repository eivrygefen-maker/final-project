"""Streamlit custom component: ROM-aligned 3D Design Studio (Three.js)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

# Production: serve static index.html from this directory (never localhost:3001).
_RELEASE = True

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_INDEX_HTML = os.path.join(_COMPONENT_DIR, "index.html")
_BRIDGE_JS = os.path.join(_COMPONENT_DIR, "streamlit_bridge.js")

for _label, _path in (("index.html", _INDEX_HTML), ("streamlit_bridge.js", _BRIDGE_JS)):
    if not os.path.isfile(_path):
        raise FileNotFoundError(
            f"fast_preview component: {_label} not found at {_path}. "
            f"Component directory: {_COMPONENT_DIR}"
        )

if _RELEASE:
    _fast_preview = components.declare_component(
        "fast_preview",
        path=_COMPONENT_DIR,
    )
else:
    _fast_preview = components.declare_component(
        "fast_preview",
        url="http://localhost:3001",
    )


def fast_preview(
    *,
    initial: Optional[Dict[str, Any]] = None,
    key: Optional[str] = None,
    height: int = 720,
) -> Optional[Dict[str, Any]]:
    """
    3D Design Studio — instant Three.js preview; returns a dict on user actions.

    Actions: ``save_sync``, ``run_rom``, ``run_fem`` (see ``index.html``).
    """
    return _fast_preview(initial=initial or {}, key=key, default=None, height=height)
