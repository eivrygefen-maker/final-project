"""Streamlit custom component: ROM-aligned 3D Design Studio (Three.js)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_RELEASE = True

if _RELEASE:
    # Point directly to the folder containing index.html
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    _fast_preview = components.declare_component("fast_preview", path=parent_dir)
else:
    # Disable development server fallback
    _fast_preview = components.declare_component("fast_preview", url="http://localhost:3001")


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
