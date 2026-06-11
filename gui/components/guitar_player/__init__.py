"""Streamlit custom component: interactive fretboard note player (Web Audio, no reruns)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_RELEASE = True

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_INDEX_HTML = os.path.join(_COMPONENT_DIR, "index.html")
_BRIDGE_JS = os.path.join(_COMPONENT_DIR, "streamlit_bridge.js")

for _label, _path in (("index.html", _INDEX_HTML), ("streamlit_bridge.js", _BRIDGE_JS)):
    if not os.path.isfile(_path):
        raise FileNotFoundError(
            f"guitar_player component: {_label} not found at {_path}. "
            f"Component directory: {_COMPONENT_DIR}"
        )

if _RELEASE:
    _guitar_player = components.declare_component(
        "guitar_player",
        path=_COMPONENT_DIR,
    )
else:
    _guitar_player = components.declare_component(
        "guitar_player",
        url="http://localhost:3002",
    )


def guitar_player(
    *,
    player: Optional[Dict[str, Any]] = None,
    key: Optional[str] = None,
    height: int = 520,
) -> None:
    """
    Interactive guitar fretboard — plays preloaded WAVs in-browser (no Streamlit reruns).

    ``player`` payload from ``note_cache_ui.build_player_payload``; status ``hidden`` renders idle UI.
    """
    _guitar_player(
        player=player or {"status": "hidden"},
        key=key,
        default=None,
        height=height,
    )


def component_dir() -> str:
    return _COMPONENT_DIR
