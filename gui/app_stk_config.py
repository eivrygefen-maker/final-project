#!/usr/bin/env python3
"""Load APP STK integration config with safe defaults."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "app_stk_config.json"

_DEFAULTS: Dict[str, Any] = {
    "instrument": "classical",
    "sample_rate": 44100,
    "default_duration_s": 4.5,
    "min_duration_s": 4.0,
    "low_note_duration_s": 5.0,
    "high_note_duration_s": 3.8,
    "fifo_max_guitars": 3,
    "enable_ready_fifo_stack": True,
    "enable_generate_intent": False,
    "enable_overlapping_playback": True,
    "user_show_stk_progress_detail": False,
    "debug_diagnostics_default": False,
    "render_mode": "parallel_batch",
    "parallel_workers": 3,
    "priority_notes": ["A2", "A4", "E5"],
    "fret_count": 19,
    "auto_refresh_interval_s": 12,
    "target_runtime_s": 180,
}


def load_app_stk_config(repo_root: Path | None = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    path = root / "config" / "app_stk_config.json"
    cfg = dict(_DEFAULTS)
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    cfg["priority_notes"] = list(cfg.get("priority_notes") or _DEFAULTS["priority_notes"])
    return cfg


def priority_notes_from_config(cfg: Dict[str, Any] | None = None) -> List[str]:
    return list((cfg or load_app_stk_config()).get("priority_notes") or _DEFAULTS["priority_notes"])
