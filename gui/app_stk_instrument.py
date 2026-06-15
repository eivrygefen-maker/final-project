#!/usr/bin/env python3
"""APP/STK instrument routing — shape/instrument namespaces and path helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

_INSTRUMENT_ALIASES = {
    "classical": "classical",
    "classic": "classical",
    "classical_guitar": "classical",
    "box": "box",
    "dreadnought": "classical",
}


def normalize_instrument(value: str) -> str:
    raw = str(value or "classical").strip().lower()
    return _INSTRUMENT_ALIASES.get(raw, raw if raw in ("classical", "box") else "classical")


def instrument_from_shape(shape_type: str) -> str:
    """Map UI shape label to APP/STK cache instrument key."""
    st = str(shape_type or "").strip().lower()
    if "box" in st:
        return "box"
    return "classical"


def rom_shape_namespace(shape_type: str) -> str:
    """Map UI shape label to ROM directory name (``ROM/<namespace>/``)."""
    st = str(shape_type or "").strip().lower()
    if "box" in st:
        return "box"
    if "dreadnought" in st:
        return "dreadnought"
    return "classic"


def lhs_pool_path(repo_root: Path, instrument: str = "classical") -> Path:
    inst = normalize_instrument(instrument)
    shape = "box" if inst == "box" else "classic"
    return Path(repo_root) / "ROM" / shape / "lhs_pool.json"


def default_sample_id(instrument: str = "classical") -> str:
    return "box_sample_000" if normalize_instrument(instrument) == "box" else "sample_000"


def reference_sample_id(instrument: str = "classical") -> str:
    """Reference sample for voicing / mix scaling within one instrument pool."""
    return default_sample_id(instrument)


def demo_version_label(instrument: str = "classical") -> str:
    inst = normalize_instrument(instrument)
    return f"app_stk_note_cache_{inst}"


def debug_reports_subdir(instrument: str = "classical") -> Path | None:
    """BOX uses ``audio/debug_reports/box/``; CLASSIC keeps flat layout."""
    if normalize_instrument(instrument) == "box":
        return Path("box")
    return None


def job_status_stem(instrument: str, parameter_hash: str) -> str:
    inst = normalize_instrument(instrument)
    if inst == "box":
        return f"app_stk_background_job_{inst}_{parameter_hash}"
    return f"app_stk_background_job_{parameter_hash}"


def background_status_stem(instrument: str, parameter_hash: str) -> str:
    inst = normalize_instrument(instrument)
    if inst == "box":
        return f"app_stk_background_status_{inst}_{parameter_hash}"
    return f"app_stk_background_status_{parameter_hash}"


def library_report_stem(instrument: str, parameter_hash: str) -> str:
    inst = normalize_instrument(instrument)
    return f"app_stk_note_library_{inst}_preview_{parameter_hash}"


def shared_shape_name(instrument: str = "classical") -> str:
    """Shared-host folder segment (lowercase per ``FEM/scripts/paths.py``)."""
    return "box" if normalize_instrument(instrument) == "box" else "classic"


def load_lhs_pool(repo_root: Path, instrument: str = "classical") -> Dict[str, Any]:
    path = lhs_pool_path(repo_root, instrument)
    if not path.is_file():
        return {}
    import json

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_lhs_sample_ids(repo_root: Path, instrument: str = "classical") -> list[str]:
    pool = load_lhs_pool(repo_root, instrument)
    inst = normalize_instrument(instrument)
    prefix = "box_sample_" if inst == "box" else "sample_"
    ids = [
        str(entry.get("id"))
        for entry in pool.get("entries") or []
        if str(entry.get("id", "")).startswith(prefix)
    ]
    return sorted(ids)


def lhs_entry_parameters(
    repo_root: Path,
    sample_id: str,
    instrument: str = "classical",
) -> Mapping[str, Any] | None:
    pool = load_lhs_pool(repo_root, instrument)
    for entry in pool.get("entries") or []:
        if str(entry.get("id")) == sample_id:
            params = entry.get("parameters")
            return params if isinstance(params, dict) else None
    return None
