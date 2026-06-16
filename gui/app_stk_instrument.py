#!/usr/bin/env python3
"""Shape/ROM path helpers — one unified STK renderer; inputs vary by shape."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def rom_shape_namespace(shape_type: str) -> str:
    """Map UI shape label to ROM directory name (``ROM/<namespace>/``)."""
    st = str(shape_type or "").strip().lower()
    if "box" in st:
        return "box"
    if "dreadnought" in st:
        return "dreadnought"
    return "classic"


def lhs_pool_path(repo_root: Path, shape_type: str = "Classical") -> Path:
    ns = rom_shape_namespace(shape_type)
    return Path(repo_root) / "ROM" / ns / "lhs_pool.json"


def default_sample_id_for_shape(shape_type: str = "Classical") -> str:
    if "box" in str(shape_type or "").strip().lower():
        return "box_sample_000"
    return "sample_000"


def reference_sample_id_for(sample_id: str) -> str:
    """Reference sample for voicing / mix scaling within one LHS pool."""
    if str(sample_id).startswith("box_sample_"):
        return "box_sample_000"
    return "sample_000"


def shape_type_label_from_sample_id(sample_id: str) -> str:
    if str(sample_id).startswith("box_sample_"):
        return "box"
    return "classic"


def load_lhs_pool(repo_root: Path, shape_type: str = "Classical") -> Dict[str, Any]:
    path = lhs_pool_path(repo_root, shape_type)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_lhs_sample_ids(repo_root: Path, shape_type: str = "Classical") -> list[str]:
    pool = load_lhs_pool(repo_root, shape_type)
    ns = rom_shape_namespace(shape_type)
    prefix = "box_sample_" if ns == "box" else "sample_"
    ids = [
        str(entry.get("id"))
        for entry in pool.get("entries") or []
        if str(entry.get("id", "")).startswith(prefix)
    ]
    return sorted(ids)


def lhs_entry_parameters(
    repo_root: Path,
    sample_id: str,
    shape_type: str | None = None,
) -> Mapping[str, Any] | None:
    if shape_type is None:
        shape_type = "Box" if str(sample_id).startswith("box_sample_") else "Classical"
    pool = load_lhs_pool(repo_root, shape_type)
    for entry in pool.get("entries") or []:
        if str(entry.get("id")) == sample_id:
            params = entry.get("parameters")
            return params if isinstance(params, dict) else None
    return None
