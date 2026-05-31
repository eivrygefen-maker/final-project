#!/usr/bin/env python3
"""Minimal PETSc helpers with no DOLFINx/FEM dependencies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return len(text.encode("utf-8"))


def mat_shape(mat: Any) -> Any:
    if mat is None:
        return None
    try:
        n_rows, n_cols = mat.getSize()
        return [int(n_rows), int(n_cols)]
    except Exception:
        return None


def petsc_mat_try_assemble(mat: Any) -> bool:
    try:
        mat.assemble()
        return True
    except Exception:
        return False
