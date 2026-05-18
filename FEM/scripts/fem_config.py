# FEM/scripts/fem_config.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

# Central I/O roots (override with env SHARED_HOST_DIR).
try:
    from paths import (  # noqa: F401
        REPO_ROOT,
        SHARED_AUDIO_DIR,
        SHARED_EXPORTS_DIR,
        SHARED_HOST_DIR,
        resolve_shared_path,
        shared_audio_path,
    )
except ImportError:
    pass

try:
    from fem_api import (
        FemCase,
        RectPlateGeometry2D,
        OrthotropicPlateMaterial,
        BoundaryCondition,
        SolverSettings,
    )
except Exception:  # pragma: no cover
    from .fem_api import (  # type: ignore
        FemCase,
        RectPlateGeometry2D,
        OrthotropicPlateMaterial,
        BoundaryCondition,
        SolverSettings,
    )


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key '{key}' in config JSON.")
    return d[key]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two dicts:
    - returns a NEW dict
    - override wins over base
    - nested dicts are merged recursively
    """
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_library_path(lib_path: str | Path) -> Path:
    lib_path = Path(lib_path)
    if lib_path.is_absolute():
        return lib_path
    project_root = Path(__file__).resolve().parents[2]  # FEM/scripts -> project root
    return project_root / lib_path


def _apply_solution_ref(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    If solution_ref exists, load the profile from library and deep-merge:
      merged = deep_merge(profile, data)
    (data overrides profile)
    """
    sol_ref = data.get("solution_ref")
    if not sol_ref:
        return data

    if not isinstance(sol_ref, dict):
        raise ValueError("solution_ref must be an object: {library, name}")

    lib_path = _resolve_library_path(sol_ref.get("library", ""))
    name = sol_ref.get("name", "")
    if not str(name):
        raise ValueError("solution_ref.name is required")
    if not lib_path.exists():
        raise ValueError(f"solution_ref library not found: {lib_path}")

    lib = json.loads(lib_path.read_text(encoding="utf-8"))
    if name not in lib:
        raise ValueError(f"Unknown solution profile '{name}' in {lib_path}")

    profile = lib[name]
    if not isinstance(profile, dict):
        raise ValueError(f"Solution profile '{name}' must be a JSON object.")

    # data overrides profile
    merged = _deep_merge(profile, data)

    # remove solution_ref so it doesn't accidentally affect downstream logic
    merged.pop("solution_ref", None)
    return merged


def load_case_json(path: str | Path, solution_type: str = "", solution_types_file: str | Path = "FEM/solutions/solution_types.json") -> FemCase:
    """
    Load a FEM case from a JSON file and return FemCase.
    Supports:
    - material_ref (wood library)
    - solution_ref (solver+bc profile library)
    """
    path = Path(path)
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if solution_type:
        lib_path = _resolve_library_path(str(solution_types_file))
        lib = json.loads(lib_path.read_text(encoding="utf-8"))
        if solution_type not in lib:
            raise ValueError(f"Unknown solution_type '{solution_type}' in {lib_path}")
        profile = lib[solution_type]
        data = _deep_merge(data, profile)

        data.pop("solution_ref", None)
    # ---- NEW: solution_ref support (profile defaults for solver/bc/etc) ----
    data = _apply_solution_ref(data)

    # ---- material_ref support: load material from a library file ----
    material_ref = data.get("material_ref")
    if material_ref and "material_ortho" not in data:
        if not isinstance(material_ref, dict):
            raise ValueError("material_ref must be an object: {library, name}")

        lib_path = _resolve_library_path(material_ref["library"])
        lib = json.loads(lib_path.read_text(encoding="utf-8"))

        name = material_ref["name"]
        if name not in lib:
            raise ValueError(f"Unknown material '{name}' in {lib_path}")
        data["material_ortho"] = lib[name]

    solver_d = _require(data, "solver")
    solver = SolverSettings(
        dimension=int(_require(solver_d, "dimension")),
        model=str(_require(solver_d, "model")),
        n_modes=int(solver_d.get("n_modes", 20)),
        mesh_size=float(solver_d.get("mesh_size", 0.01)),
    )

    geom2d = None
    mat_ortho = None
    bc = None

    if solver.dimension == 2 and solver.model == "plate_ortho_kl_2d":
        g = _require(data, "geometry_2d")
        geom2d = RectPlateGeometry2D(
            a=float(_require(g, "a")),
            b=float(_require(g, "b")),
            h=float(_require(g, "h")),
        )

        m = _require(data, "material_ortho")
        mat_ortho = OrthotropicPlateMaterial(
            rho=float(_require(m, "rho")),
            E_L=float(_require(m, "E_L")),
            E_T=float(_require(m, "E_T")),
            G_LT=float(_require(m, "G_LT")),
            nu_LT=float(_require(m, "nu_LT")),
        )

        bcd = _require(data, "bc")
        bc = BoundaryCondition(kind=str(_require(bcd, "kind")))

    return FemCase(
        geometry_2d=geom2d,
        material_ortho=mat_ortho,
        bc=bc,
        solver=solver,
    )


def save_result_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_case_json(path: str | Path, case: FemCase) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(case)
    d = {k: v for k, v in d.items() if v is not None}
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
