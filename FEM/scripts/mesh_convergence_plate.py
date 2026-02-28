# FEM/scripts/mesh_convergence_plate.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import your solver + dataclasses
try:
    from solver_2d_plate import solve_plate_ortho_kl_2d  # type: ignore
    from fem_api import RectPlateGeometry2D, OrthotropicPlateMaterial, BoundaryCondition  # type: ignore
except Exception:
    # when running as module: python -m FEM.scripts.mesh_convergence_plate ...
    from .solver_2d_plate import solve_plate_ortho_kl_2d  # type: ignore
    from .fem_api import RectPlateGeometry2D, OrthotropicPlateMaterial, BoundaryCondition  # type: ignore


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _load_material_from_case(case: dict, cfg_path) -> dict:
    """
    Load orthotropic plate material parameters from a config dict.

    Supported sources:
      - material_ortho: {rho,E_L,E_T,G_LT,nu_LT}
      - material:       {rho,E_L,E_T,G_LT,nu_LT}
      - material_file:  "path/to/material.json"
      - material_id:    key inside FEM/materials/woods_ortho.json
      - material_ref:   alias for material_id (string or dict with id/key/name)
    """
    import json
    from pathlib import Path

    def _coerce_inline(m: dict) -> dict | None:
        if not isinstance(m, dict):
            return None

        # tolerate nuLT alias
        if "nu_LT" not in m and "nuLT" in m:
            m = dict(m)
            m["nu_LT"] = m["nuLT"]

        needed = ("rho", "E_L", "E_T", "G_LT", "nu_LT")
        if not all(k in m for k in needed):
            return None

        return {
            "rho": float(m["rho"]),
            "E_L": float(m["E_L"]),
            "E_T": float(m["E_T"]),
            "G_LT": float(m["G_LT"]),
            "nu_LT": float(m["nu_LT"]),
        }

    cfg_path = Path(str(cfg_path)).resolve()
    project_root = cfg_path.parents[2]  # .../final-project/FEM/configs/<file>.json -> final-project

    # ---- 1) inline top-level ----
    m = _coerce_inline(case.get("material_ortho"))
    if m is not None:
        return m
    m = _coerce_inline(case.get("material"))
    if m is not None:
        return m

    # ---- 2) inline nested (common containers) ----
    def _try_inline_in(d: dict) -> dict | None:
        if not isinstance(d, dict):
            return None
        m2 = _coerce_inline(d.get("material_ortho"))
        if m2 is not None:
            return m2
        m2 = _coerce_inline(d.get("material"))
        if m2 is not None:
            return m2
        return None

    for key in ("case", "fem", "config", "inputs"):
        if isinstance(case.get(key), dict):
            m = _try_inline_in(case[key])
            if m is not None:
                return m

    if isinstance(case.get("steps"), dict) and isinstance(case["steps"].get("fem"), dict):
        m = _try_inline_in(case["steps"]["fem"])
        if m is not None:
            return m

    # ---- 3) material_file ----
    material_file = case.get("material_file")
    if not material_file:
        # try nested
        for key in ("case", "fem"):
            if isinstance(case.get(key), dict) and case[key].get("material_file"):
                material_file = case[key]["material_file"]
                break

    if material_file:
        p = Path(str(material_file))
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if not p.exists():
            raise ValueError(f"material_file not found: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))
        # allow either {"material_ortho": {...}} or direct dict
        if isinstance(data, dict) and "material_ortho" in data:
            m = _coerce_inline(data["material_ortho"])
            if m is not None:
                return m
        if isinstance(data, dict) and "material" in data:
            m = _coerce_inline(data["material"])
            if m is not None:
                return m

        m = _coerce_inline(data) if isinstance(data, dict) else None
        if m is not None:
            return m

        raise ValueError(f"material_file JSON has no usable material fields: {p}")

    # ---- 4) material_id OR material_ref (alias) ----
    material_id = case.get("material_id")

    # accept your current configs
    if material_id is None and "material_ref" in case:
        ref = case.get("material_ref")
        if isinstance(ref, str):
            material_id = ref
        elif isinstance(ref, dict):
            material_id = ref.get("id") or ref.get("key") or ref.get("name")

    # nested alias too (just in case)
    if material_id is None:
        for key in ("case", "fem"):
            if isinstance(case.get(key), dict):
                d = case[key]
                if d.get("material_id"):
                    material_id = d["material_id"]
                    break
                if d.get("material_ref"):
                    ref = d["material_ref"]
                    if isinstance(ref, str):
                        material_id = ref
                        break
                    if isinstance(ref, dict):
                        material_id = ref.get("id") or ref.get("key") or ref.get("name")
                        break

    if material_id:
        db_path = (project_root / "FEM" / "materials" / "woods_ortho.json").resolve()
        if not db_path.exists():
            raise ValueError(f"materials database not found: {db_path}")

        db = json.loads(db_path.read_text(encoding="utf-8"))
        if not isinstance(db, dict):
            raise ValueError(f"woods_ortho.json must be a JSON object (dict). File: {db_path}")

        key = str(material_id)
        if key not in db:
            raise ValueError(f"material_id '{key}' not found in {db_path}. Available: {list(db.keys())}")

        m = _coerce_inline(db[key])
        if m is None:
            raise ValueError(f"Material entry '{key}' in {db_path} is missing required fields.")
        return m

    # ---- fail ----
    raise ValueError(
        "Could not load material. The config dict does not contain material info in supported places.\n"
        "Provide either:\n"
        "- material_ortho: {rho,E_L,E_T,G_LT,nu_LT}\n"
        "- material: {rho,E_L,E_T,G_LT,nu_LT}\n"
        "- material_file: path/to/material.json\n"
        "- material_id: key inside FEM/materials/woods_ortho.json\n"
        "- material_ref: alias for material_id\n"
        f"Top-level keys seen: {list(case.keys())}"
    )



def _load_geometry_from_case(case: dict) -> dict:
    # Accept multiple schema variants
    if isinstance(case.get("geometry_2d"), dict):
        g = case["geometry_2d"]
        if all(k in g for k in ("a", "b", "h")):
            return {"a": float(g["a"]), "b": float(g["b"]), "h": float(g["h"])}

    if isinstance(case.get("geometry"), dict):
        g = case["geometry"]
        if all(k in g for k in ("a", "b", "h")):
            return {"a": float(g["a"]), "b": float(g["b"]), "h": float(g["h"])}

    # Root-level fallback
    if all(k in case for k in ("a", "b", "h")):
        return {"a": float(case["a"]), "b": float(case["b"]), "h": float(case["h"])}

    raise ValueError("Could not load geometry. Expected geometry_2d:{a,b,h} or geometry:{a,b,h} or root a,b,h.")


def _load_bc_from_case(case: Dict[str, Any]) -> str:
    # you can extend this later
    bc = case.get("bc")
    if isinstance(bc, dict) and "kind" in bc:
        return str(bc["kind"])
    return str(case.get("bc_kind", "SSSS"))


def _estimate_grid(a: float, b: float, mesh_size: float) -> Tuple[int, int, int]:
    # Must match solver_2d_plate.py logic (round + +1)
    nx = max(6, int(round(a / mesh_size)) + 1)
    ny = max(6, int(round(b / mesh_size)) + 1)
    nix = nx - 2
    niy = ny - 2
    ndof = max(0, nix * niy)
    return nx, ny, ndof


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mesh convergence test for the 2D orthotropic plate solver."
    )
    ap.add_argument("--config", required=True, help="Path to FEM case config JSON (rect plate).")
    ap.add_argument(
        "--mesh",
        nargs="+",
        type=float,
        default=[0.05, 0.025, 0.0125, 0.00625],
        help="List of mesh_size values to test (smaller => finer).",
    )
    ap.add_argument("--modes", type=int, default=4, help="How many lowest modes to print/compare.")
    ap.add_argument("--save", default="", help="Optional output JSON path to save the sweep results.")
    ap.add_argument(
        "--analytic",
        default="",
        help="Optional JSON file with analytic freqs list: {\"freqs_hz\": [..]}.",
    )

    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    case = _load_json(cfg_path)

    geom_d = _load_geometry_from_case(case)
    mat_d = _load_material_from_case(case, cfg_path)
    bc_kind = _load_bc_from_case(case)

    geom = RectPlateGeometry2D(a=float(geom_d["a"]), b=float(geom_d["b"]), h=float(geom_d["h"]))
    mat = OrthotropicPlateMaterial(
        E_L=float(mat_d["E_L"]),
        E_T=float(mat_d["E_T"]),
        G_LT=float(mat_d["G_LT"]),
        nu_LT=float(mat_d["nu_LT"]),
        rho=float(mat_d["rho"]),
    )
    bc = BoundaryCondition(kind=bc_kind)

    # Optional analytic values (first K modes, sorted)
    analytic_freqs: Optional[List[float]] = None
    if args.analytic:
        a_path = Path(args.analytic).resolve()
        a_data = _load_json(a_path)
        freqs = a_data.get("freqs_hz")
        if isinstance(freqs, list) and all(isinstance(x, (int, float)) for x in freqs):
            analytic_freqs = [float(x) for x in freqs]

    # If you want a built-in analytic reference for the classic plate test, uncomment:
    # analytic_freqs = [23.41, 49.85, 76.05, 95.50]  # only if your case is exactly that reference

    mesh_list = [float(x) for x in args.mesh]
    mesh_list = sorted(mesh_list, reverse=True)  # coarse -> fine for nicer convergence view

    rows: List[Dict[str, Any]] = []
    prev: Optional[np.ndarray] = None

    print("\n=== Mesh convergence sweep (2D plate) ===")
    print(f"Config: {cfg_path}")
    print(f"Geometry: a={geom.a} m, b={geom.b} m, h={geom.h} m")
    print(f"Material: rho={mat.rho}, E_L={mat.E_L}, E_T={mat.E_T}, G_LT={mat.G_LT}, nu_LT={mat.nu_LT}")
    print(f"BC: {bc.kind}")
    print(f"Requested modes: {args.modes}\n")

    header = ["mesh_size", "nx", "ny", "ndof"] + [f"f{i+1}" for i in range(args.modes)]
    if analytic_freqs:
        header += [f"err%_f{i+1}" for i in range(min(args.modes, len(analytic_freqs)))]
    header += [f"delta%_vs_prev_f{i+1}" for i in range(args.modes)]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header)) + 2))

    for ms in mesh_list:
        nx, ny, ndof = _estimate_grid(geom.a, geom.b, ms)

        res = solve_plate_ortho_kl_2d(
            geom=geom,
            mat=mat,
            bc=bc,
            n_modes=max(args.modes, 10),
            mesh_size=ms,
        )

        freqs = np.array(res.modes[: args.modes], dtype=float)
        deltas = np.full(args.modes, np.nan)
        if prev is not None:
            # relative change vs previous (coarser) run
            for i in range(args.modes):
                if i < len(freqs) and i < len(prev) and freqs[i] != 0:
                    deltas[i] = abs(freqs[i] - prev[i]) / abs(freqs[i]) * 100.0
        prev = freqs.copy()

        row: Dict[str, Any] = {
            "mesh_size": ms,
            "nx": nx,
            "ny": ny,
            "ndof": ndof,
            "freqs_hz": freqs.tolist(),
            "delta_vs_prev_percent": deltas.tolist(),
        }

        # analytic error (if provided)
        err_list: List[float] = []
        if analytic_freqs:
            K = min(args.modes, len(analytic_freqs))
            for i in range(K):
                ana = analytic_freqs[i]
                if ana != 0:
                    err_list.append(abs(freqs[i] - ana) / abs(ana) * 100.0)
                else:
                    err_list.append(float("nan"))
            row["analytic_freqs_hz"] = analytic_freqs[:K]
            row["analytic_err_percent"] = err_list

        rows.append(row)

        line_parts = [f"{ms:g}", str(nx), str(ny), str(ndof)]
        line_parts += [f"{f:.4f}" if np.isfinite(f) else "nan" for f in freqs.tolist()]

        if analytic_freqs:
            line_parts += [f"{e:.2f}" if np.isfinite(e) else "nan" for e in err_list]

        line_parts += [f"{d:.3f}" if np.isfinite(d) else "nan" for d in deltas.tolist()]
        print(" | ".join(line_parts))

    if args.save:
        out_path = Path(args.save).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(cfg_path),
            "geometry": asdict(geom),
            "material": asdict(mat),
            "bc": asdict(bc),
            "modes_printed": args.modes,
            "mesh_sizes": mesh_list,
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved sweep JSON: {out_path}")

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
