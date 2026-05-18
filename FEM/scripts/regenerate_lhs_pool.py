#!/usr/bin/env python3
"""Regenerate ``ROM/<shape>/lhs_pool.json`` with unrestricted 5×5 wood LHS (no mpi4py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from paths import REPO_ROOT
from wood_library import ALL_WOOD_IDS


def _shape_length_width_depth_bounds(shape_type: str) -> Dict[str, Dict[str, float]]:
    st = str(shape_type).lower()
    if "dreadnought" in st:
        return {
            "geometry.length": {"min": 0.45, "max": 0.70},
            "geometry.width": {"min": 0.30, "max": 0.55},
            "geometry.depth": {"min": 0.10, "max": 0.20},
        }
    if "box" in st:
        return {
            "geometry.length": {"min": 0.10, "max": 1.00},
            "geometry.width": {"min": 0.10, "max": 0.80},
            "geometry.depth": {"min": 0.01, "max": 0.50},
        }
    return {
        "geometry.length": {"min": 0.35, "max": 0.60},
        "geometry.width": {"min": 0.20, "max": 0.45},
        "geometry.depth": {"min": 0.08, "max": 0.15},
    }


def build_7d_lhs_sweep_spec(base_cfg: Dict[str, Any], sweep_cfg: Dict) -> Dict:
    shape_type = str(base_cfg.get("geometry", {}).get("shape_type", "Classical"))
    bounds = _shape_length_width_depth_bounds(shape_type)
    wood_options = list(ALL_WOOD_IDS)
    spec = {
        "geometry.length": bounds["geometry.length"],
        "geometry.width": bounds["geometry.width"],
        "geometry.depth": bounds["geometry.depth"],
        "geometry.thickness": {"min": 0.002, "max": 0.006},
        "geometry.hole_radius": {"min": 0.035, "max": 0.055},
        "top_wood_id": wood_options,
        "back_wood_id": wood_options,
    }
    for key in list(spec.keys()):
        if key in sweep_cfg:
            spec[key] = sweep_cfg[key]
    return spec


def _expand_parameter_values(spec: Any) -> List:
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict) and "values" in spec:
        return list(spec["values"])
    raise ValueError(f"Unsupported sweep spec: {spec!r}")


def _lhs_values_for_key(spec: Any, uvals: np.ndarray) -> List:
    if isinstance(spec, dict) and "min" in spec and "max" in spec:
        vmin = float(spec["min"])
        vmax = float(spec["max"])
        vals = vmin + (vmax - vmin) * uvals
        dtype = str(spec.get("dtype", "float")).lower()
        if dtype in ("int", "integer"):
            vals = np.rint(vals).astype(np.int64)
        return vals.tolist()
    options = _expand_parameter_values(spec)
    if not options:
        raise ValueError("Sweep options list cannot be empty.")
    idx = np.floor(uvals * len(options)).astype(int)
    idx = np.clip(idx, 0, len(options) - 1)
    return [options[int(i)] for i in idx]


def _latin_hypercube_unit(n_samples: int, n_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.stack([rng.permutation(n_samples) for _ in range(n_dim)], axis=0)
    u = (perm + rng.random((n_dim, n_samples))) / n_samples
    return u.T


def create_lhs_pool(
    shape_name: str,
    sweep_cfg: Dict,
    total_samples: int,
    seed: int = 123,
) -> Dict:
    keys = sorted(sweep_cfg.keys())
    unit = _latin_hypercube_unit(total_samples, len(keys), seed)
    cols = {k: _lhs_values_for_key(sweep_cfg[k], unit[:, i]) for i, k in enumerate(keys)}
    entries = []
    for i in range(total_samples):
        params = {k: cols[k][i] for k in keys}
        entries.append(
            {
                "id": f"sample_{i + 1:03d}",
                "parameters": params,
                "status": "pending",
                "snapshot_file": None,
                "error": None,
            }
        )
    return {
        "shape_name": shape_name,
        "sampling": "lhs",
        "wood_assignment": "unrestricted_5x5",
        "seed": int(seed),
        "total_samples": int(total_samples),
        "mpi_world_size": 0,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="classic")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    shapes_path = REPO_ROOT / "FEM" / "configs" / "rom_shapes.json"
    with open(shapes_path, "r", encoding="utf-8") as f:
        shapes = json.load(f)["shapes"]
    shape_cfg = shapes[args.shape]
    base_path = REPO_ROOT / shape_cfg["base_config"]
    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    sweep_cfg = build_7d_lhs_sweep_spec(base_cfg, shape_cfg.get("parameter_sweep", {}))
    pool = create_lhs_pool(args.shape, sweep_cfg, args.samples, seed=args.seed)

    out = REPO_ROOT / "ROM" / args.shape / "lhs_pool.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)
    tmp.replace(out)

    same = sum(
        1
        for e in pool["entries"]
        if e["parameters"]["top_wood_id"] == e["parameters"]["back_wood_id"]
    )
    print(f"Wrote {out} ({pool['total_samples']} samples, {same} identical top/back pairs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
