#!/usr/bin/env python3
"""Build full W-space EPS seed from acoustic-only locator pressure (experiment-only)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from v2_sensitivity_locator import pick_locator_frequency_hz
from v2_sensitivity_mesh import sample_geometry
from wood_library import apply_wood_ids_to_config

V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
ACOUSTIC_REFERENCE_SOURCE = "acoustic_only_locator_eigenvector"

MAP_KEYS = (
    "_coupled_air_u_to_W_map",
    "_coupled_air_p_to_W_map",
    "_coupled_air_pressure_restriction",
    "_coupled_air_p_air_collapsed_indices",
)


def _map_crc32(arr: np.ndarray) -> int:
    return int(zlib.crc32(np.asarray(arr, dtype=np.int32).tobytes()) & 0xFFFFFFFF)


def _validate_index_map(name: str, idx: np.ndarray, *, n_reduced_W: int) -> Dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int32).ravel()
    if idx.size == 0:
        return {
            "name": name,
            "length": 0,
            "min": None,
            "max": None,
            "bounds_valid": False,
            "crc32": _map_crc32(idx),
        }
    lo = int(idx.min())
    hi = int(idx.max())
    return {
        "name": name,
        "length": int(idx.size),
        "min": lo,
        "max": hi,
        "bounds_valid": bool(lo >= 0 and hi < int(n_reduced_W)),
        "crc32": _map_crc32(idx),
    }


def _assemble_reduced_coupled_replay(
    mesh_path: Path,
    sample: Dict[str, Any],
    *,
    coupling_enabled: bool = True,
    capture_parent_raw_blocks: bool = False,
    operator_build_profile: Any = None,
    core_config_path: Optional[Path] = None,
) -> Tuple[Any, Any, dict]:
    """Proven v2 post replay: solve_evp=False + pressure-restriction replay audit."""
    prof = operator_build_profile
    if prof is not None:
        prof.begin_replay()
        prof.attach_fem3d()
    try:
        config_source = (
            Path(core_config_path).expanduser().resolve()
            if core_config_path is not None
            else V2_CONFIG
        )
        cfg = copy.deepcopy(json.loads(config_source.read_text(encoding="utf-8")))
        sc = cfg.setdefault("solver", {})
        sc["mesh_file"] = str(mesh_path.resolve())
        sc["coupled_physical_core_v2_diagnosis"] = True
        sc["coupled_physical_core_v2_coupling_enabled"] = bool(coupling_enabled)
        sc["fsi_coupling_gain"] = 1.0
        sc["fsi_nitsche_enable"] = False
        sc["physics_integrity_capture"] = True
        sc["coupled_air_pressure_restriction_diagnosis"] = True
        sc["coupled_air_pressure_restriction_replay_audit"] = True
        sc["gnhep_block_frobenius_normalize"] = True
        sc["b3_raw_parent_block_capture_no_eps_diagnostic"] = bool(capture_parent_raw_blocks)
        if core_config_path is None:
            cfg["geometry"] = sample_geometry(sample)
            mats = sample.get("materials") or {}
            if mats.get("top_wood_id") or mats.get("back_wood_id"):
                apply_wood_ids_to_config(
                    cfg,
                    top_wood_id=mats.get("top_wood_id"),
                    back_wood_id=mats.get("back_wood_id"),
                )

        _msh, _W, A, M = fem3d._solve_coupled_evp(
            mesh_file=mesh_path.resolve(),
            config=cfg,
            num_modes=0,
            solve_evp=False,
        )
        missing = [k for k in MAP_KEYS if k not in cfg]
        if missing:
            try:
                A.destroy()
                M.destroy()
            except Exception:
                pass
            raise RuntimeError(
                "reduced replay assembly did not populate required maps. "
                f"Missing config keys: {missing}. "
                "Ensure coupled_air_pressure_restriction_replay_audit=True and "
                "coupled_air_pressure_restriction_diagnosis=True with solve_evp=False."
            )
        return A, M, cfg
    finally:
        if prof is not None:
            prof.end_replay()
            prof.detach_fem3d()


def _extract_parent_raw_block_capture() -> Dict[str, Any]:
    cap = fem3d.get_last_coupled_raw_block_capture()
    return dict(cap) if isinstance(cap, dict) else {}


def _extract_layout_maps(cfg: dict, A: Any) -> Dict[str, Any]:
    restr = dict(cfg.get("_coupled_air_pressure_restriction") or {})
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    p_air_collapsed = np.asarray(
        cfg["_coupled_air_p_air_collapsed_indices"], dtype=np.int32
    ).ravel()
    n_W = int(A.getSize()[0])
    n_u = int(restr.get("n_u_active", u_to_W.size))
    n_p = int(restr.get("n_p_active", p_to_W.size))
    n_p_full = int(restr.get("n_p_full_collapsed", p_air_collapsed.max() + 1 if p_air_collapsed.size else 0))
    layout = {
        "config_map_keys_populated": list(MAP_KEYS),
        "n_reduced_W": n_W,
        "n_u_active": n_u,
        "n_p_active": n_p,
        "n_p_full_collapsed": n_p_full,
        "len_u_to_W": int(u_to_W.size),
        "len_p_to_W": int(p_to_W.size),
        "len_p_air_collapsed_indices": int(p_air_collapsed.size),
        "u_to_W_bounds": _validate_index_map("u_to_W", u_to_W, n_reduced_W=n_W),
        "p_to_W_bounds": _validate_index_map("p_to_W", p_to_W, n_reduced_W=n_W),
        "p_air_collapsed_bounds": {
            "length": int(p_air_collapsed.size),
            "min": int(p_air_collapsed.min()) if p_air_collapsed.size else None,
            "max": int(p_air_collapsed.max()) if p_air_collapsed.size else None,
            "crc32": _map_crc32(p_air_collapsed),
        },
        "replay_flags": {
            "solve_evp": False,
            "coupled_air_pressure_restriction_diagnosis": True,
            "coupled_air_pressure_restriction_replay_audit": True,
            "coupled_physical_core_v2_diagnosis": True,
        },
    }
    if int(p_to_W.size) != int(p_air_collapsed.size):
        layout["layout_warning"] = (
            f"len(p_to_W)={p_to_W.size} != len(p_air_collapsed_indices)={p_air_collapsed.size}"
        )
    if not layout["u_to_W_bounds"]["bounds_valid"] or not layout["p_to_W_bounds"]["bounds_valid"]:
        raise RuntimeError(f"reduced map indices out of bounds for n_reduced_W={n_W}: {layout}")
    return {
        "u_to_W": u_to_W,
        "p_to_W": p_to_W,
        "p_air_collapsed": p_air_collapsed,
        "restr": restr,
        "layout": layout,
    }


def _restrict_full_collapsed_pressure(
    p_full: np.ndarray,
    p_air_collapsed: np.ndarray,
    *,
    n_p_full_expected: int,
) -> np.ndarray:
    p_full = np.asarray(p_full, dtype=np.float64).ravel()
    p_air_collapsed = np.asarray(p_air_collapsed, dtype=np.int32).ravel()
    if p_full.size < int(n_p_full_expected):
        raise RuntimeError(
            f"archived locator pressure length {p_full.size} < "
            f"n_p_full_collapsed={n_p_full_expected}"
        )
    if p_air_collapsed.size == 0:
        raise RuntimeError("active-air collapsed pressure index map is empty")
    hi = int(p_air_collapsed.max())
    if hi >= p_full.size:
        raise RuntimeError(
            f"p_air_collapsed max index {hi} >= archived pressure length {p_full.size}"
        )
    return p_full[p_air_collapsed]


def _embed_active_pressure_seed(
    p_active: np.ndarray,
    *,
    n_reduced_W: int,
    p_to_W: np.ndarray,
) -> np.ndarray:
    p_active = np.asarray(p_active, dtype=np.float64).ravel()
    p_to_W = np.asarray(p_to_W, dtype=np.int32).ravel()
    if p_active.size != p_to_W.size:
        raise RuntimeError(
            f"active pressure vector length {p_active.size} != len(p_to_W) {p_to_W.size}"
        )
    seed = np.zeros(int(n_reduced_W), dtype=np.float64)
    for k in range(int(p_to_W.size)):
        seed[int(p_to_W[k])] = float(p_active[k])
    norm = float(np.linalg.norm(seed))
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("acoustic seed has zero or non-finite norm after embedding")
    return seed / norm


def _solve_acoustic_cavity_mode(
    mesh_path: Path,
    sample: Dict[str, Any],
    *,
    locator_hz: float,
    locator_lo_hz: float,
    locator_hi_hz: float,
    num_modes: int,
) -> Tuple[np.ndarray, float, int, List[float]]:
    cfg_ac = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    sc_ac = cfg_ac.setdefault("solver", {})
    sc_ac["mesh_file"] = str(mesh_path.resolve())
    sc_ac["acoustic_cavity_only_diagnosis"] = True
    sc_ac["couple_fluid"] = False
    sc_ac["structural_only_diagnosis"] = False
    sc_ac["coupled_physical_core_v2_diagnosis"] = False
    sc_ac["physics_integrity_capture"] = False
    sc_ac["acoustic_cavity_num_modes"] = max(4, int(num_modes))
    sc_ac["acoustic_min_mode_hz"] = float(locator_lo_hz)
    sc_ac["acoustic_max_mode_hz"] = float(locator_hi_hz)
    sc_ac["acoustic_shift_target_hz"] = float(locator_hz)
    cfg_ac["geometry"] = sample_geometry(sample)

    _m1, _V1, freqs_hz, eig_ac, _nu, n_p_ac = fem3d._solve_coupled_evp(
        mesh_file=mesh_path.resolve(), config=cfg_ac, num_modes=int(num_modes)
    )
    if int(n_p_ac) <= 0 or eig_ac.size == 0:
        raise RuntimeError("acoustic cavity solve produced no pressure modes for seed")

    loc_hz, _sel = pick_locator_frequency_hz(
        list(freqs_hz),
        band_lo=float(locator_lo_hz),
        band_hi=float(locator_hi_hz),
        reference_hz=float(locator_hz),
    )
    if not math.isfinite(loc_hz):
        raise RuntimeError("cannot pick locator frequency for seed")
    j = int(np.argmin([abs(float(f) - loc_hz) for f in freqs_hz]))
    p_mode = np.asarray(eig_ac[:, j], dtype=np.float64).ravel()
    return p_mode, float(loc_hz), j, list(freqs_hz)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed acoustic locator pressure into coupled W seed")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--locator-hz", type=float, required=True)
    parser.add_argument("--locator-lo-hz", type=float, default=150.0)
    parser.add_argument("--locator-hi-hz", type=float, default=350.0)
    parser.add_argument("--reference-hz", type=float, default=244.394153389752)
    parser.add_argument("--num-modes", type=int, default=24)
    parser.add_argument("--out-npy", type=Path, required=True)
    parser.add_argument("--out-meta-json", type=Path, required=True)
    parser.add_argument(
        "--archived-pressure-npy",
        type=Path,
        default=None,
        help="Archived acoustic-only locator vector in full collapsed pressure layout.",
    )
    parser.add_argument(
        "--out-pressure-npy",
        type=Path,
        default=None,
        help="Optional write of active-air restricted pressure (n_p_active).",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        return 2

    sample = json.loads(args.sample_json.read_text(encoding="utf-8"))
    mesh_path = args.mesh.resolve()
    loc_hz = float(args.locator_hz)
    picked_j = -1
    picked_f_hz = float("nan")

    A, M, cfg = _assemble_reduced_coupled_replay(mesh_path, sample, coupling_enabled=True)
    try:
        maps = _extract_layout_maps(cfg, A)
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    layout = maps["layout"]
    u_to_W = maps["u_to_W"]
    p_to_W = maps["p_to_W"]
    p_air_collapsed = maps["p_air_collapsed"]
    restr = maps["restr"]
    n_W = int(layout["n_reduced_W"])
    n_p_full = int(layout["n_p_full_collapsed"])

    if args.archived_pressure_npy is not None and args.archived_pressure_npy.is_file():
        p_full = np.load(str(args.archived_pressure_npy.resolve()))
        pressure_source = "archived_acoustic_only_locator_eigenvector_npy"
    else:
        p_mode, loc_hz, picked_j, freqs_hz = _solve_acoustic_cavity_mode(
            mesh_path,
            sample,
            locator_hz=loc_hz,
            locator_lo_hz=float(args.locator_lo_hz),
            locator_hi_hz=float(args.locator_hi_hz),
            num_modes=int(args.num_modes),
        )
        picked_f_hz = float(freqs_hz[picked_j]) if picked_j >= 0 else float("nan")
        if int(p_mode.size) == n_p_full:
            p_full = p_mode
            pressure_source = "acoustic_cavity_solve_full_collapsed"
        elif int(p_mode.size) == int(p_to_W.size):
            p_full = np.zeros(n_p_full, dtype=np.float64)
            p_full[p_air_collapsed] = p_mode
            pressure_source = "acoustic_cavity_solve_active_subset_padded_to_full_collapsed"
        else:
            raise RuntimeError(
                f"acoustic cavity mode length {p_mode.size} neither "
                f"n_p_full_collapsed={n_p_full} nor n_p_active={p_to_W.size}"
            )

    locator_pressure_full_length = int(np.asarray(p_full).size)
    p_active = _restrict_full_collapsed_pressure(
        p_full, p_air_collapsed, n_p_full_expected=n_p_full
    )
    seed = _embed_active_pressure_seed(p_active, n_reduced_W=n_W, p_to_W=p_to_W)

    seed_norm = float(np.linalg.norm(seed))
    seed_layout_valid = (
        int(seed.size) == n_W
        and seed_norm > 0.0
        and math.isfinite(seed_norm)
        and bool(layout["u_to_W_bounds"]["bounds_valid"])
        and bool(layout["p_to_W_bounds"]["bounds_valid"])
    )

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out_npy.resolve()), seed)

    p_active_norm = float(np.linalg.norm(p_active))
    p_active_out = p_active / p_active_norm if p_active_norm > 0 else p_active
    if args.out_pressure_npy is not None:
        args.out_pressure_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.out_pressure_npy.resolve()), p_active_out)

    meta: Dict[str, Any] = {
        "locator_frequency_hz": float(loc_hz),
        "picked_mode_index": int(picked_j),
        "picked_mode_frequency_hz": float(picked_f_hz) if math.isfinite(picked_f_hz) else float(loc_hz),
        "locator_pressure_full_length": locator_pressure_full_length,
        "n_p_full_collapsed": n_p_full,
        "n_p_active": int(layout["n_p_active"]),
        "n_u_active": int(layout["n_u_active"]),
        "n_reduced_W": n_W,
        "len_p_to_W": int(layout["len_p_to_W"]),
        "len_u_to_W": int(layout["len_u_to_W"]),
        "seed_vector_length": int(seed.size),
        "seed_norm": seed_norm,
        "seed_is_finite": bool(np.all(np.isfinite(seed))),
        "seed_layout_valid": bool(seed_layout_valid),
        "seed_build_success": bool(seed_layout_valid),
        "acoustic_locator_vector_saved": True,
        "locator_pressure_reference_source": ACOUSTIC_REFERENCE_SOURCE,
        "archived_pressure_input": str(args.archived_pressure_npy) if args.archived_pressure_npy else None,
        "pressure_restriction_source": pressure_source,
        "layout_maps": layout,
        "mapping_note": (
            "Full collapsed acoustic locator pressure restricted to active-air DOFs via "
            "_coupled_air_p_air_collapsed_indices, embedded on p_to_W with u block zero."
        ),
    }
    args.out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[seed] wrote {args.out_npy} loc_hz={loc_hz:.6f} n_W={n_W} "
        f"n_p_active={layout['n_p_active']} full_p={locator_pressure_full_length}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
