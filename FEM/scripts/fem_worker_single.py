#!/usr/bin/env python3
"""
Single-use FEM worker: one SLEPc shift-invert batch at ``--target_hz``, then exit.

Loads coupled operators via ``fem_main_3d`` (expects **exactly one MPI rank** —
e.g. ``mpiexec --bind-to none -n 1 python FEM/scripts/fem_worker_single.py ...``).

Writes full eigenvectors to ``FEM/SORTING/temp_modes/mode_XXXXXX.npy`` and a small
JSON summary to ``FEM/SORTING/temp_results/result_<mHz_tag>.json`` for the master
to merge into ``candidates_log.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from mpi4py import MPI


def hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def result_json_path(sorting_root: Path, hz: float) -> Path:
    return sorting_root / "temp_results" / f"result_{hz_result_tag(hz)}.json"


def _resolve_mesh_path(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, config_path.parents[1], REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _uniqueness_for_vector(
    vec: np.ndarray,
    n_u_fe: int,
    temp_modes: Path,
    disk_exclude: Set[str],
    same_batch_vectors: List[np.ndarray],
) -> float:
    v = np.asarray(vec[:n_u_fe], dtype=np.float64)
    nv = float(np.linalg.norm(v))
    if nv <= 0.0:
        return 1.0
    max_ov = 0.0
    for p in same_batch_vectors:
        pt = np.asarray(p[:n_u_fe], dtype=np.float64)
        npv = float(np.linalg.norm(pt))
        if npv <= 0.0:
            continue
        ov = abs(float(np.vdot(v, pt))) / max(nv * npv, 1e-30)
        max_ov = max(max_ov, float(np.clip(ov, 0.0, 1.0)))
    if temp_modes.exists():
        for path in sorted(temp_modes.glob("mode_*.npy")):
            key = str(path.resolve())
            if key in disk_exclude:
                continue
            try:
                prev = np.load(key)
            except Exception:
                continue
            pt = np.asarray(prev[:n_u_fe], dtype=np.float64)
            npv = float(np.linalg.norm(pt))
            if npv <= 0.0:
                continue
            ov = abs(float(np.vdot(v, pt))) / max(nv * npv, 1e-30)
            max_ov = max(max_ov, float(np.clip(ov, 0.0, 1.0)))
    return 1.0 - max_ov


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-target-Hz FEM worker (one batch, then exit).")
    parser.add_argument("--target_hz", type=float, required=True)
    parser.add_argument("--num_modes", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "FEM" / "configs" / "guitar_3d.json",
        help="Case JSON (same as main 3D FEM driver).",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print(
                "[worker] Requires a single MPI process "
                "(e.g. `mpiexec --bind-to none -n 1 python FEM/scripts/fem_worker_single.py ...`).",
                file=sys.stderr,
            )
        return 2

    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("solver", {})
    cfg["solver"]["adaptive_mode_sifter"] = False
    cfg["_worker_target_hz"] = float(args.target_hz)
    cfg["_worker_num_modes"] = max(1, int(args.num_modes))

    mesh_file = _resolve_mesh_path(cfg, config_path)
    if MPI.COMM_WORLD.rank == 0 and not mesh_file.exists():
        print(f"[worker] Mesh not found: {mesh_file}", file=sys.stderr)
        return 1

    sorting_root = fem3d.SORTING_ROOT
    temp_modes = fem3d.SORTING_TEMP_MODES
    temp_results = sorting_root / "temp_results"
    if MPI.COMM_WORLD.rank == 0:
        temp_modes.mkdir(parents=True, exist_ok=True)
        temp_results.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.barrier()

    try:
        _msh, W, freqs_hz, eigvecs, _n_u, _n_p = fem3d._solve_coupled_evp(
            mesh_file=mesh_file,
            config=cfg,
            num_modes=max(1, int(args.num_modes)),
            status_callback=None,
        )
    except Exception as exc:
        if MPI.COMM_WORLD.rank == 0:
            print(f"[worker] Solve failed at {float(args.target_hz):.4f} Hz: {exc}", file=sys.stderr)
        return 1

    if MPI.COMM_WORLD.rank != 0:
        return 0

    tag1 = list(cfg.pop("_worker_tag1", []) or [])
    tag3 = list(cfg.pop("_worker_tag3", []) or [])
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    if len(tag1) != n_modes or len(tag3) != n_modes:
        print(
            f"[worker] Internal mismatch: modes={n_modes}, tag1={len(tag1)}, tag3={len(tag3)}",
            file=sys.stderr,
        )
        return 1

    n_u_g = int(W.sub(0).dofmap.index_map.size_global * W.sub(0).dofmap.index_map_bs)
    hz_tag = hz_result_tag(float(args.target_hz))
    candidates: List[Dict] = []
    same_batch: List[np.ndarray] = []
    exclude: Set[str] = set()

    for j in range(n_modes):
        vec = np.asarray(eigvecs[:, j], dtype=np.float64).copy()
        rt = float(tag1[j])
        rb = float(tag3[j])
        wood = max(0.0, rt + rb)
        uniq = _uniqueness_for_vector(vec, n_u_g, temp_modes, exclude, same_batch)
        rel = Path("temp_modes") / f"mode_w_{hz_tag}_{j:03d}.npy"
        abs_path = (sorting_root / rel).resolve()
        np.save(str(abs_path), vec)
        exclude.add(str(abs_path))
        same_batch.append(vec.copy())
        candidates.append(
            {
                "id": int(j),
                "hz": float(freqs_hz[j]),
                "wood_participation": float(wood),
                "uniqueness": float(uniq),
                "tag1_ratio": float(rt),
                "tag3_ratio": float(rb),
                "vector_path": str(rel).replace("\\", "/"),
            }
        )

    out = {
        "target_hz": float(args.target_hz),
        "num_modes_requested": int(args.num_modes),
        "num_modes_returned": int(n_modes),
        "candidates": candidates,
    }
    rpath = result_json_path(sorting_root, float(args.target_hz))
    _atomic_write_json(rpath, out)
    print(f"[worker] Wrote {n_modes} mode(s); metadata -> {rpath}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
