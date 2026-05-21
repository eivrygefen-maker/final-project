#!/usr/bin/env python3
"""
Single-use FEM worker: one SLEPc shift-invert batch at ``--target_hz``, then exit.

Loads coupled operators via ``fem_main_3d`` (expects **exactly one MPI rank** —
e.g. ``taskset -c 1 mpiexec -n 1 python FEM/scripts/fem_worker_single.py ...`` as spawned by the master).

Writes mode displacement vectors as **float32 CSR** sparse columns (relative noise sparsified),
one file per mode under ``<sorting-root>/temp_modes/mode_*.smx.npz`` (default: ``FEM/SORTING``),
plus JSON to ``<sorting-root>/temp_results/result_<mHz_tag>.json``. Use ``--sorting-root``
to match ``fem_master_dynamic`` (required when the master uses a non-default lab SORTING).

**Uniqueness** for each mode is computed only against (a) prior modes in the same worker batch and
(b) existing vectors under ``temp_modes/`` on disk—not against rows in ``candidates_log.json`` whose
files are missing. Modes below ``WORKER_UNIQUENESS_MIN`` are dropped before writing JSON; the master
merge applies the same floor as a safety net.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
# .../<repo>/FEM/scripts/ → repo root vs FEM package root
REPO_ROOT = SCRIPT_DIR.parents[2]
FEM_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from wood_library import resolve_plate_thicknesses
from fem_harvest_filter import (
    HARVEST_FILTER_POLICY_VERSION,
    HarvestFilterConfig,
    classify_mode_candidate,
)
from fem_mode_array_utils import (
    MODE_VECTOR_FILE_SUFFIX,
    csr_col_norm,
    csr_normalized_overlap,
    csr_u_slice,
    dense_to_csr_f32_column,
    save_mode_csr,
)

# Near-degenerate eigenpairs in one shift-invert batch (Hz); drop weak-unique duplicates.
NEAR_MODE_HZ_WORKER = 0.05
# Minimum uniqueness vs same-batch vectors + everything under ``temp_modes/`` (not candidates_log rows).
WORKER_UNIQUENESS_MIN = 0.04
from mpi4py import MPI

# SLEPc batch size for master-spawned coupled workers (LU memory / ncv stability on VMs).
_DEFAULT_WORKER_NUM_MODES_CAP = 40


def _apply_master_worker_solver_profile(
    cfg: dict,
    *,
    num_modes: int,
    structural_only: bool,
    eps_band_solver: str | None,
) -> int:
    """
    Force the production shift-invert stack used on the stable 155 Hz monolithic runs.

    Overrides stale merged JSON (e.g. ``guitar_3d.json`` with ``st_fieldsplit: true``,
    ``eps_ncv_max: 180``) so master-spawned workers always match ``sample_001`` intent.
    """
    s = cfg.setdefault("solver", {})
    s["adaptive_mode_sifter"] = False
    if eps_band_solver is not None:
        s["eps_band_solver"] = str(eps_band_solver)
    if structural_only:
        return max(1, int(num_modes))
    s["eps_band_solver"] = "shift_invert"
    s["st_use_fieldsplit"] = False
    s["st_fieldsplit"] = False
    s["st_pc_type"] = "lu"
    s["gnhep_block_frobenius_normalize"] = True
    s.setdefault("eps_pin_fix_tag5", True)
    s.setdefault("eps_algebraic_bc_zero_columns", True)
    cap = int(s.get("eps_worker_num_modes_cap", _DEFAULT_WORKER_NUM_MODES_CAP) or _DEFAULT_WORKER_NUM_MODES_CAP)
    cap = max(1, min(cap, 48))
    nm = min(max(1, int(num_modes)), cap)
    rigid_buf = int(s.get("eps_rigid_mode_buffer", 5) or 5)
    # SLEPc requires ncv >= nev+1 with nev ~= num_modes + rigid_buf.
    ncv_min = int(nm) + max(rigid_buf, 0) + 2
    ncv_max = int(s.get("eps_ncv_max", 48) or 48)
    if ncv_max <= 0:
        ncv_max = 48
    ncv_max = max(ncv_max, ncv_min)
    s["eps_ncv_max"] = ncv_max
    s["target_ncv"] = min(
        max(int(s.get("target_ncv", 0)), int(math.ceil(4.0 * nm))),
        ncv_max,
    )
    return nm


def hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def result_json_path(sorting_root: Path, hz: float) -> Path:
    return sorting_root / "temp_results" / f"result_{hz_result_tag(hz)}.json"


def _resolve_mesh_path(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, config_path.parents[1], FEM_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _uniqueness_for_sparse_column(
    vec_csr: sparse.csr_matrix,
    n_u: int,
    temp_modes: Path,
    disk_exclude: Set[str],
    same_batch: List[sparse.csr_matrix],
) -> float:
    """
    Structural uniqueness in ``[0, 1]``: ``1 - max_j |cos(u, u_j)|`` over sparse displacement blocks.

    **Scope (local + on-disk, not the JSON log)**:
    * ``same_batch``: CSR columns already accepted earlier in *this* worker invocation.
    * ``temp_modes/``: every ``mode_*`` vector file on disk except paths in ``disk_exclude``.

    Rows that exist only in ``candidates_log.json`` but whose ``vector_path`` files were removed
    are invisible here; the master merge applies an additional uniqueness floor for those cases.
    ``disk_exclude`` must include the absolute path of the slot about to be written so a stale
    file from a prior run is not treated as ``self`` (which would force uniqueness to 0.0).
    """
    v = csr_u_slice(vec_csr, n_u)
    nv = csr_col_norm(v)
    if nv <= 0.0:
        return 1.0
    max_ov = 0.0
    for p in same_batch:
        max_ov = max(max_ov, csr_normalized_overlap(v, csr_u_slice(p, n_u)))
    if temp_modes.exists():
        for path in sorted(temp_modes.glob(f"mode_*{MODE_VECTOR_FILE_SUFFIX}")):
            key = str(path.resolve())
            if key in disk_exclude:
                continue
            try:
                prev = sparse.load_npz(str(path)).tocsr().astype(np.float32, copy=False)
            except Exception:
                continue
            max_ov = max(max_ov, csr_normalized_overlap(v, csr_u_slice(prev, n_u)))
        # Legacy dense .npy on disk
        for path in sorted(temp_modes.glob("mode_*.npy")):
            key = str(path.resolve())
            if key in disk_exclude:
                continue
            try:
                arr = np.load(str(path))
            except Exception:
                continue
            prev = dense_to_csr_f32_column(arr)
            max_ov = max(max_ov, csr_normalized_overlap(v, csr_u_slice(prev, n_u)))
    return float(max(0.0, min(1.0, 1.0 - max_ov)))


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-target-Hz FEM worker (one batch, then exit).")
    parser.add_argument(
        "--target_hz",
        "--target-hz",
        dest="target_hz",
        type=float,
        required=True,
        metavar="HZ",
        help="Shift-invert band center frequency (Hz).",
    )
    parser.add_argument(
        "--num_modes",
        "--num-modes",
        dest="num_modes",
        type=int,
        required=True,
        help="Number of modes to request from SLEPc harvest.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "FEM" / "configs" / "guitar_3d.json",
        help="Case JSON (same as main 3D FEM driver).",
    )
    parser.add_argument(
        "--sorting-root",
        type=Path,
        default=None,
        help=(
            "SORTING workspace (temp_modes/, temp_results/, candidates_log.json). "
            "Must match fem_master_dynamic --sorting-root (default: FEM/SORTING next to FEM package)."
        ),
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Displacement-only shell EVP (structural_only_diagnosis); used by master fallback.",
    )
    parser.add_argument(
        "--eps-band-solver",
        "--eps_band_solver",
        dest="eps_band_solver",
        choices=("shift_invert", "ciss"),
        default=None,
        help=(
            "Override solver.eps_band_solver: 'ciss' contours [target±half_width] Hz "
            "(finds band modes); 'shift_invert' is σ-anchored (often wood-only spurious)."
        ),
    )
    parser.add_argument(
        "--harvest-lo-hz",
        type=float,
        default=None,
        help="Master spectral-band harvest lower bound (Hz); sets solver harvest window.",
    )
    parser.add_argument(
        "--harvest-hi-hz",
        type=float,
        default=None,
        help="Master spectral-band harvest upper bound (Hz).",
    )
    parser.add_argument(
        "--eps-broad-search-hz",
        type=float,
        default=None,
        help="Override solver.eps_broad_search_hz (half-width) for this shift.",
    )
    args = parser.parse_args()

    if args.sorting_root is not None:
        fem3d.set_sorting_root(args.sorting_root.resolve())

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print(
                "[worker] Requires a single MPI process "
                "(e.g. `taskset -c 1 mpiexec -n 1 python FEM/scripts/fem_worker_single.py ...`).",
                file=sys.stderr,
            )
        return 2

    if sys.platform.startswith("linux") and hasattr(os, "sched_getaffinity"):
        try:
            aff = sorted(os.sched_getaffinity(0))
            print(
                f"[worker] Linux CPU affinity for PID {os.getpid()}: "
                f"sched_getaffinity(0) = {{{', '.join(map(str, aff))}}}"
            )
            sys.stdout.flush()
        except OSError as exc:
            print(f"[worker] sched_getaffinity(0) failed: {exc}", file=sys.stderr)
            sys.stderr.flush()

    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    _patch_raw = os.environ.get("FEM_WORKER_SOLVER_OVERRIDES", "").strip()
    if _patch_raw:
        try:
            _patch = json.loads(_patch_raw)
            if isinstance(_patch, dict):
                cfg.setdefault("solver", {}).update(_patch)
        except json.JSONDecodeError as exc:
            print(f"[worker] FEM_WORKER_SOLVER_OVERRIDES JSON invalid: {exc}", file=sys.stderr)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[worker] Config file (resolved): {config_path}")
        try:
            t_top, t_back = resolve_plate_thicknesses(cfg)
            print(
                f"[worker] geometry.top_thickness = {t_top:.6f} m ({t_top * 1000.0:.4f} mm), "
                f"geometry.back_thickness = {t_back:.6f} m ({t_back * 1000.0:.4f} mm)"
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[worker] asymmetric plate thickness unavailable: {exc}")
        sys.stdout.flush()
    _worker_num_modes = _apply_master_worker_solver_profile(
        cfg,
        num_modes=int(args.num_modes),
        structural_only=bool(args.structural_only),
        eps_band_solver=args.eps_band_solver,
    )
    _band_solver = str(cfg["solver"].get("eps_band_solver", "shift_invert")).strip().lower()
    if _band_solver in ("shift_invert", "shift-invert", "sinvert"):
        # Coupled FSI harvest: keep converged modes near target and at the actual ST shift.
        cfg["solver"]["eps_reject_sigma_spurious"] = False
        cfg["solver"]["eps_reject_target_locked"] = False
        cfg["solver"]["eps_reject_decoupled_u_only"] = False
        cfg["solver"]["eps_harvest_allow_weak_coupling"] = True
        _broad = float(cfg["solver"].get("eps_broad_search_hz", 8.0))
        if args.eps_broad_search_hz is not None:
            cfg["solver"]["eps_broad_search_hz"] = max(float(args.eps_broad_search_hz), 8.0)
        else:
            cfg["solver"]["eps_broad_search_hz"] = max(_broad, 50.0)
    if args.harvest_lo_hz is not None and args.harvest_hi_hz is not None:
        cfg["solver"]["_worker_harvest_lo_hz"] = float(args.harvest_lo_hz)
        cfg["solver"]["_worker_harvest_hi_hz"] = float(args.harvest_hi_hz)
    if args.structural_only:
        cfg["solver"]["structural_only_diagnosis"] = True
    _target_hz = float(args.target_hz)
    _target_lambda = (2.0 * math.pi * _target_hz) ** 2
    # Respect merged config (e.g. eps_which=TARGET_MAGNITUDE, st_type=sinvert); do not override here.
    cfg["solver"].pop("eps_smallest_magnitude", None)
    cfg["solver"].pop("eps_use_which_user", None)
    cfg["_worker_eps_target_lambda"] = _target_lambda
    cfg["solver"]["_worker_eps_target_lambda"] = _target_lambda
    cfg["solver"]["_worker_target_hz"] = _target_hz
    cfg["_worker_eps_max_it"] = int(cfg["solver"].get("eigs_maxiter", cfg["solver"].get("eps_max_it", 3000)))
    cfg["_worker_target_hz"] = _target_hz
    cfg["_worker_num_modes"] = _worker_num_modes
    if MPI.COMM_WORLD.rank == 0 and _worker_num_modes != int(args.num_modes):
        print(
            f"[worker] Production cap: num_modes {int(args.num_modes)} -> {_worker_num_modes} "
            f"(eps_worker_num_modes_cap / monolithic LU profile)"
        )
        sys.stdout.flush()
    if MPI.COMM_WORLD.rank == 0:
        _solver = cfg.get("solver", {}) or {}
        _which = str(_solver.get("eps_which", "TARGET_MAGNITUDE"))
        _st = str(_solver.get("st_type", "shift"))
        _band = str(_solver.get("eps_band_solver", "shift_invert"))
        _half = float(_solver.get("eps_interval_half_width_hz", 5.0))
        _band_lo = _target_hz - _half
        _band_hi = _target_hz + _half
        if _band.strip().lower() in ("ciss", "contour", "contour_integral"):
            _band_str = f"CISS band=[{_band_lo:.2f}, {_band_hi:.2f}] Hz (RGINTERVAL)"
        elif _band.strip().lower() in ("interval", "spectrum_slicing", "slice", "band_interval"):
            _fb = str(_solver.get("eps_interval_fallback", "ciss"))
            _band_str = f"band=[{_band_lo:.2f}, {_band_hi:.2f}] Hz (interval→{_fb})"
        else:
            _broad_h = float(_solver.get("eps_broad_search_hz", 8.0))
            _hw_lo = _solver.get("_worker_harvest_lo_hz")
            _hw_hi = _solver.get("_worker_harvest_hi_hz")
            _harvest_note = (
                f" harvest=[{float(_hw_lo):.2f}, {float(_hw_hi):.2f}] Hz"
                if _hw_lo is not None and _hw_hi is not None
                else ""
            )
            _band_str = (
                f"shift @ {_target_hz:.4f} Hz{_harvest_note} "
                f"(harvest [{_target_hz - _broad_h:.1f}, {_target_hz + _broad_h:.1f}] Hz)"
            )
        _fs = (
            _solver.get("st_use_fieldsplit", False)
            or _solver.get("st_fieldsplit", False)
            or str(_solver.get("st_pc_type", "")).lower() in ("fieldsplit", "fs")
        )
        print(
            f"[worker] EPS target: {_band_str}, "
            f"lambda_center=(2*pi*f)^2={_target_lambda:.6e}, "
            f"band_solver={_band}, which={_which}, st={_st}, "
            f"st_fieldsplit={bool(_fs)}"
        )
        sys.stdout.flush()

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
            num_modes=_worker_num_modes,
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
    p_fracs = list(cfg.pop("_worker_p_frac", []) or [])
    p_block_maxes = list(cfg.pop("_worker_p_block_max", []) or [])
    _shift_jitter = float(cfg.get("solver", {}).get("shift_jitter_hz", 0.0))
    st_sigma_hz = float(
        cfg.pop("_worker_st_sigma_hz", max(1.0, float(args.target_hz) + _shift_jitter))
    )
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[worker] SLEPc usable modes at target {float(args.target_hz):.4f} Hz: "
            f"n_modes={n_modes} (requested {int(_worker_num_modes)}, "
            f"argv num_modes={int(args.num_modes)})"
        )
        sys.stdout.flush()
    if len(tag1) != n_modes or len(tag3) != n_modes:
        print(
            f"[worker] Internal mismatch: modes={n_modes}, tag1={len(tag1)}, tag3={len(tag3)}",
            file=sys.stderr,
        )
        return 1
    if len(p_fracs) != n_modes:
        p_fracs = [0.0] * n_modes
    if len(p_block_maxes) != n_modes:
        p_block_maxes = [0.0] * n_modes

    n_u_g = int(W.sub(0).dofmap.index_map.size_global * W.sub(0).dofmap.index_map_bs)
    hz_tag = hz_result_tag(float(args.target_hz))
    hcfg = HarvestFilterConfig.from_solver_cfg(cfg.get("solver", {}))
    # Classify first; process ROM-ready (physical) modes before σ-ritz so uniqueness keeps resonances.
    slot_order = list(range(n_modes))
    def _harvest_sort_key(j: int) -> tuple:
        fj = float(freqs_hz[j])
        _, rom, _ = classify_mode_candidate(
            {
                "hz": fj,
                "p_frac": float(p_fracs[j]) if j < len(p_fracs) else 0.0,
                "wood_participation": float(tag1[j]) + float(tag3[j]),
            },
            target_hz=float(args.target_hz),
            st_sigma_hz=st_sigma_hz,
            cfg=hcfg,
        )
        return (0 if rom else 1, abs(fj - st_sigma_hz), -fj)

    slot_order.sort(key=_harvest_sort_key)
    candidates: List[Dict] = []
    same_batch: List[sparse.csr_matrix] = []
    exclude: Set[str] = set()
    last_kept_hz: float = float("-inf")

    for j in slot_order:
        vec_csr = dense_to_csr_f32_column(eigvecs[:, j])
        rt = float(tag1[j])
        rb = float(tag3[j])
        wood = max(0.0, rt + rb)
        pbm = float(p_block_maxes[j])
        print(
            f"[worker][pre-gate] mode {j}: f={float(freqs_hz[j]):.4f} Hz "
            f"max|p|={pbm:.6e} p_frac={float(p_fracs[j]):.3e} wood={wood:.4f}",
            flush=True,
        )
        rel = Path("temp_modes") / f"mode_w_{hz_tag}_{j:03d}{MODE_VECTOR_FILE_SUFFIX}"
        pending_abs = str((sorting_root / rel).resolve())
        disk_exclude_scan = set(exclude)
        disk_exclude_scan.add(pending_abs)
        uniq = _uniqueness_for_sparse_column(
            vec_csr, n_u_g, temp_modes, disk_exclude_scan, same_batch
        )
        fj = float(freqs_hz[j])
        if (
            math.isfinite(last_kept_hz)
            and abs(fj - last_kept_hz) < NEAR_MODE_HZ_WORKER
            and float(uniq) < WORKER_UNIQUENESS_MIN
        ):
            continue
        if float(uniq) < WORKER_UNIQUENESS_MIN - 1e-15:
            continue
        h_label, h_rom, h_reason = classify_mode_candidate(
            {
                "hz": float(freqs_hz[j]),
                "p_frac": float(p_fracs[j]),
                "wood_participation": float(wood),
                "uniqueness": float(uniq),
            },
            target_hz=float(args.target_hz),
            st_sigma_hz=float(st_sigma_hz),
            cfg=hcfg,
        )
        if h_label == "sigma_ritz" and not hcfg.keep_sigma_reference:
            continue
        if not h_rom:
            continue
        u_blk = csr_u_slice(vec_csr, n_u_g)
        col_norm = float(csr_col_norm(u_blk))
        if col_norm < 1e-12:
            continue
        abs_path = (sorting_root / rel).resolve()
        save_mode_csr(abs_path, vec_csr)
        exclude.add(str(abs_path))
        same_batch.append(vec_csr.copy())
        candidates.append(
            {
                "id": int(j),
                "hz": float(freqs_hz[j]),
                "source_target_hz": float(args.target_hz),
                "wood_participation": float(wood),
                "p_frac": float(p_fracs[j]),
                "uniqueness": float(uniq),
                "tag1_ratio": float(rt),
                "tag3_ratio": float(rb),
                "vector_path": str(rel).replace("\\", "/"),
                "column_l2_norm": float(col_norm),
                "harvest_class": h_label,
                "rom_ready": bool(h_rom),
                "harvest_reason": h_reason,
                "harvest_filter_policy": HARVEST_FILTER_POLICY_VERSION,
            }
        )
        last_kept_hz = fj

    structural_only_run = bool(
        args.structural_only
        or cfg.get("solver", {}).get("structural_only_diagnosis", False)
    )
    out = {
        "target_hz": float(args.target_hz),
        "st_sigma_hz": float(st_sigma_hz),
        "harvest_filter_policy": HARVEST_FILTER_POLICY_VERSION,
        "structural_only_run": structural_only_run,
        "num_modes_requested": int(_worker_num_modes),
        "num_modes_argv": int(args.num_modes),
        "num_modes_returned": int(len(candidates)),
        "num_modes_slepc": int(n_modes),
        "candidates": candidates,
    }
    rpath = result_json_path(sorting_root, float(args.target_hz))
    _atomic_write_json(rpath, out)
    print(
        f"[worker] Wrote {len(candidates)} sparse mode(s) after uniqueness filter "
        f"(from {n_modes} SLEPc rows); metadata -> {rpath}"
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
