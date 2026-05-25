#!/usr/bin/env python3
"""
Experiment-only coupled participation / scaling audit (post-TEST-5, no eigen solve).

Replays saved coupled_near_acoustic modes against the same assembled A/M operators to
distinguish misleading p_frac from physical energy participation and FSI cross-terms.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, load_mode_column_any
from fem_worker_single import hz_result_tag
from mpi4py import MPI

from mode_diagnostics import (
    P_FRAC_FULLY_UNSCALED_DEFINITION,
    P_FRAC_PHYS_GNHEP_DEFINITION,
    P_FRAC_PRODUCTION_DEFINITION,
    block_l2_p_fraction,
    compute_mass_energy_participation,
    diagnose_mixed_mode,
    merge_scaling_metadata,
    unscale_mixed_mode_vector,
)


SIGMA_SUSPECT_HZ = 273.7168
SIGMA_SUSPECT_TOL_HZ = 0.05


def _resolve_mesh(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, EXPERIMENT_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_mode_artifact_raw(mode_path: Path) -> Any:
    """Load saved mode file without assuming dense layout."""
    if mode_path.name.endswith(MODE_VECTOR_FILE_SUFFIX):
        return load_mode_column_any(mode_path)
    if mode_path.suffix.lower() == ".npy":
        return np.load(str(mode_path))
    if mode_path.suffix.lower() == ".npz":
        data = np.load(str(mode_path))
        if "eigvec" in data:
            return data["eigvec"]
        if "arr" in data:
            return data["arr"]
        raise ValueError(f"{mode_path}: NPZ missing 'eigvec' or 'arr' keys")
    raise ValueError(f"Unsupported mode artifact: {mode_path}")


def _sparse_nnz(obj: Any) -> Optional[int]:
    if sparse.issparse(obj):
        return int(obj.nnz)
    return None


def _load_coupled_mode_dense_vector(
    mode_path: Path,
    *,
    n_coupled_W: int,
    mode_index: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a saved TEST 5 mode artifact to a dense 1-D float64 vector on W global ordering.

    Raises ValueError with file-specific diagnostics when the artifact is not a single mode.
    """
    loaded = _load_mode_artifact_raw(mode_path)
    meta: Dict[str, Any] = {
        "file": mode_path.name,
        "loaded_type": type(loaded).__name__,
        "is_sparse": bool(sparse.issparse(loaded)),
        "original_shape": list(np.asarray(loaded).shape) if not sparse.issparse(loaded) else list(loaded.shape),
        "nnz": _sparse_nnz(loaded),
    }

    if sparse.issparse(loaded):
        sp = loaded.tocsr()
        nrows, ncols = int(sp.shape[0]), int(sp.shape[1])
        meta["sparse_format"] = "csr"
        meta["sparse_shape"] = [nrows, ncols]
        if ncols != 1:
            raise ValueError(
                f"{mode_path.name}: expected a single mode column (shape (N, 1)), "
                f"got sparse shape ({nrows}, {ncols}); refusing to flatten a multi-mode matrix."
            )
        if nrows != int(n_coupled_W):
            raise ValueError(
                f"{mode_path.name}: sparse row count {nrows} != n_coupled_W={n_coupled_W}"
            )
        vec = np.asarray(sp.toarray(), dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(loaded)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                vec = np.asarray(arr[:, 0], dtype=np.float64).reshape(-1)
            elif arr.shape[0] == 1:
                vec = np.asarray(arr[0, :], dtype=np.float64).reshape(-1)
            else:
                raise ValueError(
                    f"{mode_path.name}: dense array shape {arr.shape} is not a single column/row vector; "
                    "refusing to flatten a multi-mode matrix."
                )
        elif arr.ndim == 1:
            vec = np.asarray(arr, dtype=np.float64).reshape(-1)
        else:
            raise ValueError(
                f"{mode_path.name}: unsupported ndarray rank {arr.ndim} (shape {arr.shape})"
            )

    meta["converted_length"] = int(vec.size)
    if vec.size != int(n_coupled_W):
        raise ValueError(
            f"{mode_path.name}: converted vector length {vec.size} != n_coupled_W={n_coupled_W}"
        )
    if not np.all(np.isfinite(vec)):
        n_bad = int(np.size(vec) - np.count_nonzero(np.isfinite(vec)))
        raise ValueError(
            f"{mode_path.name}: converted vector has {n_bad} non-finite entries"
        )
    amax = float(np.max(np.abs(vec))) if vec.size else 0.0
    if amax <= 0.0:
        raise ValueError(f"{mode_path.name}: converted vector is all zeros")
    meta["max_abs"] = amax
    meta["mode_index"] = int(mode_index)

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[coupled_participation_audit] mode load: file={meta['file']} "
            f"type={meta['loaded_type']} sparse={meta['is_sparse']} "
            f"shape={meta['original_shape']} nnz={meta['nnz']} "
            f"len={meta['converted_length']}",
            flush=True,
        )

    return vec, meta


def _load_modes(case_dir: Path, target_hz: float) -> Tuple[List[Dict[str, Any]], List[Path]]:
    diag_path = case_dir / "diagnostics" / "mode_physics_diagnostics.json"
    modes_meta: List[Dict[str, Any]] = []
    if diag_path.is_file():
        payload = json.loads(diag_path.read_text(encoding="utf-8"))
        modes_meta = list(payload.get("modes") or [])

    hz_tag = hz_result_tag(target_hz)
    mode_files = sorted((case_dir / "modes").glob(f"mode_{hz_tag}_*.smx.npz"))
    if not mode_files:
        mode_files = sorted((case_dir / "modes").glob("mode_*.smx.npz"))
    return modes_meta, mode_files


def _acoustic_reference_hz(solver_cfg: Dict[str, Any]) -> float:
    path = PHYSICS_ROOT / "acoustic_only" / "results" / "result_acoustic.json"
    ref = float(solver_cfg.get("acoustic_reference_hz", 244.39))
    if path.is_file():
        freqs = json.loads(path.read_text(encoding="utf-8")).get("frequencies_hz") or []
        if freqs:
            return float(freqs[0])
    return ref


def _audit_verdict(
    modes: List[Dict[str, Any]],
    *,
    ref_hz: float,
    ref_tol: float,
    min_energy_frac: float,
) -> Dict[str, Any]:
    in_tol = [
        m
        for m in modes
        if abs(float(m["frequency_hz"]) - ref_hz) <= ref_tol
        and not m.get("sigma_mapped_suspect")
    ]
    best_energy = max(in_tol, key=lambda m: float(m.get("p_frac_energy_phys", 0.0)), default=None)
    best_freq = min(modes, key=lambda m: abs(float(m["frequency_hz"]) - ref_hz), default=None)
    acoustic_branch = False
    reason = "no in-band mode within tolerance of acoustic reference"
    if best_energy and float(best_energy.get("p_frac_energy_phys", 0.0)) >= min_energy_frac:
        acoustic_branch = True
        reason = (
            f"mode f={best_energy['frequency_hz']:.4f} Hz has "
            f"p_frac_energy_phys={best_energy['p_frac_energy_phys']:.4e}"
        )
    elif best_freq and abs(float(best_freq["frequency_hz"]) - ref_hz) <= ref_tol:
        reason = (
            f"nearest mode f={best_freq['frequency_hz']:.4f} Hz has low physical acoustic energy "
            f"(p_frac_energy_phys={best_freq.get('p_frac_energy_phys', 0.0):.4e})"
        )
    return {
        "acoustic_reference_hz": ref_hz,
        "acoustic_branch_in_coupled_operator": acoustic_branch,
        "assessment": reason,
        "nearest_by_frequency": best_freq,
        "strongest_acoustic_energy_in_tol": best_energy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Coupled participation / scaling audit")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=PHYSICS_ROOT / "coupled_near_acoustic",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PHYSICS_ROOT / "configs" / "coupled_near_acoustic_244hz.json",
    )
    parser.add_argument("--band-lo", type=float, default=None)
    parser.add_argument("--band-hi", type=float, default=None)
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[coupled_participation_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    case_dir = args.case_dir.resolve()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    solver_cfg = cfg.get("solver", {})
    target_hz = float(solver_cfg.get("_worker_target_hz", 244.39))
    band_lo = float(args.band_lo if args.band_lo is not None else solver_cfg.get("_worker_harvest_lo_hz", 220.0))
    band_hi = float(args.band_hi if args.band_hi is not None else solver_cfg.get("_worker_harvest_hi_hz", 265.0))
    ref_hz = _acoustic_reference_hz(solver_cfg)
    ref_tol = float(solver_cfg.get("acoustic_reference_tolerance_hz", 8.0))
    min_energy_frac = float(solver_cfg.get("participation_audit_min_energy_frac", 0.02))

    modes_meta, mode_files = _load_modes(case_dir, target_hz)
    if not mode_files:
        print(f"[coupled_participation_audit] No mode files under {case_dir / 'modes'}", file=sys.stderr)
        return 1

    hz_tag = hz_result_tag(target_hz)
    result_path = case_dir / "results" / f"result_{hz_tag}.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    gnhep = merge_scaling_metadata(case_dir, result)
    pi_audit = cfg.get("_physics_integrity") or result.get("physics_integrity") or {}
    if isinstance(pi_audit, dict) and pi_audit.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi_audit["gnhep_scales"].items()})

    sorting = case_dir / "sorting_participation_audit"
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())
    cfg.setdefault("solver", {})["physics_integrity_capture"] = True
    cfg["solver"]["active_domain_experiment"] = {"enabled": False}

    mesh_file = _resolve_mesh(cfg, config_path)
    if MPI.COMM_WORLD.rank == 0:
        print("[coupled_participation_audit] Assembling coupled A/M (no SLEPc solve)...", flush=True)
    msh, W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )

    V_u, u_to_W = W.sub(0).collapse()
    V_p, p_to_W = W.sub(1).collapse()
    u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    n_u = int(V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs)
    n_p = int(V_p.dofmap.index_map.size_global * V_p.dofmap.index_map_bs)
    n_W = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)

    freqs_from_result = [float(f) for f in result.get("frequencies_hz") or []]
    audited: List[Dict[str, Any]] = []
    sigma_bucket: List[Dict[str, Any]] = []
    load_log: List[Dict[str, Any]] = []

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[coupled_participation_audit] n_coupled_W={n_W} (expect 136136 on validation mesh); "
            f"replaying {len(mode_files)} mode file(s)",
            flush=True,
        )

    for j, mode_path in enumerate(mode_files):
        try:
            mode_index = int(mode_path.stem.split("_")[-1])
        except ValueError:
            mode_index = j
        mode_meta = next((m for m in modes_meta if int(m.get("mode_index", -1)) == mode_index), {})
        f_hz = float(
            mode_meta.get("frequency_hz", freqs_from_result[j] if j < len(freqs_from_result) else float("nan"))
        )
        vec, load_meta = _load_coupled_mode_dense_vector(
            mode_path, n_coupled_W=n_W, mode_index=mode_index
        )
        load_log.append(load_meta)

        diag = diagnose_mixed_mode(
            vec,
            u_to_W=u_idx,
            p_to_W=p_idx,
            gnhep=gnhep,
            wood_top=float(mode_meta.get("top_plate_frac", 0.0)),
            wood_back=float(mode_meta.get("back_plate_frac", 0.0)),
            frequency_hz=f_hz,
        )
        p_prod, _, _ = block_l2_p_fraction(vec, u_to_W=u_idx, p_to_W=p_idx)
        vec_unscaled = unscale_mixed_mode_vector(
            vec, u_to_W=u_idx, p_to_W=p_idx, gnhep=gnhep, undo_pressure_dof_scale=True
        )
        p_full, _, _ = block_l2_p_fraction(vec_unscaled, u_to_W=u_idx, p_to_W=p_idx)
        energy = compute_mass_energy_participation(
            vec, M, A, u_to_W=u_idx, p_to_W=p_idx, gnhep=gnhep
        )

        sigma_suspect = abs(f_hz - SIGMA_SUSPECT_HZ) <= SIGMA_SUSPECT_TOL_HZ
        row = {
            "mode_index": mode_index,
            "frequency_hz": f_hz,
            "in_search_band": band_lo <= f_hz <= band_hi,
            "sigma_mapped_suspect": sigma_suspect,
            "sigma_cluster_hz": SIGMA_SUSPECT_HZ if sigma_suspect else None,
            "p_frac_production": float(
                mode_meta.get("p_frac_production", mode_meta.get("p_frac_raw", p_prod))
            ),
            "mode_load": load_meta,
            "p_frac_raw_audit": float(p_prod),
            "p_frac_phys_gnhep": float(diag["p_frac_phys_gnhep"]),
            "p_frac_fully_unscaled": float(p_full),
            "p_frac_energy_phys": float(energy["p_frac_energy_phys"]),
            "structural_modal_energy_phys": float(energy["structural_modal_energy_phys"]),
            "acoustic_modal_energy_phys": float(energy["acoustic_modal_energy_phys"]),
            "mass_cross_term_phys": float(energy["mass_cross_term_phys"]),
            "mass_cross_u_from_p_gnhep": float(energy["mass_cross_u_from_p_gnhep"]),
            "mass_cross_p_from_u_gnhep": float(energy["mass_cross_p_from_u_gnhep"]),
            "stiffness_u_row_load_norm": float(energy["stiffness_u_row_load_norm"]),
            "stiffness_p_row_load_norm": float(energy["stiffness_p_row_load_norm"]),
            "stiffness_cross_u_dot_Ax_u": float(energy["stiffness_cross_u_dot_Ax_u"]),
            "wood_participation": float(diag["wood_participation"]),
            "mode_class": diag["mode_class"],
            "delta_from_acoustic_ref_hz": f_hz - ref_hz,
            "vector_path": str(mode_path.relative_to(case_dir)).replace("\\", "/"),
        }
        audited.append(row)
        if sigma_suspect:
            sigma_bucket.append(row)

    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    in_band = [m for m in audited if m["in_search_band"]]
    in_band_all = list(in_band)
    rank_by_freq = sorted(in_band, key=lambda m: abs(float(m["delta_from_acoustic_ref_hz"])))
    rank_by_energy = sorted(
        in_band,
        key=lambda m: (-float(m["p_frac_energy_phys"]), abs(float(m["delta_from_acoustic_ref_hz"]))),
    )

    freq_hist = Counter(round(float(m["frequency_hz"]), 4) for m in audited)
    verdict = _audit_verdict(
        in_band,
        ref_hz=ref_hz,
        ref_tol=ref_tol,
        min_energy_frac=min_energy_frac,
    )

    p_frac_median = float(np.median([m["p_frac_production"] for m in in_band])) if in_band else 0.0
    p_energy_median = float(np.median([m["p_frac_energy_phys"] for m in in_band])) if in_band else 0.0
    diagnosis = "INCONCLUSIVE"
    notes: List[str] = []
    if p_frac_median < 1.0e-5 and p_energy_median >= min_energy_frac:
        diagnosis = "A"
        notes.append("Block L2 p_frac is tiny while mass-matrix acoustic energy fraction is not.")
    elif p_energy_median < min_energy_frac and verdict["acoustic_branch_in_coupled_operator"] is False:
        diagnosis = "C"
        notes.append("Physical energy metrics also show weak acoustic participation near 244.39 Hz.")
    elif not verdict["acoustic_branch_in_coupled_operator"] and p_frac_median < 1.0e-4:
        diagnosis = "B_or_C"
        notes.append(
            "No acoustic branch near reference; check sigma-mapped cluster and GNHEP/Nitsche scaling "
            "before concluding weak coupling."
        )
    else:
        diagnosis = "A_or_coupled_present"
        notes.append("Acoustic energy or coupled branch detected under physical metrics.")

    report = {
        "experiment": "coupled_participation_scaling_audit",
        "case_dir": str(case_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "branch": solver_cfg.get("physics_integrity_branch", "coupled-near-acoustic-244hz"),
        "harvest_policy": {
            "note": "This audit preserves all modes; no p_frac rejection.",
            "typical_test5_flags": {
                "skip_decoupled": 0,
                "allow_weak_coupling": True,
                "reject_decoupled": False,
            },
        },
        "metric_definitions": {
            "p_frac_production": P_FRAC_PRODUCTION_DEFINITION,
            "p_frac_phys_gnhep": P_FRAC_PHYS_GNHEP_DEFINITION,
            "p_frac_fully_unscaled": P_FRAC_FULLY_UNSCALED_DEFINITION,
            "p_frac_energy_phys": (
                "E_air_phys / (E_struct_phys + E_air_phys + |mass_cross|_phys) with "
                "E from x^T M x block splits on GNHEP-scaled assembled M, block energies scaled by "
                "s_uu/s_pp/s_couple to approximate physical units."
            ),
        },
        "scaling_metadata": gnhep,
        "soundhole_pressure_dof_count": int(pi_audit.get("soundhole_pressure_dof_count", 0)),
        "n_u": n_u,
        "n_p": n_p,
        "n_coupled_W_dofs": n_W,
        "search_band_hz": [band_lo, band_hi],
        "mode_vector_load_log": load_log,
        "modes_audited": audited,
        "modes_in_band": in_band_all,
        "ranking_by_proximity_to_acoustic_ref_hz": [
            {k: m[k] for k in ("mode_index", "frequency_hz", "p_frac_production", "p_frac_fully_unscaled", "p_frac_energy_phys", "sigma_mapped_suspect")}
            for m in rank_by_freq
        ],
        "ranking_by_acoustic_energy_participation": [
            {k: m[k] for k in ("mode_index", "frequency_hz", "p_frac_energy_phys", "p_frac_production", "acoustic_modal_energy_phys", "sigma_mapped_suspect")}
            for m in rank_by_energy
        ],
        "sigma_mapped_suspect_cluster": {
            "reference_hz": SIGMA_SUSPECT_HZ,
            "tolerance_hz": SIGMA_SUSPECT_TOL_HZ,
            "count": len(sigma_bucket),
            "modes": sigma_bucket,
            "frequency_histogram": {str(k): v for k, v in freq_hist.items() if v > 1},
        },
        "acoustic_branch_assessment": verdict,
        "scaling_diagnosis": {
            "label": diagnosis,
            "notes": notes,
            "median_p_frac_production_in_band": p_frac_median,
            "median_p_frac_energy_phys_in_band": p_energy_median,
        },
    }

    diag_dir = case_dir / "diagnostics"
    out_json = diag_dir / "coupled_participation_scaling_audit.json"
    out_md = diag_dir / "coupled_participation_scaling_audit.md"
    _write_json(out_json, report)

    if MPI.COMM_WORLD.rank == 0:
        lines = [
            "# Coupled participation / scaling audit",
            "",
            f"- Branch: `{report['branch']}`",
            f"- Band: {band_lo:.1f}–{band_hi:.1f} Hz | Acoustic reference: **{ref_hz:.2f} Hz**",
            f"- GNHEP: s_uu={gnhep.get('s_uu')} s_pp={gnhep.get('s_pp')} "
            f"s_couple={gnhep.get('s_couple')} pressure_dof_scale={gnhep.get('pressure_dof_scale')}",
            "",
            "## p_frac definitions",
            "",
            f"- **Production:** {P_FRAC_PRODUCTION_DEFINITION}",
            f"- **GNHEP undo:** {P_FRAC_PHYS_GNHEP_DEFINITION}",
            f"- **Fully unscaled:** {P_FRAC_FULLY_UNSCALED_DEFINITION}",
            "",
            "## Modes in band (all preserved)",
            "",
            "| f (Hz) | p_frac prod | p_frac unscaled | p_frac energy | E_air phys | E_struct phys | cross | suspect |",
            "|--------|-------------|-----------------|---------------|------------|---------------|-------|---------|",
        ]
        for m in in_band_all:
            lines.append(
                f"| {m['frequency_hz']:.4f} | {m['p_frac_production']:.3e} | "
                f"{m['p_frac_fully_unscaled']:.3e} | {m['p_frac_energy_phys']:.3e} | "
                f"{m['acoustic_modal_energy_phys']:.3e} | {m['structural_modal_energy_phys']:.3e} | "
                f"{m['mass_cross_term_phys']:.3e} | {'yes' if m['sigma_mapped_suspect'] else ''} |"
            )
        lines.extend(
            [
                "",
                "## Rankings",
                "",
                "### By proximity to acoustic reference",
                "",
            ]
        )
        for i, m in enumerate(rank_by_freq[:10], 1):
            lines.append(
                f"{i}. f={m['frequency_hz']:.4f} Hz Δref={m['delta_from_acoustic_ref_hz']:+.3f} "
                f"p_frac={m['p_frac_production']:.3e} E_part={m['p_frac_energy_phys']:.3e}"
            )
        lines.append("")
        lines.append("### By acoustic energy participation")
        lines.append("")
        for i, m in enumerate(rank_by_energy[:10], 1):
            lines.append(
                f"{i}. f={m['frequency_hz']:.4f} Hz E_part={m['p_frac_energy_phys']:.3e} "
                f"p_frac={m['p_frac_production']:.3e}"
            )
        lines.extend(
            [
                "",
                f"## Sigma-mapped suspect cluster (~{SIGMA_SUSPECT_HZ} Hz)",
                "",
                f"Count: {len(sigma_bucket)} (reported separately; audit ST mapping before interpreting)",
                "",
                "## Acoustic branch at coupled operator",
                "",
                f"- Exists (physical energy): **{verdict['acoustic_branch_in_coupled_operator']}**",
                f"- Note: {verdict['assessment']}",
                "",
                f"## Diagnosis: **{diagnosis}**",
                "",
            ]
        )
        lines.extend(f"- {n}" for n in notes)
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print("[coupled_participation_audit] branch:", report["branch"])
        print(
            f"[coupled_participation_audit] p_frac uses SLEPc/GNHEP-scaled W vector "
            f"(see {out_json.name})"
        )
        print(f"[coupled_participation_audit] modes in band={len(in_band_all)} sigma_suspect={len(sigma_bucket)}")
        print(f"[coupled_participation_audit] diagnosis={diagnosis} acoustic_branch={verdict['acoustic_branch_in_coupled_operator']}")
        print(f"[coupled_participation_audit] wrote {out_json} and {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
