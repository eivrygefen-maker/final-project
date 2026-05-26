#!/usr/bin/env python3
"""
Final report-only audit: saved-vector persistence and mass-norm control for the
completed unregularized-offset baseline diagnostic (no eigensolve).

Reads exclusively:
  seed_branch_recovery_diagnostic_unregularized_offset/
  diagnostics/acoustic_coupled_seed.npy
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_sensitivity_common import hz_result_tag
from v2_unreg_offset_report_evaluator import (
    _load_sample_spec,
    assemble_replay_operators,
    load_seed_with_diagnostics,
    resolve_maps_from_solve_result,
)

OUT_JSON = CONV_DIAG / "v2_l_mid_unregularized_saved_vector_mass_norm_audit.json"
OUT_MD = CONV_DIAG / "v2_l_mid_unregularized_saved_vector_mass_norm_audit.md"

CASE_ID = "baseline_coupled_v2"
OUT_SUBDIR = "seed_branch_recovery_diagnostic_unregularized_offset"
SEED_F_HZ = 243.0754171175576

VERDICT_REPLAY_INVALID = "REPLAY_CONTROL_INVALID_SEED_XHMX_NONFINITE"
VERDICT_PERSISTENCE_BUG = "SAVED_MODE_VECTOR_PERSISTENCE_OR_LAYOUT_BUG_CONFIRMED"
VERDICT_EPS_MASS_NULL = "EPS_RETURNED_ONLY_MASS_NULL_CANDIDATES_IN_UNREGULARIZED_SOLVE"
VERDICT_NOT_LOCALIZED = "VECTOR_MASS_NULL_ROOT_CAUSE_NOT_LOCALIZED_STOP_FOR_ARCHITECTURE_REVIEW"

XH_MX_TOL = 1.0e-30


def _rayleigh_and_residual(
    A,
    M,
    vec: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    lam_for_residual: Optional[float] = None,
) -> Dict[str, Any]:
    from physical_fsi_seed_residual_audit import (
        _block_residual_contributions,
        _rayleigh_metrics,
    )

    out: Dict[str, Any] = {
        "replay_assembly_status": "pending",
        "replay_exception_if_any": None,
        "xH_Ax": float("nan"),
        "xH_Mx": float("nan"),
        "rayleigh_lambda": float("nan"),
        "rayleigh_frequency_hz": float("nan"),
        "relative_residual": float("nan"),
    }
    try:
        ray = _rayleigh_metrics(A, M, vec, seed_f_hz=float("nan"))
        lam = float(ray["rayleigh_lambda"])
        out["xH_Ax"] = float(ray.get("xH_Ax", float("nan")))
        out["xH_Mx"] = float(ray.get("xH_Mx", float("nan")))
        out["rayleigh_lambda"] = lam
        out["rayleigh_frequency_hz"] = float(ray.get("rayleigh_f_hz", float("nan")))
        lam_use = lam if math.isfinite(lam) else lam_for_residual
        if lam_use is not None and math.isfinite(float(lam_use)):
            res = _block_residual_contributions(
                A, M, vec, lam0=float(lam_use), u_idx=u_to_W, p_idx=p_to_W
            )
            out["relative_residual"] = float(res["relative_residual"])
        out["replay_assembly_status"] = "ok"
    except Exception as exc:
        out["replay_assembly_status"] = "exception"
        out["replay_exception_if_any"] = f"{type(exc).__name__}:{exc}"
    return out


def _seed_control(
    A,
    M,
    seed: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    operator_size: int,
) -> Dict[str, Any]:
    vec = np.asarray(seed, dtype=np.float64).ravel()
    p_blk = np.asarray(vec[p_to_W], dtype=np.float64).ravel() if p_to_W.size else np.array([])
    u_blk = np.asarray(vec[u_to_W], dtype=np.float64).ravel() if u_to_W.size else np.array([])
    metrics = _rayleigh_and_residual(A, M, vec, u_to_W=u_to_W, p_to_W=p_to_W)
    return {
        "seed_vector_length": int(vec.size),
        "seed_vector_norm": float(np.linalg.norm(vec)),
        "operator_size": int(operator_size),
        "length_matches_operator": bool(vec.size == operator_size),
        "seed_xH_Ax": metrics["xH_Ax"],
        "seed_xH_Mx": metrics["xH_Mx"],
        "seed_rayleigh_lambda": metrics["rayleigh_lambda"],
        "seed_rayleigh_frequency_hz": metrics["rayleigh_frequency_hz"],
        "seed_relative_residual": metrics["relative_residual"],
        "seed_pressure_block_norm": float(np.linalg.norm(p_blk)) if p_blk.size else 0.0,
        "seed_u_block_norm": float(np.linalg.norm(u_blk)) if u_blk.size else 0.0,
        "replay_assembly_status": metrics["replay_assembly_status"],
        "replay_exception_if_any": metrics["replay_exception_if_any"],
        "seed_xH_Mx_finite_nonzero": bool(
            math.isfinite(metrics["xH_Mx"]) and abs(metrics["xH_Mx"]) > XH_MX_TOL
        ),
    }


def _inspect_npz_vector(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "vector_file": str(path),
        "file_size_bytes": path.stat().st_size if path.is_file() else None,
        "vector_file_exists": path.is_file(),
        "loader_function_used": None,
        "load_exception": None,
        "available_npz_keys": None,
        "stored_shape": None,
        "stored_dtype": None,
        "stored_nnz_if_sparse": None,
        "loaded_vector_length": None,
        "loaded_vector_norm": float("nan"),
        "loaded_vector_nonzero_count": None,
        "loaded_vector_finite": None,
        "vector_load_status": "pending",
    }
    if not path.is_file():
        row["vector_load_status"] = "missing"
        return row

    try:
        from scipy import sparse

        raw = sparse.load_npz(str(path))
        row["loader_function_used"] = "scipy.sparse.load_npz"
        row["available_npz_keys"] = "(scipy sparse matrix file)"
        if sparse.issparse(raw):
            m = raw.tocsr()
            row["stored_shape"] = list(m.shape)
            row["stored_dtype"] = str(m.dtype)
            row["stored_nnz_if_sparse"] = int(m.nnz)
            vec = np.asarray(m.toarray(), dtype=np.float64).ravel()
        else:
            row["load_exception"] = f"not_sparse:{type(raw)}"
            row["vector_load_status"] = "not_sparse_matrix"
            return row
    except Exception as exc_scipy:
        row["load_exception_scipy"] = f"{type(exc_scipy).__name__}:{exc_scipy}"
        try:
            from fem_mode_array_utils import load_mode_column_any

            m = load_mode_column_any(path)
            row["loader_function_used"] = "fem_mode_array_utils.load_mode_column_any"
            row["stored_shape"] = list(m.shape)
            row["stored_dtype"] = str(m.dtype)
            row["stored_nnz_if_sparse"] = int(m.nnz)
            vec = np.asarray(m.toarray(), dtype=np.float64).ravel()
        except Exception as exc_fem:
            row["loader_function_used"] = "failed"
            row["load_exception"] = f"{type(exc_fem).__name__}:{exc_fem}"
            row["vector_load_status"] = "load_failed"
            return row

    row["loaded_vector_length"] = int(vec.size)
    row["loaded_vector_norm"] = float(np.linalg.norm(vec))
    row["loaded_vector_nonzero_count"] = int(np.count_nonzero(vec))
    row["loaded_vector_finite"] = bool(np.all(np.isfinite(vec)))
    row["vector_load_status"] = "ok"
    row["_vec_array"] = vec
    return row


def _candidate_row(
    *,
    mode_row: Dict[str, Any],
    vec_path: Path,
    A,
    M,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    operator_size: int,
    restr: Dict[str, Any],
) -> Dict[str, Any]:
    insp = _inspect_npz_vector(vec_path)
    row: Dict[str, Any] = {
        "candidate_index": mode_row.get("mode_index"),
        "reported_frequency_hz": mode_row.get("frequency_hz"),
        "reported_p_frac_energy_phys": mode_row.get("p_frac_energy_phys"),
        "reported_mode_class_physical_energy": mode_row.get("mode_class_physical_energy"),
        "mode_energy_summary_vector_path": mode_row.get("vector_path"),
        **{k: v for k, v in insp.items() if k != "_vec_array"},
    }
    if insp.get("vector_load_status") != "ok":
        row["u_block_norm"] = float("nan")
        row["p_block_norm"] = float("nan")
        row["pressure_active_block_norm"] = float("nan")
        row["xH_Ax"] = float("nan")
        row["xH_Mx"] = float("nan")
        row["length_matches_operator"] = False
        return row

    vec = insp["_vec_array"]
    row["length_matches_operator"] = bool(int(vec.size) == int(operator_size))
    u_blk = np.asarray(vec[u_to_W], dtype=np.float64).ravel() if u_to_W.size else np.array([])
    p_blk = np.asarray(vec[p_to_W], dtype=np.float64).ravel() if p_to_W.size else np.array([])
    row["u_block_norm"] = float(np.linalg.norm(u_blk))
    row["p_block_norm"] = float(np.linalg.norm(p_blk))
    row["pressure_active_block_norm"] = row["p_block_norm"]
    row["constrained_or_dropped_support_if_available"] = {
        "n_reduced_W": restr.get("n_reduced_W"),
        "n_u_active": restr.get("n_u_active"),
        "n_p_active": restr.get("n_p_active"),
        "dropped_inactive_p": restr.get("dropped_inactive_p"),
    }
    metrics = _rayleigh_and_residual(
        A,
        M,
        vec,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        lam_for_residual=(
            (2.0 * math.pi * float(mode_row["frequency_hz"])) ** 2
            if math.isfinite(float(mode_row.get("frequency_hz", float("nan"))))
            else None
        ),
    )
    row.update(
        {
            "xH_Ax": metrics["xH_Ax"],
            "xH_Mx": metrics["xH_Mx"],
            "rayleigh_lambda": metrics["rayleigh_lambda"],
            "rayleigh_frequency_hz": metrics["rayleigh_frequency_hz"],
            "replay_relative_residual": metrics["relative_residual"],
            "replay_assembly_status": metrics["replay_assembly_status"],
            "replay_exception_if_any": metrics["replay_exception_if_any"],
        }
    )
    return row


def _static_code_trace() -> Dict[str, Any]:
    fem = FEM_SCRIPTS / "fem_main_3d.py"
    v2solve = SCRIPT_DIR / "v2_sensitivity_solve.py"
    mode_utils = FEM_SCRIPTS / "fem_mode_array_utils.py"
    return {
        "write_path": {
            "file": str(v2solve.relative_to(REPO_ROOT)),
            "functions": [
                "main() loop over eigvecs[:, j]",
                "save_mode_csr(mode_path, dense_to_csr_f32_column(vec))",
            ],
            "vector_source": (
                "Columns of eigvecs returned by fem_main_3d._solve_coupled_evp after EPS harvest."
            ),
            "written_representation": (
                "Full column vec=eigvecs[:, j] in reduced-W layout at harvest time; "
                "relative sparsification (MODE_VECTOR_RELATIVE_EPS=1e-7) then CSR float32 .smx.npz."
            ),
            "timing_vs_restriction": (
                "Harvest builds arr from eps.getEigenpair rvec.array.copy(); if active_domain "
                "metadata present, prolongate_to_full_mixed_vector before norms, but stored arr "
                "in candidates is the post-prolongation mixed vector. Replay GNHEP uses "
                "pressure-restricted reduced W (same layout as seed.npy)."
            ),
        },
        "harvest_path": {
            "file": str(fem.relative_to(REPO_ROOT)),
            "functions": [
                "_slepc_eps_solve_and_harvest (EPS loop)",
                "eps.getEigenpair(i, rvec) -> arr_red / prolongated arr",
                "candidates.append((..., arr, ...))",
            ],
        },
        "load_path": {
            "file": str(mode_utils.relative_to(REPO_ROOT)),
            "functions": ["load_mode_column_any", "scipy.sparse.load_npz"],
            "expected_layout": (
                "CSR column shape (N_reduced_W, 1) or ravel to length N_reduced_W; "
                "must match replay operator row/col size."
            ),
        },
        "answers": {
            "full_reduced_W_eigenvector_written": (
                "Intended yes: eigvecs column length equals harvest arr.size which should match "
                "n_reduced_W when no active-domain subspace shrink is used."
            ),
            "written_before_or_after_restriction": (
                "Harvest vector is EPS rvec on operator used in solve; replay audit uses separate "
                "solve_evp=False assembly with algebraic pressure restriction on GNHEP."
            ),
            "loader_same_length_as_replay_AM": (
                "Required; audit checks loaded_vector_length vs operator_size."
            ),
            "sparse_empty_support_possible": (
                "Yes if dense_to_csr_f32_column zeroes all entries (amax=0) or EPS vector is "
                "numerically zero on M support (xH_Mx=0)."
            ),
            "mode_index_one_to_one_with_files": (
                "mode_energy_summary mode_index j maps to modes/mode_{hz_tag}_{j:03d}.smx.npz "
                "written in same loop order as freqs_hz[j]."
            ),
        },
    }


def _classify_verdict(
    *,
    seed_control: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    result_json: Dict[str, Any],
    summary_modes: List[Dict[str, Any]],
) -> Tuple[str, str]:
    if not seed_control.get("seed_xH_Mx_finite_nonzero"):
        return (
            VERDICT_REPLAY_INVALID,
            "seed_xH_Mx is zero or non-finite; replay assembly/maps inconsistent for candidates.",
        )

    loads_ok = all(c.get("vector_load_status") == "ok" for c in candidates)
    lengths_ok = all(c.get("length_matches_operator") for c in candidates)
    nnz_zero_all = all(
        int(c.get("stored_nnz_if_sparse") or 0) == 0 for c in candidates if c.get("vector_load_status") == "ok"
    )
    norms_zero_all = all(
        float(c.get("loaded_vector_norm") or 0.0) < 1e-30
        for c in candidates
        if c.get("vector_load_status") == "ok"
    )
    xhmx_zero_all = all(
        (
            not math.isfinite(float(c.get("xH_Mx", float("nan"))))
            or abs(float(c.get("xH_Mx", 0.0))) <= XH_MX_TOL
        )
        for c in candidates
        if c.get("vector_load_status") == "ok"
    )

    if not loads_ok or not lengths_ok:
        return (
            VERDICT_PERSISTENCE_BUG,
            "Candidate vector load failed or length != replay operator size (save/load/layout mismatch).",
        )

    n_saved = len(candidates)
    n_summary = len(summary_modes)
    n_in_band = len(result_json.get("in_band_modes") or [])
    if n_saved != n_summary:
        return (
            VERDICT_PERSISTENCE_BUG,
            f"mode_energy_summary count ({n_summary}) != inspected mode files ({n_saved}).",
        )

    if xhmx_zero_all and (nnz_zero_all or norms_zero_all):
        return (
            VERDICT_EPS_MASS_NULL,
            "Seed replay control valid; all saved candidates load with correct length but "
            "have zero M-norm (sparse nnz=0 or ||x||=0) — EPS harvest vectors are M-null on replay GNHEP.",
        )

    if xhmx_zero_all:
        return (
            VERDICT_EPS_MASS_NULL,
            "Seed control valid; candidates have correct layout but xH_Mx=0 for every saved vector.",
        )

    return (
        VERDICT_NOT_LOCALIZED,
        "Seed control valid and candidates are not uniformly M-null; root cause not localized by this audit.",
    )


def _write_md(report: Dict[str, Any]) -> None:
    sc = report.get("seed_control") or {}
    lines = [
        "# Unregularized-offset saved-vector mass-norm audit",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        f"**Classification:** `{report.get('classification_verdict')}`",
        "",
        report.get("classification_rationale", ""),
        "",
        "## Seed control (replay assembly)",
        "",
        f"- seed_xH_Mx: {sc.get('seed_xH_Mx')}",
        f"- seed_rayleigh_frequency_hz: {sc.get('seed_rayleigh_frequency_hz')}",
        f"- seed_relative_residual: {sc.get('seed_relative_residual')}",
        f"- seed_xH_Mx_finite_nonzero: {sc.get('seed_xH_Mx_finite_nonzero')}",
        "",
        "## Candidates",
        "",
    ]
    for c in report.get("candidates") or []:
        lines.append(
            f"- idx={c.get('candidate_index')} nnz={c.get('stored_nnz_if_sparse')} "
            f"norm={c.get('loaded_vector_norm')} xH_Mx={c.get('xH_Mx')} "
            f"load={c.get('vector_load_status')}"
        )
        if c.get("load_exception"):
            lines.append(f"  - exception: {c.get('load_exception')}")
    lines.extend(["", "## Input paths", ""])
    for k, v in (report.get("input_paths") or {}).items():
        lines.append(f"- **{k}:** `{v}`")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR
    mesh_file = mesh_path("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = (
        json.loads(seed_meta_path.read_text(encoding="utf-8")) if seed_meta_path.is_file() else {}
    )
    target_hz = float(seed_meta.get("locator_frequency_hz", SEED_F_HZ))
    fallback_sample = sample_spec_from_case(case)

    input_paths = {
        "output_dir_exclusive": str(out_dir.resolve()),
        "seed_npy": str(seed_npy.resolve()),
        "mesh_file": str(mesh_file.resolve()),
        "forbidden_trees_not_used": [
            "seed_branch_recovery_diagnostic",
            "seed_branch_recovery_diagnostic_filtered",
            "seeded_retrieval",
        ],
    }

    sample, sample_path = _load_sample_spec(out_dir, fallback_sample)
    input_paths["sample_spec"] = sample_path
    result_path = out_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    if not result_path.is_file():
        results = sorted((out_dir / "results").glob("result_*.json"))
        result_path = results[-1] if results else result_path
    input_paths["solve_result_json"] = str(result_path)

    solve_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    )
    seed_info = load_seed_with_diagnostics(seed_npy)
    summary_path = out_dir / "diagnostics" / "mode_energy_summary.json"
    summary_modes: List[Dict[str, Any]] = []
    if summary_path.is_file():
        summary_modes = json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []
    input_paths["mode_energy_summary"] = str(summary_path)

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "VM_runtime_report_only_audit",
        "input_paths": input_paths,
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "baseline_eigensolve_budget_exhausted": True,
        "vm_operator_evidence": {
            "solve_operator_consistent": True,
            "all_candidates_xH_Mx_zero_in_prior_eval": True,
            "source": "reported_from_VM_operator_evidence",
        },
        "code_trace": _static_code_trace(),
    }

    if seed_info.get("seed_load_status") != "ok" or not summary_path.is_file():
        report["classification_verdict"] = VERDICT_NOT_LOCALIZED
        report["classification_rationale"] = "Missing seed or mode_energy_summary at expected paths."
        write_json(OUT_JSON, report)
        _write_md(report)
        return 1

    seed = seed_info["seed_array"]
    A, M, u_asm, p_asm, asm_meta = assemble_replay_operators(
        mesh_file, sample, out_dir=out_dir
    )
    u_to_W, p_to_W, maps_info = resolve_maps_from_solve_result(solve_result, u_asm, p_asm)
    operator_size = int(asm_meta["operator_size"])
    restr = dict(asm_meta.get("pressure_restriction") or {})

    seed_control = _seed_control(
        A, M, seed, u_to_W=u_to_W, p_to_W=p_to_W, operator_size=operator_size
    )
    candidates: List[Dict[str, Any]] = []
    try:
        for m in summary_modes:
            rel = str(m.get("vector_path", ""))
            vec_path = (out_dir / rel).resolve()
            candidates.append(
                _candidate_row(
                    mode_row=m,
                    vec_path=vec_path,
                    A=A,
                    M=M,
                    u_to_W=u_to_W,
                    p_to_W=p_to_W,
                    operator_size=operator_size,
                    restr=restr,
                )
            )
        modes_dir = out_dir / "modes"
        glob_files = sorted(modes_dir.glob("mode_*.smx.npz")) if modes_dir.is_dir() else []
        report["modes_dir_file_count"] = len(glob_files)
        report["modes_dir_glob"] = [str(p.name) for p in glob_files]
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    verdict, rationale = _classify_verdict(
        seed_control=seed_control,
        candidates=candidates,
        result_json=solve_result,
        summary_modes=summary_modes,
    )
    report["replay_assembly"] = {**asm_meta, **maps_info}
    report["seed_control"] = seed_control
    report["candidates"] = [
        {k: v for k, v in c.items() if k != "_vec_array"} for c in candidates
    ]
    report["classification_verdict"] = verdict
    report["classification_rationale"] = rationale
    report["closure"] = {
        "permitted_follow_up": (
            "report_only_re_evaluation_if_vectors_recoverable_from_existing_artifacts"
            if verdict == VERDICT_PERSISTENCE_BUG
            else "none"
        ),
        "architecture_reconsideration_required": verdict
        in (
            VERDICT_EPS_MASS_NULL,
            VERDICT_REPLAY_INVALID,
            VERDICT_NOT_LOCALIZED,
        ),
        "no_additional_baseline_eigensolve": True,
    }

    write_json(OUT_JSON, report)
    _write_md(report)
    print(f"[mass_norm_audit] verdict={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
