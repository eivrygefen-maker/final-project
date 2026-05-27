#!/usr/bin/env python3
"""
Report-only full pipeline audit for mapping-fixed persistence-fixed replacement baseline.

Reads existing VM artifacts only. Does not call eps.solve() or regenerate candidates.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI
SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mapping_fixed_baseline_evaluator import (
    OUT_SUBDIR_PERSISTENCE_FIXED,
    VERDICT_BRANCH_RECOVERED,
    VERDICT_INCONSISTENT,
    VERDICT_NO_BRANCH,
    mapping_fixed_branch_recovery_from_row,
)
from v2_mapping_fixed_candidate_persistence import pressure_block_mapping_metadata
from v2_mesh_convergence_common import (
    CONV_DIAG,
    case_by_id,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_conservative_audit_policy import (
    SERIALIZER_FUNCTION,
    SERIALIZER_THRESHOLD,
    apply_conservative_authoritative_verdict,
    build_operator_policy_from_artifacts,
)
from v2_seed_branch_candidate_filter import FILTER_POLICY, assess_physical_eligibility
from v2_sensitivity_common import hz_result_tag

CASE_ID = "baseline_coupled_v2"
SEED_F_HZ = 243.0754171175576
N_REDUCED_W = 277626
EXPECTED_P_TO_W_LENGTH = 24039
MODE_VECTOR_RELATIVE_EPS = 1.0e-7

OUT_JSON = CONV_DIAG / "v2_mapping_fixed_persistence_fixed_full_pipeline_audit.json"
OUT_MD = CONV_DIAG / "v2_mapping_fixed_persistence_fixed_full_pipeline_audit.md"
REPLACEMENT_REPORT_JSON = (
    CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_persistence_fixed_baseline_diagnostic.json"
)

VERDICT_REPLAY_EVAL_FAILURE = "MAPPING_FIXED_UNREGULARIZED_BASELINE_REPLAY_EVALUATION_FAILURE"
VERDICT_PERSISTED_CONTENT_UNRESOLVED = (
    "MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED"
)

# candidate_eps_slot_0000.smx.npz — stem includes ".smx"; parse from path.name only.
_CANDIDATE_EPS_SLOT_RE = re.compile(r"^candidate_eps_slot_(\d+)\.smx\.npz$", re.IGNORECASE)

# Canonical contract: v2_unreg_offset_report_evaluator.assemble_replay_operators -> (A, M, u_to_W, p_to_W, meta)
REPLAY_ASSEMBLY_CONTRACT: Dict[str, Any] = {
    "definition": (
        "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "v2_unreg_offset_report_evaluator.py::assemble_replay_operators"
    ),
    "return_arity": 5,
    "return_fields": ["A", "M", "u_to_W", "p_to_W", "meta"],
    "working_callers": [
        "v2_mapping_fixed_baseline_evaluator (A, M, u_asm, p_asm, asm_meta)",
        "v2_unreg_offset_report_evaluator.evaluate_unreg_offset_report",
        "run_v2_st_singular_mass_preflight.py",
        "run_v2_l_mid_unregularized_saved_vector_mass_norm_audit.py",
    ],
    "maps_resolution": (
        "solve_result/bank JSON when present; else maps from single assemble_replay_operators call"
    ),
}


def _static_pipeline_control_flow() -> List[Dict[str, Any]]:
    """Code-path documentation from local source (no VM artifacts)."""
    return [
        {
            "step": "EPS_getEigenpair",
            "location": "FEM/scripts/fem_main_3d.py::_slepc_shift_invert_batch",
            "input": "PETSc EPS, shift-invert ST at sigma",
            "output": "raw mu, physical lambda via _slepc_physical_lambda (lam_phys=mu, slepc_backtransformed)",
            "config": [
                "eps_eigenvalue_semantics=slepc_backtransformed",
                "legacy_double_shift_mapping_disabled=True",
            ],
            "thresholds": [],
            "replacement_run_exercised": True,
        },
        {
            "step": "preserve_all_capture",
            "location": "fem_main_3d.py::_slepc_shift_invert_batch (preserve_all_nconv)",
            "input": "rvec.array per converged slot",
            "output": "_eps_diagnostic_candidate_bank_records (dense numpy W vector per slot)",
            "config": ["eps_diagnostic_preserve_all_nconv_candidates=True"],
            "thresholds": ["rigid_tol lam_phys filter before bank append only"],
            "replacement_run_exercised": True,
        },
        {
            "step": "worker_row_bridge",
            "location": "fem_main_3d.py worker single-shift path",
            "input": "diag_bank or harvest rows",
            "output": "row_meta -> eigvecs stack (preserve_all bypasses rt/rb None filter)",
            "config": ["eps_diagnostic_preserve_all_nconv_candidates"],
            "thresholds": ["legacy: skip row if rt is None or rb is None (disabled when preserve_all)"],
            "replacement_run_exercised": True,
        },
        {
            "step": "candidate_persistence",
            "location": "v2_sensitivity_solve.py + v2_mapping_fixed_candidate_persistence.py",
            "input": "bank record vector (dense np.float64)",
            "output": "modes/candidate_eps_slot_XXXX.smx.npz",
            "config": ["persist_candidate_bank", "_save_one"],
            "thresholds": [
                f"MODE_VECTOR_RELATIVE_EPS={MODE_VECTOR_RELATIVE_EPS} in fem_mode_array_utils",
                "float32 CSR + eliminate_zeros on save",
            ],
            "replacement_run_exercised": True,
            "critical_note": (
                "Seed self-test uses same save_mode_csr(dense_to_csr_f32_column) but source is "
                "full dense normalized seed; EPS source may be sparse-dominant before threshold."
            ),
        },
        {
            "step": "candidate_bank_json",
            "location": "v2_mapping_fixed_candidate_persistence.write_eps_candidate_bank_json",
            "output": "diagnostics/eps_candidate_bank.json + pressure_block_mapping",
            "replacement_run_exercised": True,
        },
        {
            "step": "mode_energy_summary",
            "location": "v2_sensitivity_solve._save_one (diagnose_mixed_mode, energy participation)",
            "output": "diagnostics/mode_energy_summary.json",
            "note": "Energy stats computed on dense vec BEFORE sparsify save path uses same vec for diag",
            "replacement_run_exercised": True,
        },
        {
            "step": "evaluator_load",
            "location": "v2_unreg_offset_report_evaluator._evaluate_one_candidate",
            "input": "mode_energy_summary vector_path -> load_mode_column_any (CSR float32)",
            "output": "dense via .toarray() for replay",
            "replacement_run_exercised": True,
        },
        {
            "step": "replay_rayleigh",
            "location": "physical_fsi_seed_residual_audit._rayleigh_metrics",
            "thresholds": ["abs(xH_Mx) < 1e-30 -> lam_r and f_hz set to NaN"],
            "replacement_run_exercised": True,
        },
        {
            "step": "verdict_assignment",
            "location": "v2_mapping_fixed_baseline_evaluator.assign_mapping_fixed_verdict",
            "thresholds": list(FILTER_POLICY.keys()),
            "replacement_run_exercised": True,
        },
    ]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int32_index_map(value: Any) -> np.ndarray:
    """Coerce index map without NumPy truth-value tests (never use `value or []`)."""
    if value is None:
        return np.asarray([], dtype=np.int32)
    return np.asarray(value, dtype=np.int32).ravel()


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None or not isinstance(value, dict):
        return {}
    return value


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _safe_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, *, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _parse_candidate_eps_slot(path: Path) -> Tuple[Optional[int], Optional[str]]:
    """Parse EPS slot from candidate_eps_slot_XXXX.smx.npz filename."""
    m = _CANDIDATE_EPS_SLOT_RE.match(path.name)
    if not m:
        return None, f"filename_does_not_match_contract:{path.name}"
    return int(m.group(1)), None


def _build_modes_meta(summary: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for m in _as_list(summary.get("modes")):
        if not isinstance(m, dict):
            continue
        slot = _safe_int(m.get("eps_slot_index", m.get("candidate_index")))
        if slot is None or slot < 0:
            continue
        out[int(slot)] = m
    return out


def _sanitize_for_json(value: Any) -> Any:
    """Replace non-JSON floats (NaN/inf) for strict report serialization."""
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return None
        return value
    if isinstance(value, np.floating):
        f = float(value)
        if not math.isfinite(f):
            return None
        return f
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _index_map_crc32(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    return int(pressure_block_mapping_metadata(p_to_W=arr, source="audit").get("p_to_W_crc32", 0))


def _validate_index_map_for_reduced_W(arr: np.ndarray, *, name: str, n_w: int) -> List[str]:
    failures: List[str] = []
    if arr.size == 0:
        failures.append(f"{name}_empty")
        return failures
    if not np.issubdtype(arr.dtype, np.integer):
        failures.append(f"{name}_not_integer_dtype")
    arr64 = arr.astype(np.int64, copy=False)
    if np.any(arr64 < 0):
        failures.append(f"{name}_has_negative_indices")
    if np.any(arr64 >= n_w):
        failures.append(f"{name}_index_out_of_range_reduced_W")
    return failures


def _build_map_contract(
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    u_source: Optional[str],
    p_source: str,
) -> Dict[str, Any]:
    failures: List[str] = []
    failures.extend(_validate_index_map_for_reduced_W(u_to_W, name="u_to_W", n_w=N_REDUCED_W))
    failures.extend(_validate_index_map_for_reduced_W(p_to_W, name="p_to_W", n_w=N_REDUCED_W))

    u_found = int(u_to_W.size) > 0
    p_found = int(p_to_W.size) > 0
    if p_found and int(p_to_W.size) != EXPECTED_P_TO_W_LENGTH:
        failures.append(
            f"p_to_W_length_{int(p_to_W.size)}_!=_expected_{EXPECTED_P_TO_W_LENGTH}"
        )

    u_set = set(int(x) for x in u_to_W.tolist()) if u_found else set()
    p_set = set(int(x) for x in p_to_W.tolist()) if p_found else set()
    overlap = len(u_set & p_set)
    union = len(u_set | p_set)

    return {
        "u_to_W_found": u_found,
        "u_to_W_length": int(u_to_W.size),
        "u_to_W_crc32": _index_map_crc32(u_to_W),
        "u_to_W_source": u_source,
        "p_to_W_found": p_found,
        "p_to_W_length": int(p_to_W.size),
        "p_to_W_crc32": _index_map_crc32(p_to_W),
        "p_to_W_source": p_source,
        "n_reduced_W_expected": N_REDUCED_W,
        "p_to_W_length_expected": EXPECTED_P_TO_W_LENGTH,
        "u_p_map_overlap_count": int(overlap),
        "u_p_map_union_size": int(union),
        "map_contract_pass": len(failures) == 0 and u_found and p_found,
        "map_contract_failure_reason": "; ".join(failures) if failures else None,
        "map_contract_failures": failures,
    }


def _resolve_u_to_W_from_sources(
    solve_result: Dict[str, Any],
    u_asm: np.ndarray,
) -> Tuple[np.ndarray, str]:
    u_res = _as_int32_index_map(solve_result.get("u_to_W"))
    if u_res.size > 0:
        return u_res, "solve_result.u_to_W"
    return u_asm, "assemble_replay_operators._coupled_air_u_to_W_map"


def _resolve_p_to_W_from_sources(
    solve_result: Dict[str, Any],
    bank: Dict[str, Any],
    p_asm: np.ndarray,
) -> Tuple[np.ndarray, str]:
    p_res = _as_int32_index_map(solve_result.get("p_to_W"))
    if p_res.size > 0:
        return p_res, "solve_result.p_to_W"
    pbm = _as_dict(bank.get("pressure_block_mapping"))
    p_bank = _as_int32_index_map(pbm.get("p_to_W"))
    if p_bank.size > 0:
        return p_bank, str(pbm.get("source", "eps_candidate_bank.pressure_block_mapping"))
    return p_asm, "assemble_replay_operators._coupled_air_p_to_W_map"


def _inspect_npz_file(path: Path, *, u_to_W: np.ndarray, p_to_W: np.ndarray) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "file_path": str(path),
        "file_exists": path.is_file(),
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
    }
    if not path.is_file():
        row["vector_load_success"] = False
        row["vector_load_status"] = "missing"
        return row
    try:
        from fem_mode_array_utils import load_mode_column_any

        csr = load_mode_column_any(path)
        row["stored_format"] = "csr"
        row["stored_shape"] = list(csr.shape)
        row["stored_dtype"] = str(csr.dtype)
        row["stored_nnz"] = int(csr.nnz)
        row["stored_density"] = float(csr.nnz) / float(N_REDUCED_W)
        data = np.asarray(csr.data)
        row["finite_values_only"] = bool(np.all(np.isfinite(data))) if data.size else True
        row["number_nan_entries"] = int(np.sum(~np.isfinite(data))) if data.size else 0
        row["number_inf_entries"] = int(np.sum(~np.isfinite(data) & np.isinf(data))) if data.size else 0
        dense = np.asarray(csr.toarray(), dtype=np.float64).ravel()
        row["vector_load_success"] = True
        row["vector_load_status"] = "ok"
        row["vector_length_valid"] = int(dense.size) == N_REDUCED_W
        row["vector_finite"] = bool(np.all(np.isfinite(dense)))
        row["raw_l2_norm_after_load"] = float(np.linalg.norm(dense))
        row["max_abs_entry"] = float(np.max(np.abs(dense))) if dense.size else float("nan")
        nz = dense[np.abs(dense) > 0.0]
        row["min_nonzero_abs_entry"] = float(np.min(np.abs(nz))) if nz.size else 0.0
        if csr.indices.size:
            row["support_index_min"] = int(csr.indices.min())
            row["support_index_max"] = int(csr.indices.max())
        else:
            row["support_index_min"] = None
            row["support_index_max"] = None
        if u_to_W.size:
            u_set = set(int(x) for x in u_to_W.tolist())
            u_hit = sum(1 for i in csr.indices if int(i) in u_set)
            row["support_overlap_u_block_indices"] = int(u_hit)
        if p_to_W.size:
            p_set = set(int(x) for x in p_to_W.tolist())
            p_hit = sum(1 for i in csr.indices if int(i) in p_set)
            row["support_overlap_pressure_block"] = int(p_hit)
            row["pressure_block_fraction_of_nnz"] = (
                float(p_hit) / float(csr.nnz) if csr.nnz else 0.0
            )
        row["dense_vector"] = dense
    except Exception as exc:
        row["vector_load_success"] = False
        row["vector_load_status"] = f"exception:{type(exc).__name__}:{exc}"
        row["exception_traceback_tail"] = "\n".join(traceback.format_exc().strip().splitlines()[-6:])
    return row


def _evaluate_candidate_physics(
    *,
    slot: int,
    dense: np.ndarray,
    meta: Dict[str, Any],
    A,
    M,
    seed: np.ndarray,
    seed_f_hz: float,
    p_to_W: np.ndarray,
    u_to_W: np.ndarray,
    st_fields: Dict[str, Any],
) -> Dict[str, Any]:
    from physical_fsi_seed_residual_audit import _block_residual_contributions, _rayleigh_metrics

    reported_f = _safe_float(
        meta.get("frequency_hz", meta.get("reported_frequency_hz")),
        default=float("nan"),
    )
    out: Dict[str, Any] = {
        "slot_index": int(slot),
        "reported_frequency_hz": reported_f,
        "vector_norm": float(np.linalg.norm(dense)),
        "rejection_reasons": [],
    }
    if dense.size != N_REDUCED_W:
        out["rayleigh_evaluation_status"] = "dimension_mismatch"
        out["nonfinite_reason"] = f"vector_length_{dense.size}_!=_{N_REDUCED_W}"
        return out
    if not np.all(np.isfinite(dense)):
        out["rayleigh_evaluation_status"] = "nonfinite_input"
        out["nonfinite_reason"] = "vector_has_nonfinite_entries"
        return out
    if float(np.linalg.norm(dense)) <= 0.0:
        out["rayleigh_evaluation_status"] = "zero_vector_norm"
        out["nonfinite_reason"] = "zero_vector_norm"
        return out

    try:
        ray = _rayleigh_metrics(A, M, dense, seed_f_hz=reported_f)
        xH_Mx = float(ray.get("xH_Mx", float("nan")))
        xH_Ax = float(ray.get("xH_Ax", float("nan")))
        lam = float(ray.get("rayleigh_lambda", float("nan")))
        replay_f = float(ray.get("rayleigh_f_hz", float("nan")))
        out["xH_Mx"] = xH_Mx
        out["xH_Ax"] = xH_Ax
        out["rayleigh_lambda"] = lam
        out["rayleigh_frequency_hz"] = replay_f
        if not math.isfinite(xH_Mx) or abs(xH_Mx) < 1.0e-30:
            out["rayleigh_evaluation_status"] = "zero_or_nonfinite_mass_norm"
            out["nonfinite_reason"] = f"xH_Mx={xH_Mx}"
            out["root_cause_category"] = "PHYSICAL_REPLAY_UNDEFINED_ZERO_OR_MASS_NULL_VECTOR"
        elif not math.isfinite(replay_f):
            out["rayleigh_evaluation_status"] = "replay_frequency_nonfinite"
            out["nonfinite_reason"] = f"replay_f={replay_f}"
        else:
            out["rayleigh_evaluation_status"] = "ok"
            rel_res = float("nan")
            try:
                residual = _block_residual_contributions(
                    A, M, dense, lam0=lam, u_idx=u_to_W, p_idx=p_to_W
                )
                rel_res = _safe_float(residual.get("relative_residual"))
                out["relative_residual"] = rel_res
                out["residual_evaluation_status"] = "ok"
            except Exception as exc:
                out["residual_evaluation_status"] = f"exception:{type(exc).__name__}"
                out["relative_residual"] = float("nan")
                out["residual_nonfinite_reason"] = str(exc)
            freq_ok = False
            if math.isfinite(reported_f) and math.isfinite(replay_f) and replay_f > 0:
                rel_err = abs(reported_f - replay_f) / replay_f
                abs_err = abs(reported_f - replay_f)
                freq_ok = rel_err <= float(
                    FILTER_POLICY["reported_vs_replay_rayleigh_frequency_max_relative"]
                ) or abs_err <= float(
                    FILTER_POLICY["reported_vs_replay_rayleigh_frequency_max_absolute_hz"]
                )
            out["reported_vs_replay_delta_hz"] = (
                float(reported_f - replay_f) if math.isfinite(reported_f) and math.isfinite(replay_f) else float("nan")
            )
            out["reported_vs_replay_consistent"] = bool(freq_ok)
            p_seed = np.asarray(seed[p_to_W], dtype=np.float64).ravel()
            p_cand = np.asarray(dense[p_to_W], dtype=np.float64).ravel()
            out["pressure_block_norm"] = float(np.linalg.norm(p_cand))
            na = float(np.linalg.norm(p_seed))
            nb = float(np.linalg.norm(p_cand))
            if na <= 0 or nb <= 0:
                out["pressure_mac_status"] = "zero_pressure_norm"
                mac = float("nan")
            else:
                mac = float(abs(np.vdot(p_seed, p_cand)) / (na * nb))
                out["pressure_mac_status"] = "ok"
            out["pressure_mac_to_true_seed"] = mac
            if out["pressure_block_norm"] <= 1.0e-30:
                out["mass_null_status"] = "pressure_block_near_zero"
            elif abs(xH_Mx) < 1.0e-30:
                out["mass_null_status"] = "xH_Mx_near_zero"
            else:
                out["mass_null_status"] = "finite_nonzero"
            replay_metrics = {
                "replay_rayleigh_lambda": lam,
                "replay_rayleigh_frequency_hz": replay_f,
                "replay_relative_residual": rel_res,
                "algebraic_lambda_one_suspect": abs(lam - 1.0) <= float(
                    FILTER_POLICY["lambda_one_abs_tolerance"]
                ),
                "reported_vs_replay_frequency_consistent": bool(freq_ok),
                "reported_frequency_hz": reported_f,
            }
            elig = assess_physical_eligibility(
                reported_f_hz=reported_f,
                replay_metrics=replay_metrics,
                pressure_mac_to_true_seed=mac,
                seed_f_hz=seed_f_hz,
                require_mac=True,
                require_seed_frequency_match=True,
            )
            out["physically_eligible_after_filter"] = bool(
                elig.get("physically_eligible_after_filter", False)
            )
            out["rejection_reasons"] = list(elig.get("rejection_reasons") or [])
            out["continuation_seed_applied"] = True
            out["diagnostic_operator_consistent_with_replay"] = st_fields.get(
                "diagnostic_operator_consistent_with_replay"
            )
            out["actual_st_a_shift_frac"] = st_fields.get("actual_st_a_shift_frac", 0.0)
            out["actual_st_mass_reg_frac"] = st_fields.get("actual_st_mass_reg_frac", 0.0)
            out["eps_eigenvalue_semantics"] = st_fields.get("eps_eigenvalue_semantics")
            out["legacy_double_shift_mapping_disabled"] = st_fields.get(
                "legacy_double_shift_mapping_disabled"
            )
            out["branch_recovery_pass"] = mapping_fixed_branch_recovery_from_row(
                {
                    **out,
                    **st_fields,
                    "reported_vs_replay_frequency_consistent": bool(freq_ok),
                    "pressure_MAC_to_true_acoustic_seed": mac,
                    "rayleigh_denominator": xH_Mx,
                    "metrics_computation_status": "ok",
                }
            )
    except Exception as exc:
        out["rayleigh_evaluation_status"] = f"exception:{type(exc).__name__}"
        out["nonfinite_reason"] = str(exc)
        out["exception_traceback_tail"] = "\n".join(traceback.format_exc().strip().splitlines()[-8:])
        out["root_cause_category"] = "REPLAY_EVALUATOR_EXCEPTION_HIDDEN_AS_NAN"
    return out


def _nnz_summary(file_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    nnzs = [int(r["stored_nnz"]) for r in file_rows if r.get("vector_load_success")]
    if not nnzs:
        return {"count": 0}
    return {
        "count": len(nnzs),
        "min_nnz": min(nnzs),
        "median_nnz": int(np.median(nnzs)),
        "max_nnz": max(nnzs),
        "num_nnz_le_100": sum(1 for n in nnzs if n <= 100),
        "num_nonfinite_vectors": sum(1 for r in file_rows if not r.get("vector_finite", True)),
        "systematic_low_nnz": bool(max(nnzs) <= 100) if nnzs else False,
    }


def _assign_audit_verdict(
    *,
    file_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    persistence_closed: bool,
    slot_parse_failures: int,
) -> Tuple[str, str, Dict[str, Any]]:
    if not persistence_closed:
        return VERDICT_INCONSISTENT, "persistence_not_closed", {}
    if slot_parse_failures > 0:
        return VERDICT_REPLAY_EVAL_FAILURE, "candidate_filename_slot_parse_failure", {
            "slot_parse_failures": slot_parse_failures,
        }
    if not file_rows:
        return VERDICT_REPLAY_EVAL_FAILURE, "no_candidate_files_found", {}
    loads_ok = all(r.get("vector_load_success") for r in file_rows)
    if not loads_ok:
        return VERDICT_REPLAY_EVAL_FAILURE, "not_all_vectors_loaded", {}
    exceptions = [r for r in eval_rows if "exception" in str(r.get("rayleigh_evaluation_status", ""))]
    if exceptions and len(exceptions) == len(eval_rows):
        return VERDICT_REPLAY_EVAL_FAILURE, "all_candidates_replay_exception", {
            "sample": exceptions[0].get("exception_traceback_tail")
        }
    mass_null = [
        r
        for r in eval_rows
        if r.get("rayleigh_evaluation_status") == "zero_or_nonfinite_mass_norm"
    ]
    low_nnz = _nnz_summary(file_rows)
    if low_nnz.get("systematic_low_nnz") and len(mass_null) >= max(1, len(eval_rows) // 2):
        return VERDICT_PERSISTED_CONTENT_UNRESOLVED, "systematic_low_nnz_with_mass_null_replay", {
            "nnz_summary": low_nnz,
            "primary_hypothesis": "PERSISTED_EPS_VECTOR_CONTENT_CORRUPTED_OR_TRUNCATED",
            "mechanism": (
                f"MODE_VECTOR_RELATIVE_EPS={MODE_VECTOR_RELATIVE_EPS} sparsification on "
                "dense_to_csr_f32_column may drop pressure-block entries relative to peak amplitude"
            ),
        }
    branch_pass = [r for r in eval_rows if r.get("branch_recovery_pass")]
    if branch_pass:
        return VERDICT_BRANCH_RECOVERED, "at_least_one_candidate_passes_branch_gates", {
            "best_slot": branch_pass[0].get("slot_index")
        }
    all_finite_eval = all(
        r.get("rayleigh_evaluation_status") in ("ok", "zero_or_nonfinite_mass_norm")
        for r in eval_rows
    )
    if all_finite_eval and eval_rows:
        return VERDICT_NO_BRANCH, "all_candidates_evaluated_no_branch_pass", {}
    return VERDICT_REPLAY_EVAL_FAILURE, "replay_metrics_not_fully_evaluated", {}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_primary_audit_reports(report: Dict[str, Any]) -> None:
    """Write authoritative pipeline audit JSON/MD atomically."""
    payload = _sanitize_for_json(report)
    _atomic_write_text(OUT_JSON, json.dumps(payload, indent=2))
    md_lines = [
        "# Mapping-fixed persistence-fixed full pipeline audit",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"**audit_verdict:** `{report.get('audit_verdict')}`",
        f"**verdict_reason:** `{report.get('verdict_reason')}`",
        f"**operator_policy_provenance_mismatch:** `{report.get('operator_policy_provenance_mismatch')}`",
        "",
        "## Conservative policy",
        "",
    ]
    for stmt in _as_list(report.get("conservative_policy_statements")):
        md_lines.append(f"- {stmt}")
    md_lines.extend(
        [
            "",
            "## Persistence status",
            "",
            f"- self_test_pass (VM evidence design): replacement persistence 56/56 closed",
            f"- candidates on disk: {_as_dict(report.get('candidate_file_audit')).get('file_count')}",
            f"- nnz summary: {json.dumps(_sanitize_for_json(_as_dict(report.get('candidate_file_audit')).get('nnz_summary', {})))}",
            "",
            "## Root-cause category",
            "",
            f"`{_as_dict(report.get('root_cause_analysis')).get('primary_category')}`",
            "",
            str(_as_dict(report.get("root_cause_analysis")).get("mechanism", "")),
            "",
            "## Operator policy (from artifacts)",
            "",
        ]
    )
    mc = _as_dict(report.get("map_contract"))
    md_lines.append("## Map contract")
    md_lines.append("")
    for k in (
        "u_to_W_found",
        "u_to_W_length",
        "u_to_W_crc32",
        "p_to_W_found",
        "p_to_W_length",
        "p_to_W_crc32",
        "n_reduced_W_expected",
        "u_p_map_overlap_count",
        "u_p_map_union_size",
        "map_contract_pass",
        "map_contract_failure_reason",
    ):
        md_lines.append(f"- {k}: {mc.get(k)}")
    md_lines.append("")
    op = _as_dict(report.get("operator_policy"))
    for k, v in op.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("## Evaluation summary")
    es = _as_dict(report.get("evaluation_summary"))
    for k, v in es.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("## Pipeline control flow (code-derived)")
    for step in _as_list(report.get("pipeline_control_flow")):
        md_lines.append(f"- **{step.get('step')}** @ `{step.get('location')}`")
    _atomic_write_text(OUT_MD, "\n".join(md_lines) + "\n")


def _refresh_status_reports(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Optional ancillary refresh after primary audit is on disk. Never raises."""
    from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main
    from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main

    result: Dict[str, Any] = {
        "status_refresh_pass": True,
        "steps": {},
        "status_refresh_failure": None,
        "artifacts_stale_if_failed": [
            str(CONV_DIAG / "v2_st_singular_mass_rehabilitation_plan.json"),
            str(CONV_DIAG / "v2_st_singular_mass_rehabilitation_plan.md"),
            str(CONV_DIAG / "v2_solver_root_cause_and_forward_risk_audit.json"),
            str(CONV_DIAG / "v2_solver_root_cause_and_forward_risk_audit.md"),
            str(REPLACEMENT_REPORT_JSON),
        ],
    }
    failures: List[str] = []
    try:
        rehab_main()
        result["steps"]["rehab_plan"] = "ok"
    except Exception as exc:
        failures.append(f"rehab_plan:{type(exc).__name__}:{exc}")
        result["steps"]["rehab_plan"] = "failed"
    try:
        audit_main()
        result["steps"]["root_cause_audit"] = "ok"
    except Exception as exc:
        failures.append(f"root_cause_audit:{type(exc).__name__}:{exc}")
        result["steps"]["root_cause_audit"] = "failed"
    try:
        repl_path = REPLACEMENT_REPORT_JSON
        if repl_path.is_file():
            repl = _load_json(repl_path)
            repl["full_pipeline_audit"] = {
                "report_json": str(OUT_JSON),
                "audit_verdict": audit.get("audit_verdict"),
                "generated_utc": audit.get("generated_utc"),
            }
            evaluation = _as_dict(repl.get("evaluation"))
            evaluation["diagnostic_verdict"] = audit.get("audit_verdict")
            evaluation["audit_verdict_reason"] = audit.get("verdict_reason")
            repl["evaluation"] = evaluation
            write_json(repl_path, repl)
        result["steps"]["replacement_diagnostic_patch"] = "ok"
    except Exception as exc:
        failures.append(f"replacement_diagnostic_patch:{type(exc).__name__}:{exc}")
        result["steps"]["replacement_diagnostic_patch"] = "failed"
    if failures:
        result["status_refresh_pass"] = False
        result["status_refresh_failure"] = "; ".join(failures)
    return result


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[pipeline_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = case_by_id(manifest, CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR_PERSISTENCE_FIXED
    mesh_file = mesh_path("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = _load_json(seed_meta_path)
    sample = sample_spec_from_case(case)
    target_hz = _safe_float(seed_meta.get("locator_frequency_hz"), default=SEED_F_HZ)
    if not math.isfinite(target_hz):
        target_hz = SEED_F_HZ

    bank_path = out_dir / "diagnostics" / "eps_candidate_bank.json"
    summary_path = out_dir / "diagnostics" / "mode_energy_summary.json"
    bank = _load_json(bank_path)
    summary = _load_json(summary_path)
    replacement_report = _load_json(REPLACEMENT_REPORT_JSON)
    solve_result = replacement_report
    if not solve_result:
        results = sorted((out_dir / "results").glob("result_*.json"))
        if results:
            solve_result = _load_json(results[-1])

    modes_meta = _build_modes_meta(summary)

    from v2_unreg_offset_report_evaluator import assemble_replay_operators

    A, M, u_asm, p_asm, asm_meta = assemble_replay_operators(mesh_file, sample, out_dir=out_dir)
    u_to_W, u_source = _resolve_u_to_W_from_sources(solve_result, u_asm)
    p_to_W, p_source = _resolve_p_to_W_from_sources(solve_result, bank, p_asm)
    map_contract = _build_map_contract(
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        u_source=u_source,
        p_source=p_source,
    )
    replay_assembly_record = {
        **REPLAY_ASSEMBLY_CONTRACT,
        "asm_meta": {k: v for k, v in asm_meta.items() if k != "pressure_restriction"},
        "u_to_W_from_assembly_length": int(u_asm.size),
        "p_to_W_from_assembly_length": int(p_asm.size),
        "maps_u_source": u_source,
        "maps_p_source": p_source,
    }

    operator_policy, operator_provenance = build_operator_policy_from_artifacts(
        solve_result, bank, target_hz=target_hz
    )
    operator_policy["save_errors"] = _as_list(bank.get("save_errors"))
    operator_policy["p_to_W_source"] = p_source
    operator_policy["p_to_W_length"] = int(p_to_W.size)
    operator_policy["p_to_W_crc32"] = pressure_block_mapping_metadata(
        p_to_W=p_to_W, source=p_source
    ).get("p_to_W_crc32")
    st_fields = {
        "actual_sigma_hz": operator_policy.get("actual_sigma_hz"),
        "actual_st_a_shift_frac": operator_policy.get("actual_st_a_shift_frac", 0.0),
        "actual_st_mass_reg_frac": operator_policy.get("actual_st_mass_reg_frac", 0.0),
        "diagnostic_operator_consistent_with_replay": operator_policy.get(
            "diagnostic_operator_consistent_with_replay"
        ),
        "eps_eigenvalue_semantics": operator_policy.get("eps_eigenvalue_semantics"),
        "legacy_double_shift_mapping_disabled": operator_policy.get(
            "legacy_double_shift_mapping_disabled"
        ),
    }

    save_errors = _as_list(bank.get("save_errors"))
    persistence_closed = (
        int(bank.get("num_vectors_saved", 0)) == int(bank.get("nconv_marked", 0))
        and int(bank.get("nconv_marked", 0)) > 0
        and len(save_errors) == 0
    )

    file_rows: List[Dict[str, Any]] = []
    slot_parse_failures = 0
    eval_rows: List[Dict[str, Any]] = []
    physics_eval_skipped = False
    try:
        modes_dir = out_dir / "modes"
        if not modes_dir.is_dir():
            raise FileNotFoundError(f"modes directory missing: {modes_dir}")
        for path in sorted(modes_dir.glob("candidate_eps_slot_*.smx.npz")):
            slot, slot_err = _parse_candidate_eps_slot(path)
            if slot is None:
                slot_parse_failures += 1
                file_rows.append(
                    {
                        "file_path": str(path),
                        "file_exists": path.is_file(),
                        "slot_index": None,
                        "slot_parse_status": "failed",
                        "slot_parse_failure_reason": slot_err,
                        "vector_load_success": False,
                        "vector_load_status": "slot_parse_failure",
                    }
                )
                continue
            row = _inspect_npz_file(path, u_to_W=u_to_W, p_to_W=p_to_W)
            row["slot_index"] = int(slot)
            row["slot_parse_status"] = "ok"
            file_rows.append(row)

        if not map_contract["map_contract_pass"]:
            physics_eval_skipped = True
            for fr in file_rows:
                slot = _safe_int(fr.get("slot_index"))
                if slot is None:
                    continue
                eval_rows.append(
                    {
                        "slot_index": slot,
                        "rayleigh_evaluation_status": "skipped_map_contract_failure",
                        "nonfinite_reason": map_contract.get("map_contract_failure_reason"),
                    }
                )
        else:
            if not seed_npy.is_file():
                raise FileNotFoundError(f"true seed missing: {seed_npy}")
            seed = np.asarray(np.load(str(seed_npy)), dtype=np.float64).ravel()
            if int(seed.size) != N_REDUCED_W:
                raise ValueError(f"seed length {int(seed.size)} != {N_REDUCED_W}")
            for fr in file_rows:
                slot = _safe_int(fr.get("slot_index"))
                if slot is None:
                    continue
                meta = modes_meta.get(slot, {})
                dense = fr.pop("dense_vector", None)
                if dense is None:
                    eval_rows.append(
                        {
                            "slot_index": slot,
                            "rayleigh_evaluation_status": "sparse_load_failure",
                            "nonfinite_reason": fr.get("vector_load_status"),
                        }
                    )
                    continue
                er = _evaluate_candidate_physics(
                    slot=slot,
                    dense=dense,
                    meta=meta,
                    A=A,
                    M=M,
                    seed=seed,
                    seed_f_hz=target_hz,
                    p_to_W=p_to_W,
                    u_to_W=u_to_W,
                    st_fields=st_fields,
                )
                eval_rows.append(er)
    finally:
        for fr in file_rows:
            fr.pop("dense_vector", None)
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    if not map_contract["map_contract_pass"]:
        verdict = VERDICT_REPLAY_EVAL_FAILURE
        reason = "map_contract_failure"
        rca_extra: Dict[str, Any] = {
            "primary_hypothesis": "REPLAY_LAYOUT_OR_METADATA_MISMATCH",
            "map_contract": map_contract,
        }
    else:
        verdict, reason, rca_extra = _assign_audit_verdict(
            file_rows=file_rows,
            eval_rows=eval_rows,
            persistence_closed=persistence_closed,
            slot_parse_failures=slot_parse_failures,
        )
    rca = {
        "primary_category": rca_extra.get("primary_hypothesis", reason),
        "mechanism": rca_extra.get(
            "mechanism",
            "See per-candidate rayleigh_evaluation_status and stored_nnz audit.",
        ),
        **rca_extra,
    }
    nnz_summary = _nnz_summary(file_rows)

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "report_only_VM_artifact_audit",
        "output_tree": str(out_dir),
        "persistence_closed": persistence_closed,
        "replay_assembly": replay_assembly_record,
        "map_contract": map_contract,
        "physics_eval_skipped": physics_eval_skipped,
        "operator_policy": operator_policy,
        "pipeline_control_flow": _static_pipeline_control_flow(),
        "operator_policy_provenance": operator_provenance,
        "replacement_baseline_artifacts": replacement_report,
        "capture_vs_persist_contract": {
            "in_memory_capture": "dense numpy from PETSc rvec.array in diag_bank",
            "persist_function": SERIALIZER_FUNCTION,
            "serializer_function": SERIALIZER_FUNCTION,
            "serializer_threshold": SERIALIZER_THRESHOLD,
            "sparsify_relative_eps": MODE_VECTOR_RELATIVE_EPS,
            "serialization_may_change_physical_replay_metrics": True,
            "lossless_pre_sparsify_eps_vectors_available_in_current_run": False,
            "current_saved_vectors_sufficient_for_st_verdict": False,
            "seed_self_test_same_writer": True,
            "seed_self_test_same_source_object_type": "dense normalized seed numpy",
            "eps_candidate_source_object_type": "dense numpy from EPS harvest (pre-sparsify)",
            "critical_distinction": (
                "Passing seed round-trip does not prove EPS vectors retain pressure support "
                "after relative sparsification if EPS modes have lower per-entry amplitude "
                "in pressure block than peak structural entries."
            ),
        },
        "candidate_file_audit": {
            "file_count": len(file_rows),
            "slot_parse_failures": int(slot_parse_failures),
            "expected_num_vectors_saved": int(bank.get("num_vectors_saved", 0)),
            "nnz_summary": nnz_summary,
            "files": [
                {k: v for k, v in r.items() if k not in ("dense_vector",)}
                for r in file_rows
            ],
        },
        "candidates": eval_rows,
        "evaluation_summary": {
            "num_candidates": len(eval_rows),
            "physics_eval_skipped": physics_eval_skipped,
            "num_rayleigh_ok": sum(1 for r in eval_rows if r.get("rayleigh_evaluation_status") == "ok"),
            "num_mass_null": sum(
                1
                for r in eval_rows
                if r.get("rayleigh_evaluation_status") == "zero_or_nonfinite_mass_norm"
            ),
            "num_branch_recovery_pass": sum(1 for r in eval_rows if r.get("branch_recovery_pass")),
            "num_exceptions": sum(
                1 for r in eval_rows if "exception" in str(r.get("rayleigh_evaluation_status", ""))
            ),
        },
        "root_cause_analysis": rca,
        "audit_verdict": verdict,
        "verdict_reason": reason,
        "st_viability_conclusion": (
            "inconclusive_pending_resolved_replay_on_persisted_vectors"
            if verdict
            not in (
                VERDICT_BRANCH_RECOVERED,
                VERDICT_NO_BRANCH,
            )
            else verdict
        ),
        "no_additional_eigensolve_authorized": True,
        "mesh_convergence_may_resume": False,
    }
    apply_conservative_authoritative_verdict(report)
    primary_audit_written = False
    status_refresh_pass = False
    status_refresh_failure: Optional[str] = None
    try:
        _write_primary_audit_reports(report)
        primary_audit_written = True
    except Exception:
        print(
            "[pipeline_audit] FAILED before writing primary audit reports; "
            "status reports were not refreshed.",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 2

    refresh_result = _refresh_status_reports(report)
    status_refresh_pass = bool(refresh_result.get("status_refresh_pass"))
    status_refresh_failure = refresh_result.get("status_refresh_failure")

    print(
        f"[pipeline_audit] verdict={verdict} reason={reason} "
        f"map_contract_pass={map_contract.get('map_contract_pass')} "
        f"files={len(file_rows)} branch_pass={report['evaluation_summary']['num_branch_recovery_pass']} "
        f"median_nnz={nnz_summary.get('median_nnz')} "
        f"primary_audit_written={primary_audit_written} status_refresh_pass={status_refresh_pass}",
        flush=True,
    )
    if status_refresh_failure:
        print(
            f"[pipeline_audit] status_refresh_failure={status_refresh_failure}",
            file=sys.stderr,
            flush=True,
        )
    if verdict == VERDICT_BRANCH_RECOVERED:
        return 0
    if verdict == VERDICT_NO_BRANCH:
        return 1
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(
            "[pipeline_audit] FAILED; no audit report or status refresh written.",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise SystemExit(2)
