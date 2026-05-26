#!/usr/bin/env python3
"""
Report-only evaluator for completed unregularized-offset baseline diagnostic artifacts.

Reads exclusively from seed_branch_recovery_diagnostic_unregularized_offset/ (no EPS).
"""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FORBIDDEN_INPUT_SUBDIRS = (
    "seed_branch_recovery_diagnostic",
    "seed_branch_recovery_diagnostic_filtered",
    "seeded_retrieval",
)

VERDICT_BRANCH_RECOVERED = "UNREGULARIZED_OFFSET_BASELINE_BRANCH_RECOVERED"
VERDICT_NO_BRANCH = "UNREGULARIZED_OFFSET_BASELINE_NO_PHYSICAL_BRANCH_RECOVERED"
VERDICT_INCONSISTENT = "UNREGULARIZED_OFFSET_OUTPUT_OR_REPLAY_INCONSISTENT"


def _path_under(base: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _load_sample_spec(out_dir: Path, fallback_sample: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    spec_path = out_dir / "sample_spec.json"
    if spec_path.is_file():
        return json.loads(spec_path.read_text(encoding="utf-8")), str(spec_path)
    return fallback_sample, str(spec_path) + " (missing; used manifest fallback)"


def assemble_replay_operators(
    mesh_file: Path,
    sample: Dict[str, Any],
    *,
    out_dir: Path,
) -> Tuple[Any, Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    import fem_main_3d as fem3d
    from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay

    sort_dir = out_dir / "sorting_report_eval"
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    meta = {
        "sorting_root": str(sort_dir),
        "operator_size": int(A.getSize()[0]),
        "n_u_active": int(u_to_W.size),
        "n_p_active": int(p_to_W.size),
        "replay_assembly_status": "ok",
    }
    return A, M, u_to_W, p_to_W, meta


def resolve_maps_from_solve_result(
    solve_result: Dict[str, Any],
    assembled_u: np.ndarray,
    assembled_p: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    u_res = np.asarray(solve_result.get("u_to_W") or [], dtype=np.int32).ravel()
    p_res = np.asarray(solve_result.get("p_to_W") or [], dtype=np.int32).ravel()
    info: Dict[str, Any] = {
        "u_to_W_from_result_json": int(u_res.size),
        "p_to_W_from_result_json": int(p_res.size),
        "u_to_W_from_fresh_assembly": int(assembled_u.size),
        "p_to_W_from_fresh_assembly": int(assembled_p.size),
    }
    if u_res.size > 0 and p_res.size > 0:
        info["maps_source"] = "result_243075.json"
        info["u_maps_match_assembly"] = bool(np.array_equal(u_res, assembled_u))
        info["p_maps_match_assembly"] = bool(np.array_equal(p_res, assembled_p))
        return u_res, p_res, info
    info["maps_source"] = "fresh_assembly_fallback"
    info["u_maps_match_assembly"] = True
    info["p_maps_match_assembly"] = True
    return assembled_u, assembled_p, info


def load_seed_with_diagnostics(seed_npy: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "seed_file": str(seed_npy),
        "seed_file_exists": seed_npy.is_file(),
        "seed_load_status": "missing",
        "seed_vector_length": None,
        "seed_vector_norm": float("nan"),
    }
    if not seed_npy.is_file():
        return out
    try:
        seed = np.asarray(np.load(str(seed_npy)), dtype=np.float64).ravel()
        out["seed_load_status"] = "ok"
        out["seed_vector_length"] = int(seed.size)
        out["seed_vector_norm"] = float(np.linalg.norm(seed))
        out["seed_array"] = seed
    except Exception as exc:
        out["seed_load_status"] = f"load_failed:{type(exc).__name__}:{exc}"
        out["seed_array"] = None
    return out


def _compute_pressure_mac(
    p_seed: np.ndarray, p_cand: np.ndarray
) -> Tuple[float, str, float, float]:
    a = np.asarray(p_seed, dtype=np.float64).ravel()
    b = np.asarray(p_cand, dtype=np.float64).ravel()
    if a.size != b.size:
        return float("nan"), f"block_length_mismatch:{a.size}!={b.size}", float(np.linalg.norm(a)), float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0:
        return float("nan"), "zero_seed_pressure_norm", na, nb
    if nb <= 0.0:
        return float("nan"), "zero_candidate_pressure_norm", na, nb
    mac = float(abs(np.vdot(a, b)) / (na * nb))
    return mac, "ok", na, nb


def _evaluate_one_candidate(
    *,
    mode_row: Dict[str, Any],
    out_dir: Path,
    seed: np.ndarray,
    seed_f_hz: float,
    continuation: bool,
    st_fields: Dict[str, Any],
    op_ok: bool,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    operator_size: int,
    A,
    M,
    assess_physical_eligibility,
    branch_recovery_from_row,
    replay_candidate_metrics,
) -> Dict[str, Any]:
    from physical_fsi_seed_residual_audit import _rayleigh_metrics

    rel = str(mode_row.get("vector_path", ""))
    vec_path = (out_dir / rel).resolve()
    row: Dict[str, Any] = {
        "candidate_index": mode_row.get("mode_index"),
        "vector_file": str(vec_path),
        "vector_file_exists": vec_path.is_file(),
        "vector_load_status": "pending",
        "vector_length": None,
        "vector_norm": float("nan"),
        "reported_frequency_hz": float(mode_row.get("frequency_hz", float("nan"))),
        "reported_p_frac_energy_phys": mode_row.get("p_frac_energy_phys"),
        "reported_mode_class_physical_energy": mode_row.get("mode_class_physical_energy"),
        "seed_file_exists": True,
        "seed_load_status": "ok",
        "seed_vector_length": int(seed.size),
        "seed_vector_norm": float(np.linalg.norm(seed)),
        "p_to_W_available": bool(p_to_W.size > 0),
        "p_to_W_length": int(p_to_W.size),
        "operator_size": int(operator_size),
        "metrics_computation_status": "pending",
        "nonfinite_reason": None,
        "continuation_seed_applied": continuation,
        **st_fields,
    }

    forbidden_used: List[str] = []
    for sub in FORBIDDEN_INPUT_SUBDIRS:
        if sub in vec_path.parts:
            forbidden_used.append(sub)
    if forbidden_used:
        row["vector_load_status"] = "forbidden_path"
        row["metrics_computation_status"] = "forbidden_input_tree"
        row["nonfinite_reason"] = f"vector_path_under_forbidden_subdir:{forbidden_used}"
        row["rejection_reasons"] = [row["nonfinite_reason"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    if not _path_under(out_dir, vec_path):
        row["vector_load_status"] = "outside_output_tree"
        row["metrics_computation_status"] = "path_outside_unregularized_offset_tree"
        row["nonfinite_reason"] = "vector_file_not_under_seed_branch_recovery_diagnostic_unregularized_offset"
        row["rejection_reasons"] = [row["nonfinite_reason"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    if not vec_path.is_file():
        row["vector_load_status"] = "missing"
        row["metrics_computation_status"] = "vector_file_missing"
        row["nonfinite_reason"] = "vector_file_not_found"
        row["rejection_reasons"] = [row["nonfinite_reason"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    try:
        from fem_mode_array_utils import load_mode_column_any

        vec = np.asarray(load_mode_column_any(vec_path).toarray(), dtype=np.float64).ravel()
        row["vector_load_status"] = "ok"
        row["vector_length"] = int(vec.size)
        row["vector_norm"] = float(np.linalg.norm(vec))
    except Exception as exc:
        row["vector_load_status"] = f"load_failed:{type(exc).__name__}:{exc}"
        row["metrics_computation_status"] = "vector_load_failed"
        row["nonfinite_reason"] = row["vector_load_status"]
        row["rejection_reasons"] = [row["nonfinite_reason"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    if int(vec.size) != int(operator_size):
        row["metrics_computation_status"] = "vector_operator_length_mismatch"
        row["nonfinite_reason"] = f"vector_length_{vec.size}_!=_operator_size_{operator_size}"
        row["rejection_reasons"] = [row["nonfinite_reason"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    if p_to_W.size == 0:
        row["pressure_block_extraction_status"] = "p_to_W_empty"
        row["metrics_computation_status"] = "p_to_W_empty"
        row["nonfinite_reason"] = "p_to_W_empty"
        row["rejection_reasons"] = [row["nonfinite_reason"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    try:
        p_seed_blk = np.asarray(seed[p_to_W], dtype=np.float64).ravel()
        p_cand_blk = np.asarray(vec[p_to_W], dtype=np.float64).ravel()
        row["pressure_block_extraction_status"] = "ok"
    except Exception as exc:
        row["pressure_block_extraction_status"] = f"index_error:{type(exc).__name__}:{exc}"
        row["metrics_computation_status"] = "pressure_block_extraction_failed"
        row["nonfinite_reason"] = row["pressure_block_extraction_status"]
        row["rejection_reasons"] = [row["nonfinite_reason"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    mac, mac_status, seed_p_norm, cand_p_norm = _compute_pressure_mac(p_seed_blk, p_cand_blk)
    row["seed_pressure_norm"] = seed_p_norm
    row["candidate_pressure_norm"] = cand_p_norm
    row["pressure_mac_computation_status"] = mac_status
    row["pressure_MAC_to_true_acoustic_seed_raw"] = mac
    row["pressure_MAC_to_true_acoustic_seed"] = mac

    row["replay_assembly_status"] = "pending"
    row["replay_exception_if_any"] = None
    row["rayleigh_numerator"] = float("nan")
    row["rayleigh_denominator"] = float("nan")
    row["rayleigh_lambda_raw"] = float("nan")
    row["rayleigh_frequency_raw"] = float("nan")
    row["replay_residual_raw"] = float("nan")

    try:
        rayleigh = _rayleigh_metrics(
            A, M, vec, seed_f_hz=float(row["reported_frequency_hz"])
        )
        row["rayleigh_numerator"] = float(rayleigh.get("xH_Ax", float("nan")))
        row["rayleigh_denominator"] = float(rayleigh.get("xH_Mx", float("nan")))
        row["rayleigh_lambda_raw"] = float(rayleigh.get("rayleigh_lambda", float("nan")))
        row["rayleigh_frequency_raw"] = float(rayleigh.get("rayleigh_f_hz", float("nan")))
        replay = replay_candidate_metrics(
            A,
            M,
            vec,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            reported_f_hz=float(row["reported_frequency_hz"]),
        )
        row["replay_assembly_status"] = "ok"
        row["replay_relative_residual"] = float(replay["replay_relative_residual"])
        row["replay_residual_raw"] = row["replay_relative_residual"]
        row["replay_rayleigh_lambda"] = float(replay["replay_rayleigh_lambda"])
        row["replay_rayleigh_frequency_hz"] = float(replay["replay_rayleigh_frequency_hz"])
        row["algebraic_lambda_one_suspect"] = bool(replay["algebraic_lambda_one_suspect"])
        row["reported_vs_replay_frequency_consistent"] = bool(
            replay["reported_vs_replay_frequency_consistent"]
        )
    except Exception as exc:
        row["replay_assembly_status"] = "exception"
        row["replay_exception_if_any"] = f"{type(exc).__name__}:{exc}\n{traceback.format_exc()}"
        row["metrics_computation_status"] = "replay_exception"
        row["nonfinite_reason"] = row["replay_exception_if_any"].splitlines()[0]
        row["replay_rayleigh_lambda"] = float("nan")
        row["replay_rayleigh_frequency_hz"] = float("nan")
        row["replay_relative_residual"] = float("nan")
        row["algebraic_lambda_one_suspect"] = False
        row["reported_vs_replay_frequency_consistent"] = False
        row["rejection_reasons"] = ["replay_computation_failed"]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        return row

    if not math.isfinite(row["rayleigh_denominator"]) or abs(row["rayleigh_denominator"]) < 1.0e-30:
        row["metrics_computation_status"] = "rayleigh_denominator_near_zero"
        row["nonfinite_reason"] = (
            f"xH_Mx={row['rayleigh_denominator']} (Rayleigh quotient undefined)"
        )
    elif not math.isfinite(row["replay_rayleigh_frequency_hz"]):
        row["metrics_computation_status"] = "replay_frequency_nonfinite"
        row["nonfinite_reason"] = f"replay_f={row['replay_rayleigh_frequency_hz']}"
    elif mac_status != "ok":
        row["metrics_computation_status"] = f"mac_{mac_status}"
        row["nonfinite_reason"] = mac_status
    else:
        row["metrics_computation_status"] = "ok"

    if row["metrics_computation_status"] != "ok":
        row["rejection_reasons"] = [row["metrics_computation_status"]]
        row["physically_eligible_after_filter"] = False
        row["branch_recovery_pass"] = False
        if not op_ok:
            row["rejection_reasons"].append("st_regularization_used_eps_not_replay_consistent")
        return row

    elig = assess_physical_eligibility(
        reported_f_hz=float(row["reported_frequency_hz"]),
        replay_metrics=replay,
        pressure_mac_to_true_seed=mac,
        seed_f_hz=seed_f_hz,
        require_mac=True,
        require_seed_frequency_match=True,
    )
    row["physically_eligible_after_filter"] = elig["physically_eligible_after_filter"]
    row["rejection_reasons"] = list(elig["rejection_reasons"])
    if not op_ok:
        row["rejection_reasons"].append("st_regularization_used_eps_not_replay_consistent")
    row["branch_recovery_pass"] = branch_recovery_from_row(row)
    return row


def assign_unreg_verdict(
    *,
    continuation: bool,
    op_ok: bool,
    candidates: List[Dict[str, Any]],
    artifacts_ok: bool,
) -> str:
    if not artifacts_ok or not candidates:
        return VERDICT_INCONSISTENT
    if not continuation or not op_ok:
        return VERDICT_INCONSISTENT
    statuses = [str(c.get("metrics_computation_status", "")) for c in candidates]
    if not all(s == "ok" for s in statuses):
        return VERDICT_INCONSISTENT
    if any(c.get("branch_recovery_pass") for c in candidates):
        return VERDICT_BRANCH_RECOVERED
    return VERDICT_NO_BRANCH


def evaluate_unregularized_offset_artifacts(
    *,
    out_dir: Path,
    case_dir: Path,
    mesh_file: Path,
    seed_npy: Path,
    seed_meta: Dict[str, Any],
    fallback_sample: Dict[str, Any],
    solve_result: Dict[str, Any],
    target_hz: float,
) -> Dict[str, Any]:
    from v2_seed_branch_candidate_filter import (
        FILTER_POLICY,
        assess_physical_eligibility,
        branch_recovery_from_row,
        extract_st_operator_fields,
        replay_candidate_metrics,
    )

    input_paths = {
        "output_dir": str(out_dir.resolve()),
        "case_dir": str(case_dir.resolve()),
        "mesh_file": str(mesh_file.resolve()),
        "seed_npy": str(seed_npy.resolve()),
        "forbidden_trees_not_used": list(FORBIDDEN_INPUT_SUBDIRS),
    }
    sample, sample_path = _load_sample_spec(out_dir, fallback_sample)
    input_paths["sample_spec"] = sample_path
    try:
        from v2_sensitivity_common import hz_result_tag

        tag = hz_result_tag(target_hz)
    except Exception:
        tag = int(round(float(target_hz) * 1000))
    result_path = out_dir / "results" / f"result_{tag}.json"
    if not result_path.is_file():
        results = sorted((out_dir / "results").glob("result_*.json"))
        result_path = results[-1] if results else result_path
    input_paths["solve_result_json"] = str(result_path)

    seed_info = load_seed_with_diagnostics(seed_npy)
    if seed_info.get("seed_load_status") != "ok":
        return {
            "evidence_scope": "VM_runtime_artifact_evaluation",
            "input_paths": input_paths,
            "diagnostic_verdict": VERDICT_INCONSISTENT,
            "verdict_reason": seed_info.get("seed_load_status"),
            "seed_diagnostics": {k: v for k, v in seed_info.items() if k != "seed_array"},
        }

    seed = seed_info["seed_array"]
    seed_f = float(seed_meta.get("locator_frequency_hz", solve_result.get("target_hz", target_hz)))
    continuation = bool(
        solve_result.get("continuation_seed_applied")
        or (solve_result.get("eps_batch_diagnostics") or {}).get("continuation_seed_applied")
        or (solve_result.get("eps_seed") or {}).get("eps_initial_space_set")
    )
    st_fields = extract_st_operator_fields(solve_result)
    op_ok = bool(st_fields["diagnostic_operator_consistent_with_replay"])

    summary_path = out_dir / "diagnostics" / "mode_energy_summary.json"
    modes: List[Dict[str, Any]] = []
    if summary_path.is_file():
        modes = json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []

    A, M, u_asm, p_asm, asm_meta = assemble_replay_operators(
        mesh_file, sample, out_dir=out_dir
    )
    u_to_W, p_to_W, maps_info = resolve_maps_from_solve_result(solve_result, u_asm, p_asm)
    operator_size = int(asm_meta["operator_size"])
    input_paths["mode_energy_summary"] = str(summary_path)

    candidates: List[Dict[str, Any]] = []
    try:
        for m in modes:
            row = _evaluate_one_candidate(
                mode_row=m,
                out_dir=out_dir,
                seed=seed,
                seed_f_hz=seed_f,
                continuation=continuation,
                st_fields=st_fields,
                op_ok=op_ok,
                u_to_W=u_to_W,
                p_to_W=p_to_W,
                operator_size=operator_size,
                A=A,
                M=M,
                assess_physical_eligibility=assess_physical_eligibility,
                branch_recovery_from_row=branch_recovery_from_row,
                replay_candidate_metrics=replay_candidate_metrics,
            )
            candidates.append(row)
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    candidates.sort(key=lambda r: int(r.get("candidate_index") or 0))
    branch_pool = [c for c in candidates if c.get("branch_recovery_pass")]
    best = (
        max(branch_pool, key=lambda r: float(r["pressure_MAC_to_true_acoustic_seed"]))
        if branch_pool
        else {}
    )
    artifacts_ok = bool(candidates) and summary_path.is_file()
    verdict = assign_unreg_verdict(
        continuation=continuation,
        op_ok=op_ok,
        candidates=candidates,
        artifacts_ok=artifacts_ok,
    )
    eps_diag = solve_result.get("eps_batch_diagnostics") or {}
    metrics_ok_count = sum(1 for c in candidates if c.get("metrics_computation_status") == "ok")

    return {
        "evidence_scope": "VM_runtime_artifact_evaluation",
        "input_paths": input_paths,
        "replay_assembly": {**asm_meta, **maps_info},
        "seed_frequency_hz": seed_f,
        "filter_policy": FILTER_POLICY,
        "st_operator_fields": st_fields,
        "continuation_seed_applied": continuation,
        "eps_nconv_marked": eps_diag.get("nconv_marked"),
        "candidates": candidates,
        "recovered_mode": best,
        "any_branch_recovery_pass": bool(branch_pool),
        "summary": {
            "num_candidates_evaluated": len(candidates),
            "num_metrics_computation_ok": metrics_ok_count,
            "num_branch_recovery_pass": len(branch_pool),
            "num_physically_eligible": sum(
                1 for c in candidates if c.get("physically_eligible_after_filter")
            ),
        },
        "diagnostic_verdict": verdict,
        "verdict_policy": {
            "NO_PHYSICAL_BRANCH_only_if_all_metrics_ok": True,
            "INCONSISTENT_if_any_metrics_not_ok": True,
        },
    }
