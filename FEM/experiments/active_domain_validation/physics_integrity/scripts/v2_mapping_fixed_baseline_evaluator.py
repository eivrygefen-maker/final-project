#!/usr/bin/env python3
"""
Report-only evaluator for mapping-corrected unregularized baseline diagnostic artifacts.

Reads exclusively from seed_branch_recovery_diagnostic_mapping_fixed_unregularized/ (no EPS).
Physical eligibility is applied only after all converged candidates are saved.
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
    "seed_branch_recovery_diagnostic_unregularized_offset",
    "seeded_retrieval",
)

VERDICT_BRANCH_RECOVERED = "MAPPING_FIXED_UNREGULARIZED_BASELINE_BRANCH_RECOVERED"
VERDICT_NO_BRANCH = "MAPPING_FIXED_UNREGULARIZED_BASELINE_NO_PHYSICAL_BRANCH_RECOVERED"
VERDICT_INCONSISTENT = "MAPPING_FIXED_UNREGULARIZED_BASELINE_OUTPUT_OR_REPLAY_INCONSISTENT"
VERDICT_PERSISTENCE_FAILURE = (
    "MAPPING_FIXED_UNREGULARIZED_BASELINE_CANDIDATE_PERSISTENCE_FAILURE"
)

OUT_SUBDIR = "seed_branch_recovery_diagnostic_mapping_fixed_unregularized"
OUT_SUBDIR_PERSISTENCE_FIXED = (
    "seed_branch_recovery_diagnostic_mapping_fixed_unregularized_persistence_fixed"
)


def _path_under(base: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def mapping_fixed_branch_recovery_from_row(row: Dict[str, Any]) -> bool:
    if str(row.get("eps_eigenvalue_semantics", "")) != "slepc_backtransformed":
        return False
    if not bool(row.get("legacy_double_shift_mapping_disabled", False)):
        return False
    if not bool(row.get("continuation_seed_applied", True)):
        return False
    den = float(row.get("rayleigh_denominator", float("nan")))
    if not math.isfinite(den) or abs(den) < 1.0e-30:
        return False
    from v2_seed_branch_candidate_filter import branch_recovery_from_row

    return bool(branch_recovery_from_row(row))


def persistence_failure_from_artifacts(
    *,
    solve_result: Dict[str, Any],
    bank_summary: Dict[str, Any],
    num_saved: int,
) -> Optional[Dict[str, Any]]:
    from v2_mapping_fixed_candidate_persistence import check_persistence_gate

    eps_diag = solve_result.get("eps_batch_diagnostics") or {}
    nconv = int(eps_diag.get("nconv_marked", solve_result.get("nconv_marked", 0)) or 0)
    bank_count = int(
        bank_summary.get("eps_diagnostic_candidate_bank_count")
        or bank_summary.get("nconv_marked")
        or eps_diag.get("eps_diagnostic_candidate_bank_count", 0)
        or 0
    )
    save_errors = list(bank_summary.get("save_errors") or [])
    pf = solve_result.get("diagnostic_verdict")
    if pf == VERDICT_PERSISTENCE_FAILURE:
        return solve_result.get("seed_branch_recovery_diagnostic", {}).get(
            "persistence_failure"
        ) or {
            "diagnostic_verdict": VERDICT_PERSISTENCE_FAILURE,
            "verdict_reason": "recorded_at_solve_time",
            "nconv_marked": nconv,
            "eps_diagnostic_candidate_bank_count": bank_count,
            "num_vectors_saved": num_saved,
            "save_errors": save_errors,
        }
    return check_persistence_gate(
        nconv_marked=nconv,
        bank_count=bank_count,
        num_vectors_saved=num_saved,
        save_errors=save_errors,
    )


def assign_mapping_fixed_verdict(
    *,
    continuation: bool,
    op_ok: bool,
    semantics_ok: bool,
    candidates: List[Dict[str, Any]],
    artifacts_ok: bool,
    nconv_marked: Optional[int],
    num_saved: int,
    persistence_failure: Optional[Dict[str, Any]] = None,
) -> str:
    if persistence_failure is not None:
        return VERDICT_PERSISTENCE_FAILURE
    if not artifacts_ok or not candidates:
        if nconv_marked is not None and int(nconv_marked) > 0 and int(num_saved) == 0:
            return VERDICT_PERSISTENCE_FAILURE
        return VERDICT_INCONSISTENT
    if not continuation or not op_ok or not semantics_ok:
        return VERDICT_INCONSISTENT
    if nconv_marked is not None and int(nconv_marked) > 0 and num_saved < int(nconv_marked):
        return VERDICT_PERSISTENCE_FAILURE
    statuses = [str(c.get("metrics_computation_status", "")) for c in candidates]
    if not all(s == "ok" for s in statuses):
        return VERDICT_INCONSISTENT
    if any(mapping_fixed_branch_recovery_from_row(c) for c in candidates):
        return VERDICT_BRANCH_RECOVERED
    return VERDICT_NO_BRANCH


def evaluate_mapping_fixed_baseline_artifacts(
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
    from v2_unreg_offset_report_evaluator import (
        _evaluate_one_candidate,
        _load_sample_spec,
        assemble_replay_operators,
        load_seed_with_diagnostics,
        resolve_maps_from_solve_result,
    )
    from v2_seed_branch_candidate_filter import (
        FILTER_POLICY,
        assess_physical_eligibility,
        extract_st_operator_fields,
        replay_candidate_metrics,
    )

    input_paths = {
        "output_dir": str(out_dir.resolve()),
        "case_dir": str(case_dir.resolve()),
        "mesh_file": str(mesh_file.resolve()),
        "seed_npy": str(seed_npy.resolve()),
        "forbidden_trees_not_used": list(FORBIDDEN_INPUT_SUBDIRS),
        "eps_candidate_bank": str(out_dir / "diagnostics" / "eps_candidate_bank.json"),
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
    eps_diag = solve_result.get("eps_batch_diagnostics") or {}
    semantics_ok = (
        str(eps_diag.get("eps_eigenvalue_semantics", "slepc_backtransformed"))
        == "slepc_backtransformed"
        and bool(eps_diag.get("legacy_double_shift_mapping_disabled", True))
    )

    summary_path = out_dir / "diagnostics" / "mode_energy_summary.json"
    bank_path = out_dir / "diagnostics" / "eps_candidate_bank.json"
    bank_summary: Dict[str, Any] = {}
    if bank_path.is_file():
        try:
            bank_summary = json.loads(bank_path.read_text(encoding="utf-8"))
        except Exception:
            bank_summary = {"load_status": "failed"}
    num_saved = int(
        bank_summary.get("num_vectors_saved", solve_result.get("num_modes_saved", 0)) or 0
    )
    nconv = eps_diag.get("nconv_marked")
    persistence_failure = persistence_failure_from_artifacts(
        solve_result=solve_result,
        bank_summary=bank_summary,
        num_saved=num_saved,
    )
    if persistence_failure is not None:
        persistence_failure.setdefault("candidate_output_dir", str(out_dir.resolve()))
        return {
            "evidence_scope": "VM_runtime_artifact_evaluation",
            "input_paths": input_paths,
            "diagnostic_verdict": VERDICT_PERSISTENCE_FAILURE,
            "verdict_reason": persistence_failure.get("verdict_reason"),
            "interpretation": (
                "Corrected unregularized EPS produced converged candidates in memory, but "
                "diagnostic preserve-all vectors were not persisted for replay. No physical "
                "branch verdict is possible from this run."
            ),
            "persistence_failure": persistence_failure,
            "st_operator_fields": st_fields,
            "continuation_seed_applied": continuation,
            "eps_nconv_marked": nconv,
            "eps_candidate_bank_summary": {
                "path": str(bank_path),
                "loaded": bank_path.is_file(),
                **{k: bank_summary.get(k) for k in (
                    "num_vectors_saved",
                    "nconv_marked",
                    "eps_diagnostic_candidate_bank_count",
                    "save_errors",
                )},
            },
            "not_evidence_for_st_failure": True,
            "not_evidence_for_stage_2": True,
        }

    modes: List[Dict[str, Any]] = []
    if summary_path.is_file():
        modes = json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []
    input_paths["mode_energy_summary"] = str(summary_path)

    st_fields["eps_eigenvalue_semantics"] = str(
        eps_diag.get("eps_eigenvalue_semantics", "slepc_backtransformed")
    )
    st_fields["legacy_double_shift_mapping_disabled"] = bool(
        eps_diag.get("legacy_double_shift_mapping_disabled", True)
    )

    A, M, u_asm, p_asm, asm_meta = assemble_replay_operators(
        mesh_file, sample, out_dir=out_dir
    )
    u_to_W, p_to_W, maps_info = resolve_maps_from_solve_result(solve_result, u_asm, p_asm)
    operator_size = int(asm_meta["operator_size"])

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
                branch_recovery_from_row=mapping_fixed_branch_recovery_from_row,
                replay_candidate_metrics=replay_candidate_metrics,
            )
            row["eps_eigenvalue_semantics"] = st_fields["eps_eigenvalue_semantics"]
            row["legacy_double_shift_mapping_disabled"] = st_fields[
                "legacy_double_shift_mapping_disabled"
            ]
            row["mu_raw"] = m.get("mu_raw")
            row["lam_phys"] = m.get("lam_phys")
            row["lam_map_tag"] = m.get("lam_map_tag")
            row["sigma_used_hz"] = m.get("sigma_used_hz", st_fields.get("actual_sigma_hz"))
            row["st_type"] = m.get("st_type")
            candidates.append(row)
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    candidates.sort(key=lambda r: int(r.get("candidate_index") or r.get("eps_slot_index") or 0))
    branch_pool = [c for c in candidates if mapping_fixed_branch_recovery_from_row(c)]
    best = (
        max(branch_pool, key=lambda r: float(r["pressure_MAC_to_true_acoustic_seed"]))
        if branch_pool
        else {}
    )
    artifacts_ok = bool(candidates) and summary_path.is_file()
    verdict = assign_mapping_fixed_verdict(
        continuation=continuation,
        op_ok=op_ok,
        semantics_ok=semantics_ok,
        candidates=candidates,
        artifacts_ok=artifacts_ok,
        nconv_marked=int(nconv) if nconv is not None else None,
        num_saved=num_saved,
        persistence_failure=None,
    )
    metrics_ok_count = sum(1 for c in candidates if c.get("metrics_computation_status") == "ok")

    return {
        "evidence_scope": "VM_runtime_artifact_evaluation",
        "input_paths": input_paths,
        "replay_assembly": {**asm_meta, **maps_info},
        "seed_frequency_hz": seed_f,
        "filter_policy": FILTER_POLICY,
        "st_operator_fields": st_fields,
        "continuation_seed_applied": continuation,
        "eps_nconv_marked": nconv,
        "eps_candidate_bank_summary": {
            "path": str(bank_path),
            "loaded": bank_path.is_file(),
            "num_vectors_saved": bank_summary.get("num_vectors_saved"),
            "nconv_marked": bank_summary.get("nconv_marked"),
        },
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
            "requires_slepc_backtransformed_semantics": True,
            "requires_legacy_double_shift_mapping_disabled": True,
            "NO_PHYSICAL_BRANCH_only_if_all_metrics_ok": True,
            "INCONSISTENT_if_incomplete_candidate_bank": True,
        },
        "prior_pass_handling": {
            "mesh_topology_gates_preserved": True,
            "true_seed_replay_findings_preserved": True,
            "eps_frequency_labels_pending_recertification": True,
            "some_modes_valid_physics_wrong_frequency_labels_only": True,
        },
    }
