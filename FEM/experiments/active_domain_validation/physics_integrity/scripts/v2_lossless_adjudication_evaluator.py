#!/usr/bin/env python3
"""Evaluate mapping-fixed baseline from authoritative lossless dense vectors."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from v2_mapping_fixed_baseline_evaluator import (
    OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
    VERDICT_BRANCH_RECOVERED,
    VERDICT_INCONSISTENT,
    VERDICT_NO_BRANCH,
    VERDICT_PERSISTENCE_FAILURE,
    assign_mapping_fixed_verdict,
    mapping_fixed_branch_recovery_from_row,
    persistence_failure_from_artifacts,
)

VERDICT_LOSSLESS_ROUNDTRIP_FAILURE = (
    "MAPPING_FIXED_UNREGULARIZED_BASELINE_LOSSLESS_ROUNDTRIP_FAILURE"
)
VERDICT_LOSSLESS_REPLAY_FAILURE = (
    "MAPPING_FIXED_UNREGULARIZED_BASELINE_LOSSLESS_REPLAY_EVALUATION_FAILURE"
)


def counterfactual_production_filter_labels(
    *,
    mode_row: Dict[str, Any],
    replay_metrics: Dict[str, Any],
    st_fields: Dict[str, Any],
    seed_f_hz: float,
) -> Dict[str, Any]:
    """Report production filter outcomes without removing candidates from evidence pool."""
    reported_f = float(mode_row.get("frequency_hz", mode_row.get("reported_frequency_hz", float("nan"))))
    replay_f = float(replay_metrics.get("replay_rayleigh_frequency_hz", float("nan")))
    p_frac = mode_row.get("p_frac_energy_phys")
    lam_one = bool(replay_metrics.get("algebraic_lambda_one_suspect", False))
    sigma_hz = float(st_fields.get("actual_sigma_hz", float("nan")))
    sigma_ritz_reject = False
    if math.isfinite(reported_f) and math.isfinite(sigma_hz) and sigma_hz > 0:
        sigma_ritz_reject = abs(reported_f - sigma_hz) / sigma_hz > 0.05
    p_frac_low = False
    if p_frac is not None:
        try:
            p_frac_low = float(p_frac) < 0.01
        except (TypeError, ValueError):
            p_frac_low = False
    low_band_bypass = bool(
        math.isfinite(reported_f)
        and math.isfinite(seed_f_hz)
        and abs(reported_f - seed_f_hz) / seed_f_hz <= 0.01
    )
    return {
        "sigma_ritz_classification": "reject" if sigma_ritz_reject else "pass",
        "p_frac_staged_harvest_would_filter": p_frac_low,
        "low_band_bypass_would_apply": low_band_bypass,
        "algebraic_lambda_one_would_filter": lam_one,
        "production_eligibility_label": (
            "would_reject" if (sigma_ritz_reject or p_frac_low or lam_one) else "would_pass"
        ),
        "counterfactual_only": True,
        "removed_from_adjudication_pool": False,
    }


def assign_lossless_adjudication_verdict(
    *,
    continuation: bool,
    op_ok: bool,
    semantics_ok: bool,
    candidates: List[Dict[str, Any]],
    artifacts_ok: bool,
    nconv_marked: Optional[int],
    num_saved: int,
    lossless_roundtrip_failures: int,
    persistence_failure: Optional[Dict[str, Any]] = None,
) -> str:
    if persistence_failure is not None:
        return VERDICT_PERSISTENCE_FAILURE
    if lossless_roundtrip_failures > 0:
        return VERDICT_LOSSLESS_ROUNDTRIP_FAILURE
    if not artifacts_ok or not candidates:
        return VERDICT_INCONSISTENT
    if not all(c.get("metrics_computation_status") == "ok" for c in candidates):
        return VERDICT_LOSSLESS_REPLAY_FAILURE
    return assign_mapping_fixed_verdict(
        continuation=continuation,
        op_ok=op_ok,
        semantics_ok=semantics_ok,
        candidates=candidates,
        artifacts_ok=artifacts_ok,
        nconv_marked=nconv_marked,
        num_saved=num_saved,
        persistence_failure=None,
    )


def evaluate_lossless_adjudication_artifacts(
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
        "vector_authoritative": "lossless_dense_smx_dense_npy",
        "eps_candidate_bank": str(out_dir / "diagnostics" / "eps_candidate_bank.json"),
    }
    sample, sample_path = _load_sample_spec(out_dir, fallback_sample)
    input_paths["sample_spec"] = sample_path

    seed_info = load_seed_with_diagnostics(seed_npy)
    if seed_info.get("seed_load_status") != "ok":
        return {
            "evidence_scope": "lossless_adjudication_v1_evaluation",
            "input_paths": input_paths,
            "diagnostic_verdict": VERDICT_INCONSISTENT,
            "verdict_reason": seed_info.get("seed_load_status"),
        }

    seed = seed_info["seed_array"]
    seed_f = float(seed_meta.get("locator_frequency_hz", solve_result.get("target_hz", target_hz)))
    eps_diag = solve_result.get("eps_batch_diagnostics") or {}
    continuation = bool(
        solve_result.get("continuation_seed_applied")
        or eps_diag.get("continuation_seed_applied")
        or (solve_result.get("eps_seed") or {}).get("eps_initial_space_set")
    )
    st_fields = extract_st_operator_fields(solve_result)
    st_fields["st_type"] = str(eps_diag.get("st_type", st_fields.get("st_type", "sinvert")))
    op_ok = bool(st_fields["diagnostic_operator_consistent_with_replay"])
    semantics_ok = (
        str(eps_diag.get("eps_eigenvalue_semantics", "slepc_backtransformed"))
        == "slepc_backtransformed"
        and bool(eps_diag.get("legacy_double_shift_mapping_disabled", True))
    )

    bank_path = out_dir / "diagnostics" / "eps_candidate_bank.json"
    bank_summary: Dict[str, Any] = {}
    if bank_path.is_file():
        bank_summary = json.loads(bank_path.read_text(encoding="utf-8"))
    num_saved = int(bank_summary.get("num_vectors_saved", 0) or 0)
    nconv = eps_diag.get("nconv_marked")
    persistence_failure = persistence_failure_from_artifacts(
        solve_result=solve_result,
        bank_summary=bank_summary,
        num_saved=num_saved,
    )
    if persistence_failure is not None:
        persistence_failure["lossless_authoritative"] = True
        return {
            "evidence_scope": "lossless_adjudication_v1_evaluation",
            "input_paths": input_paths,
            "diagnostic_verdict": VERDICT_PERSISTENCE_FAILURE,
            "persistence_failure": persistence_failure,
            "not_evidence_for_st_failure": True,
        }

    summary_path = out_dir / "diagnostics" / "mode_energy_summary.json"
    modes: List[Dict[str, Any]] = []
    if summary_path.is_file():
        modes = json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []

    roundtrip_failures = sum(
        1 for m in modes if m.get("lossless_persistence") and not m.get("lossless_roundtrip_pass")
    )
    missing_lossless = [
        m
        for m in modes
        if not m.get("vector_file_lossless") or not (out_dir / str(m["vector_file_lossless"])).is_file()
    ]
    if missing_lossless:
        return {
            "evidence_scope": "lossless_adjudication_v1_evaluation",
            "input_paths": input_paths,
            "diagnostic_verdict": VERDICT_LOSSLESS_ROUNDTRIP_FAILURE,
            "verdict_reason": "lossless_vector_file_missing_for_one_or_more_candidates",
            "missing_lossless_count": len(missing_lossless),
            "not_evidence_for_st_failure": True,
        }
    if roundtrip_failures > 0:
        return {
            "evidence_scope": "lossless_adjudication_v1_evaluation",
            "input_paths": input_paths,
            "diagnostic_verdict": VERDICT_LOSSLESS_ROUNDTRIP_FAILURE,
            "lossless_roundtrip_failures": roundtrip_failures,
            "not_evidence_for_st_failure": True,
        }

    A, M, u_asm, p_asm, asm_meta = assemble_replay_operators(mesh_file, sample, out_dir=out_dir)
    u_to_W, p_to_W, maps_info = resolve_maps_from_solve_result(solve_result, u_asm, p_asm)
    operator_size = int(asm_meta["operator_size"])

    candidates: List[Dict[str, Any]] = []
    try:
        for m in modes:
            eval_row = dict(m)
            lossless_rel = m.get("vector_file_lossless")
            if not lossless_rel:
                continue
            eval_row["vector_path"] = lossless_rel
            eval_row["vector_authoritative"] = "lossless_dense"
            row = _evaluate_one_candidate(
                mode_row=eval_row,
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
            row["candidate_index"] = m.get(
                "candidate_index", m.get("eps_slot_index", m.get("mode_index"))
            )
            row["eps_slot_index"] = m.get("eps_slot_index", row.get("candidate_index"))
            row["vector_authoritative"] = "lossless_dense"
            row["lossless_vector_path"] = lossless_rel
            row["legacy_sparse_comparison_path"] = m.get("legacy_sparse_comparison_path")
            row["lossless_roundtrip_pass"] = m.get("lossless_roundtrip_pass")
            replay_m = {
                k: row.get(k)
                for k in (
                    "replay_rayleigh_lambda",
                    "replay_rayleigh_frequency_hz",
                    "replay_relative_residual",
                    "algebraic_lambda_one_suspect",
                    "reported_vs_replay_frequency_consistent",
                )
            }
            row["counterfactual_filter_labels"] = counterfactual_production_filter_labels(
                mode_row=m,
                replay_metrics=replay_m,
                st_fields=st_fields,
                seed_f_hz=seed_f,
            )
            row["eps_eigenvalue_semantics"] = st_fields["eps_eigenvalue_semantics"]
            row["legacy_double_shift_mapping_disabled"] = st_fields[
                "legacy_double_shift_mapping_disabled"
            ]
            row["st_type"] = st_fields["st_type"]
            row["st_type_authoritative_persisted"] = bool(eps_diag.get("st_type"))
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
    verdict = assign_lossless_adjudication_verdict(
        continuation=continuation,
        op_ok=op_ok,
        semantics_ok=semantics_ok,
        candidates=candidates,
        artifacts_ok=artifacts_ok,
        nconv_marked=int(nconv) if nconv is not None else None,
        num_saved=num_saved,
        lossless_roundtrip_failures=roundtrip_failures,
        persistence_failure=None,
    )

    return {
        "evidence_scope": "lossless_adjudication_v1_authoritative_evaluation",
        "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "input_paths": input_paths,
        "replay_assembly": {**asm_meta, **maps_info},
        "seed_frequency_hz": seed_f,
        "filter_policy": FILTER_POLICY,
        "st_operator_fields": st_fields,
        "st_type_authoritative_provenance": {
            "st_type": st_fields.get("st_type"),
            "persisted_at_eps_batch_root": bool(eps_diag.get("st_type")),
            "prior_replacement_gap_closed": bool(eps_diag.get("st_type")),
        },
        "continuation_seed_applied": continuation,
        "eps_nconv_marked": nconv,
        "lossless_candidate_count": len(modes),
        "lossless_vectors_saved": num_saved,
        "lossless_roundtrip_failures": roundtrip_failures,
        "legacy_sparse_comparison_saved": sum(
            1 for m in modes if m.get("legacy_sparse_comparison_path")
        ),
        "eps_candidate_bank_summary": {
            "path": str(bank_path),
            "num_vectors_saved": num_saved,
            "nconv_marked": bank_summary.get("nconv_marked"),
        },
        "candidates": candidates,
        "recovered_mode": best,
        "any_branch_recovery_pass": bool(branch_pool),
        "summary": {
            "num_candidates_evaluated": len(candidates),
            "num_metrics_computation_ok": sum(
                1 for c in candidates if c.get("metrics_computation_status") == "ok"
            ),
            "num_branch_recovery_pass": len(branch_pool),
            "num_mass_null": sum(
                1
                for c in candidates
                if not math.isfinite(float(c.get("xH_Mx", float("nan"))))
                or abs(float(c.get("xH_Mx", 0.0))) < 1.0e-30
            ),
        },
        "diagnostic_verdict": verdict,
        "all_candidates_evaluated_from_lossless_only": True,
        "counterfactual_filters_reported": True,
        "not_evidence_for_production_promotion": True,
    }
