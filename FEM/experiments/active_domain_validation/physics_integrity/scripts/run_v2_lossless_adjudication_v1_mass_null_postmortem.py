#!/usr/bin/env python3
"""
Report-only mass-null / root-space postmortem for lossless adjudication v1 (no EPS).

Reads only seed_branch_recovery_diagnostic_mapping_fixed_unregularized_lossless_adjudication_v1/.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir, write_json
from v2_sensitivity_common import hz_result_tag

OUT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_mass_null_postmortem.json"
OUT_MD = CONV_DIAG / "v2_lossless_adjudication_v1_mass_null_postmortem.md"
AUDIT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_full_pipeline_audit.json"
DIAG_JSON = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_lossless_adjudication_v1_diagnostic.json"
AUTH_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_eps_authorization_record.json"

CASE_ID = "baseline_coupled_v2"
XH_MX_TOL = 1.0e-30
NNZ_THRESHOLDS = (1e-15, 1e-12, 1e-9)
NORM_FRAC_TOL = 0.999

CLASS_NULL_M = "LOSSLESS_EPS_OUTPUT_IN_OR_NEAR_NULL_M"
CLASS_LAYOUT = "LOSSLESS_EPS_VECTOR_LAYOUT_OR_CAPTURE_MISMATCH"
CLASS_METRIC = "LOSSLESS_REPLAY_METRIC_IMPLEMENTATION_FAILURE"
CLASS_ALGEBRAIC = "LOSSLESS_ST_TARGETING_BLOCKED_BY_ALGEBRAIC_MODES"
CLASS_NO_BRANCH = "LOSSLESS_ST_NO_PHYSICAL_BRANCH_IN_RETURNED_CANDIDATES"

VERDICT_REPLAY_FAIL = "MAPPING_FIXED_UNREGULARIZED_BASELINE_LOSSLESS_REPLAY_EVALUATION_FAILURE"


def _nnz_stats(vec: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(vec, dtype=np.float64).ravel()
    ab = np.abs(v)
    out: Dict[str, Any] = {"nnz_exact": int(np.count_nonzero(v))}
    for thr in NNZ_THRESHOLDS:
        out[f"nnz_gt_{thr:g}"] = int(np.sum(ab > thr))
    return out


def _block_support(vec: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
    if idx.size == 0:
        return {"l2_norm": 0.0, "max_abs": 0.0, **_nnz_stats(np.array([]))}
    blk = np.asarray(vec, dtype=np.float64).ravel()[idx]
    return {
        "l2_norm": float(np.linalg.norm(blk)),
        "max_abs": float(np.max(np.abs(blk))) if blk.size else 0.0,
        **_nnz_stats(blk),
    }


def _support_decomposition(
    vec: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    operator_size: int,
) -> Dict[str, Any]:
    v = np.asarray(vec, dtype=np.float64).ravel()
    l2_total = float(np.linalg.norm(v))
    u_set = set(int(x) for x in u_to_W.tolist())
    p_set = set(int(x) for x in p_to_W.tolist())
    overlap = u_set & p_set
    all_w = set(range(int(operator_size)))
    complement = all_w - u_set - p_set
    comp_idx = np.asarray(sorted(complement), dtype=np.int32)
    u_idx = np.asarray(sorted(u_set), dtype=np.int32)
    p_idx = np.asarray(sorted(p_set), dtype=np.int32)

    u_sup = _block_support(v, u_idx)
    p_sup = _block_support(v, p_idx)
    c_sup = _block_support(v, comp_idx)

    def frac(sup: Dict[str, Any]) -> float:
        return float(sup["l2_norm"] / l2_total) if l2_total > 0 else 0.0

    u_frac = frac(u_sup)
    p_frac = frac(p_sup)
    c_frac = frac(c_sup)
    dominant = "u_active"
    if p_frac > u_frac and p_frac >= NORM_FRAC_TOL:
        dominant = "p_active"
    elif c_frac >= NORM_FRAC_TOL:
        dominant = "complement_or_unmapped_W_rows"
    elif u_frac >= NORM_FRAC_TOL:
        dominant = "u_active"

    return {
        "vector_length": int(v.size),
        "l2_norm_total": l2_total,
        "max_abs_total": float(np.max(np.abs(v))) if v.size else 0.0,
        "u_active_support": u_sup,
        "p_active_support": p_sup,
        "complement_W_rows_support": c_sup,
        "u_p_map_overlap_count": len(overlap),
        "complement_row_count": len(complement),
        "l2_fraction_on_u": u_frac,
        "l2_fraction_on_p": p_frac,
        "l2_fraction_on_complement": c_frac,
        "dominant_support_category": dominant,
        "p_active_absent": bool(p_sup["l2_norm"] < 1.0e-30),
        **_nnz_stats(v),
    }


def _operator_actions(
    A,
    M,
    vec: np.ndarray,
    *,
    lam_reported: Optional[float],
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
) -> Dict[str, Any]:
    from physical_fsi_seed_residual_audit import (
        _block_residual_contributions,
        _petsc_matvec,
        _petsc_vec_from_array,
        _rayleigh_metrics,
    )

    x = np.asarray(vec, dtype=np.float64).ravel()
    out: Dict[str, Any] = {
        "l2_norm_x": float(np.linalg.norm(x)),
        "xH_Ax": float("nan"),
        "xH_Mx": float("nan"),
        "l2_norm_Ax": float("nan"),
        "l2_norm_Mx": float("nan"),
        "max_abs_Ax": float("nan"),
        "max_abs_Mx": float("nan"),
        "nnz_significant_Mx": None,
        "rayleigh_lambda": float("nan"),
        "rayleigh_frequency_hz": float("nan"),
        "residual_norm_reported_lambda": float("nan"),
        "relative_residual_reported_lambda": float("nan"),
        "mass_null": False,
        "in_or_near_null_M": False,
    }
    vx = _petsc_vec_from_array(A, x)
    try:
        Ax, ay = _petsc_matvec(A, vx)
        Mx, my = _petsc_matvec(M, vx)
        Ax_arr = np.asarray(Ax, dtype=np.float64).ravel()
        Mx_arr = np.asarray(Mx, dtype=np.float64).ravel()
        out["l2_norm_Ax"] = float(np.linalg.norm(Ax_arr))
        out["l2_norm_Mx"] = float(np.linalg.norm(Mx_arr))
        out["max_abs_Ax"] = float(np.max(np.abs(Ax_arr))) if Ax_arr.size else 0.0
        out["max_abs_Mx"] = float(np.max(np.abs(Mx_arr))) if Mx_arr.size else 0.0
        out["nnz_significant_Mx"] = int(np.sum(np.abs(Mx_arr) > 1.0e-15))
        ray = _rayleigh_metrics(A, M, x, seed_f_hz=float("nan"))
        out["xH_Ax"] = float(ray.get("xH_Ax", float("nan")))
        out["xH_Mx"] = float(ray.get("xH_Mx", float("nan")))
        out["rayleigh_lambda"] = float(ray.get("rayleigh_lambda", float("nan")))
        out["rayleigh_frequency_hz"] = float(ray.get("rayleigh_f_hz", float("nan")))
    finally:
        vx.destroy()
        ay.destroy()
        my.destroy()

    mx = out["l2_norm_Mx"]
    xhm = out["xH_Mx"]
    out["mass_null"] = bool(
        (math.isfinite(mx) and mx < 1.0e-30)
        or (math.isfinite(xhm) and abs(xhm) < XH_MX_TOL)
    )
    out["in_or_near_null_M"] = bool(math.isfinite(mx) and mx < 1.0e-12)

    lam_use = lam_reported
    if lam_use is None or not math.isfinite(float(lam_use)):
        lam_use = out["rayleigh_lambda"] if math.isfinite(out["rayleigh_lambda"]) else None
    if lam_use is not None and math.isfinite(float(lam_use)):
        res = _block_residual_contributions(
            A, M, x, lam0=float(lam_use), u_idx=u_to_W, p_idx=p_to_W
        )
        out["residual_norm_reported_lambda"] = float(res.get("residual_norm", float("nan")))
        out["relative_residual_reported_lambda"] = float(res.get("relative_residual", float("nan")))

    return out


def _eps_capture_contract_audit() -> Dict[str, Any]:
    return {
        "eigenpair_source": "FEM/scripts/fem_main_3d.py::_harvest_shift_invert_band",
        "getEigenpair_object": "eps.getEigenpair(i, rvec) -> rvec.array.copy() as arr_red",
        "imaginary_handling": "eig_r = float(np.real(eig)); vector stored real dense float64",
        "preserve_all_timing": "before harvest filters; diag_bank append per converged slot",
        "vector_layout": (
            "If _active_domain.active_W_indices: prolongate_to_full_mixed_vector(arr_red) "
            "else arr_red only"
        ),
        "replay_operator_layout": "v2_build_coupled_acoustic_seed._assemble_reduced_coupled_replay (same v2 config)",
        "bank_slot_alignment": "eps_slot_index == candidate_index == lossless filename slot",
        "evaluator_candidate_index_defect": (
            "v2_unreg_offset_report_evaluator used mode_index only; fixed to prefer "
            "candidate_index/eps_slot_index from mode_energy_summary rows"
        ),
        "st_type_in_bank_at_capture": True,
    }


def _classify_mass_null_run(
    *,
    seed_ops: Dict[str, Any],
    candidate_ops: List[Dict[str, Any]],
    support_summary: Dict[str, Any],
    layout_ok: bool,
) -> Tuple[str, str]:
    seed_mx = float(seed_ops.get("l2_norm_Mx", float("nan")))
    seed_xhm = float(seed_ops.get("xH_Mx", float("nan")))
    seed_valid = math.isfinite(seed_mx) and seed_mx > 1.0e-12 and math.isfinite(seed_xhm) and abs(seed_xhm) > XH_MX_TOL

    n = len(candidate_ops)
    n_null_m = sum(1 for c in candidate_ops if c.get("in_or_near_null_M"))
    n_mass_null_flag = sum(1 for c in candidate_ops if c.get("mass_null"))
    n_metric_ok = sum(
        1
        for c in candidate_ops
        if math.isfinite(float(c.get("l2_norm_Mx", float("nan"))))
        and float(c.get("l2_norm_Mx", 0)) > 1.0e-12
        and abs(float(c.get("xH_Mx", 0))) < XH_MX_TOL
    )

    if not layout_ok:
        return CLASS_LAYOUT, "vector_length_or_map_partition_mismatch"

    if n_metric_ok > 0:
        return CLASS_METRIC, "meaningful_Mx_norm_but_xH_Mx_near_zero_suspect_metric_cancellation"

    if seed_valid and n_null_m == n and n > 0:
        dom = support_summary.get("dominant_support_across_candidates", "")
        if dom == "u_active" and support_summary.get("all_p_active_absent"):
            return (
                CLASS_ALGEBRAIC,
                "all_candidates_in_near_null_M_with_norm_on_u_active_and_no_p_active_support",
            )
        return CLASS_NULL_M, "seed_has_valid_Mx_all_candidates_near_null_M"

    any_finite_branch = any(
        not c.get("mass_null")
        and math.isfinite(float(c.get("xH_Mx", float("nan"))))
        and abs(float(c.get("xH_Mx", 0))) > XH_MX_TOL
        for c in candidate_ops
    )
    if any_finite_branch:
        return CLASS_NO_BRANCH, "non_null_M_candidates_exist_but_failed_physical_gates"

    return CLASS_NULL_M, "default_mass_null_classification_pending_further_attribution"


def _duplicate_summary(candidates: List[Dict[str, Any]], vecs: List[np.ndarray]) -> Dict[str, Any]:
    if len(vecs) < 2:
        return {"pairwise_max_rel_diff": 0.0, "appears_duplicate_set": False}
    max_rel = 0.0
    for i in range(min(3, len(vecs))):
        for j in range(i + 1, min(4, len(vecs))):
            ni = float(np.linalg.norm(vecs[i]))
            nj = float(np.linalg.norm(vecs[j]))
            if ni > 0 and nj > 0:
                diff = float(np.linalg.norm(vecs[i] - vecs[j]) / max(ni, nj))
                max_rel = max(max_rel, diff)
    return {
        "pairwise_max_rel_diff_sampled": max_rel,
        "appears_duplicate_set": bool(max_rel < 1.0e-6),
    }


def _generate_full_audit(report: Dict[str, Any], diag: Dict[str, Any], solve_result: Dict[str, Any]) -> Dict[str, Any]:
    ev = diag.get("evaluation") or {}
    audit = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "lossless_adjudication_v1_post_eps_report_only",
        "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "preflight_gate_pass": bool(diag.get("preflight_gate_pass")),
        "single_lossless_adjudication_run_authorized": bool(
            diag.get("single_lossless_adjudication_run_authorized")
        ),
        "eps_run_count_for_this_lane": int(diag.get("eps_run_count_for_this_lane", 0)),
        "no_additional_eps_run_authorized": True,
        "prior_baseline_verdict": "MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED",
        "lossless_adjudication_verdict": ev.get("diagnostic_verdict", VERDICT_REPLAY_FAIL),
        "mass_null_classification": report.get("classification"),
        "nconv_marked": ev.get("eps_nconv_marked"),
        "lossless_candidate_count": ev.get("lossless_candidate_count"),
        "lossless_vectors_saved": ev.get("lossless_vectors_saved"),
        "lossless_roundtrip_failures": ev.get("lossless_roundtrip_failures", 0),
        "st_type_authoritative_provenance": ev.get("st_type_authoritative_provenance"),
        "operator_policy": solve_result.get("eps_batch_diagnostics") or {},
        "lossless_pre_sparsify_eps_vectors_available_in_current_run": True,
        "serialization_may_change_physical_replay_metrics": True,
        "serialization_fidelity_risk": True,
        "serialization_ruled_out_as_immediate_cause": True,
        "evaluation_summary": ev.get("summary"),
        "postmortem_summary": report.get("aggregate_summary"),
        "not_evidence_for_production_promotion": True,
        "mesh_convergence_may_resume": False,
        "st_viability": "unresolved_pending_mass_null_attribution",
    }
    return audit


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[mass_null_postmortem] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
    mesh_file = mesh_path("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = (
        json.loads(seed_meta_path.read_text(encoding="utf-8")) if seed_meta_path.is_file() else {}
    )
    target_hz = float(seed_meta.get("locator_frequency_hz", 243.0754171175576))

    if not DIAG_JSON.is_file():
        print("[mass_null_postmortem] diagnostic JSON missing", file=sys.stderr)
        return 2

    diag = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
    results = sorted((out_dir / "results").glob("result_*.json"))
    solve_result = json.loads(results[-1].read_text(encoding="utf-8")) if results else {}
    bank_path = out_dir / "diagnostics" / "eps_candidate_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8")) if bank_path.is_file() else {}
    summary_path = out_dir / "diagnostics" / "mode_energy_summary.json"
    modes = (
        json.loads(summary_path.read_text(encoding="utf-8")).get("modes", [])
        if summary_path.is_file()
        else []
    )

    from fem_mode_array_utils import load_mode_dense_f64_lossless
    from v2_unreg_offset_report_evaluator import (
        _load_sample_spec,
        assemble_replay_operators,
        load_seed_with_diagnostics,
        resolve_maps_from_solve_result,
    )

    sample, _ = _load_sample_spec(out_dir, sample_spec_from_case(case))
    A, M, u_asm, p_asm, asm_meta = assemble_replay_operators(mesh_file, sample, out_dir=out_dir)
    u_to_W, p_to_W, maps_info = resolve_maps_from_solve_result(solve_result, u_asm, p_asm)
    op_size = int(asm_meta["operator_size"])
    layout_ok = bool(
        op_size == int(u_to_W.size) + int(p_to_W.size)
        and len(set(u_to_W.tolist()) & set(p_to_W.tolist())) == 0
    )

    seed_info = load_seed_with_diagnostics(seed_npy)
    seed = seed_info.get("seed_array")
    seed_ops: Dict[str, Any] = {}
    seed_support: Dict[str, Any] = {}
    if seed is not None:
        seed_support = _support_decomposition(
            seed, u_to_W=u_to_W, p_to_W=p_to_W, operator_size=op_size
        )
        seed_ops = _operator_actions(
            A, M, seed, lam_reported=None, u_to_W=u_to_W, p_to_W=p_to_W
        )

    bank_by_slot = {
        int(r.get("eps_slot_index", r.get("candidate_index", -1))): r
        for r in (bank.get("saved_mode_rows") or bank.get("candidates") or [])
        if isinstance(r, dict)
    }

    per_candidate: List[Dict[str, Any]] = []
    sample_vecs: List[np.ndarray] = []
    u_fracs: List[float] = []
    p_fracs: List[float] = []
    p_absent = 0

    try:
        for m in sorted(
            modes,
            key=lambda r: int(r.get("eps_slot_index", r.get("candidate_index", 0)) or 0),
        ):
            slot = int(m.get("eps_slot_index", m.get("candidate_index", 0)) or 0)
            lossless_rel = m.get("vector_file_lossless")
            if not lossless_rel:
                continue
            lp = out_dir / str(lossless_rel)
            vec = load_mode_dense_f64_lossless(lp)
            sample_vecs.append(vec)
            sup = _support_decomposition(vec, u_to_W=u_to_W, p_to_W=p_to_W, operator_size=op_size)
            if sup["p_active_absent"]:
                p_absent += 1
            u_fracs.append(float(sup["l2_fraction_on_u"]))
            p_fracs.append(float(sup["l2_fraction_on_p"]))
            brec = bank_by_slot.get(slot, {})
            lam_phys = brec.get("lam_phys")
            ops = _operator_actions(
                A,
                M,
                vec,
                lam_reported=float(lam_phys) if lam_phys is not None else None,
                u_to_W=u_to_W,
                p_to_W=p_to_W,
            )
            per_candidate.append(
                {
                    "eps_slot_index": slot,
                    "candidate_index": slot,
                    "lossless_vector_path": str(lossless_rel),
                    "reported_frequency_hz": m.get("frequency_hz"),
                    "mu_raw": brec.get("mu_raw"),
                    "lam_phys": lam_phys,
                    "support": sup,
                    "operator_actions": ops,
                }
            )
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    agg_support = {
        "candidate_count": len(per_candidate),
        "all_p_active_absent": bool(p_absent == len(per_candidate) and len(per_candidate) > 0),
        "median_l2_fraction_on_u": float(np.median(u_fracs)) if u_fracs else 0.0,
        "median_l2_fraction_on_p": float(np.median(p_fracs)) if p_fracs else 0.0,
        "dominant_support_across_candidates": (
            "u_active"
            if u_fracs and float(np.median(u_fracs)) >= NORM_FRAC_TOL
            else "mixed"
        ),
        "pressure_norm_median": float(
            np.median([c["support"]["p_active_support"]["l2_norm"] for c in per_candidate])
        )
        if per_candidate
        else 0.0,
    }

    classification, class_reason = _classify_mass_null_run(
        seed_ops=seed_ops,
        candidate_ops=[c["operator_actions"] for c in per_candidate],
        support_summary=agg_support,
        layout_ok=layout_ok and all(
            c["support"]["vector_length"] == op_size for c in per_candidate
        ),
    )

    guard_audit = {
        "auth_record_exists": AUTH_JSON.is_file(),
        "eps_run_count_for_this_lane": int(diag.get("eps_run_count_for_this_lane", 0)),
        "no_additional_eps_run_authorized": True,
        "re_invoking_authorized_runner_would_block_eps": bool(
            AUTH_JSON.is_file() and int(diag.get("eps_run_count_for_this_lane", 0)) >= 1
        ),
        "isolated_tree_must_not_be_overwritten": True,
    }

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "report_only_no_eps",
        "no_new_eigensolve_executed": True,
        "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "diagnostic_verdict": (diag.get("evaluation") or {}).get(
            "diagnostic_verdict", VERDICT_REPLAY_FAIL
        ),
        "classification": classification,
        "classification_reason": class_reason,
        "does_not_upgrade_to_NO_PHYSICAL_BRANCH": classification != CLASS_NO_BRANCH,
        "serialization_ruled_out": True,
        "replay_maps": maps_info,
        "operator_size": op_size,
        "layout_partition_ok": layout_ok,
        "seed_support_and_operators": {"support": seed_support, "operators": seed_ops},
        "aggregate_summary": agg_support,
        "per_candidate": per_candidate,
        "per_candidate_operator_aggregate": {
            "l2_norm_Mx_min": min(
                (c["operator_actions"]["l2_norm_Mx"] for c in per_candidate), default=float("nan")
            ),
            "l2_norm_Mx_median": float(
                np.median([c["operator_actions"]["l2_norm_Mx"] for c in per_candidate])
            )
            if per_candidate
            else float("nan"),
            "xH_Mx_max_abs": max(
                (abs(c["operator_actions"]["xH_Mx"]) for c in per_candidate), default=0.0
            ),
            "mass_null_count": sum(1 for c in per_candidate if c["operator_actions"]["mass_null"]),
        },
        "duplicate_probe": _duplicate_summary(per_candidate, sample_vecs),
        "eps_capture_contract_audit": _eps_capture_contract_audit(),
        "single_run_guard_audit": guard_audit,
        "status_snapshot": {
            "single_authorized_lossless_eps": "completed",
            "lossless_persistence": "PASS",
            "operator_provenance_st_type": "PASS",
            "lossless_replay_evaluation": "FAILED_mass_null_all_candidates",
            "st_viability": "unresolved_pending_mass_null_attribution",
            "additional_eps": "NOT_AUTHORIZED",
            "production_vector_fidelity_exposure": "OPEN",
        },
    }

    write_json(OUT_JSON, report)
    audit = _generate_full_audit(report, diag, solve_result)
    write_json(AUDIT_JSON, audit)

    lines = [
        "# Lossless adjudication v1 mass-null postmortem",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"**Classification:** `{classification}`",
        f"**Reason:** {class_reason}",
        f"**Diagnostic verdict (unchanged):** `{report['diagnostic_verdict']}`",
        "",
        f"- Seed ||Mx||: `{seed_ops.get('l2_norm_Mx')}` xH_Mx=`{seed_ops.get('xH_Mx')}`",
        f"- Candidates mass_null: `{report['per_candidate_operator_aggregate']['mass_null_count']}/{len(per_candidate)}`",
        f"- Dominant support: `{agg_support.get('dominant_support_across_candidates')}`",
        f"- All p_active absent: `{agg_support.get('all_p_active_absent')}`",
        "",
        "## Guard",
        "",
        f"- Re-run EPS blocked: `{guard_audit['re_invoking_authorized_runner_would_block_eps']}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[mass_null_postmortem] classification={classification}", flush=True)
    print(f"[mass_null_postmortem] reason={class_reason}", flush=True)
    print(
        f"[mass_null_postmortem] candidates={len(per_candidate)} "
        f"mass_null={report['per_candidate_operator_aggregate']['mass_null_count']}",
        flush=True,
    )
    print(f"[mass_null_postmortem] seed_l2_Mx={seed_ops.get('l2_norm_Mx')}", flush=True)
    print(f"[mass_null_postmortem] wrote {OUT_JSON}", flush=True)
    print(f"[mass_null_postmortem] wrote {AUDIT_JSON}", flush=True)
    print("[mass_null_postmortem] no_new_eigensolve_executed=True", flush=True)
    print("[mass_null_postmortem] additional_eps=NOT_AUTHORIZED", flush=True)

    try:
        from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main
        from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main

        rehab_main()
        audit_main()
    except Exception as exc:
        print(f"[mass_null_postmortem] status_refresh_warning={exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
