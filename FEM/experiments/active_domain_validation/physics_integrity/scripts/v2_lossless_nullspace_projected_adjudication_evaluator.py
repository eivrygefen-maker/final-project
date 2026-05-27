#!/usr/bin/env python3
"""Evaluate nullspace-projected lossless adjudication v1 artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1
from v2_certified_null_projection_lib import (
    load_persisted_Q_certified,
    orthogonality_fraction_to_Q,
)
from v2_lossless_adjudication_evaluator import (
    VERDICT_LOSSLESS_ROUNDTRIP_FAILURE,
    VERDICT_PERSISTENCE_FAILURE,
    assign_lossless_adjudication_verdict,
    counterfactual_production_filter_labels,
)
from v2_mapping_fixed_baseline_evaluator import mapping_fixed_branch_recovery_from_row

VERDICT_PROJECTED_BRANCH_RECOVERED = (
    "MAPPING_FIXED_UNREGULARIZED_PROJECTED_BASELINE_BRANCH_RECOVERED"
)
VERDICT_PROJECTED_NO_BRANCH = (
    "MAPPING_FIXED_UNREGULARIZED_PROJECTED_BASELINE_NO_PHYSICAL_BRANCH_RECOVERED"
)
VERDICT_PROJECTED_REPLAY_FAILURE = (
    "MAPPING_FIXED_UNREGULARIZED_PROJECTED_BASELINE_LOSSLESS_REPLAY_EVALUATION_FAILURE"
)
VERDICT_PROJECTED_ST_BLOCKER = (
    "MAPPING_FIXED_UNREGULARIZED_PROJECTED_BASELINE_ST_FORMULATION_BLOCKER"
)
VERDICT_INCONSISTENT = "MAPPING_FIXED_UNREGULARIZED_PROJECTED_BASELINE_OUTPUT_OR_REPLAY_INCONSISTENT"


def assign_projected_adjudication_verdict(
    *,
    base_verdict: str,
    candidates: List[Dict[str, Any]],
    mass_null_count: int,
    continuation: bool,
    op_ok: bool,
    semantics_ok: bool,
) -> str:
    if mass_null_count > 0 and mass_null_count >= max(1, len(candidates) // 2):
        return VERDICT_PROJECTED_ST_BLOCKER
    if base_verdict == VERDICT_PERSISTENCE_FAILURE:
        return VERDICT_PERSISTENCE_FAILURE
    if base_verdict == VERDICT_LOSSLESS_ROUNDTRIP_FAILURE:
        return VERDICT_LOSSLESS_ROUNDTRIP_FAILURE
    if base_verdict.endswith("REPLAY_EVALUATION_FAILURE"):
        return VERDICT_PROJECTED_REPLAY_FAILURE
    if not continuation or not op_ok or not semantics_ok:
        return VERDICT_INCONSISTENT
    if any(mapping_fixed_branch_recovery_from_row(c) for c in candidates):
        return VERDICT_PROJECTED_BRANCH_RECOVERED
    if candidates and all(c.get("metrics_computation_status") == "ok" for c in candidates):
        return VERDICT_PROJECTED_NO_BRANCH
    return VERDICT_INCONSISTENT


def evaluate_projected_lossless_adjudication_artifacts(
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
    from v2_lossless_adjudication_evaluator import evaluate_lossless_adjudication_artifacts

    base = evaluate_lossless_adjudication_artifacts(
        out_dir=out_dir,
        case_dir=case_dir,
        mesh_file=mesh_file,
        seed_npy=seed_npy,
        seed_meta=seed_meta,
        fallback_sample=fallback_sample,
        solve_result=solve_result,
        target_hz=target_hz,
    )
    base_verdict = str(base.get("diagnostic_verdict", VERDICT_INCONSISTENT))
    candidates = list(base.get("candidates") or [])

    Q_cert, proj_meta = load_persisted_Q_certified(out_dir)
    u_to_W = np.asarray(
        (base.get("replay_assembly") or {}).get("u_to_W")
        or solve_result.get("u_to_W")
        or [],
        dtype=np.int32,
    ).ravel()

    from fem_mode_array_utils import load_mode_dense_f64_lossless

    ortho_fracs: List[float] = []
    mass_null_after = 0
    for c in candidates:
        rel = c.get("lossless_vector_path") or c.get("vector_path")
        if not rel:
            continue
        lp = out_dir / str(rel)
        if not lp.is_file():
            continue
        vec = np.asarray(load_mode_dense_f64_lossless(lp), dtype=np.float64).ravel()
        if u_to_W.size:
            ortho = orthogonality_fraction_to_Q(vec[u_to_W], Q_cert)
            ortho_fracs.append(ortho)
            c["orthogonality_to_Q_certified_null_fraction"] = ortho
        xhm = c.get("xH_Mx")
        if xhm is None or not math.isfinite(float(xhm)) or abs(float(xhm)) < 1e-30:
            mass_null_after += 1
        c["mass_null_after_projection_run"] = bool(
            xhm is None or not math.isfinite(float(xhm)) or abs(float(xhm)) < 1e-30
        )

    eps_diag = solve_result.get("eps_batch_diagnostics") or {}
    verdict = assign_projected_adjudication_verdict(
        base_verdict=base_verdict,
        candidates=candidates,
        mass_null_count=mass_null_after,
        continuation=bool(base.get("continuation_seed_applied")),
        op_ok=bool((base.get("st_operator_fields") or {}).get("diagnostic_operator_consistent_with_replay")),
        semantics_ok=True,
    )

    branch_pool = [c for c in candidates if mapping_fixed_branch_recovery_from_row(c)]
    return {
        **base,
        "evidence_scope": "lossless_nullspace_projected_adjudication_v1_authoritative_evaluation",
        "output_subdir": OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1,
        "projection_metadata": proj_meta,
        "projection_runtime": eps_diag,
        "certified_null_orthogonality_summary": {
            "median_fraction": float(np.median(ortho_fracs)) if ortho_fracs else 0.0,
            "max_fraction": float(np.max(ortho_fracs)) if ortho_fracs else 0.0,
        },
        "mass_null_candidate_count_after_projection": mass_null_after,
        "branch_recovery_pass_count": len(branch_pool),
        "diagnostic_verdict": verdict,
        "base_lossless_evaluator_verdict": base_verdict,
        "final_projected_adjudication_verdict": verdict,
        "no_additional_eps_run_authorized": True,
        "stop_rule_note": (
            "Final ST adjudication attempt. No second projected EPS without new authorization."
        ),
    }
