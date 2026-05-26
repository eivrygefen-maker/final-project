#!/usr/bin/env python3
"""
Report-only triage of saved baseline seed-branch recovery diagnostic candidates.

No new eigensolves. Applies diagnostic-only physical eligibility filters.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fem_main_3d as fem3d
from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_seed_branch_candidate_filter import (
    FILTER_POLICY,
    VERDICT_SPURIOUS_SELECTED,
    assess_physical_eligibility,
    branch_recovery_from_row,
    replay_candidate_metrics,
)

DIAG_REPORT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_diagnostic.json"
DIAG_REPORT_MD = CONV_DIAG / "v2_l_mid_seed_branch_recovery_diagnostic.md"
AUDIT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_candidate_filter_audit.json"
AUDIT_MD = CONV_DIAG / "v2_l_mid_seed_branch_candidate_filter_audit.md"

CASE_ID = "baseline_coupled_v2"
SEED_F_HZ = 243.0754171175576
VERDICT = VERDICT_SPURIOUS_SELECTED


def _load_mode_vec(path: Path) -> np.ndarray:
    from fem_mode_array_utils import load_mode_column_any

    return np.asarray(load_mode_column_any(path).toarray(), dtype=np.float64).ravel()


def _assemble_replay(mesh_file: Path, sample: Dict[str, Any]):
    from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay

    diag_dir = solve_case_dir("L_mid", CASE_ID) / "seed_branch_recovery_diagnostic"
    sort_dir = diag_dir / "sorting_candidate_filter_audit"
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    return A, M, u_to_W, p_to_W


def _pressure_mac(seed_block: np.ndarray, p_block: np.ndarray) -> float:
    a = np.asarray(seed_block, dtype=np.complex128).ravel()
    b = np.asarray(p_block, dtype=np.complex128).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / (na * nb))


def _filtered_rerun_plan(case_dir: Path, seed_npy: Path, target_hz: float) -> Dict[str, Any]:
    out_dir = case_dir / "seed_branch_recovery_diagnostic_filtered"
    cmd = (
        "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "run_v2_l_mid_seed_branch_recovery_diagnostic_filtered.sh"
    )
    return {
        "execute_automatically": False,
        "requires_explicit_user_approval": True,
        "output_case_dir": str(out_dir),
        "solver_cfg_delta_vs_prior_diagnostic": {
            "eps_reject_sigma_spurious": True,
            "eps_reject_target_locked": True,
            "seed_branch_recovery_diagnostic": True,
            "post_evaluate_physical_filter": True,
            "note": (
                "Harvest may still save all converged modes for inspection; "
                "branch recovery verdict uses replay-based physical filter."
            ),
        },
        "recommended_command": cmd,
        "seed_file": str(seed_npy),
        "target_hz": float(target_hz),
    }


def main() -> int:
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    diag_dir = case_dir / "seed_branch_recovery_diagnostic"
    mesh_file = mesh_path("L_mid", CASE_ID)
    sample = sample_spec_from_case(case)

    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_w = _load_mode_vec(seed_npy) if seed_npy.is_file() else np.array([])

    summary_path = diag_dir / "diagnostics" / "mode_energy_summary.json"
    modes = (
        json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []
        if summary_path.is_file()
        else []
    )

    A, M, u_to_W, p_to_W = _assemble_replay(mesh_file, sample)
    p_seed = np.asarray(seed_w[p_to_W], dtype=np.float64).ravel() if seed_w.size else np.array([])

    candidates: List[Dict[str, Any]] = []
    try:
        for m in modes:
            rel = str(m.get("vector_path", ""))
            path = diag_dir / rel
            if not path.is_file():
                continue
            vec = _load_mode_vec(path)
            f_hz = float(m["frequency_hz"])
            mac = _pressure_mac(p_seed, np.asarray(vec[p_to_W], dtype=np.float64).ravel())
            replay = replay_candidate_metrics(
                A, M, vec, u_to_W=u_to_W, p_to_W=p_to_W, reported_f_hz=f_hz
            )
            elig = assess_physical_eligibility(
                reported_f_hz=f_hz,
                replay_metrics=replay,
                pressure_mac_to_true_seed=mac,
                seed_f_hz=SEED_F_HZ,
                require_mac=True,
                require_seed_frequency_match=True,
            )
            row = {
                "vector_file": str(path),
                "mode_index": m.get("mode_index"),
                "reported_frequency_hz": f_hz,
                "reported_p_frac_energy_phys": m.get("p_frac_energy_phys"),
                "pressure_MAC_to_true_seed": mac,
                "replay_rayleigh_lambda": replay["replay_rayleigh_lambda"],
                "replay_rayleigh_frequency_hz": replay["replay_rayleigh_frequency_hz"],
                "replay_relative_residual": replay["replay_relative_residual"],
                "algebraic_lambda_one_suspect": replay["algebraic_lambda_one_suspect"],
                "reported_vs_replay_frequency_consistent": replay[
                    "reported_vs_replay_frequency_consistent"
                ],
                "physically_eligible_after_filter": elig["physically_eligible_after_filter"],
                "rejection_reasons": elig["rejection_reasons"],
                "recovers_true_seed_branch": elig["recovers_true_seed_branch"],
            }
            candidates.append(row)
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    candidates.sort(
        key=lambda r: (
            not bool(r.get("recovers_true_seed_branch")),
            not bool(r.get("physically_eligible_after_filter")),
            -float(r.get("pressure_MAC_to_true_seed") or 0.0),
        )
    )

    any_recovers = any(branch_recovery_from_row(r) for r in candidates)
    eligible = [r for r in candidates if r.get("physically_eligible_after_filter")]
    branch_recoverers = [r for r in candidates if r.get("recovers_true_seed_branch")]

    focal = next(
        (r for r in candidates if "mode_243075_004" in str(r.get("vector_file", ""))),
        None,
    )

    audit = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "verdict": VERDICT,
        "true_acoustic_seed_hz": SEED_F_HZ,
        "true_acoustic_seed_remains_valid": True,
        "filter_policy": FILTER_POLICY,
        "candidate_enumeration": candidates,
        "summary": {
            "num_saved_candidates": len(candidates),
            "num_physically_eligible_after_filter": len(eligible),
            "num_recovers_true_seed_branch": len(branch_recoverers),
            "any_existing_saved_candidate_recovers_true_seed_branch": bool(any_recovers),
        },
        "focal_prior_false_recovery": focal,
        "interpretation": {
            "prior_selected_mode_was_algebraic_spurious": bool(
                focal and focal.get("algebraic_lambda_one_suspect")
            ),
            "do_not_claim_dirichlet_row_concentration": True,
            "artifact_description": (
                "Confirmed lambda≈1 sigma/mapping algebraic spurious mode; exact algebraic "
                "origin may remain unlocalized."
            ),
            "active_blocker": "EPS diagnostic candidate filtering/interpretation, not v2 coupling loss",
        },
        "filtered_diagnostic_rerun": _filtered_rerun_plan(case_dir, seed_npy, SEED_F_HZ)
        if not any_recovers
        else {
            "execute_automatically": False,
            "required": False,
            "reason": "At least one saved candidate passes physical filter and branch recovery gates.",
        },
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }

    write_json(AUDIT_JSON, audit)

    lines = [
        "# L_mid seed-branch candidate filter audit (baseline)",
        "",
        f"Generated: {audit['generated_utc']}",
        "",
        f"**Verdict:** `{VERDICT}`",
        "",
        "## Summary",
        "",
        f"- Saved candidates: {audit['summary']['num_saved_candidates']}",
        f"- Physically eligible after filter: {audit['summary']['num_physically_eligible_after_filter']}",
        f"- Recovers true seed branch: {audit['summary']['num_recovers_true_seed_branch']}",
        f"- **any_existing_saved_candidate_recovers_true_seed_branch:** "
        f"`{audit['summary']['any_existing_saved_candidate_recovers_true_seed_branch']}`",
        "",
        "## Focal false recovery (mode_243075_004)",
        "",
    ]
    if focal:
        lines.extend(
            [
                f"- reported f: {focal['reported_frequency_hz']} Hz",
                f"- replay Rayleigh f: {focal['replay_rayleigh_frequency_hz']} Hz",
                f"- replay λ: {focal['replay_rayleigh_lambda']}",
                f"- physically_eligible: {focal['physically_eligible_after_filter']}",
                f"- rejection: {focal.get('rejection_reasons')}",
                "",
            ]
        )
    if not any_recovers:
        plan = audit["filtered_diagnostic_rerun"]
        lines.extend(
            [
                "## Prepared filtered diagnostic rerun (not executed)",
                "",
                f"- Recommended: `{plan.get('recommended_command')}`",
                f"- Output dir: `{plan.get('output_case_dir')}`",
                "",
            ]
        )
    lines.append("**mesh_convergence_may_resume:** `False`\n")
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")

    if DIAG_REPORT_JSON.is_file():
        diag = json.loads(DIAG_REPORT_JSON.read_text(encoding="utf-8"))
        row = diag.setdefault("baseline_coupled_v2", {})
        ev = row.setdefault("evaluation", {})
        ev["diagnostic_verdict"] = VERDICT
        ev["candidate_filter_audit_json"] = str(AUDIT_JSON)
        ev["any_existing_saved_candidate_recovers_true_seed_branch"] = bool(any_recovers)
        row["diagnostic_verdict"] = VERDICT
        diag["candidate_filter_audit_verdict"] = VERDICT
        diag["mesh_convergence_may_resume"] = False
        write_json(DIAG_REPORT_JSON, diag)

    if DIAG_REPORT_MD.is_file():
        text = DIAG_REPORT_MD.read_text(encoding="utf-8")
        marker = "## Candidate filter audit"
        block = (
            f"\n{marker}\n\n**`{VERDICT}`** — "
            f"any_existing_saved_candidate_recovers_true_seed_branch="
            f"`{any_recovers}`\n"
        )
        if marker not in text:
            DIAG_REPORT_MD.write_text(text + block, encoding="utf-8")

    print(
        f"[candidate_filter_audit] any_recovers={any_recovers} eligible={len(eligible)}",
        flush=True,
    )
    print(f"[candidate_filter_audit] wrote {AUDIT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
