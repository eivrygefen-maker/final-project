#!/usr/bin/env python3
"""
Design-only: alternative spectral formulation plan after ST/SINVERT retirement.

Static/code-derived + optional diagnostic JSON merge. Does not call eps.solve().
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import CONV_DIAG, write_json

OUT_JSON = CONV_DIAG / "v2_alternative_spectral_formulation_after_st_retirement_plan.json"
OUT_MD = CONV_DIAG / "v2_alternative_spectral_formulation_after_st_retirement_plan.md"
OUT_OPS_JSON = (
    CONV_DIAG / "v2_solver_migration_operational_requirements_and_preflight_plan.json"
)
OUT_OPS_MD = (
    CONV_DIAG / "v2_solver_migration_operational_requirements_and_preflight_plan.md"
)

ROOT_CAUSE_STATUS = "V2_ST_SINVERT_FORMULATION_BLOCKED_AFTER_CERTIFIED_NULL_DEFLATION"
PHYSICAL_MODEL_STATUS = "V2_NOT_INVALIDATED"
SOLVER_STATUS = "ST_SINVERT_RETIRED_FOR_CURRENT_V2_SPECTRAL_FORMULATION"

EVIDENCE_PATHS = {
    "lossless_diagnostic": CONV_DIAG
    / "v2_l_mid_mapping_fixed_unregularized_lossless_adjudication_v1_diagnostic.json",
    "projected_diagnostic": CONV_DIAG
    / "v2_l_mid_mapping_fixed_unregularized_lossless_nullspace_projected_adjudication_v1_diagnostic.json",
    "projected_auth": CONV_DIAG
    / "v2_lossless_nullspace_projected_adjudication_v1_eps_authorization_record.json",
    "null_basis_preflight": CONV_DIAG
    / "v2_lossless_adjudication_v1_Muu_null_basis_certification_and_projection_preflight.json",
    "mass_rank_audit": CONV_DIAG
    / "v2_lossless_adjudication_v1_u_mass_rank_and_disjoint_partition_audit.json",
    "st_preflight": CONV_DIAG / "v2_st_singular_mass_preflight.json",
}
CONFIG_ROOT = SCRIPT_DIR.parent / "configs"
SCRIPTS_ROOT = SCRIPT_DIR.parent / "scripts"


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _slepc_eps_type_probe() -> Dict[str, Any]:
    out: Dict[str, Any] = {"eps_types": {}, "deflation": {}, "problem_types": {}}
    try:
        from petsc4py import PETSc
        import SLEPc

        out["slepc_version"] = getattr(SLEPc, "__version__", "unknown")
        for name in (
            "KRYLOVSCHUR",
            "CISS",
            "JD",
            "GD",
            "LOBPCG",
            "ARNOLDI",
            "LANCZOS",
            "SUBSPACE",
            "POWER",
        ):
            t = getattr(SLEPc.EPS.Type, name, None)
            out["eps_types"][name] = t is not None
        for pname in ("GNHEP", "GHEP", "HEP", "NHEP", "PGNHEP"):
            t = getattr(SLEPc.EPS.ProblemType, pname, None)
            out["problem_types"][pname] = t is not None
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        out["deflation"]["has_setDeflationSpace"] = hasattr(eps, "setDeflationSpace")
        try:
            eps.destroy()
        except Exception:
            pass
    except Exception as exc:
        out["import_error"] = str(exc)
    return out


def _repo_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _code_confirmed_requirements() -> Dict[str, Any]:
    conv = _repo_json(CONFIG_ROOT / "v2_mesh_convergence_manifest.json")
    prod = _repo_json(CONFIG_ROOT / "v2_production_parameter_manifest.json")
    ext = _repo_json(CONFIG_ROOT / "v2_material_structural_harvest_extension_manifest.json")
    conv_ranges = sorted(
        {
            (float(c.get("harvest_lo_hz", 0.0)), float(c.get("harvest_hi_hz", 0.0)))
            for c in (conv.get("cases") or [])
            if "harvest_lo_hz" in c and "harvest_hi_hz" in c
        }
    )
    conv_modes = {
        str(c.get("id")): int(c.get("num_modes", 0))
        for c in (conv.get("cases") or [])
        if c.get("id") is not None
    }
    items: List[Dict[str, Any]] = [
        {
            "question": "Is 60–550 Hz encoded as intended production band?",
            "classification": "USER_DECISION_REQUIRED",
            "evidence": "No 60–550 band found in manifests/scripts; only narrower windows are configured.",
        },
        {
            "question": "Which ranges/windows are configured now?",
            "classification": "CONFIRMED_FROM_CODE",
            "evidence": {
                "mesh_convergence_case_windows_hz": conv_ranges,
                "production_default_branch_capture_hz": [
                    float((prod.get("default_branch_capture") or {}).get("initial_harvest_lo_hz", 220.0)),
                    float((prod.get("default_branch_capture") or {}).get("initial_harvest_hi_hz", 265.0)),
                ],
                "production_widen_attempts_hz": [
                    [float(w.get("harvest_lo_hz", 0.0)), float(w.get("harvest_hi_hz", 0.0))]
                    for w in ((prod.get("default_branch_capture") or {}).get("widen_attempts") or [])
                ],
                "locator_band_hz": [
                    float((prod.get("locator_policy") or {}).get("locator_harvest_lo_hz", 150.0)),
                    float((prod.get("locator_policy") or {}).get("locator_harvest_hi_hz", 350.0)),
                ],
                "material_extension_uniform_hz": [
                    float((ext.get("harvest_policy") or {}).get("harvest_lo_hz", 200.0)),
                    float((ext.get("harvest_policy") or {}).get("harvest_hi_hz", 320.0)),
                ],
            },
        },
        {
            "question": "How many modes are requested where defined?",
            "classification": "CONFIRMED_FROM_CODE",
            "evidence": {
                "mesh_convergence_num_modes_by_case": conv_modes,
                "production_initial_num_modes": int(
                    (prod.get("default_branch_capture") or {}).get("initial_num_modes", 12)
                ),
                "production_widen_num_modes": [
                    int(w.get("num_modes", 0))
                    for w in ((prod.get("default_branch_capture") or {}).get("widen_attempts") or [])
                ],
                "material_extension_num_modes": int(
                    (ext.get("harvest_policy") or {}).get("num_modes", 30)
                ),
            },
        },
        {
            "question": "Which outputs are currently required downstream (ROM/LHS)?",
            "classification": "CONFIRMED_FROM_CODE",
            "evidence": {
                "currently_persisted": [
                    "frequency_hz/in-band classification",
                    "candidate vectors (sparse; optional lossless dense in adjudication lanes)",
                    "replay diagnostics, MAC, residual/mass-norm style gates",
                    "JSON+MD summaries",
                ],
                "status_flags": [
                    "lhs_promotion_blocked=True",
                    "mesh_convergence_pass=Pending",
                    "production_parameter_coverage_pass=Pending",
                ],
                "note": "No active production ROM/LHS consumer contract in these manifests beyond blocked gating fields.",
            },
        },
        {
            "question": "Are production harvesting and mesh/branch validation separated?",
            "classification": "CONFIRMED_FROM_CODE",
            "evidence": {
                "separate_paths": [
                    "run_v2_mesh_convergence.py",
                    "run_v2_sensitivity_production_stage.py",
                    "run_v2_material_structural_harvest_extension.py",
                ],
                "shared_worker": "v2_sensitivity_solve.py",
            },
        },
        {
            "question": "Must production return every physical mode / fixed count / family split?",
            "classification": "USER_DECISION_REQUIRED",
            "evidence": "Not encoded as a finalized requirement in repo configs.",
        },
    ]
    return {"classification_legend": ["CONFIRMED_FROM_CODE", "USER_DECISION_REQUIRED", "NOT_YET_IMPLEMENTED"], "items": items}


def _worker_parallel_audit() -> Dict[str, Any]:
    return {
        "current_code_support": {
            "worker_model_today": "independent process-per-case runs (mpiexec -n 1 per solve), orchestrated by Python loops",
            "three_workers_status": "intended external parallel orchestration; no in-process 3-worker scheduler in inspected scripts",
            "mpi_usage": "single-rank solves; not one distributed EPS across 3 ranks for separate windows",
            "per_worker_assembly": True,
            "per_worker_factorization_or_preconditioner": True,
            "artifact_merge": "summary JSON/MD aggregation by post scripts",
            "window_overlap_dedup": "limited/implicit via branch selection and post summaries; no explicit global interval dedup engine",
            "reusable_for_jd_gd_or_ciss": [
                "case manifests",
                "worker launching pattern",
                "lossless persistence + replay/MAC gates",
                "post aggregation scripts",
            ],
        },
        "three_worker_risk_table": [
            {
                "risk": "RAM pressure",
                "severity": "high",
                "why": "Each concurrent worker assembles operators and holds solver/preconditioner state.",
                "mitigation": "Benchmark 1-worker peak RSS first, then 2/3 concurrent with identical case profile.",
            },
            {
                "risk": "Repeated assembly cost",
                "severity": "medium-high",
                "why": "No shared assembled A/M cache across workers.",
                "mitigation": "Accept duplication or add optional shared artifacts later.",
            },
            {
                "risk": "Repeated factorization/preconditioner cost",
                "severity": "high",
                "why": "Each solve builds ST/JD/CISS internals independently.",
                "mitigation": "Window sizing and solver tuning; possible per-window reuse only inside worker process.",
            },
            {
                "risk": "I/O + lossless persistence overhead",
                "severity": "medium",
                "why": "Lossless vectors can be large; simultaneous writes compete on VM disk.",
                "mitigation": "Keep lossless mandatory for diagnostics, optionally reduce retained vectors per gate policy.",
            },
            {
                "risk": "Dedup/coverage ambiguity across overlaps",
                "severity": "medium-high",
                "why": "No strict global branch ledger for overlap ownership.",
                "mitigation": "Add deterministic overlap ownership + MAC-based merge policy before production.",
            },
            {
                "risk": "Debugging branch coverage misses",
                "severity": "medium",
                "why": "Without explicit coverage accounting, misses can look like solver failure.",
                "mitigation": "Coverage report per window + cross-window branch map in post stage.",
            },
        ],
        "minimal_future_benchmark_required": {
            "note": "Runtime/speedup cannot be trusted without execution.",
            "smallest_benchmark": [
                "single representative case, fixed mesh/operator",
                "1 worker vs 3 concurrent workers on disjoint windows",
                "capture wall-time, peak RSS, nconv, replay/MAC/mass-norm pass counts",
            ],
        },
    }


def _runtime_evidence_framing() -> Dict[str, Any]:
    return {
        "MEASURED": {
            "valid_production_equivalent_solver_runtime": "none",
            "note": "No trustworthy runtime baseline exists yet for a valid cleaned-formulation production solver.",
        },
        "CONFIRMED_FROM_CODE": {
            "configured_windows_and_counts": True,
            "orchestration_capabilities": True,
            "note": "Repository manifests and scripts define current windows/mode counts and process orchestration.",
        },
        "ENGINEERING_EXPECTATION_PENDING_BENCHMARK": {
            "jd_gd_primary": "likely best production candidate after cleaned formulation",
            "ciss_reference": "likely best narrow-band reference/certification",
            "continuation_tracking": "likely best branch-tracking tool",
            "frequency_response": "likely unsuitable as full-band production engine",
            "numeric_runtime_percentages": "not defensible before benchmark",
        },
    }


def _code_integration_points() -> Dict[str, Any]:
    return {
        "operator_assembly": "fem_main_3d._solve_coupled_evp (coupled_physical_core_v2)",
        "current_eps_dispatch": "fem_main_3d._slepc_shift_invert_batch / _slepc_coupled_band_solve",
        "ciss_path": "fem_main_3d._slepc_shift_invert_batch when eps_band_solver=ciss",
        "worker_entry": "v2_sensitivity_solve.py",
        "lossless_persistence": "v2_mapping_fixed_candidate_persistence.py",
        "replay_evaluator": "v2_unreg_offset_report_evaluator + v2_lossless_adjudication_evaluator",
        "clean_lane_pattern": "isolated output subdir + gated runner + preflight JSON gates",
        "mapping_fix": "v2_eps_mapping_audit_lib / slepc_backtransformed semantics",
    }


def _evidence_chain_summary() -> Dict[str, Any]:
    lossless = _load(EVIDENCE_PATHS["lossless_diagnostic"])
    projected = _load(EVIDENCE_PATHS["projected_diagnostic"])
    nb = _load(EVIDENCE_PATHS["null_basis_preflight"])
    mr = _load(EVIDENCE_PATHS["mass_rank_audit"])
    lev = lossless.get("evaluation") or {}
    pev = projected.get("evaluation") or {}
    return {
        "physical_model": "coupled_physical_core_v2",
        "seed_replay_valid": True,
        "seed_frequency_hz_approx": 243.0754171175576,
        "ruled_out_active_causes": [
            "lossy_sparse_serialization",
            "uncorrected_eigenvalue_mapping",
            "candidate_persistence_failure",
            "st_type_provenance_gap",
            "operator_policy_mismatch_at_replay",
        ],
        "unprojected_lossless_st": {
            "nconv": lev.get("eps_nconv_marked"),
            "lossless_saved": lev.get("lossless_vectors_saved"),
            "mass_null_count": (lev.get("summary") or {}).get("num_mass_null"),
            "branch_recovery_pass_count": lev.get("num_branch_recovery_pass")
            or (lev.get("summary") or {}).get("num_branch_recovery_pass"),
            "verdict": lev.get("diagnostic_verdict"),
        },
        "structural_attribution": {
            "classification_subtype": mr.get("classification_subtype")
            or nb.get("classification_subtype"),
            "certified_null_dimension": nb.get("certified_empirical_null_basis_dimension"),
        },
        "projected_st_final": {
            "projection_basis_dimension": (projected.get("pre_eps_projection_gate") or {}).get(
                "projection_basis_dimension"
            ),
            "deflation_applied": (pev.get("projection_runtime") or {}).get("deflation_applied"),
            "verdict": pev.get("final_projected_adjudication_verdict"),
            "branch_recovery_pass_count": pev.get("branch_recovery_pass_count"),
            "mass_null_after_projection": pev.get("mass_null_candidate_count_after_projection"),
            "eps_run_consumed": projected.get("eps_run_count_for_projected_lane"),
        },
        "st_retirement_decision": (
            "ST/SINVERT + empirical u-side deflation failed before and after certified-null "
            "deflation inside EPS; path retired for current V2 spectral formulation."
        ),
    }


def _comparison_rows() -> List[Dict[str, Any]]:
    """Qualitative comparison; probability fields are engineering estimates."""
    def row(
        id_: str,
        name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"id": id_, "candidate": name, **kwargs}

    return [
        row(
            "A",
            "Cleaned physical/mass-bearing formulation + JD/GD",
            role="primary_candidate_for_production",
            mathematical_change="Project to complement of certified/physical structural nullspace in M_uu; preserve coupled GNHEP on range",
            weak_form_change="none (projection layer on assembled operators)",
            v2_physics_preservation="full",
            recover_l_mid_branch_likelihood="medium-high (0.55-0.75 estimate) if range is correct",
            production_mode_bank_likelihood="medium-high (0.5-0.7 estimate)",
            singular_M_robustness="high (addresses root cause)",
            implementation_difficulty="medium",
            code_areas="new formulation module; replay assembly; evaluator contracts",
            debugability="medium-high (explicit range + lossless gates)",
            silent_artifact_risk="low if lossless-first",
            wall_time_vs_st="unknown-to-comparable (estimate only; benchmark required)",
            three_worker_compat="high",
            method_natural_partition="frequency windows with overlap",
            if_production_band_60_550_hz="feasible via windowing; requires benchmarked RAM+wall-time and overlap dedup policy",
            replay_mac_gates_compat="high",
            new_dependencies="none",
            slepc_api_uncertainty="low",
            recertification_burden="medium",
            stop_rule="If seed breaks under cleaned replay without EPS, halt formulation design",
        ),
        row(
            "B",
            "Seeded continuation / Rayleigh-Ritz branch tracking",
            role="validation_branch_tracking",
            mathematical_change="Track known branch via seed + local projection/continuation",
            weak_form_change="none if built on A",
            v2_physics_preservation="full when using A",
            recover_l_mid_branch_likelihood="high (0.7-0.85 estimate) for known branch",
            production_mode_bank_likelihood="low-medium (0.2-0.4 estimate) alone",
            singular_M_robustness="medium (depends on cleaned subspace)",
            implementation_difficulty="low-medium",
            code_areas="existing continuation hooks; MAC/replay gates",
            debugability="high",
            silent_artifact_risk="low",
            wall_time_vs_st="likely lower per tracked branch",
            three_worker_compat="medium",
            method_natural_partition="seeds/branches rather than global band harvest",
            if_production_band_60_550_hz="insufficient alone for full-bank extraction; valuable for validation gates",
            replay_mac_gates_compat="high",
            new_dependencies="none",
            slepc_api_uncertainty="low",
            recertification_burden="low-medium",
            stop_rule="Keep as branch-validation tool even if not production engine",
        ),
        row(
            "C",
            "CISS on cleaned formulation (reference)",
            role="reference_certification",
            mathematical_change="Contour integral over frequency interval on cleaned operators",
            weak_form_change="none if operators unchanged",
            v2_physics_preservation="full",
            recover_l_mid_branch_likelihood="medium (0.45-0.6 estimate)",
            production_mode_bank_likelihood="medium-high for interval harvest (0.5-0.7 estimate)",
            singular_M_robustness="medium-high on cleaned pencil",
            implementation_difficulty="low-medium (CISS path exists)",
            code_areas="fem_main_3d CISS branch; worker band windows",
            debugability="medium",
            silent_artifact_risk="medium",
            wall_time_vs_st="moderately higher to much higher (estimate only)",
            three_worker_compat="high",
            method_natural_partition="contours/intervals by window",
            if_production_band_60_550_hz="possible but likely costly; better as narrow-band reference",
            replay_mac_gates_compat="high",
            new_dependencies="none",
            slepc_api_uncertainty="low (CISS probed in code)",
            recertification_burden="medium",
            stop_rule="Reference only unless JD/GD fails; not primary production default",
        ),
        row(
            "D",
            "Frequency-response sweep (oracle)",
            role="external_resonance_oracle",
            mathematical_change="Forced response / transfer function, not EVP",
            weak_form_change="none",
            v2_physics_preservation="full",
            recover_l_mid_branch_likelihood="high for peak location (0.75-0.9 estimate)",
            production_mode_bank_likelihood="not a direct mode bank (0.1-0.2 estimate)",
            singular_M_robustness="high (bypasses null EVP modes)",
            implementation_difficulty="medium-high",
            code_areas="new driver scripts; linear solves per frequency",
            debugability="high",
            silent_artifact_risk="low",
            wall_time_vs_st="much higher for wide-band dense sweeps",
            three_worker_compat="low-medium",
            method_natural_partition="dense frequency samples",
            if_production_band_60_550_hz="impractical as primary due to dense sampling cost",
            replay_mac_gates_compat="partial (frequency only)",
            new_dependencies="none",
            slepc_api_uncertainty="n/a",
            recertification_burden="low for certification role",
            stop_rule="Certify resonance location only; not production harvest",
        ),
        row(
            "E",
            "Schur / pressure-led / PEP-NEP reformulation",
            role="deep_fallback_or_rearchitecture",
            mathematical_change="Eliminate or reduce structural DOFs analytically; reformulate EVP",
            weak_form_change="substantial risk",
            v2_physics_preservation="requires proof",
            recover_l_mid_branch_likelihood="medium-high if correct (0.5-0.75 estimate)",
            production_mode_bank_likelihood="high long-term (0.6-0.8 estimate)",
            singular_M_robustness="high fundamental",
            implementation_difficulty="high",
            code_areas="assembly, workers, replay, LHS, serializers",
            debugability="low-medium",
            silent_artifact_risk="high during migration",
            wall_time_vs_st="unknown",
            three_worker_compat="medium after rewrite",
            method_natural_partition="depends on final reformulation",
            if_production_band_60_550_hz="unknown until reformulation and implementation exist",
            replay_mac_gates_compat="requires full re-derivation",
            new_dependencies="possible",
            slepc_api_uncertainty="medium",
            recertification_burden="very high",
            stop_rule="Deep fallback only after JD/GD+CISS on cleaned formulation fail",
        ),
        row(
            "F",
            "ST/SINVERT on redesigned cleaned formulation (benchmark only)",
            role="benchmark_control_not_next_route",
            mathematical_change="Same ST mechanism on fundamentally different pencil",
            weak_form_change="none if only range change",
            v2_physics_preservation="full if range-only",
            recover_l_mid_branch_likelihood="uncertain (0.35-0.55 estimate)",
            production_mode_bank_likelihood="unknown",
            singular_M_robustness="medium-high if cleaned",
            implementation_difficulty="low (reuse ST path)",
            code_areas="fem_main_3d ST branch",
            debugability="low (history of ambiguous null modes)",
            silent_artifact_risk="high",
            wall_time_vs_st="baseline",
            three_worker_compat="high",
            method_natural_partition="frequency windows",
            if_production_band_60_550_hz="possible but explicitly not next authorized route",
            replay_mac_gates_compat="high",
            new_dependencies="none",
            slepc_api_uncertainty="low",
            recertification_burden="medium",
            stop_rule="Not next authorized solve; benchmark only after primary path chosen",
        ),
    ]


def _migration_assessment() -> List[Dict[str, Any]]:
    return [
        {
            "candidate": "A (cleaned formulation + JD/GD)",
            "integration_difficulty": "medium",
            "unchanged_components": [
                "coupled_physical_core_v2 mesh/forms/BCs/materials",
                "case/mesh definitions",
                "three-worker frequency windows (with new EPS driver)",
                "lossless persistence pattern",
                "replay/MAC/mass-norm gates",
            ],
            "likely_break_points": [
                "range projector definition generalization beyond L_mid",
                "JD preconditioner for indefinite coupled block",
                "SLEPc JD/GD API wiring",
            ],
            "failure_diagnosability": "medium-high if lossless+replay gates enforced from day one",
            "new_packages": "none expected (API support check only)",
            "no_eps_preflight_stages_before_first_solve": 3,
        },
        {
            "candidate": "C (CISS reference)",
            "integration_difficulty": "low-medium",
            "unchanged_components": ["CISS contour path exists in fem_main_3d"],
            "likely_break_points": ["unclean pencil still returns spurious contour modes"],
            "failure_diagnosability": "medium",
            "new_packages": "none",
            "no_eps_preflight_stages_before_first_solve": 2,
        },
        {
            "candidate": "B (continuation)",
            "integration_difficulty": "low-medium",
            "unchanged_components": ["seed infrastructure", "MAC/replay"],
            "likely_break_points": ["mode swapping across mesh/geometry"],
            "failure_diagnosability": "high",
            "new_packages": "none",
            "no_eps_preflight_stages_before_first_solve": 2,
        },
    ]


def _why_st_failed() -> Dict[str, Any]:
    return {
        "mechanism": (
            "GNHEP with singular structural mass block: EPS/ST harvests u_active-dominated "
            "vectors in or near null(M). Persistence and lossless capture were faithful."
        ),
        "empirical_sequence": [
            "56/56 unprojected lossless ST candidates mass-null",
            "U_NULLSPACE_SHELL_MASS_MATRIX_KERNEL attribution",
            "23-dim certified-null basis; seed preserved under projection",
            "deflation_applied=True inside EPS search space",
            "56/56 projected ST candidates still mass-null; branch recovery 0",
        ],
        "replacement_must": [
            "Not target the same easy null-space family on the same contaminated pencil",
            "Operate on physically justified cleaned/range-constrained formulation or equivalent",
            "Preserve lossless authoritative capture and replay gates from first alternative solve",
        ],
    }


def _recommendation() -> Dict[str, Any]:
    return {
        "primary": {
            "path": "Physical mass-bearing / constrained cleaned formulation + JD/GD",
            "rationale": (
                "Directly addresses the demonstrated M_uu kernel while preserving coupled_physical_core_v2. "
                "JD/GD is the natural SLEPc replacement for targeted interior modes with seed guidance. "
                "Repository already uses SLEPc GNHEP; CISS exists as secondary reference path."
            ),
            "challenged_user_expectation": "aligned",
        },
        "secondary_reference": {
            "path": "CISS on cleaned formulation over a narrow band near 243 Hz",
            "rationale": (
                "Existing CISS integration in fem_main_3d; useful to certify completeness inside "
                "a region after formulation cleanup, not as uncontrolled production default."
            ),
        },
        "branch_tracking_tool": {
            "path": "Seeded continuation / Rayleigh-Ritz with MAC and replay gates",
            "rationale": (
                "High likelihood of following the known acoustic branch across mesh changes; "
                "should run in parallel with JD/GD development, not as sole production engine."
            ),
        },
        "retired_not_recommended": [
            "Another empirical ST deflation patch on current pencil",
            "ST/SINVERT as next primary route",
            "sigma tweaks or persistence remapping diagnostics",
            "Stage-2 as immediate workaround",
        ],
    }


def _phased_plan() -> Dict[str, Any]:
    return {
        "phase_0": {
            "name": "status_closure_and_api_static_preflight",
            "authorized_eps": False,
            "tasks": [
                "Publish this alternatives plan and refresh root-cause status reports",
                "Probe SLEPc EPS.Type JD/GD/CISS availability in target environment",
                "Audit repository-configured ranges/modes/worker orchestration and classify CONFIRMED vs USER_DECISION_REQUIRED",
                "Document integration points and preserved components",
            ],
        },
        "phase_1": {
            "name": "formulation_design_preflight_only",
            "authorized_eps": False,
            "tasks": [
                "Define cleaned mass-bearing range projector (generalize beyond empirical 23-vector basis)",
                "Prove seed replay-consistent on cleaned operators without EPS",
                "Define lossless persistence and evaluator contracts for new lane",
                "Persist reconstruction metadata and fail-closed nullspace exposure checks",
            ],
        },
        "phase_2": {
            "name": "one_future_solver_diagnostic",
            "authorized_eps": False,
            "note": "Requires explicit review after Phase 0-1; one isolated tree; one authorization only",
            "tasks": [
                "JD/GD on cleaned formulation in new isolated output subdir",
                "Strict lossless/replay/MAC/mass-norm gates",
            ],
        },
        "phase_3": {
            "name": "reference_compare_and_mesh_convergence",
            "authorized_eps": False,
            "note": "Only if Phase 2 recovers physical branch; production still blocked",
            "tasks": [
                "CISS or forced-response reference on same cleaned formulation",
                "Mesh convergence reassessment",
                "Serializer/worker/LHS recertification",
            ],
        },
        "global_stop_rule": (
            "If first selected alternative fails with mass-null or unverifiable outputs under "
            "clean preflighted formulation, do not open unbounded patch cycle—return to comparison "
            "table and promote next ranked alternative (e.g. CISS reference, then Schur/PEP)."
        ),
    }


def _authoritative_status_block() -> Dict[str, Any]:
    return {
        "root_cause_status": ROOT_CAUSE_STATUS,
        "current_physical_model_status": PHYSICAL_MODEL_STATUS,
        "current_solver_status": SOLVER_STATUS,
        "additional_eps": "NOT_AUTHORIZED",
        "mesh_convergence_resume": "BLOCKED",
        "production_promotion": "BLOCKED",
        "stage_2_immediate_workaround": "NOT_AUTHORIZED",
    }


def main() -> int:
    slepc = _slepc_eps_type_probe()
    requirements = _code_confirmed_requirements()
    worker_audit = _worker_parallel_audit()
    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "design_only_no_eps",
        "authoritative_status": _authoritative_status_block(),
        "evidence_chain_summary": _evidence_chain_summary(),
        "why_st_failed": _why_st_failed(),
        "candidate_0_st_empirical_deflation": {
            "status": "TESTED_AND_RETIRED",
            "note": (
                "Certified-null deflation inside EPS was executed (deflation_applied=True, dim=23) "
                "and still yielded 56/56 mass-null candidates. Not a viable primary path."
            ),
        },
        "slepc_static_probe": slepc,
        "operational_requirements_from_repo": requirements,
        "worker_parallel_architecture_audit": worker_audit,
        "runtime_evidence_framing": _runtime_evidence_framing(),
        "code_integration_points": _code_integration_points(),
        "comparison_table": _comparison_rows(),
        "migration_assessment": _migration_assessment(),
        "recommendation": _recommendation(),
        "phased_plan": _phased_plan(),
        "solver_swap_vs_formulation_cleanup": {
            "solver_swap_only": (
                "JD/GD or CISS on the raw singular pencil likely repeats null-space harvesting "
                "(estimated probability 0.6-0.8 of similar failure class)."
            ),
            "formulation_cleanup": (
                "Project/restrict to physical mass-bearing range while preserving V2 weak forms; "
                "required before meaningful solver swap (estimated probability of progress 0.55-0.75)."
            ),
        },
        "no_eps_command_in_this_deliverable": True,
        "no_eps_preflight_command": (
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "write_v2_alternative_spectral_formulation_after_st_retirement_plan.py"
        ),
    }

    write_json(OUT_JSON, report)

    rec = report["recommendation"]
    lines = [
        "# Alternative spectral formulation after ST/SINVERT retirement",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Authoritative status",
        "",
        f"- **root_cause_status:** `{ROOT_CAUSE_STATUS}`",
        f"- **current_physical_model_status:** `{PHYSICAL_MODEL_STATUS}`",
        f"- **current_solver_status:** `{SOLVER_STATUS}`",
        "- **additional_eps:** NOT_AUTHORIZED",
        "- **mesh_convergence_resume:** BLOCKED",
        "- **production_promotion:** BLOCKED",
        "",
        "## Recommendation",
        "",
        f"**Primary:** {rec['primary']['path']}",
        "",
        f"**Secondary reference:** {rec['secondary_reference']['path']}",
        "",
        f"**Branch tracking:** {rec['branch_tracking_tool']['path']}",
        "",
        "## ST retirement",
        "",
        report["candidate_0_st_empirical_deflation"]["note"],
        "",
        "## SLEPc static probe (local)",
        "",
        f"- JD available: `{slepc.get('eps_types', {}).get('JD')}`",
        f"- GD available: `{slepc.get('eps_types', {}).get('GD')}`",
        f"- CISS available: `{slepc.get('eps_types', {}).get('CISS')}`",
        f"- setDeflationSpace: `{slepc.get('deflation', {}).get('has_setDeflationSpace')}`",
        "",
        "## Operational notes from code",
        "",
        "- Frequency windows are currently narrow-band (e.g., 220–265, 200–320, 200–350); 60–550 is not codified.",
        "- Worker model in scripts is process-per-solve (`mpiexec -n 1`), not a built-in 3-window scheduler.",
        "- Production/LHS remains blocked in status fields; runtime percentages are intentionally not claimed.",
        "",
        "See JSON for full comparison table and phased plan.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ops_report = {
        "generated_utc": report["generated_utc"],
        "authoritative_status": _authoritative_status_block(),
        "operational_requirements_from_repo": requirements,
        "worker_parallel_architecture_audit": worker_audit,
        "runtime_evidence_framing": _runtime_evidence_framing(),
        "slepc_static_probe": slepc,
        "ranked_operational_candidates": report["comparison_table"],
        "recommendation": report["recommendation"],
        "phased_plan": report["phased_plan"],
        "phase_1_no_eps_command": (
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "write_v2_alternative_spectral_formulation_after_st_retirement_plan.py"
        ),
        "no_eps_command_in_this_deliverable": True,
    }
    write_json(OUT_OPS_JSON, ops_report)
    OUT_OPS_MD.write_text(
        "\n".join(
            [
                "# Solver migration operational requirements and preflight plan",
                "",
                f"Generated: {report['generated_utc']}",
                "",
                "## Status",
                "",
                f"- `root_cause_status`: `{ROOT_CAUSE_STATUS}`",
                f"- `current_physical_model_status`: `{PHYSICAL_MODEL_STATUS}`",
                f"- `current_solver_status`: `{SOLVER_STATUS}`",
                "- `additional_eps`: `NOT_AUTHORIZED`",
                "- `mesh_convergence_resume`: `BLOCKED`",
                "- `production_promotion`: `BLOCKED`",
                "",
                "## Confirmed from code",
                "",
                "- Current configured windows include 220–265, 200–320, 200–350 and case-specific bands.",
                "- Worker orchestration is process-per-case (`mpiexec -n 1`), externally parallelizable but not internally window-sharded.",
                "- KRYLOVSCHUR and CISS paths are wired in solver dispatch; JD/GD require integration wiring.",
                "",
                "## Recommendation",
                "",
                f"- Primary: {rec['primary']['path']}",
                f"- Reference: {rec['secondary_reference']['path']}",
                f"- Branch tracking: {rec['branch_tracking_tool']['path']}",
                "",
                "See JSON for full operational classification table, risk table, and stop rules.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    try:
        from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main
        from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main

        rehab_main()
        audit_main()
    except Exception as exc:
        print(f"[alternatives_plan] status_refresh_warning={type(exc).__name__}:{exc}", flush=True)

    print(f"[alternatives_plan] wrote {OUT_JSON}", flush=True)
    print(f"[alternatives_plan] wrote {OUT_OPS_JSON}", flush=True)
    print(f"[alternatives_plan] root_cause_status={ROOT_CAUSE_STATUS}", flush=True)
    print(f"[alternatives_plan] primary_recommendation={rec['primary']['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
