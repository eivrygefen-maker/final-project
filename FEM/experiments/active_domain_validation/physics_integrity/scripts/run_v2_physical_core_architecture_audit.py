#!/usr/bin/env python3
"""
Static architecture and hidden-policy audit for coupled_physical_core_v2 (local source only).

Does not read VM runtime artifacts or run EPS.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
PHYSICS_SCRIPTS = SCRIPT_DIR
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_conservative_audit_policy import SERIALIZER_FUNCTION, SERIALIZER_THRESHOLD
from v2_eps_mapping_audit_lib import MAPPING_FIX_SUMMARY, static_exposure_inventory
from v2_mesh_convergence_common import CONV_DIAG

OUT_JSON = CONV_DIAG / "v2_physical_core_architecture_and_hidden_policy_audit.json"
OUT_MD = CONV_DIAG / "v2_physical_core_architecture_and_hidden_policy_audit.md"

SEARCH_ROOTS = (
    FEM_SCRIPTS,
    PHYSICS_SCRIPTS,
)
KEYWORD_PATTERNS = (
    r"\bfilter\b",
    r"\breject\b",
    r"\bthreshold\b",
    r"\btol\b",
    r"\brtol\b",
    r"\batol\b",
    r"\bMAC\b",
    r"\bp_frac\b",
    r"\bmass.?null\b",
    r"\bRayleigh\b",
    r"\bsigma\b",
    r"\bst_a_shift\b",
    r"\bmass_reg\b",
    r"\bPGNHEP\b",
    r"\bpurif",
    r"\bpressure_dof_scale\b",
    r"\bfsi_coupling_gain\b",
    r"dense_to_csr",
    r"MODE_VECTOR_RELATIVE_EPS",
    r"\bcontinuation_seed\b",
    r"\bpreserve_all\b",
    r"\bverdict\b",
    r"\bfallback\b",
    r"\blegacy\b",
)


def _system_map() -> List[Dict[str, Any]]:
    return [
        {
            "stage": "geometry_sample_parameters",
            "location": "FEM/geometry/build_3d_guitar.py, manifest JSON",
            "outputs": "mesh inputs, hole radius, material IDs",
            "physical_meaning": "parameterization only",
        },
        {
            "stage": "mesh_creation_tagging",
            "location": "v2_mesh_convergence_mesh.py, gmsh pipelines",
            "outputs": ".msh + mesh_audit.json",
            "physical_meaning": "discretization",
        },
        {
            "stage": "coupled_v2_form_assembly",
            "location": "fem_main_3d.py, v2_build_coupled_acoustic_seed._assemble_reduced_coupled_replay",
            "outputs": "PETSc A, M, cfg maps u_to_W, p_to_W",
            "physical_meaning": "yes",
            "diagnostic_vs_production": "same assembly kernel; diagnostic flags select ST/EPS path",
        },
        {
            "stage": "air_pressure_restriction_reduced_W",
            "location": "fem_main_3d coupled air restriction",
            "outputs": "reduced W dimension, p_to_W map",
            "physical_meaning": "yes",
        },
        {
            "stage": "EPS_ST_solve",
            "location": "fem_main_3d._slepc_shift_invert_batch, v2_sensitivity_solve._solve_coupled_evp",
            "outputs": "eigenpairs, eps_batch_diagnostics",
            "physical_meaning": "yes",
        },
        {
            "stage": "eigenvalue_mapping",
            "location": "fem_main_3d._slepc_physical_lambda",
            "outputs": "lam_phys, reported_frequency_hz",
            "physical_meaning": "labeling / Rayleigh interpretation",
        },
        {
            "stage": "preserve_all_capture",
            "location": "fem_main_3d preserve_all_nconv",
            "outputs": "_eps_diagnostic_candidate_bank_records dense numpy",
            "physical_meaning": "in-memory only until persist",
        },
        {
            "stage": "worker_filter_bridge",
            "location": "fem_main_3d worker path, fem_worker_single.py",
            "outputs": "eigvecs stack or dropped rows",
            "physical_meaning": "can drop candidates when rt/rb None unless preserve_all",
        },
        {
            "stage": "persistence_serialization",
            "location": "v2_mapping_fixed_candidate_persistence, fem_mode_array_utils",
            "outputs": ".smx.npz (+ optional .smx.dense.npy diagnostic)",
            "physical_meaning": "storage transform may alter replay",
        },
        {
            "stage": "load_replay",
            "location": "v2_unreg_offset_report_evaluator, physical_fsi_seed_residual_audit",
            "outputs": "Rayleigh, residual, MAC",
            "physical_meaning": "evaluation only",
        },
        {
            "stage": "verdict_reporting",
            "location": "v2_mapping_fixed_baseline_evaluator, pipeline audit scripts",
            "outputs": "diagnostic_verdict JSON/MD",
            "physical_meaning": "reporting only",
        },
    ]


def _active_paths() -> List[Dict[str, Any]]:
    rows = list(static_exposure_inventory())
    rows.append(
        {
            "path_id": "mapping_fixed_unregularized_persistence_fixed",
            "description": "L_mid replacement baseline with preserve-all + persistence fix",
            "entrypoint": "run_v2_l_mid_mapping_fixed_unregularized_persistence_fixed_baseline_diagnostic.py",
            "configuration_trigger": "seed_branch_recovery_diagnostic_mapping_fixed_unregularized_persistence_fixed",
            "status": "executed_once_VM",
            "trusted_for_branch_verdict": False,
            "blocked": "physical_verdict until lossless persistence",
        }
    )
    return rows


def _grep_mechanisms() -> List[Dict[str, Any]]:
    combined = re.compile("|".join(KEYWORD_PATTERNS), re.IGNORECASE)
    hits: List[Dict[str, Any]] = []
    seen: set = set()
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "venv" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if not combined.search(line):
                    continue
                key = (str(path.relative_to(REPO_ROOT)), i)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "mechanism_id": f"{path.stem}:{i}",
                        "file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                        "line": i,
                        "snippet": line.strip()[:200],
                        "risk_classification": "review_required",
                    }
                )
                if len(hits) >= 400:
                    return hits
    return hits


def _serialization_contract() -> Dict[str, Any]:
    return {
        "serializer_function": SERIALIZER_FUNCTION,
        "MODE_VECTOR_RELATIVE_EPS": 1e-7,
        "serializer_threshold": SERIALIZER_THRESHOLD,
        "source_vector_object_type": "dense numpy float64 (PETSc rvec.array / bank record)",
        "normalization_before_save": "optional per-path; seed normalized; EPS modes as harvested",
        "dtype_before_save": "float64 dense",
        "dtype_after_save": "float32 CSR",
        "dense_to_sparse_rules": "drop |x| < eps*max(|x|); eliminate_zeros",
        "serialization_may_change_physical_replay_metrics": True,
        "lossless_pre_sparsify_eps_vectors_available_in_current_run": False,
        "current_saved_vectors_sufficient_for_st_verdict": False,
        "diagnostic_lossless_option": (
            "v2_mapping_fixed_candidate_persistence.persist_candidate_bank_with_optional_lossless "
            "(.smx.dense.npy); isolated future output tree only"
        ),
        "production_paths_share_lossy_writer": True,
        "downstream_consumers_assuming_fidelity": [
            "v2_mapping_fixed_baseline_evaluator",
            "run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit",
            "v2_unreg_offset_report_evaluator",
        ],
    }


def _provenance_graph() -> List[Dict[str, Any]]:
    fields = [
        "continuation_seed_applied",
        "actual_sigma_hz",
        "st_type",
        "eps_eigenvalue_semantics",
        "legacy_double_shift_mapping_disabled",
        "diagnostic_operator_consistent_with_replay",
        "actual_st_a_shift_frac",
        "actual_st_mass_reg_frac",
        "preserve_all_enabled",
        "nconv_marked",
        "num_vectors_saved",
        "p_to_W",
        "serializer_function",
    ]
    graph: List[Dict[str, Any]] = []
    for name in fields:
        graph.append(
            {
                "field": name,
                "set_in": [
                    "fem_main_3d eps_batch_diagnostics",
                    "v2_sensitivity_solve solve result",
                    "worker result payload",
                ],
                "persisted_in": [
                    "results/result_*.json",
                    "diagnostics/eps_candidate_bank.json",
                    "diagnostics/mode_energy_summary.json",
                    "CONV_DIAG *baseline_diagnostic.json",
                ],
                "read_by": [
                    "v2_mapping_fixed_baseline_evaluator",
                    "run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit",
                    "v2_conservative_audit_policy.build_operator_policy_from_artifacts",
                ],
                "known_fragility": (
                    "top-level solve_result may omit fields present only in eps_batch_diagnostics"
                ),
            }
        )
    return graph


def _trust_boundary() -> Dict[str, Any]:
    return {
        "trusted": [
            "true seed replay near 243.075417 Hz (self-test)",
            "runtime p_to_W map contract length 24039 crc32 2027087254",
            "56/56 candidate .smx.npz files persisted",
            "sparse persisted candidates replay as mass-null (VM audit)",
        ],
        "not_yet_trustworthy": [
            "dense EPS candidates were mass-null in memory",
            "ST failed to recover physical branch",
            "prior EPS-based frequency labels before mapping recertification",
            "production readiness / mesh_convergence_resume",
        ],
        "minimal_evidence_for_baseline_st_viability": [
            "lossless pre-sparsify candidate vectors persisted",
            "finite xH_Mx and Rayleigh on >=1 candidate from lossless replay",
            "operator policy fields proven consistent across artifacts",
        ],
        "minimal_evidence_for_hole_radius_large": [
            "authoritative baseline branch verdict with lossless vectors",
        ],
        "minimal_evidence_for_L_prod_L_check": [
            "mesh_convergence gate + production promotion audit",
        ],
    }


def main() -> int:
    mechanisms = _grep_mechanisms()
    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "local_source_code_only",
        "mapping_fix_summary": MAPPING_FIX_SUMMARY,
        "system_map": _system_map(),
        "active_paths": _active_paths(),
        "hidden_policy_inventory": mechanisms,
        "hidden_policy_inventory_count": len(mechanisms),
        "serialization_and_vector_contract": _serialization_contract(),
        "config_metadata_provenance_graph": _provenance_graph(),
        "physical_interpretation_trust_boundary": _trust_boundary(),
        "no_eigensolve_executed": True,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# v2 physical core architecture and hidden-policy audit",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Serialization contract",
        "",
        f"- serializer: `{report['serialization_and_vector_contract']['serializer_function']}`",
        f"- threshold: `{report['serialization_and_vector_contract']['serializer_threshold']}`",
        f"- may change replay metrics: `{report['serialization_and_vector_contract']['serialization_may_change_physical_replay_metrics']}`",
        f"- current run lossless vectors: `{report['serialization_and_vector_contract']['lossless_pre_sparsify_eps_vectors_available_in_current_run']}`",
        "",
        "## Trust boundary",
        "",
    ]
    for item in report["physical_interpretation_trust_boundary"]["trusted"]:
        lines.append(f"- trusted: {item}")
    for item in report["physical_interpretation_trust_boundary"]["not_yet_trustworthy"]:
        lines.append(f"- not yet trustworthy: {item}")
    lines.append("")
    lines.append(f"Mechanism hits (sample cap): {len(mechanisms)}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[architecture_audit] wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
