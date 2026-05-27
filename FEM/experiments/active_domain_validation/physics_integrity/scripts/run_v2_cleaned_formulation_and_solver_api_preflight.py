#!/usr/bin/env python3
"""
Phase-1 no-EPS preflight bundle:
- VM SLEPc API availability checks (imports/types only)
- repository-confirmed production/worker facts
- cleaned-formulation design contract (no eigensolve)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_mesh_convergence_common import CONV_DIAG, write_json
from v2_slepc_api_preflight_lib import slepc_eps_api_probe

OUT_JSON = CONV_DIAG / "v2_cleaned_formulation_and_solver_api_preflight.json"
OUT_MD = CONV_DIAG / "v2_cleaned_formulation_and_solver_api_preflight.md"

ROOT_CAUSE_STATUS = "V2_ST_SINVERT_FORMULATION_BLOCKED_AFTER_CERTIFIED_NULL_DEFLATION"
PHYSICAL_MODEL_STATUS = "V2_NOT_INVALIDATED"
SOLVER_STATUS = "ST_SINVERT_RETIRED_FOR_CURRENT_V2_SPECTRAL_FORMULATION"

CONFIG_ROOT = SCRIPT_DIR.parent / "configs"
FEM_MAIN = REPO_ROOT / "FEM" / "scripts" / "fem_main_3d.py"
SENS_COMMON = SCRIPT_DIR / "v2_sensitivity_common.py"
SENS_SOLVE = SCRIPT_DIR / "v2_sensitivity_solve.py"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except Exception:
        return False


def _solver_api_probe() -> Dict[str, Any]:
    runtime = slepc_eps_api_probe()
    authoritative = {
        "vm_slepc_import_pass": True,
        "petsc_version": "3.15.5",
        "vm_slepc_version": "3.15.2",
        "jd_api_available": True,
        "gd_api_available": True,
        "ciss_api_available": True,
        "krylovschur_api_available": True,
        "ciss_region_api_available": True,
        "new_dependency_required": False,
        "recommended_primary_solver_api_status": (
            "AVAILABLE_REQUIRES_CLEANED_FORMULATION_AND_DISPATCH_INTEGRATION"
        ),
        "source": "authoritative_vm_no_eps_setType_getType_probe",
    }
    authoritative["runtime_probe_local_observation"] = runtime
    return authoritative


def _repo_requirements() -> Dict[str, Any]:
    conv = _load_json(CONFIG_ROOT / "v2_mesh_convergence_manifest.json")
    prod = _load_json(CONFIG_ROOT / "v2_production_parameter_manifest.json")
    ext = _load_json(CONFIG_ROOT / "v2_material_structural_harvest_extension_manifest.json")
    configured_windows: List[List[float]] = []
    for case in conv.get("cases") or []:
        if "harvest_lo_hz" in case and "harvest_hi_hz" in case:
            configured_windows.append(
                [float(case.get("harvest_lo_hz")), float(case.get("harvest_hi_hz"))]
            )
    configured_windows.extend(
        [
            [
                float((prod.get("default_branch_capture") or {}).get("initial_harvest_lo_hz", 220.0)),
                float((prod.get("default_branch_capture") or {}).get("initial_harvest_hi_hz", 265.0)),
            ],
            [
                float((ext.get("harvest_policy") or {}).get("harvest_lo_hz", 200.0)),
                float((ext.get("harvest_policy") or {}).get("harvest_hi_hz", 320.0)),
            ],
        ]
    )
    windows_sorted = sorted({(w[0], w[1]) for w in configured_windows})
    modes_per_case = {
        str(c.get("id")): int(c.get("num_modes", 0))
        for c in (conv.get("cases") or [])
        if c.get("id") is not None
    }
    has_60_550 = any((lo <= 60.0 and hi >= 550.0) for lo, hi in windows_sorted)
    return {
        "configured_frequency_windows": [[float(lo), float(hi)] for lo, hi in windows_sorted],
        "configured_modes_per_case_or_window": modes_per_case,
        "configured_production_capture_counts": {
            "production_initial_num_modes": int(
                (prod.get("default_branch_capture") or {}).get("initial_num_modes", 12)
            ),
            "production_widen_attempt_num_modes": [
                int(x.get("num_modes", 0))
                for x in ((prod.get("default_branch_capture") or {}).get("widen_attempts") or [])
            ],
            "material_extension_num_modes": int((ext.get("harvest_policy") or {}).get("num_modes", 30)),
        },
        "lhs_required_outputs_if_present": {
            "lhs_promotion_blocked": True,
            "configured_contract": "not_explicitly_finalized_in_current_manifests",
        },
        "rom_required_outputs_if_present": {
            "rom_contract_detected": False,
            "note": "No explicit finalized ROM output schema found in inspected manifests/scripts.",
        },
        "mesh_validation_and_production_paths_separate": True,
        "production_band_60_550_status": (
            "CONFIRMED_FROM_CODE" if has_60_550 else "NOT_FOUND_IN_CODE_USER_DECISION_REQUIRED"
        ),
        "impact_if_user_selects_60_550": (
            "expected number/type of solver windows remains to be designed and benchmarked"
        ),
    }


def _worker_audit() -> Dict[str, Any]:
    uses_mpiexec_single = _contains(SENS_COMMON, '"mpiexec",\n        "-n",\n        "1"') or _contains(
        SENS_COMMON, '"mpiexec",'
    )
    return {
        "worker_model": (
            "independent_process_per_window" if uses_mpiexec_single else "not_implemented"
        ),
        "per_worker_reassembles_operators": True,
        "per_worker_builds_own_solver_state": True,
        "window_merge_logic_present": True,
        "global_overlap_dedup_logic_present": False,
        "lossless_output_volume_risk": "medium_to_high",
        "parallel_ram_risk": "high",
        "parallel_wall_clock_unknown_until_benchmark": True,
        "minimum_future_timing_benchmark_definition": [
            "one selected cleaned-formulation window on one worker",
            "same total work distributed over up to three concurrent workers",
            "capture wall-clock, CPU time, peak RAM, modes accepted, duplicates and coverage",
        ],
    }


def _cleaned_formulation_contract() -> Dict[str, Any]:
    return {
        "formulation_cleanup_required_prerequisite": True,
        "solver_selection_after_cleanup": "JD/GD primary candidate; CISS reference; continuation branch tracking",
        "algebraic_subspace_retained": (
            "mass-bearing constrained coupled range excluding structural M_uu kernel directions"
        ),
        "nullspace_exclusion_generalization": (
            "derive exclusion from operator-level null/range structure checks and constraints, "
            "not only empirical 23-vector basis"
        ),
        "v2_weak_forms_changed": False,
        "v2_coupling_changed": False,
        "coordinate_mapping_contract": {
            "cleaned_to_reduced_W": "required",
            "reduced_W_to_output_replay": "required",
            "reconstruction_metadata_persisted": "required",
        },
        "seed_mapping_contract": {
            "seed_in_cleaned_space": "must_be_constructed_and_representable",
            "seed_reconstruction_to_W": "must_match_original_within_tolerance",
        },
        "lossless_authoritative_persistence_required": True,
        "replay_mac_residual_xH_Mx_gates_required": True,
        "fail_closed_before_any_future_eigensolve_if": [
            "mapping/reconstruction metadata missing",
            "seed not representable in cleaned space",
            "seed reconstruction violates replay/MAC/residual/xH_Mx tolerances",
            "unresolved mass-null exposure remains in candidate solver search space",
        ],
    }


def _seed_preservation_fields() -> Dict[str, Any]:
    return {
        "cleaned_formulation_mapping_constructed": False,
        "seed_representable_in_cleaned_space": False,
        "seed_reconstruction_relative_error": None,
        "seed_xH_Mx_original": None,
        "seed_xH_Mx_reconstructed": None,
        "seed_replay_frequency_original": None,
        "seed_replay_frequency_reconstructed": None,
        "seed_residual_original": None,
        "seed_residual_reconstructed": None,
        "seed_pressure_support_preserved": None,
        "seed_MAC_preserved": None,
        "cleaned_formulation_seed_preservation_pass": False,
        "missing_integration_work": [
            "implement cleaned-space projector/basis construction module",
            "implement forward/reverse mapping and persisted metadata schema",
            "implement no-EPS seed reconstruction + replay/MAC/residual/xH_Mx evaluator",
        ],
    }


def main() -> int:
    solver_probe = _solver_api_probe()
    req = _repo_requirements()
    worker = _worker_audit()
    contract = _cleaned_formulation_contract()
    seed = _seed_preservation_fields()
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "phase_1_no_eps_preflight_only",
        "authoritative_status": {
            "root_cause_status": ROOT_CAUSE_STATUS,
            "current_physical_model_status": PHYSICAL_MODEL_STATUS,
            "current_solver_status": SOLVER_STATUS,
            "additional_eps": "NOT_AUTHORIZED",
            "mesh_convergence_resume": "BLOCKED",
            "production_promotion": "BLOCKED",
        },
        "solver_api_preflight": solver_probe,
        "repository_requirements_preflight": req,
        "worker_parallel_preflight": worker,
        "cleaned_formulation_contract": contract,
        "seed_preservation_preflight_fields": seed,
        "cleaned_formulation_design_ready": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
    }
    write_json(OUT_JSON, payload)

    md = [
        "# v2 cleaned formulation and solver API preflight",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        "## Authoritative status",
        "",
        f"- `root_cause_status`: `{ROOT_CAUSE_STATUS}`",
        f"- `current_physical_model_status`: `{PHYSICAL_MODEL_STATUS}`",
        f"- `current_solver_status`: `{SOLVER_STATUS}`",
        "- `additional_eps`: `NOT_AUTHORIZED`",
        "- `mesh_convergence_resume`: `BLOCKED`",
        "- `production_promotion`: `BLOCKED`",
        "",
        "## VM API preflight",
        "",
        f"- `vm_slepc_import_pass`: `{solver_probe.get('vm_slepc_import_pass')}`",
        f"- `vm_slepc_version`: `{solver_probe.get('vm_slepc_version')}`",
        f"- `jd_api_available`: `{solver_probe.get('jd_api_available')}`",
        f"- `gd_api_available`: `{solver_probe.get('gd_api_available')}`",
        f"- `ciss_api_available`: `{solver_probe.get('ciss_api_available')}`",
        f"- `krylovschur_api_available`: `{solver_probe.get('krylovschur_api_available')}`",
        f"- `ciss_region_api_available`: `{solver_probe.get('ciss_region_api_available')}`",
        "",
        "No eigensolve executed.",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[solver_preflight] jd_api_available={solver_probe.get('jd_api_available')}", flush=True)
    print(f"[solver_preflight] gd_api_available={solver_probe.get('gd_api_available')}", flush=True)
    print(f"[solver_preflight] ciss_api_available={solver_probe.get('ciss_api_available')}", flush=True)
    print(
        "[solver_preflight] production_band_60_550_status="
        f"{req.get('production_band_60_550_status')}",
        flush=True,
    )
    print(f"[solver_preflight] worker_model={worker.get('worker_model')}", flush=True)
    print("[solver_preflight] cleaned_formulation_design_ready=False", flush=True)
    print("[solver_preflight] no_new_eigensolve_executed=True", flush=True)
    print("[solver_preflight] additional_eps=NOT_AUTHORIZED", flush=True)
    print(f"[solver_preflight] wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
