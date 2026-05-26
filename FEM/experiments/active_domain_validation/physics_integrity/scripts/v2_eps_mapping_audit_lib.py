#!/usr/bin/env python3
"""Shared helpers for EPS eigenvalue-mapping impact and replay recertification."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PHYSICS_ROOT / "scripts"
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"

MAPPING_FIX_SUMMARY = {
    "function": "fem_main_3d._slepc_physical_lambda",
    "native_path": "EPS + ST.SINVERT; eps.getEigenpair returns back-transformed λ",
    "old_behavior": "sinvert fallback sigma+1/mu; shift ST used lam_phys=mu+sigma",
    "new_behavior": "lam_phys=mu when eps_eigenvalue_semantics=slepc_backtransformed",
    "legacy_double_shift_mapping_disabled": True,
    "manual_semantics_required_for": ["manual_sinvert_theta", "manual_st_shift"],
}


def static_exposure_inventory() -> List[Dict[str, Any]]:
    """Code-derived exposure table (artifact availability resolved at runtime)."""
    rows = [
        {
            "path_id": "L0_coupled_physical_core_v2_baseline",
            "description": "Frozen L0 coupled_physical_core_v2 subcases (coupling on/off)",
            "entrypoint": "v2_sensitivity_solve / fem_main_3d._solve_coupled_evp",
            "uses_native_STSINVERT_EPS": True,
            "uses_legacy_mu_plus_sigma_mapping": "shift_st_only_historically; sinvert used mu or sigma+1/mu fallback",
            "artifact_root_guess": "coupled_physical_core_v2/{subcase}",
            "mesh_topology_gate_separate": True,
        },
        {
            "path_id": "v2_geometry_sensitivity",
            "description": "Geometry sensitivity manifest samples (hole radius, etc.)",
            "entrypoint": "run_v2_sensitivity_validation.py -> v2_sensitivity_solve",
            "uses_native_STSINVERT_EPS": True,
            "uses_legacy_mu_plus_sigma_mapping": True,
            "artifact_root_guess": "v2_sensitivity_validation/samples/{sample_id}",
            "mesh_topology_gate_separate": True,
        },
        {
            "path_id": "v2_material_validation",
            "description": "Material validation rows (wood IDs / scales)",
            "entrypoint": "run_v2_sensitivity_validation.py",
            "uses_native_STSINVERT_EPS": True,
            "uses_legacy_mu_plus_sigma_mapping": True,
            "artifact_root_guess": "v2_sensitivity_validation/samples/{sample_id}",
            "mesh_topology_gate_separate": True,
        },
        {
            "path_id": "material_structural_harvest_extension",
            "description": "Structural spectrum harvest extension (weak coupling allowed)",
            "entrypoint": "v2_sensitivity_solve --structural-spectrum-harvest",
            "uses_native_STSINVERT_EPS": True,
            "uses_legacy_mu_plus_sigma_mapping": True,
            "artifact_root_guess": "v2_sensitivity_validation/material_structural_harvest_extension",
            "mesh_topology_gate_separate": True,
        },
        {
            "path_id": "L_mid_mesh_convergence",
            "description": "L_mid mesh-convergence baseline and diagnostics",
            "entrypoint": "v2_mesh_convergence runners -> v2_sensitivity_solve",
            "uses_native_STSINVERT_EPS": True,
            "uses_legacy_mu_plus_sigma_mapping": True,
            "artifact_root_guess": "v2_mesh_convergence/solves/L_mid/{case_id}",
            "mesh_topology_gate_separate": True,
        },
        {
            "path_id": "seed_branch_diagnostics",
            "description": "Experiment-only seed-branch recovery diagnostics",
            "entrypoint": "v2_sensitivity_solve --seed-branch-*-diagnostic",
            "uses_native_STSINVERT_EPS": True,
            "uses_legacy_mu_plus_sigma_mapping": True,
            "artifact_root_guess": "v2_mesh_convergence/solves/L_mid/baseline_coupled_v2/*diagnostic*",
            "mesh_topology_gate_separate": False,
            "note": "Diagnostic verdicts superseded; not prior PASS production claims",
        },
        {
            "path_id": "future_FOM_LHS_production",
            "description": "Future FOM / LHS three-worker production path",
            "entrypoint": "fem_master_dynamic / worker ST ladder (not yet promoted)",
            "uses_native_STSINVERT_EPS": True,
            "uses_legacy_mu_plus_sigma_mapping": "unknown_until_promotion",
            "artifact_root_guess": None,
            "mesh_topology_gate_separate": True,
            "status_after_mapping_fix": "not_examined",
        },
    ]
    for r in rows:
        if r.get("status_after_mapping_fix") is None:
            r["status_after_mapping_fix"] = "pending_vm_replay_recertification"
        r["saved_modes_available_for_replay_recertification"] = "vm_runtime_check"
    return rows


def discover_replay_targets(repo_root: Path) -> List[Dict[str, Any]]:
    """Find case directories with modes/ and optional result JSON (VM may have more)."""
    roots = [
        repo_root / "FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_core_v2",
        repo_root / "FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/samples",
        repo_root / "FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation",
        repo_root / "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/solves",
    ]
    targets: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for summary in base.rglob("diagnostics/mode_energy_summary.json"):
            case_dir = summary.parent.parent
            key = str(case_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            modes = sorted((case_dir / "modes").glob("mode_*.smx.npz")) if (case_dir / "modes").is_dir() else []
            results = sorted((case_dir / "results").glob("result_*.json")) if (case_dir / "results").is_dir() else []
            targets.append(
                {
                    "case_dir": key,
                    "mode_energy_summary": str(summary),
                    "num_modes_on_disk": len(modes),
                    "result_json": str(results[0]) if results else None,
                    "has_replay_inputs": bool(modes and results),
                }
            )
    return targets
