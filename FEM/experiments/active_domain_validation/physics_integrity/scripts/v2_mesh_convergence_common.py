#!/usr/bin/env python3
"""Paths and helpers for v2_mesh_convergence (experiment-only)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PHYSICS_ROOT / "scripts"
CONV_ROOT = PHYSICS_ROOT / "v2_mesh_convergence"
CONV_MESH = CONV_ROOT / "mesh"
CONV_SOLVES = CONV_ROOT / "solves"
CONV_DIAG = CONV_ROOT / "diagnostics"
CONV_MANIFEST = PHYSICS_ROOT / "configs" / "v2_mesh_convergence_manifest.json"
SENS_ROOT = PHYSICS_ROOT / "v2_sensitivity_validation"
HARVEST_EXT_ROOT = SENS_ROOT / "material_structural_harvest_extension"
VALIDATION_MESH = (
    Path(__file__).resolve().parents[2] / "mesh" / "validation_tiny_guitar_3d.msh"
).resolve()
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
SUMMARY_JSON = CONV_DIAG / "v2_mesh_convergence_summary.json"
SUMMARY_MD = CONV_DIAG / "v2_mesh_convergence_summary.md"
INCREMENTAL_JSON = CONV_DIAG / "v2_mesh_convergence_incremental.json"
VALIDATION_STATUS_JSON = SENS_ROOT / "diagnostics" / "v2_validation_status.json"

COUPLED_BASELINE_F_HZ = 244.394153389752


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest() -> Dict[str, Any]:
    return json.loads(CONV_MANIFEST.read_text(encoding="utf-8"))


def case_by_id(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    for c in manifest.get("cases") or []:
        if str(c.get("id")) == case_id:
            return c
    raise KeyError(case_id)


def mesh_path(level_id: str, case_id: str) -> Path:
    return CONV_MESH / level_id / f"{case_id}.msh"


def mesh_audit_path(level_id: str, case_id: str) -> Path:
    return CONV_MESH / level_id / f"{case_id}_mesh_audit.json"


def solve_case_dir(level_id: str, case_id: str) -> Path:
    return CONV_SOLVES / level_id / case_id


def solve_result_path(level_id: str, case_id: str, target_hz: float) -> Path:
    from v2_sensitivity_common import hz_result_tag

    return solve_case_dir(level_id, case_id) / "results" / f"result_{hz_result_tag(target_hz)}.json"


def _acoustic_branch_ok(data: Dict[str, Any]) -> bool:
    import math

    branch = data.get("acoustic_branch_by_energy") or data.get("nearest_acoustic_branch")
    if not branch:
        return False
    f_hz = float(branch.get("frequency_hz", float("nan")))
    p_frac = float(branch.get("p_frac_energy_phys", float("nan")))
    if not math.isfinite(f_hz) or not math.isfinite(p_frac):
        return False
    cls = str(branch.get("mode_class_physical_energy", ""))
    if cls == "acoustic_dominated":
        return True
    return p_frac >= 0.85


def solve_done(level_id: str, case: Dict[str, Any]) -> bool:
    p = solve_result_path(level_id, str(case["id"]), float(case["target_hz"]))
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not bool(data.get("v2_converged")):
            return False
        if str(case.get("case_type")) == "acoustic":
            return _acoustic_branch_ok(data)
        if bool(case.get("structural_spectrum_harvest")):
            return True
        return True
    except Exception:
        return False


def run_mpi_case_solve(
    sample: Dict[str, Any],
    mesh_path_file: Path,
    *,
    target_hz: float,
    harvest_lo_hz: float,
    harvest_hi_hz: float,
    num_modes: int,
    log_path: Path,
    case_dir: Path,
    select_by_energy: bool = False,
    structural_spectrum_harvest: bool = False,
    reference_f_hz: Optional[float] = None,
) -> Tuple[int, Dict[str, Any]]:
    from v2_sensitivity_common import hz_result_tag

    sample_id = str(sample["id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    sample_json = case_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(SOLVE_SCRIPT),
        "--sample-id",
        sample_id,
        "--mesh",
        str(mesh_path_file.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--target-hz",
        str(float(target_hz)),
        "--harvest-lo-hz",
        str(float(harvest_lo_hz)),
        "--harvest-hi-hz",
        str(float(harvest_hi_hz)),
        "--num-modes",
        str(int(num_modes)),
        "--case-root",
        str(case_dir.parent.resolve()),
    ]
    if select_by_energy:
        cmd.append("--select-by-energy")
    if structural_spectrum_harvest:
        cmd.append("--structural-spectrum-harvest")
    if reference_f_hz is not None:
        cmd.extend(["--reference-f-hz", str(float(reference_f_hz))])
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    rp = case_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    result: Dict[str, Any] = {}
    if rp.is_file():
        result = json.loads(rp.read_text(encoding="utf-8"))
    return proc.returncode, result


def sample_spec_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": case["id"],
        "geometry": case.get("geometry") or {},
        "materials": case.get("materials") or {},
    }
