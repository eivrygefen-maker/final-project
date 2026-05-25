#!/usr/bin/env python3
"""Shared helpers for v2_sensitivity_validation (experiment-only)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PHYSICS_ROOT / "scripts"
SENS_ROOT = PHYSICS_ROOT / "v2_sensitivity_validation"
DIAG_DIR = SENS_ROOT / "diagnostics"
V2_ROOT = PHYSICS_ROOT / "coupled_physical_core_v2"
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
MANIFEST_PATH = PHYSICS_ROOT / "configs" / "v2_sensitivity_manifest.json"
PRODUCTION_MANIFEST_PATH = PHYSICS_ROOT / "configs" / "v2_production_parameter_manifest.json"
PRODUCTION_SUMMARY_JSON = DIAG_DIR / "v2_production_validation_summary.json"
VALIDATION_STATUS_JSON = DIAG_DIR / "v2_validation_status.json"
SUMMARY_JSON = DIAG_DIR / "v2_sensitivity_validation_summary.json"
LOCATOR_SCRIPT = SCRIPT_DIR / "v2_sensitivity_locator.py"
VALIDATION_MESH = (EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh").resolve()
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"

COUPLED_BASELINE_F_HZ = 244.394153389752
COUPLED_BASELINE_P_FRAC = 0.9998
N_REDUCED_W_VALIDATION = 112100
DEFAULT_HARVEST_LO = 220.0
DEFAULT_HARVEST_HI = 265.0
DEFAULT_TARGET_HZ = 244.39
DEFAULT_NUM_MODES = 12
WIDEN_NUM_MODES = 16
ENERGY_ACOUSTIC_THRESHOLD = 0.85


def hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_production_manifest() -> Dict[str, Any]:
    return json.loads(PRODUCTION_MANIFEST_PATH.read_text(encoding="utf-8"))


def production_sample_by_id(manifest: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    for s in manifest.get("samples") or []:
        if str(s["id"]) == sample_id:
            return s
    raise KeyError(sample_id)


def sample_by_id(manifest: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    for s in manifest.get("samples") or []:
        if str(s["id"]) == sample_id:
            return s
    raise KeyError(sample_id)


def coupled_baseline_row(manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    frozen = (manifest or load_manifest()).get("frozen_baseline") or {}
    return {
        "sample_id": "baseline_coupled_v2",
        "ingest_only": True,
        "status": "ok",
        "mesh_gates_skipped": True,
        "v2_converged": True,
        "nearest_acoustic_f_hz": COUPLED_BASELINE_F_HZ,
        "p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
        "acoustic_modal_energy_phys": None,
        "structural_modal_energy_phys": None,
        "mode_class_physical_energy": "acoustic_dominated",
        "nearest_acoustic_branch": {
            "frequency_hz": COUPLED_BASELINE_F_HZ,
            "p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
            "mode_class_physical_energy": "acoustic_dominated",
            "source": "validated coupled_physical_core_v2 post report",
        },
        "source": str(frozen.get("subcase_coupled", "physical_coupling_enabled")),
    }


def is_acoustic_branch(branch: Optional[Dict[str, Any]]) -> bool:
    if not branch:
        return False
    if str(branch.get("mode_class_physical_energy")) == "acoustic_dominated":
        return True
    return float(branch.get("p_frac_energy_phys", 0.0)) >= ENERGY_ACOUSTIC_THRESHOLD


def branch_capture_plan(sample: Dict[str, Any], manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered solve attempts: initial band then optional widen bands."""
    bc = dict(manifest.get("default_branch_capture") or {})
    bc.update(sample.get("branch_capture") or {})
    initial = {
        "label": "initial",
        "harvest_lo_hz": float(bc.get("initial_harvest_lo_hz", DEFAULT_HARVEST_LO)),
        "harvest_hi_hz": float(bc.get("initial_harvest_hi_hz", DEFAULT_HARVEST_HI)),
        "target_hz": float(bc.get("initial_target_hz", DEFAULT_TARGET_HZ)),
        "num_modes": int(bc.get("initial_num_modes", DEFAULT_NUM_MODES)),
    }
    attempts = [initial]
    for i, w in enumerate(bc.get("widen_attempts") or []):
        attempts.append(
            {
                "label": str(w.get("label", f"widen_{i + 1}")),
                "harvest_lo_hz": float(w["harvest_lo_hz"]),
                "harvest_hi_hz": float(w["harvest_hi_hz"]),
                "target_hz": float(w.get("target_hz", DEFAULT_TARGET_HZ)),
                "num_modes": int(w.get("num_modes", WIDEN_NUM_MODES)),
            }
        )
    return attempts


def run_mpi_solve(
    sample: Dict[str, Any],
    mesh_path: Path,
    *,
    target_hz: float,
    harvest_lo_hz: float,
    harvest_hi_hz: float,
    num_modes: int,
    log_path: Path,
    select_by_energy: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    sample_id = str(sample["id"])
    case_dir = SENS_ROOT / "samples" / sample_id
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
        str(mesh_path.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--target-hz",
        str(float(target_hz)),
        "--harvest-lo-hz",
        str(float(harvest_lo_hz)),
        "--harvest-hi-hz",
        str(float(harvest_hi_hz)),
        "--reference-f-hz",
        str(COUPLED_BASELINE_F_HZ),
        "--num-modes",
        str(int(num_modes)),
    ]
    if select_by_energy:
        cmd.append("--select-by-energy")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = case_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    if not result_path.is_file():
        return int(proc.returncode), {
            "error": f"solve worker exit {proc.returncode}; missing {result_path}",
            "solve_log": str(log_path),
            "mpi_command": " ".join(cmd),
        }
    solve = json.loads(result_path.read_text(encoding="utf-8"))
    solve["solve_exit_code"] = int(proc.returncode)
    solve["solve_log"] = str(log_path)
    solve["mpi_command"] = " ".join(cmd)
    return int(proc.returncode), solve


def structural_branches_summary(in_band: List[Dict[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    rows = [
        m
        for m in in_band
        if str(m.get("mode_class_physical_energy")) == "structural_dominated"
        or float(m.get("p_frac_energy_phys", 1.0)) <= 0.15
    ]
    rows.sort(key=lambda m: float(m.get("frequency_hz", 0.0)))
    out: List[Dict[str, Any]] = []
    for m in rows[:limit]:
        out.append(
            {
                "frequency_hz": float(m["frequency_hz"]),
                "p_frac_energy_phys": float(m.get("p_frac_energy_phys", float("nan"))),
                "structural_modal_energy_phys": float(
                    m.get("structural_modal_energy_phys", float("nan"))
                ),
                "acoustic_modal_energy_phys": float(
                    m.get("acoustic_modal_energy_phys", float("nan"))
                ),
                "mass_cross_term_phys": float(m.get("mass_cross_term_phys", float("nan"))),
                "mode_class_physical_energy": m.get("mode_class_physical_energy"),
            }
        )
    return out


def run_acoustic_locator(
    sample: Dict[str, Any],
    mesh_path: Path,
    *,
    policy: Dict[str, Any],
    log_path: Path,
) -> Tuple[int, Dict[str, Any]]:
    sample_id = str(sample["id"])
    case_dir = SENS_ROOT / "samples" / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    sample_json = case_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    out_json = case_dir / "diagnostics" / "acoustic_locator.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(LOCATOR_SCRIPT),
        "--mesh",
        str(mesh_path.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--locator-lo-hz",
        str(float(policy.get("locator_harvest_lo_hz", 150.0))),
        "--locator-hi-hz",
        str(float(policy.get("locator_harvest_hi_hz", 350.0))),
        "--reference-hz",
        str(COUPLED_BASELINE_F_HZ),
        "--out-json",
        str(out_json.resolve()),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not out_json.is_file():
        return int(proc.returncode), {
            "error": f"locator failed exit {proc.returncode}",
            "locator_log": str(log_path),
        }
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload["locator_log"] = str(log_path)
    payload["locator_exit_code"] = int(proc.returncode)
    return int(proc.returncode), payload


def targeted_harvest_from_locator(
    locator_hz: float,
    policy: Dict[str, Any],
) -> Dict[str, float]:
    half = float(policy.get("coupled_harvest_half_width_hz", 18.0))
    half = max(
        float(policy.get("coupled_harvest_half_width_min_hz", 12.0)),
        min(half, float(policy.get("coupled_harvest_half_width_max_hz", 25.0))),
    )
    loc = float(locator_hz)
    return {
        "harvest_lo_hz": loc - half,
        "harvest_hi_hz": loc + half,
        "target_hz": loc,
    }


def capture_branch_with_locator_then_coupled(
    sample: Dict[str, Any],
    mesh_path: Path,
    manifest: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Acoustic locator → one targeted coupled band → optional widen retries."""
    policy = dict(manifest.get("locator_policy") or {})
    sample_id = str(sample["id"])
    loc_log = SENS_ROOT / "samples" / sample_id / "logs" / "acoustic_locator.log"
    loc_rc, locator = run_acoustic_locator(sample, mesh_path, policy=policy, log_path=loc_log)
    locator_hz = float(locator.get("locator_frequency_hz", float("nan")))
    attempts_log: List[Dict[str, Any]] = []
    locator_meta = {
        "locator_frequency_hz": locator_hz,
        "locator_selection_method": locator.get("locator_selection_method"),
        "locator_band_hz": locator.get("locator_band_hz"),
        "locator_exit_code": loc_rc,
        "locator_log": locator.get("locator_log"),
    }
    if not math.isfinite(locator_hz):
        return (
            {"v2_converged": False, "error": "acoustic locator failed", **locator_meta},
            attempts_log,
            locator_meta,
        )
    targeted = targeted_harvest_from_locator(locator_hz, policy)
    locator_meta["coupled_target_hz"] = targeted["target_hz"]
    locator_meta["harvest_band_hz"] = [targeted["harvest_lo_hz"], targeted["harvest_hi_hz"]]
    log_path = SENS_ROOT / "samples" / sample_id / "logs" / "v2_solve_locator_targeted.log"
    rc, solve = run_mpi_solve(
        sample,
        mesh_path,
        target_hz=targeted["target_hz"],
        harvest_lo_hz=targeted["harvest_lo_hz"],
        harvest_hi_hz=targeted["harvest_hi_hz"],
        num_modes=int((manifest.get("default_branch_capture") or {}).get("initial_num_modes", DEFAULT_NUM_MODES)),
        log_path=log_path,
        select_by_energy=True,
    )
    branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
    ok = bool(solve.get("v2_converged")) and is_acoustic_branch(branch)
    attempts_log.append(
        {
            "label": "locator_targeted",
            "locator_frequency_hz": locator_hz,
            "harvest_band_hz": [targeted["harvest_lo_hz"], targeted["harvest_hi_hz"]],
            "target_hz": targeted["target_hz"],
            "solve_exit_code": rc,
            "v2_converged": bool(solve.get("v2_converged")),
            "acoustic_captured": ok,
            "targeted_retry_required": False,
            "branch_frequency_hz": float((branch or {}).get("frequency_hz", float("nan"))),
            "branch_p_frac_energy_phys": float(
                (branch or {}).get("p_frac_energy_phys", float("nan"))
            ),
        }
    )
    if ok:
        solve.update(locator_meta)
        solve["branch_capture_attempt"] = "locator_targeted"
        solve["branch_captured"] = True
        return solve, attempts_log, locator_meta

    bc_manifest = dict(manifest.get("default_branch_capture") or {})
    bc_manifest.update(sample.get("branch_capture") or {})
    for w in bc_manifest.get("widen_attempts") or []:
        label = str(w.get("label", "widen"))
        log_w = SENS_ROOT / "samples" / sample_id / "logs" / f"v2_solve_{label}.log"
        rc_w, solve_w = run_mpi_solve(
            sample,
            mesh_path,
            target_hz=float(w.get("target_hz", targeted["target_hz"])),
            harvest_lo_hz=float(w["harvest_lo_hz"]),
            harvest_hi_hz=float(w["harvest_hi_hz"]),
            num_modes=int(w.get("num_modes", WIDEN_NUM_MODES)),
            log_path=log_w,
            select_by_energy=True,
        )
        branch_w = solve_w.get("acoustic_branch_by_energy") or solve_w.get("nearest_acoustic_branch")
        ok_w = bool(solve_w.get("v2_converged")) and is_acoustic_branch(branch_w)
        attempts_log.append(
            {
                "label": label,
                "locator_frequency_hz": locator_hz,
                "harvest_band_hz": [w["harvest_lo_hz"], w["harvest_hi_hz"]],
                "target_hz": float(w.get("target_hz", targeted["target_hz"])),
                "solve_exit_code": rc_w,
                "v2_converged": bool(solve_w.get("v2_converged")),
                "acoustic_captured": ok_w,
                "targeted_retry_required": True,
                "branch_frequency_hz": float((branch_w or {}).get("frequency_hz", float("nan"))),
                "branch_p_frac_energy_phys": float(
                    (branch_w or {}).get("p_frac_energy_phys", float("nan"))
                ),
            }
        )
        if ok_w:
            solve_w.update(locator_meta)
            solve_w["branch_capture_attempt"] = label
            solve_w["branch_captured"] = True
            return solve_w, attempts_log, locator_meta
        solve = solve_w

    solve.update(locator_meta)
    solve["branch_capture_attempts"] = attempts_log
    solve["branch_captured"] = False
    return solve, attempts_log, locator_meta


def capture_branch_with_retries(
    sample: Dict[str, Any],
    mesh_path: Path,
    manifest: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (final_solve_payload, attempt_log)."""
    sample_id = str(sample["id"])
    attempts_log: List[Dict[str, Any]] = []
    last_solve: Dict[str, Any] = {}
    for attempt in branch_capture_plan(sample, manifest):
        label = str(attempt["label"])
        log_path = SENS_ROOT / "samples" / sample_id / "logs" / f"v2_solve_{label}.log"
        rc, solve = run_mpi_solve(
            sample,
            mesh_path,
            target_hz=float(attempt["target_hz"]),
            harvest_lo_hz=float(attempt["harvest_lo_hz"]),
            harvest_hi_hz=float(attempt["harvest_hi_hz"]),
            num_modes=int(attempt["num_modes"]),
            log_path=log_path,
            select_by_energy=True,
        )
        branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
        ok = bool(solve.get("v2_converged")) and is_acoustic_branch(branch)
        attempts_log.append(
            {
                "label": label,
                "harvest_band_hz": [attempt["harvest_lo_hz"], attempt["harvest_hi_hz"]],
                "target_hz": attempt["target_hz"],
                "solve_exit_code": rc,
                "v2_converged": bool(solve.get("v2_converged")),
                "acoustic_captured": ok,
                "branch_frequency_hz": float((branch or {}).get("frequency_hz", float("nan"))),
                "branch_p_frac_energy_phys": float(
                    (branch or {}).get("p_frac_energy_phys", float("nan"))
                ),
            }
        )
        last_solve = solve
        if ok:
            solve["branch_capture_attempt"] = label
            return solve, attempts_log
    last_solve["branch_capture_attempts"] = attempts_log
    return last_solve, attempts_log


def load_v2_mode_vector_dense(path: Path, n_expected: int) -> "np.ndarray":
    """Load ``*.smx.npz`` (or legacy ``.npy``) to a finite dense reduced vector."""
    import numpy as np
    from scipy import sparse

    from fem_mode_array_utils import load_mode_column_any

    col = load_mode_column_any(path)
    if not sparse.issparse(col):
        raise TypeError(f"Expected sparse CSR column in {path}")
    if col.shape[1] != 1:
        raise ValueError(f"Mode column must have shape (N, 1); got {col.shape} in {path}")
    dense = np.asarray(col.toarray(), dtype=np.float64).reshape(-1)
    n_exp = int(n_expected)
    if dense.size != n_exp:
        raise ValueError(
            f"Mode length {dense.size} != expected reduced length {n_exp} in {path}"
        )
    if not np.isfinite(dense).all():
        raise ValueError(f"Non-finite entries in mode vector {path}")
    if not np.any(np.abs(dense) > 0.0):
        raise ValueError(f"Zero mode vector {path}")
    return dense


def displacement_subspace_mac(
    vec_a: "np.ndarray",
    vec_b: "np.ndarray",
    u_to_W: "np.ndarray",
) -> float:
    import numpy as np

    u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    full_a = np.asarray(vec_a, dtype=np.float64).ravel()
    full_b = np.asarray(vec_b, dtype=np.float64).ravel()
    if full_a.size <= int(u_idx.max(initial=-1) + 1) or full_b.size <= int(u_idx.max(initial=-1) + 1):
        raise ValueError(
            f"u_to_W index out of range for vectors of length {full_a.size}/{full_b.size}"
        )
    ua = full_a[u_idx]
    ub = full_b[u_idx]
    mac = float(abs(np.vdot(ua, ub)) / (float(np.linalg.norm(ua)) * float(np.linalg.norm(ub))))
    if not math.isfinite(mac):
        raise ValueError("displacement MAC is non-finite")
    return mac


def best_sample_result_json(sample_id: str) -> Optional[Path]:
    results_dir = SENS_ROOT / "samples" / sample_id / "results"
    if not results_dir.is_dir():
        return None
    paths = sorted(results_dir.glob("result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def sample_has_completed_solve(sample_id: str) -> bool:
    result_path = best_sample_result_json(sample_id)
    if not result_path:
        return False
    solve = json.loads(result_path.read_text(encoding="utf-8"))
    if not solve.get("v2_converged"):
        return False
    branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
    if branch:
        return is_acoustic_branch(branch)
    in_band = solve.get("in_band_modes") or []
    acoustic = [m for m in in_band if is_acoustic_branch(m)]
    return bool(acoustic)


def load_saved_mesh_gates(sample_id: str, *, hole_radius_m: float = 0.047) -> Optional[Dict[str, Any]]:
    gates_dir = SENS_ROOT / "samples" / sample_id / "diagnostics" / "gates"
    summary_json = gates_dir / "mesh_gates_summary.json"
    if summary_json.is_file():
        return json.loads(summary_json.read_text(encoding="utf-8"))
    aperture_json = gates_dir / "soundhole_aperture_audit" / "soundhole_aperture_geometry_report.json"
    adjacency_json = gates_dir / "soundhole_air_audit" / "soundhole_air_adjacency_report.json"
    if not aperture_json.is_file() or not adjacency_json.is_file():
        return None
    aperture = json.loads(aperture_json.read_text(encoding="utf-8"))
    adjacency = json.loads(adjacency_json.read_text(encoding="utf-8"))
    fc = adjacency.get("facet_counts") or {}
    dof = adjacency.get("pressure_dof_audit") or {}
    combined = (
        bool(aperture.get("gate_pass"))
        and int(fc.get("tag2_adjacent_to_air_10", 0)) > 0
        and int(fc.get("tag2_adjacent_wood_only", 0)) == 0
        and int(dof.get("n_p_soundhole_in_air_subgraph", 0)) > 0
    )
    return {
        "mesh_file": str((SENS_ROOT / "mesh" / f"{sample_id}.msh")),
        "expected_hole_radius_m": float(hole_radius_m),
        "combined_mesh_gate_pass": bool(combined),
        "reloaded_from_saved_gate_audits": True,
        "aperture_summary": {
            "area_ratio_vs_pi_r2": aperture.get("area_ratio_vs_pi_r2"),
        },
        "tag2_adjacent_to_air_10": int(fc.get("tag2_adjacent_to_air_10", 0)),
        "n_p_soundhole_in_air_subgraph": int(dof.get("n_p_soundhole_in_air_subgraph", 0)),
    }


def load_phase2_reusable_rows(phase2_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Rows to reuse from partial/final summaries or rebuild from solve artifacts."""
    reusable: Dict[str, Dict[str, Any]] = {}
    for path in (
        DIAG_DIR / "v2_production_validation_summary.partial.json",
        PRODUCTION_SUMMARY_JSON,
    ):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for sid in phase2_ids:
            row = (data.get("samples") or {}).get(sid)
            if row and row.get("status") == "ok" and sid not in reusable:
                row = dict(row)
                row["reused_from_summary"] = str(path)
                reusable[sid] = row
    return reusable


def write_phase2_incremental(
    results: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    *,
    phase2_ids: List[str],
) -> None:
    phase1_ids = list(manifest.get("preserve_phase1_sample_ids") or [])
    summary = {
        "suite": manifest.get("suite"),
        "phase": manifest.get("phase"),
        "frozen_formulation": manifest.get("frozen_formulation"),
        "coupled_baseline": manifest.get("coupled_baseline"),
        "preserve_phase1_sample_ids": phase1_ids,
        "phase2_sample_ids": phase2_ids,
        "samples": {k: v for k, v in results.items() if not str(k).startswith("_")},
        "incremental": True,
    }
    write_json(DIAG_DIR / "v2_production_validation_summary.partial.json", summary)
    write_json(PRODUCTION_SUMMARY_JSON, summary)
    write_validation_status(
        {k: v for k, v in results.items() if k in phase1_ids},
        {k: v for k, v in results.items() if k in phase2_ids},
        production_manifest=manifest,
    )


def load_baseline_structural_reference() -> Dict[str, Any]:
    """Structural-dominated reference mode from frozen baseline coupled v2 artifacts."""
    import numpy as np

    case_dir = V2_ROOT / "physical_coupling_enabled"
    result_path = case_dir / "results" / f"result_{hz_result_tag(COUPLED_BASELINE_F_HZ)}.json"
    if not result_path.is_file():
        for p in sorted((case_dir / "results").glob("result_*.json"), reverse=True):
            result_path = p
            break
    if not result_path.is_file():
        return {}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    u_to_W = result.get("u_to_W")
    in_band = list(result.get("in_band_modes") or [])
    struct = [
        m
        for m in in_band
        if str(m.get("mode_class_physical_energy")) == "structural_dominated"
        or float(m.get("p_frac_energy_phys", 1.0)) <= 0.15
    ]
    if not struct:
        return {}
    ref = max(struct, key=lambda m: float(m.get("structural_modal_energy_phys", 0.0)))
    mode_path = case_dir / str(ref.get("vector_path", ""))
    if not mode_path.is_file():
        modes = sorted(case_dir.glob("modes/mode_*"))
        if not modes:
            return {"reference_frequency_hz": float(ref["frequency_hz"]), "vector_path": None}
        mode_path = modes[0]
    n_W = int(result.get("n_reduced_W", N_REDUCED_W_VALIDATION))
    try:
        ref_vec = load_v2_mode_vector_dense(mode_path, n_W)
    except Exception as exc:
        return {"unavailable_reason": f"baseline reference mode load failed: {exc}"}
    u_to_W = np.asarray(u_to_W, dtype=np.int32) if u_to_W is not None else None
    out = {
        "reference_frequency_hz": float(ref["frequency_hz"]),
        "reference_mode_index": int(ref.get("mode_index", -1)),
        "vector_path": str(mode_path),
        "p_frac_energy_phys": float(ref.get("p_frac_energy_phys", float("nan"))),
        "n_reduced_W": n_W,
    }
    if u_to_W is not None:
        out["u_to_W"] = u_to_W
        out["reference_vec"] = ref_vec
    return out


def structural_mac_against_baseline(
    solve: Dict[str, Any],
    sample_id: str,
    *,
    match_tol_hz: float = 8.0,
) -> Dict[str, Any]:
    import numpy as np

    ref = load_baseline_structural_reference()
    if ref.get("unavailable_reason"):
        return {
            "status": "structural_mac_unavailable",
            "reason": str(ref["unavailable_reason"]),
            "matches": [],
        }
    if ref.get("reference_vec") is None or ref.get("u_to_W") is None:
        return {
            "status": "structural_mac_unavailable",
            "reason": "baseline structural reference mode or u_to_W map missing",
            "matches": [],
        }
    u_to_W = np.asarray(ref["u_to_W"], dtype=np.int32)
    ref_vec = ref["reference_vec"]
    ref_f = float(ref["reference_frequency_hz"])
    n_W = int(solve.get("n_reduced_W", ref.get("n_reduced_W", N_REDUCED_W_VALIDATION)))
    u_map = solve.get("u_to_W")
    if u_map is not None:
        cand_u = np.asarray(u_map, dtype=np.int32)
        if cand_u.size != u_to_W.size or not np.array_equal(cand_u, u_to_W):
            return {
                "status": "structural_mac_unavailable",
                "reason": "sample u_to_W map differs from baseline (not same-mesh comparable)",
                "matches": [],
            }
    case_dir = SENS_ROOT / "samples" / sample_id
    rows: List[Dict[str, Any]] = []
    for m in solve.get("in_band_modes") or []:
        if is_acoustic_branch(m):
            continue
        f_hz = float(m["frequency_hz"])
        if abs(f_hz - ref_f) > match_tol_hz:
            continue
        rel = str(m.get("vector_path", ""))
        mode_path = case_dir / rel if rel else None
        if mode_path is None or not mode_path.is_file():
            continue
        try:
            vec = load_v2_mode_vector_dense(mode_path, n_W)
            mac = displacement_subspace_mac(ref_vec, vec, u_to_W)
        except Exception as exc:
            rows.append(
                {
                    "frequency_hz": f_hz,
                    "status": "structural_mac_unavailable",
                    "reason": str(exc),
                }
            )
            continue
        rows.append(
            {
                "frequency_hz": f_hz,
                "delta_f_hz_from_baseline_structural": f_hz - ref_f,
                "displacement_mac_vs_baseline_structural": mac,
                "p_frac_energy_phys": float(m.get("p_frac_energy_phys", float("nan"))),
                "structural_modal_energy_phys": float(
                    m.get("structural_modal_energy_phys", float("nan"))
                ),
            }
        )
    if not rows:
        return {
            "status": "structural_mac_unavailable",
            "reason": "no structural in-band modes matched baseline frequency window",
            "matches": [],
        }
    ok_rows = [r for r in rows if "displacement_mac_vs_baseline_structural" in r]
    return {
        "status": "ok" if ok_rows else "structural_mac_unavailable",
        "reason": None if ok_rows else "structural modes found but MAC could not be computed",
        "matches": rows,
    }


def row_from_existing_solve_artifacts(
    sample: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Rebuild summary row from saved solve results (no re-solve)."""
    from v2_sensitivity_mesh import sample_geometry, sample_mesh_path

    sample_id = str(sample["id"])
    geom = sample_geometry(sample)
    if not sample_has_completed_solve(sample_id):
        return None
    result_path = best_sample_result_json(sample_id)
    if not result_path:
        return None
    solve = json.loads(result_path.read_text(encoding="utf-8"))
    solve["sample_id"] = sample_id
    mesh_path = VALIDATION_MESH if sample.get("reuse_baseline_mesh") else sample_mesh_path(sample_id)
    if sample.get("requires_remesh"):
        mp = sample_mesh_path(sample_id)
        if mp.is_file():
            mesh_path = mp
    solve["mesh_file"] = str(mesh_path)
    if sample.get("reuse_baseline_mesh"):
        mesh_gates = {
            "combined_mesh_gate_pass": True,
            "reused_baseline_validation_mesh": True,
            "mesh_file": str(VALIDATION_MESH),
        }
    else:
        mesh_gates = load_saved_mesh_gates(
            sample_id, hole_radius_m=float(geom["hole_radius"])
        ) or {
            "combined_mesh_gate_pass": True,
            "reused_saved_gates_assumed": True,
            "mesh_file": str(mesh_path),
        }
    locator_meta: Dict[str, Any] = {}
    loc_path = SENS_ROOT / "samples" / sample_id / "diagnostics" / "acoustic_locator.json"
    if loc_path.is_file():
        locator_meta = json.loads(loc_path.read_text(encoding="utf-8"))
    attempts_log = list(solve.get("branch_capture_attempts") or [])
    if not attempts_log and locator_meta:
        attempts_log = [
            {
                "label": "locator_targeted",
                "locator_frequency_hz": locator_meta.get("locator_frequency_hz"),
                "harvest_band_hz": locator_meta.get("locator_band_hz"),
            }
        ]
    row = row_from_solve(
        sample,
        solve,
        mesh_gates=mesh_gates,
        attempts_log=attempts_log,
        locator_meta=locator_meta or None,
    )
    row["reused_solve_artifacts"] = True
    row["result_json"] = str(result_path)
    return row


def row_from_solve(
    sample: Dict[str, Any],
    solve: Dict[str, Any],
    *,
    mesh_gates: Dict[str, Any],
    attempts_log: List[Dict[str, Any]],
    locator_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from v2_sensitivity_mesh import sample_geometry

    geom = sample_geometry(sample)
    mats = sample.get("materials") or {}
    branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
    in_band = list(solve.get("in_band_modes") or [])
    acoustic_ok = is_acoustic_branch(branch)
    status = "ok" if acoustic_ok and solve.get("v2_converged") else (
        "acoustic_branch_not_captured" if solve.get("v2_converged") else "solve_failed"
    )
    varied: Dict[str, Any] = {
        k: geom[k]
        for k in ("hole_radius", "depth", "top_thickness", "length", "width")
        if k in geom
    }
    if mats.get("top_wood_id"):
        varied["top_wood_id"] = mats["top_wood_id"]
    if mats.get("back_wood_id"):
        varied["back_wood_id"] = mats["back_wood_id"]
    mo = sample.get("materials_override") or {}
    if (mo.get("top") or {}).get("E_L_scale") is not None:
        varied["top_E_L_scale"] = float(mo["top"]["E_L_scale"])
    row: Dict[str, Any] = {
        "sample_id": str(sample["id"]),
        "status": status,
        "requires_remesh": bool(sample.get("requires_remesh")),
        "reuse_baseline_mesh": bool(sample.get("reuse_baseline_mesh")),
        "varied_parameter_or_material": sample.get("expected_direction", {}).get("parameter"),
        "varied_parameters": varied,
        "geometry_values": {k: geom[k] for k in ("length", "width", "depth", "hole_radius", "top_thickness")},
        "material_assignment": dict(mats) if mats else None,
        "materials_override": mo,
        "expected_direction_or_interpretation": (sample.get("expected_direction") or {}).get(
            "interpretation"
        ),
        "mesh_file": solve.get("mesh_file"),
        "mesh_gates": mesh_gates,
        "v2_converged": bool(solve.get("v2_converged")),
        "branch_capture_attempts": attempts_log,
        "branch_capture_attempt": solve.get("branch_capture_attempt"),
        "branch_captured": solve.get("branch_captured", acoustic_ok),
        "locator_frequency_hz": (locator_meta or solve).get("locator_frequency_hz"),
        "coupled_target_hz": (locator_meta or solve).get("coupled_target_hz"),
        "harvest_band_hz": (locator_meta or solve).get("harvest_band_hz")
        or solve.get("harvest_band_hz"),
        "targeted_retry_required": any(
            a.get("targeted_retry_required") for a in attempts_log if "targeted_retry_required" in a
        ),
        "acoustic_branch_selection": solve.get("acoustic_branch_selection"),
        "structural_branches_in_band": structural_branches_summary(in_band),
        "in_band_modes_count": len(in_band),
    }
    if branch:
        row["nearest_acoustic_branch"] = branch
        row["acoustic_branch_by_energy"] = branch
        row["nearest_acoustic_f_hz"] = float(branch["frequency_hz"])
        row["delta_f_hz_from_coupled_baseline"] = float(branch["frequency_hz"]) - COUPLED_BASELINE_F_HZ
        row["p_frac_energy_phys"] = float(branch.get("p_frac_energy_phys", float("nan")))
        row["structural_modal_energy_phys"] = float(
            branch.get("structural_modal_energy_phys", float("nan"))
        )
        row["acoustic_modal_energy_phys"] = float(
            branch.get("acoustic_modal_energy_phys", float("nan"))
        )
        row["mass_cross_term_phys"] = float(branch.get("mass_cross_term_phys", float("nan")))
        row["mode_class_physical_energy"] = branch.get("mode_class_physical_energy")
    if sample.get("materials") and not sample.get("requires_remesh"):
        try:
            mac_payload = structural_mac_against_baseline(solve, str(sample["id"]))
        except Exception as exc:
            mac_payload = {
                "status": "structural_mac_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
                "matches": [],
            }
        row["structural_mac_status"] = mac_payload.get("status", "structural_mac_unavailable")
        row["structural_mac_reason"] = mac_payload.get("reason")
        row["structural_displacement_mac"] = mac_payload.get("matches", [])
    if status == "solve_failed":
        row["error"] = solve.get("error", "mpi solve failed")
    elif status == "acoustic_branch_not_captured":
        row["error"] = (
            "no acoustic_dominated branch in initial or widen harvest bands; "
            "do not use nearest-frequency structural mode"
        )
    return row


def load_phase1_preserved_results(production_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Preserve completed phase-1 acoustic/geometric rows from prior summary."""
    out: Dict[str, Dict[str, Any]] = {"baseline_coupled_v2": coupled_baseline_row()}
    preserve = list(production_manifest.get("preserve_phase1_sample_ids") or [])
    if not SUMMARY_JSON.is_file():
        return out
    prior = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    samples = prior.get("samples") or {}
    for sid in preserve:
        if sid == "baseline_coupled_v2":
            continue
        if sid in samples:
            row = dict(samples[sid])
            row["preserved_from_phase1"] = True
            out[sid] = row
    if prior.get("radius_trend_evaluation"):
        out["_radius_trend_evaluation"] = prior["radius_trend_evaluation"]
    return out


def write_validation_status(
    phase1_results: Dict[str, Dict[str, Any]],
    phase2_results: Dict[str, Dict[str, Any]],
    *,
    production_manifest: Dict[str, Any],
) -> None:
    exploratory = set(production_manifest.get("exploratory_not_production_gate") or [])
    phase2_ids = list(production_manifest.get("phase2_sample_ids") or [])
    material_ids = [s for s in phase2_ids if str(s).startswith("material_")]
    geometry_ids = [s for s in phase2_ids if str(s).startswith(("length_", "width_"))]
    material_pass = bool(material_ids) and all(
        (phase2_results.get(sid) or {}).get("status") == "ok" for sid in material_ids
    )
    geometry_pass = bool(geometry_ids) and all(
        (phase2_results.get(sid) or {}).get("status") == "ok" for sid in geometry_ids
    )
    acoustic_geo_ids = [
        s
        for s in production_manifest.get("preserve_phase1_sample_ids") or []
        if s not in ("baseline_coupled_v2",)
    ]
    acoustic_geo_pass = all(
        (phase1_results.get(sid) or {}).get("status") == "ok"
        for sid in acoustic_geo_ids
        if sid in phase1_results
    ) and bool(
        (phase1_results.get("_radius_trend_evaluation") or {}).get("pilot_radius_trend_pass")
    )
    status = {
        "coupled_physical_core_v2_baseline_validation": "PASS",
        "acoustic_geometric_validation_pass": acoustic_geo_pass,
        "material_species_validation_pass": "PASS" if material_pass else "Pending",
        "production_parameter_coverage_pass": "PASS" if geometry_pass else "Pending",
        "mesh_convergence_pass": "Pending",
        "lhs_promotion_blocked": True,
        "exploratory_not_production_gate": sorted(exploratory),
        "note": (
            "top_stiffness_soft/stiff (E_L×0.9/×1.1) are exploratory acoustic-branch stability "
            "evidence only; not the production material-validation gate. "
            "Production materials use full wood_library records (25 top×back combos deferred)."
        ),
        "confirmed_acoustic_geometric_results": {
            "baseline_f_hz": COUPLED_BASELINE_F_HZ,
            "baseline_p_frac": COUPLED_BASELINE_P_FRAC,
            "hole_radius_small": phase1_results.get("hole_radius_small"),
            "hole_radius_large": phase1_results.get("hole_radius_large"),
            "depth_small": phase1_results.get("depth_small"),
            "depth_large": phase1_results.get("depth_large"),
            "top_thickness_small": phase1_results.get("top_thickness_small"),
            "top_thickness_large": phase1_results.get("top_thickness_large"),
        },
        "phase2_pending_sample_ids": [
            sid for sid in phase2_ids if (phase2_results.get(sid) or {}).get("status") != "ok"
        ],
        "phase2_geometry_pass": geometry_pass,
        "phase2_material_pass": material_pass,
        "full_25_material_combinations": "deferred_after_controlled_species_and_mesh_convergence",
    }
    write_json(VALIDATION_STATUS_JSON, status)


def load_pilot_preserved_results(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Load passed radius-pilot rows from existing summary."""
    out: Dict[str, Dict[str, Any]] = {"baseline_coupled_v2": coupled_baseline_row(manifest)}
    if not SUMMARY_JSON.is_file():
        return out
    prior = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    samples = prior.get("samples") or {}
    for sid in ("hole_radius_small", "hole_radius_large"):
        if sid in samples and samples[sid].get("status") == "ok":
            row = dict(samples[sid])
            row["preserved_from_radius_pilot"] = True
            out[sid] = row
    if prior.get("radius_trend_evaluation"):
        out["_radius_trend_evaluation"] = prior["radius_trend_evaluation"]
    return out


def evaluate_expected_direction(
    sample: Dict[str, Any],
    result: Dict[str, Any],
    *,
    peer_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    exp = sample.get("expected_direction") or {}
    branch = result.get("nearest_acoustic_branch") or {}
    f_hz = float(branch.get("frequency_hz", float("nan")))
    param = str(exp.get("parameter", ""))
    out: Dict[str, Any] = {
        "parameter": param,
        "perturbation": exp.get("perturbation"),
        "expect": exp.get("expect"),
        "interpretation": exp.get("interpretation"),
        "delta_f_hz_from_coupled_baseline": (
            f_hz - COUPLED_BASELINE_F_HZ if math.isfinite(f_hz) else float("nan")
        ),
        "acoustic_branch_captured": is_acoustic_branch(branch),
        "recorded": math.isfinite(f_hz),
    }
    if param == "depth":
        f_small = float(
            (peer_results.get("depth_small", {}).get("nearest_acoustic_f_hz", float("nan")))
        )
        f_large = float(
            (peer_results.get("depth_large", {}).get("nearest_acoustic_f_hz", float("nan")))
        )
        if math.isfinite(f_small) and math.isfinite(f_large):
            out["depth_small_f_hz"] = f_small
            out["depth_large_f_hz"] = f_large
            out["depth_trend"] = "increasing" if f_large > f_small else "decreasing"
    if param == "top_thickness":
        f_thin = float(
            (peer_results.get("top_thickness_small", {}).get("nearest_acoustic_f_hz", float("nan")))
        )
        f_thick = float(
            (peer_results.get("top_thickness_large", {}).get("nearest_acoustic_f_hz", float("nan")))
        )
        out["top_thickness_small_f_hz"] = f_thin
        out["top_thickness_large_f_hz"] = f_thick
    if param == "top_plate_stiffness":
        f_soft = float(
            (peer_results.get("top_stiffness_soft", {}).get("nearest_acoustic_f_hz", float("nan")))
        )
        f_stiff = float(
            (peer_results.get("top_stiffness_stiff", {}).get("nearest_acoustic_f_hz", float("nan")))
        )
        out["top_stiffness_soft_f_hz"] = f_soft
        out["top_stiffness_stiff_f_hz"] = f_stiff
    return out


def write_suite_summary(
    results: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    *,
    controlled_suite: bool,
    pilot_mode: bool,
) -> None:
    pilot_ids = set(manifest.get("pilot_sample_ids") or [])
    controlled_ids = set(manifest.get("controlled_sample_ids") or [])
    radius_trend = results.pop("_radius_trend_evaluation", None)
    if radius_trend is None and all(
        sid in results for sid in ("hole_radius_small", "hole_radius_large")
    ):
        f_s = float(results["hole_radius_small"].get("nearest_acoustic_f_hz", float("nan")))
        f_l = float(results["hole_radius_large"].get("nearest_acoustic_f_hz", float("nan")))
        radius_trend = {
            "f_hole_radius_small_hz": f_s,
            "f_coupled_baseline_hz": COUPLED_BASELINE_F_HZ,
            "f_hole_radius_large_hz": f_l,
            "monotonic_increasing_with_radius": (
                math.isfinite(f_s)
                and math.isfinite(f_l)
                and f_s < COUPLED_BASELINE_F_HZ < f_l
            ),
            "pilot_radius_trend_pass": (
                math.isfinite(f_s)
                and math.isfinite(f_l)
                and f_s < COUPLED_BASELINE_F_HZ < f_l
            ),
        }

    def _sample_ok(sid: str) -> bool:
        r = results.get(sid) or {}
        if r.get("ingest_only"):
            return True
        return r.get("status") == "ok"

    pilot_pass = all(_sample_ok(s) for s in pilot_ids) if pilot_ids else None
    controlled_pass = all(_sample_ok(s) for s in controlled_ids) if controlled_ids else None
    suite_pass = (
        (pilot_pass is not False)
        and (controlled_pass is True if controlled_suite else True)
        and bool(radius_trend and radius_trend.get("pilot_radius_trend_pass"))
    )

    summary = {
        "suite": "v2_sensitivity_validation",
        "frozen_baseline": manifest.get("frozen_baseline"),
        "coupled_baseline_acoustic_f_hz": COUPLED_BASELINE_F_HZ,
        "coupled_baseline_p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
        "pilot_mode": pilot_mode,
        "controlled_suite": controlled_suite,
        "radius_pilot_passed": bool(radius_trend and radius_trend.get("pilot_radius_trend_pass")),
        "radius_trend_evaluation": radius_trend,
        "samples": results,
        "promotion": {
            "lhs_promotion_blocked_until_suite_pass": True,
            "lhs_promotion_blocked": True,
            "lhs_blocked": True,
            "mesh_convergence_blocked": True,
            "acoustic_geometric_validation_pass": suite_pass,
            "material_species_validation_pass": "Pending",
            "production_parameter_coverage_pass": "Pending",
            "mesh_convergence_pass": "Pending",
            "exploratory_not_production_gate": [
                "top_stiffness_soft",
                "top_stiffness_stiff",
            ],
            "pilot_all_gates_and_v2_pass": pilot_pass,
            "controlled_suite_pass": controlled_pass,
            "full_nonrandom_suite_pass": suite_pass,
            "note": "full_nonrandom_suite_pass reflects phase-1 acoustic/geometric only; not LHS promotion.",
        },
        "note": (
            "coupled_physical_core_v2 frozen; branch selection by physical energy; "
            "radius pilot recorded as first parametric validation."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    md = [
        "# v2 sensitivity validation summary",
        "",
        f"**Coupled baseline:** {COUPLED_BASELINE_F_HZ:.6f} Hz, "
        f"`p_frac_energy_phys` = {COUPLED_BASELINE_P_FRAC:.4f}",
        "",
        f"Radius pilot passed: `{summary['radius_pilot_passed']}`",
        f"Controlled suite pass: `{controlled_pass}`",
        f"Full non-random suite pass: `{suite_pass}`",
        "",
        "| sample | gates | v2 | f_acoustic | Δf coupled | p_frac | E_air | E_struct | class |",
        "|--------|-------|----:|-----------:|-----------:|-------:|------:|--------:|:------|",
    ]
    order = [
        "baseline_coupled_v2",
        "hole_radius_small",
        "hole_radius_large",
        "depth_small",
        "depth_large",
        "top_thickness_small",
        "top_thickness_large",
        "top_stiffness_soft",
        "top_stiffness_stiff",
    ]
    for sid in order:
        row = results.get(sid)
        if not row:
            continue
        gates = "—" if row.get("mesh_gates_skipped") else (
            "pass" if (row.get("mesh_gates") or {}).get("combined_mesh_gate_pass") else "FAIL"
        )
        v2 = "—" if row.get("ingest_only") else ("ok" if row.get("v2_converged") else "no")
        f_a = row.get("nearest_acoustic_f_hz", float("nan"))
        d_f = 0.0 if sid == "baseline_coupled_v2" else float(f_a) - COUPLED_BASELINE_F_HZ
        p_e = row.get("p_frac_energy_phys", float("nan"))
        e_a = row.get("acoustic_modal_energy_phys")
        e_s = row.get("structural_modal_energy_phys")
        e_a_s = f"{float(e_a):.2e}" if e_a is not None and math.isfinite(float(e_a)) else "—"
        e_s_s = f"{float(e_s):.2e}" if e_s is not None and math.isfinite(float(e_s)) else "—"
        cls = row.get("mode_class_physical_energy", row.get("status", "—"))
        md.append(
            f"| {sid} | {gates} | {v2} | {float(f_a):.3f} | {d_f:+.3f} | "
            f"{float(p_e):.4f} | {e_a_s} | {e_s_s} | {cls} |"
        )
    (DIAG_DIR / "v2_sensitivity_validation_summary.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
