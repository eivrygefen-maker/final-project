#!/usr/bin/env python3
"""M4.2 — full per-guitar LHS pipeline dry-run (plan only; will_execute=false)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
PIPELINE_RUNS = PHYSICS_ROOT / "pipeline_runs"
GUITARS_ROOT = PIPELINE_RUNS / "guitars"
SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")

DEFAULT_PROD_PYTHON = "/home/vboxuser/final-project/.venv/bin/python"
DEFAULT_PROD_VENV = "/home/vboxuser/final-project/.venv"
DEFAULT_SOLVER_PYTHON = "/home/vboxuser/solver-mkl/venv/bin/python"
DEFAULT_SOLVER_VENV = "/home/vboxuser/solver-mkl/venv"

MESH_CASE_BASE = "baseline_coupled_v2"
SCOUT_MESH_SCRIPT = f"{SCRIPTS_REL.as_posix()}/run_v2_B3_scout_coarse_mesh_build.py"
STAGE_A_SCRIPT = f"{SCRIPTS_REL.as_posix()}/v2_b3_checkpoint_export.py"
STAGE_B_SCRIPT = f"{SCRIPTS_REL.as_posix()}/v2_b3_checkpoint_target_density_experiment.py"
LPROD_MESH_SCRIPT = f"{SCRIPTS_REL.as_posix()}/run_v2_mesh_convergence.py"

PROD_ENV_VARS = {
    "PETSC_DIR": "/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real",
    "SLEPC_DIR": "/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real",
    "PYTHONPATH": (
        "$PETSC_DIR/lib/python3/dist-packages:"
        "$SLEPC_DIR/lib/python3/dist-packages:/usr/lib/python3/dist-packages"
    ),
}

ZONE_IDS = ("ZONE_1_dense", "ZONE_2_medium", "ZONE_3_sparse")
CHUNK_MIN_HZ = 15.0
CHUNK_MAX_HZ = 50.0
CHUNK_PREF_LO = 20.0
CHUNK_PREF_HI = 40.0
DENSITY_BIN_WIDTH_HZ = 25.0

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _detect_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur.parent != cur:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not detect repository root (missing .git ancestor)")


def _rel(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_sample(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: sample JSON must be an object")
    sample_id = str(data.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError(f"{path}: missing sample_id")
    return data


def _extract_geometry_material(sample: Mapping[str, Any]) -> Dict[str, Any]:
    params = sample.get("parameters")
    if isinstance(params, dict):
        return dict(params)
    out: Dict[str, Any] = {}
    if isinstance(sample.get("geometry"), dict):
        for k, v in sample["geometry"].items():
            out[f"geometry.{k}"] = v
    for k in ("top_wood_id", "back_wood_id"):
        if k in sample:
            out[k] = sample[k]
    return out


def _env_profiles(*, prod_python: str, solver_python: str) -> Dict[str, Any]:
    return {
        "production_fem": {
            "profile": "production_venv_strict",
            "python": prod_python,
            "virtual_env": DEFAULT_PROD_VENV,
            "used_for": ["stage0_resolve", "stage1_scout_mesh", "stage1_scout_export", "stage4_lprod_mesh", "stage4_lprod_export"],
            "env_vars": dict(PROD_ENV_VARS),
            "note": "Build explicit env dict at execution; do not inherit parent VIRTUAL_ENV.",
        },
        "solver_mkl": {
            "profile": "solver_mkl_strict",
            "python": solver_python,
            "virtual_env": DEFAULT_SOLVER_VENV,
            "used_for": ["stage2_scout_discovery", "stage5_lprod_workers"],
            "env_vars": {},
            "unset_at_execution": ["PYTHONPATH", "PETSC_DIR", "SLEPC_DIR", "PYTHONHOME"],
            "note": "Isolated solver-mkl; dolfinx must not import.",
        },
    }


def _cmd_scout_mesh_build(*, prod_python: str, sample_id: str) -> str:
    return (
        f"{prod_python} {SCOUT_MESH_SCRIPT} "
        f"# sample-specific geometry: {sample_id} (M4.3+ may pass FEM geometry overrides)"
    )


def _cmd_stage_a(
    *,
    prod_python: str,
    mesh_level: str,
    core_config: str,
    output_dir: str,
    synthesis_region_dofs: str = "off",
) -> str:
    return (
        f"{prod_python} {STAGE_A_SCRIPT} "
        f"--mesh-level {mesh_level} "
        "--B3-block-compose-backend csr_bulk "
        f"--B3-synthesis-region-dofs {synthesis_region_dofs} "
        f'--core-config "{core_config}" '
        f'--output-dir "{output_dir}"'
    )


def _cmd_stage_b_discovery(
    *,
    solver_python: str,
    checkpoint_dir: str,
    output_dir: str,
    freq_min: float,
    freq_max: float,
    spacing_hz: float,
    half_width_hz: float,
) -> str:
    return (
        f"{solver_python} {STAGE_B_SCRIPT} "
        f'--checkpoint-dir "{checkpoint_dir}" '
        f"--start-hz {freq_min} --stop-hz {freq_max} "
        f"--spacings-hz {spacing_hz} "
        f"--B3-discovery-mode --discovery-band-hz {freq_min} {freq_max} "
        f"--target-window-half-width-hz {half_width_hz} "
        f'--output-dir "{output_dir}"'
    )


def _cmd_lprod_mesh_placeholder(*, prod_python: str, sample_id: str) -> str:
    return (
        f"{prod_python} {LPROD_MESH_SCRIPT} "
        f"# planned L_prod build for sample {sample_id}; geometry-aware path TBD M4.3+"
    )


def _placeholder_density_bins(
    *,
    freq_min: float,
    freq_max: float,
    bin_width: float,
    zone_spacing: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bins: List[Dict[str, Any]] = []
    lo = freq_min
    while lo < freq_max - 1e-9:
        hi = min(freq_max, lo + bin_width)
        bw = hi - lo
        zone_id = "pending_scout"
        spacing = None
        bins.append(
            {
                "freq_lo_hz": round(lo, 6),
                "freq_hi_hz": round(hi, 6),
                "bin_width_hz": round(bw, 6),
                "mode_count": None,
                "density_modes_per_hz": None,
                "zone_id": zone_id,
                "recommended_lprod_spacing_hz": spacing,
            }
        )
        lo = hi
    segments = [
        {
            "freq_lo_hz": freq_min,
            "freq_hi_hz": freq_max,
            "zone_id": "pending_scout",
            "recommended_lprod_spacing_hz": None,
            "note": "Segments assigned after Stage 2 scout discovery (M4.3).",
        }
    ]
    return bins, segments


def _plan_placeholder_chunks(
    *,
    sample_id: str,
    run_id: str,
    freq_min: float,
    freq_max: float,
) -> List[Dict[str, Any]]:
    """Dry-run chunks over full band; targets pending scout."""
    chunks: List[Dict[str, Any]] = []
    lo = freq_min
    idx = 1
    width = CHUNK_PREF_HI
    while lo < freq_max - 1e-9:
        hi = min(freq_max, lo + width)
        span = hi - lo
        if span > CHUNK_MAX_HZ:
            hi = lo + CHUNK_MAX_HZ
        elif span < CHUNK_MIN_HZ and hi < freq_max:
            hi = min(freq_max, lo + CHUNK_MIN_HZ)
        chunk_id = f"{sample_id}_chunk_{idx:02d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "freq_range_hz": [round(lo, 6), round(hi, 6)],
                "zone_ids": ["pending_scout"],
                "targets_hz": [],
                "target_windows_hz": [],
                "target_count": 0,
                "estimated_cost": {
                    "target_count": 0,
                    "estimated_seconds": None,
                    "relative_weight": None,
                    "placeholder": True,
                },
                "status": "pending_target_plan",
                "assigned_worker_id": None,
                "priority": 0,
                "run_id": run_id,
            }
        )
        lo = hi
        idx += 1
        if idx > 64:
            break
    return chunks


def _stage_env_preview() -> Dict[str, Any]:
    return _env_profiles(prod_python=DEFAULT_PROD_PYTHON, solver_python=DEFAULT_SOLVER_PYTHON)


def build_dry_run_plan(
    *,
    repo_root: Path,
    sample: Dict[str, Any],
    run_id: str,
    freq_min: float,
    freq_max: float,
    scout_spacing_hz: float,
    scout_half_width_hz: float,
    zone_spacing_dense: float,
    zone_spacing_medium: float,
    zone_spacing_sparse: float,
    workers: int,
    prod_python: str,
    solver_python: str,
    mesh_profile: Optional[str] = None,
    dataset_version: Optional[str] = None,
) -> Dict[str, Any]:
    from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
        apply_mesh_profile_to_sample_input,
        resolve_mesh_profile,
    )

    sample_id = str(sample["sample_id"])
    resolved = resolve_mesh_profile(mesh_profile=mesh_profile, dataset_version=dataset_version)
    sample = apply_mesh_profile_to_sample_input(dict(sample), resolved)
    lprod_mesh_level = resolved.mesh_level_id
    run_root = GUITARS_ROOT / sample_id / "runs" / run_id
    rel_run = _rel(run_root, repo_root=repo_root)

    overlay_dir = PIPELINE_RUNS / "config_overlays" / sample_id
    resolved_core = overlay_dir / "resolved_core_config.json"
    readiness = overlay_dir / "readiness_check.json"

    sample_dir = run_root / "sample"
    scout_dir = run_root / "scout"
    lprod_dir = run_root / "lprod"
    worker_results_dir = run_root / "worker_results"
    aggregation_dir = run_root / "aggregation"
    logs_dir = run_root / "logs"

    scout_mesh = scout_dir / "mesh" / "L_scout_coarse" / f"{sample_id}.msh"
    scout_checkpoint = scout_dir / "checkpoint"
    scout_discovery = scout_dir / "discovery"
    lprod_mesh = lprod_dir / "mesh" / lprod_mesh_level / f"{sample_id}.msh"
    lprod_checkpoint = lprod_dir / "checkpoint"

    zone_spacing = {
        "ZONE_1_dense": zone_spacing_dense,
        "ZONE_2_medium": zone_spacing_medium,
        "ZONE_3_sparse": zone_spacing_sparse,
    }

    geom_mat = _extract_geometry_material(sample)
    core_config_planned = _rel(resolved_core, repo_root=repo_root)
    readiness_planned = _rel(readiness, repo_root=repo_root)

    cmd_scout_mesh = _cmd_scout_mesh_build(prod_python=prod_python, sample_id=sample_id)
    cmd_scout_a = _cmd_stage_a(
        prod_python=prod_python,
        mesh_level="L_scout_coarse",
        core_config=core_config_planned,
        output_dir=_rel(scout_checkpoint, repo_root=repo_root),
    )
    cmd_scout_b = _cmd_stage_b_discovery(
        solver_python=solver_python,
        checkpoint_dir=_rel(scout_checkpoint, repo_root=repo_root),
        output_dir=_rel(scout_discovery, repo_root=repo_root),
        freq_min=freq_min,
        freq_max=freq_max,
        spacing_hz=scout_spacing_hz,
        half_width_hz=scout_half_width_hz,
    )
    cmd_lprod_mesh = _cmd_lprod_mesh_placeholder(prod_python=prod_python, sample_id=sample_id)
    from v2_b3_m4_lprod_interfaces import LPROD_SYNTHESIS_REGION_DOFS_DEFAULT  # noqa: E402

    cmd_lprod_a = _cmd_stage_a(
        prod_python=prod_python,
        mesh_level=lprod_mesh_level,
        core_config=core_config_planned,
        output_dir=_rel(lprod_checkpoint, repo_root=repo_root),
        synthesis_region_dofs=LPROD_SYNTHESIS_REGION_DOFS_DEFAULT,
    )

    bins, segments = _placeholder_density_bins(
        freq_min=freq_min,
        freq_max=freq_max,
        bin_width=DENSITY_BIN_WIDTH_HZ,
        zone_spacing=zone_spacing,
    )
    chunks = _plan_placeholder_chunks(
        sample_id=sample_id,
        run_id=run_id,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    paths_tree = {
        "run_root": rel_run,
        "sample_dir": _rel(sample_dir, repo_root=repo_root),
        "scout_dir": _rel(scout_dir, repo_root=repo_root),
        "lprod_dir": _rel(lprod_dir, repo_root=repo_root),
        "worker_results_dir": _rel(worker_results_dir, repo_root=repo_root),
        "aggregation_dir": _rel(aggregation_dir, repo_root=repo_root),
        "logs_dir": _rel(logs_dir, repo_root=repo_root),
        "pipeline_run_manifest": f"{rel_run}/pipeline_run_manifest.json",
        "dry_run_summary": f"{rel_run}/dry_run_summary.md",
    }

    scout_plan = {
        "schema": "m4_scout_plan_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "mesh_level": "L_scout_coarse",
        "mesh_case_id": MESH_CASE_BASE,
        "sample_specific_mesh": True,
        "planned_mesh_path": _rel(scout_mesh, repo_root=repo_root),
        "checkpoint_dir": _rel(scout_checkpoint, repo_root=repo_root),
        "discovery_dir": _rel(scout_discovery, repo_root=repo_root),
        "scout_policy": {
            "version": "v1",
            "frequency_range_hz": [freq_min, freq_max],
            "spacing_hz": scout_spacing_hz,
            "half_width_hz": scout_half_width_hz,
            "discovery_mode": True,
        },
        "commands": {
            "mesh_build": cmd_scout_mesh,
            "stage_a_export": cmd_scout_a,
            "stage_b_discovery": cmd_scout_b,
        },
        "env_profile": "production_fem",
    }

    lprod_plan = {
        "schema": "m4_lprod_plan_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "mesh_level": lprod_mesh_level,
        "mesh_profile": resolved.mesh_profile,
        "dataset_version": resolved.dataset_version,
        "planned_mesh_path": _rel(lprod_mesh, repo_root=repo_root),
        "checkpoint_dir": _rel(lprod_checkpoint, repo_root=repo_root),
        "commands": {
            "mesh_build": cmd_lprod_mesh,
            "stage_a_export": cmd_lprod_a,
        },
        "env_profile": "production_fem",
        "worker_pool": {
            "planned_workers": workers,
            "assignment_policy": "fcfs_v1",
            "chunk_plan_path": f"{rel_run}/lprod/worker_chunk_plan.placeholder.json",
        },
    }

    density_zones_ph = {
        "schema": "m4_density_zones_v1",
        "will_execute": False,
        "status": "pending_scout",
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "zone_policy_version": "v1",
        "scout_policy_version": "v1",
        "frequency_range_hz": [freq_min, freq_max],
        "bin_width_hz": DENSITY_BIN_WIDTH_HZ,
        "zone_spacing_hz": zone_spacing,
        "classification_rule": "pending_scout — assign ZONE_1_dense / ZONE_2_medium / ZONE_3_sparse after discovery",
        "allowed_zone_ids": list(ZONE_IDS),
        "bins": bins,
        "segments": segments,
        "warnings": [
            "Placeholder only (M4.2 dry-run). mode_count and zone_id filled after Stage 2.",
        ],
    }

    lprod_target_ph = {
        "schema": "m4_lprod_target_plan_v1",
        "will_execute": False,
        "status": "pending_scout",
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "zone_policy_version": "v1",
        "target_generation_policy": "pending_scout_v1",
        "frequency_range_hz": [freq_min, freq_max],
        "mesh_level": lprod_mesh_level,
        "mesh_profile": resolved.mesh_profile,
        "targets_hz": [],
        "target_windows_hz": [],
        "target_metadata": [],
        "coverage_check": {
            "pass": False,
            "pending_scout": True,
            "band_hz": [freq_min, freq_max],
            "max_gap_hz": None,
            "gap_tolerance_hz": 0.01,
            "target_count": 0,
            "notes": "Gapless grid generated after density_zones from scout (M4.3).",
        },
        "estimated_runtime": {
            "target_count": 0,
            "estimated_seconds_per_target": None,
            "estimated_total_seconds": None,
            "placeholder": True,
        },
        "zone_spacing_policy_hz": zone_spacing,
        "warnings": [
            "No concrete targets until scout discovery completes.",
            f"Planned spacings when zoned: dense={zone_spacing_dense}, medium={zone_spacing_medium}, sparse={zone_spacing_sparse} Hz.",
        ],
    }

    worker_chunk_ph = {
        "schema": "m4_worker_chunk_plan_v1",
        "will_execute": False,
        "status": "pending_target_plan",
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "chunk_policy_version": "v1",
        "frequency_range_hz": [freq_min, freq_max],
        "chunk_policy": {
            "preferred_width_hz": [CHUNK_PREF_LO, CHUNK_PREF_HI],
            "min_width_hz": CHUNK_MIN_HZ,
            "max_width_hz": CHUNK_MAX_HZ,
            "respect_zone_boundaries": True,
        },
        "lprod_target_plan_path": f"{rel_run}/lprod/lprod_target_plan.placeholder.json",
        "chunks": chunks,
        "warnings": ["Chunk targets_hz empty until lprod_target_plan is finalized."],
    }

    scout_result_ph = {
        "schema": "m4_scout_result_v1",
        "will_execute": False,
        "status": "PLANNED",
        "sample_id": sample_id,
        "run_id": run_id,
        "scout_policy_version": "v1",
        "mesh_level": "L_scout_coarse",
        "mesh_path": _rel(scout_mesh, repo_root=repo_root),
        "checkpoint_dir": _rel(scout_checkpoint, repo_root=repo_root),
        "checkpoint_status": "PENDING",
        "discovery": {
            "frequency_range_hz": [freq_min, freq_max],
            "spacing_hz": scout_spacing_hz,
            "half_width_hz": scout_half_width_hz,
            "discovery_mode": True,
            "density_result_path": f"{_rel(scout_discovery, repo_root=repo_root)}/density_result.json",
            "unique_accepted_count": None,
            "experiment_status": "PENDING",
        },
        "artifacts": {
            "scout_density_report_json": f"{_rel(scout_dir, repo_root=repo_root)}/reports/scout_density_report.json",
            "density_result_json": f"{_rel(scout_discovery, repo_root=repo_root)}/density_result.json",
        },
    }

    resolved_manifest = {
        "schema": "m4_sample_manifest_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "status": "PLANNED",
        "resolved_core_config_path": core_config_planned,
        "readiness_check_path": readiness_planned,
        "overlay_applied_path": _rel(overlay_dir / "overlay_applied.json", repo_root=repo_root),
        "core_config_sha256": None,
        "mesh_paths": {
            "scout": {
                "mesh_level": "L_scout_coarse",
                "mesh_file": _rel(scout_mesh, repo_root=repo_root),
                "case_id": sample_id,
            },
            "lprod": {
                "mesh_level": lprod_mesh_level,
                "mesh_profile": resolved.mesh_profile,
                "mesh_file": _rel(lprod_mesh, repo_root=repo_root),
                "case_id": sample_id,
            },
        },
        "solver_policy": {"clamp_ribs": False},
        "geometry_and_material": geom_mat,
        "requires_mesh_regeneration": bool(sample.get("requires_mesh_regeneration", True)),
        "provenance": {
            "baseline_case_id": MESH_CASE_BASE,
            "shape_name": sample.get("shape_name"),
            "placeholder": True,
        },
        "warnings": ["Dry-run manifest; resolve overlay not executed (M4.3)."],
    }

    readiness_ph = {
        "schema": "m4_readiness_check_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "status": "PLANNED",
        "mesh_exists": {"L_scout_coarse": False, lprod_mesh_level: False},
        "core_config_readable": False,
        "notes": "Populated by Stage 0 resolve on execution (M4.3+).",
    }

    aggregation_ph = {
        "schema": "m4_aggregation_result_v1",
        "will_execute": False,
        "status": "PLANNED",
        "sample_id": sample_id,
        "run_id": run_id,
        "all_worker_results": [],
        "dedupe_tolerance_hz": 0.5,
        "unique_modes": [],
        "mode_catalog_path": _rel(
            aggregation_dir / "mode_catalog.json", repo_root=repo_root
        ),
        "modal_npz_path": _rel(aggregation_dir / "modal_modes.npz", repo_root=repo_root),
        "plots": {
            "mode_density_png": _rel(
                aggregation_dir / "mode_density_60_550.png", repo_root=repo_root
            ),
            "spectrum_png": _rel(aggregation_dir / "spectrum_overlay.png", repo_root=repo_root),
            "zone_overlay_png": _rel(
                aggregation_dir / "zone_overlay.png", repo_root=repo_root
            ),
        },
        "warnings": ["Placeholder aggregation (M4.2)."],
        "failures": [],
        "frequency_range_hz": [freq_min, freq_max],
    }

    runtime_ph = {
        "schema": "m4_runtime_summary_v1",
        "will_execute": False,
        "placeholder": True,
        "run_id": run_id,
        "sample_id": sample_id,
        "generated_utc": _utc_now(),
        "stages": {
            "stage0_resolve": {"wall_seconds": None, "status": "PLANNED"},
            "stage1_scout_mesh": {"wall_seconds": None, "status": "PLANNED"},
            "stage1_scout_export": {"wall_seconds": None, "status": "PLANNED"},
            "stage2_scout_discovery": {"wall_seconds": None, "status": "PLANNED"},
            "stage3_zones_plan": {"wall_seconds": None, "status": "PLANNED"},
            "stage4_lprod_mesh": {"wall_seconds": None, "status": "PLANNED"},
            "stage4_lprod_export": {"wall_seconds": None, "status": "PLANNED"},
            "stage5_workers": {"wall_seconds": None, "status": "PLANNED"},
            "stage6_aggregate": {"wall_seconds": None, "status": "PLANNED"},
        },
        "totals": {
            "wall_seconds": None,
            "target_count": 0,
            "unique_mode_count": None,
            "worker_count": workers,
        },
    }

    stages_manifest = {
        "stage0_resolve": {
            "status": "PLANNED",
            "artifact_paths": [
                f"{rel_run}/sample/sample_input.json",
                f"{rel_run}/sample/sample_resolved_config_manifest.json",
                f"{rel_run}/sample/readiness_check.json",
            ],
            "command_preview": f"# resolve overlay for {sample_id} (v2_b3_resolve_pilot_core_config.py pattern)",
        },
        "stage1_scout_mesh": {
            "status": "PLANNED",
            "artifact_paths": [_rel(scout_mesh, repo_root=repo_root)],
            "command_preview": cmd_scout_mesh,
        },
        "stage1_scout_export": {
            "status": "PLANNED",
            "artifact_paths": [_rel(scout_checkpoint, repo_root=repo_root)],
            "command_preview": cmd_scout_a,
        },
        "stage2_scout_discovery": {
            "status": "PLANNED",
            "artifact_paths": [_rel(scout_discovery / "density_result.json", repo_root=repo_root)],
            "command_preview": cmd_scout_b,
        },
        "stage3_zones_plan": {
            "status": "PLANNED",
            "artifact_paths": [
                f"{rel_run}/scout/density_zones.placeholder.json",
                f"{rel_run}/lprod/lprod_target_plan.placeholder.json",
                f"{rel_run}/lprod/worker_chunk_plan.placeholder.json",
            ],
        },
        "stage4_lprod_mesh": {
            "status": "PLANNED",
            "artifact_paths": [_rel(lprod_mesh, repo_root=repo_root)],
            "command_preview": cmd_lprod_mesh,
        },
        "stage4_lprod_export": {
            "status": "PLANNED",
            "artifact_paths": [_rel(lprod_checkpoint, repo_root=repo_root)],
            "command_preview": cmd_lprod_a,
        },
        "stage5_workers": {
            "status": "PLANNED",
            "artifact_paths": [_rel(worker_results_dir, repo_root=repo_root)],
            "command_preview": f"# FCFS workers W0..W{workers - 1} (M4.4)",
        },
        "stage6_aggregate": {
            "status": "PLANNED",
            "artifact_paths": [
                _rel(aggregation_dir / "aggregation_result.placeholder.json", repo_root=repo_root),
                _rel(aggregation_dir / "runtime_summary.placeholder.json", repo_root=repo_root),
            ],
        },
    }

    pipeline_manifest = {
        "schema": "m4_pipeline_run_manifest_v1",
        "sample_id": sample_id,
        "run_id": run_id,
        "mode": "dry_run",
        "will_execute": False,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "terminal_status": "PLANNED",
        "policy_versions": {
            "scout_policy_version": "v1",
            "zone_policy_version": "v1",
            "chunk_policy_version": "v1",
        },
        "frequency_policy": {
            "band_hz": [freq_min, freq_max],
            "scout_spacing_hz": scout_spacing_hz,
            "scout_half_width_hz": scout_half_width_hz,
            "zone_spacing_hz": zone_spacing,
        },
        "stages": stages_manifest,
        "output_tree": paths_tree,
        "command_previews": {
            "scout_mesh_build": cmd_scout_mesh,
            "scout_stage_a": cmd_scout_a,
            "scout_stage_b": cmd_scout_b,
            "lprod_mesh_build": cmd_lprod_mesh,
            "lprod_stage_a": cmd_lprod_a,
        },
        "environment_profiles": _stage_env_preview(),
        "no_execution_guarantee": True,
        "planned_workers": workers,
        "mesh_profile": resolved.mesh_profile,
        "mesh_level_id": lprod_mesh_level,
        "dataset_version": resolved.dataset_version,
        "provenance": {
            "core_config_sha256": None,
            "mesh_hashes": {"L_scout_coarse": None, lprod_mesh_level: None},
        },
    }

    return {
        "run_root": run_root,
        "rel_run": rel_run,
        "sample_id": sample_id,
        "lprod_mesh_level": lprod_mesh_level,
        "mesh_profile": resolved.mesh_profile,
        "files": {
            sample_dir / "sample_input.json": copy.deepcopy(sample),
            sample_dir / "sample_resolved_config_manifest.json": resolved_manifest,
            sample_dir / "readiness_check.json": readiness_ph,
            scout_dir / "scout_plan.json": scout_plan,
            scout_dir / "scout_result.placeholder.json": scout_result_ph,
            scout_dir / "density_zones.placeholder.json": density_zones_ph,
            lprod_dir / "lprod_plan.json": lprod_plan,
            lprod_dir / "lprod_target_plan.placeholder.json": lprod_target_ph,
            lprod_dir / "worker_chunk_plan.placeholder.json": worker_chunk_ph,
            aggregation_dir / "aggregation_result.placeholder.json": aggregation_ph,
            aggregation_dir / "runtime_summary.placeholder.json": runtime_ph,
            run_root / "pipeline_run_manifest.json": pipeline_manifest,
        },
        "readme_dirs": [worker_results_dir, logs_dir],
        "dry_run_summary": {
            "run_id": run_id,
            "sample_id": sample_id,
            "rel_run": rel_run,
            "chunk_count": len(chunks),
            "density_bin_count": len(bins),
            "workers": workers,
        },
    }


def _write_tree(plan: Dict[str, Any], *, force: bool) -> None:
    run_root: Path = plan["run_root"]
    if run_root.exists() and not force:
        raise FileExistsError(
            f"Run directory already exists (use --force to overwrite): {run_root}"
        )
    lprod_level = str(plan.get("lprod_mesh_level") or "L_prod_reference")
    for sub in (
        plan["run_root"] / "sample",
        plan["run_root"] / "scout" / "mesh" / "L_scout_coarse",
        plan["run_root"] / "scout" / "checkpoint",
        plan["run_root"] / "scout" / "discovery",
        plan["run_root"] / "lprod" / "mesh" / lprod_level,
        plan["run_root"] / "lprod" / "checkpoint",
        plan["run_root"] / "worker_results",
        plan["run_root"] / "aggregation",
        plan["run_root"] / "logs",
    ):
        sub.mkdir(parents=True, exist_ok=True)

    for path, payload in plan["files"].items():
        write_json_atomic(path, payload)

    readme_text = (
        "# M4 worker results (dry-run)\n\n"
        "Planned per-chunk JSON will appear here during M4.4 execution.\n"
        "`will_execute=false` — no solver runs in M4.2.\n"
    )
    logs_text = (
        "# M4 run logs (dry-run)\n\n"
        "Stage and worker logs will be written here on execution (M4.3+).\n"
        "Env probes are planned in manifest only for M4.2.\n"
    )
    for d in plan["readme_dirs"]:
        name = "README.md"
        p = d / name
        if force or not p.exists():
            p.write_text(readme_text if "worker" in d.name else logs_text, encoding="utf-8")


def _render_dry_run_summary(plan: Dict[str, Any], *, repo_root: Path) -> str:
    s = plan["dry_run_summary"]
    lines = [
        "# M4.2 pipeline dry-run summary",
        "",
        f"- **will_execute:** false",
        f"- **sample_id:** {s['sample_id']}",
        f"- **run_id:** {s['run_id']}",
        f"- **run_root:** `{s['rel_run']}/`",
        "",
        "## Policy",
        "",
        "- Scout: discovery on `L_scout_coarse`, sample-specific mesh path.",
        "- Zones / targets: `pending_scout` until Stage 2 completes (M4.3).",
        "- Workers: placeholder chunks only; status `pending_target_plan`.",
        "",
        "## Planned counts",
        "",
        f"- Density bins (placeholder): {s['density_bin_count']}",
        f"- Worker chunks (placeholder): {s['chunk_count']}",
        f"- Planned FCFS workers: {s['workers']}",
        "",
        "## Artifacts",
        "",
        "| Path | Role |",
        "|------|------|",
        "| `sample/sample_input.json` | Copied LHS input |",
        "| `sample/sample_resolved_config_manifest.json` | Stage 0 manifest stub |",
        "| `scout/scout_plan.json` | Stage 1–2 command plan |",
        "| `lprod/lprod_plan.json` | Stage 4–5 command plan |",
        "| `pipeline_run_manifest.json` | Terminal manifest stub |",
        "",
        "## Safety",
        "",
        "- No mesh build, Stage A/B/C, or worker execution.",
        "- Does not modify `v2_mesh_convergence/` outputs or legacy M2/M3 trees.",
        "",
        f"Generated: {_utc_now()}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4.2 full pipeline dry-run planner (no execution).")
    parser.add_argument("--sample-json", type=Path, required=True, help="Sample input JSON path.")
    parser.add_argument("--run-id", required=True, help="Run identifier under guitars/<sample_id>/runs/.")
    parser.add_argument("--freq-min-hz", type=float, default=60.0)
    parser.add_argument("--freq-max-hz", type=float, default=550.0)
    parser.add_argument("--scout-spacing-hz", type=float, default=7.5)
    parser.add_argument("--scout-half-width-hz", type=float, default=3.75)
    parser.add_argument("--zone-spacing-dense-hz", type=float, default=6.0)
    parser.add_argument("--zone-spacing-medium-hz", type=float, default=9.0)
    parser.add_argument("--zone-spacing-sparse-hz", type=float, default=12.5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--prod-python", default=DEFAULT_PROD_PYTHON)
    parser.add_argument("--solver-python", default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Planning only (default: true).",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Refused: this script is dry-run only.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing run directory.")
    parser.add_argument(
        "--mesh-profile",
        choices=("reference", "rom"),
        default=None,
        help="Production mesh profile (default: reference canonical).",
    )
    parser.add_argument("--dataset-version", default=None, help="Canonical dataset paired with --mesh-profile.")
    args = parser.parse_args(argv)

    if not args.dry_run:
        print("error: M4.2 dry-run script refuses --no-dry-run (execution not implemented)", file=sys.stderr)
        return 2

    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2

    repo_root = _detect_repo_root(SCRIPT_DIR)
    sample_path = args.sample_json if args.sample_json.is_absolute() else repo_root / args.sample_json
    if not sample_path.is_file():
        print(f"error: sample JSON not found: {sample_path}", file=sys.stderr)
        return 2

    sample = _load_sample(sample_path)
    if "schema" not in sample:
        sample = dict(sample)
        sample["schema"] = "m4_sample_input_v1"

    plan = build_dry_run_plan(
        repo_root=repo_root,
        sample=sample,
        run_id=str(args.run_id).strip(),
        freq_min=float(args.freq_min_hz),
        freq_max=float(args.freq_max_hz),
        scout_spacing_hz=float(args.scout_spacing_hz),
        scout_half_width_hz=float(args.scout_half_width_hz),
        zone_spacing_dense=float(args.zone_spacing_dense_hz),
        zone_spacing_medium=float(args.zone_spacing_medium_hz),
        zone_spacing_sparse=float(args.zone_spacing_sparse_hz),
        workers=int(args.workers),
        prod_python=str(args.prod_python),
        solver_python=str(args.solver_python),
        mesh_profile=args.mesh_profile,
        dataset_version=args.dataset_version,
    )

    try:
        _write_tree(plan, force=bool(args.force))
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary_path = plan["run_root"] / "dry_run_summary.md"
    summary_path.write_text(_render_dry_run_summary(plan, repo_root=repo_root), encoding="utf-8")

    print("will_execute=false")
    print(f"sample_id={plan['sample_id']}")
    print(f"run_id={args.run_id}")
    print("created planned run tree")
    print("wrote pipeline_run_manifest.json")
    print("wrote dry_run_summary.md")
    print("no execution performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
