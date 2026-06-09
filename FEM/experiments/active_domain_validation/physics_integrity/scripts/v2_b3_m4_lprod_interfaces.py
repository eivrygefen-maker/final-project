"""M4.4.1a — shared L_prod execution interfaces (mesh readiness, chunk targets, worker placeholders)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")
V2_MESH_ROOT = PHYSICS_ROOT / "v2_mesh_convergence"

CHUNK_TARGETS_SCHEMA = "m4_worker_chunk_targets_v1"
LPROD_MESH_LEVEL = "L_prod"
# M4 production default: lightweight region_dof_indices.npz for top/back/air participation (not Stage C).
LPROD_SYNTHESIS_REGION_DOFS_DEFAULT = "best_effort"
BASELINE_CASE_ID = "baseline_coupled_v2"
BASELINE_L_PROD_MESH = V2_MESH_ROOT / "mesh" / LPROD_MESH_LEVEL / f"{BASELINE_CASE_ID}.msh"

GEOMETRY_FINGERPRINT_KEYS = (
    "length",
    "width",
    "depth",
    "hole_radius",
    "top_thickness",
    "back_thickness",
)

# Numeric FEM body dimensions only (meters). Metadata must not appear here.
GEOMETRY_NUMERIC_KEYS = frozenset(GEOMETRY_FINGERPRINT_KEYS)

GEOMETRY_METADATA_KEYS = frozenset(
    {
        "shape_type",
        "mesh_mode",
        "shape_name",
        "dataset_version",
    }
)


class GeometryNumericCoercionError(ValueError):
    """Raised when a geometry key expected to be numeric cannot be coerced."""

    def __init__(self, key: str, value: Any) -> None:
        self.key = key
        self.value = value
        super().__init__(f"geometry numeric key {key!r} cannot be coerced to float: {value!r}")


def coerce_geometry_numeric(key: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryNumericCoercionError(key, value) from exc

# Canonical baseline geometry (classic coupled v2 reference body).
BASELINE_GEOMETRY: Dict[str, float] = {
    "length": 0.48,
    "width": 0.325,
    "depth": 0.1,
    "hole_radius": 0.047,
    "top_thickness": 0.003,
    "back_thickness": 0.003,
}

DEFAULT_DISCOVERY_BAND_HZ = (60.0, 550.0)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def extract_run_metadata(sample_or_params: Mapping[str, Any]) -> Dict[str, str]:
    """Non-numeric run metadata (shape, dataset marker) kept separate from body dimensions."""
    out: Dict[str, str] = {}
    for top_key in ("shape_name", "dataset_version"):
        if top_key in sample_or_params and sample_or_params[top_key] is not None:
            out[top_key] = str(sample_or_params[top_key])
    meta = sample_or_params.get("m4_run_metadata")
    if isinstance(meta, dict):
        for key in ("shape_name", "dataset_version", "shape_type", "mesh_mode"):
            if key in meta and meta[key] is not None:
                out[key] = str(meta[key])
    geom = sample_or_params.get("geometry")
    if isinstance(geom, dict):
        for key in GEOMETRY_METADATA_KEYS:
            if key in geom and geom[key] is not None:
                out.setdefault(key, str(geom[key]))
    return out


def extract_geometry_dict(sample_or_params: Mapping[str, Any]) -> Dict[str, float]:
    """Normalize numeric body geometry from parameters, geometry block, or geometry_numeric_parameters."""
    out: Dict[str, float] = {}

    numeric_block = sample_or_params.get("geometry_numeric_parameters")
    if isinstance(numeric_block, dict):
        for raw_key, raw_val in numeric_block.items():
            key = str(raw_key)
            if key in GEOMETRY_METADATA_KEYS:
                continue
            if key not in GEOMETRY_NUMERIC_KEYS:
                raise GeometryNumericCoercionError(key, raw_val)
            out[key] = coerce_geometry_numeric(key, raw_val)

    params = sample_or_params.get("parameters")
    if isinstance(params, dict):
        for k, v in params.items():
            ks = str(k)
            if not ks.startswith("geometry."):
                continue
            key = ks.split(".", 1)[1]
            if key in GEOMETRY_METADATA_KEYS:
                continue
            if key not in GEOMETRY_NUMERIC_KEYS:
                raise GeometryNumericCoercionError(ks, v)
            out[key] = coerce_geometry_numeric(key, v)

    geom = sample_or_params.get("geometry")
    if isinstance(geom, dict):
        for raw_key, raw_val in geom.items():
            key = str(raw_key)
            if key in GEOMETRY_METADATA_KEYS:
                continue
            if key not in GEOMETRY_NUMERIC_KEYS:
                continue
            out[key] = coerce_geometry_numeric(key, raw_val)

    return out


def geometry_fingerprint(geometry: Mapping[str, float], *, rel_tol: float = 1.0e-6) -> str:
    """Stable hash for geometry compatibility checks."""
    normalized: Dict[str, str] = {}
    for key in GEOMETRY_FINGERPRINT_KEYS:
        if key not in geometry:
            continue
        v = float(geometry[key])
        normalized[key] = f"{v:.12g}"
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def geometries_match(
    sample_geometry: Mapping[str, float],
    baseline_geometry: Mapping[str, float],
    *,
    abs_tol: float = 1.0e-9,
) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    ok = True
    for key in GEOMETRY_FINGERPRINT_KEYS:
        if key not in sample_geometry and key not in baseline_geometry:
            continue
        if key not in sample_geometry:
            notes.append(f"missing sample geometry key: {key}")
            ok = False
            continue
        if key not in baseline_geometry:
            notes.append(f"missing baseline geometry key: {key}")
            ok = False
            continue
        a = float(sample_geometry[key])
        b = float(baseline_geometry[key])
        if abs(a - b) > abs_tol:
            notes.append(f"{key}: sample={a} baseline={b}")
            ok = False
    return ok, notes


def evaluate_lprod_mesh_checkpoint_readiness(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    sample_input: Mapping[str, Any],
    rel_path_fn,
    mesh_level_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Decide L_prod mesh/checkpoint reuse vs planned build (M4.4.1a dry-run)."""
    from v2_b3_m4_mesh_profile_lib import (  # noqa: WPS433
        LEVEL_L_PROD_LEGACY,
        resolve_mesh_profile_from_mapping,
        run_tree_lprod_mesh_path,
    )

    profile = resolve_mesh_profile_from_mapping(sample_input)
    level = mesh_level_id or profile.mesh_level_id
    lprod_dir = run_root / "lprod"
    sample_mesh = run_tree_lprod_mesh_path(run_root, sample_id, level)
    if not sample_mesh.is_file() and profile.mesh_profile == "reference":
        legacy = lprod_dir / "mesh" / LEVEL_L_PROD_LEGACY / f"{sample_id}.msh"
        if legacy.is_file():
            sample_mesh = legacy
    checkpoint_dir = lprod_dir / "checkpoint"
    resolved_lprod = lprod_dir / "resolved_core_config.json"
    resolved_sample = run_root / "sample" / "resolved_core_config.json"

    sample_geom = extract_geometry_dict(sample_input)
    if not sample_geom and resolved_sample.is_file():
        sample_geom = extract_geometry_dict(_load_json(resolved_sample))
    sample_fp = geometry_fingerprint(sample_geom) if sample_geom else None
    baseline_fp = geometry_fingerprint(BASELINE_GEOMETRY)
    geom_match, geom_notes = geometries_match(sample_geom, BASELINE_GEOMETRY) if sample_geom else (False, ["no geometry in sample"])

    requires_remesh = bool(sample_input.get("requires_mesh_regeneration"))
    baseline_mesh_exists = BASELINE_L_PROD_MESH.is_file()
    sample_mesh_exists = sample_mesh.is_file()

    export_manifest = checkpoint_dir / "checkpoint_export_manifest.json"
    checkpoint_pass = False
    if export_manifest.is_file():
        try:
            manifest = _load_json(export_manifest)
            checkpoint_pass = str(manifest.get("status") or "").upper() == "PASS"
        except (OSError, ValueError, json.JSONDecodeError):
            checkpoint_pass = False

    if (
        profile.allow_baseline_mesh_reuse
        and geom_match
        and baseline_mesh_exists
        and not requires_remesh
    ):
        lprod_mesh_status = "reusable_existing"
        mesh_source = rel_path_fn(BASELINE_L_PROD_MESH, repo_root=repo_root)
        mesh_note = (
            "Geometry fingerprint matches baseline_coupled_v2; "
            f"reuse {BASELINE_CASE_ID} L_prod mesh after explicit compatibility check."
        )
    elif sample_mesh_exists:
        lprod_mesh_status = "reusable_existing"
        mesh_source = rel_path_fn(sample_mesh, repo_root=repo_root)
        mesh_note = "Per-sample L_prod mesh file already present on disk."
    else:
        lprod_mesh_status = "planned_build_required"
        mesh_source = rel_path_fn(sample_mesh, repo_root=repo_root)
        mesh_note = (
            "Geometry differs from baseline or requires_mesh_regeneration=true; "
            "plan sample-specific L_prod mesh build (no execution in M4.4.1a)."
        )

    if checkpoint_pass:
        lprod_checkpoint_status = "existing_pass"
    elif checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
        lprod_checkpoint_status = "blocked"
    else:
        lprod_checkpoint_status = "planned"

    cmd_stage_a = (
        f"# production .venv\n"
        f"python {SCRIPTS_REL.as_posix()}/v2_b3_checkpoint_export.py "
        f"--mesh-level {level} "
        "--B3-block-compose-backend csr_bulk "
        f"--B3-synthesis-region-dofs {LPROD_SYNTHESIS_REGION_DOFS_DEFAULT} "
        f'--core-config "{rel_path_fn(resolved_lprod if resolved_lprod.is_file() else resolved_sample, repo_root=repo_root)}" '
        f'--output-dir "{rel_path_fn(checkpoint_dir, repo_root=repo_root)}"'
    )
    cmd_mesh = (
        f"# production .venv — sample-specific production mesh when geometry != baseline\n"
        f"python {SCRIPTS_REL.as_posix()}/v2_b3_m4_lprod_mesh_build.py "
        f"# planned for {sample_id}; mesh_level_id={level}"
    )

    return {
        "schema": "m4_lprod_mesh_checkpoint_readiness_v1",
        "will_execute": False,
        "sample_id": sample_id,
        **profile.provenance_fields(),
        "lprod_mesh_status": lprod_mesh_status,
        "lprod_checkpoint_status": lprod_checkpoint_status,
        "geometry_compatibility": {
            "baseline_case_id": BASELINE_CASE_ID,
            "geometry_fingerprint_sample": sample_fp,
            "geometry_fingerprint_baseline": baseline_fp,
            "geometry_hash_match": geom_match,
            "requires_mesh_regeneration": requires_remesh,
            "mismatch_notes": geom_notes,
        },
        "paths": {
            "sample_mesh_path": rel_path_fn(sample_mesh, repo_root=repo_root),
            "baseline_mesh_path": rel_path_fn(BASELINE_L_PROD_MESH, repo_root=repo_root),
            "baseline_mesh_exists": baseline_mesh_exists,
            "sample_mesh_exists": sample_mesh_exists,
            "mesh_source_recommended": mesh_source,
            "checkpoint_dir": rel_path_fn(checkpoint_dir, repo_root=repo_root),
            "resolved_core_config_lprod": rel_path_fn(resolved_lprod, repo_root=repo_root),
            "resolved_core_config_sample": rel_path_fn(resolved_sample, repo_root=repo_root),
        },
        "commands": {
            "mesh_build_planned": cmd_mesh,
            "stage_a_export_planned": cmd_stage_a,
        },
        "notes": mesh_note,
    }


def build_chunk_targets_payload(
    *,
    sample_id: str,
    run_id: str,
    chunk: Mapping[str, Any],
    target_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build m4_worker_chunk_targets_v1 from chunk plan + lprod_target_plan metadata."""
    chunk_id = str(chunk.get("chunk_id"))
    targets_hz = [float(t) for t in (chunk.get("targets_hz") or [])]
    windows_raw = chunk.get("target_windows_hz") or []
    meta_by_target = {
        float(m.get("target_hz")): m
        for m in (target_plan.get("target_metadata") or [])
        if m.get("target_hz") is not None
    }
    plan_windows = target_plan.get("target_windows_hz") or []
    plan_targets = [float(t) for t in (target_plan.get("targets_hz") or [])]
    window_by_target = {
        plan_targets[i]: list(plan_windows[i])
        for i in range(min(len(plan_targets), len(plan_windows)))
    }

    targets_out: List[Dict[str, Any]] = []
    for i, thz in enumerate(targets_hz):
        win = None
        if i < len(windows_raw):
            win = list(windows_raw[i])
        if win is None or len(win) != 2:
            win = window_by_target.get(thz)
        if win is None:
            meta = meta_by_target.get(thz, {})
            hw = meta.get("half_width_hz")
            if hw is not None:
                win = [round(thz - float(hw), 6), round(thz + float(hw), 6)]
        if win is None:
            raise ValueError(f"{chunk_id}: missing window for target_hz={thz}")
        meta = meta_by_target.get(thz, {})
        targets_out.append(
            {
                "target_hz": thz,
                "window_hz": [float(win[0]), float(win[1])],
                "zone_id": str(meta.get("zone_id") or "unknown"),
                "spacing_hz": float(meta.get("spacing_hz") or 0.0),
                "source": "adaptive_lprod_target_plan",
            }
        )

    return {
        "schema": CHUNK_TARGETS_SCHEMA,
        "sample_id": sample_id,
        "run_id": run_id,
        "chunk_id": chunk_id,
        "freq_range_hz": list(chunk.get("freq_range_hz") or []),
        "targets": targets_out,
    }


def validate_chunk_targets_doc(doc: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if doc.get("schema") != CHUNK_TARGETS_SCHEMA:
        errors.append(f"schema={doc.get('schema')!r} expected {CHUNK_TARGETS_SCHEMA!r}")
    if not doc.get("chunk_id"):
        errors.append("missing chunk_id")
    targets = doc.get("targets") or []
    if not targets:
        errors.append("empty targets")
    for i, t in enumerate(targets):
        if t.get("target_hz") is None:
            errors.append(f"targets[{i}]: missing target_hz")
        win = t.get("window_hz")
        if not win or len(win) != 2:
            errors.append(f"targets[{i}]: invalid window_hz")
    return errors


def acceptance_config_from_chunk_targets(
    doc: Mapping[str, Any],
    *,
    discovery_band_hz: Tuple[float, float] = DEFAULT_DISCOVERY_BAND_HZ,
) -> Any:
    from v2_b3_st_sinvert_solver_lib import ACCEPTANCE_POLICY_DISCOVERY, AcceptanceConfig

    per_target: Dict[float, Tuple[float, float]] = {}
    for t in doc.get("targets") or []:
        hz = float(t["target_hz"])
        win = t.get("window_hz") or []
        per_target[hz] = (float(win[0]), float(win[1]))
    return AcceptanceConfig(
        policy=ACCEPTANCE_POLICY_DISCOVERY,
        discovery_band_hz=discovery_band_hz,
        target_window_half_width_hz=6.25,
        per_target_windows_hz=per_target,
    )


def build_worker_command_line(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    chunk_targets_path: Path,
    output_dir: Path,
    solver_python: str,
) -> str:
    script = repo_root / SCRIPTS_REL / "v2_b3_checkpoint_solve_target_list.py"
    from v2_b3_resolve_pilot_core_config import _repo_relative

    return (
        f"{solver_python} {_repo_relative(script, repo_root=repo_root)} "
        f'--checkpoint-dir "{_repo_relative(checkpoint_dir, repo_root=repo_root)}" '
        f'--targets-json "{_repo_relative(chunk_targets_path, repo_root=repo_root)}" '
        f"--factor-solver mkl_pardiso "
        f'--output-dir "{_repo_relative(output_dir, repo_root=repo_root)}"'
    )


def build_worker_result_placeholder(
    *,
    chunk_id: str,
    worker_id: Optional[str],
    chunk_targets: Mapping[str, Any],
    output_dir: Path,
    mode: str = "m4_4_1a_dry_run",
) -> Dict[str, Any]:
    n = len(chunk_targets.get("targets") or [])
    solver_rel = "solver_result.json"
    return {
        "schema": "m4_worker_result_v1",
        "will_execute": False,
        "mode": mode,
        "chunk_id": chunk_id,
        "worker_id": worker_id,
        "status": "DRY_RUN_PLANNED",
        "targets_attempted": n,
        "targets_passed": 0,
        "accepted_modes": [],
        "unique_modes": [],
        "timing": {
            "wall_seconds": 0.0,
            "setup_seconds": 0.0,
            "solve_seconds": 0.0,
            "per_target_seconds": None,
        },
        "warnings": ["M4.4.1a dry-run placeholder — no solver execution."],
        "errors": [],
        "solver_result_json": str(output_dir / solver_rel),
        "generated_utc": _utc_now(),
    }


def build_solver_result_placeholder(
    *,
    chunk_targets: Mapping[str, Any],
    checkpoint_dir: Path,
    factor_solver: str,
    mode: str = "m4_4_1a_dry_run",
) -> Dict[str, Any]:
    targets_hz = [float(t["target_hz"]) for t in (chunk_targets.get("targets") or [])]
    return {
        "schema": "m4_solver_result_v1",
        "will_execute": False,
        "mode": mode,
        "status": "DRY_RUN_PLANNED",
        "benchmark_kind": "checkpoint_solve_target_list",
        "checkpoint_dir": str(checkpoint_dir),
        "factor_solver": factor_solver,
        "targets_hz": targets_hz,
        "targets": [],
        "aggregate": {
            "targets_attempted": len(targets_hz),
            "targets_succeeded": 0,
            "unique_accepted_frequencies_hz": [],
        },
        "warnings": ["M4.4.1a dry-run placeholder — no ST solve."],
        "generated_utc": _utc_now(),
    }
