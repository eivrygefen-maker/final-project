#!/usr/bin/env python3
"""M4.3 — single-guitar scout pipeline (Stages 0–3). No L_prod execution."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
PIPELINE_RUNS = PHYSICS_ROOT / "pipeline_runs"
GUITARS_ROOT = PIPELINE_RUNS / "guitars"
BASELINE_CORE = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
DEFAULT_REFERENCE = PIPELINE_RUNS / "specs" / "scout_discovery_reference_stub.json"
CONV_MESH = PHYSICS_ROOT / "v2_mesh_convergence" / "mesh"
SCOUT_MESH_SCRIPT = (
    "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
    "v2_b3_m4_scout_mesh_build.py"
)
STAGE_A_REL = (
    "FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py"
)
STAGE_B_REL = (
    "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
    "v2_b3_checkpoint_target_density_experiment.py"
)
MESH_LEVEL = "L_scout_coarse"
MESH_CASE_BASE = "baseline_coupled_v2"
BASELINE_SCOUT_MESH = CONV_MESH / MESH_LEVEL / f"{MESH_CASE_BASE}.msh"
BIN_WIDTH_HZ = 25.0

DEFAULT_PROD_PYTHON = "/home/vboxuser/final-project/.venv/bin/python"
DEFAULT_PROD_VENV = "/home/vboxuser/final-project/.venv"
DEFAULT_SOLVER_PYTHON = "/home/vboxuser/solver-mkl/venv/bin/python"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m3_orchestrator_run_one import (  # noqa: E402
    DEFAULT_PROD_PYTHON as M3_PROD_PYTHON,
    DEFAULT_SOLVER_PYTHON as M3_SOLVER_PYTHON,
    _run_subprocess,
    _verify_stage_a_export,
)
from v2_b3_m4_lprod_interfaces import (  # noqa: E402
    BASELINE_GEOMETRY,
    extract_geometry_dict,
    geometries_match,
    geometry_fingerprint,
)
from v2_b3_m4_scout_planner_lib import (  # noqa: E402
    ZONE_1,
    ZONE_2,
    ZONE_3,
    build_density_bins,
    build_density_zones_document,
    build_gapless_target_plan,
    build_worker_chunk_preview,
    classify_bins_percentile,
    estimate_runtime_summary,
    merge_zone_segments,
    render_chunk_preview_md,
    render_density_zones_md,
    render_target_plan_md,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_resolve_pilot_core_config import (  # noqa: E402
    _build_changed_material_values,
    _repo_relative,
    _sha256_file,
)
from v2_b3_run_coarse_scout_lhs_batch import (  # noqa: E402
    DEFAULT_PROD_VENV,
    STAGE_A_ENV_PROBE,
    STAGE_B_ENV_PROBE,
    _dedupe_frequencies_hz,
    _extract_unique_frequencies,
    _path_for_subprocess,
    _prod_subprocess_env_strict,
    _run_env_probe,
    _run_stage_env_probes,
    _solver_mkl_subprocess_env_strict,
    _verify_density_result,
    _verify_stage_a_env_probe,
    _verify_stage_b_env_probe,
)
from v2_b3_m4_production_contracts import DATASET_VERSION, is_strict_production_mode  # noqa: E402


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
    return _repo_relative(path, repo_root=repo_root)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def _wood_library_apply(config: Dict[str, Any], parameters: Dict[str, Any], *, repo_root: Path) -> None:
    fem_scripts = repo_root / "FEM" / "scripts"
    if str(fem_scripts) not in sys.path:
        sys.path.insert(0, str(fem_scripts))
    from wood_library import apply_lhs_parameters_to_config  # noqa: WPS433

    apply_lhs_parameters_to_config(config, parameters)


def resolve_m4_sample(
    sample: Dict[str, Any],
    *,
    repo_root: Path,
    run_root: Path,
    scout_mesh_rel: str,
    force: bool,
    skip_mesh: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    sample_id = str(sample["sample_id"])
    sample_dir = run_root / "sample"
    resolved_path = sample_dir / "resolved_core_config.json"
    readiness_path = sample_dir / "readiness_check.json"
    overlay_path = sample_dir / "overlay_applied.json"

    if resolved_path.is_file() and readiness_path.is_file() and not force:
        readiness = _load_json(readiness_path)
        cached_status = readiness.get("status")
        if cached_status == "PASS":
            return _load_json(resolved_path), readiness, resolved_path
        if cached_status == "PENDING_MESH" and not skip_mesh:
            mesh_abs = (repo_root / scout_mesh_rel).resolve()
            if mesh_abs.is_file():
                readiness = dict(readiness)
                readiness["status"] = "PASS"
                readiness["mesh_exists"] = {MESH_LEVEL: True}
                write_json_atomic(readiness_path, readiness)
            return _load_json(resolved_path), readiness, resolved_path

    sample_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE_CORE.read_text(encoding="utf-8"))
    baseline_sha = _sha256_file(BASELINE_CORE)
    resolved = copy.deepcopy(baseline)

    params = sample.get("parameters")
    if not isinstance(params, dict):
        params = {}
    shape = str(sample.get("shape_name") or "classic")
    m4_meta = resolved.setdefault("m4_run_metadata", {})
    if isinstance(m4_meta, dict):
        m4_meta["shape_name"] = shape
    geom = resolved.setdefault("geometry", {})
    if isinstance(geom, dict):
        geom["shape_type"] = shape
    sample_geom = extract_geometry_dict(sample)
    if sample_geom:
        resolved["geometry_numeric_parameters"] = dict(sample_geom)
        if isinstance(geom, dict):
            for key, val in sample_geom.items():
                geom[key] = val

    if params:
        _wood_library_apply(resolved, params, repo_root=repo_root)

    solver = resolved.setdefault("solver", {})
    solver["mesh_file"] = scout_mesh_rel
    solver["clamp_ribs"] = False
    for forbidden_key in (
        "eps_interval_fallback",
        "eps_ciss_fallback_shift_invert",
        "resolvent_static_fallback",
    ):
        solver.pop(forbidden_key, None)
    m4_meta = resolved.setdefault("m4_run_metadata", {})
    if isinstance(m4_meta, dict):
        m4_meta.setdefault("dataset_version", "m4_geometry_corrected_v1")

    generated = _utc_now()
    write_json_atomic(resolved_path, resolved)
    resolved_sha = _sha256_file(resolved_path)

    mesh_abs = (repo_root / scout_mesh_rel).resolve()
    mesh_exists = mesh_abs.is_file()
    mats = resolved.get("materials") or {}
    errors: List[str] = []
    warnings: List[str] = []
    if mesh_exists:
        readiness_status = "PASS"
    elif skip_mesh:
        readiness_status = "FAIL"
        errors.append(
            "scout mesh missing and mesh build/reuse disabled (skip_mesh=True; expected existing PASS mesh)"
        )
    else:
        readiness_status = "PENDING_MESH"
        warnings.append(
            f"scout mesh not present yet at {scout_mesh_rel}; Stage 1 will build or copy before checkpoint"
        )

    overlay = {
        "schema": "m4_sample_overlay_applied_v1",
        "generated_utc": generated,
        "sample_id": sample_id,
        "base_config_sha256": baseline_sha,
        "mesh_level": MESH_LEVEL,
        "mesh_file": scout_mesh_rel,
        "parameters": params,
        "shape_name": shape,
        "geometry_numeric_parameters": sample_geom,
        "geometry_fingerprint": geometry_fingerprint(sample_geom) if sample_geom else None,
    }
    write_json_atomic(overlay_path, overlay)

    readiness = {
        "schema": "m4_readiness_check_v1",
        "generated_utc": generated,
        "sample_id": sample_id,
        "status": readiness_status,
        "stage0_outcome": "PASS" if readiness_status in ("PASS", "PENDING_MESH") else "FAIL",
        "mesh_pending_stage1": readiness_status == "PENDING_MESH",
        "resolved_config_path": _rel(resolved_path, repo_root=repo_root),
        "mesh_file": scout_mesh_rel,
        "mesh_exists": {MESH_LEVEL: mesh_exists},
        "core_config_readable": True,
        "core_config_sha256": resolved_sha,
        "solver_clamp_ribs": False,
        "effective_materials": {
            "top.density": float((mats.get("top") or {}).get("density", float("nan")))
            if isinstance(mats.get("top"), dict)
            else None,
            "back.density": float((mats.get("back") or {}).get("density", float("nan")))
            if isinstance(mats.get("back"), dict)
            else None,
        },
        "changed_material_values": _build_changed_material_values(baseline, resolved, {}),
        "warnings": warnings,
        "errors": errors,
    }
    write_json_atomic(readiness_path, readiness)

    manifest_stub = {
        "schema": "m4_sample_manifest_v1",
        "sample_id": sample_id,
        "run_id": run_root.name,
        "generated_utc": generated,
        "status": readiness["status"],
        "resolved_core_config_path": _rel(resolved_path, repo_root=repo_root),
        "readiness_check_path": _rel(readiness_path, repo_root=repo_root),
        "overlay_applied_path": _rel(overlay_path, repo_root=repo_root),
        "core_config_sha256": resolved_sha,
        "mesh_paths": {
            "scout": {
                "mesh_level": MESH_LEVEL,
                "mesh_file": scout_mesh_rel,
                "case_id": sample_id,
            },
            "lprod": {
                "mesh_level": "L_prod",
                "mesh_file": _rel(run_root / "lprod" / "mesh" / "L_prod" / f"{sample_id}.msh", repo_root=repo_root),
                "case_id": sample_id,
            },
        },
        "solver_policy": {"clamp_ribs": False},
        "geometry_and_material": params,
        "requires_mesh_regeneration": bool(sample.get("requires_mesh_regeneration", True)),
    }
    write_json_atomic(sample_dir / "sample_resolved_config_manifest.json", manifest_stub)

    if readiness["status"] == "FAIL":
        raise RuntimeError(f"Stage 0 readiness FAIL: {errors}")
    return resolved, readiness, resolved_path


def _mesh_pass(mesh_path: Path) -> bool:
    return mesh_path.is_file() and mesh_path.stat().st_size > 1000


def _install_scout_mesh(*, repo_root: Path, sample_id: str, dst: Path) -> None:
    src = (CONV_MESH / MESH_LEVEL / f"{sample_id}.msh").resolve()
    if not src.is_file():
        raise FileNotFoundError(
            f"sample-specific scout mesh build output missing: {src} "
            f"(refusing baseline fallback {MESH_CASE_BASE}.msh)"
        )
    if BASELINE_SCOUT_MESH.is_file() and src.resolve() == BASELINE_SCOUT_MESH.resolve():
        raise RuntimeError(
            f"scout_mesh_baseline_contamination: refusing to install baseline mesh for {sample_id}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    summary_src = CONV_MESH / MESH_LEVEL / f"{sample_id}_mesh_build_summary.json"
    if summary_src.is_file():
        shutil.copy2(summary_src, dst.parent / f"{sample_id}_mesh_build_summary.json")


def _cmd_stage_a(
    *,
    repo_root: Path,
    prod_python: str,
    core_config_rel: str,
    checkpoint_dir: Path,
    operator_mesh_rel: str,
) -> List[str]:
    return [
        prod_python,
        _path_for_subprocess(repo_root / STAGE_A_REL, repo_root=repo_root),
        "--mesh-level",
        MESH_LEVEL,
        "--B3-block-compose-backend",
        "csr_bulk",
        "--B3-synthesis-region-dofs",
        "off",
        "--core-config",
        core_config_rel,
        "--operator-mesh-file",
        operator_mesh_rel,
        "--output-dir",
        _path_for_subprocess(checkpoint_dir, repo_root=repo_root),
    ]


def _cmd_scout_mesh_build(
    *,
    repo_root: Path,
    prod_python: str,
    run_root: Path,
    sample_id: str,
) -> List[str]:
    return [
        prod_python,
        _path_for_subprocess(repo_root / SCOUT_MESH_SCRIPT, repo_root=repo_root),
        "--sample-id",
        sample_id,
        "--run-dir",
        _path_for_subprocess(run_root, repo_root=repo_root),
    ]


def _cmd_stage_b(
    *,
    repo_root: Path,
    solver_python: str,
    checkpoint_dir: Path,
    discovery_dir: Path,
    reference_json: Path,
    freq_min: float,
    freq_max: float,
    spacing_hz: float,
    half_width_hz: float,
) -> List[str]:
    return [
        solver_python,
        _path_for_subprocess(repo_root / STAGE_B_REL, repo_root=repo_root),
        "--checkpoint-dir",
        _path_for_subprocess(checkpoint_dir, repo_root=repo_root),
        "--reference-json",
        _path_for_subprocess(reference_json, repo_root=repo_root),
        "--start-hz",
        str(freq_min),
        "--stop-hz",
        str(freq_max),
        "--spacings-hz",
        str(spacing_hz),
        "--B3-discovery-mode",
        "--discovery-band-hz",
        str(freq_min),
        str(freq_max),
        "--target-window-half-width-hz",
        str(half_width_hz),
        "--output-dir",
        _path_for_subprocess(discovery_dir, repo_root=repo_root),
    ]


def _update_manifest(
    manifest_path: Path,
    *,
    stage_updates: Dict[str, str],
    terminal_status: str,
    failure_reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    manifest.setdefault("stages", {})
    for key, status in stage_updates.items():
        st = manifest["stages"].setdefault(key, {})
        st["status"] = status
        st["updated_utc"] = _utc_now()
    manifest["terminal_status"] = terminal_status
    manifest["updated_utc"] = _utc_now()
    manifest["will_execute"] = True
    manifest["mode"] = "scout_execute"
    if failure_reason:
        manifest["failure_reason"] = failure_reason
    if extra:
        manifest.update(extra)
    write_json_atomic(manifest_path, manifest)
    return manifest


def build_execution_plan(
    *,
    repo_root: Path,
    run_root: Path,
    sample: Dict[str, Any],
    freq_min: float,
    freq_max: float,
    scout_spacing: float,
    scout_half_width: float,
    workers: int,
    prod_python: str,
    solver_python: str,
    reference_json: Path,
    force: bool,
) -> Dict[str, Any]:
    sample_id = str(sample["sample_id"])
    run_id = run_root.name
    scout_mesh = run_root / "scout" / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    scout_mesh_rel = _rel(scout_mesh, repo_root=repo_root)
    checkpoint_dir = run_root / "scout" / "checkpoint"
    discovery_dir = run_root / "scout" / "discovery"
    density_json = discovery_dir / "density_result.json"
    export_manifest = checkpoint_dir / "checkpoint_export_manifest.json"

    mesh_ok = _mesh_pass(scout_mesh)
    ckpt_ok = False
    if export_manifest.is_file():
        ckpt_ok, _ = _verify_stage_a_export(export_manifest)
    dens_ok = False
    if density_json.is_file():
        dens_ok, _, _ = _verify_density_result(
            density_json,
            strict=is_strict_production_mode(dataset_version=DATASET_VERSION),
        )

    resolved_rel = _rel(run_root / "sample" / "resolved_core_config.json", repo_root=repo_root)

    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": _rel(run_root, repo_root=repo_root),
        "scout_mesh": scout_mesh_rel,
        "mesh_pass": mesh_ok,
        "checkpoint_pass": ckpt_ok,
        "density_pass": dens_ok,
        "skip_mesh": mesh_ok and not force,
        "skip_checkpoint": ckpt_ok and not force,
        "skip_discovery": dens_ok and not force,
        "workers": workers,
        "discovery": {
            "freq_min_hz": freq_min,
            "freq_max_hz": freq_max,
            "spacing_hz": scout_spacing,
            "half_width_hz": scout_half_width,
        },
        "argv_mesh": _cmd_scout_mesh_build(
            repo_root=repo_root,
            prod_python=prod_python,
            run_root=run_root,
            sample_id=sample_id,
        ),
        "argv_stage_a": _cmd_stage_a(
            repo_root=repo_root,
            prod_python=prod_python,
            core_config_rel=resolved_rel,
            checkpoint_dir=checkpoint_dir,
            operator_mesh_rel=scout_mesh_rel,
        ),
        "argv_stage_b": _cmd_stage_b(
            repo_root=repo_root,
            solver_python=solver_python,
            checkpoint_dir=checkpoint_dir,
            discovery_dir=discovery_dir,
            reference_json=reference_json,
            freq_min=freq_min,
            freq_max=freq_max,
            spacing_hz=scout_spacing,
            half_width_hz=scout_half_width,
        ),
    }


def run_stage3_zones_target_plan(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    freq_min: float,
    freq_max: float,
    scout_spacing: float,
    scout_half_width: float,
    workers: int,
    zone_spacing_hz: Dict[str, float],
    scout_mesh_rel: str,
    manifest_path: Path,
    explicit_target_plan_path: Optional[Path] = None,
) -> int:
    """Stage 3 only: density zones + gapless L_prod target plan from existing discovery."""
    logs = run_root / "logs"
    log3 = logs / "stage3_zones_target_plan.log"
    discovery_dir = run_root / "scout" / "discovery"
    density_json = discovery_dir / "density_result.json"
    checkpoint_dir = run_root / "scout" / "checkpoint"
    scout_dir = run_root / "scout"
    lprod_dir = run_root / "lprod"

    strict_density = is_strict_production_mode(dataset_version=DATASET_VERSION)
    ok_d, detail_d, _ = _verify_density_result(density_json, strict=strict_density)
    if not ok_d:
        print(f"[m4_scout] Stage 3 abort: missing or invalid density_result: {detail_d}", flush=True)
        return 1

    try:
        density_body = _load_json(density_json)
        if strict_density:
            if str(density_body.get("status") or "") != "PASS":
                raise RuntimeError(
                    f"strict_density_status={density_body.get('status') or 'missing'} "
                    f"intrinsic_pass={density_body.get('intrinsic_coverage_pass')}"
                )
            if not bool(density_body.get("intrinsic_coverage_pass")):
                raise RuntimeError(
                    f"intrinsic_coverage_failures={density_body.get('intrinsic_coverage_failures')}"
                )
            if bool(density_body.get("external_reference_gate_enabled")) and (
                density_body.get("external_reference_classification") == "discovery_stub"
            ):
                raise RuntimeError("external_reference_gate_enabled_with_stub")
        unique_hz = _extract_unique_frequencies(density_body)
        bins = build_density_bins(
            unique_hz,
            freq_min_hz=freq_min,
            freq_max_hz=freq_max,
            bin_width_hz=BIN_WIDTH_HZ,
        )
        class_meta = classify_bins_percentile(bins)
        for b in bins:
            zid = str(b["zone_id"])
            b["recommended_lprod_spacing_hz"] = float(zone_spacing_hz[zid])
        segments = merge_zone_segments(bins)
        for s in segments:
            s["recommended_lprod_spacing_hz"] = float(zone_spacing_hz[str(s["zone_id"])])
        density_doc = build_density_zones_document(
            sample_id=sample_id,
            run_id=run_id,
            bins=bins,
            segments=segments,
            classification_meta=class_meta,
            unique_hz=unique_hz,
            freq_min_hz=freq_min,
            freq_max_hz=freq_max,
            bin_width_hz=BIN_WIDTH_HZ,
            density_result_path=_rel(density_json, repo_root=repo_root),
        )
        density_doc["zone_spacing_hz"] = dict(zone_spacing_hz)
        write_json_atomic(scout_dir / "density_zones.json", density_doc)
        (scout_dir / "density_zones.md").write_text(
            render_density_zones_md(density_doc), encoding="utf-8"
        )

        if explicit_target_plan_path is not None:
            from v2_b3_m4_mesh_profile_lib import install_explicit_target_plan  # noqa: WPS433

            target_plan = install_explicit_target_plan(
                run_root=run_root,
                target_plan_path=explicit_target_plan_path,
                sample_id=sample_id,
                run_id=run_id,
            )
        else:
            target_plan = build_gapless_target_plan(
                segments,
                sample_id=sample_id,
                run_id=run_id,
                freq_min_hz=freq_min,
                freq_max_hz=freq_max,
            )
        if explicit_target_plan_path is not None:
            cov = target_plan.get("coverage_check") or {"pass": True, "repair_targets_added": 0}
            repair_count = 0
            target_plan.setdefault("coverage_check", cov)
        else:
            cov = target_plan["coverage_check"]
            repair_count = int(cov.get("repair_targets_added") or 0)
            target_plan["raw_discovery_provenance"] = {
                "intrinsic_coverage_pass": bool(density_body.get("intrinsic_coverage_pass")),
                "raw_unique_accepted_count": density_body.get("raw_unique_accepted_count"),
                "raw_max_gap_hz": density_body.get("raw_max_gap_hz"),
                "coverage_policy": density_body.get("coverage_policy"),
                "target_plan_repair_targets_added": repair_count,
            }
            if not cov.get("pass"):
                raise RuntimeError(f"coverage_check failed after repair: {cov}")
            if repair_count > 0 and not bool(density_body.get("intrinsic_coverage_pass")):
                raise RuntimeError(
                    f"target_plan_repair_disallowed_without_intrinsic_pass:repair_count={repair_count}"
                )
        scout_wall = float(density_body.get("experiment_wall_s") or 0.0)
        runtime_est = estimate_runtime_summary(
            target_count=len(target_plan["targets_hz"]),
            scout_wall_seconds=scout_wall,
            workers=workers,
            freq_min_hz=freq_min,
            freq_max_hz=freq_max,
        )
        target_plan["estimated_runtime"] = runtime_est
        target_plan["density_zones_path"] = _rel(scout_dir / "density_zones.json", repo_root=repo_root)
        write_json_atomic(lprod_dir / "lprod_target_plan.json", target_plan)
        (lprod_dir / "lprod_target_plan.md").write_text(
            render_target_plan_md(target_plan, runtime_est), encoding="utf-8"
        )

        chunk_preview = build_worker_chunk_preview(
            segments,
            sample_id=sample_id,
            run_id=run_id,
            freq_min_hz=freq_min,
            freq_max_hz=freq_max,
            targets_hz=target_plan["targets_hz"],
            target_windows_hz=target_plan["target_windows_hz"],
        )
        write_json_atomic(lprod_dir / "worker_chunk_plan.preview.json", chunk_preview)
        (lprod_dir / "worker_chunk_plan.preview.md").write_text(
            render_chunk_preview_md(chunk_preview), encoding="utf-8"
        )

        scout_result = {
            "schema": "m4_scout_result_v1",
            "will_execute": True,
            "status": "PASS",
            "sample_id": sample_id,
            "run_id": run_id,
            "generated_utc": _utc_now(),
            "scout_policy_version": "v1",
            "mesh_level": MESH_LEVEL,
            "mesh_path": scout_mesh_rel,
            "checkpoint_dir": _rel(checkpoint_dir, repo_root=repo_root),
            "checkpoint_status": "PASS",
            "discovery": {
                "frequency_range_hz": [freq_min, freq_max],
                "spacing_hz": scout_spacing,
                "half_width_hz": scout_half_width,
                "discovery_mode": True,
                "density_result_path": _rel(density_json, repo_root=repo_root),
                "unique_accepted_count": len(unique_hz),
                "experiment_status": density_body.get("status"),
                "experiment_wall_s": scout_wall,
            },
            "artifacts": {
                "density_zones_json": _rel(scout_dir / "density_zones.json", repo_root=repo_root),
                "lprod_target_plan_json": _rel(lprod_dir / "lprod_target_plan.json", repo_root=repo_root),
            },
        }
        write_json_atomic(scout_dir / "scout_result.json", scout_result)
        _append_log(
            log3,
            f"Stage 3 OK targets={len(target_plan['targets_hz'])} "
            f"repair={cov.get('repair_targets_added', 0)}\n",
        )
    except Exception as exc:
        _append_log(log3, f"FAIL: {exc}\n")
        _update_manifest(
            manifest_path,
            stage_updates={"stage3_zones_plan": "FAIL"},
            terminal_status="FAIL",
            failure_reason=f"stage3_failed:{exc}",
        )
        print(f"Stage 3 FAIL: {exc}", flush=True)
        return 1

    _update_manifest(
        manifest_path,
        stage_updates={
            "stage3_zones_plan": "PASS",
            "stage4_lprod_mesh": "PLANNED",
            "stage4_lprod_export": "PLANNED",
            "stage5_workers": "PLANNED",
            "stage6_aggregate": "PLANNED",
        },
        terminal_status="SCOUT_PASS_TARGET_PLAN_READY",
        extra={
            "scout_complete": True,
            "target_count": len(target_plan["targets_hz"]),
            "repair_targets_added": cov.get("repair_targets_added", 0),
            "no_lprod_executed": True,
        },
    )
    print("Stage 3 PASS", flush=True)
    print(
        f"coverage_check.pass=true target_count={len(target_plan['targets_hz'])} "
        f"repair_targets_added={cov.get('repair_targets_added', 0)}",
        flush=True,
    )
    print("SCOUT_PASS_TARGET_PLAN_READY", flush=True)
    print("no L_prod executed", flush=True)
    return 0


def run_scout_pipeline(
    *,
    repo_root: Path,
    run_root: Path,
    sample: Dict[str, Any],
    freq_min: float,
    freq_max: float,
    scout_spacing: float,
    scout_half_width: float,
    workers: int,
    prod_python: str,
    prod_venv: str,
    solver_python: str,
    solver_venv: str,
    reference_json: Path,
    force: bool,
    dry_run: bool,
    stage3_only: bool,
    zone_spacing_hz: Dict[str, float],
    explicit_target_plan_path: Optional[Path] = None,
) -> int:
    sample_id = str(sample["sample_id"])
    run_id = run_root.name
    logs = run_root / "logs"
    manifest_path = run_root / "pipeline_run_manifest.json"

    plan = build_execution_plan(
        repo_root=repo_root,
        run_root=run_root,
        sample=sample,
        freq_min=freq_min,
        freq_max=freq_max,
        scout_spacing=scout_spacing,
        scout_half_width=scout_half_width,
        workers=workers,
        prod_python=prod_python,
        solver_python=solver_python,
        reference_json=reference_json,
        force=force,
    )

    scout_mesh = run_root / "scout" / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    scout_mesh_rel = _rel(scout_mesh, repo_root=repo_root)

    if dry_run:
        print("[m4_scout] DRY-RUN preview (no execution)", flush=True)
        print(f"[m4_scout] run_root={plan['run_root']}", flush=True)
        if stage3_only:
            print("[m4_scout] mode=stage3_only (would reuse PASS mesh/checkpoint/discovery)", flush=True)
        else:
            print(
                f"[m4_scout] skip_mesh={plan['skip_mesh']} "
                f"skip_checkpoint={plan['skip_checkpoint']} "
                f"skip_discovery={plan['skip_discovery']}",
                flush=True,
            )
            print(f"[m4_scout] mesh: {' '.join(plan['argv_mesh'])}", flush=True)
            print(f"[m4_scout] stage_a: {' '.join(plan['argv_stage_a'])}", flush=True)
            print(f"[m4_scout] stage_b: {' '.join(plan['argv_stage_b'])}", flush=True)
        print("[m4_scout] stage3: regenerate density_zones + lprod_target_plan from discovery", flush=True)
        return 0

    if stage3_only:
        print("[m4_scout] stage3-only: reusing PASS Stages 0–2 artifacts", flush=True)
        print("Stage 0 PASS / reused", flush=True)
        print("Stage 1 PASS / reused", flush=True)
        print("Stage 2 PASS / reused", flush=True)
        return run_stage3_zones_target_plan(
            repo_root=repo_root,
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
            freq_min=freq_min,
            freq_max=freq_max,
            scout_spacing=scout_spacing,
            scout_half_width=scout_half_width,
            workers=workers,
            zone_spacing_hz=zone_spacing_hz,
            scout_mesh_rel=scout_mesh_rel,
            manifest_path=manifest_path,
            explicit_target_plan_path=explicit_target_plan_path,
        )

    env_probe_path = logs / "env_probe.json"
    if env_probe_path.is_file() and not force:
        probe_body = _load_json(env_probe_path)
        env_ok = bool((probe_body.get("stage_a") or {}).get("ok")) and bool(
            (probe_body.get("stage_b") or {}).get("ok")
        )
    else:
        env_ok, probe_body = _run_stage_env_probes(
            repo_root=repo_root,
            prod_python=prod_python,
            prod_venv=prod_venv,
            solver_python=solver_python,
            solver_venv=solver_venv,
            log_dir=logs,
        )
        write_json_atomic(env_probe_path, probe_body)
    if not env_ok:
        _update_manifest(
            manifest_path,
            stage_updates={"stage0_resolve": "FAIL"},
            terminal_status="FAIL",
            failure_reason="env_probe_failed",
        )
        print("[m4_scout] FAIL env probes", flush=True)
        return 1

    # Stage 0
    log0 = logs / "stage0_config.log"
    try:
        _append_log(log0, f"[{_utc_now()}] Stage 0 resolve sample_id={sample_id}\n")
        resolved, readiness, _ = resolve_m4_sample(
            sample,
            repo_root=repo_root,
            run_root=run_root,
            scout_mesh_rel=scout_mesh_rel,
            force=force,
            skip_mesh=bool(plan["skip_mesh"]),
        )
        if readiness.get("status") == "PENDING_MESH":
            _append_log(log0, "readiness PENDING_MESH (Stage 1 will build/copy mesh)\n")
        stage0_status = str(readiness.get("stage0_outcome") or "FAIL")
    except Exception as exc:
        stage0_status = "FAIL"
        _append_log(log0, f"FAIL: {exc}\n")
        _update_manifest(manifest_path, stage_updates={"stage0_resolve": "FAIL"}, terminal_status="FAIL", failure_reason=str(exc))
        print(f"[m4_scout] Stage 0 FAIL: {exc}", flush=True)
        return 1

    _update_manifest(manifest_path, stage_updates={"stage0_resolve": stage0_status}, terminal_status="RUNNING")
    if readiness.get("status") == "PENDING_MESH":
        print("Stage 0 PASS (PENDING_MESH — Stage 1 will build/copy mesh)", flush=True)
    else:
        print("Stage 0 PASS", flush=True)

    env_a = _prod_subprocess_env_strict(prod_python=prod_python, prod_venv=prod_venv)
    env_b = _solver_mkl_subprocess_env_strict(solver_python=solver_python, solver_venv=solver_venv)

    # Stage 1 mesh
    log_mesh = logs / "stage1_scout_mesh.log"
    if plan["skip_mesh"]:
        mesh_ok = _mesh_pass(scout_mesh)
        if mesh_ok:
            _append_log(log_mesh, f"[{_utc_now()}] reuse PASS mesh {scout_mesh_rel}\n")
            stage1_mesh = "PASS"
        else:
            _append_log(
                log_mesh,
                f"[{_utc_now()}] FAIL: skip_mesh=True but mesh missing at {scout_mesh_rel}\n",
            )
            stage1_mesh = "FAIL"
    else:
        rc_mesh = _run_subprocess(
            plan["argv_mesh"],
            env=env_a,
            cwd=repo_root,
            log_path=log_mesh,
            label="scout_mesh_build",
        )
        try:
            _install_scout_mesh(repo_root=repo_root, sample_id=sample_id, dst=scout_mesh)
            mesh_ok = _mesh_pass(scout_mesh)
        except Exception as exc:
            mesh_ok = False
            _append_log(log_mesh, f"mesh install FAIL: {exc}\n")
        stage1_mesh = "PASS" if rc_mesh == 0 and mesh_ok else "FAIL"
        if stage1_mesh == "PASS":
            sample_geom = extract_geometry_dict(sample)
            geom_match, _ = (
                geometries_match(sample_geom, BASELINE_GEOMETRY) if sample_geom else (False, ["no geometry"])
            )
            _append_log(
                log_mesh,
                f"scout_mesh_provenance: path={scout_mesh_rel} sample_specific=True "
                f"geometry_matches_baseline={geom_match}\n",
            )
            resolve_m4_sample(
                sample,
                repo_root=repo_root,
                run_root=run_root,
                scout_mesh_rel=scout_mesh_rel,
                force=True,
                skip_mesh=True,
            )

    if stage1_mesh != "PASS":
        _update_manifest(
            manifest_path,
            stage_updates={"stage1_scout_mesh": "FAIL"},
            terminal_status="FAIL",
            failure_reason="stage1_scout_mesh_failed",
        )
        print("Stage 1 FAIL (mesh)", flush=True)
        return 1
    _update_manifest(manifest_path, stage_updates={"stage1_scout_mesh": "PASS"}, terminal_status="RUNNING")
    print("Stage 1 PASS", flush=True)

    checkpoint_dir = run_root / "scout" / "checkpoint"
    export_manifest = checkpoint_dir / "checkpoint_export_manifest.json"
    log_ckpt = logs / "stage1_scout_checkpoint.log"

    if plan["skip_checkpoint"]:
        _append_log(log_ckpt, f"[{_utc_now()}] reuse PASS checkpoint\n")
        stage1_ckpt = "PASS"
    else:
        rc_a = _run_subprocess(
            plan["argv_stage_a"],
            env=env_a,
            cwd=repo_root,
            log_path=log_ckpt,
            label="scout_stage_a",
        )
        ok_a, detail_a = _verify_stage_a_export(export_manifest)
        stage1_ckpt = "PASS" if rc_a == 0 and ok_a else "FAIL"
        if stage1_ckpt != "PASS":
            _append_log(log_ckpt, f"verify FAIL: {detail_a}\n")

    if stage1_ckpt != "PASS":
        _update_manifest(
            manifest_path,
            stage_updates={"stage1_scout_export": "FAIL"},
            terminal_status="FAIL",
            failure_reason="stage1_scout_checkpoint_failed",
        )
        print("Stage 1 FAIL (checkpoint)", flush=True)
        return 1
    _update_manifest(manifest_path, stage_updates={"stage1_scout_export": "PASS"}, terminal_status="RUNNING")

    # Stage 2 discovery
    discovery_dir = run_root / "scout" / "discovery"
    density_json = discovery_dir / "density_result.json"
    log_b = logs / "stage2_scout_discovery.log"

    if plan["skip_discovery"]:
        _append_log(log_b, f"[{_utc_now()}] reuse PASS density_result\n")
        stage2 = "PASS"
    else:
        rc_b = _run_subprocess(
            plan["argv_stage_b"],
            env=env_b,
            cwd=repo_root,
            log_path=log_b,
            label="scout_stage_b",
        )
        ok_b, detail_b, _ = _verify_density_result(
            density_json,
            strict=is_strict_production_mode(dataset_version=DATASET_VERSION),
        )
        stage2 = "PASS" if rc_b == 0 and ok_b else "FAIL"
        if stage2 != "PASS":
            _append_log(log_b, f"verify FAIL: {detail_b}\n")

    if stage2 != "PASS":
        _update_manifest(
            manifest_path,
            stage_updates={"stage2_scout_discovery": "FAIL"},
            terminal_status="FAIL",
            failure_reason="stage2_scout_discovery_failed",
        )
        print("Stage 2 FAIL", flush=True)
        return 1
    _update_manifest(manifest_path, stage_updates={"stage2_scout_discovery": "PASS"}, terminal_status="RUNNING")
    print("Stage 2 PASS", flush=True)

    return run_stage3_zones_target_plan(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        freq_min=freq_min,
        freq_max=freq_max,
        scout_spacing=scout_spacing,
        scout_half_width=scout_half_width,
        workers=workers,
        zone_spacing_hz=zone_spacing_hz,
        scout_mesh_rel=scout_mesh_rel,
        manifest_path=manifest_path,
        explicit_target_plan_path=explicit_target_plan_path,
    )


def _ensure_run_tree(run_root: Path, sample: Dict[str, Any]) -> None:
    sample_dir = run_root / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    inp = sample_dir / "sample_input.json"
    if not inp.is_file():
        write_json_atomic(inp, sample)
    for sub in (
        run_root / "scout" / "mesh" / MESH_LEVEL,
        run_root / "scout" / "checkpoint",
        run_root / "scout" / "discovery",
        run_root / "lprod",
        run_root / "logs",
        run_root / "worker_results",
        run_root / "aggregation",
    ):
        sub.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4.3 scout pipeline (Stages 0–3 only).")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing M4.2 run directory.")
    parser.add_argument("--sample-json", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--freq-min-hz", type=float, default=60.0)
    parser.add_argument("--freq-max-hz", type=float, default=550.0)
    parser.add_argument("--scout-spacing-hz", type=float, default=7.5)
    parser.add_argument("--scout-half-width-hz", type=float, default=3.75)
    parser.add_argument("--zone-spacing-dense-hz", type=float, default=6.0)
    parser.add_argument("--zone-spacing-medium-hz", type=float, default=9.0)
    parser.add_argument("--zone-spacing-sparse-hz", type=float, default=12.5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Preview only; no subprocess execution.")
    parser.add_argument("--execute-scout", action="store_true", help="Run Stages 0–3.")
    parser.add_argument(
        "--stage3-only",
        action="store_true",
        help="Reuse PASS mesh/checkpoint/discovery; regenerate Stage 3 only.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run stages even if PASS artifacts exist.")
    parser.add_argument("--prod-python", default=os.environ.get("B3_PROD_PYTHON", M3_PROD_PYTHON))
    parser.add_argument("--prod-venv", default=os.environ.get("B3_PROD_VENV", DEFAULT_PROD_VENV))
    parser.add_argument("--solver-python", default=os.environ.get("B3_SOLVER_PYTHON", M3_SOLVER_PYTHON))
    parser.add_argument("--solver-venv", default=os.environ.get("B3_SOLVER_MKL_VENV", "/home/vboxuser/solver-mkl/venv"))
    parser.add_argument("--reference-json", type=Path, default=str(DEFAULT_REFERENCE))
    parser.add_argument(
        "--target-plan-file",
        type=Path,
        help="Validation-only: install explicit frozen lprod_target_plan.json (SHA256 recorded).",
    )
    args = parser.parse_args(argv)

    if args.stage3_only and not args.dry_run:
        args.execute_scout = True

    if not args.dry_run and not args.execute_scout:
        print("error: specify --dry-run, --execute-scout, or --stage3-only", file=sys.stderr)
        return 2

    repo_root = _detect_repo_root(SCRIPT_DIR)

    if args.run_dir:
        run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
        run_root = run_root.resolve()
        sample_path = run_root / "sample" / "sample_input.json"
        if not sample_path.is_file():
            print(f"error: missing {sample_path}", file=sys.stderr)
            return 2
        sample = _load_json(sample_path)
    elif args.sample_json and args.run_id:
        sample_path = args.sample_json if args.sample_json.is_absolute() else repo_root / args.sample_json
        sample = _load_json(sample_path)
        sample_id = str(sample["sample_id"])
        run_root = (GUITARS_ROOT / sample_id / "runs" / str(args.run_id)).resolve()
        _ensure_run_tree(run_root, sample)
    else:
        print("error: provide --run-dir or (--sample-json and --run-id)", file=sys.stderr)
        return 2

    reference_json = args.reference_json if args.reference_json.is_absolute() else repo_root / args.reference_json

    if args.execute_scout and run_root.exists() and not args.force and not args.stage3_only:
        manifest_path = run_root / "pipeline_run_manifest.json"
        if manifest_path.is_file():
            term = _load_json(manifest_path).get("terminal_status")
            if term == "SCOUT_PASS_TARGET_PLAN_READY":
                print(f"[m4_scout] already complete: {term} (use --force or --stage3-only)", flush=True)
                return 0

    zone_spacing_hz = {
        ZONE_1: float(args.zone_spacing_dense_hz),
        ZONE_2: float(args.zone_spacing_medium_hz),
        ZONE_3: float(args.zone_spacing_sparse_hz),
    }

    explicit_tp: Optional[Path] = None
    if args.target_plan_file:
        explicit_tp = args.target_plan_file if args.target_plan_file.is_absolute() else repo_root / args.target_plan_file

    return run_scout_pipeline(
        repo_root=repo_root,
        run_root=run_root,
        sample=sample,
        freq_min=float(args.freq_min_hz),
        freq_max=float(args.freq_max_hz),
        scout_spacing=float(args.scout_spacing_hz),
        scout_half_width=float(args.scout_half_width_hz),
        workers=int(args.workers),
        prod_python=str(args.prod_python),
        prod_venv=str(args.prod_venv),
        solver_python=str(args.solver_python),
        solver_venv=str(args.solver_venv),
        reference_json=reference_json.resolve(),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        stage3_only=bool(args.stage3_only),
        zone_spacing_hz=zone_spacing_hz,
        explicit_target_plan_path=explicit_tp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
