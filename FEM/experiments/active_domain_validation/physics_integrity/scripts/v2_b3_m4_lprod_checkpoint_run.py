#!/usr/bin/env python3
"""M4.4.1b-0 — Stage 4 only: L_prod mesh + checkpoint export (no worker solves)."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
STAGE_A_REL = (
    "FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py"
)
MESH_BUILD_REL = (
    "FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_lprod_mesh_build.py"
)

DEFAULT_PROD_PYTHON = "/home/vboxuser/final-project/.venv/bin/python"
DEFAULT_PROD_VENV = "/home/vboxuser/final-project/.venv"

MESH_LEVEL = "L_prod"
TERMINAL_READY = "LPROD_CHECKPOINT_READY"
ALLOWED_INPUT_TERMINAL = frozenset(
    {
        "SCOUT_PASS_TARGET_PLAN_READY",
        "LPROD_WORKER_PLAN_READY",
        "LPROD_CHECKPOINT_READY",
    }
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m3_orchestrator_run_one import (  # noqa: E402
    _run_subprocess,
    _verify_stage_a_export,
)
from v2_b3_m4_production_contracts import DATASET_VERSION  # noqa: E402
from v2_b3_m4_lprod_interfaces import (  # noqa: E402
    BASELINE_GEOMETRY,
    BASELINE_L_PROD_MESH,
    LPROD_MESH_LEVEL,
    LPROD_SYNTHESIS_REGION_DOFS_DEFAULT,
    evaluate_lprod_mesh_checkpoint_readiness,
    extract_geometry_dict,
    extract_run_metadata,
    geometries_match,
)
from v2_b3_rich_modal_lib import (  # noqa: E402
    REGION_DOF_INDICES_NPZ,
    SYNTHESIS_METADATA_JSON,
    load_region_dof_bundle,
)
from v2_b3_synthesis_export import region_dof_status_is_pass  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_resolve_pilot_core_config import _repo_relative  # noqa: E402
from v2_b3_run_coarse_scout_lhs_batch import (  # noqa: E402
    DEFAULT_PROD_VENV,
    STAGE_A_ENV_PROBE,
    _path_for_subprocess,
    _prod_subprocess_env_strict,
    _run_env_probe,
    _verify_stage_a_env_probe,
)
from v2_mesh_convergence_common import mesh_path  # noqa: E402


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


def _mesh_pass(mesh_path: Path) -> bool:
    return mesh_path.is_file() and mesh_path.stat().st_size > 1000


def _checkpoint_pass(export_manifest: Path) -> bool:
    if not export_manifest.is_file():
        return False
    ok, _ = _verify_stage_a_export(export_manifest)
    return ok


def _verify_lprod_checkpoint_export(
    *,
    checkpoint_dir: Path,
    export_manifest: Path,
    expected_core_config: Path,
    repo_root: Path,
) -> Tuple[bool, str, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    if not export_manifest.is_file():
        return False, "missing checkpoint_export_manifest.json", detail

    try:
        data = _load_json(export_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}", detail

    detail["manifest"] = {
        "status": data.get("status"),
        "export_pass": data.get("export_pass"),
        "matrix_verify_pass": data.get("matrix_verify_pass"),
        "mesh_level": data.get("mesh_level"),
    }

    ok_base, msg_base = _verify_stage_a_export(export_manifest)
    if not ok_base:
        return False, msg_base, detail

    built_path = checkpoint_dir / "built_metadata.json"
    if not built_path.is_file():
        return False, "missing built_metadata.json", detail
    built = _load_json(built_path)
    detail["built_metadata"] = built

    active_dim = data.get("active_dimension") or built.get("active_dimension")
    if active_dim is None:
        return False, "missing active_dimension", detail
    detail["active_dimension"] = active_dim

    mesh_level = str(data.get("mesh_level") or built.get("mesh_level") or "")
    if mesh_level != LPROD_MESH_LEVEL:
        return False, f"mesh_level={mesh_level!r} expected L_prod", detail

    prov = data.get("core_config_provenance") or {}
    core_path_raw = str(
        data.get("core_config_path") or prov.get("core_config_path") or ""
    ).replace("\\", "/")
    expected_rel = _rel(expected_core_config, repo_root=repo_root).replace("\\", "/")
    if expected_rel not in core_path_raw and not core_path_raw.endswith(expected_rel):
        return False, f"core_config_path mismatch: {core_path_raw!r} expected *{expected_rel}", detail

    for name in ("A_active_csr.npz", "M_active_csr.npz"):
        p = checkpoint_dir / name
        if not p.is_file():
            return False, f"missing {name}", detail

    return True, "ok", detail


def resolve_lprod_core_config(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    lprod_mesh_rel: str,
    force: bool,
    sample_input: Optional[Mapping[str, Any]] = None,
) -> Path:
    lprod_dir = run_root / "lprod"
    out_path = lprod_dir / "resolved_core_config.json"
    sample_resolved = run_root / "sample" / "resolved_core_config.json"
    if not sample_resolved.is_file():
        raise FileNotFoundError(f"missing scout resolved config: {sample_resolved}")

    if out_path.is_file() and not force:
        existing = _load_json(out_path)
        if str((existing.get("solver") or {}).get("mesh_file", "")).replace("\\", "/") == lprod_mesh_rel.replace(
            "\\", "/"
        ):
            return out_path

    resolved = copy.deepcopy(_load_json(sample_resolved))
    resolved.setdefault("solver", {})["mesh_file"] = lprod_mesh_rel
    resolved.setdefault("solver", {})["clamp_ribs"] = False
    if sample_input:
        geom = extract_geometry_dict(sample_input)
        if geom:
            resolved["geometry_numeric_parameters"] = dict(geom)
            geo_block = resolved.setdefault("geometry", {})
            if isinstance(geo_block, dict):
                for key, val in geom.items():
                    geo_block[key] = val
            params = resolved.setdefault("parameters", {})
            if isinstance(params, dict):
                for key, val in geom.items():
                    params[f"geometry.{key}"] = float(val)
        meta = extract_run_metadata(sample_input)
        if meta:
            m4_meta = resolved.setdefault("m4_run_metadata", {})
            if isinstance(m4_meta, dict):
                m4_meta.update(meta)
            if meta.get("shape_name"):
                resolved["shape_name"] = str(meta["shape_name"])
    resolved["dataset_version"] = DATASET_VERSION
    m4_meta = resolved.setdefault("m4_run_metadata", {})
    if isinstance(m4_meta, dict):
        m4_meta["dataset_version"] = DATASET_VERSION
    write_json_atomic(out_path, resolved)
    return out_path


def _install_mesh_from_src(*, src: Path, dst: Path, sample_id: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"L_prod mesh source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    summary_candidates = [
        src.parent / f"{src.stem.replace('.msh', '')}_mesh_build_summary.json",
        src.parent / f"{sample_id}_mesh_build_summary.json",
    ]
    for summary_src in summary_candidates:
        if summary_src.is_file():
            shutil.copy2(summary_src, dst.parent / f"{sample_id}_mesh_build_summary.json")
            break


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
    manifest["mode"] = "m4_4_1b_0_lprod_checkpoint"
    if failure_reason:
        manifest["failure_reason"] = failure_reason
    elif "failure_reason" in manifest and terminal_status == TERMINAL_READY:
        manifest.pop("failure_reason", None)
    if extra:
        manifest.update(extra)
    write_json_atomic(manifest_path, manifest)
    return manifest


def build_execution_plan(
    *,
    repo_root: Path,
    run_root: Path,
    sample_input: Dict[str, Any],
    manifest: Dict[str, Any],
    prod_python: str,
    force: bool,
) -> Dict[str, Any]:
    sample_id = str(sample_input.get("sample_id") or manifest.get("sample_id"))
    run_id = str(manifest.get("run_id") or run_root.name)
    lprod_mesh = run_root / "lprod" / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    lprod_mesh_rel = _rel(lprod_mesh, repo_root=repo_root)
    checkpoint_dir = run_root / "lprod" / "checkpoint"
    resolved_lprod = run_root / "lprod" / "resolved_core_config.json"
    resolved_lprod_rel = _rel(resolved_lprod, repo_root=repo_root)
    export_manifest = checkpoint_dir / "checkpoint_export_manifest.json"

    readiness = evaluate_lprod_mesh_checkpoint_readiness(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        sample_input=sample_input,
        rel_path_fn=lambda p, **kw: _rel(p, repo_root=repo_root),
    )

    mesh_ok = _mesh_pass(lprod_mesh)
    ckpt_ok = _checkpoint_pass(export_manifest)
    geom = extract_geometry_dict(sample_input)
    geom_match, _ = geometries_match(geom, BASELINE_GEOMETRY)

    mesh_action = "reuse_run_tree"
    if not mesh_ok:
        if readiness.get("lprod_mesh_status") == "reusable_existing" and geom_match:
            mesh_action = "copy_baseline_mesh"
        else:
            mesh_action = "build_sample_geometry"

    # --force re-exports checkpoint; it must not rebuild an existing run-tree mesh.
    skip_mesh = mesh_ok and mesh_action == "reuse_run_tree"

    return {
        "schema": "m4_lprod_checkpoint_run_plan_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": _rel(run_root, repo_root=repo_root),
        "mesh_action": mesh_action,
        "lprod_mesh_status": readiness.get("lprod_mesh_status"),
        "lprod_checkpoint_status": readiness.get("lprod_checkpoint_status"),
        "readiness": readiness,
        "skip_mesh": skip_mesh,
        "skip_checkpoint": ckpt_ok and not force,
        "paths": {
            "lprod_mesh": lprod_mesh_rel,
            "lprod_resolved_core_config": resolved_lprod_rel,
            "lprod_checkpoint_dir": _rel(checkpoint_dir, repo_root=repo_root),
        },
        "argv_mesh_build": [],
        "argv_stage_a": [
            prod_python,
            _path_for_subprocess(repo_root / Path(STAGE_A_REL), repo_root=repo_root),
            "--mesh-level",
            MESH_LEVEL,
            "--B3-block-compose-backend",
            "csr_bulk",
            "--B3-synthesis-region-dofs",
            LPROD_SYNTHESIS_REGION_DOFS_DEFAULT,
            "--core-config",
            resolved_lprod_rel,
            "--operator-mesh-file",
            lprod_mesh_rel,
            "--output-dir",
            _rel(checkpoint_dir, repo_root=repo_root),
        ],
    }


def _fix_mesh_argv(plan: Dict[str, Any], *, repo_root: Path, run_root: Path, sample_id: str, prod_python: str) -> None:
    plan["argv_mesh_build"] = [
        prod_python,
        _path_for_subprocess(repo_root / Path(MESH_BUILD_REL), repo_root=repo_root),
        "--sample-id",
        sample_id,
        "--run-dir",
        _rel(run_root, repo_root=repo_root),
    ]


def _log_lprod_region_dof_index_status(
    *,
    checkpoint_dir: Path,
    log_path: Path,
    readiness_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record region DOF index availability; warn if top/back masks missing (non-blocking)."""
    built_path = checkpoint_dir / "built_metadata.json"
    built_meta: Dict[str, Any] = {}
    if built_path.is_file():
        try:
            built_meta = _load_json(built_path)
        except (OSError, ValueError, json.JSONDecodeError):
            built_meta = {}

    ctx = load_region_dof_bundle(checkpoint_dir, built_meta)
    synth_status = None
    synth_path = checkpoint_dir / SYNTHESIS_METADATA_JSON
    if synth_path.is_file():
        try:
            synth_status = _load_json(synth_path).get("region_dof_indices_status")
        except (OSError, ValueError, json.JSONDecodeError):
            synth_status = None

    summary = {
        "region_dof_indices_npz": (checkpoint_dir / REGION_DOF_INDICES_NPZ).is_file(),
        "region_dof_source": ctx.get("region_dof_source"),
        "structural_indices_available": bool(ctx.get("structural_indices_available")),
        "pressure_indices_available": bool(ctx.get("pressure_indices_available")),
        "synthesis_region_dof_indices_status": synth_status,
        "synthesis_region_dofs_mode": LPROD_SYNTHESIS_REGION_DOFS_DEFAULT,
    }
    if readiness_out is not None:
        readiness_out["region_dof_indices"] = summary

    if summary["structural_indices_available"] and region_dof_status_is_pass(synth_status):
        msg = (
            f"[{_utc_now()}] region_dof_indices: present "
            f"(source={summary['region_dof_source']}); "
            "worker solves may compute top/back/air participation."
        )
        _append_log(log_path, msg + "\n")
        print(msg, flush=True)
    else:
        warn = (
            f"[{_utc_now()}] warning: top/back region DOF indices unavailable "
            f"(npz={summary['region_dof_indices_npz']}, "
            f"synthesis_status={synth_status!r}, source={summary['region_dof_source']}). "
            "Worker participation will use air/norm fallback only; production continues."
        )
        _append_log(log_path, warn + "\n")
        print(warn, flush=True)

    return summary


def _validate_inputs(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    target_plan: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    term = str(manifest.get("terminal_status") or "")
    if term not in ALLOWED_INPUT_TERMINAL:
        errors.append(f"terminal_status={term!r} not in {sorted(ALLOWED_INPUT_TERMINAL)}")

    st3 = (manifest.get("stages") or {}).get("stage3_zones_plan") or {}
    if str(st3.get("status")) != "PASS":
        errors.append(f"stage3_zones_plan.status={st3.get('status')!r} expected PASS")

    cov = target_plan.get("coverage_check") or {}
    if not cov.get("pass"):
        errors.append("lprod_target_plan.coverage_check.pass is false")

    if not (run_root / "sample" / "resolved_core_config.json").is_file():
        errors.append("missing sample/resolved_core_config.json (run scout first)")

    return errors


def run_dry_run(*, repo_root: Path, run_root: Path, prod_python: str, force: bool) -> int:
    manifest_path = run_root / "pipeline_run_manifest.json"
    sample_input = _load_json(run_root / "sample" / "sample_input.json")
    manifest = _load_json(manifest_path)
    target_plan = _load_json(run_root / "lprod" / "lprod_target_plan.json")

    val_errors = _validate_inputs(run_root=run_root, manifest=manifest, target_plan=target_plan)
    if val_errors:
        print("error: input validation failed:", file=sys.stderr)
        for e in val_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    plan = build_execution_plan(
        repo_root=repo_root,
        run_root=run_root,
        sample_input=sample_input,
        manifest=manifest,
        prod_python=prod_python,
        force=force,
    )
    _fix_mesh_argv(
        plan,
        repo_root=repo_root,
        run_root=run_root,
        sample_id=str(plan["sample_id"]),
        prod_python=prod_python,
    )

    write_json_atomic(run_root / "lprod" / "lprod_checkpoint_run_plan.json", plan)

    print("will_execute=false")
    print(f"sample_id={plan.get('sample_id')}")
    print(f"mesh_action={plan.get('mesh_action')}")
    print(f"lprod_mesh_status={plan.get('lprod_mesh_status')}")
    print(f"lprod_checkpoint_status={plan.get('lprod_checkpoint_status')}")
    print(f"skip_mesh={plan.get('skip_mesh')} skip_checkpoint={plan.get('skip_checkpoint')}")
    print(f"mesh: {' '.join(plan['argv_mesh_build'])}")
    print(f"stage_a: {' '.join(plan['argv_stage_a'])}")
    return 0


def run_execute(
    *,
    repo_root: Path,
    run_root: Path,
    prod_python: str,
    prod_venv: str,
    force: bool,
) -> int:
    logs = run_root / "logs"
    manifest_path = run_root / "pipeline_run_manifest.json"
    sample_input = _load_json(run_root / "sample" / "sample_input.json")
    manifest = _load_json(manifest_path)
    target_plan = _load_json(run_root / "lprod" / "lprod_target_plan.json")
    sample_id = str(sample_input.get("sample_id") or manifest.get("sample_id"))

    val_errors = _validate_inputs(run_root=run_root, manifest=manifest, target_plan=target_plan)
    if val_errors:
        print("error: input validation failed:", file=sys.stderr)
        for e in val_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    plan = build_execution_plan(
        repo_root=repo_root,
        run_root=run_root,
        sample_input=sample_input,
        manifest=manifest,
        prod_python=prod_python,
        force=force,
    )
    _fix_mesh_argv(plan, repo_root=repo_root, run_root=run_root, sample_id=sample_id, prod_python=prod_python)

    lprod_mesh = run_root / "lprod" / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    lprod_mesh_rel = _rel(lprod_mesh, repo_root=repo_root)
    checkpoint_dir = run_root / "lprod" / "checkpoint"
    export_manifest = checkpoint_dir / "checkpoint_export_manifest.json"
    readiness = plan.get("readiness") or {}

    env_a = _prod_subprocess_env_strict(prod_python=prod_python, prod_venv=prod_venv)
    log_env = logs / "stage4_env_probe.log"
    rc_probe, out_probe = _run_env_probe(
        python=prod_python,
        script=STAGE_A_ENV_PROBE,
        env=env_a,
        cwd=repo_root,
    )
    log_env.write_text(out_probe, encoding="utf-8")
    ok_probe, detail_probe = _verify_stage_a_env_probe(out_probe, prod_python=prod_python, prod_venv=prod_venv)
    write_json_atomic(
        logs / "stage4_env_probe.json",
        {"ok": ok_probe and rc_probe == 0, "exit_code": rc_probe, "detail": detail_probe},
    )
    if rc_probe != 0 or not ok_probe:
        _update_manifest(
            manifest_path,
            stage_updates={"stage4_lprod_mesh": "FAIL"},
            terminal_status="FAIL",
            failure_reason=f"stage4_env_probe_failed:{detail_probe}",
        )
        print("Stage 4 FAIL (production env probe)", flush=True)
        return 1

    # Resolve lprod core config (materials from scout; L_prod mesh path)
    try:
        resolve_lprod_core_config(
            repo_root=repo_root,
            run_root=run_root,
            sample_id=sample_id,
            lprod_mesh_rel=lprod_mesh_rel,
            force=force or not plan["skip_mesh"],
            sample_input=sample_input,
        )
    except Exception as exc:
        _update_manifest(
            manifest_path,
            stage_updates={"stage4_lprod_mesh": "FAIL"},
            terminal_status="FAIL",
            failure_reason=f"lprod_resolve_config:{exc}",
        )
        print(f"Stage 4 FAIL (resolve lprod config): {exc}", flush=True)
        return 1

    log_mesh = logs / "stage4_lprod_mesh.log"
    mesh_action = str(plan.get("mesh_action"))
    if plan["skip_mesh"]:
        if _mesh_pass(lprod_mesh):
            _append_log(
                log_mesh,
                f"[{_utc_now()}] [B3_lprod_mesh] status=PASS mesh_action={mesh_action} "
                f"mesh_path={lprod_mesh.resolve()}\n",
            )
            stage4_mesh = "PASS"
        else:
            _append_log(log_mesh, f"[{_utc_now()}] FAIL skip_mesh but mesh missing\n")
            stage4_mesh = "FAIL"
    else:
        try:
            if mesh_action == "reuse_run_tree":
                if not _mesh_pass(lprod_mesh):
                    raise FileNotFoundError(f"reuse_run_tree but mesh missing: {lprod_mesh}")
                _append_log(
                    log_mesh,
                    f"[{_utc_now()}] [B3_lprod_mesh] status=PASS mesh_action=reuse_run_tree "
                    f"mesh_path={lprod_mesh.resolve()}\n",
                )
                rc_mesh = 0
            elif mesh_action == "copy_baseline_mesh":
                _append_log(log_mesh, f"[{_utc_now()}] geometry matches baseline — copy {BASELINE_L_PROD_MESH}\n")
                _install_mesh_from_src(
                    src=BASELINE_L_PROD_MESH,
                    dst=lprod_mesh,
                    sample_id=sample_id,
                )
                rc_mesh = 0 if _mesh_pass(lprod_mesh) else 2
            elif mesh_action == "build_sample_geometry":
                _append_log(log_mesh, f"[{_utc_now()}] build sample-specific L_prod mesh\n")
                rc_mesh = _run_subprocess(
                    plan["argv_mesh_build"],
                    env=env_a,
                    cwd=repo_root,
                    log_path=log_mesh,
                    label="lprod_mesh_build",
                )
                built_conv = mesh_path(MESH_LEVEL, sample_id)
                if rc_mesh == 0 and built_conv.is_file():
                    _install_mesh_from_src(src=built_conv, dst=lprod_mesh, sample_id=sample_id)
            else:
                raise RuntimeError(f"unsupported mesh_action={mesh_action!r}")
            mesh_ok = _mesh_pass(lprod_mesh)
        except Exception as exc:
            rc_mesh = 2
            mesh_ok = False
            _append_log(log_mesh, f"mesh stage exception: {exc}\n")

        stage4_mesh = "PASS" if rc_mesh == 0 and mesh_ok else "FAIL"
        if stage4_mesh == "PASS":
            resolve_lprod_core_config(
                repo_root=repo_root,
                run_root=run_root,
                sample_id=sample_id,
                lprod_mesh_rel=lprod_mesh_rel,
                force=True,
                sample_input=sample_input,
            )

    if stage4_mesh != "PASS":
        blocker = {
            "schema": "m4_lprod_mesh_blocker_v1",
            "sample_id": sample_id,
            "mesh_action": plan.get("mesh_action"),
            "lprod_mesh_status": readiness.get("lprod_mesh_status"),
            "note": (
                "L_prod mesh build failed or mesh missing. "
                "Requires production .venv + FEM/geometry/build_3d_guitar.py (FEM_ALLOW_FOM=1). "
                f"Interface: {MESH_BUILD_REL}"
            ),
        }
        write_json_atomic(run_root / "lprod" / "lprod_mesh_blocker.json", blocker)
        _update_manifest(
            manifest_path,
            stage_updates={"stage4_lprod_mesh": "FAIL"},
            terminal_status="FAIL",
            failure_reason="stage4_lprod_mesh_failed",
        )
        print("Stage 4 L_prod mesh FAIL", flush=True)
        return 1

    _update_manifest(manifest_path, stage_updates={"stage4_lprod_mesh": "PASS"}, terminal_status="RUNNING")
    print("Stage 4 L_prod mesh PASS", flush=True)

    resolved_lprod = run_root / "lprod" / "resolved_core_config.json"
    log_ckpt = logs / "stage4_lprod_checkpoint.log"

    if plan["skip_checkpoint"]:
        _append_log(log_ckpt, f"[{_utc_now()}] reuse PASS lprod checkpoint\n")
        stage4_ckpt = "PASS"
    else:
        rc_a = _run_subprocess(
            plan["argv_stage_a"],
            env=env_a,
            cwd=repo_root,
            log_path=log_ckpt,
            label="lprod_stage_a",
        )
        ok_a, detail_a, verify_detail = _verify_lprod_checkpoint_export(
            checkpoint_dir=checkpoint_dir,
            export_manifest=export_manifest,
            expected_core_config=resolved_lprod,
            repo_root=repo_root,
        )
        stage4_ckpt = "PASS" if rc_a == 0 and ok_a else "FAIL"
        if stage4_ckpt != "PASS":
            _append_log(log_ckpt, f"verify FAIL: {detail_a} rc={rc_a}\n")
        write_json_atomic(
            run_root / "lprod" / "lprod_checkpoint_verify.json",
            {
                "ok": ok_a,
                "detail": detail_a,
                "verify_detail": verify_detail,
                "exit_code": rc_a,
            },
        )

    if stage4_ckpt != "PASS":
        _update_manifest(
            manifest_path,
            stage_updates={"stage4_lprod_export": "FAIL"},
            terminal_status="FAIL",
            failure_reason="stage4_lprod_checkpoint_failed",
        )
        print("Stage 4 L_prod checkpoint FAIL", flush=True)
        return 1

    active_dim = None
    if export_manifest.is_file():
        man = _load_json(export_manifest)
        built_path = checkpoint_dir / "built_metadata.json"
        built = _load_json(built_path) if built_path.is_file() else {}
        active_dim = man.get("active_dimension") or built.get("active_dimension")

    readiness_out = evaluate_lprod_mesh_checkpoint_readiness(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        sample_input=sample_input,
        rel_path_fn=lambda p, **kw: _rel(p, repo_root=repo_root),
    )
    readiness_out["lprod_mesh_status"] = "reusable_existing" if _mesh_pass(lprod_mesh) else readiness_out.get(
        "lprod_mesh_status"
    )
    readiness_out["lprod_checkpoint_status"] = "existing_pass"
    readiness_out["will_execute"] = True
    readiness_out["stage4_completed_utc"] = _utc_now()
    if active_dim is not None:
        readiness_out["active_dimension"] = active_dim
    _log_lprod_region_dof_index_status(
        checkpoint_dir=checkpoint_dir,
        log_path=log_ckpt,
        readiness_out=readiness_out,
    )
    write_json_atomic(run_root / "lprod" / "lprod_mesh_checkpoint_readiness.json", readiness_out)

    _update_manifest(
        manifest_path,
        stage_updates={
            "stage4_lprod_export": "PASS",
            "stage5_workers": "PLANNED_READY",
            "stage6_aggregate": "PLANNED_READY",
        },
        terminal_status=TERMINAL_READY,
        extra={
            "lprod_checkpoint": {
                "active_dimension": active_dim,
                "checkpoint_dir": _rel(checkpoint_dir, repo_root=repo_root),
                "core_config": _rel(resolved_lprod, repo_root=repo_root),
            },
        },
    )

    preview_path = run_root / "pipeline_run_manifest.m4_4_1b_checkpoint_preview.json"
    preview = _load_json(manifest_path)
    preview["will_execute"] = False
    write_json_atomic(preview_path, preview)

    print(f"terminal_status={TERMINAL_READY}")
    print(f"active_dim={active_dim}")
    print("Stage 4 L_prod checkpoint PASS")
    print("no workers executed")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4.4.1b-0 L_prod mesh + checkpoint (Stage 4 only).")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no mesh/checkpoint execution.")
    parser.add_argument("--execute", action="store_true", help="Run Stage 4 mesh + checkpoint export.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export checkpoint even if PASS; does not rebuild existing run-tree L_prod mesh.",
    )
    parser.add_argument("--prod-python", default=DEFAULT_PROD_PYTHON)
    parser.add_argument("--prod-venv", default=DEFAULT_PROD_VENV)
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        print("error: use either --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2

    repo_root = _detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    if args.dry_run:
        return run_dry_run(
            repo_root=repo_root,
            run_root=run_root,
            prod_python=str(args.prod_python),
            force=bool(args.force),
        )
    return run_execute(
        repo_root=repo_root,
        run_root=run_root,
        prod_python=str(args.prod_python),
        prod_venv=str(args.prod_venv),
        force=bool(args.force),
    )


if __name__ == "__main__":
    raise SystemExit(main())
