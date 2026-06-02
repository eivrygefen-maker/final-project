#!/usr/bin/env python3
"""Dry-run resolver: merge pilot JSONL material_delta into per-sample core configs (no FEM execution)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
DEFAULT_OUTPUT_ROOT = PHYSICS_ROOT / "pipeline_runs" / "config_overlays"
MESH_CASE_ID = "baseline_coupled_v2"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _detect_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur.parent != cur:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not detect repository root (missing .git ancestor)")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSONL on line {i}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {i} is not a JSON object")
            rows.append(row)
    return rows


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(text.encode("utf-8"))


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _mesh_file_for_level(mesh_level: str, *, repo_root: Path) -> str:
    rel = (
        Path("FEM")
        / "experiments"
        / "active_domain_validation"
        / "physics_integrity"
        / "v2_mesh_convergence"
        / "mesh"
        / mesh_level
        / f"{MESH_CASE_ID}.msh"
    )
    return rel.as_posix()


def _coerce_json_number(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    return val


def _build_changed_material_values(
    baseline_cfg: Dict[str, Any],
    resolved: Dict[str, Any],
    material_delta: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per-plate field map: baseline vs resolved for each key in material_delta."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    base_mats = baseline_cfg.get("materials") or {}
    res_mats = resolved.get("materials") or {}
    if not isinstance(material_delta, dict):
        return out
    for plate in ("top", "back"):
        plate_delta = material_delta.get(plate)
        if not isinstance(plate_delta, dict) or not plate_delta:
            continue
        base_block = base_mats.get(plate) if isinstance(base_mats.get(plate), dict) else {}
        res_block = res_mats.get(plate) if isinstance(res_mats.get(plate), dict) else {}
        for field in plate_delta:
            base_val = base_block.get(field) if isinstance(base_block, dict) else None
            res_val = res_block.get(field) if isinstance(res_block, dict) else None
            entry: Dict[str, Any] = {
                "baseline": _coerce_json_number(base_val) if isinstance(base_val, (int, float)) else base_val,
                "resolved": _coerce_json_number(res_val) if isinstance(res_val, (int, float)) else res_val,
            }
            out.setdefault(plate, {})[field] = entry
    return out


def _apply_material_delta(cfg: Dict[str, Any], material_delta: Dict[str, Any]) -> Dict[str, str]:
    """Merge material_delta into cfg['materials']; return dotted keys changed."""
    changed: Dict[str, str] = {}
    if not material_delta:
        return changed
    materials = cfg.setdefault("materials", {})
    for plate in ("top", "back"):
        plate_delta = material_delta.get(plate)
        if not isinstance(plate_delta, dict) or not plate_delta:
            continue
        before = copy.deepcopy(materials.get(plate) or {})
        merged = _deep_merge_dict(before if isinstance(before, dict) else {}, plate_delta)
        materials[plate] = merged
        for field, new_val in plate_delta.items():
            old_val = before.get(field) if isinstance(before, dict) else None
            if old_val != new_val:
                changed[f"materials.{plate}.{field}"] = str(new_val)
    return changed


def _expected_density_checks(
    resolved: Dict[str, Any],
    material_delta: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    mats = resolved.get("materials") or {}
    for plate in ("top", "back"):
        plate_delta = material_delta.get(plate) or {}
        if not isinstance(plate_delta, dict):
            continue
        if "density" not in plate_delta:
            continue
        expected = float(plate_delta["density"])
        block = mats.get(plate) or {}
        if not isinstance(block, dict):
            errors.append(f"missing materials.{plate} block in resolved config")
            continue
        if "density" not in block:
            errors.append(f"missing materials.{plate}.density in resolved config")
            continue
        actual = float(block["density"])
        if abs(actual - expected) > 1e-9:
            errors.append(
                f"materials.{plate}.density mismatch: expected {expected}, got {actual}"
            )
    return errors, warnings


def _resolve_one_sample(
    row: Dict[str, Any],
    *,
    base_config_path: Path,
    baseline_cfg: Dict[str, Any],
    baseline_sha256: str,
    output_root: Path,
    repo_root: Path,
    generated_utc: str,
) -> Dict[str, Any]:
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample row missing sample_id")

    payload = row.get("parameter_payload") or {}
    geometry_delta = payload.get("geometry_delta") or {}
    material_delta = payload.get("material_delta") or {}
    requires_mesh = bool(payload.get("requires_mesh_regeneration", False))
    mesh_level = str(row.get("mesh_level") or "L_prod")

    sample_dir = output_root / sample_id
    resolved_path = sample_dir / "resolved_core_config.json"
    overlay_path = sample_dir / "overlay_applied.json"
    readiness_path = sample_dir / "readiness_check.json"

    errors: List[str] = []
    warnings: List[str] = []

    resolved = copy.deepcopy(baseline_cfg)
    changed_fields = _apply_material_delta(resolved, material_delta if isinstance(material_delta, dict) else {})

    mesh_file = _mesh_file_for_level(mesh_level, repo_root=repo_root)
    solver_cfg = resolved.setdefault("solver", {})
    solver_cfg["mesh_file"] = mesh_file
    # Preserve official coupled BC policy when not explicitly set.
    solver_cfg.setdefault("clamp_ribs", False)

    overlay_payload = {
        "schema": "b3_pilot_config_overlay_applied_v1",
        "generated_utc": generated_utc,
        "sample_id": sample_id,
        "base_config_path": _repo_relative(base_config_path, repo_root=repo_root),
        "base_config_sha256": baseline_sha256,
        "mesh_level": mesh_level,
        "mesh_case_id": MESH_CASE_ID,
        "mesh_file": mesh_file,
        "geometry_delta": geometry_delta,
        "material_delta": material_delta,
        "requires_mesh_regeneration": requires_mesh,
        "fields_changed": changed_fields,
    }
    overlay_sha256 = _sha256_json(overlay_payload)

    density_errors, density_warnings = _expected_density_checks(
        resolved,
        material_delta if isinstance(material_delta, dict) else {},
    )
    errors.extend(density_errors)
    warnings.extend(density_warnings)

    if geometry_delta != {}:
        errors.append("geometry_delta must be empty for M2.4.1 material-only pilot")
    if requires_mesh:
        errors.append("requires_mesh_regeneration must be false for M2.4.1")
    if mesh_level != "L_prod":
        warnings.append(f"mesh_level={mesh_level!r} (expected L_prod for first physical pilot)")

    solver_mesh = str((resolved.get("solver") or {}).get("mesh_file") or "")
    if "L_prod" not in solver_mesh.replace("\\", "/"):
        errors.append(f"solver.mesh_file does not reference L_prod: {solver_mesh!r}")
    if MESH_CASE_ID not in solver_mesh.replace("\\", "/"):
        errors.append(f"solver.mesh_file does not reference {MESH_CASE_ID}: {solver_mesh!r}")
    clamp_ribs = bool((resolved.get("solver") or {}).get("clamp_ribs", False))
    if clamp_ribs:
        errors.append("solver.clamp_ribs must be false for official coupled pilot BC policy")

    changed_material_values = _build_changed_material_values(
        baseline_cfg,
        resolved,
        material_delta if isinstance(material_delta, dict) else {},
    )

    if not material_delta:
        errors.append("material_delta is empty")
    elif not changed_fields:
        errors.append("material_delta present but no material fields changed vs baseline copy")
    elif not changed_material_values:
        errors.append("changed_material_values is empty despite non-empty material_delta")

    geom_empty = isinstance(geometry_delta, dict) and len(geometry_delta) == 0
    mat_nonempty = isinstance(material_delta, dict) and len(material_delta) > 0
    physical_lhs_ready = bool(geom_empty and mat_nonempty and not requires_mesh)

    mats = resolved.get("materials") or {}
    effective_materials = {
        "top.density": float((mats.get("top") or {}).get("density", float("nan")))
        if isinstance(mats.get("top"), dict)
        else None,
        "back.density": float((mats.get("back") or {}).get("density", float("nan")))
        if isinstance(mats.get("back"), dict)
        else None,
    }

    write_json_atomic(resolved_path, resolved)
    resolved_sha256 = _sha256_file(resolved_path)
    write_json_atomic(overlay_path, {**overlay_payload, "overlay_payload_sha256": overlay_sha256})

    post_baseline_sha256 = _sha256_file(base_config_path)
    if post_baseline_sha256 != baseline_sha256:
        errors.append("canonical baseline config was modified on disk during resolver run")

    status = "PASS" if not errors else "FAIL"
    readiness = {
        "schema": "b3_pilot_config_readiness_check_v1",
        "generated_utc": generated_utc,
        "sample_id": sample_id,
        "status": status,
        "base_config_path": _repo_relative(base_config_path, repo_root=repo_root),
        "resolved_config_path": _repo_relative(resolved_path, repo_root=repo_root),
        "overlay_applied_path": _repo_relative(overlay_path, repo_root=repo_root),
        "effective_materials": effective_materials,
        "changed_material_values": changed_material_values,
        "material_fields_changed": changed_fields,
        "sha256": {
            "baseline_config": baseline_sha256,
            "baseline_config_after_run": post_baseline_sha256,
            "resolved_config": resolved_sha256,
            "overlay_payload": overlay_sha256,
        },
        "mesh_regeneration_required": requires_mesh,
        "mesh_level": mesh_level,
        "mesh_file": mesh_file,
        "physical_lhs_ready": physical_lhs_ready,
        "warnings": warnings,
        "errors": errors,
    }
    write_json_atomic(readiness_path, readiness)

    return {
        "sample_id": sample_id,
        "status": status,
        "resolved_config_path": str(resolved_path),
        "readiness_path": str(readiness_path),
        "effective_materials": effective_materials,
        "changed_material_values": changed_material_values,
        "warnings": warnings,
        "errors": errors,
    }


def run_resolve(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve per-sample core configs from pilot JSONL (dry-run, no FEM execution).",
    )
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_BASE_CONFIG),
        help="Canonical baseline core config (read-only).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root for per-sample overlay directories.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing overlay outputs.")
    args = parser.parse_args(argv)

    repo_root = _detect_repo_root(SCRIPT_DIR)
    samples_path = Path(args.samples_jsonl).expanduser()
    if not samples_path.is_absolute():
        samples_path = (repo_root / samples_path).resolve()
    base_config_path = Path(args.base_config).expanduser()
    if not base_config_path.is_absolute():
        base_config_path = (repo_root / base_config_path).resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    if not samples_path.is_file():
        raise SystemExit(f"samples JSONL not found: {samples_path}")
    if not base_config_path.is_file():
        raise SystemExit(f"base config not found: {base_config_path}")

    output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(samples_path)
    if not rows:
        raise SystemExit(f"no samples in {samples_path}")

    for row in rows:
        sid = str(row.get("sample_id") or "").strip()
        sample_dir = output_root / sid
        for name in ("resolved_core_config.json", "overlay_applied.json", "readiness_check.json"):
            p = sample_dir / name
            if p.exists() and not args.force:
                raise SystemExit(f"output exists: {p} (use --force)")

    baseline_sha256 = _sha256_file(base_config_path)
    baseline_cfg = json.loads(base_config_path.read_text(encoding="utf-8"))
    generated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
            _resolve_one_sample(
                row,
                base_config_path=base_config_path,
                baseline_cfg=baseline_cfg,
                baseline_sha256=baseline_sha256,
                output_root=output_root,
                repo_root=repo_root,
                generated_utc=generated_utc,
            )
        )

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = len(results) - n_pass
    print(f"[B3_resolve_pilot_config] samples={len(results)} pass={n_pass} fail={n_fail}", flush=True)
    for r in results:
        em = r["effective_materials"]
        cmv = r.get("changed_material_values") or {}
        print(
            f"[B3_resolve_pilot_config] {r['sample_id']}: status={r['status']} "
            f"top.density={em.get('top.density')} back.density={em.get('back.density')} "
            f"changed={list(cmv.keys()) or 'none'}",
            flush=True,
        )
        for w in r.get("warnings") or []:
            print(f"[B3_resolve_pilot_config]   WARN: {w}", flush=True)
        for err in r.get("errors") or []:
            print(f"[B3_resolve_pilot_config]   ERROR: {err}", flush=True)
    print("[B3_resolve_pilot_config] no Stage A/B/C execution performed", flush=True)
    return 0 if n_fail == 0 else 2


def main() -> int:
    return run_resolve()


if __name__ == "__main__":
    raise SystemExit(main())
