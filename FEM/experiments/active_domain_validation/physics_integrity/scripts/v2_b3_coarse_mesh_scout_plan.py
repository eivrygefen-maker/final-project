#!/usr/bin/env python3
"""M3.4 coarse-mesh modal-density scout planner — inspection / dry-run only."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PHYSICS_ROOT = SCRIPT_DIR.parent
MANIFEST_PATH = PHYSICS_ROOT / "configs" / "v2_mesh_convergence_manifest.json"
CONV_MESH = PHYSICS_ROOT / "v2_mesh_convergence" / "mesh"
CONV_DIAG = PHYSICS_ROOT / "v2_mesh_convergence" / "diagnostics"
DEFAULT_CASE_ID = "baseline_coupled_v2"
DEFAULT_RUN_ID = "scout_l_scout_coarse_m34"
SCOUT_LEVEL_ID = "L_scout_coarse"
PLAN_SCHEMA = "b3_coarse_mesh_scout_plan_v1_1"
SCOUT_MESH_BUILD_SCRIPT = (
    "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
    "run_v2_B3_scout_coarse_mesh_build.py"
)

PLANNING_BAND_LO_HZ = 60.0
PLANNING_BAND_HI_HZ = 550.0
DEFAULT_DISCOVERY_SPACING_HZ = 15.0
L_PROD_ACTIVE_DIM_REFERENCE = 316_017

from v2_mesh_convergence_mesh import effective_controls_from_level_def  # noqa: E402

WRONG_DIRECTION_RUN_ID = "target_density_discovery_60_550_step15_m3exec2"
STAGE_B_OUTPUT_STEM = "target_density_discovery_60_550_step15_L_scout_coarse_m34"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _detect_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur.parent != cur:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not detect repository root (missing .git ancestor)")


def _format_path(path: Path, *, repo_root: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _controls_for_profile(
    profile: str,
    lc_scale: float,
) -> Dict[str, float]:
    """Mirror v2_mesh_convergence_mesh.py effective_controls_m computation."""
    if profile == "validation":
        base = {
            "wood_surface_size_m": 0.014,
            "wood_thickness_size_m": 0.003,
            "air_threshold_size_min_m": 0.009,
            "air_threshold_size_max_m": 0.04,
        }
    else:
        base = {
            "wood_surface_size_m": 0.007,
            "wood_thickness_size_m": 0.001,
            "air_threshold_size_min_m": 0.004,
            "air_threshold_size_max_m": 0.05,
        }
    s = float(lc_scale)
    return {k: float(v) * s for k, v in base.items()}


def _uniform_lc_scale_table(
    profile: str,
    scales: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sc in scales:
        ctrl = _controls_for_profile(profile, sc)
        rows.append(
            {
                "lc_scale": sc,
                "wood_thickness_mm": round(ctrl["wood_thickness_size_m"] * 1000, 2),
                "wood_surface_mm": round(ctrl["wood_surface_size_m"] * 1000, 2),
                "air_min_mm": round(ctrl["air_threshold_size_min_m"] * 1000, 2),
            }
        )
    return rows


def _estimate_scout_active_dim(
    prod_active_dim: int,
    prod_controls: Dict[str, float],
    scout_controls: Dict[str, float],
) -> Dict[str, Any]:
    """Rough DOF scaling ~ (mean LC)^-3 between prod and scout."""
    prod_vals = [
        prod_controls["wood_thickness_size_m"],
        prod_controls["wood_surface_size_m"],
        prod_controls["air_threshold_size_min_m"],
    ]
    scout_vals = [
        scout_controls["wood_thickness_size_m"],
        scout_controls["wood_surface_size_m"],
        scout_controls["air_threshold_size_min_m"],
    ]
    mean_prod = sum(prod_vals) / len(prod_vals)
    mean_scout = sum(scout_vals) / len(scout_vals)
    ratio = mean_scout / mean_prod if mean_prod > 0 else 1.0
    est = int(round(prod_active_dim / (ratio**3)))
    lo = int(round(prod_active_dim / ((ratio * 1.15) ** 3)))
    hi = int(round(prod_active_dim / ((ratio * 0.85) ** 3)))
    return {
        "method": "mean_characteristic_length_cubed_scaling",
        "mean_lc_ratio_scout_over_prod": round(ratio, 4),
        "point_estimate": est,
        "rough_band": [min(lo, hi), max(lo, hi)],
        "prod_active_dim_reference": prod_active_dim,
        "note": "Measure active_dimension from Stage A export after mesh build; estimate is planning-only.",
    }


def _mesh_level_report(
    level_id: str,
    level_def: Dict[str, Any],
    *,
    case_id: str,
    repo_root: Path,
    absolute_paths: bool,
) -> Dict[str, Any]:
    build_env = dict(level_def.get("build_env") or {})
    lc_scale = float(level_def.get("lc_scale", 1.0))
    profile = "validation" if "FEM_VALIDATION_MESH" in build_env else "fom"
    controls = effective_controls_from_level_def(level_def)
    explicit = level_def.get("explicit_controls_m")
    if isinstance(explicit, dict) and explicit:
        controls_source = "effective_controls_from_level_def_with_explicit_overrides"
    else:
        controls_source = f"effective_controls_from_level_def_{profile}_times_lc_scale"

    msh = CONV_MESH / level_id / f"{case_id}.msh"
    audit = CONV_MESH / level_id / f"{case_id}_mesh_audit.json"
    summary = CONV_MESH / level_id / f"{case_id}_mesh_build_summary.json"

    row: Dict[str, Any] = {
        "mesh_level": level_id,
        "label": level_def.get("label"),
        "build_env": build_env,
        "lc_scale": lc_scale,
        "profile": profile,
        "controls_source": controls_source,
        "effective_controls_mm": {
            k: round(v * 1000, 3) for k, v in controls.items() if k.endswith("_m")
        },
        "mesh_path": _format_path(msh, repo_root=repo_root, absolute_paths=absolute_paths),
        "mesh_exists": msh.is_file(),
        "mesh_audit_path": _format_path(audit, repo_root=repo_root, absolute_paths=absolute_paths),
        "mesh_audit_exists": audit.is_file(),
        "mesh_build_summary_exists": summary.is_file(),
        "solver_smoke_test_only": bool(level_def.get("solver_smoke_test_only")),
        "not_authorized_for_final_physics_validation": bool(
            level_def.get("not_authorized_for_final_physics_validation")
        ),
        "purpose": level_def.get("purpose"),
        "production_physics": level_def.get("production_physics"),
        "final_results": level_def.get("final_results"),
        "modal_density_scout_only": level_def.get("modal_density_scout_only"),
    }
    if audit.is_file():
        try:
            aud = json.loads(audit.read_text(encoding="utf-8"))
            row["audit_n_nodes"] = aud.get("n_nodes")
            row["audit_n_tetrahedra"] = aud.get("n_tetrahedra")
            if aud.get("effective_controls_m"):
                row["audit_effective_controls_mm"] = {
                    k: round(float(v) * 1000, 3)
                    for k, v in aud["effective_controls_m"].items()
                    if str(k).endswith("_m")
                }
        except (json.JSONDecodeError, OSError):
            row["audit_read_error"] = True
    return row


def _checkpoint_glob_status(
    *,
    repo_root: Path,
    absolute_paths: bool,
) -> List[Dict[str, Any]]:
    diag = CONV_DIAG
    patterns = [
        "st_worker_scaling_L_prod_*",
        "st_worker_scaling_L_scout_coarse_*",
        "st_worker_scaling_L_dev_*",
        "st_worker_scaling_L_mid_*",
    ]
    found: List[Path] = []
    if diag.is_dir():
        for pat in patterns:
            found.extend(sorted(diag.glob(pat)))
    rows: List[Dict[str, Any]] = []
    for p in found[:40]:
        manifest = p / "checkpoint_export_manifest.json"
        rows.append(
            {
                "checkpoint_dir": _format_path(p, repo_root=repo_root, absolute_paths=absolute_paths),
                "export_manifest_exists": manifest.is_file(),
                "name": p.name,
            }
        )
    if len(found) > 40:
        rows.append({"truncated": True, "total_matching_dirs": len(found)})
    return rows


def _stage_a_command_preview(
    *,
    mesh_level: str,
    run_id: str,
    core_config: str,
    output_dir: Path,
) -> str:
    return (
        "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        f"v2_b3_checkpoint_export.py --mesh-level {mesh_level} "
        "--B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off "
        f'--core-config "{core_config}" '
        f'--output-dir "{output_dir.as_posix()}"'
    )


def _mesh_build_command_preview() -> str:
    return f"python {SCOUT_MESH_BUILD_SCRIPT}"


def _stage_b_discovery_preview(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    spacing_hz: float,
) -> str:
    half_w = spacing_hz / 2.0
    return (
        "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "v2_b3_checkpoint_target_density_experiment.py "
        f'--checkpoint-dir "{checkpoint_dir.as_posix()}" '
        f"--start-hz {PLANNING_BAND_LO_HZ} --stop-hz {PLANNING_BAND_HI_HZ} "
        f"--spacings-hz {spacing_hz} "
        f"--B3-discovery-mode --discovery-band-hz {PLANNING_BAND_LO_HZ} {PLANNING_BAND_HI_HZ} "
        f"--target-window-half-width-hz {half_w} "
        f'--output-dir "{output_dir.as_posix()}"'
    )


def build_scout_plan(
    *,
    repo_root: Path,
    run_id: str,
    case_id: str,
    core_config: Optional[Path],
    absolute_paths: bool,
) -> Dict[str, Any]:
    manifest = _load_manifest()
    levels = manifest.get("mesh_levels") or {}
    l_prod_def = levels.get("L_prod") or {}
    l_dev_coarse_def = levels.get("L_dev_coarse") or {}
    scout_def = levels.get(SCOUT_LEVEL_ID) or {}

    prod_controls = effective_controls_from_level_def(l_prod_def) if l_prod_def else _controls_for_profile(
        "fom", 1.0
    )
    scout_controls = effective_controls_from_level_def(scout_def) if scout_def else {}

    core_cfg = core_config or (
        PHYSICS_ROOT
        / "pipeline_runs"
        / "config_overlays"
        / "lhs_pilot_001_timing"
        / "resolved_core_config.json"
    )
    core_cfg_rel = _format_path(core_cfg, repo_root=repo_root, absolute_paths=False)

    scout_msh = CONV_MESH / SCOUT_LEVEL_ID / f"{case_id}.msh"
    ckpt_dir = CONV_DIAG / f"st_worker_scaling_{SCOUT_LEVEL_ID}_{run_id}"
    stage_b_out = CONV_DIAG / "solver_benchmarks" / STAGE_B_OUTPUT_STEM

    mesh_levels_report = [
        _mesh_level_report(lid, ldef, case_id=case_id, repo_root=repo_root, absolute_paths=absolute_paths)
        for lid, ldef in sorted(levels.items())
    ]

    l_prod_mesh = CONV_MESH / "L_prod" / f"{case_id}.msh"
    warnings: List[str] = []
    if not l_prod_mesh.is_file():
        warnings.append(f"L_prod mesh file not found on this host: {l_prod_mesh}")
    if not scout_msh.is_file():
        warnings.append(f"Scout mesh not built yet: {scout_msh}")
    if ckpt_dir.is_dir():
        warnings.append(f"Checkpoint dir already exists (preview only): {ckpt_dir}")

    scout_recognized = SCOUT_LEVEL_ID in levels
    try:
        from v2_b3_checkpoint_export import ALLOWED_MESH_LEVELS  # type: ignore

        stage_a_allowed = sorted(ALLOWED_MESH_LEVELS)
    except Exception:
        stage_a_allowed = ["L_mid", "L_dev_dense", "L_prod", SCOUT_LEVEL_ID]

    stage_a_allows_scout = SCOUT_LEVEL_ID in stage_a_allowed
    if not scout_recognized:
        warnings.append(f"Manifest missing mesh_levels.{SCOUT_LEVEL_ID}")
    if not stage_a_allows_scout:
        warnings.append(f"Stage A allowlist missing {SCOUT_LEVEL_ID}")

    half_width = DEFAULT_DISCOVERY_SPACING_HZ / 2.0
    plan: Dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "generated_at_utc": _utc_now(),
        "will_execute": False,
        "strategy": "coarse_fem_mesh_modal_density_scout",
        "supersedes_wrong_direction": {
            "run_id": WRONG_DIRECTION_RUN_ID,
            "reason": "Wide-band discovery on full L_prod checkpoint — wrong primary scout strategy; discard as zone evidence.",
        },
        "planning_band_hz": [PLANNING_BAND_LO_HZ, PLANNING_BAND_HI_HZ],
        "verified_l_prod_sizing": {
            "source_manifest": _format_path(MANIFEST_PATH, repo_root=repo_root, absolute_paths=absolute_paths),
            "source_builder": "FEM/geometry/build_3d_guitar.py (FEM_ALLOW_FOM=1, FEM_MESH_LC_SCALE=1)",
            "manifest_l_prod_source_controls_mm": {
                k: round(float(v) * 1000, 3)
                for k, v in (manifest.get("l_prod_source") or {}).get("controls", {}).items()
                if str(k).endswith("_m")
            },
            "effective_controls_mm": {
                k: round(v * 1000, 3) for k, v in prod_controls.items()
            },
            "active_dim_reference_m3exec2": L_PROD_ACTIVE_DIM_REFERENCE,
        },
        "run_id": run_id,
        "mesh_level": SCOUT_LEVEL_ID,
        "L_scout_coarse_recognized_in_manifest": scout_recognized,
        "stage_a_allows_L_scout_coarse": stage_a_allows_scout,
        "explicit_controls_wired": True,
        "explicit_controls_env": "FEM_MESH_EXPLICIT_CONTROLS_JSON",
        "paths": {
            "mesh": _format_path(scout_msh, repo_root=repo_root, absolute_paths=absolute_paths),
            "mesh_exists": scout_msh.is_file(),
            "checkpoint_dir": _format_path(ckpt_dir, repo_root=repo_root, absolute_paths=absolute_paths),
            "checkpoint_dir_exists": ckpt_dir.is_dir(),
            "stage_b_discovery_output_dir": _format_path(
                stage_b_out, repo_root=repo_root, absolute_paths=absolute_paths
            ),
        },
        "implementation_m34_1": {
            "manifest_entry": scout_recognized,
            "build_hook": "v2_mesh_convergence_mesh.build_level_mesh + FEM_MESH_EXPLICIT_CONTROLS_JSON",
            "builder_hook": "FEM/geometry/build_3d_guitar.py",
            "mesh_build_script": SCOUT_MESH_BUILD_SCRIPT,
            "stage_a_allowlist": stage_a_allows_scout,
        },
        "proposed_scout_level": {
            "recommended_id": SCOUT_LEVEL_ID,
            "reuse_L_dev_coarse": False,
            "reuse_rationale": (
                "L_dev_coarse uses FEM_VALIDATION_MESH with lc_scale=2.0 "
                f"(~{_controls_for_profile('validation', 2.0)['wood_surface_size_m']*1000:.0f} mm wood shell), "
                "not FOM production geometry."
            ),
            "manifest_level_def": scout_def if scout_recognized else None,
            "explicit_controls_m": scout_def.get("explicit_controls_m") if scout_def else None,
            "effective_controls_m": scout_controls,
            "effective_controls_mm": {
                k: round(v * 1000, 3) for k, v in scout_controls.items() if k.endswith("_m")
            },
            "operator_target_mm_mapping": {
                "plate_thickness_detail": {
                    "requested_mm": 3.0,
                    "field": "wood_thickness_size_m",
                    "resolved_mm": round(scout_controls.get("wood_thickness_size_m", 0) * 1000, 3),
                },
                "wood_shell": {
                    "requested_mm": 8.5,
                    "field": "wood_surface_size_m",
                    "resolved_mm": round(scout_controls.get("wood_surface_size_m", 0) * 1000, 3),
                },
                "air_graded_min": {
                    "requested_mm": 11.0,
                    "field": "air_threshold_size_min_m",
                    "resolved_mm": round(scout_controls.get("air_threshold_size_min_m", 0) * 1000, 3),
                },
            },
        },
        "uniform_lc_scale_compromise_table_fom": _uniform_lc_scale_table(
            "fom", [1.0, 1.21, 1.5, 2.0, 3.0]
        ),
        "L_dev_coarse_effective_mm": {
            k: round(v * 1000, 3)
            for k, v in _controls_for_profile(
                "validation", float(l_dev_coarse_def.get("lc_scale", 2.0))
            ).items()
        },
        "active_dimension_estimate": _estimate_scout_active_dim(
            L_PROD_ACTIVE_DIM_REFERENCE, prod_controls, scout_controls
        ),
        "mesh_levels": mesh_levels_report,
        "checkpoints_on_host": _checkpoint_glob_status(
            repo_root=repo_root, absolute_paths=absolute_paths
        ),
        "stage_a_allowed_mesh_levels": stage_a_allowed,
        "core_config": {
            "path": _format_path(core_cfg, repo_root=repo_root, absolute_paths=absolute_paths),
            "exists": core_cfg.is_file(),
        },
        "command_previews": {
            "mesh_build": _mesh_build_command_preview(),
            "stage_a": _stage_a_command_preview(
                mesh_level=SCOUT_LEVEL_ID,
                run_id=run_id,
                core_config=core_cfg_rel,
                output_dir=ckpt_dir,
            ),
            "stage_b_discovery": _stage_b_discovery_preview(
                checkpoint_dir=ckpt_dir,
                output_dir=stage_b_out,
                spacing_hz=DEFAULT_DISCOVERY_SPACING_HZ,
            ),
        },
        "discovery_half_width_hz": {
            "applied": half_width,
            "rule": "spacing_hz / 2 for 15 Hz grid (touching windows)",
        },
        "prerequisites_before_execution": [
            "M3.4.1 code wiring complete (manifest, explicit controls, Stage A allowlist)",
            "Approve and run mesh build on VM: run_v2_B3_scout_coarse_mesh_build.py",
            "Stage A export on production venv (after mesh exists)",
            "Stage B discovery on solver-mkl venv (after checkpoint exists)",
        ],
        "warnings": warnings,
        "documentation": (
            "FEM/experiments/active_domain_validation/physics_integrity/docs/"
            "B3_M3_4_COARSE_MESH_MODAL_DENSITY_SCOUT_PLAN.md"
        ),
    }
    return plan


def _write_markdown(plan: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Coarse-mesh modal-density scout plan (dry-run)",
        "",
        f"- **Generated:** `{plan.get('generated_at_utc')}`",
        f"- **will_execute:** `{plan.get('will_execute')}`",
        f"- **Strategy:** {plan.get('strategy')}",
        "",
        "## L_prod verified sizing (mm)",
        "",
    ]
    v = plan.get("verified_l_prod_sizing") or {}
    for k, val in (v.get("effective_controls_mm") or {}).items():
        lines.append(f"- `{k}`: **{val}**")
    lines.append(f"- **active_dim (m3exec2 ref):** {v.get('active_dim_reference_m3exec2')}")
    lines.append("")
    lines.append("## Proposed L_scout_coarse (mm)")
    lines.append("")
    prop = plan.get("proposed_scout_level") or {}
    for k, val in (prop.get("explicit_controls_mm") or {}).items():
        lines.append(f"- `{k}`: **{val}**")
    lines.append(f"- **reuse L_dev_coarse:** {prop.get('reuse_L_dev_coarse')}")
    lines.append("")
    est = plan.get("active_dimension_estimate") or {}
    lines.append("## Active dimension estimate")
    lines.append("")
    lines.append(f"- Point estimate: **{est.get('point_estimate')}**")
    lines.append(f"- Rough band: **{est.get('rough_band')}**")
    lines.append("")
    if plan.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in plan["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    cmds = plan.get("command_previews") or {}
    lines.append("## Command previews (do not run without approval)")
    lines.append("")
    lines.append("### Mesh build")
    lines.append("```bash")
    lines.append(cmds.get("mesh_build", ""))
    lines.append("```")
    lines.append("")
    lines.append("### Stage A")
    lines.append("```bash")
    lines.append(cmds.get("stage_a", ""))
    lines.append("```")
    lines.append("")
    lines.append("### Stage B discovery")
    lines.append("```bash")
    lines.append(cmds.get("stage_b_discovery", ""))
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run coarse-mesh modal-density scout inspection (no mesh/solver execution).",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Suffix for preview checkpoint dir (default: scout_l_scout_coarse_m34)",
    )
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--core-config", type=Path, default=None, help="Overlay core config for Stage A preview")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PHYSICS_ROOT / "pipeline_runs" / "specs" / "scout_plans" / "preview",
        help="Directory for scout_plan.json and scout_plan.md",
    )
    parser.add_argument("--absolute-paths", action="store_true", help="Emit absolute paths in JSON")
    args = parser.parse_args(argv)

    repo_root = _detect_repo_root(SCRIPT_DIR)
    out_dir: Path = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_scout_plan(
        repo_root=repo_root,
        run_id=str(args.run_id),
        case_id=str(args.case_id),
        core_config=args.core_config.resolve() if args.core_config else None,
        absolute_paths=bool(args.absolute_paths),
    )

    json_path = out_dir / "scout_plan.json"
    md_path = out_dir / "scout_plan.md"
    json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    _write_markdown(plan, md_path)

    print(f"[scout_plan] will_execute={plan['will_execute']}", flush=True)
    print(f"[scout_plan] wrote {json_path}", flush=True)
    print(f"[scout_plan] wrote {md_path}", flush=True)
    for w in plan.get("warnings") or []:
        print(f"[scout_plan] WARN: {w}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
