#!/usr/bin/env python3
"""M3.4-pre coarse frequency planner — dry-run planning only (no solver execution)."""
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

from v2_b3_petsc_util import write_json_atomic  # noqa: E402

# Mirrored from v2_b3_st_sinvert_solver_lib (avoid PETSc import for dry-run planning).
ACCEPTANCE_FREQ_LO_HZ = 220.0
ACCEPTANCE_FREQ_HI_HZ = 265.0
L_PROD_ST_FULL9_TARGETS_HZ = [221.5, 227.0, 232.5, 238.0, 243.5, 249.0, 254.5, 260.0, 264.0]

PLANNING_BAND_LO_HZ = 60.0
PLANNING_BAND_HI_HZ = 550.0
DEFAULT_DISCOVERY_HALF_WIDTH_HZ = 7.5
FULL9_REF_LO_HZ = min(L_PROD_ST_FULL9_TARGETS_HZ)
FULL9_REF_HI_HZ = max(L_PROD_ST_FULL9_TARGETS_HZ)

PLAN_SCHEMA = "b3_coarse_frequency_plan_v2"
DEFAULT_PLANS_ROOT = (
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/frequency_plans"
)

# Conservative per-target ST wall time on L_prod ~316k DOF (m3exec2 class); not measured exactly.
EST_PER_TARGET_WALL_S_LO = 120.0
EST_PER_TARGET_WALL_S_HI = 480.0


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


def _coarse_targets_hz(freq_min: float, freq_max: float, step_hz: float) -> List[float]:
    if step_hz <= 0:
        raise ValueError(f"coarse_step_hz must be positive, got {step_hz}")
    if freq_max < freq_min:
        raise ValueError(f"freq_max_hz ({freq_max}) must be >= freq_min_hz ({freq_min})")
    targets: List[float] = [float(freq_min)]
    f = float(freq_min) + float(step_hz)
    while f < float(freq_max) - 1.0e-9:
        targets.append(float(f))
        f += float(step_hz)
    if abs(targets[-1] - float(freq_max)) > 1.0e-9:
        targets.append(float(freq_max))
    return targets


def _adaptive_targets_v0(freq_min: float, freq_max: float) -> List[float]:
    """Planning-only adaptive grid: coarser outside full9 neighborhood."""
    segments = [
        (float(freq_min), 200.0, 20.0),
        (200.0, 280.0, 10.0),
        (280.0, float(freq_max), 20.0),
    ]
    out: List[float] = []
    for lo, hi, step in segments:
        lo_eff = max(lo, float(freq_min))
        hi_eff = min(hi, float(freq_max))
        if hi_eff <= lo_eff + 1.0e-9:
            continue
        seg = _coarse_targets_hz(lo_eff, hi_eff, step)
        for t in seg:
            if not out or abs(t - out[-1]) > 1.0e-9:
                out.append(t)
    if not out:
        return _coarse_targets_hz(freq_min, freq_max, 15.0)
    if abs(out[0] - float(freq_min)) > 1.0e-9:
        out.insert(0, float(freq_min))
    if abs(out[-1] - float(freq_max)) > 1.0e-9:
        out.append(float(freq_max))
    return out


def _spacing_alternatives(freq_min: float, freq_max: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in (10.0, 15.0, 20.0):
        targets = _coarse_targets_hz(freq_min, freq_max, step)
        rows.append(
            {
                "policy": f"uniform_{int(step)}hz",
                "coarse_step_hz": step,
                "coarse_target_count": len(targets),
                "calibration_status": "not_calibrated_yet",
                "notes": (
                    "moderate cost"
                    if step == 10.0
                    else "recommended first pass" if step == 15.0 else "coarser discovery"
                ),
            }
        )
    adaptive = _adaptive_targets_v0(freq_min, freq_max)
    rows.append(
        {
            "policy": "adaptive_v0",
            "coarse_step_hz": None,
            "coarse_target_count": len(adaptive),
            "segment_steps_hz": [20.0, 10.0, 20.0],
            "calibration_status": "not_calibrated_yet",
            "notes": "20 Hz wings + 10 Hz around 200–280 Hz; planning hypothesis only",
        }
    )
    return rows


def _target_executable_legacy(target_hz: float) -> bool:
    return ACCEPTANCE_FREQ_LO_HZ <= float(target_hz) <= ACCEPTANCE_FREQ_HI_HZ


def _target_executable_discovery(
    target_hz: float,
    *,
    band_lo: float,
    band_hi: float,
) -> bool:
    return band_lo <= float(target_hz) <= band_hi


def _executable_feasibility(
    *,
    freq_min: float,
    freq_max: float,
    coarse_targets: Sequence[float],
    assume_gate_a_discovery: bool = False,
    discovery_band_hz: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    band = discovery_band_hz or (float(freq_min), float(freq_max))

    if assume_gate_a_discovery:
        exec_targets = [
            t for t in coarse_targets if _target_executable_discovery(t, band_lo=band[0], band_hi=band[1])
        ]
        blocked_targets = [t for t in coarse_targets if t not in exec_targets]
        overall = "executable_with_discovery_mode_opt_in"
        legacy_note = "Gate A implemented; use --B3-discovery-mode on Stage B / density experiment."
    else:
        exec_targets = [t for t in coarse_targets if _target_executable_legacy(t)]
        blocked_targets = [t for t in coarse_targets if not _target_executable_legacy(t)]
        planning_outside_acceptance = (
            freq_min < ACCEPTANCE_FREQ_LO_HZ - 1.0e-9 or freq_max > ACCEPTANCE_FREQ_HI_HZ + 1.0e-9
        )
        if planning_outside_acceptance:
            overall = "requires_acceptance_band/general_target_set_support_before_execution"
        else:
            overall = "executable_with_current_stage_b_acceptance_filter"
        legacy_note = (
            "Default Stage B (no discovery flags) uses legacy acceptance [220, 265] Hz only."
        )

    executable_now_count = len(exec_targets)
    blocked_count = len(blocked_targets)

    blocked_slices: List[List[float]] = []
    if not assume_gate_a_discovery:
        if freq_min < ACCEPTANCE_FREQ_LO_HZ:
            blocked_slices.append([float(freq_min), float(ACCEPTANCE_FREQ_LO_HZ)])
        if freq_max > ACCEPTANCE_FREQ_HI_HZ:
            blocked_slices.append([float(ACCEPTANCE_FREQ_HI_HZ), float(freq_max)])

    return {
        "planning_band_hz": [float(freq_min), float(freq_max)],
        "solver_acceptance_band_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        "discovery_band_hz_assumed": list(band) if assume_gate_a_discovery else None,
        "executable_now_band_hz": list(band) if assume_gate_a_discovery else [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        "blocked_until_acceptance_extension_hz": blocked_slices,
        "executable_now_count": executable_now_count,
        "blocked_count": blocked_count,
        "executable_now_targets_hz": exec_targets,
        "blocked_targets_hz": blocked_targets,
        "coarse_target_count_total": len(coarse_targets),
        "coarse_target_count_executable_now": executable_now_count,
        "coarse_target_count_blocked_now": blocked_count,
        "assume_gate_a_discovery": bool(assume_gate_a_discovery),
        "overall_execution_status": overall,
        "code_basis": legacy_note,
    }


def _cost_and_parallel_estimate(
    *,
    coarse_targets: Sequence[float],
    active_dimension: Optional[int],
    mesh_level: Optional[str],
) -> Dict[str, Any]:
    n = len(coarse_targets)
    wall_lo = n * EST_PER_TARGET_WALL_S_LO
    wall_hi = n * EST_PER_TARGET_WALL_S_HI
    return {
        "coarse_target_count": n,
        "estimated_stage_b_wall_seconds_range": [wall_lo, wall_hi],
        "estimated_stage_b_wall_minutes_range": [wall_lo / 60.0, wall_hi / 60.0],
        "assumptions": {
            "per_target_wall_seconds_range": [EST_PER_TARGET_WALL_S_LO, EST_PER_TARGET_WALL_S_HI],
            "basis": "Conservative extrapolation from L_prod m3exec2 class (~316k active DOF); "
            "measure from first approved slice.",
            "active_dimension": active_dimension,
            "mesh_level": mesh_level,
        },
        "parallel_execution_guidance": {
            "dry_run": "Safe to run in parallel with other work (no solver).",
            "actual_solver_scan": (
                "Run alone on VM: MKL/PETSc ST solves are CPU/RAM heavy on L_prod; "
                "do not assume a few minutes for wide-band scans; avoid concurrent solver benchmarks."
            ),
            "recommended_concurrency": "exclusive_solver_slot",
        },
    }


def _planning_regions(*, freq_min: float, freq_max: float) -> List[Dict[str, Any]]:
    """Placeholder region table — not calibrated until coarse-scan data exists."""
    regions: List[Dict[str, Any]] = []

    if freq_min < FULL9_REF_LO_HZ - 1.0e-9:
        regions.append(
            {
                "region_id": "R_low_60_220",
                "range_hz": [float(freq_min), float(FULL9_REF_LO_HZ)],
                "density_policy": "unknown_until_coarse_scan",
                "calibration_status": "not_calibrated_yet",
                "recommended_step_hz": None,
                "reason": "Low/mid guitar band; requires acceptance-band extension before Stage B discovery",
            }
        )

    regions.append(
        {
            "region_id": "R_full9_validated_220_265",
            "range_hz": [float(FULL9_REF_LO_HZ), float(FULL9_REF_HI_HZ)],
            "density_policy": "validated_reference_slice",
            "calibration_status": "validated_by_m3_pilot_full9",
            "recommended_step_hz": None,
            "reason": (
                "M3 m3exec2 timing 9/9 PASS; overlaps solver acceptance band; "
                "not the full 60–550 Hz planning range"
            ),
        }
    )

    if freq_max > FULL9_REF_HI_HZ + 1.0e-9:
        regions.append(
            {
                "region_id": "R_mid_high_265_550",
                "range_hz": [float(FULL9_REF_HI_HZ), float(freq_max)],
                "density_policy": "unknown_until_coarse_scan",
                "calibration_status": "not_calibrated_yet",
                "recommended_step_hz": None,
                "reason": "Upper guitar/modal band; requires acceptance-band extension before Stage B discovery",
            }
        )

    return regions


def _window_bins(
    freq_min: float, freq_max: float, *, window_width_hz: float
) -> List[Dict[str, Any]]:
    bins: List[Dict[str, Any]] = []
    lo = float(freq_min)
    idx = 0
    while lo < float(freq_max) - 1.0e-9:
        hi = min(lo + float(window_width_hz), float(freq_max))
        bins.append(
            {
                "window_id": f"W{idx:02d}",
                "range_hz": [round(lo, 4), round(hi, 4)],
                "mode_count": None,
                "modes_per_hz": None,
                "calibration_status": "not_calibrated_yet",
            }
        )
        lo = hi
        idx += 1
    return bins


def _stage_b_windows(
    targets_hz: Sequence[float], *, half_width_hz: float
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for hz in targets_hz:
        hz_f = float(hz)
        out.append(
            {
                "target_hz": hz_f,
                "window_hz": [hz_f - half_width_hz, hz_f + half_width_hz],
                "half_width_hz": float(half_width_hz),
                "executable_with_legacy_acceptance": _target_executable_legacy(hz_f),
                "reason": (
                    "full9 validated reference target"
                    if hz_f in L_PROD_ST_FULL9_TARGETS_HZ
                    else "coarse grid proposal (not executed)"
                ),
            }
        )
    return out


def _gate_a_cli_flags(
    *,
    band_lo: float,
    band_hi: float,
    half_width_hz: float,
) -> List[str]:
    return [
        "--B3-discovery-mode",
        "--discovery-band-hz",
        str(float(band_lo)),
        str(float(band_hi)),
        "--target-window-half-width-hz",
        str(float(half_width_hz)),
    ]


def _recommended_next_step(
    *,
    feasibility: Dict[str, Any],
    spacing_rec: Dict[str, Any],
    ckpt_str: str,
    ref_str: Optional[str],
    target_window_half_width_hz: float,
) -> Dict[str, Any]:
    step = spacing_rec.get("recommended_first_pass_hz")
    ref = ref_str or (
        "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/"
        "diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_"
        "lhs_pilot_001_timing_m3exec2/result.json"
    )
    discovery_flags = " ".join(
        _gate_a_cli_flags(
            band_lo=PLANNING_BAND_LO_HZ,
            band_hi=PLANNING_BAND_HI_HZ,
            half_width_hz=target_window_half_width_hz,
        )
    )
    cmd = (
        "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "v2_b3_checkpoint_target_density_experiment.py \\\n"
        f"  --checkpoint-dir {ckpt_str} \\\n"
        f"  --reference-json {ref} \\\n"
        f"  --start-hz {PLANNING_BAND_LO_HZ} --stop-hz {PLANNING_BAND_HI_HZ} "
        f"--spacings-hz {step} \\\n"
        f"  {discovery_flags}"
    )
    return {
        "summary": (
            "Gate A discovery mode is implemented (opt-in). "
            "1) Approve coarse scan plan (15 Hz uniform, exclusive VM). "
            "2) Run density experiment with --B3-discovery-mode on a new output dir. "
            "3) Post-process unique_accepted_frequencies_hz per window; calibrate zones."
        ),
        "blocked_until": None,
        "gate_a_status": "implemented_option_c",
        "suggested_spacing_hz": step,
        "suggested_discovery_band_hz": [PLANNING_BAND_LO_HZ, PLANNING_BAND_HI_HZ],
        "suggested_target_window_half_width_hz": float(target_window_half_width_hz),
        "gate_a_cli_flags": _gate_a_cli_flags(
            band_lo=PLANNING_BAND_LO_HZ,
            band_hi=PLANNING_BAND_HI_HZ,
            half_width_hz=target_window_half_width_hz,
        ),
        "suggested_command_after_approval": cmd,
        "default_full9_unchanged": (
            "python .../v2_b3_checkpoint_solve.py --checkpoint-dir <ckpt> "
            "--target-set full9  # no discovery flags → legacy [220,265] acceptance"
        ),
    }


def _write_plan_md(path: Path, body: Dict[str, Any]) -> None:
    inp = body.get("input_summary") or {}
    feas = body.get("executable_feasibility") or {}
    cost = body.get("cost_estimate") or {}
    lines = [
        "# Coarse frequency plan (M3.4-pre)",
        "",
        f"- schema: `{body.get('schema')}`",
        f"- mode: `{body.get('mode')}`",
        f"- will_execute: `{body.get('will_execute')}`",
        f"- calibration_status: `{body.get('calibration_status')}`",
        f"- zone_policy_status: `{body.get('zone_policy_status')}`",
        f"- freq_range_hz: `{inp.get('freq_range_hz')}`",
        f"- coarse_step_hz: `{inp.get('coarse_step_hz')}`",
        f"- coarse_target_count: `{len(body.get('coarse_targets_hz') or [])}`",
        f"- execution_status: `{feas.get('overall_execution_status')}`",
        "",
        "## Regions (placeholder — not calibrated)",
        "",
    ]
    for reg in body.get("regions") or []:
        lines.append(
            f"- **{reg.get('region_id')}** `{reg.get('range_hz')}` — "
            f"`{reg.get('calibration_status')}`: {reg.get('reason')}"
        )
    lines.extend(
        [
            "",
            "## Spacing alternatives (planning)",
            "",
            "| policy | step_hz | targets | note |",
            "|--------|---------|---------|------|",
        ]
    )
    for row in body.get("spacing_alternatives") or []:
        lines.append(
            f"| {row.get('policy')} | {row.get('coarse_step_hz')} | "
            f"{row.get('coarse_target_count')} | {row.get('notes')} |"
        )
    lines.extend(
        [
            "",
            "## Cost / parallel guidance",
            "",
            f"- estimated_wall_minutes: `{cost.get('estimated_stage_b_wall_minutes_range')}`",
            f"- parallel (solver): `{cost.get('parallel_execution_guidance', {}).get('actual_solver_scan')}`",
            "",
            "## Recommended next step",
            "",
            str((body.get("recommended_next_step") or {}).get("summary")),
            "",
        ]
    )
    for note in body.get("diagnostic_notes") or []:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_plan(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    reference_result_json: Optional[Path],
    freq_min_hz: float,
    freq_max_hz: float,
    coarse_step_hz: Optional[float],
    spacing_policy: str,
    target_window_half_width_hz: float,
    mode: str,
    absolute_paths: bool,
    assume_gate_a_discovery: bool,
) -> Dict[str, Any]:
    if mode != "dry-run":
        raise ValueError(f"only mode=dry-run is implemented in M3.4-pre; got {mode!r}")

    spacing_alts = _spacing_alternatives(freq_min_hz, freq_max_hz)
    if spacing_policy == "adaptive_v0":
        coarse_targets = _adaptive_targets_v0(freq_min_hz, freq_max_hz)
        step_used: Optional[float] = None
    else:
        step_used = float(coarse_step_hz if coarse_step_hz is not None else 15.0)
        coarse_targets = _coarse_targets_hz(freq_min_hz, freq_max_hz, step_used)

    window_width = max(step_used or 15.0, 10.0)
    ckpt_str = _format_path(checkpoint_dir, repo_root=repo_root, absolute_paths=absolute_paths)

    mesh_level = None
    active_dimension = None
    built_meta = checkpoint_dir / "built_metadata.json"
    if built_meta.is_file():
        try:
            meta = json.loads(built_meta.read_text(encoding="utf-8"))
            mesh_level = meta.get("mesh_level")
            active_dimension = meta.get("active_dimension")
        except json.JSONDecodeError:
            pass

    ref_str = None
    if reference_result_json is not None:
        ref_str = _format_path(reference_result_json, repo_root=repo_root, absolute_paths=absolute_paths)

    feasibility = _executable_feasibility(
        freq_min=freq_min_hz,
        freq_max=freq_max_hz,
        coarse_targets=coarse_targets,
        assume_gate_a_discovery=assume_gate_a_discovery,
        discovery_band_hz=(freq_min_hz, freq_max_hz),
    )
    cost = _cost_and_parallel_estimate(
        coarse_targets=coarse_targets,
        active_dimension=active_dimension,
        mesh_level=mesh_level,
    )
    regions = _planning_regions(freq_min=freq_min_hz, freq_max=freq_max_hz)

    spacing_rec = {
        "recommended_first_pass_hz": 15.0,
        "recommended_policy": "uniform_15hz",
        "rationale": (
            "15 Hz over 60–550 Hz yields ~34 targets — coarse enough for discovery, "
            "not assuming 5 Hz is correct band-wide. 10 Hz (~50 targets) if first pass "
            "shows gaps; 20 Hz (~26 targets) if runtime too high. "
            "Blocked until acceptance band supports full planning range."
        ),
        "alternatives": spacing_alts,
    }

    diagnostic_notes = [
        "Planning band 60–550 Hz is the guitar/modal exploration target; 220–265 Hz is validated full9 reference only.",
        "Zone density thresholds are not_calibrated_yet; regions are placeholders.",
        "Gate A: opt-in --B3-discovery-mode uses discovery_band + per-target window (Option C).",
        "Default full9 / timing runs without discovery flags keep legacy [220, 265] acceptance.",
        "Do not overwrite m3exec1/m3exec2 runtime diagnostics.",
    ]
    if not checkpoint_dir.is_dir():
        diagnostic_notes.append(f"WARN: checkpoint_dir not found on this host: {checkpoint_dir}")

    recommended = _recommended_next_step(
        feasibility=feasibility,
        spacing_rec=spacing_rec,
        ckpt_str=ckpt_str,
        ref_str=ref_str,
        target_window_half_width_hz=target_window_half_width_hz,
    )

    return {
        "gate_a_implementation": "option_c_opt_in_discovery_mode",
        "schema": PLAN_SCHEMA,
        "generated_utc": _utc_now(),
        "mode": mode,
        "will_execute": False,
        "calibration_status": "not_calibrated_yet",
        "zone_policy_status": "not_calibrated_yet",
        "input_summary": {
            "checkpoint_dir": ckpt_str,
            "mesh_level": mesh_level,
            "active_dimension": active_dimension,
            "freq_range_hz": [float(freq_min_hz), float(freq_max_hz)],
            "freq_min_hz": float(freq_min_hz),
            "freq_max_hz": float(freq_max_hz),
            "coarse_step_hz": step_used,
            "spacing_policy": spacing_policy,
            "target_window_half_width_hz": float(target_window_half_width_hz),
            "reference_result_json": ref_str,
            "solver_acceptance_band_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
            "validated_full9_reference_band_hz": [FULL9_REF_LO_HZ, FULL9_REF_HI_HZ],
        },
        "known_evidence": {
            "full9_targets_hz": list(L_PROD_ST_FULL9_TARGETS_HZ),
            "full9_span_hz": [FULL9_REF_LO_HZ, FULL9_REF_HI_HZ],
            "m3_validated_reference_runs": ["lhs_pilot_001_timing_m3exec2"],
            "note": "full9 is a validated reference slice inside the 60–550 Hz planning band",
        },
        "coarse_targets_hz": coarse_targets,
        "coarse_target_count": len(coarse_targets),
        "regions": regions,
        "spacing_recommendation": spacing_rec,
        "executable_feasibility": feasibility,
        "cost_estimate": cost,
        "frequency_windows": _window_bins(freq_min_hz, freq_max_hz, window_width_hz=window_width),
        "stage_b_target_windows": _stage_b_windows(
            coarse_targets, half_width_hz=target_window_half_width_hz
        ),
        "diagnostic_notes": diagnostic_notes,
        "recommended_next_step": recommended,
        "orchestrator_feed_preview": {
            "future_target_set_name": "coarse_calibrated_v0",
            "note": "Planner feeds orchestrator via approved targets_hz / JSONL after calibration",
        },
    }


def run_planner(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M3.4-pre coarse frequency planner (dry-run only; no solver)."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--freq-min-hz", type=float, default=PLANNING_BAND_LO_HZ)
    parser.add_argument("--freq-max-hz", type=float, default=PLANNING_BAND_HI_HZ)
    parser.add_argument(
        "--coarse-step-hz",
        type=float,
        default=15.0,
        help="Uniform coarse spacing (default 15 Hz for 60–550 band). Ignored if --spacing-policy adaptive_v0.",
    )
    parser.add_argument(
        "--spacing-policy",
        choices=("uniform", "adaptive_v0"),
        default="uniform",
        help="uniform uses --coarse-step-hz; adaptive_v0 uses 20/10/20 Hz segments.",
    )
    parser.add_argument("--target-window-half-width-hz", type=float, default=1.5)
    parser.add_argument("--reference-result-json", help="Optional validated result.json")
    parser.add_argument("--mode", choices=("dry-run", "execute"), default="dry-run")
    parser.add_argument(
        "--output-dir",
        default=f"{DEFAULT_PLANS_ROOT}/m3_4_pre_coarse_demo",
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite plan outputs")
    parser.add_argument(
        "--assume-gate-a-discovery",
        action="store_true",
        help="Feasibility counts assume --B3-discovery-mode over planning band (post Gate A).",
    )
    args = parser.parse_args(argv)

    if args.mode == "execute":
        raise SystemExit(
            "mode=execute is not implemented in M3.4-pre; use target_density_experiment after approval"
        )

    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else _detect_repo_root(SCRIPT_DIR)
    )
    checkpoint = Path(args.checkpoint_dir).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (repo_root / checkpoint).resolve()
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    ref_path = None
    if args.reference_result_json:
        ref_path = Path(args.reference_result_json).expanduser()
        if not ref_path.is_absolute():
            ref_path = (repo_root / ref_path).resolve()

    json_path = out_dir / "coarse_frequency_plan.json"
    md_path = out_dir / "coarse_frequency_plan.md"
    if not args.force and (json_path.exists() or md_path.exists()):
        raise SystemExit(f"output exists: {out_dir} (use --force)")

    step = None if args.spacing_policy == "adaptive_v0" else float(args.coarse_step_hz)
    body = build_plan(
        repo_root=repo_root,
        checkpoint_dir=checkpoint,
        reference_result_json=ref_path,
        freq_min_hz=float(args.freq_min_hz),
        freq_max_hz=float(args.freq_max_hz),
        coarse_step_hz=step,
        spacing_policy=str(args.spacing_policy),
        target_window_half_width_hz=float(args.target_window_half_width_hz),
        mode=str(args.mode),
        absolute_paths=bool(args.absolute_paths),
        assume_gate_a_discovery=bool(args.assume_gate_a_discovery),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(json_path, body)
    _write_plan_md(md_path, body)
    preview_path = out_dir / "command_preview_gate_a.txt"
    preview_path.write_text(
        str((body.get("recommended_next_step") or {}).get("suggested_command_after_approval") or "")
        + "\n",
        encoding="utf-8",
    )

    feas = body["executable_feasibility"]
    print(f"[B3_freq_planner] wrote {json_path}", flush=True)
    print(f"[B3_freq_planner] wrote {md_path}", flush=True)
    print(
        f"[B3_freq_planner] mode={body['mode']} will_execute={body['will_execute']} "
        f"targets={body['coarse_target_count']} step={body['input_summary']['coarse_step_hz']} "
        f"range={body['input_summary']['freq_range_hz']} "
        f"execution_status={feas['overall_execution_status']}",
        flush=True,
    )
    return 0


def main() -> int:
    return run_planner()


if __name__ == "__main__":
    raise SystemExit(main())
