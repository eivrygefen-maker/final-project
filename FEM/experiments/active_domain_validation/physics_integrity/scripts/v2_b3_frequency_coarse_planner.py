#!/usr/bin/env python3
"""M3.4-pre coarse frequency planner — dry-run planning only (no solver execution)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_petsc_util import write_json_atomic  # noqa: E402

# Mirrored from v2_b3_st_sinvert_solver_lib (avoid PETSc import for dry-run planning).
ACCEPTANCE_FREQ_LO_HZ = 220.0
ACCEPTANCE_FREQ_HI_HZ = 265.0
L_PROD_ST_FULL9_TARGETS_HZ = [221.5, 227.0, 232.5, 238.0, 243.5, 249.0, 254.5, 260.0, 264.0]

PLAN_SCHEMA = "b3_coarse_frequency_plan_v1"
DEFAULT_PLANS_ROOT = (
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/frequency_plans"
)


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
    targets = [float(freq_min)]
    f = float(freq_min) + float(step_hz)
    while f < float(freq_max) - 1.0e-9:
        targets.append(float(f))
        f += float(step_hz)
    if abs(targets[-1] - float(freq_max)) > 1.0e-9:
        targets.append(float(freq_max))
    return targets


def _window_bins(
    freq_min: float, freq_max: float, *, window_width_hz: float
) -> List[Dict[str, Any]]:
    if window_width_hz <= 0:
        raise ValueError("window_width_hz must be positive")
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


def _proposed_regions(
    *,
    freq_min: float,
    freq_max: float,
    full9: Sequence[float],
) -> List[Dict[str, Any]]:
    f9_lo = min(full9)
    f9_hi = max(full9)
    regions: List[Dict[str, Any]] = []

    if freq_min < f9_lo - 1.0e-9:
        regions.append(
            {
                "region_id": "R_below_full9",
                "range_hz": [float(freq_min), float(f9_lo)],
                "density_policy": "unknown_until_coarse_scan",
                "calibration_status": "not_calibrated_yet",
                "recommended_step_hz": None,
                "recommended_window_half_width_hz": None,
                "reason": "Inside planner band but below validated full9 targets",
            }
        )

    regions.append(
        {
            "region_id": "R_full9_validated",
            "range_hz": [float(f9_lo), float(f9_hi)],
            "density_policy": "known_validated_band",
            "calibration_status": "validated_by_m3_pilot_full9",
            "recommended_step_hz": None,
            "recommended_window_half_width_hz": 1.5,
            "reason": (
                "M3 pilot timing 9/9 PASS on m3exec2; historical ~5.5 Hz spacing — not proven optimal"
            ),
        }
    )

    if freq_max > f9_hi + 1.0e-9:
        regions.append(
            {
                "region_id": "R_above_full9",
                "range_hz": [float(f9_hi), float(freq_max)],
                "density_policy": "unknown_until_coarse_scan",
                "calibration_status": "not_calibrated_yet",
                "recommended_step_hz": None,
                "recommended_window_half_width_hz": None,
                "reason": "Inside planner band but above validated full9 targets",
            }
        )

    if freq_min < ACCEPTANCE_FREQ_LO_HZ or freq_max > ACCEPTANCE_FREQ_HI_HZ:
        regions.append(
            {
                "region_id": "R_acceptance_band_note",
                "range_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
                "density_policy": "solver_acceptance_filter",
                "calibration_status": "code_constant_not_calibrated",
                "recommended_step_hz": None,
                "recommended_window_half_width_hz": None,
                "reason": (
                    f"ST acceptance currently filters to {ACCEPTANCE_FREQ_LO_HZ}-"
                    f"{ACCEPTANCE_FREQ_HI_HZ} Hz; wider scan needs separate review"
                ),
            }
        )

    return regions


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
                "reason": (
                    "full9 validated target"
                    if hz_f in L_PROD_ST_FULL9_TARGETS_HZ
                    else "coarse grid proposal (not executed)"
                ),
            }
        )
    return out


def _write_plan_md(path: Path, body: Dict[str, Any]) -> None:
    inp = body.get("input_summary") or {}
    lines = [
        "# Coarse frequency plan (M3.4-pre)",
        "",
        f"- generated_utc: `{body.get('generated_utc')}`",
        f"- schema: `{body.get('schema')}`",
        f"- mode: `{inp.get('mode')}`",
        f"- calibration_status: `{body.get('calibration_status')}`",
        f"- checkpoint_dir: `{inp.get('checkpoint_dir')}`",
        f"- frequency_range_hz: `{inp.get('freq_min_hz')}` – `{inp.get('freq_max_hz')}`",
        f"- coarse_step_hz: `{inp.get('coarse_step_hz')}`",
        f"- coarse_target_count: `{len(body.get('coarse_targets_hz') or [])}`",
        "",
        "## Validated full9 evidence",
        "",
        f"- targets_hz: `{body.get('known_evidence', {}).get('full9_targets_hz')}`",
        f"- note: `{body.get('known_evidence', {}).get('note')}`",
        "",
        "## Proposed regions (thresholds not calibrated)",
        "",
    ]
    for reg in body.get("proposed_regions") or []:
        lines.append(
            f"- **{reg.get('region_id')}** `{reg.get('range_hz')}` — "
            f"`{reg.get('calibration_status')}`: {reg.get('reason')}"
        )
    lines.extend(
        [
            "",
            "## Frequency windows (mode counts pending coarse scan)",
            "",
            "| window | range_hz | mode_count | modes_per_hz | status |",
            "|--------|----------|------------|--------------|--------|",
        ]
    )
    for w in body.get("frequency_windows") or []:
        lines.append(
            f"| {w.get('window_id')} | {w.get('range_hz')} | {w.get('mode_count')} | "
            f"{w.get('modes_per_hz')} | {w.get('calibration_status')} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic notes",
            "",
        ]
    )
    for note in body.get("diagnostic_notes") or []:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Next step if approved (not executed by this tool)",
            "",
            f"```bash",
            str(body.get("next_approved_command_hint") or "(none)"),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_plan(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    reference_result_json: Optional[Path],
    freq_min_hz: float,
    freq_max_hz: float,
    coarse_step_hz: float,
    target_window_half_width_hz: float,
    mode: str,
    absolute_paths: bool,
) -> Dict[str, Any]:
    if mode != "dry-run":
        raise ValueError(f"only mode=dry-run is implemented in M3.4-pre; got {mode!r}")

    coarse_targets = _coarse_targets_hz(freq_min_hz, freq_max_hz, coarse_step_hz)
    window_width = max(coarse_step_hz, 5.0)
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

    diagnostic_notes = [
        "Zone density thresholds are not calibrated yet; regions are hypotheses only.",
        "full9 band is validated M3 pilot evidence, not proof of global spectral coverage.",
        f"Solver acceptance band is {ACCEPTANCE_FREQ_LO_HZ}-{ACCEPTANCE_FREQ_HI_HZ} Hz in "
        "v2_b3_st_sinvert_solver_lib.py.",
        "First coarse solve should use existing Stage B / target_density_experiment on a PASS checkpoint.",
        "Do not overwrite m3exec1/m3exec2 runtime diagnostics.",
    ]

    if not checkpoint_dir.is_dir():
        diagnostic_notes.append(
            f"WARN: checkpoint_dir not found on this host: {checkpoint_dir}"
        )

    execute_hint = (
        "# After explicit approval — solver-only, new output dir, isolated solver-mkl env:\n"
        f"# checkpoint: {ckpt_str}\n"
        "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "v2_b3_checkpoint_target_density_experiment.py \\\n"
        f"  --checkpoint-dir {ckpt_str} \\\n"
        f"  --reference-json {ref_str or '<full9_result.json>'} \\\n"
        f"  --start-hz {freq_min_hz} --stop-hz {freq_max_hz} --spacings-hz {coarse_step_hz}"
    )

    return {
        "schema": PLAN_SCHEMA,
        "generated_utc": _utc_now(),
        "calibration_status": "not_calibrated_yet",
        "will_execute": False,
        "input_summary": {
            "mode": mode,
            "checkpoint_dir": ckpt_str,
            "mesh_level": mesh_level,
            "active_dimension": active_dimension,
            "freq_min_hz": float(freq_min_hz),
            "freq_max_hz": float(freq_max_hz),
            "coarse_step_hz": float(coarse_step_hz),
            "target_window_half_width_hz": float(target_window_half_width_hz),
            "target_set_policy": "custom_coarse_grid_proposal",
            "reference_result_json": ref_str,
            "solver_acceptance_band_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        },
        "known_evidence": {
            "full9_targets_hz": list(L_PROD_ST_FULL9_TARGETS_HZ),
            "full9_span_hz": [min(L_PROD_ST_FULL9_TARGETS_HZ), max(L_PROD_ST_FULL9_TARGETS_HZ)],
            "m3_validated_reference_runs": ["lhs_pilot_001_timing_m3exec2"],
            "note": "Validated timing pilot band; not the whole physical range",
        },
        "coarse_targets_hz": coarse_targets,
        "proposed_regions": _proposed_regions(
            freq_min=freq_min_hz, freq_max=freq_max_hz, full9=L_PROD_ST_FULL9_TARGETS_HZ
        ),
        "frequency_windows": _window_bins(
            freq_min_hz, freq_max_hz, window_width_hz=window_width
        ),
        "stage_b_target_windows": _stage_b_windows(
            coarse_targets, half_width_hz=target_window_half_width_hz
        ),
        "diagnostic_notes": diagnostic_notes,
        "next_approved_command_hint": execute_hint,
        "orchestrator_feed_preview": {
            "future_target_set_name": "coarse_calibrated_v0",
            "targets_hz_field": "coarse_targets_hz",
            "note": "Orchestrator full9 hardcoding may need schema extension before LHS",
        },
    }


def run_planner(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M3.4-pre coarse frequency planner (dry-run only; no solver)."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--freq-min-hz", type=float, default=220.0)
    parser.add_argument("--freq-max-hz", type=float, default=265.0)
    parser.add_argument("--coarse-step-hz", type=float, default=5.0)
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

    body = build_plan(
        repo_root=repo_root,
        checkpoint_dir=checkpoint,
        reference_result_json=ref_path,
        freq_min_hz=float(args.freq_min_hz),
        freq_max_hz=float(args.freq_max_hz),
        coarse_step_hz=float(args.coarse_step_hz),
        target_window_half_width_hz=float(args.target_window_half_width_hz),
        mode=str(args.mode),
        absolute_paths=bool(args.absolute_paths),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(json_path, body)
    _write_plan_md(md_path, body)

    print(f"[B3_freq_planner] wrote {json_path}", flush=True)
    print(f"[B3_freq_planner] wrote {md_path}", flush=True)
    print(
        f"[B3_freq_planner] dry_run targets={len(body['coarse_targets_hz'])} "
        f"calibration_status={body['calibration_status']}",
        flush=True,
    )
    return 0


def main() -> int:
    return run_planner()


if __name__ == "__main__":
    raise SystemExit(main())
