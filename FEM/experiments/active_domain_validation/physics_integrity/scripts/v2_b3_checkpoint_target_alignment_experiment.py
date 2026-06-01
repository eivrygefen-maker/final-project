#!/usr/bin/env python3
"""Solver-only ST shift-center alignment experiment (no FEM/DOLFINx/scipy/gmsh)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_checkpoint_pipeline_lib import default_target_alignment_output_dir  # noqa: E402
from v2_b3_checkpoint_target_density_experiment import (  # noqa: E402
    compare_reference_coverage,
    extract_reference_frequencies_hz,
    parse_spacing_list,
)
from v2_b3_operator_checkpoint_portable import load_operators_with_portable_fallback  # noqa: E402
from v2_b3_petsc_util import mat_shape, write_json_atomic  # noqa: E402
from v2_b3_st_sinvert_solver_lib import (  # noqa: E402
    ACCEPTANCE_FREQ_HI_HZ,
    ACCEPTANCE_FREQ_LO_HZ,
    L_PROD_ST_FULL9_TARGETS_HZ,
    built_from_checkpoint_metadata,
    deduplicate_frequencies_hz,
    mat_global_nnz_used,
    run_checkpoint_st_target,
    safe_float,
    threading_env_snapshot,
    version_snapshot,
)

DEFAULT_START_HZ = 220.0
DEFAULT_STOP_HZ = 264.0
DEFAULT_REFERENCE_ANCHOR_HZ = 221.5
DEFAULT_SPACINGS_HZ = "8,10,12,15,18,20,22"
DEFAULT_TOLERANCE_HZ = 0.1

GridBuilder = Callable[..., List[float]]


def _ensure_stop_included(targets: List[float], stop_hz: float) -> List[float]:
    out = sorted(set(float(t) for t in targets))
    if not out:
        return [float(stop_hz)]
    if abs(out[-1] - float(stop_hz)) > 1.0e-9:
        out.append(float(stop_hz))
    return out


def grid_rounded_aligned(*, start_hz: float, stop_hz: float, spacing_hz: float, **_kw: Any) -> List[float]:
    """Multiples of spacing from the floor anchor at start_hz (e.g. 10 Hz → 220,230,…,264)."""
    anchor = math.floor(float(start_hz) / float(spacing_hz)) * float(spacing_hz)
    if anchor < float(start_hz) - 1.0e-9:
        anchor += float(spacing_hz)
    targets: List[float] = []
    f = anchor
    while f <= float(stop_hz) + 1.0e-9:
        targets.append(float(f))
        f += float(spacing_hz)
    return _ensure_stop_included(targets, stop_hz)


def grid_half_offset(*, start_hz: float, stop_hz: float, spacing_hz: float, **_kw: Any) -> List[float]:
    """Half-spacing offset from rounded anchor (e.g. 10 Hz → 225,235,…,264)."""
    half = float(spacing_hz) / 2.0
    anchor = math.floor(float(start_hz) / float(spacing_hz)) * float(spacing_hz) + half
    if anchor < float(start_hz) - 1.0e-9:
        anchor += float(spacing_hz)
    targets: List[float] = []
    f = anchor
    while f <= float(stop_hz) + 1.0e-9:
        targets.append(float(f))
        f += float(spacing_hz)
    return _ensure_stop_included(targets, stop_hz)


def grid_reference_anchored(
    *,
    start_hz: float,
    stop_hz: float,
    spacing_hz: float,
    reference_anchor_hz: float,
    **_kw: Any,
) -> List[float]:
    """Start at reference anchor (default 221.5) and step by spacing."""
    _ = start_hz
    anchor = float(reference_anchor_hz)
    targets: List[float] = [anchor]
    f = anchor + float(spacing_hz)
    while f < float(stop_hz) - 1.0e-9:
        targets.append(float(f))
        f += float(spacing_hz)
    return _ensure_stop_included(targets, stop_hz)


def grid_reference_subsample(
    *,
    start_hz: float,
    stop_hz: float,
    spacing_hz: float,
    reference_shift_centers_hz: Sequence[float],
    **_kw: Any,
) -> List[float]:
    """Greedy subset of reference shift centers approximating the requested spacing."""
    centers = sorted(
        float(c)
        for c in reference_shift_centers_hz
        if float(start_hz) - 1.0e-9 <= float(c) <= float(stop_hz) + 1.0e-9
    )
    if not centers:
        return grid_reference_anchored(
            start_hz=start_hz,
            stop_hz=stop_hz,
            spacing_hz=spacing_hz,
            reference_anchor_hz=DEFAULT_REFERENCE_ANCHOR_HZ,
        )

    chosen: List[float] = [centers[0]]
    while chosen[-1] < float(stop_hz) - 1.0e-9:
        target_next = chosen[-1] + float(spacing_hz)
        candidates = [c for c in centers if c > chosen[-1] + 1.0e-9]
        if not candidates:
            break
        best = min(candidates, key=lambda c: abs(c - target_next))
        if best in chosen:
            break
        chosen.append(best)
    return _ensure_stop_included(chosen, stop_hz)


ALIGNMENT_BUILDERS: Dict[str, GridBuilder] = {
    "rounded": grid_rounded_aligned,
    "half_offset": grid_half_offset,
    "reference_anchored": grid_reference_anchored,
    "reference_subsample": grid_reference_subsample,
}


def extract_reference_shift_centers_hz(reference_json: Dict[str, Any]) -> List[float]:
    if reference_json.get("targets_hz"):
        return [float(x) for x in reference_json["targets_hz"]]
    rows = reference_json.get("targets") or []
    shift = [float(r["target_frequency_hz"]) for r in rows if r.get("target_frequency_hz") is not None]
    if shift:
        return sorted(shift)
    return list(L_PROD_ST_FULL9_TARGETS_HZ)


def alignment_sort_key(row: Dict[str, Any]) -> Tuple[float, int, int, float]:
    cov = float(row.get("coverage_ratio") or 0.0)
    missed = int(row.get("missed_reference_count") or 9999)
    target_count = int(row.get("target_count") or 9999)
    st_s = float(row.get("total_st_s") or 1.0e18)
    return (-cov, missed, target_count, st_s)


def recommendation_for_row(row: Dict[str, Any], *, best_overall: bool) -> str:
    if best_overall:
        return "best_overall"
    if row.get("coverage_pass"):
        return "full_coverage"
    cov = float(row.get("coverage_ratio") or 0.0)
    if cov >= 0.95:
        return "near_full_coverage"
    if cov >= 0.75:
        return "good_coverage"
    return "low_coverage"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Target-alignment experiment: for each spacing, compare phase/offset grids "
            "against validated reference accepted modes (solver-only, load A/M once)."
        ),
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--start-hz", type=float, default=DEFAULT_START_HZ)
    parser.add_argument("--stop-hz", type=float, default=DEFAULT_STOP_HZ)
    parser.add_argument("--reference-anchor-hz", type=float, default=DEFAULT_REFERENCE_ANCHOR_HZ)
    parser.add_argument("--spacings-hz", default=DEFAULT_SPACINGS_HZ)
    parser.add_argument("--factor-solver", default="mkl_pardiso")
    parser.add_argument("--nev", type=int, default=12)
    parser.add_argument("--ncv", type=int, default=24)
    parser.add_argument("--tolerance-hz", type=float, default=DEFAULT_TOLERANCE_HZ)
    parser.add_argument(
        "--alignments",
        default=",".join(ALIGNMENT_BUILDERS.keys()),
        help="Comma-separated alignment names (default: all).",
    )
    parser.add_argument("--output-dir", help="Default: solver_benchmarks/target_alignment_experiment_<utc>/")
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _write_alignment_md(path: Path, body: Dict[str, Any]) -> None:
    lines = [
        "# Target alignment experiment",
        "",
        f"- generated_utc: `{body.get('generated_utc')}`",
        f"- checkpoint_dir: `{body.get('checkpoint_dir')}`",
        f"- reference_json: `{body.get('reference_json')}`",
        f"- range_hz: `[{body.get('start_hz')}, {body.get('stop_hz')}]`",
        f"- reference_anchor_hz: `{body.get('reference_anchor_hz')}`",
        f"- reference_shift_centers_hz: `{body.get('reference_shift_centers_hz')}`",
        f"- factor_solver: `{body.get('factor_solver')}`",
        f"- nev: `{body.get('nev')}`",
        f"- ncv: `{body.get('ncv')}`",
        f"- tolerance_hz: `{body.get('tolerance_hz')}`",
        f"- reference_unique_accepted_count: `{body.get('reference_unique_accepted_count')}`",
        f"- status: `{body.get('status')}`",
        "",
        "## Best overall",
        "",
        f"```json",
        json.dumps(body.get("best_overall"), indent=2),
        "```",
        "",
        "## Best per spacing",
        "",
        f"```json",
        json.dumps(body.get("best_per_spacing"), indent=2),
        "```",
        "",
        "## Ranking (coverage desc, then fewer missed, fewer targets, lower ST)",
        "",
        "| rank | spacing_hz | alignment | target_count | coverage_ratio | missed | total_st_s | "
        "coverage_pass | recommendation |",
        "|------|------------|-----------|--------------|----------------|--------|------------|"
        "---------------|------------------|",
    ]
    for i, row in enumerate(body.get("ranking") or [], start=1):
        lines.append(
            f"| {i} | {row.get('spacing_hz')} | {row.get('alignment_name')} | "
            f"{row.get('target_count')} | {row.get('coverage_ratio')} | "
            f"{row.get('missed_reference_count')} | {row.get('total_st_s')} | "
            f"{row.get('coverage_pass')} | {row.get('recommendation')} |"
        )
    lines.extend(["", "## Per-case details", ""])
    for row in body.get("cases") or []:
        lines.extend(
            [
                f"### spacing={row.get('spacing_hz')} alignment={row.get('alignment_name')}",
                "",
                f"- generated_targets_hz: `{row.get('generated_targets_hz')}`",
                f"- missed_reference_frequencies_hz: `{row.get('missed_reference_frequencies_hz')}`",
                f"- extra_candidate_frequencies_hz: `{row.get('extra_candidate_frequencies_hz')}`",
                f"- status: `{row.get('status')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_target_alignment_experiment(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    reference_path = Path(args.reference_json).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_target_alignment_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_solver = str(args.factor_solver).strip().lower()
    tol_hz = float(args.tolerance_hz)
    start_hz = float(args.start_hz)
    stop_hz = float(args.stop_hz)
    reference_anchor_hz = float(args.reference_anchor_hz)
    spacings = parse_spacing_list(str(args.spacings_hz))
    alignment_names = [a.strip() for a in str(args.alignments).split(",") if a.strip()]
    for name in alignment_names:
        if name not in ALIGNMENT_BUILDERS:
            raise ValueError(f"unknown alignment={name!r}; expected one of {sorted(ALIGNMENT_BUILDERS)}")

    nev = int(args.nev)
    ncv = int(args.ncv)

    reference_json = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_freqs = extract_reference_frequencies_hz(reference_json, tol_hz=tol_hz)
    reference_shift_centers = extract_reference_shift_centers_hz(reference_json)
    if not reference_freqs:
        body = {
            "status": "FAIL",
            "failure_reason": f"no reference accepted frequencies in {reference_path}",
            "reference_json": str(reference_path),
        }
        write_json_atomic(output_dir / "alignment_result.json", body)
        _write_alignment_md(output_dir / "alignment_result.md", body)
        return 2

    meta_path = checkpoint / "built_metadata.json"
    if not meta_path.is_file():
        body = {
            "status": "FAIL",
            "failure_reason": f"missing built metadata: {meta_path}",
            "checkpoint_dir": str(checkpoint),
        }
        write_json_atomic(output_dir / "alignment_result.json", body)
        _write_alignment_md(output_dir / "alignment_result.md", body)
        return 2

    built_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mesh_level = str(built_meta.get("mesh_level") or "unknown")

    experiment: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_kind": "checkpoint_target_alignment",
        "checkpoint_dir": str(checkpoint),
        "reference_json": str(reference_path),
        "output_dir": str(output_dir),
        "mesh_level": mesh_level,
        "factor_solver": factor_solver,
        "nev": nev,
        "ncv": ncv,
        "start_hz": start_hz,
        "stop_hz": stop_hz,
        "reference_anchor_hz": reference_anchor_hz,
        "reference_shift_centers_hz": reference_shift_centers,
        "spacings_hz": spacings,
        "alignment_names": alignment_names,
        "tolerance_hz": tol_hz,
        "acceptance_interval_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        "reference_unique_accepted_count": len(reference_freqs),
        "reference_unique_accepted_frequencies_hz": reference_freqs,
        "versions": version_snapshot(),
        "threading_env": threading_env_snapshot(),
        "cases": [],
        "ranking": [],
        "best_overall": None,
        "best_per_spacing": {},
        "status": "FAIL",
        "failure_reason": None,
    }

    mats: List[Any] = []
    t_experiment0 = time.perf_counter()
    case_rows: List[Dict[str, Any]] = []

    try:
        A_active, M_active, load_diag = load_operators_with_portable_fallback(checkpoint)
        mats.extend([A_active, M_active])
        built, built_diag = built_from_checkpoint_metadata(
            built_meta,
            A_active=A_active,
            M_active=M_active,
        )
        experiment["checkpoint_load"] = load_diag
        experiment["built_metadata_diag"] = built_diag
        experiment["matrix_contract"] = {
            "A_shape": mat_shape(A_active),
            "M_shape": mat_shape(M_active),
            "A_nnz_used": mat_global_nnz_used(A_active),
            "M_nnz_used": mat_global_nnz_used(M_active),
            "load_path_summary": load_diag.get("load_path_summary"),
        }

        grid_kw = {
            "start_hz": start_hz,
            "stop_hz": stop_hz,
            "reference_anchor_hz": reference_anchor_hz,
            "reference_shift_centers_hz": reference_shift_centers,
        }

        for spacing_hz in spacings:
            for alignment_name in alignment_names:
                builder = ALIGNMENT_BUILDERS[alignment_name]
                generated_targets = builder(spacing_hz=float(spacing_hz), **grid_kw)
                print(
                    f"[B3_target_alignment] spacing={spacing_hz} alignment={alignment_name} "
                    f"targets={len(generated_targets)} list={generated_targets}",
                    flush=True,
                )
                t_case0 = time.perf_counter()
                per_target_rows: List[Dict[str, Any]] = []
                all_accepted: List[float] = []
                total_st = 0.0
                targets_failed = 0
                case_failure_reason: Optional[str] = None

                for ti, target_hz in enumerate(generated_targets):
                    try:
                        row = run_checkpoint_st_target(
                            A_active=A_active,
                            M_active=M_active,
                            built=built,
                            target_hz=float(target_hz),
                            factor_solver=factor_solver,
                            mesh_level=mesh_level,
                            nev=nev,
                            ncv=ncv,
                            target_index=int(ti),
                        )
                    except Exception as exc:
                        row = {
                            "target_index": ti,
                            "target_frequency_hz": float(target_hz),
                            "status": "FAIL",
                            "failure_reason": f"{type(exc).__name__}:{exc}",
                            "accepted_frequencies_hz": [],
                        }
                    per_target_rows.append(row)
                    if row.get("status") == "PASS":
                        all_accepted.extend(list(row.get("accepted_frequencies_hz") or []))
                        total_st += float(row.get("st_total_elapsed_seconds") or 0.0)
                    else:
                        targets_failed += 1
                        if case_failure_reason is None:
                            case_failure_reason = str(row.get("failure_reason") or row.get("status"))

                unique_accepted = deduplicate_frequencies_hz(all_accepted, tol_hz=tol_hz)
                coverage = compare_reference_coverage(
                    reference_freqs,
                    unique_accepted,
                    tol_hz=tol_hz,
                )
                case_wall_s = time.perf_counter() - t_case0
                if targets_failed == len(generated_targets):
                    case_status = "FAIL"
                elif targets_failed > 0:
                    case_status = "PARTIAL"
                else:
                    case_status = "PASS"

                case_row: Dict[str, Any] = {
                    "spacing_hz": float(spacing_hz),
                    "alignment_name": alignment_name,
                    "generated_targets_hz": generated_targets,
                    "target_count": len(generated_targets),
                    "total_wall_s": safe_float(case_wall_s),
                    "total_st_s": safe_float(total_st),
                    "unique_accepted_count": len(unique_accepted),
                    "unique_accepted_frequencies_hz": unique_accepted,
                    "matched_reference_count": coverage["matched_reference_count"],
                    "missed_reference_count": coverage["missed_reference_count"],
                    "missed_reference_frequencies_hz": coverage["missed_reference_frequencies_hz"],
                    "extra_candidate_frequencies_hz": coverage["extra_candidate_frequencies_hz"],
                    "coverage_ratio": coverage["coverage_ratio"],
                    "coverage_pass": coverage["coverage_pass"],
                    "targets_failed": targets_failed,
                    "targets_succeeded": len(generated_targets) - targets_failed,
                    "status": case_status,
                    "failure_reason": case_failure_reason,
                    "per_target": per_target_rows,
                }
                case_rows.append(case_row)
                print(
                    f"[B3_target_alignment] spacing={spacing_hz} alignment={alignment_name} "
                    f"status={case_status} coverage={coverage['matched_reference_count']}/"
                    f"{len(reference_freqs)} wall={case_wall_s:.1f}s",
                    flush=True,
                )

        ranked = sorted(case_rows, key=alignment_sort_key)
        best_overall = ranked[0] if ranked else None
        best_per_spacing: Dict[str, Any] = {}
        for spacing_hz in spacings:
            spacing_cases = [r for r in ranked if float(r["spacing_hz"]) == float(spacing_hz)]
            if spacing_cases:
                best_per_spacing[str(spacing_hz)] = spacing_cases[0]

        ranking_table: List[Dict[str, Any]] = []
        for rank_i, row in enumerate(ranked, start=1):
            is_best = best_overall is not None and row is best_overall
            ranking_table.append(
                {
                    "rank": rank_i,
                    "spacing_hz": row["spacing_hz"],
                    "alignment_name": row["alignment_name"],
                    "target_count": row["target_count"],
                    "coverage_ratio": row["coverage_ratio"],
                    "missed_reference_count": row["missed_reference_count"],
                    "total_st_s": row["total_st_s"],
                    "total_wall_s": row["total_wall_s"],
                    "coverage_pass": row["coverage_pass"],
                    "score": safe_float(
                        float(row.get("coverage_ratio") or 0.0)
                        - 0.001 * int(row.get("missed_reference_count") or 0)
                        - 1.0e-5 * int(row.get("target_count") or 0)
                    ),
                    "recommendation": recommendation_for_row(row, best_overall=is_best),
                }
            )

        experiment["cases"] = case_rows
        experiment["ranking"] = ranking_table
        experiment["best_overall"] = best_overall
        experiment["best_per_spacing"] = best_per_spacing
        experiment["experiment_wall_s"] = safe_float(time.perf_counter() - t_experiment0)

        any_pass = any(bool(r.get("coverage_pass")) for r in case_rows)
        if not case_rows:
            experiment["status"] = "FAIL"
        elif any_pass:
            experiment["status"] = "PASS"
        else:
            experiment["status"] = "PARTIAL"

        write_json_atomic(output_dir / "alignment_result.json", experiment)
        _write_alignment_md(output_dir / "alignment_result.md", experiment)
        print(
            f"[B3_target_alignment] {experiment['status']} cases={len(case_rows)} "
            f"best={best_overall.get('alignment_name') if best_overall else None}@"
            f"{best_overall.get('spacing_hz') if best_overall else None} "
            f"-> {output_dir / 'alignment_result.json'}",
            flush=True,
        )
        return 0 if experiment["status"] in ("PASS", "PARTIAL") else 2
    except Exception as exc:
        experiment["failure_reason"] = f"{type(exc).__name__}:{exc}"
        experiment["experiment_wall_s"] = safe_float(time.perf_counter() - t_experiment0)
        experiment["cases"] = case_rows
        write_json_atomic(output_dir / "alignment_result.json", experiment)
        _write_alignment_md(output_dir / "alignment_result.md", experiment)
        print(f"[B3_target_alignment] FAIL {exc}", flush=True)
        return 2
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    return run_target_alignment_experiment(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
