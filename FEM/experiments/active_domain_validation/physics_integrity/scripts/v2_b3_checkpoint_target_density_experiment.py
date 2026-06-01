#!/usr/bin/env python3
"""Solver-only target-spacing density experiment (no FEM/DOLFINx/scipy/gmsh)."""
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

from v2_b3_checkpoint_pipeline_lib import default_target_density_output_dir  # noqa: E402
from v2_b3_operator_checkpoint_portable import load_operators_with_portable_fallback  # noqa: E402
from v2_b3_petsc_util import mat_shape, write_json_atomic  # noqa: E402
from v2_b3_st_sinvert_solver_lib import (  # noqa: E402
    ACCEPTANCE_FREQ_HI_HZ,
    ACCEPTANCE_FREQ_LO_HZ,
    built_from_checkpoint_metadata,
    deduplicate_frequencies_hz,
    extract_summary_view,
    mat_global_nnz_used,
    parse_hz_list,
    run_checkpoint_st_target,
    safe_float,
    threading_env_snapshot,
    version_snapshot,
)

DEFAULT_START_HZ = 221.5
DEFAULT_STOP_HZ = 264.0
DEFAULT_SPACINGS_HZ = "6,8,10,15,20"
DEFAULT_TOLERANCE_HZ = 0.1


def generate_spaced_targets_hz(
    start_hz: float,
    stop_hz: float,
    spacing_hz: float,
) -> List[float]:
    """Build target list from start to stop; always include both endpoints."""
    if spacing_hz <= 0.0:
        raise ValueError(f"spacing_hz must be positive, got {spacing_hz}")
    if stop_hz < start_hz:
        raise ValueError(f"stop_hz ({stop_hz}) must be >= start_hz ({start_hz})")

    targets: List[float] = [float(start_hz)]
    f = float(start_hz) + float(spacing_hz)
    while f < float(stop_hz) - 1.0e-9:
        targets.append(float(f))
        f += float(spacing_hz)
    if not targets or abs(targets[-1] - float(stop_hz)) > 1.0e-9:
        targets.append(float(stop_hz))
    return targets


def parse_spacing_list(text: str) -> List[float]:
    spacings = parse_hz_list(str(text))
    return sorted(set(spacings))


def extract_reference_frequencies_hz(reference_json: Dict[str, Any], *, tol_hz: float) -> List[float]:
    summary = extract_summary_view(reference_json)
    if summary.get("unique_accepted_hz"):
        return deduplicate_frequencies_hz(list(summary["unique_accepted_hz"]), tol_hz=tol_hz)
    agg = reference_json.get("aggregate") or {}
    if agg.get("unique_accepted_frequencies_hz"):
        return deduplicate_frequencies_hz(list(agg["unique_accepted_frequencies_hz"]), tol_hz=tol_hz)
    if reference_json.get("accepted_frequencies_hz"):
        return deduplicate_frequencies_hz(list(reference_json["accepted_frequencies_hz"]), tol_hz=tol_hz)
    freqs: List[float] = []
    for row in reference_json.get("targets") or []:
        freqs.extend(float(x) for x in (row.get("accepted_frequencies_hz") or []))
    return deduplicate_frequencies_hz(freqs, tol_hz=tol_hz)


def compare_reference_coverage(
    reference_hz: Sequence[float],
    candidate_hz: Sequence[float],
    *,
    tol_hz: float,
) -> Dict[str, Any]:
    ref_sorted = sorted(float(x) for x in reference_hz)
    cand_sorted = sorted(float(x) for x in candidate_hz)
    used_cand = [False] * len(cand_sorted)
    matched_reference: List[float] = []
    missed_reference: List[float] = []

    for rf in ref_sorted:
        match_idx: Optional[int] = None
        for i, cf in enumerate(cand_sorted):
            if used_cand[i]:
                continue
            if abs(rf - cf) <= tol_hz:
                match_idx = i
                break
        if match_idx is not None:
            used_cand[match_idx] = True
            matched_reference.append(rf)
        else:
            missed_reference.append(rf)

    extra_candidate = [cf for i, cf in enumerate(cand_sorted) if not used_cand[i]]
    ref_count = len(ref_sorted)
    matched_count = len(matched_reference)
    coverage_ratio = (matched_count / ref_count) if ref_count else 1.0
    return {
        "matched_reference_count": matched_count,
        "missed_reference_count": len(missed_reference),
        "missed_reference_frequencies_hz": missed_reference,
        "extra_candidate_frequencies_hz": extra_candidate,
        "coverage_ratio": safe_float(coverage_ratio),
        "coverage_pass": bool(len(missed_reference) == 0),
        "matched_reference_frequencies_hz": matched_reference,
    }


def load_previous_density_by_spacing(previous_body: Dict[str, Any]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for row in previous_body.get("spacings") or []:
        if row.get("spacing_hz") is None:
            continue
        out[float(row["spacing_hz"])] = row
    return out


def compare_spacing_to_previous(
    *,
    spacing_hz: float,
    current_coverage: Dict[str, Any],
    previous_row: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if previous_row is None:
        return None
    prev_matched = int(previous_row.get("matched_reference_count") or 0)
    cur_matched = int(current_coverage.get("matched_reference_count") or 0)
    prev_ratio = previous_row.get("coverage_ratio")
    cur_ratio = current_coverage.get("coverage_ratio")
    return {
        "previous_matched_reference_count": prev_matched,
        "previous_missed_reference_count": int(previous_row.get("missed_reference_count") or 0),
        "previous_coverage_ratio": prev_ratio,
        "previous_coverage_pass": bool(previous_row.get("coverage_pass")),
        "matched_reference_delta": cur_matched - prev_matched,
        "coverage_ratio_delta": (
            safe_float(float(cur_ratio) - float(prev_ratio))
            if cur_ratio is not None and prev_ratio is not None
            else None
        ),
        "coverage_improved": bool(cur_matched > prev_matched),
        "coverage_pass_improved": bool(
            current_coverage.get("coverage_pass") and not previous_row.get("coverage_pass")
        ),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overnight target-density experiment: sweep ST target spacings against a "
            "validated reference accepted-frequency set (solver-only, load A/M once)."
        ),
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--reference-json", required=True, help="Validated full9 result.json reference.")
    parser.add_argument("--start-hz", type=float, default=DEFAULT_START_HZ)
    parser.add_argument("--stop-hz", type=float, default=DEFAULT_STOP_HZ)
    parser.add_argument(
        "--spacings-hz",
        default=DEFAULT_SPACINGS_HZ,
        help=f"Comma-separated spacing values in Hz (default: {DEFAULT_SPACINGS_HZ}).",
    )
    parser.add_argument("--factor-solver", default="mkl_pardiso")
    parser.add_argument("--nev", type=int, default=12)
    parser.add_argument("--ncv", type=int, default=24)
    parser.add_argument("--tolerance-hz", type=float, default=DEFAULT_TOLERANCE_HZ)
    parser.add_argument(
        "--output-dir",
        help="Default: solver_benchmarks/target_density_experiment[_nevN_ncvM]_<utc>/",
    )
    parser.add_argument(
        "--previous-density-result-json",
        help="Optional prior density_result.json (e.g. nev12/ncv24) for per-spacing coverage comparison.",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _write_density_md(path: Path, body: Dict[str, Any]) -> None:
    lines = [
        "# Target density experiment",
        "",
        f"- generated_utc: `{body.get('generated_utc')}`",
        f"- checkpoint_dir: `{body.get('checkpoint_dir')}`",
        f"- reference_json: `{body.get('reference_json')}`",
        f"- nev: `{body.get('nev')}`",
        f"- ncv: `{body.get('ncv')}`",
        f"- previous_density_result_json: `{body.get('previous_density_result_json')}`",
        f"- factor_solver: `{body.get('factor_solver')}`",
        f"- start_hz: `{body.get('start_hz')}`",
        f"- stop_hz: `{body.get('stop_hz')}`",
        f"- tolerance_hz: `{body.get('tolerance_hz')}`",
        f"- reference_unique_accepted_count: `{body.get('reference_unique_accepted_count')}`",
        f"- reference_unique_accepted_frequencies_hz: `{body.get('reference_unique_accepted_frequencies_hz')}`",
        f"- experiment_status: `{body.get('status')}`",
        f"- sparsest_coverage_pass_spacing_hz: `{body.get('sparsest_coverage_pass_spacing_hz')}`",
        "",
        "## Spacing summary",
        "",
        "| spacing_hz | target_count | total_wall_s | total_st_s | unique_accepted | "
        "matched_ref | missed_ref | coverage_ratio | coverage_pass | "
        "prev_matched | delta | improved | status |",
        "|------------|--------------|--------------|------------|-----------------|"
        "-------------|------------|----------------|---------------|"
        "-------------|-------|----------|--------|",
    ]
    for row in body.get("spacings") or []:
        prev_cmp = row.get("previous_comparison") or {}
        lines.append(
            f"| {row.get('spacing_hz')} | {row.get('target_count')} | "
            f"{row.get('total_wall_s')} | {row.get('total_st_s')} | "
            f"{row.get('unique_accepted_count')} | {row.get('matched_reference_count')} | "
            f"{row.get('missed_reference_count')} | {row.get('coverage_ratio')} | "
            f"{row.get('coverage_pass')} | "
            f"{prev_cmp.get('previous_matched_reference_count', '')} | "
            f"{prev_cmp.get('matched_reference_delta', '')} | "
            f"{prev_cmp.get('coverage_improved', '')} | "
            f"{row.get('status')} |"
        )
    lines.extend(["", "## Per spacing details", ""])
    for row in body.get("spacings") or []:
        lines.extend(
            [
                f"### spacing {row.get('spacing_hz')} Hz",
                "",
                f"- generated_targets_hz: `{row.get('generated_targets_hz')}`",
                f"- missed_reference_frequencies_hz: `{row.get('missed_reference_frequencies_hz')}`",
                f"- extra_candidate_frequencies_hz: `{row.get('extra_candidate_frequencies_hz')}`",
                f"- targets_failed: `{row.get('targets_failed')}`",
                f"- failure_reason: `{row.get('failure_reason')}`",
                f"- previous_comparison: `{row.get('previous_comparison')}`",
                "",
            ]
        )
    prev_summary = body.get("previous_density_comparison")
    if prev_summary:
        lines.extend(
            [
                "## Previous run comparison",
                "",
                f"- previous_density_result_json: `{prev_summary.get('previous_density_result_json')}`",
                f"- previous_nev: `{prev_summary.get('previous_nev')}`",
                f"- previous_ncv: `{prev_summary.get('previous_ncv')}`",
                f"- spacings_compared: `{prev_summary.get('spacings_compared')}`",
                f"- spacings_improved: `{prev_summary.get('spacings_improved')}`",
                f"- best_matched_reference_delta: `{prev_summary.get('best_matched_reference_delta')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_target_density_experiment(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    reference_path = Path(args.reference_json).expanduser().resolve()
    nev = int(args.nev)
    ncv = int(args.ncv)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_target_density_output_dir(nev=nev, ncv=ncv)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_solver = str(args.factor_solver).strip().lower()
    tol_hz = float(args.tolerance_hz)
    spacings = parse_spacing_list(str(args.spacings_hz))
    start_hz = float(args.start_hz)
    stop_hz = float(args.stop_hz)

    previous_path: Optional[Path] = None
    previous_body: Optional[Dict[str, Any]] = None
    previous_by_spacing: Dict[float, Dict[str, Any]] = {}
    if args.previous_density_result_json:
        previous_path = Path(args.previous_density_result_json).expanduser().resolve()
        previous_body = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_by_spacing = load_previous_density_by_spacing(previous_body)

    reference_json = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_freqs = extract_reference_frequencies_hz(reference_json, tol_hz=tol_hz)
    if not reference_freqs:
        body = {
            "status": "FAIL",
            "failure_reason": f"no reference accepted frequencies in {reference_path}",
            "reference_json": str(reference_path),
        }
        write_json_atomic(output_dir / "density_result.json", body)
        _write_density_md(output_dir / "density_result.md", body)
        return 2

    meta_path = checkpoint / "built_metadata.json"
    if not meta_path.is_file():
        body = {
            "status": "FAIL",
            "failure_reason": f"missing built metadata: {meta_path}",
            "checkpoint_dir": str(checkpoint),
        }
        write_json_atomic(output_dir / "density_result.json", body)
        _write_density_md(output_dir / "density_result.md", body)
        return 2

    built_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mesh_level = str(built_meta.get("mesh_level") or "unknown")

    experiment: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_kind": "checkpoint_target_density",
        "checkpoint_dir": str(checkpoint),
        "reference_json": str(reference_path),
        "previous_density_result_json": str(previous_path) if previous_path else None,
        "output_dir": str(output_dir),
        "mesh_level": mesh_level,
        "factor_solver": factor_solver,
        "nev": nev,
        "ncv": ncv,
        "start_hz": start_hz,
        "stop_hz": stop_hz,
        "spacings_hz": spacings,
        "tolerance_hz": tol_hz,
        "acceptance_interval_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        "reference_unique_accepted_count": len(reference_freqs),
        "reference_unique_accepted_frequencies_hz": reference_freqs,
        "versions": version_snapshot(),
        "threading_env": threading_env_snapshot(),
        "checkpoint_load": None,
        "matrix_contract": None,
        "spacings": [],
        "sparsest_coverage_pass_spacing_hz": None,
        "previous_density_comparison": None,
        "status": "FAIL",
        "failure_reason": None,
    }

    mats: List[Any] = []
    t_experiment0 = time.perf_counter()
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

        spacing_rows: List[Dict[str, Any]] = []

        for spacing_hz in spacings:
            generated_targets = generate_spaced_targets_hz(start_hz, stop_hz, spacing_hz)
            print(
                f"[B3_target_density] spacing={spacing_hz} Hz targets={len(generated_targets)} "
                f"nev={nev} ncv={ncv} range=[{generated_targets[0]}, {generated_targets[-1]}]",
                flush=True,
            )
            t_spacing0 = time.perf_counter()
            per_target_rows: List[Dict[str, Any]] = []
            all_accepted: List[float] = []
            total_st = 0.0
            targets_failed = 0
            spacing_failure_reason: Optional[str] = None

            for ti, target_hz in enumerate(generated_targets):
                print(
                    f"[B3_target_density] spacing={spacing_hz} target {ti + 1}/{len(generated_targets)} "
                    f"hz={target_hz}",
                    flush=True,
                )
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
                    if spacing_failure_reason is None:
                        spacing_failure_reason = str(row.get("failure_reason") or row.get("status"))

            unique_accepted = deduplicate_frequencies_hz(all_accepted, tol_hz=tol_hz)
            coverage = compare_reference_coverage(
                reference_freqs,
                unique_accepted,
                tol_hz=tol_hz,
            )
            spacing_wall_s = time.perf_counter() - t_spacing0
            spacing_status = "PASS"
            if targets_failed == len(generated_targets):
                spacing_status = "FAIL"
            elif targets_failed > 0:
                spacing_status = "PARTIAL"

            spacing_row: Dict[str, Any] = {
                "spacing_hz": float(spacing_hz),
                "generated_targets_hz": generated_targets,
                "target_count": len(generated_targets),
                "total_wall_s": safe_float(spacing_wall_s),
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
                "status": spacing_status,
                "failure_reason": spacing_failure_reason,
                "per_target": per_target_rows,
            }
            prev_cmp = compare_spacing_to_previous(
                spacing_hz=float(spacing_hz),
                current_coverage=coverage,
                previous_row=previous_by_spacing.get(float(spacing_hz)),
            )
            if prev_cmp is not None:
                spacing_row["previous_comparison"] = prev_cmp
            spacing_rows.append(spacing_row)
            print(
                f"[B3_target_density] spacing={spacing_hz} status={spacing_status} "
                f"coverage={coverage['matched_reference_count']}/{len(reference_freqs)} "
                f"wall={spacing_wall_s:.1f}s st={total_st:.1f}s",
                flush=True,
            )

        passing_spacings = [
            float(r["spacing_hz"]) for r in spacing_rows if bool(r.get("coverage_pass"))
        ]
        sparsest_pass = max(passing_spacings) if passing_spacings else None

        if previous_body is not None and previous_path is not None:
            compared = [
                r for r in spacing_rows if r.get("previous_comparison") is not None
            ]
            improved = [
                r
                for r in compared
                if bool((r.get("previous_comparison") or {}).get("coverage_improved"))
            ]
            deltas = [
                int((r.get("previous_comparison") or {}).get("matched_reference_delta") or 0)
                for r in compared
            ]
            experiment["previous_density_comparison"] = {
                "previous_density_result_json": str(previous_path),
                "previous_nev": previous_body.get("nev"),
                "previous_ncv": previous_body.get("ncv"),
                "previous_experiment_status": previous_body.get("status"),
                "spacings_compared": len(compared),
                "spacings_improved": len(improved),
                "improved_spacing_hz": [float(r["spacing_hz"]) for r in improved],
                "best_matched_reference_delta": max(deltas) if deltas else None,
                "per_spacing": [
                    {
                        "spacing_hz": float(r["spacing_hz"]),
                        **(r.get("previous_comparison") or {}),
                    }
                    for r in compared
                ],
            }

        experiment["spacings"] = spacing_rows
        experiment["sparsest_coverage_pass_spacing_hz"] = sparsest_pass
        experiment["experiment_wall_s"] = safe_float(time.perf_counter() - t_experiment0)

        any_pass = any(bool(r.get("coverage_pass")) for r in spacing_rows)
        all_fail = all(str(r.get("status")) == "FAIL" for r in spacing_rows)
        if all_fail:
            experiment["status"] = "FAIL"
        elif any_pass:
            experiment["status"] = "PASS"
        else:
            experiment["status"] = "PARTIAL"

        write_json_atomic(output_dir / "density_result.json", experiment)
        _write_density_md(output_dir / "density_result.md", experiment)
        print(
            f"[B3_target_density] {experiment['status']} spacings={len(spacing_rows)} "
            f"sparsest_pass={sparsest_pass} -> {output_dir / 'density_result.json'}",
            flush=True,
        )
        return 0 if experiment["status"] in ("PASS", "PARTIAL") else 2
    except Exception as exc:
        experiment["failure_reason"] = f"{type(exc).__name__}:{exc}"
        experiment["experiment_wall_s"] = safe_float(time.perf_counter() - t_experiment0)
        write_json_atomic(output_dir / "density_result.json", experiment)
        _write_density_md(output_dir / "density_result.md", experiment)
        print(f"[B3_target_density] FAIL {exc}", flush=True)
        return 2
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    return run_target_density_experiment(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
