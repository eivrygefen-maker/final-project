#!/usr/bin/env python3
"""Solver-only multi-target checkpoint ST/EPS benchmark (no DOLFINx/FEM imports)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from petsc4py import PETSc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_operator_checkpoint_portable import load_operators_with_portable_fallback  # noqa: E402
from v2_b3_petsc_util import mat_shape, write_json_atomic  # noqa: E402
from v2_b3_rich_modal_lib import (  # noqa: E402
    RICH_MODAL_DIRNAME,
    RICH_MODAL_MANIFEST_JSON,
    RichModalCollector,
    SYNTHESIS_METADATA_JSON,
)
from v2_b3_checkpoint_pipeline_lib import (  # noqa: E402
    B3_EXPORT_RICH_MODAL_DATA_ARG,
    default_solve_output_dir,
    ensure_rich_modal_export_allowed,
    rich_modal_export_manifest_block,
)
from v2_b3_st_sinvert_solver_lib import (  # noqa: E402
    ACCEPTANCE_FREQ_HI_HZ,
    ACCEPTANCE_FREQ_LO_HZ,
    L_PROD_ST_FULL9_TARGETS_HZ,
    built_from_checkpoint_metadata,
    compare_checkpoint_results_to_baseline,
    build_stable_summary,
    deduplicate_frequencies_hz,
    mat_global_nnz_used,
    parse_hz_list,
    run_checkpoint_st_target,
    safe_float,
    threading_env_snapshot,
    version_snapshot,
)

ALLOWED_FACTOR_SOLVERS = frozenset({"mumps", "mkl_pardiso"})
ALLOWED_TARGET_SETS = frozenset({"full9"})


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load checkpoint A/M once and run sequential multi-target ST benchmarks.",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--factor-solver", choices=sorted(ALLOWED_FACTOR_SOLVERS), required=True)
    parser.add_argument(
        "--targets-hz",
        help="Comma-separated target frequencies in Hz. Overrides --target-set when set.",
    )
    parser.add_argument(
        "--target-set",
        choices=sorted(ALLOWED_TARGET_SETS),
        default="full9",
        help="Named target list (default: L_prod full9).",
    )
    parser.add_argument("--nev", type=int, default=12)
    parser.add_argument("--ncv", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        help="Default: solver_benchmarks/checkpoint_solve_<factor>_<set>_<utc>",
    )
    parser.add_argument(
        "--baseline-json",
        help="Optional baseline result.json for accepted-frequency parity comparison.",
    )
    parser.add_argument(
        B3_EXPORT_RICH_MODAL_DATA_ARG,
        dest="export_rich_modal_data",
        action="store_true",
        default=False,
        help="Opt-in rich modal export (NOT implemented; disabled by default for benchmarks).",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _resolve_targets(args: argparse.Namespace) -> List[float]:
    if args.targets_hz:
        return parse_hz_list(str(args.targets_hz))
    target_set = str(args.target_set).strip().lower()
    if target_set == "full9":
        return list(L_PROD_ST_FULL9_TARGETS_HZ)
    raise ValueError(f"unsupported target-set={args.target_set!r}")


def _write_result_md(path: Path, result: Dict[str, Any]) -> None:
    agg = result.get("aggregate") or {}
    lines = [
        "# Checkpoint solver multi-target benchmark",
        "",
        f"- checkpoint_dir: `{result.get('checkpoint_dir')}`",
        f"- factor_solver: `{result.get('factor_solver')}`",
        f"- targets_hz: `{result.get('targets_hz')}`",
        f"- status: `{result.get('status')}`",
        f"- aggregate_wall_s: `{agg.get('total_wall_seconds')}`",
        f"- total_setup_s: `{agg.get('total_setup_seconds')}`",
        f"- total_solve_s: `{agg.get('total_solve_seconds')}`",
        f"- total_st_s: `{agg.get('total_st_seconds')}`",
        f"- targets_succeeded: `{agg.get('targets_succeeded')}` / `{agg.get('targets_attempted')}`",
        f"- unique_accepted_hz: `{agg.get('unique_accepted_frequencies_hz')}`",
        "",
        "## Per target",
        "",
        "| idx | target_hz | status | setup_s | solve_s | st_total_s | accepted_n |",
        "|-----|-----------|--------|---------|---------|------------|------------|",
    ]
    for row in result.get("targets") or []:
        lines.append(
            f"| {row.get('target_index')} | {row.get('target_frequency_hz')} | {row.get('status')} | "
            f"{row.get('setup_elapsed_seconds')} | {row.get('solve_elapsed_seconds')} | "
            f"{row.get('st_total_elapsed_seconds')} | {row.get('accepted_mode_count_in_interval')} |"
        )
    if result.get("baseline_comparison"):
        bc = result["baseline_comparison"]
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                f"- parity_pass: `{bc.get('parity_pass')}`",
                f"- aggregate_match: `{bc.get('aggregate_accepted_frequencies_match')}`",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_checkpoint_solver_multi_benchmark(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rich_modal_requested = bool(args.export_rich_modal_data)
    ensure_rich_modal_export_allowed(requested=rich_modal_requested, context="B3_checkpoint_solver_multi")
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    factor_solver = str(args.factor_solver).strip().lower()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_solve_output_dir(factor_solver=factor_solver, target_set=str(args.target_set))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_hz = _resolve_targets(args)

    meta_path = checkpoint / "built_metadata.json"
    if not meta_path.is_file():
        body = {
            "status": "FAIL",
            "failure_reason": f"missing built metadata: {meta_path}",
            "checkpoint_dir": str(checkpoint),
        }
        write_json_atomic(output_dir / "result.json", body)
        return 2

    built_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mesh_level = str(built_meta.get("mesh_level") or "unknown")

    result: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_kind": "checkpoint_solver_multi_target",
        "checkpoint_dir": str(checkpoint),
        "output_dir": str(output_dir),
        "mesh_level": mesh_level,
        "factor_solver": factor_solver,
        "targets_hz": targets_hz,
        "target_set": str(args.target_set),
        "nev": int(args.nev),
        "ncv": int(args.ncv),
        "acceptance_interval_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        "versions": version_snapshot(),
        "threading_env": threading_env_snapshot(),
        "checkpoint_load": None,
        "matrix_contract": None,
        "targets": [],
        "aggregate": {},
        "baseline_comparison": None,
        "rich_modal_export": rich_modal_export_manifest_block(requested=rich_modal_requested),
        "status": "FAIL",
        "failure_reason": None,
    }

    mats: List[Any] = []
    t_wall0 = time.perf_counter()
    try:
        A_active, M_active, load_diag = load_operators_with_portable_fallback(checkpoint)
        mats.extend([A_active, M_active])
        built, built_diag = built_from_checkpoint_metadata(
            built_meta,
            A_active=A_active,
            M_active=M_active,
        )
        result["checkpoint_load"] = load_diag
        result["built_metadata_diag"] = built_diag
        result["matrix_contract"] = {
            "A_shape": mat_shape(A_active),
            "M_shape": mat_shape(M_active),
            "A_nnz_used": mat_global_nnz_used(A_active),
            "M_nnz_used": mat_global_nnz_used(M_active),
            "load_path_summary": load_diag.get("load_path_summary"),
        }

        per_target_rows: List[Dict[str, Any]] = []
        all_accepted: List[float] = []
        total_setup = 0.0
        total_solve = 0.0
        total_st = 0.0
        succeeded = 0
        rich_collector = RichModalCollector() if rich_modal_requested else None

        for ti, target_hz in enumerate(targets_hz):
            print(
                f"[B3_checkpoint_solver_multi] target {ti + 1}/{len(targets_hz)} "
                f"hz={target_hz} factor={factor_solver}",
                flush=True,
            )
            row = run_checkpoint_st_target(
                A_active=A_active,
                M_active=M_active,
                built=built,
                target_hz=float(target_hz),
                factor_solver=factor_solver,
                mesh_level=mesh_level,
                nev=int(args.nev),
                ncv=int(args.ncv),
                target_index=int(ti),
                export_vectors=rich_modal_requested,
            )
            per_target_rows.append(row)
            if rich_modal_requested and row.get("status") == "PASS":
                for am in row.get("accepted_modes") or []:
                    if "x_active" in am:
                        rich_collector.add_mode(
                            x_active=am["x_active"],
                            target_index=int(ti),
                            target_hz=float(target_hz),
                            record={k: v for k, v in am.items() if k != "x_active"},
                        )
                        del am["x_active"]
            if row.get("status") == "PASS":
                succeeded += 1
                all_accepted.extend(list(row.get("accepted_frequencies_hz") or []))
                total_setup += float(row.get("setup_elapsed_seconds") or 0.0)
                total_solve += float(row.get("solve_elapsed_seconds") or 0.0)
                total_st += float(row.get("st_total_elapsed_seconds") or 0.0)
            else:
                result["failure_reason"] = row.get("failure_reason")
                break

        unique_accepted = deduplicate_frequencies_hz(all_accepted)
        wall_s = time.perf_counter() - t_wall0
        result["targets"] = per_target_rows
        result["aggregate"] = {
            "targets_attempted": len(targets_hz),
            "targets_succeeded": succeeded,
            "total_setup_seconds": safe_float(total_setup),
            "total_solve_seconds": safe_float(total_solve),
            "total_st_seconds": safe_float(total_st),
            "total_wall_seconds": safe_float(wall_s),
            "unique_accepted_frequencies_hz": unique_accepted,
            "unique_accepted_mode_count": len(unique_accepted),
        }

        if args.baseline_json:
            baseline_path = Path(args.baseline_json).expanduser().resolve()
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline["_baseline_path"] = str(baseline_path)
            result["baseline_comparison"] = compare_checkpoint_results_to_baseline(
                current=result,
                baseline=baseline,
            )

        if succeeded == len(targets_hz):
            result["status"] = "PASS"
        elif succeeded > 0:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "FAIL"

        result["summary"] = build_stable_summary(result)

        if rich_collector is not None:
            synth_path = checkpoint / SYNTHESIS_METADATA_JSON
            rm_manifest = rich_collector.write_bundle(
                output_dir / RICH_MODAL_DIRNAME,
                checkpoint_dir=checkpoint,
                solve_output_dir=output_dir,
                factor_solver=factor_solver,
                nev=int(args.nev),
                ncv=int(args.ncv),
                target_set=str(args.target_set),
                targets_hz=targets_hz,
                acceptance_interval_hz=[ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
                synthesis_metadata_path=synth_path,
            )
            result["rich_modal_export"] = {
                **rich_modal_export_manifest_block(requested=True),
                "manifest": str((output_dir / RICH_MODAL_DIRNAME / RICH_MODAL_MANIFEST_JSON).resolve()),
                "modes_active_npz": rm_manifest.get("modes_active_npz"),
                "mode_count": rm_manifest.get("mode_count"),
            }

        write_json_atomic(output_dir / "result.json", result)
        _write_result_md(output_dir / "result.md", result)
        print(
            f"[B3_checkpoint_solver_multi] {result['status']} "
            f"targets={succeeded}/{len(targets_hz)} wall={wall_s:.1f}s "
            f"st_total={total_st:.1f}s -> {output_dir / 'result.json'}",
            flush=True,
        )
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}:{exc}"
        result["aggregate"] = {
            "total_wall_seconds": safe_float(time.perf_counter() - t_wall0),
        }
        result["summary"] = build_stable_summary(result)
        write_json_atomic(output_dir / "result.json", result)
        _write_result_md(output_dir / "result.md", result)
        print(f"[B3_checkpoint_solver_multi] FAIL {exc}", flush=True)
        return 2
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    return run_checkpoint_solver_multi_benchmark(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
