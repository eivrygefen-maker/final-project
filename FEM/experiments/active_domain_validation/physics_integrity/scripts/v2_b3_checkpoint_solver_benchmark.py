#!/usr/bin/env python3
"""Solver-only checkpoint ST/EPS benchmark (no DOLFINx/FEM assembly imports)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_operator_checkpoint_portable import load_operators_with_portable_fallback  # noqa: E402
from v2_b3_petsc_util import mat_shape, write_json_atomic  # noqa: E402
from v2_b3_checkpoint_pipeline_lib import (  # noqa: E402
    B3_EXPORT_RICH_MODAL_DATA_ARG,
    ensure_rich_modal_export_allowed,
    rich_modal_export_manifest_block,
)
from v2_b3_st_sinvert_solver_lib import (  # noqa: E402
    ACCEPTANCE_FREQ_HI_HZ,
    ACCEPTANCE_FREQ_LO_HZ,
    built_from_checkpoint_metadata,
    build_stable_summary,
    compare_checkpoint_results_to_baseline,
    mat_global_nnz_used,
    run_checkpoint_st_target,
    safe_float,
    threading_env_snapshot,
    version_snapshot,
)

ALLOWED_FACTOR_SOLVERS = frozenset({"mumps", "mkl_pardiso"})
ALLOWED_EPS_TYPES = frozenset({"krylovschur"})
ALLOWED_ST_TYPES = frozenset({"sinvert"})


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load checkpoint A/M and run one KRYLOVSCHUR + ST.SINVERT benchmark case.",
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Operator checkpoint directory.")
    parser.add_argument("--target-hz", type=float, default=244.39)
    parser.add_argument("--factor-solver", choices=sorted(ALLOWED_FACTOR_SOLVERS), required=True)
    parser.add_argument("--eps-type", default="krylovschur")
    parser.add_argument("--st-type", default="sinvert")
    parser.add_argument("--nev", type=int, default=12)
    parser.add_argument("--ncv", type=int, default=24)
    parser.add_argument("--output-dir", required=True, help="Directory for result.json/result.md.")
    parser.add_argument("--baseline-json", help="Optional baseline result.json for parity comparison.")
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


def _validate_solver_semantics(args: argparse.Namespace) -> None:
    eps_type = str(args.eps_type).strip().lower()
    st_type = str(args.st_type).strip().lower()
    if eps_type not in ALLOWED_EPS_TYPES:
        raise ValueError(f"unsupported eps-type={args.eps_type!r}; expected krylovschur")
    if st_type not in ALLOWED_ST_TYPES:
        raise ValueError(f"unsupported st-type={args.st_type!r}; expected sinvert")


def _write_result_md(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Checkpoint solver benchmark",
        "",
        f"- checkpoint_dir: `{result.get('checkpoint_dir')}`",
        f"- factor_solver: `{result.get('factor_solver')}`",
        f"- target_hz: `{result.get('target_frequency_hz')}`",
        f"- status: `{result.get('status')}`",
        f"- setup_s: `{result.get('setup_elapsed_seconds')}`",
        f"- solve_s: `{result.get('solve_elapsed_seconds')}`",
        f"- total_s: `{result.get('total_elapsed_seconds')}`",
        f"- converged: `{result.get('converged_mode_count')}`",
        f"- accepted_n: `{result.get('accepted_mode_count_in_interval')}`",
        f"- accepted_hz: `{result.get('accepted_frequencies_hz')}`",
        f"- factor_solver_effective: `{result.get('factor_solver_effective')}`",
        f"- mumps_policy_effective: `{result.get('mumps_policy_effective')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_checkpoint_solver_benchmark(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rich_modal_requested = bool(args.export_rich_modal_data)
    ensure_rich_modal_export_allowed(requested=rich_modal_requested, context="B3_checkpoint_solver_bench")
    _validate_solver_semantics(args)

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = checkpoint / "built_metadata.json"
    if not meta_path.is_file():
        body = {
            "status": "FAIL",
            "failure_reason": f"missing built metadata: {meta_path}",
            "checkpoint_dir": str(checkpoint),
        }
        write_json_atomic(output_dir / "result.json", body)
        print(f"[B3_checkpoint_solver_bench] FAIL missing metadata -> {output_dir / 'result.json'}", flush=True)
        return 2

    built_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mesh_level = str(built_meta.get("mesh_level") or "unknown")
    factor_solver = str(args.factor_solver).strip().lower()
    target_hz = float(args.target_hz)

    result: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_kind": "checkpoint_solver_single_target",
        "checkpoint_dir": str(checkpoint),
        "output_dir": str(output_dir),
        "mesh_level": mesh_level,
        "target_frequency_hz": target_hz,
        "factor_solver": factor_solver,
        "eps_type_requested": str(args.eps_type).lower(),
        "st_type_requested": str(args.st_type).lower(),
        "nev": int(args.nev),
        "ncv": int(args.ncv),
        "acceptance_interval_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        "versions": version_snapshot(),
        "threading_env": threading_env_snapshot(),
        "checkpoint_load": None,
        "matrix_contract": None,
        "baseline_comparison": None,
        "rich_modal_export": rich_modal_export_manifest_block(requested=rich_modal_requested),
        "status": "FAIL",
        "failure_reason": None,
    }

    mats: List[Any] = []
    t_total0 = time.perf_counter()
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

        row = run_checkpoint_st_target(
            A_active=A_active,
            M_active=M_active,
            built=built,
            target_hz=target_hz,
            factor_solver=factor_solver,
            mesh_level=mesh_level,
            nev=int(args.nev),
            ncv=int(args.ncv),
            target_index=0,
        )
        result.update(row)
        result["total_elapsed_seconds"] = safe_float(time.perf_counter() - t_total0)

        if args.baseline_json:
            baseline_path = Path(args.baseline_json).expanduser().resolve()
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline["_baseline_path"] = str(baseline_path)
            result["baseline_comparison"] = compare_checkpoint_results_to_baseline(
                current={"targets": [row], "aggregate": {"unique_accepted_frequencies_hz": row.get("accepted_frequencies_hz")}},
                baseline=baseline,
            )

        result["summary"] = build_stable_summary(result)

        write_json_atomic(output_dir / "result.json", result)
        _write_result_md(output_dir / "result.md", result)
        print(
            f"[B3_checkpoint_solver_bench] {result.get('status')} factor={factor_solver} "
            f"setup={result.get('setup_elapsed_seconds')}s solve={result.get('solve_elapsed_seconds')}s "
            f"accepted_n={result.get('accepted_mode_count_in_interval')} -> {output_dir / 'result.json'}",
            flush=True,
        )
        return 0 if result.get("status") == "PASS" else 2
    except Exception as exc:
        from v2_b3_st_sinvert_solver_lib import extract_st_failure_diagnostics

        result["failure_reason"] = f"{type(exc).__name__}:{exc}"
        result["failure_class"] = extract_st_failure_diagnostics(exc).get("failure_class")
        result["total_elapsed_seconds"] = safe_float(time.perf_counter() - t_total0)
        result["summary"] = build_stable_summary(result)
        write_json_atomic(output_dir / "result.json", result)
        _write_result_md(output_dir / "result.md", result)
        print(f"[B3_checkpoint_solver_bench] FAIL {exc} -> {output_dir / 'result.json'}", flush=True)
        return 2
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


def main() -> int:
    return run_checkpoint_solver_benchmark()


if __name__ == "__main__":
    raise SystemExit(main())
