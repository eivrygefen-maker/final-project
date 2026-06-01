#!/usr/bin/env python3
"""Solver-mkl stage checkpoint solve (load checkpoint + ST/EPS; no FEM assembly)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_checkpoint_pipeline_lib import (  # noqa: E402
    B3_EXPORT_RICH_MODAL_DATA_ARG,
    PIPELINE_SOLVE_MANIFEST,
    default_solve_output_dir,
    ensure_rich_modal_export_allowed,
    fail_with_messages,
    rich_modal_export_manifest_block,
    verify_checkpoint_complete,
    verify_checkpoint_matrices,
    verify_mumps_available,
    verify_solver_mkl_stage_environment,
    write_json,
)
from v2_b3_checkpoint_solver_multi_benchmark import run_checkpoint_solver_multi_benchmark  # noqa: E402
from v2_b3_st_sinvert_solver_lib import version_snapshot, threading_env_snapshot  # noqa: E402

ALLOWED_FACTOR_SOLVERS = ("mkl_pardiso", "mumps")
ALLOWED_TARGET_SETS = ("full9",)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solver-mkl stage: load checkpoint and run multi-target ST solve.",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", help="Default: solver_benchmarks/checkpoint_solve_<factor>_<set>_<utc>")
    parser.add_argument("--factor-solver", choices=ALLOWED_FACTOR_SOLVERS, default="mkl_pardiso")
    parser.add_argument("--target-set", choices=ALLOWED_TARGET_SETS, default="full9")
    parser.add_argument("--targets-hz", help="Optional comma-separated override for target-set.")
    parser.add_argument("--nev", type=int, default=12)
    parser.add_argument("--ncv", type=int, default=24)
    parser.add_argument("--baseline-json", help="Optional baseline result.json for parity comparison.")
    parser.add_argument(
        "--skip-mkl-probe",
        action="store_true",
        help="Skip MKL PARDISO availability probe (not recommended).",
    )
    parser.add_argument(
        B3_EXPORT_RICH_MODAL_DATA_ARG,
        dest="export_rich_modal_data",
        action="store_true",
        default=False,
        help="Opt-in rich modal export (active eigenvectors under rich_modal/).",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def run_checkpoint_solve(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rich_modal_requested = bool(args.export_rich_modal_data)
    ensure_rich_modal_export_allowed(requested=rich_modal_requested, context="B3_checkpoint_solve")
    factor_solver = str(args.factor_solver).strip().lower()
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_solve_output_dir(factor_solver=factor_solver, target_set=str(args.target_set))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    require_mkl = factor_solver == "mkl_pardiso" and not args.skip_mkl_probe
    ok, messages = verify_solver_mkl_stage_environment(require_mkl_pardiso=require_mkl)
    if not ok:
        fail_with_messages("B3_checkpoint_solve", messages)

    if factor_solver == "mumps":
        mumps_ok, mumps_err = verify_mumps_available()
        if not mumps_ok:
            fail_with_messages("B3_checkpoint_solve", [f"mumps unavailable: {mumps_err}"])

    ckpt_ok, ckpt_errors, ckpt_detail = verify_checkpoint_complete(checkpoint, require_csr=False)
    if not ckpt_ok:
        fail_with_messages("B3_checkpoint_solve", ckpt_errors)

    mat_ok, mat_errors, mat_detail = verify_checkpoint_matrices(checkpoint)
    if not mat_ok:
        fail_with_messages("B3_checkpoint_solve", mat_errors)

    checkpoint_warnings = list(ckpt_detail.get("warnings") or [])

    pre_manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "solver_mkl_precheck",
        "status": "PASS",
        "checkpoint_dir": str(checkpoint),
        "output_dir": str(output_dir),
        "factor_solver": factor_solver,
        "target_set": str(args.target_set),
        "csr_present": bool(ckpt_detail.get("csr_present")),
        "csr_required": False,
        "load_path": mat_detail.get("load_path") or mat_detail.get("load_path_summary"),
        "warnings": checkpoint_warnings,
        "checkpoint_detail": ckpt_detail,
        "matrix_verify_detail": mat_detail,
        "versions": version_snapshot(),
        "threading_env": threading_env_snapshot(),
        "environment_messages": [m for m in messages if str(m).startswith("WARN:")],
        "rich_modal_export": rich_modal_export_manifest_block(requested=rich_modal_requested),
    }
    write_json(output_dir / PIPELINE_SOLVE_MANIFEST, pre_manifest)

    bench_argv: List[str] = [
        "--checkpoint-dir",
        str(checkpoint),
        "--factor-solver",
        factor_solver,
        "--target-set",
        str(args.target_set),
        "--nev",
        str(int(args.nev)),
        "--ncv",
        str(int(args.ncv)),
        "--output-dir",
        str(output_dir),
    ]
    if args.targets_hz:
        bench_argv.extend(["--targets-hz", str(args.targets_hz)])
    if args.baseline_json:
        bench_argv.extend(["--baseline-json", str(args.baseline_json)])

    rc = run_checkpoint_solver_multi_benchmark(bench_argv)

    result_path = output_dir / "result.json"
    if result_path.is_file():
        result_body = json.loads(result_path.read_text(encoding="utf-8"))
        pre_manifest["stage"] = "solver_mkl_solve"
        pre_manifest["solve_completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pre_manifest["solve_return_code"] = int(rc)
        pre_manifest["status"] = result_body.get("status", "FAIL")
        pre_manifest["summary"] = result_body.get("summary")
        pre_manifest["result_json"] = str(result_path)
    else:
        pre_manifest["stage"] = "solver_mkl_solve"
        pre_manifest["solve_return_code"] = int(rc)
        pre_manifest["status"] = "FAIL"
        pre_manifest["failure_reason"] = "result.json not written"
    write_json(output_dir / PIPELINE_SOLVE_MANIFEST, pre_manifest)

    print(f"[B3_checkpoint_solve] completed rc={rc} output={output_dir / 'result.json'}", flush=True)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    return run_checkpoint_solve(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
