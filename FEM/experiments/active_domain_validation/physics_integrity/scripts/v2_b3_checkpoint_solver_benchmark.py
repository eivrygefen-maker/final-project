#!/usr/bin/env python3
"""Solver-only checkpoint ST/EPS benchmark (no DOLFINx/FEM assembly imports)."""
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

from v2_b3_operator_checkpoint_portable import (  # noqa: E402
    load_operators_with_portable_fallback,
)
from v2_b3_petsc_util import mat_shape, write_json_atomic  # noqa: E402
from v2_b3_st_sinvert_solver_lib import (  # noqa: E402
    ACCEPTANCE_FREQ_HI_HZ,
    ACCEPTANCE_FREQ_LO_HZ,
    built_from_checkpoint_metadata,
    collect_accepted_st_modes,
    collect_converged_modes,
    configure_eps_krylovschur_sinvert,
    extract_st_failure_diagnostics,
    hz_to_lambda_sq,
    mat_global_nnz_used,
    mumps_policy_chain,
    peak_rss_mb,
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
    target_lambda = float(hz_to_lambda_sq(target_hz))

    result: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_dir": str(checkpoint),
        "output_dir": str(output_dir),
        "mesh_level": mesh_level,
        "target_frequency_hz": target_hz,
        "target_lambda": safe_float(target_lambda),
        "factor_solver": factor_solver,
        "eps_type_requested": str(args.eps_type).lower(),
        "st_type_requested": str(args.st_type).lower(),
        "nev": int(args.nev),
        "ncv": int(args.ncv),
        "acceptance_interval_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
        "versions": version_snapshot(),
        "threading_env": threading_env_snapshot(),
        "setup_succeeded": False,
        "solve_succeeded": False,
        "setup_elapsed_seconds": None,
        "solve_elapsed_seconds": None,
        "total_elapsed_seconds": None,
        "peak_rss_mb": None,
        "converged_mode_count": None,
        "converged_modes": [],
        "accepted_mode_count_in_interval": None,
        "accepted_frequencies_hz": [],
        "accepted_modes": [],
        "factor_solver_effective": None,
        "mumps_policy_effective": None,
        "mumps_policies_tried": [],
        "petsc_options_written": None,
        "configure_meta": None,
        "checkpoint_load": None,
        "matrix_contract": None,
        "status": "FAIL",
        "failure_reason": None,
        "failure_class": None,
    }

    mats: List[Any] = []
    eps = None
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

        if factor_solver == "mumps":
            policies = mumps_policy_chain(mesh_level=mesh_level)
        else:
            policies = [None]

        setup_succeeded = False
        last_setup_exc: Optional[BaseException] = None
        configure_meta: Dict[str, Any] = {}

        for policy in policies:
            if eps is not None:
                try:
                    eps.destroy()
                except Exception:
                    pass
                eps = None
            if policy is not None:
                result["mumps_policies_tried"].append(str(policy))
            try:
                from slepc4py import SLEPc

                eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
                configure_meta = configure_eps_krylovschur_sinvert(
                    eps,
                    A_active,
                    M_active,
                    target_hz=target_hz,
                    target_lambda=target_lambda,
                    factor_solver=factor_solver,
                    nev=int(args.nev),
                    ncv=int(args.ncv),
                    mumps_policy=policy,
                )
                t0 = time.perf_counter()
                eps.setUp()
                setup_s = time.perf_counter() - t0
                result["setup_elapsed_seconds"] = safe_float(setup_s)
                result["setup_succeeded"] = True
                result["configure_meta"] = configure_meta
                result["factor_solver_effective"] = configure_meta.get("factor_solver_effective")
                result["mumps_policy_effective"] = configure_meta.get("mumps_policy_applied")
                result["petsc_options_written"] = configure_meta.get("petsc_options_written")
                setup_succeeded = True
                break
            except Exception as exc:
                last_setup_exc = exc
                diag = extract_st_failure_diagnostics(exc)
                result["failure_class"] = diag.get("failure_class")
                result["failure_reason"] = f"{type(exc).__name__}:{exc}"

        if not setup_succeeded:
            result["status"] = "FAIL_SETUP"
            if last_setup_exc is not None:
                result["failure_diagnostics"] = extract_st_failure_diagnostics(last_setup_exc)
            write_json_atomic(output_dir / "result.json", result)
            _write_result_md(output_dir / "result.md", result)
            print(f"[B3_checkpoint_solver_bench] FAIL setup -> {output_dir / 'result.json'}", flush=True)
            return 2

        t0 = time.perf_counter()
        try:
            eps.solve()
            solve_s = time.perf_counter() - t0
            result["solve_elapsed_seconds"] = safe_float(solve_s)
            result["solve_succeeded"] = True
        except Exception as exc:
            result["failure_reason"] = f"{type(exc).__name__}:{exc}"
            result["failure_class"] = extract_st_failure_diagnostics(exc).get("failure_class")
            result["status"] = "FAIL_SOLVE"
            write_json_atomic(output_dir / "result.json", result)
            _write_result_md(output_dir / "result.md", result)
            print(f"[B3_checkpoint_solver_bench] FAIL solve -> {output_dir / 'result.json'}", flush=True)
            return 2

        nconv, converged_modes = collect_converged_modes(eps, A_active)
        _nconv2, accepted_modes = collect_accepted_st_modes(
            eps,
            A_active,
            built,
            target_hz=target_hz,
        )
        accepted_freqs = sorted(float(m["frequency_hz"]) for m in accepted_modes)

        result["converged_mode_count"] = int(nconv)
        result["converged_modes"] = converged_modes
        result["accepted_mode_count_in_interval"] = len(accepted_modes)
        result["accepted_modes"] = accepted_modes
        result["accepted_frequencies_hz"] = accepted_freqs
        result["peak_rss_mb"] = peak_rss_mb()
        result["total_elapsed_seconds"] = safe_float(time.perf_counter() - t_total0)
        result["status"] = "PASS"
        write_json_atomic(output_dir / "result.json", result)
        _write_result_md(output_dir / "result.md", result)
        print(
            f"[B3_checkpoint_solver_bench] PASS factor={factor_solver} "
            f"setup={result['setup_elapsed_seconds']}s solve={result['solve_elapsed_seconds']}s "
            f"accepted_n={len(accepted_freqs)} -> {output_dir / 'result.json'}",
            flush=True,
        )
        return 0
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}:{exc}"
        result["failure_class"] = extract_st_failure_diagnostics(exc).get("failure_class")
        result["total_elapsed_seconds"] = safe_float(time.perf_counter() - t_total0)
        write_json_atomic(output_dir / "result.json", result)
        _write_result_md(output_dir / "result.md", result)
        print(f"[B3_checkpoint_solver_bench] FAIL {exc} -> {output_dir / 'result.json'}", flush=True)
        return 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


def main() -> int:
    return run_checkpoint_solver_benchmark()


if __name__ == "__main__":
    raise SystemExit(main())
