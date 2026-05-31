#!/usr/bin/env python3
"""Dev-only ST/EPS solver benchmark harness (organized under diagnostics/solver_benchmarks/<run_id>/).

Does not alter production solver defaults or fem_main_3d behavior.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit
import v2_b3_dev_solver_benchmark as dev_bench
import v2_b3_st_worker_scaling_benchmark as st_scaling
from v2_b3_block_compose_backend import apply_compose_backend_from_argv
from v2_mesh_convergence_common import CONV_DIAG

B3_ST_SOLVER_BENCHMARK_ARG = "--B3-ST-solver-benchmark-only"
B3_ST_SOLVER_BENCHMARK_SUITE_ARG = "--B3-ST-solver-benchmark-suite"

BENCHMARK_ROOT = CONV_DIAG / "solver_benchmarks"
ALLOWED_SUITES = frozenset({"factor_solver", "eps_params", "interval_slicing", "all"})

FACTOR_SOLVER_CANDIDATES = (
    "mumps",
    "superlu_dist",
    "mkl_pardiso",
    "pastix",
    "strumpack",
)

BASELINE_CASE_ID = "baseline_mumps_nev12_ncv24"
BASELINE_FACTOR = "mumps"
BASELINE_NEV = int(dev_bench.B3_DEV_COARSE_NEV)
BASELINE_NCV = int(dev_bench.B3_DEV_COARSE_NCV)

INTERVAL_SLICING_HZ_DEFAULT = (240.0, 250.0)


def is_st_solver_benchmark_mode(argv: Sequence[str]) -> bool:
    return B3_ST_SOLVER_BENCHMARK_ARG in argv


def _parse_arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    return st_scaling._parse_arg_value(argv, flag)


def _parse_suite(argv: Sequence[str]) -> str:
    raw = (_parse_arg_value(argv, B3_ST_SOLVER_BENCHMARK_SUITE_ARG) or "all").strip().lower()
    if raw not in ALLOWED_SUITES:
        raise ValueError(f"{B3_ST_SOLVER_BENCHMARK_SUITE_ARG} must be one of {sorted(ALLOWED_SUITES)}")
    return raw


def _run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _safe_float(x: Any) -> Optional[float]:
    return dev_bench._safe_float(x)


def _write_json(path: Path, body: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audit._write_json_atomic(path, body)


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def probe_factor_solver_packages() -> Dict[str, Any]:
    """Probe PETSc LU factor backends without failing the whole benchmark."""
    from slepc4py import SLEPc

    out: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "petsc_version": None,
        "slepc_version": None,
        "mpi_comm_world_size": None,
        "default_mat_type_on_comm_world": None,
        "eps_types": {},
        "factor_solvers": {},
    }
    try:
        out["petsc_version"] = str(PETSc.Sys.getVersion())
        out["slepc_version"] = str(getattr(SLEPc, "__version__", "unknown"))
        out["mpi_comm_world_size"] = int(MPI.COMM_WORLD.Get_size())
        A = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        A.setSizes([4, 4])
        A.setType("aij")
        A.setUp()
        out["default_mat_type_on_comm_world"] = str(A.getType())
        A.destroy()
    except Exception as exc:
        out["environment_probe_error"] = f"{type(exc).__name__}:{exc}"
        return out

    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    for name in ("krylovschur", "ciss", "primme", "jd", "gd"):
        try:
            eps.setType(name)
            out["eps_types"][name] = {"available": True, "effective": str(eps.getType())}
        except Exception as exc:
            out["eps_types"][name] = {"available": False, "error": f"{type(exc).__name__}:{exc}"}
    try:
        eps.destroy()
    except Exception:
        pass

    for pkg in FACTOR_SOLVER_CANDIDATES:
        entry: Dict[str, Any] = {"requested": pkg, "available": False}
        A = None
        pc = None
        try:
            A = PETSc.Mat().createAIJ(size=(4, 4), comm=PETSc.COMM_WORLD)
            A.setUp()
            pc = PETSc.PC().create(comm=PETSc.COMM_WORLD)
            pc.setOperators(A)
            pc.setType("lu")
            pc.setFactorSolverType(pkg)
            pc.setUp()
            effective = None
            try:
                effective = str(pc.getFactorSolverType())
            except Exception:
                effective = None
            entry["available"] = True
            entry["effective_factor_solver"] = effective
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}:{exc}"
        finally:
            if pc is not None:
                try:
                    pc.destroy()
                except Exception:
                    pass
            if A is not None:
                try:
                    A.destroy()
                except Exception:
                    pass
        out["factor_solvers"][pkg] = entry
    return out


def _benchmark_solver_cfg(*, factor_solver: str) -> Dict[str, Any]:
    cfg = dict(audit._b3_ciss_direct_stable_solver_cfg())
    fs = str(factor_solver).strip().lower()
    cfg["st_pc_factor_mat_solver_type"] = fs
    cfg["st_factor_solver_type"] = fs
    return cfg


def configure_eps_st_benchmark(
    eps: Any,
    A_active: Any,
    M_active: Any,
    *,
    target_hz: float,
    target_lambda: float,
    variant: Dict[str, Any],
    meta_out: Dict[str, Any],
) -> None:
    """Parameterized KRYLOVSCHUR + ST.SINVERT (dev benchmark only)."""
    from slepc4py import SLEPc

    import fem_main_3d as fem3d

    factor_solver = str(variant.get("factor_solver", BASELINE_FACTOR)).strip().lower()
    nev = int(variant.get("nev", BASELINE_NEV))
    ncv = int(variant.get("ncv", BASELINE_NCV))
    eps_tol = variant.get("eps_tol")
    eps_max_it = variant.get("eps_max_it")
    interval_hz = variant.get("interval_hz")
    targeting = str(variant.get("targeting", "per_target_sigma"))

    meta_out["benchmark_factor_solver_requested"] = factor_solver
    meta_out["benchmark_nev"] = nev
    meta_out["benchmark_ncv"] = ncv
    meta_out["benchmark_eps_tol"] = _safe_float(eps_tol) if eps_tol is not None else None
    meta_out["benchmark_eps_max_it"] = int(eps_max_it) if eps_max_it is not None else None
    meta_out["benchmark_targeting_mode"] = targeting
    meta_out["benchmark_interval_hz"] = list(interval_hz) if interval_hz else None

    eps.setOperators(A_active, M_active)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(float(target_lambda))
    try:
        eps.setDimensions(nev=nev, ncv=ncv)
    except TypeError:
        eps.setDimensions(nev, ncv)

    if eps_tol is not None or eps_max_it is not None:
        tol = float(eps_tol) if eps_tol is not None else 1.0e-3
        max_it = int(eps_max_it) if eps_max_it is not None else 3000
        eps.setTolerances(tol, max_it)
        conv_rel = getattr(SLEPc.EPS.Conv, "REL", None)
        if conv_rel is not None:
            try:
                eps.setConvergenceTest(conv_rel)
            except Exception:
                pass

    shift_lambda = float(target_lambda)
    if interval_hz is not None:
        lo_hz, hi_hz = float(interval_hz[0]), float(interval_hz[1])
        lam_lo = float(audit._b3_hz_to_lambda_sq(lo_hz))
        lam_hi = float(audit._b3_hz_to_lambda_sq(hi_hz))
        rg_type = audit._b3_ciss_configure_rg_interval(eps, lam_lo=lam_lo, lam_hi=lam_hi)
        meta_out["benchmark_rg_interval_lambda"] = [_safe_float(lam_lo), _safe_float(lam_hi)]
        meta_out["benchmark_rg_type"] = rg_type
        center_hz = 0.5 * (lo_hz + hi_hz)
        shift_lambda = float(audit._b3_hz_to_lambda_sq(center_hz))
        eps.setTarget(shift_lambda)
        meta_out["benchmark_interval_center_hz"] = center_hz
        meta_out["benchmark_interval_center_lambda"] = _safe_float(shift_lambda)

    st = eps.getST()
    try:
        st.setType(SLEPc.ST.Type.SINVERT)
    except Exception:
        st.setType("sinvert")
    st.setShift(shift_lambda)

    ksp = st.getKSP()
    pc = ksp.getPC()
    solver_cfg = _benchmark_solver_cfg(factor_solver=factor_solver)
    fem3d._slepc_configure_st_ksp_pc(
        ksp,
        pc,
        solver_cfg,
        block_is=None,
        opts_prefix="st_",
        use_ciss=True,
    )
    if factor_solver == "mumps":
        audit._b3_ciss_record_direct_stable_factor_shift_request(pc, meta_out)
    try:
        effective = audit._b3_ciss_pc_factor_solver_effective_label(pc)
        meta_out["benchmark_factor_solver_effective"] = effective
        if effective is None or factor_solver not in str(effective).lower():
            if factor_solver != "mumps":
                meta_out["benchmark_factor_solver_effective_warning"] = (
                    f"requested={factor_solver} effective={effective}"
                )
    except Exception:
        pass

    mumps_policy = str(variant.get("mumps_policy", "default"))
    if factor_solver == "mumps":
        mumps_written = st_scaling._apply_st_mumps_petsc_policy(mumps_policy)
        meta_out.update(mumps_written)


def run_single_st_benchmark_case(
    built: Dict[str, Any],
    *,
    target_hz: float,
    variant: Dict[str, Any],
    mesh_level: str,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one EPSSetUp + EPSSolve for a benchmark variant."""
    from slepc4py import SLEPc

    case_id = str(variant.get("case_id", "unnamed"))
    target_lambda = float(audit._b3_hz_to_lambda_sq(float(target_hz)))
    freq_lo = float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    freq_hi = float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    A_active = built["A_active"]
    M_active = built["M_active"]

    result: Dict[str, Any] = {
        "case_id": case_id,
        "suite": variant.get("suite"),
        "target_frequency_hz": float(target_hz),
        "target_lambda": _safe_float(target_lambda),
        "mesh_level": mesh_level,
        "skipped": False,
        "skip_reason": None,
        "setup_succeeded": False,
        "solve_succeeded": False,
        "setup_elapsed_seconds": None,
        "solve_elapsed_seconds": None,
        "st_total_elapsed_seconds": None,
        "peak_rss_mb": None,
        "converged_mode_count": None,
        "accepted_mode_count_in_interval": None,
        "accepted_frequencies_hz": [],
        "failure_reason": None,
        "failure_class": None,
        "variant": dict(variant),
    }

    if variant.get("skip") and variant.get("skip_reason"):
        result["skipped"] = True
        result["skip_reason"] = str(variant["skip_reason"])
        return result

    if not variant.get("factor_solver_available", True):
        result["skipped"] = True
        result["skip_reason"] = f"factor_solver_unavailable:{variant.get('factor_solver')}"
        return result

    eps = None
    t_st0 = time.perf_counter()
    try:
        policy_chain = (
            st_scaling._mumps_policy_chain(mesh_level)
            if str(variant.get("factor_solver", BASELINE_FACTOR)) == "mumps"
            else ["default"]
        )
        setup_meta: Dict[str, Any] = {}
        last_exc: Optional[BaseException] = None
        for policy in policy_chain:
            if eps is not None:
                try:
                    eps.destroy()
                except Exception:
                    pass
                eps = None
            try:
                eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
                configure_eps_st_benchmark(
                    eps,
                    A_active,
                    M_active,
                    target_hz=float(target_hz),
                    target_lambda=target_lambda,
                    variant={**variant, "mumps_policy": policy},
                    meta_out=setup_meta,
                )
                t0 = time.perf_counter()
                eps.setUp()
                setup_s = time.perf_counter() - t0
                result["setup_succeeded"] = True
                result["setup_elapsed_seconds"] = _safe_float(setup_s)
                result["mumps_policy_effective"] = policy if policy != "default" or mesh_level == "L_prod" else policy
                result["configure_meta"] = setup_meta
                intro = dev_bench._dev_introspect_st_targeting_after_setup(eps)
                result["effective_target"] = intro.get("B3_DEV_ST_target_effective")
                result["effective_shift"] = intro.get("B3_DEV_ST_shift_effective")
                result["effective_which"] = intro.get("B3_DEV_ST_which_effective_normalized")
                break
            except Exception as exc:
                last_exc = exc
                diag = st_scaling._extract_st_failure_diagnostics(exc)
                result.update({k: v for k, v in diag.items() if not k.startswith("exception_")})
                result["failure_reason"] = f"{type(exc).__name__}:{exc}"
                if log_path:
                    _append_log(log_path, traceback.format_exc())

        if not result["setup_succeeded"]:
            if last_exc is not None:
                result["failure_class"] = result.get("failure_class") or "ST_SETUP_FAILED"
            return result

        t1 = time.perf_counter()
        eps.solve()
        solve_s = time.perf_counter() - t1
        result["solve_succeeded"] = True
        result["solve_elapsed_seconds"] = _safe_float(solve_s)
        nconv, accepted = dev_bench._dev_collect_accepted_st_modes(
            eps,
            A_active,
            built,
            target_hz=float(target_hz),
            freq_lo=freq_lo,
            freq_hi=freq_hi,
        )
        result["converged_mode_count"] = int(nconv)
        result["accepted_mode_count_in_interval"] = int(len(accepted))
        result["accepted_frequencies_hz"] = [float(m["frequency_hz"]) for m in accepted]
        result["accepted_mode_records"] = accepted
    except Exception as exc:
        diag = st_scaling._extract_st_failure_diagnostics(exc)
        result.update({k: v for k, v in diag.items() if not k.startswith("exception_")})
        result["failure_reason"] = f"{type(exc).__name__}:{exc}"
        result["failure_class"] = diag.get("failure_class", "ST_SOLVE_FAILED")
        if log_path:
            _append_log(log_path, traceback.format_exc())
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        result["st_total_elapsed_seconds"] = _safe_float(time.perf_counter() - t_st0)
        result["peak_rss_mb"] = st_scaling._peak_rss_mb()

    return result


def _freq_lists_match(a: List[float], b: List[float], *, tol_hz: float = 0.05) -> bool:
    if len(a) != len(b):
        return False
    aa = sorted(float(x) for x in a)
    bb = sorted(float(x) for x in b)
    return all(abs(x - y) <= tol_hz for x, y in zip(aa, bb))


def _parity_vs_baseline(case: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    b_freqs = list(baseline.get("accepted_frequencies_hz") or [])
    c_freqs = list(case.get("accepted_frequencies_hz") or [])
    b_st = float(baseline.get("st_total_elapsed_seconds") or 0.0)
    c_st = float(case.get("st_total_elapsed_seconds") or 0.0)
    speedup = (b_st / c_st) if b_st > 0.0 and c_st > 0.0 else None
    parity = {
        "accepted_frequencies_match_baseline": _freq_lists_match(c_freqs, b_freqs),
        "baseline_accepted_frequencies_hz": b_freqs,
        "case_accepted_frequencies_hz": c_freqs,
        "accepted_mode_count_delta": int(case.get("accepted_mode_count_in_interval") or 0)
        - int(baseline.get("accepted_mode_count_in_interval") or 0),
        "converged_count_delta": int(case.get("converged_mode_count") or 0)
        - int(baseline.get("converged_mode_count") or 0),
        "speedup_vs_baseline_st_total": _safe_float(speedup),
        "st_total_seconds_delta": _safe_float(c_st - b_st) if c_st and b_st else None,
        "parity_pass": False,
        "parity_notes": [],
    }
    if case.get("skipped"):
        parity["parity_notes"].append(f"skipped:{case.get('skip_reason')}")
    elif not case.get("solve_succeeded"):
        parity["parity_notes"].append(f"solve_failed:{case.get('failure_reason')}")
    elif parity["accepted_frequencies_match_baseline"]:
        parity["parity_pass"] = True
        parity["parity_notes"].append("accepted_frequencies_match_baseline")
    else:
        parity["parity_notes"].append("accepted_frequencies_differ_from_baseline")
    return parity


def _summary_table_md(cases: List[Dict[str, Any]], *, baseline_st: Optional[float]) -> str:
    lines = [
        "| case_id | status | setup_s | solve_s | st_total_s | speedup_vs_baseline | accepted_n | parity |",
        "|---------|--------|---------|---------|------------|---------------------|------------|--------|",
    ]
    for c in cases:
        cid = c.get("case_id", "")
        if c.get("skipped"):
            status = f"SKIP ({c.get('skip_reason', '')[:40]})"
        elif c.get("solve_succeeded"):
            status = "OK"
        elif c.get("setup_succeeded"):
            status = "SETUP_ONLY"
        else:
            status = "FAIL"
        setup_s = c.get("setup_elapsed_seconds")
        solve_s = c.get("solve_elapsed_seconds")
        st_tot = c.get("st_total_elapsed_seconds")
        sp = c.get("parity", {}).get("speedup_vs_baseline_st_total")
        if sp is None and baseline_st and st_tot:
            sp = baseline_st / float(st_tot) if float(st_tot) > 0 else None
        acc_n = c.get("accepted_mode_count_in_interval")
        parity_ok = c.get("parity", {}).get("parity_pass")
        if c.get("case_id") == BASELINE_CASE_ID:
            parity_s = "baseline"
        elif parity_ok is True:
            parity_s = "match"
        elif parity_ok is False:
            parity_s = "**DIFF**"
        else:
            parity_s = "n/a"
        lines.append(
            f"| {cid} | {status} | {setup_s} | {solve_s} | {st_tot} | "
            f"{_safe_float(sp) if sp is not None else ''} | {acc_n} | {parity_s} |"
        )
    return "\n".join(lines) + "\n"


def _build_operators_once(
    argv: Sequence[str],
    *,
    mesh_level: str,
    pre: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
    mats: List[Any] = []
    seen: set[int] = set()
    if not pre.get("preassembly_contract_pass"):
        raise RuntimeError("preassembly_contract_pass=False")
    apply_compose_backend_from_argv(argv, mesh_level=mesh_level)
    built = audit._b3_build_corrected_structural_active_operators(
        mats_to_destroy=mats,
        mat_destroy_seen=seen,
        mesh_level=mesh_level,
        struct_active_count_policy=st_scaling._struct_active_count_policy(mesh_level),
        operator_build_profile=None,
    )
    payload: Dict[str, Any] = {}
    contract_pass = st_scaling._st_scaling_operator_contract_pass(payload, built=built, mesh_level=mesh_level)
    if not contract_pass:
        raise RuntimeError(f"operator_contract_failed:{payload.get('failure_reason')}")
    meta = {
        "mesh_level": mesh_level,
        "active_dimension": int(built["active_local"].size),
        "A_shape": audit._mat_shape(built["A_active"]),
        "M_shape": audit._mat_shape(built["M_active"]),
        "compose_backend": os.environ.get("B3_BLOCK_COMPOSE_BACKEND", "direct_row_loop"),
    }
    return built, meta, mats


def _factor_solver_variants(package_probe: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = [
        {
            "case_id": BASELINE_CASE_ID,
            "suite": "factor_solver",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "factor_solver_available": True,
        }
    ]
    fs_map = package_probe.get("factor_solvers") or {}
    for pkg in FACTOR_SOLVER_CANDIDATES:
        if pkg == BASELINE_FACTOR:
            continue
        info = fs_map.get(pkg) or {}
        variants.append(
            {
                "case_id": f"factor_{pkg}",
                "suite": "factor_solver",
                "factor_solver": pkg,
                "nev": BASELINE_NEV,
                "ncv": BASELINE_NCV,
                "factor_solver_available": bool(info.get("available")),
                "skip": not bool(info.get("available")),
                "skip_reason": info.get("error") if not info.get("available") else None,
            }
        )
    return variants


def _eps_params_variants() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": BASELINE_CASE_ID,
            "suite": "eps_params",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
        },
        {
            "case_id": "nev8_ncv16",
            "suite": "eps_params",
            "factor_solver": BASELINE_FACTOR,
            "nev": 8,
            "ncv": 16,
        },
        {
            "case_id": "nev16_ncv32",
            "suite": "eps_params",
            "factor_solver": BASELINE_FACTOR,
            "nev": 16,
            "ncv": 32,
        },
        {
            "case_id": "tol_1e-3",
            "suite": "eps_params",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "eps_tol": 1.0e-3,
            "eps_max_it": 3000,
        },
        {
            "case_id": "tol_1e-4",
            "suite": "eps_params",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "eps_tol": 1.0e-4,
            "eps_max_it": 3000,
        },
        {
            "case_id": "max_it_1500",
            "suite": "eps_params",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "eps_tol": 1.0e-3,
            "eps_max_it": 1500,
        },
        {
            "case_id": "max_it_5000",
            "suite": "eps_params",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "eps_tol": 1.0e-3,
            "eps_max_it": 5000,
        },
    ]


def _interval_slicing_variants() -> List[Dict[str, Any]]:
    lo, hi = INTERVAL_SLICING_HZ_DEFAULT
    return [
        {
            "case_id": "baseline_per_target_sigma",
            "suite": "interval_slicing",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "targeting": "per_target_sigma",
        },
        {
            "case_id": f"rg_interval_{lo:g}_{hi:g}_hz",
            "suite": "interval_slicing",
            "factor_solver": BASELINE_FACTOR,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "targeting": "rg_interval_prototype",
            "interval_hz": [lo, hi],
        },
    ]


def _run_suite(
    suite_name: str,
    variants: List[Dict[str, Any]],
    *,
    built: Dict[str, Any],
    mesh_level: str,
    targets_hz: List[float],
    suite_dir: Path,
    baseline_case: Optional[Dict[str, Any]],
    argv: Sequence[str],
) -> Dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    target_hz = float(targets_hz[0])
    command_manifest = {
        "suite": suite_name,
        "mesh_level": mesh_level,
        "targets_hz": list(targets_hz),
        "target_hz_primary": target_hz,
        "argv": list(argv),
        "variants": [v.get("case_id") for v in variants],
        "compose_backend": os.environ.get("B3_BLOCK_COMPOSE_BACKEND"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(suite_dir / "command_manifest.json", command_manifest)

    cases: List[Dict[str, Any]] = []
    for variant in variants:
        case_id = str(variant["case_id"])
        log_path = logs_dir / f"{case_id}.log"
        _append_log(log_path, f"=== {suite_name}/{case_id} target_hz={target_hz} ===")
        print(f"[B3_ST_solver_bench] {suite_name}/{case_id} target={target_hz} Hz", flush=True)
        case_result = run_single_st_benchmark_case(
            built,
            target_hz=target_hz,
            variant=variant,
            mesh_level=mesh_level,
            log_path=log_path,
        )
        ref = baseline_case
        if ref is not None and case_id != ref.get("case_id"):
            case_result["parity"] = _parity_vs_baseline(case_result, ref)
        elif case_id == BASELINE_CASE_ID or case_id == "baseline_per_target_sigma":
            case_result["parity"] = {
                "parity_pass": True,
                "parity_notes": ["baseline_reference_case"],
                "speedup_vs_baseline_st_total": 1.0,
            }
        cases.append(case_result)
        _write_json(logs_dir / f"{case_id}_result.json", case_result)

    baseline_st = None
    if baseline_case:
        baseline_st = baseline_case.get("st_total_elapsed_seconds")
    for c in cases:
        if c.get("parity") is None and baseline_case is not None:
            c["parity"] = _parity_vs_baseline(c, baseline_case)
        if baseline_st and c.get("st_total_elapsed_seconds"):
            c.setdefault("parity", {})
            if c["parity"].get("speedup_vs_baseline_st_total") is None:
                c["parity"]["speedup_vs_baseline_st_total"] = _safe_float(
                    float(baseline_st) / float(c["st_total_elapsed_seconds"])
                )

    suite_result = {
        "suite": suite_name,
        "mesh_level": mesh_level,
        "targets_hz": list(targets_hz),
        "baseline_case_id": baseline_case.get("case_id") if baseline_case else BASELINE_CASE_ID,
        "baseline_st_total_seconds": baseline_st,
        "cases": cases,
        "summary_table_markdown": _summary_table_md(cases, baseline_st=baseline_st),
    }
    _write_json(suite_dir / "result.json", suite_result)

    md_lines = [
        f"# ST solver benchmark — {suite_name}",
        "",
        f"- Mesh: `{mesh_level}`",
        f"- Target Hz: `{target_hz}`",
        f"- Baseline ST total (s): `{baseline_st}`",
        "",
        "## Summary",
        "",
        suite_result["summary_table_markdown"],
        "",
        "## Correctness / parity",
        "",
    ]
    for c in cases:
        p = c.get("parity") or {}
        md_lines.append(
            f"- **{c.get('case_id')}**: parity_pass={p.get('parity_pass')}; "
            f"notes={p.get('parity_notes')}; accepted={c.get('accepted_frequencies_hz')}"
        )
    md_lines.extend(
        [
            "",
            "## Parallelism",
            "",
            "Run suites **serially** on this VM. Each case performs a full ST shift-invert "
            "factorization; concurrent runs risk RAM exhaustion and distorted timings.",
            "",
        ]
    )
    (suite_dir / "result.md").write_text("\n".join(md_lines), encoding="utf-8")
    (suite_dir / "summary_table.md").write_text(suite_result["summary_table_markdown"], encoding="utf-8")
    return suite_result


def run_st_solver_benchmark(argv: Sequence[str], pre: Dict[str, Any]) -> int:
    mpi_ok, mpi_size = st_scaling.st_worker_scaling_mpi_world_ok()
    if not mpi_ok:
        print(
            f"[B3_ST_solver_bench] requires MPI COMM_WORLD size 1 (got {mpi_size}); "
            "use plain python or mpiexec -n 1",
            flush=True,
        )
        return 2

    try:
        mesh_level = st_scaling._parse_mesh_level(argv)
        suite = _parse_suite(argv)
        targets_hz = st_scaling._parse_targets_hz(argv, mesh_level=mesh_level)
    except ValueError as exc:
        print(f"[B3_ST_solver_bench] {exc}", flush=True)
        return 2

    if len(targets_hz) > 1:
        print(
            f"[B3_ST_solver_bench] WARN: multiple targets {targets_hz}; "
            "using first target only for solver benchmarks",
            flush=True,
        )
    targets_hz = [float(targets_hz[0])]

    rid = _run_id()
    run_dir = BENCHMARK_ROOT / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    apply_compose_backend_from_argv(argv, mesh_level=mesh_level)
    compose_backend = os.environ.get("B3_BLOCK_COMPOSE_BACKEND", "direct_row_loop")

    print(f"[B3_ST_solver_bench] run_dir={run_dir}", flush=True)
    print(f"[B3_ST_solver_bench] suite={suite} mesh={mesh_level} targets={targets_hz}", flush=True)

    package_probe = probe_factor_solver_packages()
    _write_json(run_dir / "package_availability.json", package_probe)

    mats: List[Any] = []
    built: Optional[Dict[str, Any]] = None
    operator_meta: Dict[str, Any] = {}
    try:
        t_build = time.perf_counter()
        built, operator_meta, mats = _build_operators_once(argv, mesh_level=mesh_level, pre=pre)
        operator_build_s = time.perf_counter() - t_build
        operator_meta["operator_build_elapsed_seconds"] = _safe_float(operator_build_s)
    except Exception as exc:
        body = {
            "failure_reason": f"{type(exc).__name__}:{exc}",
            "package_availability": str(run_dir / "package_availability.json"),
        }
        _write_json(run_dir / "run_failure.json", body)
        print(f"[B3_ST_solver_bench] operator build failed: {exc}", flush=True)
        return 2

    run_manifest = {
        "run_id": rid,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_ST_solver_benchmark_only",
        "mesh_level": mesh_level,
        "targets_hz": targets_hz,
        "suite_requested": suite,
        "compose_backend": compose_backend,
        "argv": list(argv),
        "benchmark_root": str(run_dir.resolve()),
        "operator_meta": operator_meta,
        "production_promotion": "BLOCKED",
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)

    suites_to_run: List[str]
    if suite == "all":
        suites_to_run = ["factor_solver", "eps_params", "interval_slicing"]
    else:
        suites_to_run = [suite]

    baseline_reference: Optional[Dict[str, Any]] = None
    suite_results: Dict[str, Any] = {}

    for suite_name in suites_to_run:
        if suite_name == "factor_solver":
            variants = _factor_solver_variants(package_probe)
        elif suite_name == "eps_params":
            variants = _eps_params_variants()
        else:
            variants = _interval_slicing_variants()

        if baseline_reference is None:
            baseline_variant = next(
                (v for v in variants if v.get("case_id") in (BASELINE_CASE_ID, "baseline_per_target_sigma")),
                variants[0],
            )
            log_path = run_dir / suite_name / "logs" / f"_global_baseline_{baseline_variant['case_id']}.log"
            baseline_reference = run_single_st_benchmark_case(
                built,
                target_hz=float(targets_hz[0]),
                variant=baseline_variant,
                mesh_level=mesh_level,
                log_path=log_path,
            )
            _write_json(run_dir / "baseline_reference.json", baseline_reference)

        suite_results[suite_name] = _run_suite(
            suite_name,
            variants,
            built=built,
            mesh_level=mesh_level,
            targets_hz=targets_hz,
            suite_dir=run_dir / suite_name,
            baseline_case=baseline_reference,
            argv=argv,
        )

    _write_json(
        run_dir / "aggregate_result.json",
        {
            "run_id": rid,
            "baseline_reference": baseline_reference,
            "suites": suite_results,
            "package_availability_path": str((run_dir / "package_availability.json").resolve()),
        },
    )

    print(f"[B3_ST_solver_bench] complete run_dir={run_dir}", flush=True)
    print(f"[B3_ST_solver_bench] package_availability={run_dir / 'package_availability.json'}", flush=True)
    print(f"[B3_ST_solver_bench] parallelism: {parallel_run_guidance()}", flush=True)
    return 0


def parallel_run_guidance() -> str:
    return (
        "Run solver benchmarks **one case at a time** (this harness is already serial). "
        "Do not launch multiple benchmark processes in parallel on the same VM: each "
        "EPSSetUp performs a sparse LU (MUMPS) on the full active operator (~100k+ DOFs "
        "on L_prod), typically holding multiple GB of factor workspace. Parallel runs "
        "contend for CPU and RAM, inflate wall time, and can trigger MUMPS INFOG(1)=-13 "
        "or PETSc error 76. Operator build (~5 min) should be done once per run_id; "
        "still avoid overlapping runs if memory is tight."
    )
