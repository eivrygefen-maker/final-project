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
B3_ST_CASE_FILTER_ARG = "--B3-ST-solver-benchmark-cases"

# Minimum EPSSetUp wall time (s) for a valid MUMPS baseline on L_prod-scale operators.
BASELINE_MIN_SETUP_SECONDS = 60.0

BENCHMARK_ROOT = CONV_DIAG / "solver_benchmarks"
ALLOWED_SUITES = frozenset(
    {
        "factor_solver",
        "mumps_policy",
        "mumps_ordering",
        "eps_params",
        "interval_slicing",
        "all",
    }
)

# Suites run for `--B3-ST-solver-benchmark-suite all` (eps_params omitted: solve phase is ~5% of ST).
ALL_SUITE_ORDER = ("factor_solver", "mumps_policy", "mumps_ordering", "interval_slicing")

MUMPS_POLICY_CASE_BASELINE = "mumps_default"
MUMPS_ORDERING_ICNTL7_PROBE_VALUES = (0, 7, 3, 4, 5)  # auto, auto(7), Scotch, PORD, METIS
MUMPS_ORDERING_ICNTL7_LABELS = {
    0: "icntl7_0_amd_default",
    7: "icntl7_7_automatic",
    3: "icntl7_3_scotch",
    4: "icntl7_4_pord",
    5: "icntl7_5_metis",
}

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


def _parse_case_filter(argv: Sequence[str]) -> Optional[frozenset[str]]:
    raw = _parse_arg_value(argv, B3_ST_CASE_FILTER_ARG)
    if not raw:
        return None
    ids = frozenset(part.strip() for part in str(raw).split(",") if part.strip())
    return ids if ids else None


def _collect_benchmark_petsc_option_keys() -> Tuple[str, ...]:
    keys: set[str] = set()
    for policy_key in ("mumps_default", "mumps_relaxed", "mumps_maximum", "mumps_fast_incore"):
        keys.update(_benchmark_mumps_policy_spec(policy_key).keys())
    for policy in ("default", "L_prod_relaxed", "L_prod_maximum"):
        keys.update(st_scaling._st_mumps_policy_spec(policy).keys())
    keys.update(
        {
            "st_pc_factor_mat_solver_type",
            "st_pc_factor_shift_type",
            "st_pc_factor_shift_amount",
            "st_ksp_type",
            "st_pc_type",
            "mat_mumps_icntl_3",
            "mat_mumps_icntl_4",
            "mat_mumps_icntl_6",
            "mat_mumps_icntl_7",
            "mat_mumps_icntl_11",
            "mat_mumps_icntl_12",
            "mat_mumps_icntl_14",
            "mat_mumps_icntl_22",
            "mat_mumps_icntl_23",
            "mat_mumps_icntl_24",
        }
    )
    return tuple(sorted(keys))


def _reset_petsc_options_before_case(case_id: str) -> Dict[str, Any]:
    """Clear ST/MUMPS PETSc options so prior benchmark cases cannot leak settings."""
    opts = PETSc.Options()
    cleared: List[str] = []
    failed: List[str] = []
    for key in _collect_benchmark_petsc_option_keys():
        try:
            if hasattr(opts, "delValue"):
                opts.delValue(key)
            else:
                opts[key] = None
            cleared.append(key)
        except Exception as exc:
            failed.append(f"{key}:{type(exc).__name__}")
    return {
        "case_id": str(case_id),
        "petsc_options_reset_attempted": len(_collect_benchmark_petsc_option_keys()),
        "petsc_options_cleared_count": len(cleared),
        "petsc_options_clear_failures": failed[:16],
    }


def _verify_factor_solver_effective(pc: Any, *, requested: str, meta_out: Dict[str, Any]) -> None:
    effective = audit._b3_ciss_pc_factor_solver_effective_label(pc)
    meta_out["benchmark_factor_solver_requested"] = str(requested)
    meta_out["benchmark_factor_solver_effective"] = effective
    req = str(requested).strip().lower()
    eff = str(effective or "").strip().lower()
    ok = bool(eff and (req in eff or eff in req))
    meta_out["factor_solver_verification_pass"] = bool(ok)
    if not ok:
        raise RuntimeError(
            f"factor_solver_verification_failed:requested={requested!r} effective={effective!r}"
        )


def _baseline_case_is_valid(case: Optional[Dict[str, Any]]) -> bool:
    if case is None or case.get("skipped"):
        return False
    if not case.get("setup_succeeded") or not case.get("solve_succeeded"):
        return False
    try:
        setup_s = float(case.get("setup_elapsed_seconds") or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(setup_s >= BASELINE_MIN_SETUP_SECONDS)


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


def _create_probe_assembled_aij(*, n: int = 12, comm: Any = None) -> Any:
    """Small diagonally dominant assembled AIJ matrix (valid for PC LU probes)."""
    mat_comm = comm if comm is not None else PETSc.COMM_WORLD
    n_local = int(n)
    A = PETSc.Mat().create(comm=mat_comm)
    A.setSizes([n_local, n_local])
    A.setType("aij")
    try:
        A.setPreallocationNNZ(max(4, n_local))
    except Exception:
        pass
    A.setUp()
    for i in range(n_local):
        A.setValue(i, i, 2.0 + float(i))
        if i > 0:
            A.setValue(i, i - 1, -0.05)
        if i < n_local - 1:
            A.setValue(i, i + 1, -0.05)
    A.assemble()
    return A


def _probe_pc_lu_factor_solver(
    pkg: str,
    *,
    mumps_icntl_7: Optional[int] = None,
    mumps_extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Probe one factor solver on an assembled AIJ matrix."""
    entry: Dict[str, Any] = {"requested": pkg, "available": False, "matrix_assembled": True}
    A = None
    pc = None
    try:
        A = _create_probe_assembled_aij()
        pc = PETSc.PC().create(comm=PETSc.COMM_WORLD)
        pc.setOperators(A)
        pc.setType("lu")
        pc.setFactorSolverType(str(pkg))
        if str(pkg).lower() == "mumps":
            petsc_opts = PETSc.Options()
            if mumps_icntl_7 is not None:
                petsc_opts["mat_mumps_icntl_7"] = int(mumps_icntl_7)
            for key, val in (mumps_extra_options or {}).items():
                petsc_opts[str(key)] = val
        pc.setUp()
        effective = None
        try:
            effective = str(pc.getFactorSolverType())
        except Exception:
            effective = None
        entry["available"] = True
        entry["effective_factor_solver"] = effective
        if mumps_icntl_7 is not None:
            entry["mumps_icntl_7"] = int(mumps_icntl_7)
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
    return entry


def probe_mumps_ordering_icntl7() -> Dict[str, Any]:
    """Probe MUMPS ICNTL(7) ordering values on a small assembled matrix."""
    out: Dict[str, Any] = {"ordering_probes": {}, "available_icntl7_values": []}
    for icntl7 in MUMPS_ORDERING_ICNTL7_PROBE_VALUES:
        label = MUMPS_ORDERING_ICNTL7_LABELS.get(int(icntl7), f"icntl7_{icntl7}")
        probe = _probe_pc_lu_factor_solver("mumps", mumps_icntl_7=int(icntl7))
        probe["label"] = label
        out["ordering_probes"][str(icntl7)] = probe
        if probe.get("available"):
            out["available_icntl7_values"].append(int(icntl7))
    return out


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
        "mumps_ordering_probe": {},
        "probe_matrix_note": "assembled_aij_with_diagonal_and_offdiagonal_values",
    }
    try:
        out["petsc_version"] = str(PETSc.Sys.getVersion())
        out["slepc_version"] = str(getattr(SLEPc, "__version__", "unknown"))
        out["mpi_comm_world_size"] = int(MPI.COMM_WORLD.Get_size())
        A = _create_probe_assembled_aij()
        out["default_mat_type_on_comm_world"] = str(A.getType())
        out["probe_matrix_assembled"] = True
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
        out["factor_solvers"][pkg] = _probe_pc_lu_factor_solver(pkg)

    out["mumps_ordering_probe"] = probe_mumps_ordering_icntl7()
    return out


def _benchmark_mumps_policy_spec(policy_key: str) -> Dict[str, Any]:
    """Dev-benchmark MUMPS PETSc option sets (st_ prefix); does not change production specs."""
    base_default = dict(st_scaling._st_mumps_policy_spec("default"))
    if policy_key == "mumps_fast_incore":
        return {
            **base_default,
            "st_mat_mumps_icntl_14": 400,
            "st_mat_mumps_icntl_24": 0,
            "st_mat_mumps_icntl_22": 0,
            "st_mat_mumps_icntl_7": 0,
            "st_mat_mumps_icntl_4": 0,
        }
    mapped = {
        "mumps_default": "default",
        "mumps_relaxed": "L_prod_relaxed",
        "mumps_maximum": "L_prod_maximum",
    }
    if policy_key in mapped:
        return dict(st_scaling._st_mumps_policy_spec(mapped[policy_key]))
    raise ValueError(f"unknown_benchmark_mumps_policy_key={policy_key!r}")


def _apply_benchmark_mumps_petsc_options(
    policy_key: str,
    *,
    icntl7_override: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec = _benchmark_mumps_policy_spec(policy_key)
    if icntl7_override is not None:
        spec["st_mat_mumps_icntl_7"] = int(icntl7_override)
    if extra_options:
        spec.update(dict(extra_options))
    petsc_opts = PETSc.Options()
    for key, val in spec.items():
        petsc_opts[str(key)] = val
    return {
        "mumps_policy_key": str(policy_key),
        "mumps_icntl_7_effective": spec.get("st_mat_mumps_icntl_7"),
        "petsc_options_written": dict(spec),
    }


def _mumps_policy_chain_for_variant(variant: Dict[str, Any], mesh_level: str) -> List[Optional[str]]:
    """Single policy for benchmark cases; retry chain only when explicitly requested."""
    if variant.get("use_production_st_configure"):
        return [str(variant.get("production_mumps_policy", "default"))]
    if variant.get("mumps_policy_key"):
        return [str(variant["mumps_policy_key"])]
    if variant.get("use_mumps_retry_chain"):
        return [str(p) for p in st_scaling._mumps_policy_chain(mesh_level)]
    if str(variant.get("factor_solver", BASELINE_FACTOR)) == "mumps":
        return ["mumps_default"]
    return [None]


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

    if variant.get("use_production_st_configure"):
        policy = str(variant.get("production_mumps_policy", "default"))
        meta_out["configure_path"] = "st_scaling_configure_eps_st_sinvert"
        meta_out["production_mumps_policy"] = policy
        meta_out["factor_solver_requested"] = "mumps"
        cfg = st_scaling._configure_eps_st_sinvert(
            eps,
            A_active,
            M_active,
            target_hz=float(target_hz),
            target_lambda=float(target_lambda),
            mumps_policy=policy,
        )
        meta_out.update(cfg)
        st = eps.getST()
        ksp = st.getKSP()
        pc = ksp.getPC()
        meta_out["benchmark_KSP_type_effective"] = str(ksp.getType())
        meta_out["benchmark_PC_type_effective"] = str(pc.getType())
        meta_out["benchmark_ST_type_effective"] = str(st.getType())
        _verify_factor_solver_effective(pc, requested="mumps", meta_out=meta_out)
        meta_out["petsc_options_written"] = dict(cfg.get("petsc_options_written") or {})
        return

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

    if factor_solver == "mumps" or variant.get("mumps_policy_key"):
        policy_key = str(variant.get("mumps_policy_key", "mumps_default"))
        mumps_written = _apply_benchmark_mumps_petsc_options(
            policy_key,
            icntl7_override=variant.get("mumps_icntl_7_override"),
            extra_options=variant.get("mumps_petsc_options_extra"),
        )
        meta_out.update(mumps_written)

    meta_out["configure_path"] = "benchmark_configure_eps_st_benchmark"
    _verify_factor_solver_effective(pc, requested=factor_solver, meta_out=meta_out)


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
        "mumps_policy_key": variant.get("mumps_policy_key"),
        "petsc_options_written": None,
        "configure_path": None,
        "factor_solver_verification_pass": None,
        "petsc_options_reset": None,
        "variant": dict(variant),
    }

    result["petsc_options_reset"] = _reset_petsc_options_before_case(case_id)

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
        policy_chain = _mumps_policy_chain_for_variant(variant, mesh_level)
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
                variant_run = dict(variant)
                if policy is not None and not variant.get("use_production_st_configure"):
                    variant_run["mumps_policy_key"] = str(policy)
                elif variant.get("use_production_st_configure") and policy is not None:
                    variant_run["production_mumps_policy"] = str(policy)
                configure_eps_st_benchmark(
                    eps,
                    A_active,
                    M_active,
                    target_hz=float(target_hz),
                    target_lambda=target_lambda,
                    variant=variant_run,
                    meta_out=setup_meta,
                )
                t0 = time.perf_counter()
                eps.setUp()
                setup_s = time.perf_counter() - t0
                result["setup_succeeded"] = True
                result["setup_elapsed_seconds"] = _safe_float(setup_s)
                result["mumps_policy_key_effective"] = str(policy)
                result["mumps_policy_effective"] = str(policy)
                result["petsc_options_written"] = dict(setup_meta.get("petsc_options_written") or {})
                result["configure_meta"] = setup_meta
                result["configure_path"] = setup_meta.get("configure_path")
                result["factor_solver_verification_pass"] = setup_meta.get("factor_solver_verification_pass")
                result["benchmark_factor_solver_effective"] = setup_meta.get("benchmark_factor_solver_effective")
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
                result["failure_exception_type"] = type(last_exc).__name__
            if not result.get("failure_reason") and last_exc is not None:
                result["failure_reason"] = f"{type(last_exc).__name__}:{last_exc}"
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


def _summary_table_md(
    cases: List[Dict[str, Any]],
    *,
    baseline_st: Optional[float],
    suite_baseline_case_id: str,
) -> str:
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
        if str(c.get("case_id")) == str(suite_baseline_case_id):
            parity_s = "baseline"
        elif parity_ok is True:
            parity_s = "match"
        elif parity_ok is False:
            parity_s = "**DIFF**"
        elif c.get("parity", {}).get("parity_skipped"):
            parity_s = "n/a (baseline invalid)"
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
    mumps_probe = (package_probe.get("factor_solvers") or {}).get("mumps") or {}
    variants: List[Dict[str, Any]] = [
        {
            "case_id": BASELINE_CASE_ID,
            "suite": "factor_solver",
            "factor_solver": BASELINE_FACTOR,
            "use_production_st_configure": True,
            "production_mumps_policy": "default",
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
            "factor_solver_available": True,
            "probe_reports_mumps_available": bool(mumps_probe.get("available")),
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


def _mumps_policy_variants() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": MUMPS_POLICY_CASE_BASELINE,
            "suite": "mumps_policy",
            "factor_solver": BASELINE_FACTOR,
            "mumps_policy_key": "mumps_default",
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
        },
        {
            "case_id": "mumps_relaxed",
            "suite": "mumps_policy",
            "factor_solver": BASELINE_FACTOR,
            "mumps_policy_key": "mumps_relaxed",
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
        },
        {
            "case_id": "mumps_maximum",
            "suite": "mumps_policy",
            "factor_solver": BASELINE_FACTOR,
            "mumps_policy_key": "mumps_maximum",
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
        },
        {
            "case_id": "mumps_fast_incore",
            "suite": "mumps_policy",
            "factor_solver": BASELINE_FACTOR,
            "mumps_policy_key": "mumps_fast_incore",
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
        },
    ]


def _mumps_ordering_variants(package_probe: Dict[str, Any]) -> List[Dict[str, Any]]:
    ordering_probe = package_probe.get("mumps_ordering_probe") or {}
    available = set(int(x) for x in (ordering_probe.get("available_icntl7_values") or []))
    variants: List[Dict[str, Any]] = [
        {
            "case_id": f"{MUMPS_POLICY_CASE_BASELINE}_{MUMPS_ORDERING_ICNTL7_LABELS[0]}",
            "suite": "mumps_ordering",
            "factor_solver": BASELINE_FACTOR,
            "mumps_policy_key": MUMPS_POLICY_CASE_BASELINE,
            "mumps_icntl_7_override": 0,
            "nev": BASELINE_NEV,
            "ncv": BASELINE_NCV,
        },
    ]
    for icntl7 in MUMPS_ORDERING_ICNTL7_PROBE_VALUES:
        if int(icntl7) == 0:
            continue
        label = MUMPS_ORDERING_ICNTL7_LABELS.get(int(icntl7), f"icntl7_{icntl7}")
        ok = int(icntl7) in available
        variants.append(
            {
                "case_id": f"{MUMPS_POLICY_CASE_BASELINE}_{label}",
                "suite": "mumps_ordering",
                "factor_solver": BASELINE_FACTOR,
                "mumps_policy_key": MUMPS_POLICY_CASE_BASELINE,
                "mumps_icntl_7_override": int(icntl7),
                "nev": BASELINE_NEV,
                "ncv": BASELINE_NCV,
                "skip": not ok,
                "skip_reason": None if ok else f"mumps_ordering_probe_unavailable_icntl7={icntl7}",
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


def _suite_baseline_case_id(suite_name: str) -> str:
    if suite_name == "mumps_ordering":
        return f"{MUMPS_POLICY_CASE_BASELINE}_{MUMPS_ORDERING_ICNTL7_LABELS[0]}"
    if suite_name == "mumps_policy":
        return MUMPS_POLICY_CASE_BASELINE
    if suite_name == "interval_slicing":
        return "baseline_per_target_sigma"
    return BASELINE_CASE_ID


def _run_suite(
    suite_name: str,
    variants: List[Dict[str, Any]],
    *,
    built: Dict[str, Any],
    mesh_level: str,
    targets_hz: List[float],
    suite_dir: Path,
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

    suite_baseline_id = _suite_baseline_case_id(suite_name)
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
        cases.append(case_result)
        _write_json(logs_dir / f"{case_id}_result.json", case_result)

    baseline_in_suite = next((c for c in cases if str(c.get("case_id")) == suite_baseline_id), None)
    if baseline_in_suite is None and cases:
        baseline_in_suite = cases[0]
        suite_baseline_id = str(baseline_in_suite.get("case_id"))

    baseline_valid = _baseline_case_is_valid(baseline_in_suite)
    suite_status = "PASS" if baseline_valid else "FAILED_BASELINE"

    baseline_st = None
    baseline_setup = None
    baseline_solve = None
    if baseline_in_suite is not None:
        baseline_st = baseline_in_suite.get("st_total_elapsed_seconds")
        baseline_setup = baseline_in_suite.get("setup_elapsed_seconds")
        baseline_solve = baseline_in_suite.get("solve_elapsed_seconds")

    for c in cases:
        cid = str(c.get("case_id"))
        if not baseline_valid:
            c["parity"] = {
                "parity_pass": None,
                "parity_skipped": True,
                "parity_skipped_reason": "baseline_failed_or_invalid",
                "baseline_reference_case_id": suite_baseline_id,
                "baseline_failure_reason": (baseline_in_suite or {}).get("failure_reason"),
                "baseline_setup_succeeded": (baseline_in_suite or {}).get("setup_succeeded"),
                "baseline_solve_succeeded": (baseline_in_suite or {}).get("solve_succeeded"),
                "baseline_st_total_seconds": _safe_float(baseline_st),
            }
        elif cid == suite_baseline_id:
            c["parity"] = {
                "parity_pass": True,
                "parity_notes": ["in_suite_baseline_reference"],
                "speedup_vs_baseline_st_total": 1.0,
                "baseline_reference_case_id": suite_baseline_id,
            }
        else:
            c["parity"] = _parity_vs_baseline(c, baseline_in_suite)
            c["parity"]["baseline_reference_case_id"] = suite_baseline_id
            c["parity"]["baseline_st_total_seconds"] = _safe_float(baseline_st)

    suite_result = {
        "suite": suite_name,
        "suite_status": suite_status,
        "mesh_level": mesh_level,
        "targets_hz": list(targets_hz),
        "baseline_reference_source": "in_suite_case",
        "baseline_valid": bool(baseline_valid),
        "baseline_case_id": suite_baseline_id,
        "baseline_failure_reason": (baseline_in_suite or {}).get("failure_reason"),
        "baseline_setup_elapsed_seconds": _safe_float(baseline_setup),
        "baseline_solve_elapsed_seconds": _safe_float(baseline_solve),
        "baseline_st_total_seconds": _safe_float(baseline_st) if baseline_valid else None,
        "cases": cases,
        "summary_table_markdown": _summary_table_md(
            cases,
            baseline_st=baseline_st,
            suite_baseline_case_id=suite_baseline_id,
        ),
    }
    _write_json(suite_dir / "result.json", suite_result)

    md_lines = [
        f"# ST solver benchmark — {suite_name}",
        "",
        f"- Mesh: `{mesh_level}`",
        f"- Target Hz: `{target_hz}`",
        f"- Suite status: `{suite_status}`",
        f"- Baseline case (in-suite): `{suite_baseline_id}`",
        f"- Baseline valid: `{baseline_valid}`",
        f"- Baseline setup (s): `{baseline_setup}`",
        f"- Baseline solve (s): `{baseline_solve}`",
        f"- Baseline ST total (s): `{suite_result.get('baseline_st_total_seconds')}`",
        f"- Baseline failure: `{(baseline_in_suite or {}).get('failure_reason')}`",
        "",
        "## Summary",
        "",
        suite_result["summary_table_markdown"],
        "",
        "## PETSc / MUMPS options (baseline case)",
        "",
    ]
    if baseline_in_suite is not None:
        md_lines.append(f"```json\n{json.dumps(baseline_in_suite.get('petsc_options_written') or {}, indent=2)}\n```")
        md_lines.append("")
    md_lines.extend(["## Correctness / parity", ""])
    for c in cases:
        p = c.get("parity") or {}
        opts = c.get("petsc_options_written") or {}
        md_lines.append(
            f"- **{c.get('case_id')}**: parity_pass={p.get('parity_pass')}; "
            f"setup={c.get('setup_elapsed_seconds')}s; solve={c.get('solve_elapsed_seconds')}s; "
            f"st_total={c.get('st_total_elapsed_seconds')}s; "
            f"accepted_n={c.get('accepted_mode_count_in_interval')}; "
            f"accepted_hz={c.get('accepted_frequencies_hz')}; "
            f"mumps_policy={c.get('mumps_policy_key_effective')}; "
            f"notes={p.get('parity_notes')}"
        )
        if opts and str(c.get("case_id")) != suite_baseline_id:
            md_lines.append(f"  - ICNTL/options: `{opts}`")
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
        suites_to_run = list(ALL_SUITE_ORDER)
    else:
        suites_to_run = [suite]

    suite_results: Dict[str, Any] = {}

    case_filter = _parse_case_filter(argv)

    for suite_name in suites_to_run:
        if suite_name == "factor_solver":
            variants = _factor_solver_variants(package_probe)
        elif suite_name == "mumps_policy":
            variants = _mumps_policy_variants()
        elif suite_name == "mumps_ordering":
            variants = _mumps_ordering_variants(package_probe)
        elif suite_name == "eps_params":
            variants = _eps_params_variants()
        else:
            variants = _interval_slicing_variants()

        if case_filter is not None:
            variants = [v for v in variants if str(v.get("case_id")) in case_filter]
            if not variants:
                print(
                    f"[B3_ST_solver_bench] no variants match {B3_ST_CASE_FILTER_ARG}={sorted(case_filter)} "
                    f"in suite={suite_name}",
                    flush=True,
                )
                continue

        suite_results[suite_name] = _run_suite(
            suite_name,
            variants,
            built=built,
            mesh_level=mesh_level,
            targets_hz=targets_hz,
            suite_dir=run_dir / suite_name,
            argv=argv,
        )

    baseline_reference: Optional[Dict[str, Any]] = None
    for preferred_suite in ("mumps_policy", "factor_solver"):
        suite_data = suite_results.get(preferred_suite) or {}
        baseline_id = str(suite_data.get("baseline_case_id") or MUMPS_POLICY_CASE_BASELINE)
        for case in suite_data.get("cases") or []:
            if str(case.get("case_id")) == baseline_id and _baseline_case_is_valid(case):
                baseline_reference = case
                break
        if baseline_reference is not None:
            break
    if baseline_reference is not None:
        baseline_reference = {
            **baseline_reference,
            "baseline_reference_source": "in_suite_case_from_mumps_policy_or_factor_solver",
            "canonical_baseline_case_id": MUMPS_POLICY_CASE_BASELINE,
        }
        _write_json(run_dir / "baseline_reference.json", baseline_reference)

    _write_json(
        run_dir / "aggregate_result.json",
        {
            "run_id": rid,
            "baseline_reference": baseline_reference,
            "baseline_reference_note": (
                "Parity and speedup use the in-suite baseline case row "
                f"({MUMPS_POLICY_CASE_BASELINE} for mumps_policy/mumps_ordering; "
                f"{BASELINE_CASE_ID} for factor_solver). "
                "No separate pre-suite baseline run."
            ),
            "suites": suite_results,
            "package_availability_path": str((run_dir / "package_availability.json").resolve()),
            "recommended_next_suite": "mumps_policy",
            "eps_params_deferred_reason": "EPSSolve is ~5% of ST total; tune MUMPS/setup first.",
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
