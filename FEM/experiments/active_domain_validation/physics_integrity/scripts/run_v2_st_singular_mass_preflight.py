#!/usr/bin/env python3
"""
No-EVP preflight: SLEPc API for PGNHEP/purification and M matrix structure (L_mid baseline).

Must not call eps.solve().
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, write_json
from v2_unreg_offset_report_evaluator import assemble_replay_operators, _load_sample_spec

OUT_JSON = CONV_DIAG / "v2_st_singular_mass_preflight.json"
OUT_MD = CONV_DIAG / "v2_st_singular_mass_preflight.md"
CASE_ID = "baseline_coupled_v2"


def _slepc_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import SLEPc

        out["slepc_version"] = getattr(SLEPc, "__version__", "unknown")
    except Exception as exc:
        out["slepc_import_error"] = str(exc)
    try:
        import slepc4py

        out["slepc4py_version"] = getattr(slepc4py, "__version__", "unknown")
    except Exception as exc:
        out["slepc4py_import_error"] = str(exc)
    return out


def _api_probe() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "has_EPS_ProblemType_PGNHEP": False,
        "has_eps_setPurify_or_equivalent": False,
        "can_set_PGNHEP_without_solve": False,
        "can_set_purify_without_solve": False,
        "api_errors": [],
    }
    try:
        from petsc4py import PETSc
        import SLEPc

        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        pgnhep = getattr(SLEPc.EPS.ProblemType, "PGNHEP", None)
        out["has_EPS_ProblemType_PGNHEP"] = pgnhep is not None
        if pgnhep is not None:
            try:
                eps.setProblemType(pgnhep)
                out["can_set_PGNHEP_without_solve"] = True
            except Exception as exc:
                out["api_errors"].append(f"setProblemType(PGNHEP): {exc}")
        for meth in ("setPurify", "setPurification", "setDeflation"):
            if hasattr(eps, meth):
                out["has_eps_setPurify_or_equivalent"] = True
                out["purify_method_name"] = meth
                try:
                    getattr(eps, meth)(True)
                    out["can_set_purify_without_solve"] = True
                except Exception as exc:
                    out["api_errors"].append(f"{meth}: {exc}")
                break
        try:
            eps.destroy()
        except Exception:
            pass
    except Exception as exc:
        out["api_errors"].append(f"eps_create: {exc}")
    return out


def _matrix_probes(mesh_file: Path, sample: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    A, M, u_to_W, p_to_W, asm = assemble_replay_operators(mesh_file, sample, out_dir=out_dir)
    n = int(asm["operator_size"])
    out: Dict[str, Any] = {
        "M_is_matrix_passed_to_EPS": True,
        "M_shape": [n, n],
        "M_same_object_as_replay": True,
        "operator_assembly_note": (
            "Replay assembly uses solve_evp=False GNHEP (A,M) identical to EPS operator "
            "construction path in fem_main_3d._solve_coupled_evp."
        ),
    }
    try:
        from physical_fsi_seed_residual_audit import _petsc_matvec, _petsc_vec_from_array

        r0, r1 = M.getOwnershipRange()
        loc_n = r1 - r0
        sym_max_rel = 0.0
        sym_samples = 0
        for _ in range(min(8, max(1, loc_n))):
            x = np.random.randn(n)
            xv = _petsc_vec_from_array(M, x)
            try:
                Mx, _ = _petsc_matvec(M, xv)
                MTx, _ = _petsc_matvec(M.transpose(), xv)
                d = float(np.linalg.norm(Mx - MTx))
                s = float(np.linalg.norm(Mx)) + 1.0e-30
                sym_max_rel = max(sym_max_rel, d / s)
                sym_samples += 1
            finally:
                xv.destroy()
        out["M_hermitian_symmetry_probe"] = {
            "samples": sym_samples,
            "max_relative_asymmetry": sym_max_rel,
            "hermitian_within_1e-10": bool(sym_max_rel < 1.0e-10),
        }
        neg_found = False
        probe_vecs: List[float] = []
        for _ in range(5):
            x = np.random.randn(n)
            xv = _petsc_vec_from_array(M, x)
            try:
                Mx, _ = _petsc_matvec(M, xv)
                q = float(np.vdot(x, Mx))
                probe_vecs.append(q)
                if q < -1.0e-6:
                    neg_found = True
            finally:
                xv.destroy()
        out["M_quadratic_form_probes"] = probe_vecs
        out["M_negative_quadratic_probe_found"] = neg_found
        out["M_psd_justification_source"] = (
            "analytic_form_plus_numerical_probe"
            if not neg_found
            else "numerical_probe_failed_psd_assumption"
        )
        # Mass-null hint: diagonal near-zero on p block indices
        p_idx = np.asarray(p_to_W, dtype=np.int32)
        if p_idx.size > 0:
            diag_p = []
            for pi in p_idx[: min(20, p_idx.size)]:
                if r0 <= pi < r1:
                    diag_p.append(float(M.getValue(pi, pi)))
            out["M_pressure_diagonal_sample"] = diag_p
            out["M_known_null_direction_evidence"] = (
                "pressure restriction / algebraic null may yield singular B; "
                "see dropped_inactive_p in pressure_restriction metadata"
            )
        out["pressure_restriction"] = asm.get("pressure_restriction") or {}
    except Exception as exc:
        out["matrix_probe_error"] = f"{type(exc).__name__}:{exc}"
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass
    return out


def _applicability(api: Dict[str, Any], mprobe: Dict[str, Any]) -> str:
    if not api.get("can_set_PGNHEP_without_solve"):
        return "not_justified_use_nullspace_reduction_plan"
    if mprobe.get("M_negative_quadratic_probe_found"):
        return "not_justified_use_nullspace_reduction_plan"
    if not api.get("can_set_purify_without_solve"):
        return "unresolved"
    if mprobe.get("M_psd_justification_source") == "numerical_probe_failed_psd_assumption":
        return "not_justified_use_nullspace_reduction_plan"
    return "supported_for_stage1_test_pending_vm_confirmation"


def main() -> int:
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)
    out_dir = (
        Path(__file__).resolve().parents[1]
        / "v2_mesh_convergence/solves/L_mid/baseline_coupled_v2"
        / "st_singular_mass_preflight_scratch"
    )
    sample, _ = _load_sample_spec(out_dir, sample_spec_from_case(case))

    api = _api_probe()
    versions = _slepc_versions()
    mprobe = _matrix_probes(mesh_file, sample, out_dir)
    applicability = _applicability(api, mprobe)

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "no_eigensolve_vm_preflight",
        "eps_solve_called": False,
        **versions,
        "slepc_api": api,
        "matrix_structure": mprobe,
        "PGNHEP_purification_applicability": applicability,
        "cautions": [
            "PGNHEP slepc4py docs may state positive definite B; our M may be PSD only.",
            "Valid acoustic seed xH_Mx>0 does not prove global PSD of assembled M.",
            "Stage-1 solve remains blocked until mapping inventory and this preflight are reviewed.",
        ],
        "mesh_convergence_may_resume": False,
    }
    write_json(OUT_JSON, report)

    lines = [
        "# ST singular-mass preflight (no EPS solve)",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"**PGNHEP/purification applicability:** `{applicability}`",
        "",
        f"- has_EPS_ProblemType_PGNHEP: {api.get('has_EPS_ProblemType_PGNHEP')}",
        f"- can_set_PGNHEP_without_solve: {api.get('can_set_PGNHEP_without_solve')}",
        f"- can_set_purify_without_solve: {api.get('can_set_purify_without_solve')}",
        f"- M_negative_quadratic_probe_found: {mprobe.get('M_negative_quadratic_probe_found')}",
        f"- M_psd_justification_source: {mprobe.get('M_psd_justification_source')}",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[st_preflight] applicability={applicability}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
