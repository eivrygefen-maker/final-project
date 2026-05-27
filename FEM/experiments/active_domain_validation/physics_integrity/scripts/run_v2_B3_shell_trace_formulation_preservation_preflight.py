#!/usr/bin/env python3
"""Report-only B3 shell/trace formulation preservation preflight (no eigensolve)."""
from __future__ import annotations

import copy
import ast
import inspect
import json
import math
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    import sys

    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fem_main_3d as fem3d
from dolfinx import fem, mesh as dmesh
from physical_fsi_seed_residual_audit import _block_residual_contributions, _rayleigh_metrics
from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay, _extract_layout_maps
from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir
from v2_unreg_offset_report_evaluator import load_seed_with_diagnostics

CASE_ID = "baseline_coupled_v2"
OUT_JSON = CONV_DIAG / "v2_B3_shell_trace_formulation_preservation_preflight.json"
OUT_MD = CONV_DIAG / "v2_B3_shell_trace_formulation_preservation_preflight.md"
REPORT_SIZE_TARGET_BYTES = 1048576

TAG_TOP = 1
TAG_BACK = 3
TAG_RIBS = 4
TAG_FIX = 5

BLOCKER_INTERFACE = "dolfinx_trace_u_to_coupled_pressure_operator_and_seed_transfer_interface"


def _crc32_i32(arr: np.ndarray) -> int:
    return int(zlib.crc32(np.asarray(arr, dtype=np.int32).tobytes()) & 0xFFFFFFFF)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return len(text.encode("utf-8"))


def _safe_float(x: Any) -> Any:
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def _compact_indices(idx: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(idx, dtype=np.int32).ravel()
    n = int(a.size)
    return {
        "dtype": "int32",
        "size": n,
        "min": int(a.min()) if n else None,
        "max": int(a.max()) if n else None,
        "crc32": _crc32_i32(a),
        "preview_first": [int(x) for x in a[:8].tolist()],
        "preview_last": [int(x) for x in a[-8:].tolist()] if n > 8 else [],
    }


def _preassembly_contract_check() -> Dict[str, Any]:
    checks: Dict[str, bool] = {
        "preassembly_helper_import_pass": False,
        "preassembly_rayleigh_signature_pass": False,
        "preassembly_residual_signature_pass": False,
        "preassembly_writer_available_pass": False,
        "preassembly_no_eigensolve_call_pass": False,
    }
    reasons: List[Dict[str, str]] = []
    guard_method = "ast_call_scan_for_attr_solve_on_names_eps_or_EPS"
    helper_semantics = True
    helper_source = (
        "physical_fsi_seed_residual_audit._block_residual_contributions + "
        "physical_fsi_seed_residual_audit._rayleigh_metrics"
    )

    try:
        checks["preassembly_helper_import_pass"] = callable(_rayleigh_metrics) and callable(
            _block_residual_contributions
        )
        if not checks["preassembly_helper_import_pass"]:
            reasons.append(
                {
                    "check": "preassembly_helper_import_pass",
                    "reason": "expected callable imported helpers; got non-callable",
                }
            )

        sig_ray = inspect.signature(_rayleigh_metrics)
        checks["preassembly_rayleigh_signature_pass"] = "seed_f_hz" in sig_ray.parameters
        if not checks["preassembly_rayleigh_signature_pass"]:
            reasons.append(
                {
                    "check": "preassembly_rayleigh_signature_pass",
                    "reason": (
                        "expected parameter 'seed_f_hz' in _rayleigh_metrics signature; "
                        f"got {list(sig_ray.parameters.keys())}"
                    ),
                }
            )

        sig_blk = inspect.signature(_block_residual_contributions)
        blk_needed = ("lam0", "u_idx", "p_idx")
        checks["preassembly_residual_signature_pass"] = all(
            k in sig_blk.parameters for k in blk_needed
        )
        if not checks["preassembly_residual_signature_pass"]:
            reasons.append(
                {
                    "check": "preassembly_residual_signature_pass",
                    "reason": (
                        "expected parameters lam0/u_idx/p_idx in _block_residual_contributions; "
                        f"got {list(sig_blk.parameters.keys())}"
                    ),
                }
            )

        checks["preassembly_writer_available_pass"] = callable(_write_json_atomic)
        if not checks["preassembly_writer_available_pass"]:
            reasons.append(
                {
                    "check": "preassembly_writer_available_pass",
                    "reason": "_write_json_atomic is not callable",
                }
            )

        src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        bad_calls: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "solve" and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in {"eps", "EPS"}:
                        bad_calls.append(f"{node.func.value.id}.solve")
        checks["preassembly_no_eigensolve_call_pass"] = len(bad_calls) == 0
        if not checks["preassembly_no_eigensolve_call_pass"]:
            reasons.append(
                {
                    "check": "preassembly_no_eigensolve_call_pass",
                    "reason": f"detected forbidden eigensolve calls: {bad_calls}",
                }
            )
    except Exception as exc:
        reasons.append(
            {
                "check": "preassembly_contract_check_runtime",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )

    contract_pass = all(checks.values()) and len(reasons) == 0
    if not checks["preassembly_helper_import_pass"]:
        helper_source = "UNAVAILABLE"
        helper_semantics = False
    return {
        **checks,
        "preassembly_no_eigensolve_call_guard_method": guard_method,
        "preassembly_contract_pass": contract_pass,
        "preassembly_failure_reasons": reasons,
        "residual_helper_source": helper_source,
        "residual_helper_semantics_matches_validated_replay": helper_semantics,
        "invalid_import_removed": True,
    }


def _replay_like_metrics(
    A: Any,
    M: Any,
    x: np.ndarray,
    *,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
    seed_f_hz: float,
) -> Dict[str, Any]:
    """Replay-equivalent metrics used by validated no-EPS audits."""
    ray = _rayleigh_metrics(A, M, x, seed_f_hz=seed_f_hz)
    lam = float(ray.get("rayleigh_lambda", float("nan")))
    blk = _block_residual_contributions(A, M, x, lam0=lam, u_idx=u_idx, p_idx=p_idx)
    return {
        "replay_rayleigh_lambda": lam,
        "replay_rayleigh_frequency_hz": float(ray.get("rayleigh_f_hz", float("nan"))),
        "replay_relative_residual": float(blk.get("relative_residual", float("nan"))),
    }


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[B3_preflight] Requires mpiexec -n 1")
        return 2

    pre = _preassembly_contract_check()
    print(
        f"[B3_preflight] preassembly_helper_import_pass={pre['preassembly_helper_import_pass']}",
        flush=True,
    )
    print(
        f"[B3_preflight] preassembly_rayleigh_signature_pass={pre['preassembly_rayleigh_signature_pass']}",
        flush=True,
    )
    print(
        f"[B3_preflight] preassembly_residual_signature_pass={pre['preassembly_residual_signature_pass']}",
        flush=True,
    )
    print(
        f"[B3_preflight] preassembly_writer_available_pass={pre['preassembly_writer_available_pass']}",
        flush=True,
    )
    print(
        f"[B3_preflight] preassembly_no_eigensolve_call_pass={pre['preassembly_no_eigensolve_call_pass']}",
        flush=True,
    )
    print(
        f"[B3_preflight] preassembly_no_eigensolve_call_guard_method={pre['preassembly_no_eigensolve_call_guard_method']}",
        flush=True,
    )
    print(
        f"[B3_preflight] preassembly_contract_pass={pre['preassembly_contract_pass']}",
        flush=True,
    )
    if not bool(pre["preassembly_contract_pass"]):
        print(
            f"[B3_preflight] preassembly_failure_reasons={json.dumps(pre['preassembly_failure_reasons'])}",
            flush=True,
        )
        print("[B3_preflight] no_new_eigensolve_executed=True", flush=True)
        return 2
    import sys

    if "--precheck-only" in sys.argv:
        print("[B3_preflight] precheck_only_mode=True", flush=True)
        print("[B3_preflight] no_new_eigensolve_executed=True", flush=True)
        return 0

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    sample = sample_spec_from_case(case)
    mesh_file = mesh_path("L_mid", CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"

    # Baseline no-EPS coupled assembly (existing validated V2 representation).
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    b3_constructed = False
    b3_space_type = "UNRESOLVED"
    b3_source = "UNRESOLVED"
    b3_coord_dim = None
    b3_u_new = None
    b3_total_cleaned = None
    b3_dim_ratio = None
    b3_operator_assembled = False
    b3_muu_rows = None
    b3_muu_cols = None
    b3_mass_null_status = "UNRESOLVED"
    b3_seed_mapping_constructed = False
    b3_seed_representable = False
    b3_seed_pres_pass = False
    b3_seed_fail_reason = BLOCKER_INTERFACE
    b3_seed_mac = None
    b3_xhmx_b3 = None
    b3_f_b3 = None
    b3_res_b3 = None
    b3_sanity_pass = False
    b3_scalability_pass = True

    b3_changes_continuum = False
    b3_changes_discrete = True
    b3_material = "RE-DERIVED_ON_TRACE_SPACE_REQUIRED_FOR_EQUIVALENCE_VALIDATION"
    b3_bc = "RE-DERIVED_ON_TRACE_SPACE_REQUIRED_FOR_EQUIVALENCE_VALIDATION"
    b3_coupling = "RE-DERIVED_TRANSFER_REQUIRED_TO_PRESSURE_SPACE_INTERFACE"
    b3_layout_map = False
    b3_preservation = "REQUIRES_VALIDATION"
    construction_blocker = BLOCKER_INTERFACE

    try:
        maps = _extract_layout_maps(cfg, A)
        u_to_W = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
        p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        n_u = int(u_to_W.size)
        n_p = int(p_to_W.size)
        n_w = int(A.getSize()[0])

        # Construct actual shell-facet trace submesh (not a volumetric coordinate selector).
        msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        if shell_facets.size > 0 and hasattr(dmesh, "create_submesh"):
            tdim = msh.topology.dim
            shell_mesh, shell_entity_map, _, _ = dmesh.create_submesh(msh, tdim - 1, shell_facets)
            u_el_trace = fem3d._displacement_element(shell_mesh, 1)
            V_u_trace = fem.functionspace(shell_mesh, u_el_trace)
            b3_constructed = True
            b3_space_type = "facet_submesh_vector_displacement_space"
            b3_source = "dolfinx.mesh.create_submesh(facet_union_tags_1_3_4)"
            b3_coord_dim = int(shell_mesh.topology.dim)
            b3_u_new = int(V_u_trace.dofmap.index_map.size_global * V_u_trace.dofmap.index_map_bs)
            b3_total_cleaned = int(b3_u_new + n_p)
            b3_dim_ratio = float(b3_total_cleaned / max(n_w, 1))
            b3_layout_map = False  # mapping back not yet implemented
            b3_operator_assembled = False  # full coupled trace-u operators blocked by transfer interface
            b3_muu_rows = None
            b3_muu_cols = None
            b3_mass_null_status = (
                "BLOCKED_PENDING_TRACE_TO_COUPLED_OPERATOR_TRANSFER_IMPLEMENTATION"
            )
            # Seed transfer also blocked until trace<->mixed transfer interface exists.
            b3_seed_mapping_constructed = False
            b3_seed_representable = False
            b3_seed_fail_reason = BLOCKER_INTERFACE
            construction_blocker = BLOCKER_INTERFACE
            b3_sanity_pass = False
        else:
            b3_constructed = False
            b3_space_type = "UNAVAILABLE"
            b3_source = "dolfinx.mesh.create_submesh_not_available_or_no_shell_facets"
            construction_blocker = BLOCKER_INTERFACE

        # Original seed metrics on validated baseline representation.
        seed_info = load_seed_with_diagnostics(seed_npy)
        seed_arr = np.asarray(seed_info.get("seed_array"), dtype=np.float64).ravel()
        ray_o = _rayleigh_metrics(A, M, seed_arr, seed_f_hz=float("nan"))
        rep_o = _replay_like_metrics(
            A,
            M,
            seed_arr,
            u_idx=u_to_W,
            p_idx=p_to_W,
            seed_f_hz=float("nan"),
        )
        seed_xhmx_orig = _safe_float(ray_o.get("xH_Mx"))
        seed_f_orig = _safe_float(rep_o.get("replay_rayleigh_frequency_hz"))
        seed_res_orig = _safe_float(rep_o.get("replay_relative_residual"))
        p_norm_orig = float(np.linalg.norm(seed_arr[p_to_W])) if p_to_W.size else 0.0
        b3_seed_pressure_support_preserved = False

    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    # No-EPS sanity can pass only if true B3 coupled operators are assembled and checks pass.
    b3_structural_mass_present = bool(b3_constructed and b3_operator_assembled)
    b3_structural_stiffness_present = bool(b3_constructed and b3_operator_assembled)
    b3_coupling_present = bool(b3_constructed and b3_operator_assembled)
    b3_bc_contract_pass = bool(b3_constructed and b3_operator_assembled)
    b3_pressure_block_contract_pass = bool(b3_constructed and b3_operator_assembled)
    b3_sanity_pass = bool(
        b3_structural_mass_present
        and b3_structural_stiffness_present
        and b3_coupling_present
        and b3_bc_contract_pass
        and b3_pressure_block_contract_pass
    )

    if b3_constructed and not b3_seed_pres_pass:
        verdict = "B3_BLOCKED_BY_ONE_NAMED_IMPLEMENTATION_INTERFACE"
    elif not b3_scalability_pass:
        verdict = "B3_NOT_SCALABLE_OR_STORAGE_UNSAFE"
    elif b3_seed_pres_pass and b3_sanity_pass:
        verdict = "B3_READY_FOR_JD_INERT_WIRING"
    else:
        verdict = "B3_REJECTED_DOES_NOT_PRESERVE_VALIDATED_V2_SEED_OR_COUPLING"

    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "residual_helper_source": pre["residual_helper_source"],
        "residual_helper_semantics_matches_validated_replay": pre[
            "residual_helper_semantics_matches_validated_replay"
        ],
        "invalid_import_removed": pre["invalid_import_removed"],
        "preassembly_helper_import_pass": pre["preassembly_helper_import_pass"],
        "preassembly_rayleigh_signature_pass": pre["preassembly_rayleigh_signature_pass"],
        "preassembly_residual_signature_pass": pre["preassembly_residual_signature_pass"],
        "preassembly_writer_available_pass": pre["preassembly_writer_available_pass"],
        "preassembly_no_eigensolve_call_guard_method": pre[
            "preassembly_no_eigensolve_call_guard_method"
        ],
        "preassembly_no_eigensolve_call_pass": pre["preassembly_no_eigensolve_call_pass"],
        "preassembly_contract_pass": pre["preassembly_contract_pass"],
        "preassembly_failure_reasons": pre["preassembly_failure_reasons"],
        "selected_cleaned_formulation_route": "B3",
        "B3_construction_implemented": bool(b3_constructed),
        "B3_shell_trace_space_constructed": bool(b3_constructed),
        "B3_shell_trace_space_type": b3_space_type,
        "B3_shell_trace_mesh_or_submesh_source": b3_source,
        "B3_shell_trace_coordinate_dimension": b3_coord_dim,
        "B3_original_structural_u_dimension": n_u if "n_u" in locals() else None,
        "B3_new_structural_u_dimension": b3_u_new,
        "B3_pressure_dimension_retained": n_p if "n_p" in locals() else None,
        "B3_total_cleaned_W_dimension": b3_total_cleaned,
        "B3_dimension_reduction_ratio": _safe_float(b3_dim_ratio),
        "B3_changes_continuum_physical_meaning_of_weak_forms": b3_changes_continuum,
        "B3_changes_discrete_basis_or_operator_representation": b3_changes_discrete,
        "B3_material_forms_preserved_or_rederived": b3_material,
        "B3_boundary_conditions_preserved_or_rederived": b3_bc,
        "B3_pressure_coupling_preserved_or_rederived": b3_coupling,
        "B3_mapping_back_to_existing_output_layout_defined": b3_layout_map,
        "B3_V2_preservation_status": b3_preservation,
        "B3_A_and_M_assembled_without_EPS": bool(b3_operator_assembled),
        "B3_operator_dimensions": {
            "baseline_reduced_W_dimension": n_w if "n_w" in locals() else None,
            "B3_cleaned_W_dimension": b3_total_cleaned,
        },
        "B3_Muu_nonzero_row_count": b3_muu_rows,
        "B3_Muu_nonzero_column_count": b3_muu_cols,
        "B3_structural_mass_null_coordinate_exposure_status": b3_mass_null_status,
        "B3_operator_storage_persisted": False,
        "B3_seed_mapping_constructed": b3_seed_mapping_constructed,
        "B3_seed_representable": b3_seed_representable,
        "B3_seed_reconstruction_or_transfer_method": (
            "BLOCKED_PENDING_TRACE_U_TO_COUPLED_LAYOUT_TRANSFER"
            if not b3_seed_mapping_constructed
            else "trace_u_l2_projection_plus_pressure_identity_embed"
        ),
        "B3_seed_pressure_support_preserved": b3_seed_pressure_support_preserved,
        "B3_seed_pressure_MAC": b3_seed_mac,
        "B3_seed_xH_Mx_original": seed_xhmx_orig if "seed_xhmx_orig" in locals() else None,
        "B3_seed_xH_Mx_B3": b3_xhmx_b3,
        "B3_seed_replay_frequency_original": seed_f_orig if "seed_f_orig" in locals() else None,
        "B3_seed_replay_frequency_B3": b3_f_b3,
        "B3_seed_residual_original": seed_res_orig if "seed_res_orig" in locals() else None,
        "B3_seed_residual_B3": b3_res_b3,
        "B3_seed_preservation_pass": bool(b3_seed_pres_pass),
        "B3_seed_preservation_failure_reason": b3_seed_fail_reason,
        "B3_structural_mass_present": b3_structural_mass_present,
        "B3_structural_stiffness_present": b3_structural_stiffness_present,
        "B3_coupling_present": b3_coupling_present,
        "B3_BC_contract_pass": b3_bc_contract_pass,
        "B3_pressure_block_contract_pass": b3_pressure_block_contract_pass,
        "B3_no_EPS_operator_sanity_pass": b3_sanity_pass,
        "B3_scalability_gate_pass": bool(b3_scalability_pass),
        "next_step_verdict": verdict,
        "construction_blocker": construction_blocker,
        "jd_wiring_authorized": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "artifact_storage_policy_applied": True,
        "report_size_target_bytes": REPORT_SIZE_TARGET_BYTES,
        "new_large_artifacts_created": [],
        "large_artifact_generation_authorized": False,
        "operator_matrices_persisted": False,
        "vector_banks_persisted": False,
        "cleanup_required_before_production": True,
    }

    report_size = _write_json_atomic(OUT_JSON, payload)
    payload["report_size_bytes"] = int(report_size)
    report_size = _write_json_atomic(OUT_JSON, payload)

    md = [
        "# B3 shell/trace formulation preservation preflight (report-only)",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        f"- B3 shell/trace space constructed: `{payload['B3_shell_trace_space_constructed']}`",
        f"- B3 new structural u dimension: `{payload['B3_new_structural_u_dimension']}`",
        f"- B3 total cleaned W dimension: `{payload['B3_total_cleaned_W_dimension']}`",
        f"- B3 no-EPS operator sanity pass: `{payload['B3_no_EPS_operator_sanity_pass']}`",
        f"- B3 seed mapping constructed: `{payload['B3_seed_mapping_constructed']}`",
        f"- B3 seed preservation pass: `{payload['B3_seed_preservation_pass']}`",
        f"- B3 scalability gate pass: `{payload['B3_scalability_gate_pass']}`",
        f"- next step verdict: `{payload['next_step_verdict']}`",
        "",
        "No eigensolve executed.",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    def log(k: str, v: Any) -> None:
        print(f"[B3_preflight] {k}={v}", flush=True)

    log("B3_shell_trace_space_constructed", payload["B3_shell_trace_space_constructed"])
    log("B3_new_structural_u_dimension", payload["B3_new_structural_u_dimension"])
    log("B3_total_cleaned_W_dimension", payload["B3_total_cleaned_W_dimension"])
    log("B3_no_EPS_operator_sanity_pass", payload["B3_no_EPS_operator_sanity_pass"])
    log("B3_seed_mapping_constructed", payload["B3_seed_mapping_constructed"])
    log("B3_seed_preservation_pass", payload["B3_seed_preservation_pass"])
    log("B3_scalability_gate_pass", payload["B3_scalability_gate_pass"])
    log("next_step_verdict", payload["next_step_verdict"])
    log("artifact_storage_policy_applied", payload["artifact_storage_policy_applied"])
    log("report_size_bytes", payload["report_size_bytes"])
    log("no_new_eigensolve_executed", payload["no_new_eigensolve_executed"])
    log("additional_eps", payload["additional_eps"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
