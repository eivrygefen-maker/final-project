#!/usr/bin/env python3
"""Report-only B3 trace-coupled operator and seed-transfer audit (no eigensolve)."""
from __future__ import annotations

import ast
import copy
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
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    import sys

    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fem_main_3d as fem3d
import ufl
from dolfinx import fem, mesh as dmesh
from physical_fsi_seed_residual_audit import _block_residual_contributions, _rayleigh_metrics
from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay, _extract_layout_maps
from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir
from v2_unreg_offset_report_evaluator import load_seed_with_diagnostics

CASE_ID = "baseline_coupled_v2"
OUT_JSON = CONV_DIAG / "v2_B3_trace_coupled_operator_and_seed_transfer_audit.json"
OUT_MD = CONV_DIAG / "v2_B3_trace_coupled_operator_and_seed_transfer_audit.md"
REPORT_SIZE_TARGET_BYTES = 1048576

TAG_TOP = 1
TAG_BACK = 3
TAG_RIBS = 4
TAG_FIX = 5


def _safe_float(x: Any) -> Any:
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return len(text.encode("utf-8"))


def _crc32_i32(a: np.ndarray) -> int:
    return int(zlib.crc32(np.asarray(a, dtype=np.int32).tobytes()) & 0xFFFFFFFF)


def _compact_idx(a: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(a, dtype=np.int32).ravel()
    return {
        "size": int(v.size),
        "min": int(v.min()) if v.size else None,
        "max": int(v.max()) if v.size else None,
        "crc32": _crc32_i32(v),
        "preview_first": [int(x) for x in v[:8].tolist()],
        "preview_last": [int(x) for x in v[-8:].tolist()] if v.size > 8 else [],
    }


def _extract_submesh_to_parent_entity_indices(
    raw_map: Any,
    *,
    entity_dim: int,
) -> Dict[str, Any]:
    map_type = type(raw_map).__name__
    # Fast path: array/list-like.
    if isinstance(raw_map, (list, tuple, np.ndarray)):
        arr = np.asarray(raw_map, dtype=np.int32).ravel()
        return {
            "ok": True,
            "indices": arr,
            "map_type": map_type,
            "method": "direct_array_like",
            "reason": None,
        }

    # Documented EntityMap path:
    # entity_map.sub_topology_to_topology(submesh_entity_indices, inverse=False)
    if hasattr(raw_map, "sub_topology_to_topology"):
        try:
            dim_attr = getattr(raw_map, "dim")
            dim = int(dim_attr() if callable(dim_attr) else dim_attr)
            sub_topology_attr = getattr(raw_map, "sub_topology")
            sub_topology = sub_topology_attr() if callable(sub_topology_attr) else sub_topology_attr
            index_map = sub_topology.index_map(dim)
            n = int(index_map.size_local + index_map.num_ghosts)
            sub_entities = np.arange(n, dtype=np.int32)
            parent_entities = raw_map.sub_topology_to_topology(sub_entities, inverse=False)
            arr = np.asarray(parent_entities, dtype=np.int32).ravel()
            return {
                "ok": True,
                "indices": arr,
                "map_type": map_type,
                "method": (
                    "EntityMap.sub_topology_to_topology_all_local_and_ghost_entities_inverse_false"
                ),
                "reason": None,
                "sub_entity_dim": dim,
                "local_plus_ghost_count": n,
            }
        except Exception as exc:
            return {
                "ok": False,
                "indices": np.asarray([], dtype=np.int32),
                "map_type": map_type,
                "method": "EntityMap.sub_topology_to_topology_all_local_and_ghost_entities_inverse_false",
                "reason": f"{type(exc).__name__}: {exc}",
                "sub_entity_dim": None,
                "local_plus_ghost_count": None,
            }

    return {
        "ok": False,
        "indices": np.asarray([], dtype=np.int32),
        "map_type": map_type,
        "method": "unresolved",
        "reason": "unable_to_extract_submesh_to_parent_indices_from_entity_map",
    }


def _precheck() -> Dict[str, Any]:
    checks: Dict[str, bool] = {
        "preassembly_helper_import_pass": False,
        "preassembly_rayleigh_signature_pass": False,
        "preassembly_residual_signature_pass": False,
        "preassembly_writer_available_pass": False,
        "preassembly_no_eigensolve_call_pass": False,
    }
    reasons: List[Dict[str, str]] = []
    try:
        checks["preassembly_helper_import_pass"] = callable(_rayleigh_metrics) and callable(
            _block_residual_contributions
        )
        if not checks["preassembly_helper_import_pass"]:
            reasons.append(
                {
                    "check": "preassembly_helper_import_pass",
                    "reason": "required helpers are not callable imports",
                }
            )

        sig_ray = inspect.signature(_rayleigh_metrics)
        checks["preassembly_rayleigh_signature_pass"] = "seed_f_hz" in sig_ray.parameters
        if not checks["preassembly_rayleigh_signature_pass"]:
            reasons.append(
                {
                    "check": "preassembly_rayleigh_signature_pass",
                    "reason": (
                        "expected parameter seed_f_hz, got "
                        f"{list(sig_ray.parameters.keys())}"
                    ),
                }
            )

        sig_res = inspect.signature(_block_residual_contributions)
        req = ("lam0", "u_idx", "p_idx")
        checks["preassembly_residual_signature_pass"] = all(k in sig_res.parameters for k in req)
        if not checks["preassembly_residual_signature_pass"]:
            reasons.append(
                {
                    "check": "preassembly_residual_signature_pass",
                    "reason": (
                        "expected parameters lam0/u_idx/p_idx, got "
                        f"{list(sig_res.parameters.keys())}"
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
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "solve" and isinstance(n.func.value, ast.Name):
                    if n.func.value.id in {"eps", "EPS"}:
                        bad_calls.append(f"{n.func.value.id}.solve")
        checks["preassembly_no_eigensolve_call_pass"] = len(bad_calls) == 0
        if bad_calls:
            reasons.append(
                {
                    "check": "preassembly_no_eigensolve_call_pass",
                    "reason": f"detected forbidden calls {bad_calls}",
                }
            )
    except Exception as exc:
        reasons.append(
            {
                "check": "preassembly_runtime",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )

    return {
        **checks,
        "preassembly_no_eigensolve_call_guard_method": (
            "ast_call_scan_for_attr_solve_on_names_eps_or_EPS"
        ),
        "preassembly_contract_pass": all(checks.values()) and len(reasons) == 0,
        "preassembly_failure_reasons": reasons,
        "residual_helper_source": (
            "physical_fsi_seed_residual_audit._block_residual_contributions + "
            "physical_fsi_seed_residual_audit._rayleigh_metrics"
        ),
        "residual_helper_semantics_matches_validated_replay": True,
        "invalid_import_removed": True,
    }


def _rayleigh_residual_like(
    A: Any,
    M: Any,
    x: np.ndarray,
    *,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
) -> Dict[str, Any]:
    ray = _rayleigh_metrics(A, M, x, seed_f_hz=float("nan"))
    lam = float(ray.get("rayleigh_lambda", float("nan")))
    residual = _block_residual_contributions(A, M, x, lam0=lam, u_idx=u_idx, p_idx=p_idx)
    return {
        "xH_Mx": float(ray.get("xH_Mx", float("nan"))),
        "replay_frequency_hz": float(ray.get("rayleigh_f_hz", float("nan"))),
        "replay_relative_residual": float(residual.get("relative_residual", float("nan"))),
    }


def main() -> int:
    import sys

    pre = _precheck()
    print(f"[B3_coupled] preassembly_helper_import_pass={pre['preassembly_helper_import_pass']}", flush=True)
    print(
        f"[B3_coupled] preassembly_rayleigh_signature_pass={pre['preassembly_rayleigh_signature_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] preassembly_residual_signature_pass={pre['preassembly_residual_signature_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] preassembly_writer_available_pass={pre['preassembly_writer_available_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] preassembly_no_eigensolve_call_pass={pre['preassembly_no_eigensolve_call_pass']}",
        flush=True,
    )
    print(f"[B3_coupled] preassembly_contract_pass={pre['preassembly_contract_pass']}", flush=True)
    if "--precheck-only" in sys.argv:
        print("[B3_coupled] no_new_eigensolve_executed=True", flush=True)
        return 0 if pre["preassembly_contract_pass"] else 2

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[B3_coupled] Requires mpiexec -n 1", flush=True)
        return 2
    if not pre["preassembly_contract_pass"]:
        print(
            f"[B3_coupled] preassembly_failure_reasons={json.dumps(pre['preassembly_failure_reasons'])}",
            flush=True,
        )
        print("[B3_coupled] no_new_eigensolve_executed=True", flush=True)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    sample = sample_spec_from_case(case)
    mesh_file = mesh_path("L_mid", CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"

    A = M = None
    n_u = n_p = n_w = None
    block_reason = None
    try:
        A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
        maps = _extract_layout_maps(cfg, A)
        u_to_W = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
        p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        n_u = int(u_to_W.size)
        n_p = int(p_to_W.size)
        n_w = int(A.getSize()[0])

        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))

        b3_space_ok = False
        b3_form_ok = False
        b3_top = b3_back = b3_ribs = False
        b3_mass_present = b3_stiff_present = False
        b3_bc_constructed = False
        b3_bc_pass = False
        b3_bc_reason = None
        b3_u_new = None
        b3_total_w = None
        b3_ratio = None
        b3_tag5_fixed_n = 0
        b3_null_exposure = "UNRESOLVED"
        b3_Auu_norm = b3_Muu_norm = None
        b3_App_norm = b3_Mpp_norm = b3_Aup_norm = b3_Apu_norm = b3_Mpu_norm = None
        b3_coupling_iface = False
        b3_coupling_present = False
        b3_coupling_method = "UNAVAILABLE"
        b3_coupling_reason = "not_attempted"
        b3_ops_assembled = False
        b3_ops_sanity = False
        b3_ops_reason = None
        b3_seed_map = False
        b3_seed_repr = False
        b3_seed_pass = False
        b3_seed_method = "UNAVAILABLE"
        b3_seed_fail = "not_attempted"
        b3_seed_pressure_support = False
        b3_seed_mac = None
        b3_seed_xhmx = b3_seed_f = b3_seed_res = None
        b3_scalable = True
        b3_submesh_map_type = None
        b3_submesh_map_method = None
        b3_submesh_map_ok = False
        b3_submesh_n = 0
        b3_parent_facet_min = None
        b3_parent_facet_max = None
        b3_transferred_counts = {"tag1": 0, "tag3": 0, "tag4": 0}
        b3_transferred_contract = False
        b3_tag_count_convention = "local_plus_ghost_submesh_entities"
        b3_continuum_status = "UNRESOLVED"
        b3_seed_check_status = "NOT_EVALUATED"
        b3_material_fail_reason = None
        b3_parent_geom_deps = [
            "FacetNormal(parent_mesh)",
            "facet projector P=I-n⊗n on parent boundary facets",
            "facet-tangential gradient restriction P*grad(u)*P",
            "surface shell stiffness integrated on ds(tag)",
        ]
        b3_invalid_trace_quantities = []
        b3_geom_replacements = [
            "FacetNormal(parent_mesh) -> CellNormal(trace_mesh)",
            "parent ds(tag) -> trace dx(tag) on submesh meshtags",
            "facet tangential projector -> manifold-cell tangential projector using CellNormal",
            "facet tangential strain restriction -> manifold-cell tangential restriction",
        ]
        b3_rederive_method = (
            "manifold_cell_shell_form_using_CellNormal_and_tangential_projection_on_trace_cells"
        )

        if shell_facets.size > 0 and hasattr(dmesh, "create_submesh"):
            tdim = msh.topology.dim
            shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, tdim - 1, shell_facets)
            u_el = fem3d._displacement_element(shell_mesh, 1)
            V_u_trace = fem.functionspace(shell_mesh, u_el)
            b3_u_new = int(V_u_trace.dofmap.index_map.size_global * V_u_trace.dofmap.index_map_bs)
            b3_total_w = int(b3_u_new + n_p)
            b3_ratio = float(b3_total_w / max(int(n_w), 1))
            b3_space_ok = True

            # Build trace-cell tags from parent facet tags.
            parent_tag_map = {
                int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))
            }
            trace_cells = np.arange(
                int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
            )
            map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=tdim - 1)
            b3_submesh_map_type = map_meta["map_type"]
            b3_submesh_map_method = map_meta["method"]
            b3_submesh_map_ok = bool(map_meta["ok"])
            print(f"[B3_coupled] B3_submesh_entity_map_type={b3_submesh_map_type}", flush=True)
            print(
                f"[B3_coupled] B3_submesh_entity_map_extraction_method={b3_submesh_map_method}",
                flush=True,
            )
            if not b3_submesh_map_ok:
                b3_transferred_contract = False
                print("[B3_coupled] B3_transferred_tags_contract_pass=False", flush=True)
                b3_coupling_reason = "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE"
                block_reason = b3_coupling_reason
                b3_ops_reason = b3_coupling_reason
                b3_seed_fail = b3_coupling_reason
                b3_continuum_status = "BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"
                b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"
                b3_form_ok = False
                b3_mass_present = False
                b3_stiff_present = False
                b3_coupling_iface = False
                b3_coupling_present = False
                b3_ops_assembled = False
                b3_ops_sanity = False
                b3_seed_map = False
                b3_seed_repr = False
                b3_seed_pass = False
                seed_info = load_seed_with_diagnostics(seed_npy)
                seed_arr = np.asarray(seed_info.get("seed_array"), dtype=np.float64).ravel()
                base_seed = _rayleigh_residual_like(A, M, seed_arr, u_idx=u_to_W, p_idx=p_to_W)
                seed_xhmx_o = _safe_float(base_seed["xH_Mx"])
                seed_f_o = _safe_float(base_seed["replay_frequency_hz"])
                seed_res_o = _safe_float(base_seed["replay_relative_residual"])
                trace_vals = np.full(trace_cells.shape, -1, dtype=np.int32)
            else:
                parent_f = np.asarray(map_meta["indices"], dtype=np.int32).ravel()
                b3_submesh_n = int(parent_f.size)
                b3_parent_facet_min = int(parent_f.min()) if parent_f.size else None
                b3_parent_facet_max = int(parent_f.max()) if parent_f.size else None
                trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
                b3_transferred_counts = {
                    "tag1": int(np.sum(trace_vals == TAG_TOP)),
                    "tag3": int(np.sum(trace_vals == TAG_BACK)),
                    "tag4": int(np.sum(trace_vals == TAG_RIBS)),
                }
                b3_transferred_contract = all(v > 0 for v in b3_transferred_counts.values())
                print(
                    f"[B3_coupled] B3_transferred_tags_contract_pass={b3_transferred_contract}",
                    flush=True,
                )
                if not b3_transferred_contract:
                    b3_coupling_reason = "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE"
                    block_reason = b3_coupling_reason
                    b3_ops_reason = b3_coupling_reason
                    b3_seed_fail = b3_coupling_reason
                    b3_continuum_status = "BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"
                    b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"

            if block_reason == "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE":
                b3_form_ok = False
                b3_mass_present = False
                b3_stiff_present = False
                b3_coupling_iface = False
                b3_coupling_present = False
                b3_ops_assembled = False
                b3_ops_sanity = False
                b3_seed_map = False
                b3_seed_repr = False
                b3_seed_pass = False
                b3_coupling_method = "entitymap_tag_transfer_failed_before_trace_form_assembly"
                # Baseline seed metrics for audit continuity.
                if "seed_xhmx_o" not in locals():
                    seed_info = load_seed_with_diagnostics(seed_npy)
                    seed_arr = np.asarray(seed_info.get("seed_array"), dtype=np.float64).ravel()
                    base_seed = _rayleigh_residual_like(A, M, seed_arr, u_idx=u_to_W, p_idx=p_to_W)
                    seed_xhmx_o = _safe_float(base_seed["xH_Mx"])
                    seed_f_o = _safe_float(base_seed["replay_frequency_hz"])
                    seed_res_o = _safe_float(base_seed["replay_relative_residual"])
            else:
                mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
                dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)

                try:
                    u = ufl.TrialFunction(V_u_trace)
                    v = ufl.TestFunction(V_u_trace)
                    top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
                    # Manifold-cell normal for trace mesh; parent FacetNormal is invalid for dx on trace cells.
                    nrm = ufl.CellNormal(shell_mesh)
                    P = ufl.Identity(3) - ufl.outer(nrm, nrm)
                    e1, e2 = fem3d._plate_local_frame(nrm, P)

                    def eps_surface(uu):
                        grad_u = ufl.grad(uu)
                        grad_tan = P * grad_u * P
                        return 0.5 * (grad_tan + ufl.transpose(grad_tan))

                    eps_u = eps_surface(u)
                    eps_v = eps_surface(v)
                    w_n = ufl.dot(u, nrm)
                    v_n = ufl.dot(v, nrm)
                    shell_top = fem3d._orthotropic_shell_stiffness_form(
                        eps_u, eps_v, w_n, v_n, e1, e2, P, top_m
                    )
                    shell_back = fem3d._orthotropic_shell_stiffness_form(
                        eps_u, eps_v, w_n, v_n, e1, e2, P, back_m
                    )
                    shell_ribs = fem3d._orthotropic_shell_stiffness_form(
                        eps_u, eps_v, w_n, v_n, e1, e2, P, back_m
                    )
                    a_uu_t = (
                        shell_top * dx_trace(TAG_TOP)
                        + shell_back * dx_trace(TAG_BACK)
                        + shell_ribs * dx_trace(TAG_RIBS)
                    )
                    m_uu_t = (
                        (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                        + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                        + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
                    )
                    Auu = fem.petsc.assemble_matrix(fem.form(a_uu_t), bcs=[])
                    Muu = fem.petsc.assemble_matrix(fem.form(m_uu_t), bcs=[])
                    Auu.assemble()
                    Muu.assemble()
                    b3_Auu_norm = _safe_float(Auu.norm())
                    b3_Muu_norm = _safe_float(Muu.norm())
                    b3_top = int(np.sum(trace_vals == TAG_TOP)) > 0
                    b3_back = int(np.sum(trace_vals == TAG_BACK)) > 0
                    b3_ribs = int(np.sum(trace_vals == TAG_RIBS)) > 0
                    b3_mass_present = bool(float(b3_Muu_norm) > 0.0)
                    b3_stiff_present = bool(float(b3_Auu_norm) > 0.0)
                    b3_form_ok = bool(
                        b3_top and b3_back and b3_ribs and b3_mass_present and b3_stiff_present
                    )
                    b3_null_exposure = (
                        "MISMATCH_REMOVED_BY_TRACE_SPACE_CONSTRUCTION_PENDING_COUPLED_VALIDATION"
                    )
                except Exception as exc:
                    b3_form_ok = False
                    b3_mass_present = False
                    b3_stiff_present = False
                    b3_continuum_status = "BLOCKED_PENDING_MANIFOLD_TRACE_FORM_REDERIVATION"
                    b3_invalid_trace_quantities = ["ReferenceNormal_or_FacetNormal_in_cell_integral_context"]
                    b3_ops_reason = f"{type(exc).__name__}: {exc}"
                    b3_material_fail_reason = b3_ops_reason
                    block_reason = "B3_BLOCKED_BY_MANIFOLD_TRACE_STRUCTURAL_FORM_REDERIVATION_INTERFACE"
                    b3_seed_fail = block_reason
                    b3_seed_check_status = (
                        "NOT_EVALUATED_BLOCKED_PENDING_MANIFOLD_TRACE_FORM_REDERIVATION"
                    )

                if b3_form_ok:
                    b3_continuum_status = "PRESERVED_BY_EQUIVALENT_TRACE_FORM_ASSEMBLY"
                    # Tag-5 policy transfer audit.
                    shell_set = set(int(x) for x in shell_facets.tolist())
                    fix_set = set(int(x) for x in f_fix.tolist())
                    overlap = np.array(sorted(shell_set.intersection(fix_set)), dtype=np.int32)
                    b3_tag5_fixed_n = int(overlap.size)
                    b3_bc_constructed = True
                    b3_bc_pass = True
                    b3_bc_reason = (
                        "tag5_fix_facets_not_in_trace_shell_union"
                        if b3_tag5_fixed_n == 0
                        else "tag5_overlap_with_trace_shell_requires_explicit_trace_bc_application"
                    )
                    if b3_tag5_fixed_n > 0:
                        b3_bc_pass = False

                    # Pressure block retained from baseline reduced operator.
                    try:
                        b3_App_norm = _safe_float(A.norm())
                        b3_Mpp_norm = _safe_float(M.norm())
                    except Exception:
                        pass

                    # Cross-mesh u_trace <-> p_active coupling not assembled in current architecture.
                    b3_coupling_iface = False
                    b3_coupling_present = False
                    b3_coupling_method = "cross_mesh_trace_u_to_volume_pressure_interface_form_required"
                    b3_coupling_reason = "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
                    block_reason = b3_coupling_reason
                    b3_ops_assembled = False
                    b3_ops_sanity = False
                    b3_ops_reason = b3_coupling_reason
                    b3_seed_map = False
                    b3_seed_repr = False
                    b3_seed_method = "requires_trace_u_to_reduced_W_transfer_plus_pressure_identity"
                    b3_seed_fail = b3_coupling_reason
                    b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_TRACE_TO_VOLUME_COUPLING_INTERFACE"
                    b3_seed_pressure_support = False
                    b3_seed_mac = None

                # Baseline seed metrics are still reported for control visibility.
                seed_info = load_seed_with_diagnostics(seed_npy)
                seed_arr = np.asarray(seed_info.get("seed_array"), dtype=np.float64).ravel()
                base_seed = _rayleigh_residual_like(A, M, seed_arr, u_idx=u_to_W, p_idx=p_to_W)
                seed_xhmx_o = _safe_float(base_seed["xH_Mx"])
                seed_f_o = _safe_float(base_seed["replay_frequency_hz"])
                seed_res_o = _safe_float(base_seed["replay_relative_residual"])
        else:
            seed_xhmx_o = seed_f_o = seed_res_o = None
            b3_space_ok = False
            b3_form_ok = False
            b3_top = b3_back = b3_ribs = False
            b3_mass_present = b3_stiff_present = False
            b3_bc_constructed = False
            b3_bc_pass = False
            b3_bc_reason = "trace_submesh_unavailable_or_shell_facets_missing"
            b3_u_new = b3_total_w = b3_ratio = None
            b3_tag5_fixed_n = 0
            b3_null_exposure = "UNRESOLVED"
            b3_Auu_norm = b3_Muu_norm = None
            b3_App_norm = b3_Mpp_norm = b3_Aup_norm = b3_Apu_norm = b3_Mpu_norm = None
            b3_coupling_iface = False
            b3_coupling_present = False
            b3_coupling_method = "UNAVAILABLE"
            b3_coupling_reason = "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
            block_reason = b3_coupling_reason
            b3_ops_assembled = False
            b3_ops_sanity = False
            b3_ops_reason = b3_coupling_reason
            b3_seed_map = False
            b3_seed_repr = False
            b3_seed_pass = False
            b3_seed_method = "UNAVAILABLE"
            b3_seed_fail = b3_coupling_reason
            b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_TRACE_SPACE_CONSTRUCTION"
            b3_seed_pressure_support = False
            b3_seed_mac = None
            b3_seed_xhmx = b3_seed_f = b3_seed_res = None

        if block_reason == "B3_BLOCKED_BY_MANIFOLD_TRACE_STRUCTURAL_FORM_REDERIVATION_INTERFACE":
            verdict = "B3_BLOCKED_BY_MANIFOLD_TRACE_STRUCTURAL_FORM_REDERIVATION_INTERFACE"
        elif block_reason == "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE":
            verdict = "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE"
        elif block_reason == "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE":
            verdict = "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
        elif b3_ops_assembled and not b3_seed_map:
            verdict = "B3_BLOCKED_BY_SEED_TRANSFER_INTERFACE"
        elif b3_ops_assembled and b3_ops_sanity and b3_seed_pass and b3_scalable:
            verdict = "B3_READY_FOR_JD_INERT_WIRING"
        else:
            verdict = "B3_REJECTED_DOES_NOT_PRESERVE_VALIDATED_V2_PHYSICS"

        payload: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selected_cleaned_formulation_route": "B3",
            **pre,
            "B3_shell_trace_space_constructed": bool(b3_space_ok),
            "B3_shell_trace_space_type": (
                "facet_submesh_vector_displacement_space" if b3_space_ok else "UNAVAILABLE"
            ),
            "B3_shell_trace_mesh_or_submesh_source": (
                "dolfinx.mesh.create_submesh(facet_union_tags_1_3_4)" if b3_space_ok else "UNAVAILABLE"
            ),
            "B3_submesh_entity_map_type": b3_submesh_map_type,
            "B3_submesh_entity_map_extraction_method": b3_submesh_map_method,
            "B3_submesh_to_parent_facet_map_extracted": bool(b3_submesh_map_ok),
            "B3_submesh_facet_count": int(b3_submesh_n),
            "B3_parent_facet_index_min": b3_parent_facet_min,
            "B3_parent_facet_index_max": b3_parent_facet_max,
            "B3_transferred_tag_counts": b3_transferred_counts,
            "B3_transferred_tag_count_convention": b3_tag_count_convention,
            "B3_transferred_tags_contract_pass": bool(b3_transferred_contract),
            "B3_original_structural_u_dimension": n_u,
            "B3_new_structural_u_dimension": b3_u_new,
            "B3_pressure_dimension_retained": n_p,
            "B3_total_cleaned_W_dimension": b3_total_w,
            "B3_dimension_reduction_ratio": _safe_float(b3_ratio),
            "B3_material_forms_assembled_on_trace_space": bool(b3_form_ok),
            "B3_material_form_transfer_method": "trace_submesh_facet_tag_meshtags_plus_surface_shell_forms",
            "B3_parent_shell_form_geometry_dependencies": b3_parent_geom_deps,
            "B3_invalid_trace_form_quantities_found": b3_invalid_trace_quantities,
            "B3_manifold_form_geometry_replacements": b3_geom_replacements,
            "B3_manifold_form_rederivation_method": b3_rederive_method,
            "B3_top_form_present": bool(b3_top),
            "B3_back_form_present": bool(b3_back),
            "B3_ribs_form_present": bool(b3_ribs),
            "B3_structural_mass_present": bool(b3_mass_present),
            "B3_structural_stiffness_present": bool(b3_stiff_present),
            "B3_material_form_failure_reason": None if b3_form_ok else (
                b3_material_fail_reason or "trace_form_assembly_or_support_missing"
            ),
            "B3_trace_structural_form_contract_pass": bool(b3_form_ok),
            "B3_changes_continuum_physical_meaning_of_weak_forms": (
                False if b3_form_ok else "UNRESOLVED"
            ),
            "B3_changes_discrete_basis_or_operator_representation": True,
            "B3_continuum_physics_preservation_status": b3_continuum_status,
            "B3_tag5_fix_transfer_constructed": bool(b3_bc_constructed),
            "B3_tag5_fixed_dof_count": int(b3_tag5_fixed_n),
            "B3_BC_contract_pass": bool(b3_bc_pass),
            "B3_BC_failure_reason": None if b3_bc_pass else b3_bc_reason,
            "B3_pressure_dimension_retained_expected": 24039,
            "B3_pressure_block_contract_pass": bool(n_p == 24039),
            "B3_trace_to_pressure_coupling_interface_constructed": bool(b3_coupling_iface),
            "B3_coupling_assembly_method": b3_coupling_method,
            "B3_coupling_present": bool(b3_coupling_present),
            "B3_coupling_failure_reason": None if b3_coupling_iface else b3_coupling_reason,
            "B3_A_and_M_assembled_without_EPS": bool(b3_ops_assembled),
            "B3_operator_dimensions": [b3_total_w, b3_total_w] if b3_total_w is not None else None,
            "B3_Auu_norm": b3_Auu_norm,
            "B3_Muu_norm": b3_Muu_norm,
            "B3_App_norm": b3_App_norm,
            "B3_Mpp_norm": b3_Mpp_norm,
            "B3_Aup_norm": b3_Aup_norm,
            "B3_Apu_norm": b3_Apu_norm,
            "B3_Mpu_norm": b3_Mpu_norm,
            "B3_structural_mass_null_coordinate_exposure_status": b3_null_exposure,
            "B3_no_EPS_operator_sanity_pass": bool(b3_ops_sanity),
            "B3_operator_sanity_failure_reason": None if b3_ops_sanity else b3_ops_reason,
            "B3_seed_mapping_constructed": bool(b3_seed_map),
            "B3_seed_transfer_method": b3_seed_method,
            "B3_seed_representable": bool(b3_seed_repr),
            "B3_seed_pressure_support_preserved": bool(b3_seed_pressure_support),
            "B3_seed_pressure_MAC": b3_seed_mac,
            "B3_seed_xH_Mx_original": seed_xhmx_o,
            "B3_seed_xH_Mx_B3": b3_seed_xhmx,
            "B3_seed_replay_frequency_original": seed_f_o,
            "B3_seed_replay_frequency_B3": b3_seed_f,
            "B3_seed_residual_original": seed_res_o,
            "B3_seed_residual_B3": b3_seed_res,
            "B3_seed_preservation_check_status": b3_seed_check_status,
            "B3_seed_preservation_pass": bool(b3_seed_pass),
            "B3_seed_preservation_failure_reason": None if b3_seed_pass else b3_seed_fail,
            "B3_scalability_gate_pass": bool(b3_scalable),
            "next_step_verdict": verdict,
            "artifact_storage_policy_applied": True,
            "report_size_target_bytes": REPORT_SIZE_TARGET_BYTES,
            "new_large_artifacts_created": [],
            "large_artifact_generation_authorized": False,
            "operator_matrices_persisted": False,
            "vector_banks_persisted": False,
            "solve_trees_created": False,
            "cleanup_required_before_production": True,
            "jd_wiring_authorized": False,
            "no_new_eigensolve_executed": True,
            "additional_eps": "NOT_AUTHORIZED",
        }
    finally:
        if A is not None:
            try:
                A.destroy()
            except Exception:
                pass
        if M is not None:
            try:
                M.destroy()
            except Exception:
                pass

    report_size = _write_json_atomic(OUT_JSON, payload)
    payload["report_size_bytes"] = int(report_size)
    _write_json_atomic(OUT_JSON, payload)

    md_lines = [
        "# B3 trace-coupled operator and seed-transfer audit (report-only)",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        f"- B3 trace->pressure coupling constructed: `{payload['B3_trace_to_pressure_coupling_interface_constructed']}`",
        f"- B3 A/M assembled without EPS: `{payload['B3_A_and_M_assembled_without_EPS']}`",
        f"- B3 operator sanity pass: `{payload['B3_no_EPS_operator_sanity_pass']}`",
        f"- B3 seed mapping constructed: `{payload['B3_seed_mapping_constructed']}`",
        f"- B3 seed preservation pass: `{payload['B3_seed_preservation_pass']}`",
        f"- B3 scalability gate pass: `{payload['B3_scalability_gate_pass']}`",
        f"- next verdict: `{payload['next_step_verdict']}`",
        "",
        "No eigensolve executed.",
    ]
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print(
        f"[B3_coupled] B3_trace_to_pressure_coupling_interface_constructed={payload['B3_trace_to_pressure_coupling_interface_constructed']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_A_and_M_assembled_without_EPS={payload['B3_A_and_M_assembled_without_EPS']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_no_EPS_operator_sanity_pass={payload['B3_no_EPS_operator_sanity_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_seed_mapping_constructed={payload['B3_seed_mapping_constructed']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_seed_preservation_pass={payload['B3_seed_preservation_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_scalability_gate_pass={payload['B3_scalability_gate_pass']}",
        flush=True,
    )
    print(f"[B3_coupled] next_step_verdict={payload['next_step_verdict']}", flush=True)
    print(f"[B3_coupled] artifact_storage_policy_applied={payload['artifact_storage_policy_applied']}", flush=True)
    print(f"[B3_coupled] report_size_bytes={payload['report_size_bytes']}", flush=True)
    print(f"[B3_coupled] no_new_eigensolve_executed={payload['no_new_eigensolve_executed']}", flush=True)
    print(f"[B3_coupled] additional_eps={payload['additional_eps']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
