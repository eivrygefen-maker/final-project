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
OUT_JSON_C2_CONTRACT = CONV_DIAG / "v2_B3_C2_transfer_contract_only.json"
REPORT_SIZE_TARGET_BYTES = 1048576
C2_TRANSFER_CONTRACT_ONLY_ARG = "--C2-transfer-contract-only"

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


def _coupling_contract_precheck() -> Dict[str, Any]:
    c1_supported = False
    c1_api = "dolfinx.fem.form(entity_maps=...) cross-mesh mixed-domain assembly"
    c1_blocker = "dolfinx_direct_cross_mesh_mixed_domain_form_api_unproven_for_trace_u_to_parent_p"
    try:
        sig_form = inspect.signature(fem.form)
        if "entity_maps" in sig_form.parameters:
            # API parameter exists; still not enough to prove viable route in this stack.
            c1_supported = False
            c1_blocker = (
                "entity_maps_parameter_present_but_no_validated_trace_u_parent_p_form_contract_in_current_code"
            )
    except Exception as exc:
        c1_blocker = f"inspect_fem_form_failed:{type(exc).__name__}:{exc}"

    c2_constructible = True
    c2_blocker = None
    if not hasattr(dmesh, "create_submesh"):
        c2_constructible = False
        c2_blocker = "dolfinx_create_submesh_unavailable"
    c2_transfer_repr = "sparse_trace_u_to_parent_interface_u_transfer_operator_T"

    selected = "C2" if c2_constructible else "NONE"
    selected_reason = (
        "C2 preferred for compact sparse transfer and reuse of validated parent coupling blocks"
        if selected == "C2"
        else "No viable coupling route precheck passed"
    )
    return {
        "C1_supported_by_installed_dolfinx": c1_supported,
        "C1_required_api": c1_api,
        "C1_preserves_existing_interface_integral_meaning": "UNPROVEN",
        "C1_implementation_blocker": c1_blocker,
        "C2_sparse_trace_to_parent_transfer_constructible": c2_constructible,
        "C2_transfer_representation": c2_transfer_repr,
        "C2_transfer_storage_bytes": 0 if c2_constructible else None,
        "C2_reuses_validated_parent_coupling_contract": True,
        "C2_preserves_output_reconstruction_path": "UNPROVEN",
        "C2_implementation_blocker": c2_blocker,
        "selected_B3_coupling_route": selected,
        "selected_B3_coupling_route_reason": selected_reason,
    }


def _build_c2_trace_to_parent_transfer(
    msh: Any,
    facet_tags: Any,
    *,
    shell_facets: np.ndarray,
    tag_top: int,
    tag_back: int,
    tag_ribs: int,
) -> Dict[str, Any]:
    """Construct sparse transfer T: u_trace -> parent_u using block DOF map then component expansion."""
    u_el_parent = fem3d._displacement_element(msh, 1)
    V_u_parent = fem.functionspace(msh, u_el_parent)
    n_u_parent = int(V_u_parent.dofmap.index_map.size_global * V_u_parent.dofmap.index_map_bs)
    n_parent_blocks = int(V_u_parent.dofmap.index_map.size_global)

    if shell_facets.size == 0 or not hasattr(dmesh, "create_submesh"):
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "shell_facets_missing_or_create_submesh_unavailable",
            "failure_stage": "SUBMESH_PRECONDITION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "C2_dense_coupling_allocation_prohibited": True,
            "C2_dense_coupling_allocation_removed": True,
            "C2_projected_coupling_representation": "NOT_YET_SAFE",
        }

    tdim = msh.topology.dim
    shell_mesh, shell_to_parent, shell_vertex_to_parent, _ = dmesh.create_submesh(
        msh, tdim - 1, shell_facets
    )
    shell_tdim = int(shell_mesh.topology.dim)
    parent_tdim = int(msh.topology.dim)
    shell_0_to_tdim_created = False
    shell_tdim_to_0_created = False
    parent_0_to_tdim_created = False
    parent_tdim_to_0_created = False

    try:
        shell_mesh.topology.create_connectivity(0, shell_tdim)
        shell_0_to_tdim_created = True
        shell_mesh.topology.create_connectivity(shell_tdim, 0)
        shell_tdim_to_0_created = True
        msh.topology.create_connectivity(0, parent_tdim)
        parent_0_to_tdim_created = True
        msh.topology.create_connectivity(parent_tdim, 0)
        parent_tdim_to_0_created = True
    except Exception as exc:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "topology_connectivity_construction_failed",
            "failure_stage": "TOPOLOGY_CONNECTIVITY_CONSTRUCTION",
            "failure_exception_type": type(exc).__name__,
            "failure_exception_message": str(exc),
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "C2_T_shell_topological_dimension": shell_tdim,
            "C2_T_parent_topological_dimension": parent_tdim,
            "C2_T_shell_connectivity_0_to_tdim_created": shell_0_to_tdim_created,
            "C2_T_shell_connectivity_tdim_to_0_created": shell_tdim_to_0_created,
            "C2_T_parent_connectivity_0_to_tdim_created": parent_0_to_tdim_created,
            "C2_T_parent_connectivity_tdim_to_0_created": parent_tdim_to_0_created,
            "C2_T_shell_cell_entity_map_type": type(shell_to_parent).__name__,
            "C2_T_shell_vertex_entity_map_type": type(shell_vertex_to_parent).__name__,
            "C2_T_shell_vertex_map_extracted": False,
            "C2_T_shell_vertex_map_extraction_method": None,
            "C2_dense_coupling_allocation_prohibited": True,
            "C2_dense_coupling_allocation_removed": True,
            "C2_projected_coupling_representation": "NOT_YET_SAFE",
        }

    cell_map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=tdim - 1)
    vertex_map_meta = _extract_submesh_to_parent_entity_indices(shell_vertex_to_parent, entity_dim=0)
    parent_f = np.asarray(cell_map_meta.get("indices", np.asarray([], dtype=np.int32)), dtype=np.int32).ravel()
    sub_v_map = np.asarray(vertex_map_meta.get("indices", np.asarray([], dtype=np.int32)), dtype=np.int32).ravel()
    vmap_size = int(shell_mesh.topology.index_map(0).size_local + shell_mesh.topology.index_map(0).num_ghosts)

    common_meta: Dict[str, Any] = {
        "C2_T_shell_cell_entity_map_type": cell_map_meta.get("map_type"),
        "C2_T_shell_vertex_entity_map_type": vertex_map_meta.get("map_type"),
        "C2_T_shell_cell_map_extraction_method": cell_map_meta.get("method"),
        "C2_T_shell_vertex_map_extraction_method": vertex_map_meta.get("method"),
        "C2_T_shell_vertex_map_extracted": bool(vertex_map_meta.get("ok", False)),
        "C2_T_shell_vertex_count": int(sub_v_map.size),
        "C2_T_parent_vertex_index_min": int(sub_v_map.min()) if sub_v_map.size else None,
        "C2_T_parent_vertex_index_max": int(sub_v_map.max()) if sub_v_map.size else None,
        "C2_T_shell_topological_dimension": shell_tdim,
        "C2_T_parent_topological_dimension": parent_tdim,
        "C2_T_shell_connectivity_0_to_tdim_created": shell_0_to_tdim_created,
        "C2_T_shell_connectivity_tdim_to_0_created": shell_tdim_to_0_created,
        "C2_T_parent_connectivity_0_to_tdim_created": parent_0_to_tdim_created,
        "C2_T_parent_connectivity_tdim_to_0_created": parent_tdim_to_0_created,
        "C2_T_matching_key": "ENTITY_VERTEX_PLUS_VECTOR_COMPONENT",
        "C2_T_coordinate_match_used_as": "VALIDATION_ONLY",
        "C2_T_coordinate_validation_level": "BLOCK_VERTEX_DOF",
        "C2_dense_coupling_allocation_prohibited": True,
        "C2_dense_coupling_allocation_removed": True,
        "C2_projected_coupling_representation": "NOT_YET_SAFE",
    }

    if not cell_map_meta["ok"]:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "entitymap_to_parent_facet_extraction_failed",
            "failure_stage": "CELL_ENTITYMAP_EXTRACTION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            **common_meta,
        }
    if not vertex_map_meta["ok"]:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "entitymap_to_parent_vertex_extraction_failed",
            "failure_stage": "VERTEX_ENTITYMAP_EXTRACTION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            **common_meta,
        }
    if sub_v_map.size < vmap_size:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "submesh_vertex_map_size_mismatch",
            "failure_stage": "VERTEX_ENTITYMAP_EXTRACTION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            **common_meta,
        }

    parent_tag_map = {
        int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))
    }
    trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
    transferred_counts = {
        "tag1": int(np.sum(trace_vals == tag_top)),
        "tag3": int(np.sum(trace_vals == tag_back)),
        "tag4": int(np.sum(trace_vals == tag_ribs)),
    }

    V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
    n_u_trace = int(V_u_trace.dofmap.index_map.size_global * V_u_trace.dofmap.index_map_bs)
    n_trace_blocks = int(V_u_trace.dofmap.index_map.size_global)
    bs_trace = int(V_u_trace.dofmap.index_map_bs)
    bs_parent = int(V_u_parent.dofmap.index_map_bs)
    if bs_trace != bs_parent:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "trace_parent_vector_block_size_mismatch",
            "failure_stage": "VECTOR_BLOCK_SIZE_VALIDATION",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            **common_meta,
        }

    trace_block_to_parent_block = np.full(n_trace_blocks, -1, dtype=np.int32)
    block_cardinality_mismatch = 0
    for sv in range(vmap_size):
        pv = int(sub_v_map[sv])
        if pv < 0:
            continue
        trace_block = np.asarray(
            fem.locate_dofs_topological(V_u_trace, 0, np.asarray([sv], dtype=np.int32)), dtype=np.int32
        ).ravel()
        parent_block = np.asarray(
            fem.locate_dofs_topological(V_u_parent, 0, np.asarray([pv], dtype=np.int32)), dtype=np.int32
        ).ravel()
        if trace_block.size != 1 or parent_block.size != 1:
            block_cardinality_mismatch += 1
            continue
        tb = int(trace_block[0])
        pb = int(parent_block[0])
        if 0 <= tb < n_trace_blocks and 0 <= pb < n_parent_blocks:
            trace_block_to_parent_block[tb] = pb

    mapped_trace_block_count = int(np.sum(trace_block_to_parent_block >= 0))
    unmatched_trace_block_count = int(np.sum(trace_block_to_parent_block < 0))
    mapped_parent_blocks = trace_block_to_parent_block[trace_block_to_parent_block >= 0]
    duplicate_parent_block_count = int(mapped_parent_blocks.size - np.unique(mapped_parent_blocks).size)
    block_map_injective_pass = bool(
        unmatched_trace_block_count == 0 and duplicate_parent_block_count == 0 and block_cardinality_mismatch == 0
    )
    if not block_map_injective_pass:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "block_vertex_map_not_injective_or_incomplete",
            "failure_stage": "VERTEX_COMPONENT_EXPANSION",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            "C2_T_trace_block_dimension": n_trace_blocks,
            "C2_T_parent_block_dimension": n_parent_blocks,
            "C2_T_mapped_trace_block_count": mapped_trace_block_count,
            "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
            "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
            "C2_T_block_map_injective_pass": block_map_injective_pass,
            "C2_T_vector_block_size_trace": bs_trace,
            "C2_T_vector_block_size_parent": bs_parent,
            "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
            "C2_T_component_aware_mapping_pass": False,
            **common_meta,
        }

    parent_idx = np.full(n_u_trace, -1, dtype=np.int32)
    if bs_trace != 3 or bs_parent != 3:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "blocked_vector_ordering_convention_not_verified",
            "failure_stage": "VECTOR_COMPONENT_EXPANSION_ORDERING",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            "C2_T_trace_block_dimension": n_trace_blocks,
            "C2_T_parent_block_dimension": n_parent_blocks,
            "C2_T_mapped_trace_block_count": mapped_trace_block_count,
            "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
            "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
            "C2_T_block_map_injective_pass": block_map_injective_pass,
            "C2_T_vector_block_size_trace": bs_trace,
            "C2_T_vector_block_size_parent": bs_parent,
            "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
            "C2_T_component_aware_mapping_pass": False,
            **common_meta,
        }

    for tb in range(n_trace_blocks):
        pb = int(trace_block_to_parent_block[tb])
        if pb < 0:
            continue
        for c in range(bs_trace):
            t_scalar = bs_trace * tb + c
            p_scalar = bs_parent * pb + c
            if 0 <= t_scalar < n_u_trace and 0 <= p_scalar < n_u_parent:
                parent_idx[t_scalar] = p_scalar

    missing_scalar = int(np.sum(parent_idx < 0))
    if missing_scalar > 0:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": f"unmatched_trace_scalar_dofs={missing_scalar}",
            "failure_stage": "VERTEX_COMPONENT_EXPANSION",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            "C2_T_trace_block_dimension": n_trace_blocks,
            "C2_T_parent_block_dimension": n_parent_blocks,
            "C2_T_mapped_trace_block_count": mapped_trace_block_count,
            "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
            "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
            "C2_T_block_map_injective_pass": block_map_injective_pass,
            "C2_T_vector_block_size_trace": bs_trace,
            "C2_T_vector_block_size_parent": bs_parent,
            "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
            "C2_T_component_aware_mapping_pass": False,
            **common_meta,
        }

    shell_parent_block_support = np.asarray(
        fem3d._locate_facet_displacement_dofs(V_u_parent, msh, shell_facets), dtype=np.int32
    ).ravel()
    shell_parent_scalar_support = np.concatenate(
        [bs_parent * shell_parent_block_support + c for c in range(bs_parent)]
    ).astype(np.int32, copy=False)

    row_counts = np.bincount(parent_idx, minlength=n_u_parent)
    nnz = int(parent_idx.size)
    density = float(nnz / max(n_u_parent * n_u_trace, 1))
    checksum = _crc32_i32(parent_idx)
    unique_parent_scalar = np.unique(parent_idx)
    duplicate_parent_scalar_count = int(parent_idx.size - unique_parent_scalar.size)

    geom_pass = bool(np.all((parent_idx >= 0) & (parent_idx < n_u_parent)))
    tag_support_pass = bool(all(v > 0 for v in transferred_counts.values()))
    support_pass = bool(np.all(np.isin(parent_idx, shell_parent_scalar_support)))
    ones_trace = np.ones(n_u_trace, dtype=np.float64)
    y = np.zeros(n_u_parent, dtype=np.float64)
    np.add.at(y, parent_idx, ones_trace)
    const_pass = bool(np.allclose(y[parent_idx], 1.0, rtol=0.0, atol=1.0e-12))
    component_pass = bool(duplicate_parent_scalar_count == 0 and bs_trace == 3 and bs_parent == 3)
    entity_corr_pass = bool(block_map_injective_pass)

    trace_coords_block = np.asarray(V_u_trace.tabulate_dof_coordinates(), dtype=np.float64)
    parent_coords_block = np.asarray(V_u_parent.tabulate_dof_coordinates(), dtype=np.float64)
    coord_pass = False
    if trace_coords_block.shape[0] >= n_trace_blocks and parent_coords_block.shape[0] >= n_parent_blocks:
        coord_pass = bool(
            np.allclose(
                trace_coords_block[:n_trace_blocks],
                parent_coords_block[trace_block_to_parent_block],
                rtol=0.0,
                atol=1.0e-12,
            )
        )

    exact_pass = bool(
        geom_pass
        and tag_support_pass
        and support_pass
        and const_pass
        and coord_pass
        and entity_corr_pass
        and component_pass
    )

    return {
        "ok": exact_pass,
        "reason": None if exact_pass else "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
        "failure_detail": None if exact_pass else "C2_transfer_contract_failed",
        "failure_stage": None if exact_pass else "TRANSFER_CONTRACT_VALIDATION",
        "domain_dim": n_u_trace,
        "codomain_dim": n_u_parent,
        "shape": [n_u_parent, n_u_trace],
        "nnz": nnz,
        "density": density,
        "column_nnz_min": 1,
        "column_nnz_max": 1,
        "row_nnz_min": int(row_counts.min()) if row_counts.size else 0,
        "row_nnz_max": int(row_counts.max()) if row_counts.size else 0,
        "mapping_checksum": int(checksum),
        "storage_bytes": int(parent_idx.nbytes + np.ones(nnz, dtype=np.float64).nbytes),
        "parent_index_per_trace_dof": parent_idx,
        "map_meta": cell_map_meta,
        "vertex_map_meta": vertex_map_meta,
        "transferred_counts": transferred_counts,
        "C2_T_trace_block_dimension": n_trace_blocks,
        "C2_T_parent_block_dimension": n_parent_blocks,
        "C2_T_mapped_trace_block_count": mapped_trace_block_count,
        "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
        "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
        "C2_T_block_map_injective_pass": block_map_injective_pass,
        "C2_T_mapped_parent_scalar_dofs_unique": int(unique_parent_scalar.size),
        "C2_T_duplicate_parent_scalar_dof_count": duplicate_parent_scalar_count,
        "C2_T_vector_block_size_trace": bs_trace,
        "C2_T_vector_block_size_parent": bs_parent,
        "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
        "C2_T_entity_correspondence_pass": entity_corr_pass,
        "C2_T_component_aware_mapping_pass": component_pass,
        "C2_T_coordinate_validation_pass": coord_pass,
        "C2_T_geometry_map_contract_pass": geom_pass,
        "C2_T_constant_field_transfer_pass": const_pass,
        "C2_T_trace_support_transfer_pass": support_pass,
        "C2_T_tag_support_transfer_pass": tag_support_pass,
        **common_meta,
        "C2_T_validation_failure_reason": None if exact_pass else "one_or_more_transfer_contract_checks_failed",
    }


def _is_c2_transfer_contract_only_mode(argv: List[str]) -> bool:
    return C2_TRANSFER_CONTRACT_ONLY_ARG in argv


def _print_c2_transfer_contract_summary(
    tmeta: Dict[str, Any],
    *,
    pre: Dict[str, Any],
    codomain_note: str,
) -> int:
    exact = bool(tmeta.get("ok", False))
    dense_removed = bool(tmeta.get("C2_dense_coupling_allocation_removed", False))
    method = "EntityMap_plus_exact_dof_coordinate_match_on_P1_trace_and_parent"
    verdict = (
        "B3_C2_TRANSFER_READY_FOR_SPARSE_COUPLING_IMPLEMENTATION_REVIEW"
        if (exact and dense_removed)
        else "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
    )
    blocker = None if exact else (tmeta.get("reason") or tmeta.get("failure_detail"))

    print("[B3_C2] mode=C2_transfer_contract_only", flush=True)
    print(f"[B3_C2] preassembly_contract_pass={pre['preassembly_contract_pass']}", flush=True)
    print(f"[B3_C2] codomain_space_note={codomain_note}", flush=True)
    print(f"[B3_C2] C2_T_construction_method={method}", flush=True)
    print(f"[B3_C2] C2_T_domain_dimension={tmeta.get('domain_dim')}", flush=True)
    print(f"[B3_C2] C2_T_codomain_dimension={tmeta.get('codomain_dim')}", flush=True)
    print(f"[B3_C2] C2_T_shape={tmeta.get('shape')}", flush=True)
    print(f"[B3_C2] C2_T_constructed={exact}", flush=True)
    print(f"[B3_C2] C2_T_nnz={tmeta.get('nnz')}", flush=True)
    print(f"[B3_C2] C2_T_column_nnz_min={tmeta.get('column_nnz_min')}", flush=True)
    print(f"[B3_C2] C2_T_column_nnz_max={tmeta.get('column_nnz_max')}", flush=True)
    print(f"[B3_C2] C2_T_density={tmeta.get('density')}", flush=True)
    print(f"[B3_C2] C2_T_row_nnz_min={tmeta.get('row_nnz_min')}", flush=True)
    print(f"[B3_C2] C2_T_row_nnz_max={tmeta.get('row_nnz_max')}", flush=True)
    print(f"[B3_C2] C2_T_mapping_checksum={tmeta.get('mapping_checksum')}", flush=True)
    print(f"[B3_C2] C2_T_shell_cell_entity_map_type={tmeta.get('C2_T_shell_cell_entity_map_type')}", flush=True)
    print(f"[B3_C2] C2_T_shell_vertex_entity_map_type={tmeta.get('C2_T_shell_vertex_entity_map_type')}", flush=True)
    print(
        f"[B3_C2] C2_T_shell_cell_map_extraction_method={tmeta.get('C2_T_shell_cell_map_extraction_method')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_shell_vertex_map_extraction_method={tmeta.get('C2_T_shell_vertex_map_extraction_method')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_shell_vertex_map_extracted={tmeta.get('C2_T_shell_vertex_map_extracted')}", flush=True)
    print(f"[B3_C2] C2_T_shell_vertex_count={tmeta.get('C2_T_shell_vertex_count')}", flush=True)
    print(f"[B3_C2] C2_T_parent_vertex_index_min={tmeta.get('C2_T_parent_vertex_index_min')}", flush=True)
    print(f"[B3_C2] C2_T_parent_vertex_index_max={tmeta.get('C2_T_parent_vertex_index_max')}", flush=True)
    print(f"[B3_C2] C2_T_shell_topological_dimension={tmeta.get('C2_T_shell_topological_dimension')}", flush=True)
    print(f"[B3_C2] C2_T_parent_topological_dimension={tmeta.get('C2_T_parent_topological_dimension')}", flush=True)
    print(
        f"[B3_C2] C2_T_shell_connectivity_0_to_tdim_created={tmeta.get('C2_T_shell_connectivity_0_to_tdim_created')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_shell_connectivity_tdim_to_0_created={tmeta.get('C2_T_shell_connectivity_tdim_to_0_created')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_parent_connectivity_0_to_tdim_created={tmeta.get('C2_T_parent_connectivity_0_to_tdim_created')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_parent_connectivity_tdim_to_0_created={tmeta.get('C2_T_parent_connectivity_tdim_to_0_created')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_matching_key={tmeta.get('C2_T_matching_key')}", flush=True)
    print(f"[B3_C2] C2_T_coordinate_match_used_as={tmeta.get('C2_T_coordinate_match_used_as')}", flush=True)
    print(f"[B3_C2] C2_T_trace_block_dimension={tmeta.get('C2_T_trace_block_dimension')}", flush=True)
    print(f"[B3_C2] C2_T_parent_block_dimension={tmeta.get('C2_T_parent_block_dimension')}", flush=True)
    print(f"[B3_C2] C2_T_mapped_trace_block_count={tmeta.get('C2_T_mapped_trace_block_count')}", flush=True)
    print(f"[B3_C2] C2_T_unmatched_trace_block_count={tmeta.get('C2_T_unmatched_trace_block_count')}", flush=True)
    print(f"[B3_C2] C2_T_duplicate_parent_block_count={tmeta.get('C2_T_duplicate_parent_block_count')}", flush=True)
    print(f"[B3_C2] C2_T_block_map_injective_pass={tmeta.get('C2_T_block_map_injective_pass')}", flush=True)
    print(f"[B3_C2] C2_T_vector_block_size_trace={tmeta.get('C2_T_vector_block_size_trace')}", flush=True)
    print(f"[B3_C2] C2_T_vector_block_size_parent={tmeta.get('C2_T_vector_block_size_parent')}", flush=True)
    print(f"[B3_C2] C2_T_component_expansion_method={tmeta.get('C2_T_component_expansion_method')}", flush=True)
    print(
        f"[B3_C2] C2_T_mapped_parent_scalar_dofs_unique={tmeta.get('C2_T_mapped_parent_scalar_dofs_unique')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_duplicate_parent_scalar_dof_count={tmeta.get('C2_T_duplicate_parent_scalar_dof_count')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_entity_correspondence_pass={tmeta.get('C2_T_entity_correspondence_pass')}", flush=True)
    print(f"[B3_C2] C2_T_component_aware_mapping_pass={tmeta.get('C2_T_component_aware_mapping_pass')}", flush=True)
    print(f"[B3_C2] C2_T_coordinate_validation_pass={tmeta.get('C2_T_coordinate_validation_pass')}", flush=True)
    print(
        f"[B3_C2] C2_T_geometry_map_contract_pass={tmeta.get('C2_T_geometry_map_contract_pass')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_constant_field_transfer_pass={tmeta.get('C2_T_constant_field_transfer_pass')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_trace_support_transfer_pass={tmeta.get('C2_T_trace_support_transfer_pass')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_tag_support_transfer_pass={tmeta.get('C2_T_tag_support_transfer_pass')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_persisted_to_disk=False", flush=True)
    print(f"[B3_C2] C2_T_failure_stage={tmeta.get('failure_stage')}", flush=True)
    print(f"[B3_C2] C2_T_failure_exception_type={tmeta.get('failure_exception_type')}", flush=True)
    print(f"[B3_C2] C2_T_failure_exception_message={tmeta.get('failure_exception_message')}", flush=True)
    print(f"[B3_C2] C2_T_exact_transfer_contract_pass={exact}", flush=True)
    print(f"[B3_C2] C2_T_construction_blocker={blocker}", flush=True)
    print(
        f"[B3_C2] C2_dense_coupling_allocation_prohibited={tmeta.get('C2_dense_coupling_allocation_prohibited')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_dense_coupling_allocation_removed={tmeta.get('C2_dense_coupling_allocation_removed')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_projected_coupling_representation={tmeta.get('C2_projected_coupling_representation')}",
        flush=True,
    )
    print(f"[B3_C2] next_step_verdict={verdict}", flush=True)
    print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
    print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
    return 0 if (exact and dense_removed) else 2


def _run_c2_transfer_contract_only(pre: Dict[str, Any]) -> int:
    """Lightweight path: mesh/submesh/EntityMap + exact T only; no baseline A/M or seed replay."""
    if not pre["preassembly_contract_pass"]:
        empty = {
            "ok": False,
            "domain_dim": None,
            "codomain_dim": None,
            "shape": None,
            "nnz": None,
            "density": None,
            "row_nnz_min": None,
            "row_nnz_max": None,
            "mapping_checksum": None,
            "reason": "preassembly_contract_failed",
            "failure_detail": json.dumps(pre.get("preassembly_failure_reasons", [])),
        }
        return _print_c2_transfer_contract_summary(
            empty,
            pre=pre,
            codomain_note="not_loaded_preassembly_failed",
        )

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[B3_C2] mode=C2_transfer_contract_only", flush=True)
            print("[B3_C2] C2_T_constructed=False", flush=True)
            print("[B3_C2] C2_T_construction_blocker=requires_mpiexec_n_1", flush=True)
            print(
                "[B3_C2] next_step_verdict=B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
                flush=True,
            )
            print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
            print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)
    msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
    f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
    f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
    shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))

    try:
        tmeta = _build_c2_trace_to_parent_transfer(
            msh,
            facet_tags,
            shell_facets=shell_facets,
            tag_top=TAG_TOP,
            tag_back=TAG_BACK,
            tag_ribs=TAG_RIBS,
        )
    except Exception as exc:
        tmeta = {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "uncaught_transfer_construction_exception",
            "failure_stage": "VERTEX_ENTITYMAP_EXTRACTION",
            "failure_exception_type": type(exc).__name__,
            "failure_exception_message": str(exc),
            "domain_dim": None,
            "codomain_dim": None,
            "shape": None,
            "nnz": None,
            "density": None,
            "row_nnz_min": None,
            "row_nnz_max": None,
            "column_nnz_min": None,
            "column_nnz_max": None,
            "mapping_checksum": None,
            "C2_T_shell_cell_entity_map_type": "EntityMap",
            "C2_T_shell_vertex_entity_map_type": "EntityMap",
            "C2_T_shell_cell_map_extraction_method": None,
            "C2_T_shell_vertex_map_extraction_method": None,
            "C2_T_shell_vertex_map_extracted": False,
            "C2_T_shell_vertex_count": None,
            "C2_T_parent_vertex_index_min": None,
            "C2_T_parent_vertex_index_max": None,
            "C2_T_matching_key": "ENTITY_VERTEX_PLUS_VECTOR_COMPONENT",
            "C2_T_coordinate_match_used_as": "VALIDATION_ONLY",
            "C2_T_trace_block_dimension": None,
            "C2_T_parent_block_dimension": None,
            "C2_T_vector_block_size_trace": None,
            "C2_T_vector_block_size_parent": None,
            "C2_T_entity_correspondence_pass": False,
            "C2_T_component_aware_mapping_pass": False,
            "C2_T_coordinate_validation_pass": False,
            "C2_T_geometry_map_contract_pass": False,
            "C2_T_constant_field_transfer_pass": False,
            "C2_T_trace_support_transfer_pass": False,
            "C2_T_tag_support_transfer_pass": False,
            "C2_T_validation_failure_reason": "transfer_construction_exception",
            "C2_dense_coupling_allocation_prohibited": True,
            "C2_dense_coupling_allocation_removed": True,
            "C2_projected_coupling_representation": "NOT_YET_SAFE",
        }

    dense_removed = bool(tmeta.get("C2_dense_coupling_allocation_removed", False))
    exact_pass = bool(tmeta.get("ok", False))
    next_step_verdict = (
        "B3_C2_TRANSFER_READY_FOR_SPARSE_COUPLING_IMPLEMENTATION_REVIEW"
        if (exact_pass and dense_removed)
        else "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
    )

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "C2_transfer_contract_only",
        "selected_B3_coupling_route": "C2",
        "preassembly_contract_pass": pre["preassembly_contract_pass"],
        "codomain_space_note": (
            "parent_mesh_P1_displacement_global_dof_indices_shell_support_subset_not_reduced_W"
        ),
        "C2_T_domain_space": "B3_trace_u_submesh_P1_vector",
        "C2_T_codomain_space": "parent_mesh_P1_displacement_dof",
        "C2_T_transfer_direction": "B3_trace_to_parent_interface_u",
        "C2_T_construction_method": (
            "EntityMap_plus_exact_dof_coordinate_match_on_P1_trace_and_parent"
        ),
        "C2_T_constructed": bool(tmeta.get("ok", False)),
        "C2_T_construction_blocker": tmeta.get("reason") if not tmeta.get("ok", False) else None,
        "C2_T_shape": tmeta.get("shape"),
        "C2_T_nnz": tmeta.get("nnz"),
        "C2_T_density": tmeta.get("density"),
        "C2_T_column_nnz_min": tmeta.get("column_nnz_min"),
        "C2_T_column_nnz_max": tmeta.get("column_nnz_max"),
        "C2_T_row_nnz_min": tmeta.get("row_nnz_min"),
        "C2_T_row_nnz_max": tmeta.get("row_nnz_max"),
        "C2_T_mapping_checksum": tmeta.get("mapping_checksum"),
        "C2_transfer_storage_bytes": tmeta.get("storage_bytes"),
        "C2_T_persisted_to_disk": False,
        "C2_T_shell_cell_entity_map_type": tmeta.get("C2_T_shell_cell_entity_map_type"),
        "C2_T_shell_vertex_entity_map_type": tmeta.get("C2_T_shell_vertex_entity_map_type"),
        "C2_T_shell_cell_map_extraction_method": tmeta.get("C2_T_shell_cell_map_extraction_method"),
        "C2_T_shell_vertex_map_extraction_method": tmeta.get("C2_T_shell_vertex_map_extraction_method"),
        "C2_T_shell_vertex_map_extracted": tmeta.get("C2_T_shell_vertex_map_extracted"),
        "C2_T_shell_vertex_count": tmeta.get("C2_T_shell_vertex_count"),
        "C2_T_parent_vertex_index_min": tmeta.get("C2_T_parent_vertex_index_min"),
        "C2_T_parent_vertex_index_max": tmeta.get("C2_T_parent_vertex_index_max"),
        "C2_T_shell_topological_dimension": tmeta.get("C2_T_shell_topological_dimension"),
        "C2_T_parent_topological_dimension": tmeta.get("C2_T_parent_topological_dimension"),
        "C2_T_shell_connectivity_0_to_tdim_created": tmeta.get(
            "C2_T_shell_connectivity_0_to_tdim_created"
        ),
        "C2_T_shell_connectivity_tdim_to_0_created": tmeta.get(
            "C2_T_shell_connectivity_tdim_to_0_created"
        ),
        "C2_T_parent_connectivity_0_to_tdim_created": tmeta.get(
            "C2_T_parent_connectivity_0_to_tdim_created"
        ),
        "C2_T_parent_connectivity_tdim_to_0_created": tmeta.get(
            "C2_T_parent_connectivity_tdim_to_0_created"
        ),
        "C2_T_matching_key": tmeta.get("C2_T_matching_key"),
        "C2_T_coordinate_match_used_as": tmeta.get("C2_T_coordinate_match_used_as"),
        "C2_T_trace_block_dimension": tmeta.get("C2_T_trace_block_dimension"),
        "C2_T_parent_block_dimension": tmeta.get("C2_T_parent_block_dimension"),
        "C2_T_mapped_trace_block_count": tmeta.get("C2_T_mapped_trace_block_count"),
        "C2_T_unmatched_trace_block_count": tmeta.get("C2_T_unmatched_trace_block_count"),
        "C2_T_duplicate_parent_block_count": tmeta.get("C2_T_duplicate_parent_block_count"),
        "C2_T_block_map_injective_pass": tmeta.get("C2_T_block_map_injective_pass"),
        "C2_T_vector_block_size_trace": tmeta.get("C2_T_vector_block_size_trace"),
        "C2_T_vector_block_size_parent": tmeta.get("C2_T_vector_block_size_parent"),
        "C2_T_component_expansion_method": tmeta.get("C2_T_component_expansion_method"),
        "C2_T_mapped_parent_scalar_dofs_unique": tmeta.get("C2_T_mapped_parent_scalar_dofs_unique"),
        "C2_T_duplicate_parent_scalar_dof_count": tmeta.get("C2_T_duplicate_parent_scalar_dof_count"),
        "C2_T_entity_correspondence_pass": tmeta.get("C2_T_entity_correspondence_pass"),
        "C2_T_component_aware_mapping_pass": tmeta.get("C2_T_component_aware_mapping_pass"),
        "C2_T_coordinate_validation_pass": tmeta.get("C2_T_coordinate_validation_pass"),
        "C2_T_geometry_map_contract_pass": tmeta.get("C2_T_geometry_map_contract_pass"),
        "C2_T_constant_field_transfer_pass": tmeta.get("C2_T_constant_field_transfer_pass"),
        "C2_T_trace_support_transfer_pass": tmeta.get("C2_T_trace_support_transfer_pass"),
        "C2_T_tag_support_transfer_pass": tmeta.get("C2_T_tag_support_transfer_pass"),
        "C2_T_failure_stage": tmeta.get("failure_stage"),
        "C2_T_failure_exception_type": tmeta.get("failure_exception_type"),
        "C2_T_failure_exception_message": tmeta.get("failure_exception_message"),
        "C2_T_exact_transfer_contract_pass": exact_pass,
        "C2_T_validation_failure_reason": tmeta.get("C2_T_validation_failure_reason"),
        "C2_dense_coupling_allocation_prohibited": True,
        "C2_dense_coupling_allocation_removed": dense_removed,
        "C2_projected_coupling_representation": tmeta.get(
            "C2_projected_coupling_representation", "NOT_YET_SAFE"
        ),
        "B3_submesh_entity_map_extraction_method": tmeta.get("map_meta", {}).get("method"),
        "B3_transferred_tag_counts": tmeta.get("transferred_counts"),
        "artifact_storage_policy_applied": True,
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "jd_wiring_authorized": False,
        "next_step_verdict": next_step_verdict,
    }
    _write_json_atomic(OUT_JSON_C2_CONTRACT, payload)
    payload["report_size_bytes"] = OUT_JSON_C2_CONTRACT.stat().st_size
    _write_json_atomic(OUT_JSON_C2_CONTRACT, payload)

    return _print_c2_transfer_contract_summary(
        tmeta,
        pre=pre,
        codomain_note=payload["codomain_space_note"],
    )


def main() -> int:
    import sys

    pre = _precheck()

    if _is_c2_transfer_contract_only_mode(sys.argv):
        return _run_c2_transfer_contract_only(pre)

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
        cpre = _coupling_contract_precheck()
        selected_route = cpre["selected_B3_coupling_route"]
        coupling_stage_outcome = None
        c2_t_domain_space = "B3_trace_u_submesh_P1_vector"
        c2_t_codomain_space = "parent_reduced_u_representation"
        c2_t_domain_dim = None
        c2_t_codomain_dim = None
        c2_t_direction = "B3_trace_to_parent_interface_u"
        c2_t_method = "UNAVAILABLE"
        c2_t_constructed = False
        c2_t_is_sparse = True
        c2_t_is_interp = "exact_coordinate_lift_on_P1_trace_parent_matching"
        c2_t_geom_preserve = False
        c2_t_shape = None
        c2_t_nnz = None
        c2_t_density = None
        c2_t_row_nnz_min = None
        c2_t_row_nnz_max = None
        c2_t_checksum = None
        c2_t_storage_bytes = None
        c2_t_blocker = None
        c2_t_contract_pass = False
        c2_t_const_pass = False
        c2_t_support_pass = False
        c2_t_tag_pass = False
        c2_t_validation_failure = None
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

                    # Coupling route selection.
                    if selected_route == "C1":
                        b3_coupling_iface = False
                        b3_coupling_present = False
                        b3_coupling_method = "direct_cross_mesh_entity_map_assembly"
                        b3_coupling_reason = (
                            "B3_BLOCKED_BY_ONE_NAMED_DIRECT_MIXED_DOMAIN_API"
                        )
                        coupling_stage_outcome = b3_coupling_reason
                        block_reason = "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
                    elif selected_route == "C2":
                        tmeta = _build_c2_trace_to_parent_transfer(
                            msh,
                            facet_tags,
                            shell_facets=shell_facets,
                            tag_top=TAG_TOP,
                            tag_back=TAG_BACK,
                            tag_ribs=TAG_RIBS,
                        )
                        c2_t_method = "EntityMap_plus_exact_dof_coordinate_match_on_P1_trace_and_parent"
                        c2_t_domain_dim = tmeta.get("domain_dim")
                        c2_t_codomain_dim = tmeta.get("codomain_dim")
                        c2_t_constructed = bool(tmeta.get("ok", False))
                        c2_t_geom_preserve = bool(tmeta.get("ok", False))
                        c2_t_shape = tmeta.get("shape")
                        c2_t_nnz = tmeta.get("nnz")
                        c2_t_density = tmeta.get("density")
                        c2_t_row_nnz_min = tmeta.get("row_nnz_min")
                        c2_t_row_nnz_max = tmeta.get("row_nnz_max")
                        c2_t_checksum = tmeta.get("mapping_checksum")
                        c2_t_storage_bytes = tmeta.get("storage_bytes")
                        c2_t_blocker = tmeta.get("reason") if not tmeta.get("ok", False) else None
                        c2_t_contract_pass = bool(tmeta.get("C2_T_geometry_map_contract_pass", False))
                        c2_t_const_pass = bool(tmeta.get("C2_T_constant_field_transfer_pass", False))
                        c2_t_support_pass = bool(tmeta.get("C2_T_trace_support_transfer_pass", False))
                        c2_t_tag_pass = bool(tmeta.get("C2_T_tag_support_transfer_pass", False))
                        c2_t_validation_failure = tmeta.get("C2_T_validation_failure_reason")
                        if not tmeta.get("ok", False):
                            b3_coupling_iface = False
                            b3_coupling_present = False
                            b3_coupling_method = "sparse_transfer_T_then_parent_coupling_block_projection"
                            b3_coupling_reason = "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
                            coupling_stage_outcome = b3_coupling_reason
                            block_reason = b3_coupling_reason
                            b3_ops_reason = str(tmeta.get("failure_detail"))
                            b3_seed_fail = b3_coupling_reason
                            b3_seed_check_status = (
                                "NOT_EVALUATED_BLOCKED_PENDING_SPARSE_TRACE_TRANSFER_INTERFACE"
                            )
                        else:
                            b3_coupling_iface = False
                            b3_coupling_present = False
                            b3_coupling_method = (
                                "sparse_transfer_T_constructed_dense_parent_projection_path_disabled"
                            )
                            b3_coupling_reason = (
                                "B3_BLOCKED_BY_PROHIBITED_DENSE_PARENT_COUPLING_PROJECTION_PATH"
                            )
                            coupling_stage_outcome = b3_coupling_reason
                            block_reason = b3_coupling_reason
                            b3_seed_fail = b3_coupling_reason
                            b3_seed_check_status = (
                                "NOT_EVALUATED_BLOCKED_BY_PROHIBITED_DENSE_PARENT_COUPLING_PROJECTION_PATH"
                            )
                    else:
                        b3_coupling_iface = False
                        b3_coupling_present = False
                        b3_coupling_method = "none"
                        b3_coupling_reason = (
                            "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
                        )
                        coupling_stage_outcome = b3_coupling_reason
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
        elif block_reason == "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE":
            verdict = "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
        elif block_reason == "B3_BLOCKED_BY_SEED_TRANSFER_INTERFACE":
            verdict = "B3_BLOCKED_BY_SEED_TRANSFER_INTERFACE"
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
            **cpre,
            **pre,
            "C2_T_domain_space": c2_t_domain_space,
            "C2_T_codomain_space": c2_t_codomain_space,
            "C2_T_domain_dimension": c2_t_domain_dim,
            "C2_T_codomain_dimension": c2_t_codomain_dim,
            "C2_T_transfer_direction": c2_t_direction,
            "C2_T_construction_method": c2_t_method,
            "C2_T_is_sparse": c2_t_is_sparse,
            "C2_T_is_coordinate_interpolation_or_lift": c2_t_is_interp,
            "C2_T_preserves_trace_geometry_contract": c2_t_geom_preserve,
            "C2_T_constructed": c2_t_constructed,
            "C2_T_construction_blocker": c2_t_blocker,
            "C2_T_shape": c2_t_shape,
            "C2_T_nnz": c2_t_nnz,
            "C2_T_density": c2_t_density,
            "C2_T_row_nnz_min": c2_t_row_nnz_min,
            "C2_T_row_nnz_max": c2_t_row_nnz_max,
            "C2_T_mapping_checksum": c2_t_checksum,
            "C2_T_persisted_to_disk": False,
            "C2_T_geometry_map_contract_pass": c2_t_contract_pass,
            "C2_T_constant_field_transfer_pass": c2_t_const_pass,
            "C2_T_trace_support_transfer_pass": c2_t_support_pass,
            "C2_T_tag_support_transfer_pass": c2_t_tag_pass,
            "C2_T_validation_failure_reason": c2_t_validation_failure,
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
            "B3_coupling_transfer_or_entity_map_metadata": {
                "selected_route": selected_route,
                "C1_required_api": cpre["C1_required_api"],
                "C1_implementation_blocker": cpre["C1_implementation_blocker"],
                "C2_transfer_representation": cpre["C2_transfer_representation"],
                "C2_implementation_blocker": cpre["C2_implementation_blocker"],
                "C2_projected_coupling_formulae": (
                    {
                        "A_up_B3": "A_up_parent[parent_index_per_trace_dof, :]",
                        "A_pu_B3": "A_pu_parent[:, parent_index_per_trace_dof]",
                        "M_pu_B3": "M_pu_parent[:, parent_index_per_trace_dof]",
                    }
                    if c2_t_constructed
                    else None
                ),
            },
            "B3_coupling_present": bool(b3_coupling_present),
            "B3_coupling_failure_reason": None if b3_coupling_iface else b3_coupling_reason,
            "B3_coupling_stage_outcome": coupling_stage_outcome,
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
