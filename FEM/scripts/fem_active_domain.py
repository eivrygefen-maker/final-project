"""
Opt-in algebraic active-domain reduction for coupled GNHEP (experiment / validation only).

Enabled only when ``solver.active_domain_experiment.enabled`` is true in the case JSON.
Production runs without this key behave exactly as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

AIR_VOLUME_TAG = 10
WOOD_SURFACE_TAGS = (1, 3, 4)
RIBS_SURFACE_TAG = 4
WOOD_FIX_SURFACE_TAG = 5
ROOT_RANK = 0


def _solver_bool(cfg: Dict[str, Any], key: str, *, default: bool = False) -> bool:
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return default


def _u_global_dof_count(V_u) -> int:
    return int(V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs)


def _locate_facet_displacement_dofs(V_u, msh: mesh.Mesh, facet_indices: np.ndarray) -> np.ndarray:
    fdim = msh.topology.dim - 1
    return np.asarray(
        fem.locate_dofs_topological(
            V_u, fdim, np.asarray(facet_indices, dtype=np.int32)
        ),
        dtype=np.int32,
    ).ravel()

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
except ImportError:
    csr_matrix = None  # type: ignore
    connected_components = None  # type: ignore


def active_domain_experiment_enabled(solver_cfg: Dict[str, Any]) -> bool:
    block = solver_cfg.get("active_domain_experiment")
    if not isinstance(block, dict):
        return False
    return _solver_bool(block, "enabled", default=False)


def _seed_active_parent_indices(
    msh: mesh.Mesh,
    cell_tags,
    facet_tags,
    V_u_collapsed,
    V_p_collapsed,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    fsi_iface_facets: np.ndarray,
    *,
    clamp_ribs: bool,
    pin_fix_tag5: bool,
    p_bc_collapsed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """DOF seeds: air pressure, shell/FSI displacement, and constrained rows."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    u_map = np.asarray(u_to_W, dtype=np.int32).ravel()
    p_map = np.asarray(p_to_W, dtype=np.int32).ravel()

    air_cells = np.asarray(cell_tags.find(AIR_VOLUME_TAG), dtype=np.int32)
    p_coll = np.asarray(
        fem.locate_dofs_topological(V_p_collapsed, tdim, air_cells),
        dtype=np.int32,
    ).ravel() if air_cells.size > 0 else np.array([], dtype=np.int32)
    p_seed_w = np.unique(p_map[p_coll[(p_coll >= 0) & (p_coll < p_map.size)]]) if p_coll.size else np.array([], dtype=np.int32)

    u_chunks: List[np.ndarray] = []
    for tag in WOOD_SURFACE_TAGS:
        facets = np.asarray(facet_tags.find(int(tag)), dtype=np.int32)
        if facets.size > 0:
            u_chunks.append(
                np.asarray(
                    _locate_facet_displacement_dofs(V_u_collapsed, msh, facets),
                    dtype=np.int32,
                ).ravel()
            )
    if fsi_iface_facets.size > 0:
        u_chunks.append(
            np.asarray(
                _locate_facet_displacement_dofs(V_u_collapsed, msh, fsi_iface_facets),
                dtype=np.int32,
            ).ravel()
        )
    if clamp_ribs:
        ribs = np.asarray(facet_tags.find(RIBS_SURFACE_TAG), dtype=np.int32)
        if ribs.size > 0:
            u_chunks.append(
                np.asarray(
                    _locate_facet_displacement_dofs(V_u_collapsed, msh, ribs),
                    dtype=np.int32,
                ).ravel()
            )
    if pin_fix_tag5:
        fix_f = np.asarray(facet_tags.find(WOOD_FIX_SURFACE_TAG), dtype=np.int32)
        if fix_f.size > 0:
            u_chunks.append(
                np.asarray(
                    _locate_facet_displacement_dofs(V_u_collapsed, msh, fix_f),
                    dtype=np.int32,
                ).ravel()
            )

    u_coll = (
        np.unique(np.concatenate(u_chunks).astype(np.int32, copy=False))
        if u_chunks
        else np.array([], dtype=np.int32)
    )
    u_seed_w = np.unique(u_map[u_coll[(u_coll >= 0) & (u_coll < u_map.size)]]) if u_coll.size else np.array([], dtype=np.int32)

    if p_bc_collapsed.size > 0:
        pc = p_bc_collapsed[(p_bc_collapsed >= 0) & (p_bc_collapsed < p_map.size)]
        p_seed_w = np.unique(np.concatenate([p_seed_w, p_map[pc]]).astype(np.int32, copy=False))

    counts = {
        "p_air_collapsed": int(p_coll.size),
        "u_shell_iface_collapsed": int(u_coll.size),
        "p_seed_parent": int(p_seed_w.size),
        "u_seed_parent": int(u_seed_w.size),
    }
    return (
        np.unique(np.concatenate([u_seed_w, p_seed_w]).astype(np.int32, copy=False))
        if (u_seed_w.size or p_seed_w.size)
        else np.array([], dtype=np.int32),
        p_seed_w,
        counts,
    )


def _active_indices_graph_closure(
    A: PETSc.Mat,
    seed_indices: np.ndarray,
    *,
    comm: MPI.Intracomm,
) -> np.ndarray:
    """
    Connected components of the undirected graph of |A| entries, keeping the component
    that contains all seeds (preserves FSI coupling paths).
    """
    if seed_indices.size == 0:
        raise RuntimeError("active_domain: empty seed index set")
    n_global = int(A.getSize()[0])
    if csr_matrix is None or connected_components is None:
        return _active_indices_bfs_fallback(A, seed_indices, n_global=n_global)

    owned = A.getOwnershipRange()
    r0, r1 = int(owned[0]), int(owned[1])
    local_n = r1 - r0
    if local_n <= 0:
        if comm.rank == ROOT_RANK:
            return np.unique(seed_indices.astype(np.int32, copy=False))
        return np.array([], dtype=np.int32)

    indptr, indices, _ = A.getCSRSubMatrix()
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    data = np.ones(indices.shape[0], dtype=np.uint8)
    local_csr = csr_matrix((data, indices, indptr), shape=(local_n, n_global))
    sym = local_csr + local_csr.T
    sym.data[:] = 1

    n_comp, labels = connected_components(sym, directed=False, return_labels=True)
    seed_local = seed_indices[(seed_indices >= r0) & (seed_indices < r1)] - r0
    seed_labels = set(int(labels[int(i - r0)]) for i in seed_local if r0 <= i < r1)
    if not seed_labels:
        gathered = comm.gather(np.array(list(seed_labels), dtype=np.int32), root=ROOT_RANK)
        if comm.rank == ROOT_RANK:
            for part in gathered:
                seed_labels.update(int(x) for x in np.asarray(part).ravel())
        seed_labels_arr = np.array(sorted(seed_labels), dtype=np.int32)
        seed_labels_broadcast = comm.bcast(seed_labels_arr, root=ROOT_RANK)
        seed_labels = set(int(x) for x in np.asarray(seed_labels_broadcast).ravel())

    active_local = np.where(np.isin(labels, list(seed_labels)))[0] + r0
    active_local = np.unique(active_local.astype(np.int32, copy=False))
    all_active = comm.gather(active_local, root=ROOT_RANK)
    if comm.rank != ROOT_RANK:
        return np.array([], dtype=np.int32)
    out: Set[int] = set(int(x) for x in seed_indices.ravel())
    for part in all_active:
        out.update(int(x) for x in np.asarray(part, dtype=np.int32).ravel())
    return np.array(sorted(out), dtype=np.int32)


def _active_indices_bfs_fallback(
    A: PETSc.Mat,
    seed_indices: np.ndarray,
    *,
    n_global: int,
) -> np.ndarray:
    """Rank-0 BFS on matrix rows (fine for validation mesh sizes)."""
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return np.array([], dtype=np.int32)
    seeds = set(int(i) for i in np.asarray(seed_indices, dtype=np.int32).ravel())
    active: Set[int] = set(seeds)
    frontier = list(seeds)
    while frontier:
        nxt: List[int] = []
        for i in frontier:
            try:
                cols, _ = A.getRow(i)
            except Exception:
                continue
            for j in np.asarray(cols, dtype=np.int32).ravel():
                jj = int(j)
                if 0 <= jj < n_global and jj not in active:
                    active.add(jj)
                    nxt.append(jj)
        frontier = nxt
    return np.array(sorted(active), dtype=np.int32)


def restrict_operators_to_active_set(
    A: PETSc.Mat,
    M: PETSc.Mat,
    active_W_indices: np.ndarray,
) -> Tuple[PETSc.Mat, PETSc.Mat, Dict[str, Any]]:
    """Extract square submatrices on ``active_W_indices`` (global mixed ordering)."""
    comm = A.getComm()
    active_W_indices = np.unique(np.asarray(active_W_indices, dtype=np.int32).ravel())
    is_red = PETSc.IS().createGeneral(active_W_indices, comm=comm)
    try:
        is_red.setType(PETSc.IS.Type.GENERAL)
    except Exception:
        pass
    A_red = A.createSubMatrix(is_red, is_red, PETSc.Mat.Structure.SUBMATRIX)
    M_red = M.createSubMatrix(is_red, is_red, PETSc.Mat.Structure.SUBMATRIX)
    A_red.assemble()
    M_red.assemble()
    is_red.destroy()
    meta = {
        "method": "algebraic_restriction",
        "n_active": int(active_W_indices.size),
        "n_full": int(A.getSize()[0]),
        "active_W_indices": active_W_indices.tolist(),
    }
    return A_red, M_red, meta


def build_parent_to_local_map(
    active_W_indices: np.ndarray,
    n_full: int,
) -> np.ndarray:
    """``parent_to_local[parent_idx] = local reduced index, else -1``."""
    parent_to_local = np.full(int(n_full), -1, dtype=np.int32)
    idx = np.asarray(active_W_indices, dtype=np.int32).ravel()
    parent_to_local[idx] = np.arange(idx.size, dtype=np.int32)
    return parent_to_local


def remap_parent_indices_to_reduced(
    parent_indices: np.ndarray,
    parent_to_local: np.ndarray,
) -> np.ndarray:
    parent_indices = np.asarray(parent_indices, dtype=np.int32).ravel()
    out = parent_to_local[parent_indices]
    if np.any(out < 0):
        bad = int(np.sum(out < 0))
        raise RuntimeError(
            f"active_domain: {bad} parent indices are outside the active subgraph "
            "(increase seed set or check FSI closure)."
        )
    return out.astype(np.int32, copy=False)


def prolongate_to_full_mixed_vector(
    vec_red: np.ndarray,
    active_W_indices: np.ndarray,
    n_full: int,
) -> np.ndarray:
    """Scatter reduced eigenvector entries into full mixed ``W`` layout."""
    out = np.zeros(int(n_full), dtype=np.float64)
    idx = np.asarray(active_W_indices, dtype=np.int32).ravel()
    v = np.asarray(vec_red, dtype=np.float64).reshape(-1)
    if v.size != idx.size:
        raise ValueError(
            f"prolongate: reduced vector length {v.size} != active index count {idx.size}"
        )
    out[idx] = v
    return out


def apply_active_domain_reduction(
    A: PETSc.Mat,
    M: PETSc.Mat,
    W: fem.FunctionSpace,
    msh: mesh.Mesh,
    cell_tags,
    facet_tags,
    *,
    u_to_W_map: np.ndarray,
    p_to_W_map: np.ndarray,
    coupled_dirichlet_rows: np.ndarray,
    fsi_iface_facets: np.ndarray,
    p_gauge_dofs_v: np.ndarray,
    solver_cfg: Dict[str, Any],
    timing_dir: Optional[Path] = None,
) -> Tuple[PETSc.Mat, PETSc.Mat, Dict[str, Any]]:
    """
    Replace ``(A, M)`` with algebraically restricted operators on the active DOF subgraph.

    Full ``W`` is unchanged; eigenvectors must be prolongated before export.
    """
    V_u, _ = W.sub(0).collapse()
    V_p, _ = W.sub(1).collapse()
    clamp_ribs = _solver_bool(solver_cfg, "clamp_ribs", default=True)
    pin_fix = _solver_bool(solver_cfg, "eps_pin_fix_tag5", default=True)

    seed_w, p_seed_w, seed_counts = _seed_active_parent_indices(
        msh,
        cell_tags,
        facet_tags,
        V_u,
        V_p,
        u_to_W_map,
        p_to_W_map,
        fsi_iface_facets,
        clamp_ribs=clamp_ribs,
        pin_fix_tag5=pin_fix,
        p_bc_collapsed=np.asarray(p_gauge_dofs_v, dtype=np.int32).ravel(),
    )
    if coupled_dirichlet_rows.size > 0:
        seed_w = np.unique(
            np.concatenate([seed_w, np.asarray(coupled_dirichlet_rows, dtype=np.int32).ravel()]).astype(
                np.int32, copy=False
            )
        )

    active_W = _active_indices_graph_closure(A, seed_w, comm=A.getComm())
    if MPI.COMM_WORLD.rank == ROOT_RANK and active_W.size == 0:
        raise RuntimeError("active_domain: graph closure produced zero active indices")

    A_red, M_red, meta = restrict_operators_to_active_set(A, M, active_W)
    n_u_col = _u_global_dof_count(V_u)
    n_p_col = int(V_p.dofmap.index_map.size_global * V_p.dofmap.index_map_bs)
    n_full = int(A.getSize()[0])
    u_map = np.asarray(u_to_W_map, dtype=np.int32).ravel()
    p_map = np.asarray(p_to_W_map, dtype=np.int32).ravel()
    active_set = set(int(i) for i in active_W)
    parent_to_local = build_parent_to_local_map(active_W, n_full)
    meta.update(
        {
            "seed_counts": seed_counts,
            "n_u_collapsed_full": n_u_col,
            "n_p_collapsed_full": n_p_col,
            "n_u_active": int(sum(1 for i in u_map if int(i) in active_set)),
            "n_p_active": int(sum(1 for i in p_map if int(i) in active_set)),
            "soundhole_p_bc_collapsed": int(np.asarray(p_gauge_dofs_v).size),
            "fsi_iface_facets": int(fsi_iface_facets.size),
            "parent_to_local": parent_to_local.tolist(),
            "n_full": n_full,
        }
    )
    try:
        meta["A_red_norm_f"] = float(A_red.norm())
        meta["M_red_norm_f"] = float(M_red.norm())
        meta["A_full_norm_f"] = float(A.norm())
    except Exception:
        pass

    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(
            "[active_domain] algebraic restriction: "
            f"n_active={meta['n_active']}/{meta['n_full']} "
            f"(u_active={meta['n_u_active']}/{n_u_col}, "
            f"p_active={meta['n_p_active']}/{n_p_col}, "
            f"seeds={seed_counts})",
            flush=True,
        )

    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    if timing_dir is not None and MPI.COMM_WORLD.rank == ROOT_RANK:
        timing_dir.mkdir(parents=True, exist_ok=True)
        (timing_dir / "active_domain_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    return A_red, M_red, meta


def write_dof_audit(
    path: Path,
    *,
    formulation: str,
    mesh_audit: Dict[str, Any],
    operator_meta: Dict[str, Any],
) -> None:
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    payload = {
        "formulation": formulation,
        "mesh": mesh_audit,
        "operators": operator_meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
