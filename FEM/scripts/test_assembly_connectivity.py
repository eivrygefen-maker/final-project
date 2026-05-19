#!/usr/bin/env python3
"""
FSI coupling and facet-tag connectivity audit (no SLEPc eigen solve).

Loads the coupled P1+P1 mixed space W, assembles stiffness coupling blocks A_up / A_pu
(and reference blocks A_uu, A_pp), and reports Frobenius norms and nnz.

Usage (single MPI rank):
  mpiexec -n 1 python FEM/scripts/test_assembly_connectivity.py
  mpiexec -n 1 python FEM/scripts/test_assembly_connectivity.py --config FEM/configs/guitar_3d.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc  # noqa: E402
from mpi4py import MPI  # noqa: E402
import ufl  # noqa: E402
from basix.ufl import element, mixed_element  # noqa: E402
from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import assemble_matrix  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FEM_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fem_main_3d import (  # noqa: E402  # SLEPc is imported by fem_main_3d but not used here
    AIR_VOLUME_TAG,
    ROOT_RANK,
    WOOD_SURFACE_TAGS,
    _coupled_pressure_dof_scale,
    _load_mesh_and_tags,
)

TAG_TOP = 1
TAG_SOUNDHOLE = 2
TAG_BACK = 3
TAG_RIBS = 4
TAG_WOOD_FIX = 5


def _resolve_mesh_path(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, config_path.parents[1], FEM_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pressure_gauge_bcs(
    msh: mesh.Mesh,
    W: fem.FunctionSpace,
    facet_tags,
    config: dict,
) -> list:
    """Soundhole P=0 (same as production coupled solve)."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    soundhole_facets = np.array(facet_tags.find(TAG_SOUNDHOLE), dtype=np.int32)
    V_p, _ = W.sub(1).collapse()
    p_dofs = np.array(
        fem.locate_dofs_topological(V_p, fdim, soundhole_facets),
        dtype=np.int32,
    )
    if p_dofs.size == 0:
        raise RuntimeError("No pressure gauge DOFs on soundhole (tag 2).")
    p_zero = fem.Constant(msh, PETSc.ScalarType(0.0))
    return [fem.dirichletbc(p_zero, p_dofs, V_p)]


def _mat_frobenius_norm(K: PETSc.Mat) -> float:
    K.assemble()
    return float(K.norm())


def _mat_nnz_global(K: PETSc.Mat) -> int:
    K.assemble()
    info = K.getInfo()
    if isinstance(info, dict):
        return int(info.get("nz_used", info.get("nz_allocated", 0)))
    # Older petsc4py: tuple layout
    try:
        return int(info[0])
    except Exception:
        return -1


def _mixed_dof_split(W: fem.FunctionSpace) -> Tuple[int, int]:
    n_u = int(W.sub(0).dofmap.index_map.size_global * W.sub(0).dofmap.index_map_bs)
    n_p = int(W.sub(1).dofmap.index_map.size_global * W.sub(1).dofmap.index_map_bs)
    return n_u, n_p


def _count_up_pu_nnz(K: PETSc.Mat, n_u: int) -> Tuple[int, int]:
    """Count nnz in u→p (rows < n_u, cols >= n_u) and p→u blocks (blocked [u; p] ordering)."""
    K.assemble()
    nnz_up = 0
    nnz_pu = 0
    rstart, rend = K.getOwnershipRange()
    for row in range(rstart, rend):
        cols = K.getRow(row)[0]
        if row < n_u:
            nnz_up += int(np.sum(cols >= n_u))
        else:
            nnz_pu += int(np.sum(cols < n_u))
    nnz_up_g = MPI.COMM_WORLD.allreduce(nnz_up, op=MPI.SUM)
    nnz_pu_g = MPI.COMM_WORLD.allreduce(nnz_pu, op=MPI.SUM)
    return int(nnz_up_g), int(nnz_pu_g)


def _report_mat(label: str, K: PETSc.Mat, n_u: int) -> None:
    fnorm = _mat_frobenius_norm(K)
    nz = _mat_nnz_global(K)
    nz_up, nz_pu = _count_up_pu_nnz(K, n_u)
    print(
        f"  {label}: ||K||_F = {fnorm:.6e}, nnz = {nz}, "
        f"nnz(u→p) = {nz_up}, nnz(p→u) = {nz_pu}"
    )
    if nz == 0:
        print(f"    [FAIL] {label} is empty.")
    elif nz_up == 0 and nz_pu == 0 and ("A_up" in label or "M_pu" in label):
        print(f"    [FAIL] {label} has no u↔p off-diagonal entries (FSI disconnected).")
    elif nz_up > 0 or nz_pu > 0:
        print(f"    [OK] {label} has FSI coupling entries.")


def _facet_tag_counts(facet_tags) -> Dict[int, int]:
    out = {TAG_TOP: 0, TAG_SOUNDHOLE: 0, TAG_BACK: 0, TAG_RIBS: 0, TAG_WOOD_FIX: 0}
    for tag in out:
        out[tag] = int(facet_tags.find(tag).size)
    return out


def run_audit(config_path: Path) -> int:
    cfg = _load_config(config_path)
    mesh_file = _resolve_mesh_path(cfg, config_path)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print("=" * 72)
        print("FSI / assembly connectivity audit (no SLEPc)")
        print(f"  config : {config_path}")
        print(f"  mesh   : {mesh_file}")
        print("=" * 72)

    msh, cell_tags, facet_tags = _load_mesh_and_tags(mesh_file, status_callback=None)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        fac = _facet_tag_counts(facet_tags)
        print("\n[facet tags] local facet counts (find):")
        for tag, name in (
            (TAG_TOP, "Top_Plate"),
            (TAG_SOUNDHOLE, "Soundhole"),
            (TAG_BACK, "Back_Plate"),
            (TAG_RIBS, "Ribs_Sides"),
            (TAG_WOOD_FIX, "wood_fix (neck patch)"),
        ):
            print(f"  tag {tag} ({name}): {fac[tag]}")
        print(
            f"\n[note] Production WOOD_SURFACE_TAGS = {WOOD_SURFACE_TAGS} "
            f"→ coupling integrates on tags {WOOD_SURFACE_TAGS} only."
        )

    tdim = msh.topology.dim
    fdim = tdim - 1
    u_el = element("Lagrange", msh.basix_cell(), 1, shape=(3,))
    p_el = element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, mixed_element([u_el, p_el]))
    n_u, n_p = _mixed_dof_split(W)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(f"\n[mixed space] n_u = {n_u}, n_p = {n_p}, n_total = {n_u + n_p}")

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    xdmf_ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    xdmf_dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    n = ufl.FacetNormal(msh)

    solver_cfg = cfg.get("solver", {})
    p_scale = _coupled_pressure_dof_scale(solver_cfg)
    rho_air = float(cfg["materials"]["air"]["density"])

    ds_top = xdmf_ds(TAG_TOP)
    ds_back = xdmf_ds(TAG_BACK)
    ds_ribs = xdmf_ds(TAG_RIBS)
    wood_ds = ds_top + ds_back + ds_ribs

    a_up = -p_scale * p * ufl.dot(n, v) * wood_ds
    m_pu = p_scale * rho_air * ufl.dot(u, n) * q * wood_ds
    a_pp = (p_scale**2) * (1.0 / rho_air) * ufl.inner(ufl.grad(p), ufl.grad(q)) * xdmf_dx(AIR_VOLUME_TAG)
    # Minimal structural placeholder on wood (not orthotropic — norm reference only).
    a_uu = ufl.inner(ufl.grad(u), ufl.grad(v)) * wood_ds

    bcs = _pressure_gauge_bcs(msh, W, facet_tags, cfg)

    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(f"\n[forms] pressure_dof_scale = {p_scale:.4e}")
        print("[assembly] with soundhole pressure gauge BCs\n")

    mats = [
        ("A_up (p→u stiffness, full wood_ds)", assemble_matrix(fem.form(a_up), bcs=bcs)),
        ("M_pu (u→p mass, full wood_ds)", assemble_matrix(fem.form(m_pu), bcs=bcs)),
        ("A_up (tag1 top only)", assemble_matrix(fem.form(-p_scale * p * ufl.dot(n, v) * ds_top), bcs=bcs)),
        ("A_up (tag3 back only)", assemble_matrix(fem.form(-p_scale * p * ufl.dot(n, v) * ds_back), bcs=bcs)),
        ("A_up (tag4 ribs only)", assemble_matrix(fem.form(-p_scale * p * ufl.dot(n, v) * ds_ribs), bcs=bcs)),
        ("A_pp (air vol)", assemble_matrix(fem.form(a_pp), bcs=bcs)),
        ("A_uu (wood placeholder)", assemble_matrix(fem.form(a_uu), bcs=bcs)),
    ]

    if MPI.COMM_WORLD.rank == ROOT_RANK:
        for label, K in mats:
            _report_mat(label, K, n_u)
            K.destroy()

        print("\n[interpretation]")
        print("  • nnz(u→p)=0 AND nnz(p→u)=0 on A_up/A_pu → wood–air FSI is topologically disconnected.")
        print("  • FSI shell tags 1/3/4; tag 4 ribs clamped in production fem_main_3d.")
        print("  • Compare ||A_up||_F to ||A_uu||_F: large gap suggests block scaling / ill-conditioning.")
        print("=" * 72)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FSI coupling assembly audit (no SLEPc).")
    parser.add_argument(
        "--config",
        type=Path,
        default=FEM_ROOT / "configs" / "guitar_3d.json",
        help="FEM case JSON (default: FEM/configs/guitar_3d.json)",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    return run_audit(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
