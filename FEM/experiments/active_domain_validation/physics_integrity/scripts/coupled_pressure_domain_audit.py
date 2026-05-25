#!/usr/bin/env python3
"""
Coupled active-pressure-domain audit (no SLEPc solve).

Compares full-mesh coupled pressure DOFs vs acoustic-only air-restricted domain,
and audits row activity on wood-only/inactive pressure DOFs in A_pp, M_pp, and coupled A/M.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
import ufl
from basix.ufl import element
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
from fem_active_domain import restrict_operators_to_active_set
from mpi4py import MPI
from petsc4py import PETSc


def _resolve_mesh(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, EXPERIMENT_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_acoustic_only_reference() -> Dict[str, Any]:
    path = PHYSICS_ROOT / "acoustic_only" / "results" / "result_acoustic.json"
    audit = PHYSICS_ROOT / "acoustic_only" / "diagnostics" / "physics_integrity_audit.json"
    out: Dict[str, Any] = {}
    if path.is_file():
        out.update(json.loads(path.read_text(encoding="utf-8")))
    if audit.is_file():
        pi = json.loads(audit.read_text(encoding="utf-8"))
        out["acoustic_pressure_restriction"] = pi.get("acoustic_pressure_restriction") or {}
    return out


def _petsc_row_l2_norm(mat: PETSc.Mat, row: int) -> float:
    cols, vals = mat.getRow(int(row))
    try:
        v = np.asarray(vals, dtype=np.float64).ravel()
        return float(np.linalg.norm(v)) if v.size else 0.0
    finally:
        mat.restoreRow(int(row))


def _petsc_row_nnz(mat: PETSc.Mat, row: int) -> int:
    cols, vals = mat.getRow(int(row))
    try:
        return int(len(cols))
    finally:
        mat.restoreRow(int(row))


def _summarize_row_norms(
    mat: PETSc.Mat,
    rows: np.ndarray,
    *,
    label: str,
    max_rows: int = 512,
) -> Dict[str, Any]:
    rows = np.unique(np.asarray(rows, dtype=np.int32).ravel())
    if rows.size == 0:
        return {"label": label, "n_rows": 0}
    if rows.size > max_rows:
        idx = np.linspace(0, rows.size - 1, max_rows, dtype=np.int32)
        sample = rows[idx]
    else:
        sample = rows
    norms = [_petsc_row_l2_norm(mat, int(r)) for r in sample]
    nnz = [_petsc_row_nnz(mat, int(r)) for r in sample]
    arr = np.asarray(norms, dtype=np.float64)
    return {
        "label": label,
        "n_rows": int(rows.size),
        "n_sampled": int(sample.size),
        "row_l2_max": float(np.max(arr)),
        "row_l2_median": float(np.median(arr)),
        "row_l2_mean": float(np.mean(arr)),
        "row_nnz_max": int(np.max(nnz)),
        "row_nnz_median": float(np.median(nnz)),
    }


def _row_column_support_split(
    mat: PETSc.Mat,
    row: int,
    *,
    u_W: np.ndarray,
    p_air_W: np.ndarray,
    p_inactive_W: np.ndarray,
) -> Dict[str, int]:
    cols, vals = mat.getRow(int(row))
    try:
        c = np.asarray(cols, dtype=np.int32).ravel()
        v = np.asarray(vals, dtype=np.float64).ravel()
        mask = np.abs(v) > 0.0
        c = c[mask]
        u_set = set(int(x) for x in u_W)
        pa_set = set(int(x) for x in p_air_W)
        pi_set = set(int(x) for x in p_inactive_W)
        n_u = sum(1 for j in c if int(j) in u_set)
        n_pa = sum(1 for j in c if int(j) in pa_set)
        n_pi = sum(1 for j in c if int(j) in pi_set)
        n_other = int(c.size) - n_u - n_pa - n_pi
        return {
            "nnz": int(c.size),
            "cols_on_u": n_u,
            "cols_on_p_air": n_pa,
            "cols_on_p_inactive": n_pi,
            "cols_other": n_other,
        }
    finally:
        mat.restoreRow(int(row))


def _assemble_pressure_only_app_mpp(
    msh,
    cell_tags,
    config: Dict[str, Any],
) -> Tuple[PETSc.Mat, PETSc.Mat, fem.FunctionSpace, float, float]:
    """Same a_pp/m_pp forms as coupled/acoustic (dx on air tag 10 only), before restriction."""
    solver_cfg = config.get("solver", {})
    air_mat = config["materials"]["air"]
    rho_air = float(air_mat["density"])
    c_air = float(air_mat["speed_of_sound"])
    p_scale = fem3d._coupled_pressure_dof_scale(solver_cfg)
    p2 = p_scale * p_scale
    p_el = element("Lagrange", msh.basix_cell(), 1)
    V_p = fem.functionspace(msh, p_el)
    p = ufl.TrialFunction(V_p)
    q = ufl.TestFunction(V_p)
    xdmf_dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)

    a_pp = p2 * (1.0 / rho_air) * ufl.inner(ufl.grad(p), ufl.grad(q)) * xdmf_dx(fem3d.AIR_VOLUME_TAG)
    m_pp = p2 * (1.0 / (rho_air * c_air * c_air)) * p * q * xdmf_dx(fem3d.AIR_VOLUME_TAG)
    s_pp = 1.0
    if fem3d._solver_bool(solver_cfg, "gnhep_block_frobenius_normalize", default=True):
        s_pp = max(fem3d._mat_frobenius_norm(a_pp, label="audit_a_pp"), 1.0e-30)
        inv_p = 1.0 / s_pp
        a_pp = inv_p * a_pp
        m_pp = inv_p * m_pp
    A = assemble_matrix(fem.form(a_pp), bcs=[])
    A.assemble()
    M = assemble_matrix(fem.form(m_pp), bcs=[])
    M.assemble()
    return A, M, V_p, float(s_pp), float(p_scale)


def main() -> int:
    parser = argparse.ArgumentParser(description="Coupled pressure-domain / eigenbranch audit")
    parser.add_argument(
        "--config",
        type=Path,
        default=PHYSICS_ROOT / "configs" / "coupled_near_acoustic_244hz.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PHYSICS_ROOT / "diagnostics" / "coupled_pressure_domain",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[coupled_pressure_domain_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg.setdefault("solver", {})["physics_integrity_capture"] = True
    cfg["solver"]["active_domain_experiment"] = {"enabled": False}
    cfg["solver"]["coupled_air_pressure_restriction_diagnosis"] = False

    mesh_file = _resolve_mesh(cfg, args.config.resolve())
    sorting = args.out_dir / "sorting_audit"
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())

    if MPI.COMM_WORLD.rank == 0:
        print("[coupled_pressure_domain_audit] Assembling full coupled A/M (no solve)...", flush=True)

    msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    _msh, W, A_cpl, M_cpl = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )

    V_p_c, p_to_W = W.sub(1).collapse()
    p_to_W = np.asarray(p_to_W, dtype=np.int32).ravel()
    u_to_W = np.asarray(W.sub(0).collapse()[1], dtype=np.int32).ravel()
    n_p_full = int(V_p_c.dofmap.index_map.size_global * V_p_c.dofmap.index_map_bs)
    n_u = int(W.sub(0).dofmap.index_map.size_global * W.sub(0).dofmap.index_map_bs)
    n_W = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)

    p_air_v = fem3d._locate_air_volume_pressure_dofs(V_p_c, msh, cell_tags)
    all_p_v = np.arange(n_p_full, dtype=np.int32)
    p_inactive_v = np.setdiff1d(all_p_v, p_air_v, assume_unique=True)

    p_air_W = np.unique(p_to_W[p_air_v])
    p_inactive_W = np.unique(p_to_W[p_inactive_v])

    soundhole_facets = np.asarray(facet_tags.find(2), dtype=np.int32)
    sh_v = fem3d._locate_soundhole_pressure_release_dofs(V_p_c, soundhole_facets)
    sh_active_v = np.intersect1d(sh_v, p_air_v)
    sh_active_W = np.unique(p_to_W[sh_active_v]) if sh_active_v.size else np.array([], dtype=np.int32)

    acoustic_ref = _load_acoustic_only_reference()
    apr = acoustic_ref.get("acoustic_pressure_restriction") or {}
    n_p_active_acoustic = int(apr.get("n_p_active", acoustic_ref.get("n_p_active", 9998)))

    # Pressure-only operators on full V_p (air forms only — wood-only rows are structurally zero).
    A_pp, M_pp, _V_p_standalone, s_pp_audit, p_scale = _assemble_pressure_only_app_mpp(
        msh, cell_tags, cfg
    )

    inactive_row_stats_app = _summarize_row_norms(A_pp, p_inactive_v, label="A_pp_inactive_p_collapsed")
    inactive_row_stats_mpp = _summarize_row_norms(M_pp, p_inactive_v, label="M_pp_inactive_p_collapsed")
    inactive_row_stats_A = _summarize_row_norms(A_cpl, p_inactive_W, label="A_coupled_inactive_p_W")
    inactive_row_stats_M = _summarize_row_norms(M_cpl, p_inactive_W, label="M_coupled_inactive_p_W")

    sample_inactive_W = p_inactive_W[: min(32, p_inactive_W.size)]
    col_support = [
        _row_column_support_split(
            A_cpl,
            int(r),
            u_W=u_to_W,
            p_air_W=p_air_W,
            p_inactive_W=p_inactive_W,
        )
        for r in sample_inactive_W
    ]

    active_W = np.unique(np.concatenate([u_to_W, p_air_W]).astype(np.int32))
    n_reduced = int(active_W.size)
    A_red, M_red, restr_meta = restrict_operators_to_active_set(A_cpl, M_cpl, active_W)
    slepc_n_reduced = int(A_red.getSize()[0])

    feasibility = {
        "method": "restrict_operators_to_active_set (same as acoustic-only)",
        "active_W_indices_count": n_reduced,
        "expected_n_u_plus_n_p_air": int(n_u + p_air_v.size),
        "n_p_air_supported": int(p_air_v.size),
        "acoustic_only_n_p_active_reference": n_p_active_acoustic,
        "n_p_air_matches_acoustic_active": bool(int(p_air_v.size) == n_p_active_acoustic),
        "reduced_matrix_n": slepc_n_reduced,
        "full_coupled_matrix_n": int(A_cpl.getSize()[0]),
        "feasible_algebraic_restriction": bool(
            slepc_n_reduced == n_reduced and n_reduced == n_u + int(p_air_v.size)
        ),
        "restriction_meta": restr_meta,
    }

    pi = cfg.get("_physics_integrity") or {}
    report = {
        "experiment": "coupled_pressure_domain_audit",
        "mesh_file": str(mesh_file),
        "pressure_dof_counts": {
            "n_p_full": n_p_full,
            "n_p_air_supported": int(p_air_v.size),
            "n_p_wood_only_or_inactive": int(p_inactive_v.size),
            "n_u": n_u,
            "n_coupled_W": n_W,
            "soundhole_p_full_collapsed": int(sh_v.size),
            "soundhole_p_active_collapsed": int(sh_active_v.size),
            "soundhole_p_active_W": int(sh_active_W.size),
        },
        "acoustic_only_comparison": {
            "n_p_full_reference": int(apr.get("n_p_full", n_p_full)),
            "n_p_active_reference": n_p_active_acoustic,
            "method": apr.get("method", "algebraic_air_volume_dofs"),
            "delta_p_active_coupled_vs_acoustic": int(p_air_v.size) - n_p_active_acoustic,
        },
        "coupled_matrix_retains_inactive_pressure_rows": True,
        "inactive_pressure_rows_have_nontrivial_coupled_blocks": {
            "A_coupled": inactive_row_stats_A.get("row_l2_max", 0.0) > 1.0e-14,
            "M_coupled": inactive_row_stats_M.get("row_l2_max", 0.0) > 1.0e-14,
        },
        "inactive_row_norms": {
            "A_pp_physical_acoustic": inactive_row_stats_app,
            "M_pp_physical_acoustic": inactive_row_stats_mpp,
            "A_coupled_final": inactive_row_stats_A,
            "M_coupled_final": inactive_row_stats_M,
        },
        "inactive_row_column_support_sample": col_support,
        "gnhep_audit": {
            "s_pp_pressure_only": s_pp_audit,
            "pressure_dof_scale": p_scale,
            "coupled_physics_integrity": pi,
        },
        "algebraic_restriction_feasibility": feasibility,
        "diagnostic_solve_prepared": {
            "config": "configs/coupled_near_acoustic_air_p_restricted.json",
            "runner": "scripts/run_coupled_near_acoustic_air_p_restricted.sh",
            "solver_flag": "coupled_air_pressure_restriction_diagnosis",
            "note": (
                "Run separately after this audit; restricts to all u + air-supported p only, "
                "same weak forms/BCs/materials. Harvest all converged modes; rank by f≈244.39 Hz "
                "and by p_frac_energy_phys."
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "coupled_pressure_domain_audit.json", report)

    md = [
        "# Coupled pressure-domain audit",
        "",
        "## DOF counts",
        "",
        f"| Metric | Coupled | Acoustic-only ref |",
        f"|--------|---------|-------------------|",
        f"| n_p full | {n_p_full} | {apr.get('n_p_full', '—')} |",
        f"| n_p air-supported | {p_air_v.size} | {n_p_active_acoustic} |",
        f"| n_p inactive/wood-only | {p_inactive_v.size} | 0 (dropped) |",
        f"| n_u | {n_u} | 0 |",
        f"| n_W coupled | {n_W} | — |",
        f"| soundhole p active | {sh_active_v.size} | {apr.get('soundhole_p_dof_active', '—')} |",
        "",
        "## Inactive pressure rows",
        "",
        f"- Coupled **retains** {p_inactive_v.size} inactive pressure DOFs as matrix rows/columns.",
        f"- A_pp/M_pp (air dx only): inactive row max L2 ≈ "
        f"{inactive_row_stats_app.get('row_l2_max', 0):.3e} / "
        f"{inactive_row_stats_mpp.get('row_l2_max', 0):.3e}",
        f"- Final coupled A/M inactive p-row max L2 ≈ "
        f"{inactive_row_stats_A.get('row_l2_max', 0):.3e} / "
        f"{inactive_row_stats_M.get('row_l2_max', 0):.3e}",
        "",
        "## Restriction feasibility",
        "",
        f"- Reduced operator size: **{slepc_n_reduced}** (full {n_W})",
        f"- Matches n_u + n_p_air: **{feasibility['feasible_algebraic_restriction']}**",
        "",
        "## Next diagnostic solve",
        "",
        "```bash",
        "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "run_coupled_near_acoustic_air_p_restricted.sh",
        "```",
    ]
    (args.out_dir / "coupled_pressure_domain_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    try:
        A_cpl.destroy()
        M_cpl.destroy()
        A_pp.destroy()
        M_pp.destroy()
        A_red.destroy()
        M_red.destroy()
    except Exception:
        pass

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[coupled_pressure_domain_audit] n_p_full={n_p_full} n_p_air={p_air_v.size} "
            f"n_p_inactive={p_inactive_v.size} acoustic_active_ref={n_p_active_acoustic}"
        )
        print(
            f"[coupled_pressure_domain_audit] coupled inactive row max ||A||_row="
            f"{inactive_row_stats_A.get('row_l2_max', 0):.3e} "
            f"feasible_restriction={feasibility['feasible_algebraic_restriction']}"
        )
        print(f"[coupled_pressure_domain_audit] wrote {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
