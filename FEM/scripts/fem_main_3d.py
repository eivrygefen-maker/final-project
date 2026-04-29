import json
import logging
import math
import gc
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

try:
    import meshio
except Exception:
    meshio = None

LOGGER = logging.getLogger("fem3d_dolfinx")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


WOOD_SURFACE_TAGS = (1, 3)
AIR_VOLUME_TAG = 10


def _emit(message: str, status_callback=None, level: str = "info") -> None:
    if level == "error":
        LOGGER.error(message)
    elif level == "warning":
        LOGGER.warning(message)
    else:
        LOGGER.info(message)
    print(message)
    sys.stdout.flush()
    sys.stderr.flush()
    if status_callback is not None:
        status_callback(message)


def _wipe_cache_folder(cache_dir: Path, status_callback=None) -> None:
    if not cache_dir.exists():
        _emit(f"[cache] clear-on-start requested, cache does not exist: {cache_dir}", status_callback=status_callback)
        return
    removed = 0
    for path in sorted(cache_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
            removed += 1
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # Ignore non-empty dirs; later passes may remove their children.
                pass
    _emit(f"[cache] clear-on-start removed {removed} file(s) from {cache_dir}", status_callback=status_callback)


def _cleanup_xdmf_cache_keep_latest(cache_dir: Path, keep_last: int = 2, status_callback=None) -> None:
    if not cache_dir.exists():
        _emit(f"[cache] cleanup skipped, cache does not exist: {cache_dir}", status_callback=status_callback)
        return
    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    if len(files) <= keep_last:
        _emit(
            f"[cache] cleanup skipped, file count={len(files)} <= keep_last={keep_last}",
            status_callback=status_callback,
        )
        return

    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    keep_set = set(files_sorted[:keep_last])
    removed = 0
    for path in files_sorted[keep_last:]:
        if path not in keep_set:
            path.unlink()
            removed += 1

    # Prune empty directories after file cleanup.
    for d in sorted([p for p in cache_dir.rglob("*") if p.is_dir()], reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass

    _emit(
        f"[cache] cleanup complete: kept={min(keep_last, len(files_sorted))}, removed={removed}, dir={cache_dir}",
        status_callback=status_callback,
    )


def _generate_mesh_with_gmsh(status_callback=None) -> None:
    geom_script = Path(__file__).resolve().parents[1] / "geometry" / "build_3d_guitar.py"
    cmd = [sys.executable, str(geom_script), "-nopopup"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Gmsh mesh generation failed.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def _convert_msh_to_xdmf_with_meshio(mesh_file: Path, status_callback=None):
    if meshio is None:
        raise RuntimeError("meshio is required for strict Generate-Convert-Load flow.")
    if not mesh_file.exists():
        raise RuntimeError(f"Expected generated .msh not found: {mesh_file}")

    _emit(f"[mesh] converting .msh via meshio: {mesh_file}", status_callback=status_callback)
    msh = meshio.read(str(mesh_file))
    if "gmsh:physical" not in msh.cell_data_dict:
        raise RuntimeError("meshio read succeeded but gmsh:physical cell_data is missing.")

    cell_phys = msh.cell_data_dict["gmsh:physical"]
    tetra_cells = msh.get_cells_type("tetra")
    tri_cells = msh.get_cells_type("triangle")
    tetra_tags = cell_phys.get("tetra")
    tri_tags = cell_phys.get("triangle")
    if tetra_cells is None or len(tetra_cells) == 0 or tetra_tags is None:
        raise RuntimeError("Generated mesh is missing tetra cells/tags.")
    if tri_cells is None or len(tri_cells) == 0 or tri_tags is None:
        raise RuntimeError("Generated mesh is missing triangle cells/tags.")

    xdmf_dir = mesh_file.parent / "_xdmf_cache"
    xdmf_dir.mkdir(parents=True, exist_ok=True)
    vol_xdmf = xdmf_dir / "guitar_3d_volume.xdmf"
    fac_xdmf = xdmf_dir / "guitar_3d_facets.xdmf"

    vol_mesh = meshio.Mesh(
        points=msh.points,
        cells=[("tetra", tetra_cells)],
        cell_data={"name_to_read": [np.asarray(tetra_tags, dtype=np.int32)]},
    )
    fac_mesh = meshio.Mesh(
        points=msh.points,
        cells=[("triangle", tri_cells)],
        cell_data={"name_to_read": [np.asarray(tri_tags, dtype=np.int32)]},
    )
    print("[INFO] Starting meshio conversion (this may take a few minutes for large meshes)...")
    sys.stdout.flush()
    meshio.write(str(vol_xdmf), vol_mesh)
    meshio.write(str(fac_xdmf), fac_mesh)

    if not vol_xdmf.exists() or not fac_xdmf.exists():
        raise RuntimeError(
            f"XDMF conversion failed. Missing files: vol={vol_xdmf.exists()}, fac={fac_xdmf.exists()}"
        )
    print("[diag] Mesh optimized for RAM: 1.5mm wood, 12mm air.")
    sys.stdout.flush()
    return vol_xdmf, fac_xdmf


def _load_mesh_with_fallback(mesh_file: Path, status_callback=None):
    # Strict primary path: Generate -> Convert -> Load (no fallback alternatives).
    vol_xdmf, fac_xdmf = _convert_msh_to_xdmf_with_meshio(mesh_file, status_callback=status_callback)

    with io.XDMFFile(MPI.COMM_WORLD, str(vol_xdmf), "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
        cell_tags = xdmf.read_meshtags(msh, name="Grid")
    with io.XDMFFile(MPI.COMM_WORLD, str(fac_xdmf), "r") as xdmf:
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        facet_tags = xdmf.read_meshtags(msh, name="Grid")
    return msh, cell_tags, facet_tags


def _load_mesh_and_tags(mesh_file: Path, status_callback=None):
    _emit("Step 1/5: Loading mesh and physical tags...", status_callback=status_callback)
    msh, cell_tags, facet_tags = _load_mesh_with_fallback(mesh_file, status_callback=status_callback)
    if cell_tags is None:
        raise RuntimeError("No 3D physical tags detected in mesh. Expected Air_Internal=10.")
    if facet_tags is None:
        raise RuntimeError("No 2D physical tags detected in mesh. Expected Top_Plate/Body_Shell tags.")

    air_cells = np.where(cell_tags.values == AIR_VOLUME_TAG)[0]
    wood_facets = np.where(np.isin(facet_tags.values, np.asarray(WOOD_SURFACE_TAGS, dtype=np.int32)))[0]
    if air_cells.size == 0:
        raise RuntimeError("Air volume tag 10 not found in mesh.")
    if wood_facets.size == 0:
        raise RuntimeError("Wood surface tags (1/3) not found in mesh.")

    _emit(
        f"[diag] mesh loaded: num_air_cells={air_cells.size}, num_wood_facets={wood_facets.size}",
        status_callback=status_callback,
    )
    _emit(
        f"[diag] unique volume tags={np.unique(cell_tags.values).tolist()}, "
        f"unique facet tags={np.unique(facet_tags.values).tolist()}",
        status_callback=status_callback,
    )
    # Explicit per-tag sanity counts requested for fallback validation.
    vol_counts = {1: int(np.sum(cell_tags.values == 1)), 2: int(np.sum(cell_tags.values == 2)),
                  3: int(np.sum(cell_tags.values == 3)), 10: int(np.sum(cell_tags.values == 10))}
    fac_counts = {1: int(np.sum(facet_tags.values == 1)), 2: int(np.sum(facet_tags.values == 2)),
                  3: int(np.sum(facet_tags.values == 3)), 10: int(np.sum(facet_tags.values == 10))}
    _emit(f"[diag] volume tag counts: {vol_counts}", status_callback=status_callback)
    _emit(f"[diag] facet tag counts: {fac_counts}", status_callback=status_callback)
    return msh, cell_tags, facet_tags


def _solver_bool(solver: Dict, key: str, default: bool) -> bool:
    """Parse solver JSON flags robustly (bool / int / string)."""
    if key not in solver:
        return default
    v = solver[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("0", "false", "no", "off", ""):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
    return bool(v)


def _effective_wood_properties(config: Dict) -> Tuple[float, float, float, float]:
    top = config["materials"]["top"]
    back = config["materials"]["back"]
    thickness = float(config.get("geometry", {}).get("thickness", 0.003))

    E_top = float(top.get("E_L", 1.0e9))
    E_back = float(back.get("E_L", 1.0e9))
    nu_top = float(top.get("nu_LT", 0.3))
    nu_back = float(back.get("nu_LT", 0.3))
    rho_top = float(top["density"])
    rho_back = float(back["density"])

    E_eff = 0.5 * (E_top + E_back)
    nu_eff = 0.5 * (nu_top + nu_back)
    rho_eff = 0.5 * (rho_top + rho_back)
    return E_eff, nu_eff, rho_eff, thickness


def _plate_modal_energy_ratios(
    phi: PETSc.Vec,
    M: PETSc.Mat,
    M_top: Optional[PETSc.Mat],
    M_back: Optional[PETSc.Mat],
    work: PETSc.Vec,
) -> Tuple[float, float]:
    """Top (tag 1) and back/body (tag 3) shares of phi^T M phi (shell mass per facet group / |total|)."""
    M.mult(phi, work)
    e_tot = float(np.real(phi.dot(work)))
    e_top = 0.0
    e_back = 0.0
    if M_top is not None:
        M_top.mult(phi, work)
        e_top = float(np.real(phi.dot(work)))
    if M_back is not None:
        M_back.mult(phi, work)
        e_back = float(np.real(phi.dot(work)))
    denom = max(abs(e_tot), 1e-60)
    return e_top / denom, e_back / denom


def _slepc_shift_invert_batch(
    A: PETSc.Mat,
    M: PETSc.Mat,
    solver_cfg: Dict,
    shift_hz: float,
    batch: int,
    diag_shift: float,
    status_callback,
    M_top: Optional[PETSc.Mat] = None,
    M_back: Optional[PETSc.Mat] = None,
    work: Optional[PETSc.Vec] = None,
) -> Tuple[int, List[Tuple[float, np.ndarray, Optional[float], Optional[float]]]]:
    """Run one SLEPc GNHEP shift-invert solve; returns rows (freq_hz, eigenvector, top_ratio|None, back_ratio|None)."""
    target_lambda = (2.0 * math.pi * float(shift_hz)) ** 2
    eps = SLEPc.EPS().create(MPI.COMM_WORLD)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setKrylovSchurRestart(float(solver_cfg.get("krylov_schur_restart", 0.5)))
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(target_lambda)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)

    ksp = st.getKSP()
    pc = ksp.getPC()
    use_iterative = bool(solver_cfg.get("st_use_iterative_fallback", False))
    if use_iterative:
        ksp.setType(str(solver_cfg.get("st_iter_ksp_type", "gmres")))
        pc.setType(str(solver_cfg.get("st_iter_pc_type", "gamg")))
        ksp_rtol = float(solver_cfg.get("st_iter_ksp_rtol", 1e-4))
        ksp_max_it = int(solver_cfg.get("st_iter_ksp_max_it", 1000))
        ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)
        ksp.setConvergenceHistory()
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        if bool(solver_cfg.get("ksp_monitor", False)):
            ksp.setMonitor(
                lambda _ksp, its, rnorm: _emit(f"[ksp] it={its} rnorm={rnorm:.6e}", status_callback=status_callback)
            )
        if pc.getType().lower() == "hypre":
            try:
                pc.setHYPREType(str(solver_cfg.get("st_iter_hypre_type", "boomeramg")))
            except Exception:
                pass
    else:
        ksp.setType(str(solver_cfg.get("st_ksp_type", "preonly")))
        pc.setType(str(solver_cfg.get("st_pc_type", "lu")))
        pc.setFactorSolverType(str(solver_cfg.get("st_factor_solver_type", "mumps")))

    mumps_icntl_14 = int(solver_cfg.get("mat_mumps_icntl_14", 100))
    mumps_icntl_24 = int(solver_cfg.get("mat_mumps_icntl_24", 1))
    mumps_icntl_22 = int(solver_cfg.get("mat_mumps_icntl_22", 1))
    petsc_opts = PETSc.Options()
    petsc_opts["mat_mumps_icntl_14"] = mumps_icntl_14
    petsc_opts["mat_mumps_icntl_24"] = mumps_icntl_24
    petsc_opts["mat_mumps_icntl_22"] = mumps_icntl_22
    mg_levels_ksp_type = str(solver_cfg.get("mg_levels_ksp_type", "chebyshev"))
    mg_levels_pc_type = str(solver_cfg.get("mg_levels_pc_type", "sor"))
    petsc_opts["mg_levels_ksp_type"] = mg_levels_ksp_type
    petsc_opts["mg_levels_pc_type"] = mg_levels_pc_type
    petsc_opts["pc_gamg_threshold"] = float(solver_cfg.get("pc_gamg_threshold", 0.02))
    petsc_opts["pc_gamg_square_graph"] = int(solver_cfg.get("pc_gamg_square_graph", 1))
    petsc_opts["pc_gamg_agg_nsmooths"] = int(solver_cfg.get("pc_gamg_agg_nsmooths", 1))
    petsc_opts["mg_coarse_pc_type"] = str(solver_cfg.get("mg_coarse_pc_type", "jacobi"))
    petsc_opts["pc_factor_shift_type"] = str(solver_cfg.get("pc_factor_shift_type", "nonzero"))
    petsc_opts["pc_factor_shift_amount"] = float(solver_cfg.get("pc_factor_shift_amount", 1e-2))
    petsc_opts["eps_gen_non_hermitian"] = ""
    petsc_opts["bv_orthog_refine"] = str(solver_cfg.get("bv_orthog_refine", "always"))
    petsc_opts["st_ksp_type"] = str(solver_cfg.get("st_iter_ksp_type", "gmres"))
    petsc_opts["st_ksp_norm_type"] = str(solver_cfg.get("st_ksp_norm_type", "unpreconditioned"))

    ncv = int(solver_cfg.get("target_ncv", max(40, 4 * batch)))
    eps.setDimensions(batch, ncv)
    eps.setTolerances(float(solver_cfg.get("eigs_tol", 1e-4)), int(solver_cfg.get("eigs_maxiter", 2000)))

    diag_vec = A.getDiagonal()
    diag_arr = np.real(diag_vec.array)
    if diag_arr.size > 0:
        diag_min = float(np.min(diag_arr))
        diag_max = float(np.max(diag_arr))
    else:
        diag_min = float("nan")
        diag_max = float("nan")
    _emit(
        f"[solver] shift-invert batch center {shift_hz:.2f} Hz (lambda={target_lambda:.6e} s^-2), "
        f"batch={batch}, KSP={ksp.getType()}, PC={pc.getType()}, "
        f"MUMPS(ICNTL14={mumps_icntl_14}, ICNTL24={mumps_icntl_24}, ICNTL22={mumps_icntl_22}), "
        f"diag_shift={diag_shift:.2e}, A_diag_min={diag_min:.6e}, A_diag_max={diag_max:.6e}, "
        f"iterative={use_iterative}",
        status_callback=status_callback,
    )
    st.setShift(target_lambda)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setFromOptions()
    eps.solve()

    its = eps.getIterationNumber()
    nconv = eps.getConverged()
    reason = eps.getConvergedReason()
    _emit(
        f"[solver] EPS sweep @ {shift_hz:.1f} Hz: iterations={its}, converged={nconv}, reason={reason}",
        status_callback=status_callback,
    )

    out: List[Tuple[float, np.ndarray, Optional[float], Optional[float]]] = []
    rvec = A.createVecRight()
    for i in range(min(batch, nconv)):
        eig = eps.getEigenpair(i, rvec)
        eig_r = float(np.real(eig))
        if eig_r <= 1.0e-14:
            continue
        omega = math.sqrt(eig_r)
        f_hz = omega / (2.0 * math.pi)
        rt: Optional[float] = None
        rb: Optional[float] = None
        if work is not None and (M_top is not None or M_back is not None):
            rt, rb = _plate_modal_energy_ratios(rvec, M, M_top, M_back, work)
        out.append((f_hz, rvec.array.copy(), rt, rb))

    eps.destroy()
    return nconv, out


def _solve_structural_only_evp(
    msh: mesh.Mesh,
    facet_tags,
    config: Dict,
    num_modes: int,
    status_callback=None,
) -> Tuple[mesh.Mesh, fem.FunctionSpace, List[float], np.ndarray, int, int]:
    """Structural-only diagnostic EVP (displacement field only, no acoustic coupling)."""
    _emit("[diag] structural-only diagnosis enabled: solving u-field EVP only.", status_callback=status_callback)
    tdim = msh.topology.dim
    fdim = tdim - 1
    n = ufl.FacetNormal(msh)
    P = ufl.Identity(3) - ufl.outer(n, n)

    u_el = element("Lagrange", msh.basix_cell(), 1, shape=(3,))
    V_u = fem.functionspace(msh, u_el)
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)

    xdmf_ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    structural_tags = (WOOD_SURFACE_TAGS[0], 2, WOOD_SURFACE_TAGS[1])  # include soundhole tag for diag coverage
    E_eff, nu_eff, rho_eff, thickness = _effective_wood_properties(config)
    mu = E_eff / (2.0 * (1.0 + nu_eff))
    lam = E_eff * nu_eff / ((1.0 + nu_eff) * (1.0 - 2.0 * nu_eff))
    D_bend = E_eff * thickness ** 3 / (12.0 * (1.0 - nu_eff ** 2))

    wood_tag_top = int(np.sum(facet_tags.values == structural_tags[0]))
    wood_tag_hole = int(np.sum(facet_tags.values == structural_tags[1]))
    wood_tag_shell = int(np.sum(facet_tags.values == structural_tags[2]))
    if wood_tag_top + wood_tag_hole + wood_tag_shell > 0:
        wood_ds = xdmf_ds(structural_tags[0]) + xdmf_ds(structural_tags[1]) + xdmf_ds(structural_tags[2])
        _emit(
            "[diag] structural-only shell tags in stiffness: "
            f"tag{structural_tags[0]}={wood_tag_top}, tag{structural_tags[1]}={wood_tag_hole}, "
            f"tag{structural_tags[2]}={wood_tag_shell}",
            status_callback=status_callback,
        )
    else:
        wood_ds = ufl.ds(domain=msh)
        _emit(
            "[diag][warn] structural-only run: wood facet tags missing; using full exterior ds.",
            status_callback=status_callback,
            level="warning",
        )

    def eps_surface(uu):
        grad_u = ufl.grad(uu)
        grad_tan = P * grad_u * P
        return 0.5 * (grad_tan + ufl.transpose(grad_tan))

    eps_u = eps_surface(u)
    eps_v = eps_surface(v)
    w_n = ufl.dot(u, n)
    v_n = ufl.dot(v, n)

    a_uu = (
        thickness * (2.0 * mu * ufl.inner(eps_u, eps_v) + lam * ufl.tr(eps_u) * ufl.tr(eps_v))
        + D_bend * ufl.inner(P * ufl.grad(w_n), P * ufl.grad(v_n))
    ) * wood_ds
    m_uu = (rho_eff * thickness) * ufl.dot(u, v) * wood_ds

    # Same structural grounding protocol as coupled run.
    fixed_facets = np.array(facet_tags.find(4), dtype=np.int32)
    if fixed_facets.size == 0:
        fixed_facets = np.array(facet_tags.find(WOOD_SURFACE_TAGS[1]), dtype=np.int32)
    if fixed_facets.size == 0:
        fixed_facets = np.array(facet_tags.find(WOOD_SURFACE_TAGS[0]), dtype=np.int32)
    u_dofs = np.array(fem.locate_dofs_topological(V_u, fdim, fixed_facets), dtype=np.int32)
    if u_dofs.size == 0:
        # Geometric fallback (same spirit as coupled path): pin three boundary points.
        coords = msh.geometry.x
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        tol = max(1.0e-12, 1.0e-8 * max(1.0, diag))
        boundary_mask = (
            np.isclose(coords[:, 0], mins[0], atol=tol)
            | np.isclose(coords[:, 0], maxs[0], atol=tol)
            | np.isclose(coords[:, 1], mins[1], atol=tol)
            | np.isclose(coords[:, 1], maxs[1], atol=tol)
            | np.isclose(coords[:, 2], mins[2], atol=tol)
            | np.isclose(coords[:, 2], maxs[2], atol=tol)
        )
        boundary_ids = np.where(boundary_mask)[0]
        if boundary_ids.size == 0:
            boundary_ids = np.arange(coords.shape[0], dtype=np.int32)
        bcoords = coords[boundary_ids]
        i_min_x = int(np.argpartition(bcoords[:, 0], 0)[0])
        i_max_x = int(np.argpartition(bcoords[:, 0], bcoords.shape[0] - 1)[bcoords.shape[0] - 1])
        i_min_y = int(np.argpartition(bcoords[:, 1], 0)[0])
        anchor_ids = [int(boundary_ids[i_min_x]), int(boundary_ids[i_max_x]), int(boundary_ids[i_min_y])]
        u_dof_blocks = []
        for idx in anchor_ids:
            pt = coords[idx]
            def _u_anchor_marker(x, p=pt):
                return (
                    np.isclose(x[0], p[0], atol=tol)
                    & np.isclose(x[1], p[1], atol=tol)
                    & np.isclose(x[2], p[2], atol=tol)
                )
            u_dof_blocks.append(fem.locate_dofs_geometrical(V_u, _u_anchor_marker))
        if u_dof_blocks:
            u_dofs = np.array(np.unique(np.concatenate(u_dof_blocks)), dtype=np.int32)
    _emit(
        f"[diag] structural-only BCs: fixed_facets={fixed_facets.size}, u_dofs={u_dofs.size}",
        status_callback=status_callback,
    )
    if u_dofs.size == 0:
        raise RuntimeError("Structural-only diagnosis failed: u_dofs is empty even after geometric fallback.")
    bc_u = fem.dirichletbc(np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType), u_dofs, V_u)

    A = assemble_matrix(fem.form(a_uu), bcs=[bc_u])
    A.assemble()
    M = assemble_matrix(fem.form(m_uu), bcs=[bc_u])
    M.assemble()
    # Diagnostic stabilization for singular pivots: A <- A + eps*M
    struct_diag_shift = float(config.get("solver", {}).get("structural_diag_shift", 1.0e-6))
    if struct_diag_shift > 0.0:
        A.axpy(struct_diag_shift, M, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        A.assemble()
        _emit(
            f"[diag] structural-only diagonal shift applied: A += {struct_diag_shift:.2e} * M",
            status_callback=status_callback,
        )

    eps = SLEPc.EPS().create(MPI.COMM_WORLD)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    nev = int(max(1, num_modes))
    eps.setDimensions(nev, int(max(40, 4 * nev)))
    eps.setTolerances(1e-6, 2000)
    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()
    if nconv <= 0:
        raise RuntimeError("Structural-only diagnosis: no converged eigenpairs.")

    rvec = A.createVecRight()
    freqs_hz: List[float] = []
    vectors: List[np.ndarray] = []
    for i in range(min(nev, nconv)):
        eig = eps.getEigenpair(i, rvec)
        eig_r = float(np.real(eig))
        if eig_r <= 1.0e-14:
            continue
        freqs_hz.append(math.sqrt(eig_r) / (2.0 * math.pi))
        vectors.append(rvec.array.copy())
    eps.destroy()

    if not freqs_hz:
        raise RuntimeError("Structural-only diagnosis: no positive eigenvalues.")
    order = np.argsort(np.array(freqs_hz))
    freqs_hz = [freqs_hz[int(i)] for i in order]
    eigvecs = np.stack([vectors[int(i)] for i in order], axis=1)

    print(f"[DIAG] Structural-only first {len(freqs_hz)} mode(s): {[round(f, 3) for f in freqs_hz]}")
    if freqs_hz:
        print(f"[DIAG] Structural-only first mode: {freqs_hz[0]:.3f} Hz")
    sys.stdout.flush()
    n_u = int(V_u.dofmap.index_map.size_local * V_u.dofmap.index_map_bs)
    return msh, V_u, freqs_hz, eigvecs, n_u, 0


def _solve_coupled_evp(
    mesh_file: Path,
    config: Dict,
    num_modes: int,
    status_callback=None,
    solve_evp: bool = True,
):
    _solver_early = config.get("solver", {})
    _ams = _solver_early.get("adaptive_mode_sifter", "<missing>")
    _sihz = _solver_early.get("shift_invert_target_hz", "<missing>")
    print(
        f"[DEBUG] Sifter status: adaptive_mode_sifter={_ams!r} "
        f"(effective={_solver_bool(_solver_early, 'adaptive_mode_sifter', True)}), "
        f"shift_invert_target_hz={_sihz!r}, solve_evp={solve_evp}"
    )
    sys.stdout.flush()

    msh, cell_tags, facet_tags = _load_mesh_and_tags(mesh_file, status_callback=status_callback)
    if solve_evp and _solver_bool(config.get("solver", {}), "structural_only_diagnosis", default=False):
        return _solve_structural_only_evp(
            msh=msh,
            facet_tags=facet_tags,
            config=config,
            num_modes=max(1, int(config.get("solver", {}).get("structural_only_num_modes", 10))),
            status_callback=status_callback,
        )
    coords = msh.geometry.x
    print(
        f"[DIAG] Mesh Bounds - X: {coords[:, 0].min()} to {coords[:, 0].max()}"
    )
    print(
        f"[DIAG] Mesh Bounds - Y: {coords[:, 1].min()} to {coords[:, 1].max()}, "
        f"Z: {coords[:, 2].min()} to {coords[:, 2].max()}"
    )
    sys.stdout.flush()
    gc.collect()
    tdim = msh.topology.dim
    fdim = tdim - 1
    num_cells_global = msh.topology.index_map(tdim).size_global
    _emit(
        f"[diag] topology check: dim={tdim}, num_cells_global={num_cells_global}, "
        f"cell_tags={cell_tags is not None}, facet_tags={facet_tags is not None}",
        status_callback=status_callback,
    )
    if num_cells_global <= 0:
        raise RuntimeError("Mesh topology appears empty (num_cells_global <= 0). Check XDMF read/conversion.")

    _emit("Step 2/5: Building mixed spaces and weak forms...", status_callback=status_callback)
    u_el = element("Lagrange", msh.basix_cell(), 1, shape=(3,))
    p_el = element("Lagrange", msh.basix_cell(), 1)
    W_el = mixed_element([u_el, p_el])
    W = fem.functionspace(msh, W_el)
    n_p_global = int(W.sub(1).dofmap.index_map.size_global * W.sub(1).dofmap.index_map_bs)
    _emit(
        f"[form] pressure field: global P1 DOFs n_p={n_p_global} on full mesh "
        f"(acoustic forms use dx on air cells only, tag={AIR_VOLUME_TAG}).",
        status_callback=status_callback,
    )

    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(z)

    xdmf_ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    # Volume: subdomain_data=cell_tags so xdmf_dx(AIR_VOLUME_TAG) restricts to air cells.
    xdmf_dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    _emit(
        f"[form] measures: dx(domain, subdomain_data=cell_tags) for air tag {AIR_VOLUME_TAG}; "
        f"ds(domain, subdomain_data=facet_tags) for wood shell tags {WOOD_SURFACE_TAGS}.",
        status_callback=status_callback,
    )
    full_dx = ufl.Measure("dx", domain=msh)
    n = ufl.FacetNormal(msh)
    P = ufl.Identity(3) - ufl.outer(n, n)

    E_eff, nu_eff, rho_eff, thickness = _effective_wood_properties(config)
    air_mat = config["materials"]["air"]
    rho_air = float(air_mat["density"])
    c_air = float(air_mat["speed_of_sound"])
    _emit(
        f"[diag] material sanity: E_eff={E_eff:.6e} Pa, rho_eff={rho_eff:.6e} kg/m^3, "
        f"rho_air={rho_air:.6e} kg/m^3, thickness={thickness:.6e} m",
        status_callback=status_callback,
    )

    mu = E_eff / (2.0 * (1.0 + nu_eff))
    lam = E_eff * nu_eff / ((1.0 + nu_eff) * (1.0 - 2.0 * nu_eff))
    D_bend = E_eff * thickness ** 3 / (12.0 * (1.0 - nu_eff ** 2))

    def eps_surface(uu):
        grad_u = ufl.grad(uu)
        grad_tan = P * grad_u * P
        return 0.5 * (grad_tan + ufl.transpose(grad_tan))

    wood_tag_top = int(np.sum(facet_tags.values == WOOD_SURFACE_TAGS[0]))
    wood_tag_shell = int(np.sum(facet_tags.values == WOOD_SURFACE_TAGS[1]))
    if wood_tag_top + wood_tag_shell > 0:
        wood_ds = xdmf_ds(WOOD_SURFACE_TAGS[0]) + xdmf_ds(WOOD_SURFACE_TAGS[1])
        _emit(
            f"[form] structural shell integration on tagged facets: "
            f"tag{WOOD_SURFACE_TAGS[0]}={wood_tag_top}, tag{WOOD_SURFACE_TAGS[1]}={wood_tag_shell}",
            status_callback=status_callback,
        )
    else:
        # Force-physics fallback: if expected structural tags are missing, use all exterior facets.
        wood_ds = ufl.ds(domain=msh)
        _emit(
            "[form][warn] structural facet tags missing; falling back to all exterior facets (ds).",
            status_callback=status_callback,
            level="warning",
        )

    eps_u = eps_surface(u)
    eps_v = eps_surface(v)
    w_n = ufl.dot(u, n)
    v_n = ufl.dot(v, n)

    # Shell-like stiffness on wood manifold:
    # - membrane term: thickness * in-surface elasticity
    # - bending-like term: D * |grad_tan(w_n)|^2
    a_uu = (
        thickness * (2.0 * mu * ufl.inner(eps_u, eps_v) + lam * ufl.tr(eps_u) * ufl.tr(eps_v))
        + D_bend * ufl.inner(P * ufl.grad(w_n), P * ufl.grad(v_n))
    ) * wood_ds

    # Acoustic stiffness in internal air volume.
    a_pp = (1.0 / rho_air) * ufl.inner(ufl.grad(p), ufl.grad(q)) * xdmf_dx(AIR_VOLUME_TAG)

    # Pressure load on structure (stiffness-side coupling).
    a_up = -p * v_n * wood_ds

    # Acoustic mass and structure mass.
    m_uu = (rho_eff * thickness) * ufl.dot(u, v) * wood_ds
    m_pp = (1.0 / (rho_air * c_air * c_air)) * p * q * xdmf_dx(AIR_VOLUME_TAG)

    # Acceleration coupling in acoustic equation:
    # <q, u.n> on interface contributes to generalized mass block.
    m_pu = q * w_n * wood_ds

    # Small diagonal regularization to improve conditioning of the coupled system.
    # This helps avoid NaN/Inf KSP norms near near-null/rigid-body components.
    diag_shift = float(config.get("solver", {}).get("diag_shift", 1.0e3))
    # Global mixed-space regularization so every DOF gets a diagonal anchor.
    reg_u = diag_shift * ufl.dot(u, v) * full_dx
    # Pressure regularization only on interior air (tag 10); avoids smearing acoustic
    # stiffness/mass penalty into wood cells outside the cavity.
    reg_p = diag_shift * p * q * xdmf_dx(AIR_VOLUME_TAG)

    a_form = a_uu + a_pp + a_up + reg_u + reg_p
    m_form = m_uu + m_pp + m_pu

    # Per-facet-group shell mass forms for plate-specific sifter (Top tag 1, Body tag 3).
    m_uu_top_plate = (rho_eff * thickness) * ufl.dot(u, v) * xdmf_ds(WOOD_SURFACE_TAGS[0])
    m_uu_back_shell = (rho_eff * thickness) * ufl.dot(u, v) * xdmf_ds(WOOD_SURFACE_TAGS[1])
    has_top_plate_facets = wood_tag_top > 0
    has_back_shell_facets = wood_tag_shell > 0

    # Lumped masses consistent with m_uu (surface shell) and air volume (tag 10), before EVP solve.
    _wood_mass_note = (
        "Top_Plate+Body_Shell facet tags only"
        if (wood_tag_top + wood_tag_shell) > 0
        else "WARNING: full exterior ds (wood facet tags missing)"
    )
    try:
        mass_air_kg = float(fem.assemble_scalar(fem.form(rho_air * xdmf_dx(AIR_VOLUME_TAG))))
    except Exception as exc:
        mass_air_kg = float("nan")
        _emit(f"[diag] air mass integral failed: {exc}", status_callback=status_callback, level="warning")
    try:
        mass_wood_kg = float(fem.assemble_scalar(fem.form(rho_eff * thickness * wood_ds)))
    except Exception as exc:
        mass_wood_kg = float("nan")
        _emit(f"[diag] wood shell mass integral failed: {exc}", status_callback=status_callback, level="warning")
    print(
        f"[DIAG] Total wood mass (integral rho_eff*thickness over shell ds; {_wood_mass_note}): "
        f"{mass_wood_kg:.6e} kg"
    )
    print(f"[DIAG] Total air mass (integral rho_air over air volume tag {AIR_VOLUME_TAG}): {mass_air_kg:.6e} kg")
    if math.isfinite(mass_air_kg) and math.isfinite(mass_wood_kg) and mass_wood_kg > 0:
        print(f"[DIAG] Air mass / wood mass ratio: {mass_air_kg / mass_wood_kg:.3e}")
    elif math.isfinite(mass_air_kg) and math.isfinite(mass_wood_kg) and mass_wood_kg <= 0:
        print("[DIAG] Air mass / wood mass ratio: undefined (wood mass <= 0)")
    sys.stdout.flush()

    # Release no-longer-needed symbolic temporaries once forms are finalized.
    del eps_u, eps_v, w_n, v_n, wood_tag_top, wood_tag_shell

    # Dirichlet BCs using subspace-collapse strategy for strict C++ signatures.
    soundhole_facets = np.array(facet_tags.find(2), dtype=np.int32)
    pressure_gauge = str(config.get("solver", {}).get("pressure_gauge", "air_interior")).lower()
    # Structural grounding: prioritize explicit wood_fix support (tag=4).
    fixed_facets = np.array(facet_tags.find(4), dtype=np.int32)
    if fixed_facets.size == 0:
        # Fallback to Body_Shell (tag=3) if wood_fix is absent.
        fixed_facets = np.array(facet_tags.find(WOOD_SURFACE_TAGS[1]), dtype=np.int32)
    if fixed_facets.size == 0:
        fixed_facets = np.array(facet_tags.find(WOOD_SURFACE_TAGS[0]), dtype=np.int32)
    bcs = []
    try:
        V_p, _ = W.sub(1).collapse()
        V_u, _ = W.sub(0).collapse()

        u_dofs = fem.locate_dofs_topological(V_u, fdim, fixed_facets)
        u_dofs = np.array(u_dofs, dtype=np.int32)
        n_p_collapsed = int(V_p.dofmap.index_map.size_global * V_p.dofmap.index_map_bs)

        coords = msh.geometry.x
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        tol = max(1.0e-12, 1.0e-8 * max(1.0, diag))
        tol_p = max(1.0e-9, 1.0e-5 * max(1.0, diag))

        # Pressure gauge: default is one dof in air (tag 10) interior so p spans the cavity;
        # optional "soundhole" pins all pressure dofs on soundhole facets (often hundreds).
        p_dofs = np.array([], dtype=np.int32)
        if pressure_gauge == "soundhole":
            p_dofs = np.array(
                fem.locate_dofs_topological(V_p, fdim, soundhole_facets),
                dtype=np.int32,
            )
        else:
            air_cell_idx = cell_tags.find(AIR_VOLUME_TAG)
            if air_cell_idx.size > 0:
                msh.topology.create_connectivity(tdim, 0)
                c_to_v = msh.topology.connectivity(tdim, 0)
                mid_cell = int(air_cell_idx[air_cell_idx.size // 2])
                vidx = c_to_v.links(mid_cell)
                p_anchor = np.mean(coords[vidx], axis=0)

                def _p_air_anchor(x):
                    d = np.linalg.norm(x.T - p_anchor, axis=1)
                    return d < tol_p

                p_dofs = np.array(fem.locate_dofs_geometrical(V_p, _p_air_anchor), dtype=np.int32)
                if p_dofs.size > 1:
                    p_dofs = np.array([int(p_dofs[0])], dtype=np.int32)
                _emit(
                    f"[bc] pressure gauge: single interior air anchor (tag {AIR_VOLUME_TAG}) "
                    f"near {p_anchor.tolist()} (solver.pressure_gauge=air_interior).",
                    status_callback=status_callback,
                )
            if p_dofs.size == 0:
                p_dofs = np.array(
                    fem.locate_dofs_topological(V_p, fdim, soundhole_facets),
                    dtype=np.int32,
                )
                _emit(
                    "[bc][warn] air-interior pressure anchor failed; falling back to soundhole facets.",
                    status_callback=status_callback,
                    level="warning",
                )

        # Acoustic grounding fallback: if pressure dofs still empty, pin one mesh node.
        if p_dofs.size == 0:
            p_anchor = coords[np.argmin(np.linalg.norm(coords - mins, axis=1))]

            def _p_anchor_marker(x):
                return (
                    np.isclose(x[0], p_anchor[0], atol=tol)
                    & np.isclose(x[1], p_anchor[1], atol=tol)
                    & np.isclose(x[2], p_anchor[2], atol=tol)
                )

            p_dofs = np.array(fem.locate_dofs_geometrical(V_p, _p_anchor_marker), dtype=np.int32)
            _emit(
                f"[bc][warn] pressure gauge empty; using corner anchor at {p_anchor.tolist()} "
                f"(count={p_dofs.size})",
                status_callback=status_callback,
                level="warning",
            )

        # Structural grounding fallback: if facet-based displacement dofs are empty,
        # pin three distinct boundary points to suppress all rigid-body modes.
        if u_dofs.size == 0:
            boundary_mask = (
                np.isclose(coords[:, 0], mins[0], atol=tol)
                | np.isclose(coords[:, 0], maxs[0], atol=tol)
                | np.isclose(coords[:, 1], mins[1], atol=tol)
                | np.isclose(coords[:, 1], maxs[1], atol=tol)
                | np.isclose(coords[:, 2], mins[2], atol=tol)
                | np.isclose(coords[:, 2], maxs[2], atol=tol)
            )
            boundary_ids = np.where(boundary_mask)[0]
            if boundary_ids.size == 0:
                boundary_ids = np.arange(coords.shape[0], dtype=np.int32)

            bcoords = coords[boundary_ids]
            i_min_x = int(np.argpartition(bcoords[:, 0], 0)[0])
            i_max_x = int(np.argpartition(bcoords[:, 0], bcoords.shape[0] - 1)[bcoords.shape[0] - 1])
            i_min_y = int(np.argpartition(bcoords[:, 1], 0)[0])
            anchor_ids = [int(boundary_ids[i_min_x]), int(boundary_ids[i_max_x]), int(boundary_ids[i_min_y])]

            # Ensure distinct anchors; if duplicates appear, fill from farthest points.
            unique_anchor_ids = []
            for idx in anchor_ids:
                if idx not in unique_anchor_ids:
                    unique_anchor_ids.append(idx)
            if len(unique_anchor_ids) < 3:
                centroid = np.mean(bcoords, axis=0)
                dist = np.linalg.norm(bcoords - centroid, axis=1)
                far_order = np.argsort(-dist)
                for loc in far_order.tolist():
                    cand = int(boundary_ids[int(loc)])
                    if cand not in unique_anchor_ids:
                        unique_anchor_ids.append(cand)
                    if len(unique_anchor_ids) >= 3:
                        break

            u_anchor_pts = [coords[idx] for idx in unique_anchor_ids[:3]]
            u_dof_blocks = []
            for u_anchor in u_anchor_pts:
                def _u_anchor_marker(x, pt=u_anchor):
                    return (
                        np.isclose(x[0], pt[0], atol=tol)
                        & np.isclose(x[1], pt[1], atol=tol)
                        & np.isclose(x[2], pt[2], atol=tol)
                    )

                u_dof_blocks.append(fem.locate_dofs_geometrical(V_u, _u_anchor_marker))

            if u_dof_blocks:
                u_dofs = np.array(np.unique(np.concatenate(u_dof_blocks)), dtype=np.int32)
            else:
                u_dofs = np.array([], dtype=np.int32)
            _emit(
                f"[bc][warn] facet-based u_dofs empty; using 3-point displacement anchors "
                f"at {[pt.tolist() for pt in u_anchor_pts]} (count={u_dofs.size})",
                status_callback=status_callback,
                level="warning",
            )

        if p_dofs.size == 0:
            raise RuntimeError("Failed to create pressure grounding dofs (p_dofs is empty).")
        if u_dofs.size == 0:
            raise RuntimeError("Failed to create displacement grounding dofs (u_dofs is empty).")

        _emit(
            "[bc][diag] collapsed spaces ready. "
            f"pressure BC dof count={p_dofs.size} (gauge; full pressure FE unknowns=n_p_collapsed={n_p_collapsed}), "
            f"displacement BC u_dofs.shape={u_dofs.shape}, "
            f"soundhole_facets.shape={soundhole_facets.shape}, fixed_facets.shape={fixed_facets.shape}",
            status_callback=status_callback,
        )

        p_zero = fem.Constant(msh, PETSc.ScalarType(0.0))
        bc_p = fem.dirichletbc(p_zero, p_dofs, V_p)

        u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
        bc_u = fem.dirichletbc(u_zero, u_dofs, V_u)
        bcs = [bc_p, bc_u]
    except Exception as e:
        _emit(
            "[bc][error] dirichletbc creation failed. "
            f"p_dofs.dtype={p_dofs.dtype}, p_dofs.shape={p_dofs.shape}, "
            f"u_dofs.dtype={u_dofs.dtype}, u_dofs.shape={u_dofs.shape}, "
            f"soundhole_facets.dtype={soundhole_facets.dtype}, "
            f"soundhole_facets.shape={soundhole_facets.shape}, "
            f"error={e}",
            status_callback=status_callback,
            level="error",
        )
        raise

    A = assemble_matrix(fem.form(a_form), bcs=bcs)
    A.assemble()
    # Debug-only hard anchor to test if row-0 pivot singularity can be bypassed.
    A.setValue(0, 0, PETSc.ScalarType(1.0), addv=PETSc.InsertMode.INSERT_VALUES)
    A.assemble()
    M = assemble_matrix(fem.form(m_form), bcs=bcs)
    M.assemble()

    solver_cfg = config.get("solver", {})
    use_sifter = _solver_bool(solver_cfg, "adaptive_mode_sifter", default=True)
    M_top: Optional[PETSc.Mat] = None
    M_back: Optional[PETSc.Mat] = None
    if solve_evp and use_sifter:
        if has_top_plate_facets:
            M_top = assemble_matrix(fem.form(m_uu_top_plate), bcs=bcs)
            M_top.assemble()
        if has_back_shell_facets:
            M_back = assemble_matrix(fem.form(m_uu_back_shell), bcs=bcs)
            M_back.assemble()

    if not solve_evp:
        return msh, W, A, M

    # Release form objects before eigensolve; matrices are already assembled.
    del a_form, m_form, a_uu, a_pp, a_up, m_uu, m_pp, m_pu, m_uu_top_plate, m_uu_back_shell, reg_u, reg_p
    gc.collect()

    _emit("Step 3/5: Solving generalized EVP with SLEPc...", status_callback=status_callback)
    n_dofs = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)
    n_u_fe = int(W.sub(0).dofmap.index_map.size_global * W.sub(0).dofmap.index_map_bs)
    n_p_fe = int(W.sub(1).dofmap.index_map.size_global * W.sub(1).dofmap.index_map_bs)
    print(
        f"[DIAG] Final u_dofs={n_u_fe} p_dofs={n_p_fe} "
        f"(Dirichlet BC active: u_bc={u_dofs.size} p_bc={p_dofs.size})"
    )
    sys.stdout.flush()
    print(f"Starting solver with {n_dofs} DOFs and proactive memory cleanup...")
    sys.stdout.flush()

    min_valid_hz = float(solver_cfg.get("min_valid_mode_hz", 50.0))
    max_valid_hz = float(solver_cfg.get("max_valid_mode_hz", 1000.0))
    work = M.createVecRight()

    if use_sifter and (M_top is not None or M_back is not None):
        quota = int(solver_cfg.get("sifter_quota", 100))
        batch = int(solver_cfg.get("sifter_batch_modes", 50))
        f_center = float(solver_cfg.get("sifter_start_hz", 100.0))
        f_cap = float(solver_cfg.get("sifter_max_hz", 1000.0))
        df_s = float(solver_cfg.get("sifter_step_hz", 10.0))
        th_top = float(
            solver_cfg.get(
                "sifter_top_plate_energy_ratio",
                solver_cfg.get("sifter_plate_energy_ratio_floor", 0.0005),
            )
        )
        th_back = float(
            solver_cfg.get(
                "sifter_back_plate_energy_ratio",
                solver_cfg.get("sifter_plate_energy_ratio_floor", 0.0005),
            )
        )
        dup_hz = float(solver_cfg.get("sifter_dup_hz", 0.25))

        saved_freqs: List[float] = []
        saved_vecs: List[np.ndarray] = []

        def _dup(freq: float) -> bool:
            return any(abs(freq - fs) < dup_hz for fs in saved_freqs)

        _emit(
            f"[sifter] plate-specific sifter: quota={quota}, batch={batch}, "
            f"tag1>{th_top:g} or tag3>{th_back:g} (energy / |phi^T M phi|), "
            f"sweep {f_center:.0f}–{f_cap:.0f} Hz step {df_s:.0f} Hz.",
            status_callback=status_callback,
        )

        while len(saved_freqs) < quota and f_center <= f_cap + 1e-9:
            nconv, rows = _slepc_shift_invert_batch(
                A,
                M,
                solver_cfg,
                f_center,
                batch,
                diag_shift,
                status_callback,
                M_top=M_top,
                M_back=M_back,
                work=work,
            )
            scored: List[Tuple[float, float, float]] = []
            for _f, _v, rt, rb in rows:
                if rt is None or rb is None:
                    continue
                if not (np.isfinite(rt) and np.isfinite(rb)):
                    continue
                scored.append((max(rt, rb), float(rt), float(rb)))
            scored.sort(key=lambda t: -t[0])
            top5 = scored[:5]
            top_str = ", ".join(f"(tag1={a:.6f},tag3={b:.6f})" for _, a, b in top5) if top5 else ""
            print(f"[DIAG] Batch {f_center:.1f}Hz - Top Ratios: [{top_str}]")
            sys.stdout.flush()

            added = 0
            for f_hz, vec, rt, rb in rows:
                if rt is None or rb is None:
                    continue
                if not (rt > th_top or rb > th_back):
                    continue
                if not (min_valid_hz <= f_hz <= max_valid_hz):
                    continue
                if _dup(f_hz):
                    continue
                saved_freqs.append(float(f_hz))
                saved_vecs.append(vec)
                added += 1
                if len(saved_freqs) >= quota:
                    break
            _emit(
                f"[sifter] center={f_center:.1f} Hz: converged={nconv}, accepted+{added} "
                f"(total saved={len(saved_freqs)}/{quota}).",
                status_callback=status_callback,
            )
            if len(saved_freqs) >= quota:
                break
            f_center += df_s

        if not saved_freqs:
            if M_top is not None:
                M_top.destroy()
            if M_back is not None:
                M_back.destroy()
            raise RuntimeError(
                "Adaptive plate sifter found no modes with "
                f"(tag1_energy/|total| > {th_top:g} OR tag3_energy/|total| > {th_back:g}) "
                f"in [{min_valid_hz:.1f}, {max_valid_hz:.1f}] Hz. "
                "Try lowering sifter_*_plate_energy_ratio or widening the frequency band."
            )

        order = np.argsort(np.array(saved_freqs))
        freqs_hz = [saved_freqs[int(i)] for i in order]
        vectors = [saved_vecs[int(i)] for i in order]
        eigvecs = np.stack(vectors, axis=1)
        if M_top is not None:
            M_top.destroy()
        if M_back is not None:
            M_back.destroy()
        M_top = M_back = None

        print(
            f"[diag] Adaptive sifter: {len(freqs_hz)} significant modes "
            f"(range {min(freqs_hz):.2f}–{max(freqs_hz):.2f} Hz)."
        )
        sys.stdout.flush()
    else:
        if use_sifter:
            _emit(
                "[sifter][warn] adaptive sifter is on but no facet-tagged shell mass matrices "
                "(Top tag 1 / Body tag 3); using legacy single-shift EVP.",
                status_callback=status_callback,
                level="warning",
            )
        shift_target_hz = float(solver_cfg.get("shift_invert_target_hz", 150.0))
        batch_legacy = int(solver_cfg.get("eigenpair_batch_size", 100))
        nconv, rows = _slepc_shift_invert_batch(
            A,
            M,
            solver_cfg,
            shift_target_hz,
            batch_legacy,
            diag_shift,
            status_callback,
            M_top=None,
            M_back=None,
            work=None,
        )
        if nconv <= 0:
            raise RuntimeError("SLEPc did not converge any eigenpairs.")

        freqs_hz = []
        vectors = []
        for f_hz, vec, _rt, _rb in rows:
            freqs_hz.append(f_hz)
            vectors.append(vec)

        if not freqs_hz:
            raise RuntimeError("No positive eigenvalues were found.")

        order = np.argsort(np.array(freqs_hz))
        freqs_hz = [freqs_hz[idx] for idx in order]
        vectors = [vectors[idx] for idx in order]
        eigvecs = np.stack(vectors, axis=1)

        print(
            f"[diag] Solver found {len(freqs_hz)} raw modes. "
            f"Range: {min(freqs_hz):.2f} to {max(freqs_hz):.2f} Hz."
        )
        sys.stdout.flush()

        keep_idx = [i for i, f in enumerate(freqs_hz) if (f >= min_valid_hz and f <= max_valid_hz)]
        if not keep_idx:
            raise RuntimeError(
                f"No modes in [{min_valid_hz:.2f}, {max_valid_hz:.2f}] Hz after filtering."
            )
        freqs_hz = [freqs_hz[i] for i in keep_idx]
        eigvecs = eigvecs[:, keep_idx]

    # Extract split dof counts for output compatibility.
    n_u = W.sub(0).dofmap.index_map.size_local * W.sub(0).dofmap.index_map_bs
    n_p = W.sub(1).dofmap.index_map.size_local * W.sub(1).dofmap.index_map_bs
    return msh, W, freqs_hz, eigvecs, n_u, n_p


def assemble_coupled_operators_for_rom(config: Dict, status_callback=None):
    mesh_file = Path(config["solver"]["mesh_file"])
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    msh, W, A, M = _solve_coupled_evp(
        mesh_file=mesh_file,
        config=config,
        num_modes=1,
        status_callback=status_callback,
        solve_evp=False,
    )
    return msh, W, A, M


def run_fom_for_rom(config: Dict, num_modes: int = 15, status_callback=None):
    mesh_file = Path(config["solver"]["mesh_file"])
    if not mesh_file.exists():
        _emit(f"[mesh] missing .msh, generating new mesh: {mesh_file}", status_callback=status_callback)
    _generate_mesh_with_gmsh(status_callback=status_callback)
    if not mesh_file.exists():
        raise FileNotFoundError(f"Fresh mesh generation did not create expected file: {mesh_file}")
    msh, W, freqs, eigvecs, n_u, n_p = _solve_coupled_evp(
        mesh_file=mesh_file,
        config=config,
        num_modes=num_modes,
        status_callback=status_callback,
    )
    return {
        "mesh": msh,
        "space": W,
        "freqs_hz": freqs,
        "eigvecs": eigvecs,
        "n_u": n_u,
        "n_p": n_p,
    }


def _write_mode_files(
    msh: mesh.Mesh,
    W: fem.FunctionSpace,
    eigvecs: np.ndarray,
    mode_dir: Path,
    status_callback=None,
) -> List[str]:
    _emit("Step 4/5: Writing mode shapes to XDMF...", status_callback=status_callback)
    mode_dir.mkdir(parents=True, exist_ok=True)

    vtk_files: List[str] = []
    export_count = min(eigvecs.shape[1], 10)
    _emit(f"[write] exporting first {export_count} mode(s) as real-valued fields.", status_callback=status_callback)

    # Use collapsed subspaces and explicit real-part extraction to avoid XDMF
    # writer crashes on complex-valued eigenvectors.
    V_u, u_to_W = W.sub(0).collapse()
    V_p, p_to_W = W.sub(1).collapse()
    u_real = fem.Function(V_u)
    p_real = fem.Function(V_p)
    u_real.name = "u"
    p_real.name = "p"

    for i in range(export_count):
        mode_real = np.real(eigvecs[:, i])
        u_real.x.array[:] = mode_real[np.asarray(u_to_W, dtype=np.int32)]
        p_real.x.array[:] = mode_real[np.asarray(p_to_W, dtype=np.int32)]
        u_real.x.scatter_forward()
        p_real.x.scatter_forward()
        file_path = mode_dir / f"mode_{i+1:02d}.xdmf"
        xdmf = io.XDMFFile(msh.comm, str(file_path), "w")
        try:
            xdmf.write_mesh(msh)
            xdmf.write_function(u_real)
            xdmf.write_function(p_real)
        finally:
            xdmf.close()
        vtk_files.append(str(file_path.resolve()))
    return vtk_files


def run_fem_3d_simulation(config_path, status_callback=None):
    _emit(">>> FEM 3D dolfinx entrypoint reached.", status_callback=status_callback)
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    mesh_file = Path(config["solver"]["mesh_file"])
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    cache_dir = mesh_file.parent / "_xdmf_cache"
    if bool(config.get("solver", {}).get("clear_cache_on_start", False)):
        _wipe_cache_folder(cache_dir, status_callback=status_callback)

    num_modes = int(config.get("solver", {}).get("num_modes", 3))
    msh, W, freqs, eigvecs, n_u, n_p = _solve_coupled_evp(
        mesh_file=mesh_file,
        config=config,
        num_modes=num_modes,
        status_callback=status_callback,
    )

    out_dir = config_path.parents[1] / "outputs"
    mode_dir = out_dir / "modes_3d"
    npz_file = mode_dir / "coupled_modes_raw.npz"
    mode_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_file, eigvecs=eigvecs, n_u=n_u, n_p=n_p)
    if int(n_p) > 0:
        vtk_files = _write_mode_files(msh, W, eigvecs, mode_dir, status_callback=status_callback)
    else:
        vtk_files = []
        _emit("[diag] structural-only diagnosis run: skipping mixed-field XDMF mode export.", status_callback=status_callback)

    output_data = {
        "analysis": "acoustic_structural_coupled_eigen",
        "modes_hz": freqs,
        "num_modes": len(freqs),
        "mode_vectors_file": str(npz_file.resolve()),
        "vtk_mode_files": vtk_files,
        "tag_protocol": {
            "Top_Plate": 1,
            "Soundhole": 2,
            "Body_Shell": 3,
            "wood_fix": 4,
            "Air_Internal": 10,
        },
    }

    output_path = out_dir / "fem_3d_output.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    config.setdefault("results", {})
    config["results"]["modes_hz"] = freqs
    config["results"]["mode_vectors_file"] = output_data["mode_vectors_file"]
    config["results"]["vtk_mode_files"] = vtk_files
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # Keep only the latest cache artifacts after a successful run.
    _cleanup_xdmf_cache_keep_latest(cache_dir, keep_last=2, status_callback=status_callback)

    _emit(f"Step 5/5: SUCCESS -> {output_path}", status_callback=status_callback)
    return output_path


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parents[1] / "configs" / "guitar_3d.json"
    run_fem_3d_simulation(str(default_config))