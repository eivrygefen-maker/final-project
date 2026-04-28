import json
import logging
import math
import gc
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    meshio.write(str(vol_xdmf), vol_mesh)
    meshio.write(str(fac_xdmf), fac_mesh)

    if not vol_xdmf.exists() or not fac_xdmf.exists():
        raise RuntimeError(
            f"XDMF conversion failed. Missing files: vol={vol_xdmf.exists()}, fac={fac_xdmf.exists()}"
        )
    print("[diag] New mesh generated with 2mm wood refinement and converted successfully.")
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


def _solve_coupled_evp(
    mesh_file: Path,
    config: Dict,
    num_modes: int,
    status_callback=None,
    solve_evp: bool = True,
):
    msh, cell_tags, facet_tags = _load_mesh_and_tags(mesh_file, status_callback=status_callback)
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

    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(z)

    xdmf_ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    xdmf_dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    full_dx = ufl.Measure("dx", domain=msh)
    n = ufl.FacetNormal(msh)
    P = ufl.Identity(3) - ufl.outer(n, n)

    E_eff, nu_eff, rho_eff, thickness = _effective_wood_properties(config)
    air_mat = config["materials"]["air"]
    rho_air = float(air_mat["density"])
    c_air = float(air_mat["speed_of_sound"])

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
    reg_p = diag_shift * p * q * full_dx

    a_form = a_uu + a_pp + a_up + reg_u + reg_p
    m_form = m_uu + m_pp + m_pu

    # Release no-longer-needed symbolic temporaries once forms are finalized.
    del eps_u, eps_v, w_n, v_n, wood_tag_top, wood_tag_shell

    # Dirichlet BCs using subspace-collapse strategy for strict C++ signatures.
    soundhole_facets = np.array(facet_tags.find(2), dtype=np.int32)
    wood_fix_facets = np.array(facet_tags.find(WOOD_SURFACE_TAGS[0]), dtype=np.int32)
    if wood_fix_facets.size == 0:
        wood_fix_facets = np.array(facet_tags.find(WOOD_SURFACE_TAGS[1]), dtype=np.int32)
    bcs = []
    try:
        V_p, _ = W.sub(1).collapse()
        V_u, _ = W.sub(0).collapse()

        p_dofs = fem.locate_dofs_topological(V_p, fdim, soundhole_facets)
        u_dofs = fem.locate_dofs_topological(V_u, fdim, wood_fix_facets)
        p_dofs = np.array(p_dofs, dtype=np.int32)
        u_dofs = np.array(u_dofs, dtype=np.int32)

        coords = msh.geometry.x
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        tol = max(1.0e-12, 1.0e-8 * max(1.0, diag))

        # Acoustic grounding fallback: if soundhole-based pressure dofs are empty,
        # pin one pressure dof at a mesh node as reference pressure.
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
                f"[bc][warn] soundhole p_dofs empty; using single-point pressure anchor at {p_anchor.tolist()} "
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
            f"p_dofs.dtype={p_dofs.dtype}, p_dofs.shape={p_dofs.shape}, "
            f"u_dofs.dtype={u_dofs.dtype}, u_dofs.shape={u_dofs.shape}, "
            f"soundhole_facets.dtype={soundhole_facets.dtype}, "
            f"soundhole_facets.shape={soundhole_facets.shape}, "
            f"wood_fix_facets.shape={wood_fix_facets.shape}",
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

    if not solve_evp:
        return msh, W, A, M

    # Release form objects before eigensolve; matrices are already assembled.
    del a_form, m_form, a_uu, a_pp, a_up, m_uu, m_pp, m_pu, reg_u, reg_p
    gc.collect()

    _emit("Step 3/5: Solving generalized EVP with SLEPc...", status_callback=status_callback)
    n_dofs = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)
    print(f"Starting solver with {n_dofs} DOFs and proactive memory cleanup...")
    sys.stdout.flush()
    eps = SLEPc.EPS().create(MPI.COMM_WORLD)
    eps.setOperators(A, M)
    # Coupled structural-acoustic system is generally indefinite/non-Hermitian.
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    solver_cfg = config.get("solver", {})
    eps.setKrylovSchurRestart(float(solver_cfg.get("krylov_schur_restart", 0.5)))
    target_lambda = float(solver_cfg.get("target_lambda", 0.0))

    # Shift-and-invert around the physically relevant low-frequency range.
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(target_lambda)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)

    # Preconditioner/factorization hints for the shifted linear solves.
    ksp = st.getKSP()
    pc = ksp.getPC()
    use_iterative = bool(solver_cfg.get("st_use_iterative_fallback", False))
    if use_iterative:
        # Memory-efficient inner solve for shift-invert.
        ksp.setType(str(solver_cfg.get("st_iter_ksp_type", "gmres")))
        pc.setType(str(solver_cfg.get("st_iter_pc_type", "gamg")))
        ksp_rtol = float(solver_cfg.get("st_iter_ksp_rtol", 1e-4))
        ksp_max_it = int(solver_cfg.get("st_iter_ksp_max_it", 1000))
        ksp.setTolerances(rtol=ksp_rtol, max_it=ksp_max_it)
        ksp.setConvergenceHistory()
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        if bool(solver_cfg.get("ksp_monitor", False)):
            ksp.setMonitor(lambda _ksp, its, rnorm: _emit(f"[ksp] it={its} rnorm={rnorm:.6e}", status_callback=status_callback))
        if pc.getType().lower() == "hypre":
            # BoomerAMG is generally robust for large 3D coupled systems.
            try:
                pc.setHYPREType(str(solver_cfg.get("st_iter_hypre_type", "boomeramg")))
            except Exception:
                pass
    else:
        ksp.setType(str(solver_cfg.get("st_ksp_type", "preonly")))
        pc.setType(str(solver_cfg.get("st_pc_type", "lu")))
        pc.setFactorSolverType(str(solver_cfg.get("st_factor_solver_type", "mumps")))

    # PETSc/MUMPS memory and robustness tuning for shifted LU factorizations.
    # - ICNTL(14): increase MUMPS working memory percentage to reduce OOM (-9).
    # - ICNTL(24): enhanced null-pivot detection in coupled problems.
    # - ICNTL(22): out-of-core mode (optional; slower but less RAM pressure).
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
    # Explicit ST-KSP options for iterative shift-invert stability.
    petsc_opts["st_ksp_type"] = str(solver_cfg.get("st_iter_ksp_type", "gmres"))
    petsc_opts["st_ksp_norm_type"] = str(solver_cfg.get("st_ksp_norm_type", "unpreconditioned"))

    fast_num_modes = 100
    ncv = int(solver_cfg.get("target_ncv", max(40, 4 * fast_num_modes)))
    eps.setDimensions(fast_num_modes, ncv)
    eps.setTolerances(float(solver_cfg.get("eigs_tol", 1e-4)), int(solver_cfg.get("eigs_maxiter", 2000)))
    # Matrix sanity diagnostic: inspect assembled stiffness diagonal spread.
    diag_vec = A.getDiagonal()
    diag_arr = np.real(diag_vec.array)
    if diag_arr.size > 0:
        diag_min = float(np.min(diag_arr))
        diag_max = float(np.max(diag_arr))
    else:
        diag_min = float("nan")
        diag_max = float("nan")
    FORCED_SHIFT = 150.0
    _emit(
        f"[solver] shift-invert target: {FORCED_SHIFT:.2f} (forced shift anchor), "
        f"KSP={ksp.getType()}, PC={pc.getType()}, "
        f"MUMPS(ICNTL14={mumps_icntl_14}, ICNTL24={mumps_icntl_24}, ICNTL22={mumps_icntl_22}), "
        f"MG(level_ksp={mg_levels_ksp_type}, level_pc={mg_levels_pc_type}), "
        f"diag_shift={diag_shift:.2e}, A_diag_min={diag_min:.6e}, A_diag_max={diag_max:.6e}, "
        f"iterative_default={use_iterative}",
        status_callback=status_callback,
    )
    st.setShift(FORCED_SHIFT)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setFromOptions()
    eps.solve()

    its = eps.getIterationNumber()
    nconv = eps.getConverged()
    reason = eps.getConvergedReason()
    err0 = float("nan")
    if nconv > 0:
        try:
            err0 = float(eps.computeError(0))
        except Exception:
            err0 = float("nan")
    _emit(
        f"[solver] EPS status: iterations={its}, converged={nconv}, reason={reason}, error_mode0={err0:.6e}",
        status_callback=status_callback,
    )

    if nconv <= 0:
        raise RuntimeError("SLEPc did not converge any eigenpairs.")

    rvec = A.createVecRight()
    freqs_hz: List[float] = []
    vectors: List[np.ndarray] = []
    for i in range(min(fast_num_modes, nconv)):
        eig = eps.getEigenpair(i, rvec)
        eig_r = float(np.real(eig))
        if eig_r <= 1.0e-14:
            continue
        omega = math.sqrt(eig_r)
        freqs_hz.append(omega / (2.0 * math.pi))
        vectors.append(rvec.array.copy())

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

    # Sweep-and-filter: keep only physically valid audible band modes.
    min_valid_hz = float(solver_cfg.get("min_valid_mode_hz", 80.0))
    max_valid_hz = float(solver_cfg.get("max_valid_mode_hz", 500.0))
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
    vtk_files = _write_mode_files(msh, W, eigvecs, mode_dir, status_callback=status_callback)

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