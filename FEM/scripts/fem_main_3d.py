import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


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


def _load_mesh_and_tags(mesh_file: Path, status_callback=None):
    _emit("Step 1/5: Loading Gmsh mesh with dolfinx.gmshio...", status_callback=status_callback)
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(
        str(mesh_file),
        MPI.COMM_WORLD,
        rank=0,
        gdim=3,
    )
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


def _solve_coupled_evp(mesh_file: Path, config: Dict, num_modes: int, status_callback=None):
    msh, cell_tags, facet_tags = _load_mesh_and_tags(mesh_file, status_callback=status_callback)
    tdim = msh.topology.dim
    fdim = tdim - 1

    _emit("Step 2/5: Building mixed spaces and weak forms...", status_callback=status_callback)
    Vu = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1, shape=(3,)))
    Vp = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1))
    W = fem.functionspace(msh, ufl.MixedElement([Vu.ufl_element(), Vp.ufl_element()]))

    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(z)

    xdmf_ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    xdmf_dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
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

    wood_ds = xdmf_ds(WOOD_SURFACE_TAGS[0]) + xdmf_ds(WOOD_SURFACE_TAGS[1])

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

    a_form = a_uu + a_pp + a_up
    m_form = m_uu + m_pp + m_pu

    # Eliminate pressure on soundhole to avoid null-space drift.
    soundhole_facets = facet_tags.find(2)
    p_sub, _ = W.sub(1).collapse()
    p_dofs = fem.locate_dofs_topological((W.sub(1), p_sub), fdim, soundhole_facets)
    bc_p0 = fem.dirichletbc(PETSc.ScalarType(0.0), p_dofs, W.sub(1))
    bcs = [bc_p0]

    A = assemble_matrix(fem.form(a_form), bcs=bcs)
    A.assemble()
    M = assemble_matrix(fem.form(m_form), bcs=bcs)
    M.assemble()

    _emit("Step 3/5: Solving generalized EVP with SLEPc...", status_callback=status_callback)
    eps = SLEPc.EPS().create(MPI.COMM_WORLD)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    solver_cfg = config.get("solver", {})
    target_freq_hz = float(solver_cfg.get("target_freq_hz", 90.0))
    target_lambda = (2.0 * math.pi * target_freq_hz) ** 2

    # Shift-and-invert around the physically relevant low-frequency range.
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eps.setTarget(target_lambda)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)

    # Preconditioner/factorization hints for the shifted linear solves.
    ksp = st.getKSP()
    pc = ksp.getPC()
    ksp.setType(str(solver_cfg.get("st_ksp_type", "preonly")))
    pc.setType(str(solver_cfg.get("st_pc_type", "lu")))
    pc.setFactorSolverType(str(solver_cfg.get("st_factor_solver_type", "mumps")))

    ncv = max(4 * num_modes, 40)
    eps.setDimensions(num_modes, ncv)
    eps.setTolerances(float(solver_cfg.get("eigs_tol", 1e-8)), int(solver_cfg.get("eigs_maxiter", 1000)))
    _emit(
        f"[solver] shift-invert target: {target_freq_hz:.2f} Hz (lambda={target_lambda:.6e}), "
        f"KSP={ksp.getType()}, PC={pc.getType()}",
        status_callback=status_callback,
    )
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    if nconv <= 0:
        raise RuntimeError("SLEPc did not converge any eigenpairs.")

    rvec = A.createVecRight()
    freqs_hz: List[float] = []
    vectors: List[np.ndarray] = []
    for i in range(min(num_modes, nconv)):
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

    # Extract split dof counts for output compatibility.
    n_u = W.sub(0).dofmap.index_map.size_local * W.sub(0).dofmap.index_map_bs
    n_p = W.sub(1).dofmap.index_map.size_local * W.sub(1).dofmap.index_map_bs
    return msh, W, freqs_hz, eigvecs, n_u, n_p


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
    mixed_mode = fem.Function(W)
    u_fun, p_fun = mixed_mode.split()
    u_fun.name = "u"
    p_fun.name = "p"

    for i in range(eigvecs.shape[1]):
        mixed_mode.x.array[:] = eigvecs[:, i]
        mixed_mode.x.scatter_forward()
        file_path = mode_dir / f"mode_{i+1:02d}.xdmf"
        with io.XDMFFile(msh.comm, str(file_path), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(u_fun)
            xdmf.write_function(p_fun)
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

    num_modes = int(config.get("solver", {}).get("num_modes", 10))
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

    _emit(f"Step 5/5: SUCCESS -> {output_path}", status_callback=status_callback)
    return output_path


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parents[1] / "configs" / "guitar_3d.json"
    run_fem_3d_simulation(str(default_config))