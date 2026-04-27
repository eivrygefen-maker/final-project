import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import eigsh

from sfepy.discrete import Equation, Equations, FieldVariable, Integral, Material, Problem
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.terms import Term

try:
    import meshio
except Exception:
    meshio = None

LOGGER = logging.getLogger("fem3d")
if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _emit(message, status_callback=None, level="info"):
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


def _mesh_tag_diagnostics(mesh_file, status_callback=None):
    if meshio is None:
        _emit("[diag] meshio not available, skipping gmsh-tag diagnostics.", status_callback=status_callback)
        return
    try:
        m = meshio.read(str(mesh_file))
        _emit(f"[diag] meshio cells: {[c.type for c in m.cells]}", status_callback=status_callback)

        cell_phys = m.cell_data_dict.get("gmsh:physical", {})
        tet_count = 0
        tag10_tet = 0
        for block in m.cells:
            tags = cell_phys.get(block.type)
            if block.type in ("tetra", "tetra10"):
                tet_count += len(block.data)
                if tags is not None:
                    tag10_tet += int(np.sum(np.asarray(tags) == 10))
        _emit(
            f"[diag] tetra cells={tet_count}, tetra-with-tag10={tag10_tet}",
            status_callback=status_callback,
        )
        if tet_count == 0:
            _emit("[diag][warn] No tetra volume cells found in mesh.", status_callback=status_callback, level="warning")
        if tag10_tet == 0:
            _emit(
                "[diag][warn] No tetra cells with physical tag 10 detected. Acoustic volume may be missing.",
                status_callback=status_callback,
                level="warning",
            )
    except Exception as e:
        _emit(f"[diag][warn] mesh tag diagnostics failed: {e}", status_callback=status_callback, level="warning")


def _build_tag_submesh(mesh_file, out_path, target_tags, allowed_cell_types, status_callback=None):
    """
    Build a filtered submesh containing only selected cell types and physical tags.
    """
    if meshio is None:
        raise RuntimeError("meshio is required to build filtered submeshes.")

    m = meshio.read(str(mesh_file))
    cell_phys = m.cell_data_dict.get("gmsh:physical", {})

    new_cells = []
    new_phys = []
    for block in m.cells:
        if block.type not in allowed_cell_types:
            continue
        tags = cell_phys.get(block.type)
        if tags is None:
            continue
        tags = np.asarray(tags)
        mask = np.isin(tags, np.asarray(list(target_tags)))
        if not np.any(mask):
            continue
        new_cells.append(meshio.CellBlock(block.type, block.data[mask]))
        new_phys.append(tags[mask])

    if not new_cells:
        raise RuntimeError(
            f"No cells found for tags={list(target_tags)} and cell_types={list(allowed_cell_types)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh = meshio.Mesh(
        points=m.points,
        cells=new_cells,
        cell_data={"gmsh:physical": new_phys},
    )
    meshio.write(str(out_path), out_mesh, file_format="gmsh22")
    _emit(
        f"[diag] wrote submesh {out_path.name}: "
        f"blocks={[c.type for c in new_cells]}, total_cells={sum(len(c.data) for c in new_cells)}",
        status_callback=status_callback,
    )
    return out_path


def isotropic_stiffness(E, nu):
    """Build 3D isotropic stiffness tensor (6x6 Voigt)."""
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6), dtype=np.float64)
    C[:3, :3] = lam
    np.fill_diagonal(C[:3, :3], lam + 2.0 * mu)
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


def plane_stress_stiffness(E, nu):
    """Build 2D plane-stress stiffness tensor (3x3 Voigt)."""
    c = E / (1.0 - nu * nu)
    C = np.zeros((3, 3), dtype=np.float64)
    C[0, 0] = c
    C[1, 1] = c
    C[0, 1] = c * nu
    C[1, 0] = c * nu
    C[2, 2] = c * (1.0 - nu) * 0.5
    return C


def _assemble_matrix(problem, equation):
    mtx = problem.equations.create_matrix_graph()
    out = equation.evaluate(mode="weak", dw_mode="matrix", asm_obj=mtx)
    if isinstance(out, tuple):
        return out[0]
    return mtx


def _pick_region(domain, name, expr, kind):
    return domain.create_region(name, expr, kind)


def _matrix_diagnostics(name, mtx, status_callback=None):
    mtx = mtx.tocsr()
    if mtx.nnz > 0:
        data_min = float(mtx.data.min())
        data_max = float(mtx.data.max())
    else:
        data_min = 0.0
        data_max = 0.0

    zero_rows = int(np.sum(np.diff(mtx.indptr) == 0))
    mtx_csc = mtx.tocsc()
    zero_cols = int(np.sum(np.diff(mtx_csc.indptr) == 0))

    _emit(
        f"[diag] {name}: shape={mtx.shape}, nnz={mtx.nnz}, "
        f"min={data_min:.6e}, max={data_max:.6e}, "
        f"zero_rows={zero_rows}, zero_cols={zero_cols}",
        status_callback=status_callback,
    )


def _build_acoustic_only_matrices(mesh_file, config, status_callback=None):
    """
    Debug mode: solve only acoustic pressure on Tag 10 tetra volume.
    """
    _emit("[debug] Acoustic-only mode enabled: structural field disabled.", status_callback=status_callback, level="warning")
    tmp_dir = Path(config.get("solver", {}).get("submesh_dir", "FEM/outputs/submeshes"))
    air_submesh_path = _build_tag_submesh(
        mesh_file=mesh_file,
        out_path=tmp_dir / "air_tag10_only.msh",
        target_tags={10},
        allowed_cell_types={"tetra", "tetra10"},
        status_callback=status_callback,
    )

    mesh_air = Mesh.from_file(str(air_submesh_path))
    domain_air = FEDomain("air_domain", mesh_air, dim=3) if "dim" in FEDomain.__init__.__code__.co_varnames else FEDomain("air_domain", mesh_air)
    air = _pick_region(domain_air, "Air", "all", "cell")

    # Explicit field setup as requested.
    fp = Field.from_args("fp", np.float64, 1, air, approx_order=1, space="H1")
    p = FieldVariable("p", "unknown", fp)
    q = FieldVariable("q", "test", fp, primary_var_name="p")

    air_mat = config["materials"]["air"]
    c0 = float(air_mat["speed_of_sound"])
    rho_air = float(air_mat["density"])
    m_a = Material(
        "m_a",
        inv_rho=np.array([[[1.0 / rho_air]]], dtype=np.float64),
        inv_rho_c2=np.array([[[1.0 / (rho_air * c0 * c0)]]], dtype=np.float64),
    )
    integ_v = Integral("ivol", order=2)
    eq_kp = Equation("Kpp", Term.new("dw_laplace(m_a.inv_rho, q, p)", integ_v, air, m_a=m_a, q=q, p=p))
    eq_mp = Equation("Mpp", Term.new("dw_volume_dot(m_a.inv_rho_c2, q, p)", integ_v, air, m_a=m_a, q=q, p=p))

    pb_p = Problem("acous_only", equations=Equations([eq_kp, eq_mp]))
    pb_p.time_update()
    pb_p.update_materials()
    pb_p.get_variables().init_state()

    _emit("[assemble] Before acoustic-only stiffness assembly (Kpp).", status_callback=status_callback)
    Kpp = _assemble_matrix(pb_p, eq_kp).tocsr()
    _emit("[assemble] After acoustic-only stiffness assembly (Kpp).", status_callback=status_callback)
    _emit("[assemble] Before acoustic-only mass assembly (Mpp).", status_callback=status_callback)
    Mpp = _assemble_matrix(pb_p, eq_mp).tocsr()
    _emit("[assemble] After acoustic-only mass assembly (Mpp).", status_callback=status_callback)

    _matrix_diagnostics("Kpp(acoustic-only)", Kpp, status_callback=status_callback)
    _matrix_diagnostics("Mpp(acoustic-only)", Mpp, status_callback=status_callback)
    return Kpp, Mpp, mesh_air, 0, Kpp.shape[0]


def build_coupled_matrices(mesh_file, config, status_callback=None):
    """
    Build block matrices for coupled acoustic-structural modal analysis:
      [Kuu  Kup][u] = w^2 [Muu   0][u]
      [Kpu  Kpp][p]       [ 0   Mpp][p]
    """
    _emit("Loading mesh into SfePy...", status_callback=status_callback)
    _mesh_tag_diagnostics(mesh_file, status_callback=status_callback)
    if bool(config.get("solver", {}).get("acoustic_only", False)):
        return _build_acoustic_only_matrices(mesh_file, config, status_callback=status_callback)

    # Build separate internal submeshes to avoid mixed-cell connectivity hangs.
    tmp_dir = Path(config.get("solver", {}).get("submesh_dir", "FEM/outputs/submeshes"))
    wood_submesh_path = _build_tag_submesh(
        mesh_file=mesh_file,
        out_path=tmp_dir / "wood_tag1_3_only.msh",
        target_tags={1, 3},
        allowed_cell_types={"triangle", "triangle6"},
        status_callback=status_callback,
    )
    air_submesh_path = _build_tag_submesh(
        mesh_file=mesh_file,
        out_path=tmp_dir / "air_tag10_only.msh",
        target_tags={10},
        allowed_cell_types={"tetra", "tetra10"},
        status_callback=status_callback,
    )

    mesh_wood = Mesh.from_file(str(wood_submesh_path))
    mesh_air = Mesh.from_file(str(air_submesh_path))
    _emit(f"[diag] wood mesh descs={mesh_wood.descs}, n_nod={mesh_wood.n_nod}", status_callback=status_callback)
    _emit(f"[diag] air mesh descs={mesh_air.descs}, n_nod={mesh_air.n_nod}", status_callback=status_callback)

    # Independent domains by dimension.
    try:
        domain_wood = FEDomain("wood_domain", mesh_wood, dim=2)
    except TypeError:
        domain_wood = FEDomain("wood_domain", mesh_wood)
    try:
        domain_air = FEDomain("air_domain", mesh_air, dim=3)
    except TypeError:
        domain_air = FEDomain("air_domain", mesh_air)

    wood_surf = _pick_region(domain_wood, "WoodSurf", "all", "cell")
    air = _pick_region(domain_air, "Air", "all", "cell")
    _emit(
        f"[diag] region sizes: WoodSurf vertices={wood_surf.vertices.shape[0]}, "
        f"Air vertices={air.vertices.shape[0]}",
        status_callback=status_callback,
    )
    if air.vertices.shape[0] == 0:
        raise RuntimeError("Acoustic region Tag 10 is empty. No volume domain to assemble.")

    # Structural field: scalar normal-displacement membrane on 2D wood manifold.
    fu = Field.from_args("fu", np.float64, 1, wood_surf, approx_order=1, space="H1")
    # Acoustic field: explicit 3D volume (tag 10 only).
    fp = Field.from_args("fp", np.float64, 1, air, approx_order=1, space="H1")

    u = FieldVariable("u", "unknown", fu)
    v = FieldVariable("v", "test", fu, primary_var_name="u")
    p = FieldVariable("p", "unknown", fp)
    q = FieldVariable("q", "test", fp, primary_var_name="p")

    top = config["materials"]["top"]
    back = config["materials"]["back"]
    air_mat = config["materials"]["air"]

    # Effective structural properties:
    # blend top/back by area participation for the shell representation.
    E_top = float(top.get("E_L", 1.0e9))
    E_back = float(back.get("E_L", 1.0e9))
    nu_top = float(top.get("nu_LT", 0.3))
    nu_back = float(back.get("nu_LT", 0.3))
    rho_top = float(top["density"])
    rho_back = float(back["density"])

    E_eff = 0.5 * (E_top + E_back)
    nu_eff = 0.5 * (nu_top + nu_back)
    rho_eff = 0.5 * (rho_top + rho_back)
    thick = float(config.get("geometry", {}).get("thickness", 0.003))

    c0 = float(air_mat["speed_of_sound"])
    rho_air = float(air_mat["density"])

    # Scalar membrane properties (thickness-weighted).
    k_val = max(E_eff * thick, 1e-12)
    rho_val = max(rho_eff * thick, 1e-12)

    m_s = Material(
        "m_s",
        k_val=np.array([[[k_val]]], dtype=np.float64),
        rho_val=np.array([[[rho_val]]], dtype=np.float64),
    )
    m_a = Material(
        "m_a",
        inv_rho=np.array([[[1.0 / rho_air]]], dtype=np.float64),
        inv_rho_c2=np.array([[[1.0 / (rho_air * c0 * c0)]]], dtype=np.float64),
    )
    integ_s = Integral("isurf", order=2)  # surface/manifold terms
    integ_v = Integral("ivol", order=2)   # volume terms

    # Derivative-free manifold spring model (Winkler-like) for maximum stability.
    eq_ku = Equation("Kuu", Term.new("dw_volume_dot(m_s.k_val, v, u)", integ_s, wood_surf, m_s=m_s, v=v, u=u))
    eq_mu = Equation("Muu", Term.new("dw_volume_dot(m_s.rho_val, v, u)", integ_s, wood_surf, m_s=m_s, v=v, u=u))
    eq_kp = Equation("Kpp", Term.new("dw_laplace(m_a.inv_rho, q, p)", integ_v, air, m_a=m_a, q=q, p=p))
    eq_mp = Equation("Mpp", Term.new("dw_volume_dot(m_a.inv_rho_c2, q, p)", integ_v, air, m_a=m_a, q=q, p=p))

    # Separate problems for structural manifold and 3D acoustic assembly.
    pb_u = Problem("struct", equations=Equations([eq_ku, eq_mu]))
    pb_p = Problem("acous", equations=Equations([eq_kp, eq_mp]))

    for pb in (pb_u, pb_p):
        pb.time_update()
        pb.update_materials()
        vars_ = pb.get_variables()
        vars_.init_state()

    _emit("Step 1/4: Assembling structural and acoustic matrices...", status_callback=status_callback)
    _emit("[assemble] Before structural stiffness assembly (Kuu).", status_callback=status_callback)
    Kuu = _assemble_matrix(pb_u, eq_ku).tocsr()
    _emit("[assemble] After structural stiffness assembly (Kuu).", status_callback=status_callback)
    _emit("[assemble] Before structural mass assembly (Muu).", status_callback=status_callback)
    Muu = _assemble_matrix(pb_u, eq_mu).tocsr()
    _emit("[assemble] After structural mass assembly (Muu).", status_callback=status_callback)

    _emit("[assemble] Before acoustic stiffness assembly (Kpp).", status_callback=status_callback)
    Kpp = _assemble_matrix(pb_p, eq_kp).tocsr()
    _emit("[assemble] After acoustic stiffness assembly (Kpp).", status_callback=status_callback)
    _emit("[assemble] Before acoustic mass assembly (Mpp).", status_callback=status_callback)
    Mpp = _assemble_matrix(pb_p, eq_mp).tocsr()
    _emit("[assemble] After acoustic mass assembly (Mpp).", status_callback=status_callback)

    # Interface coupling blocks (lightweight consistent coupling).
    # The shared-node mesh from fragment enforces spatial compatibility.
    n_u = Kuu.shape[0]
    n_p = Kpp.shape[0]
    alpha = float(config.get("solver", {}).get("coupling_alpha", 1.0))
    beta = float(config.get("solver", {}).get("coupling_beta", 1.0))

    # Direct scalar-DOF coupling between membrane displacement and pressure.
    k = min(n_u, n_p)
    if k == 0:
        raise RuntimeError("Coupling map is empty: no structural or pressure DOFs.")

    rows_up = np.arange(k, dtype=np.int32)
    cols_up = np.arange(k, dtype=np.int32)
    data_up = np.full(k, alpha, dtype=np.float64)
    Kup = csr_matrix((data_up, (rows_up, cols_up)), shape=(n_u, n_p))

    rows_pu = np.arange(k, dtype=np.int32)
    cols_pu = np.arange(k, dtype=np.int32)
    data_pu = np.full(k, beta, dtype=np.float64)
    Kpu = csr_matrix((data_pu, (rows_pu, cols_pu)), shape=(n_p, n_u))

    K = bmat([[Kuu, Kup], [Kpu, Kpp]], format="csr")
    M = bmat([[Muu, None], [None, Mpp]], format="csr")

    # Diagnostics for conditioning / singularity debugging.
    _emit("Step 2/4: Running matrix diagnostics...", status_callback=status_callback)
    _matrix_diagnostics("Kuu", Kuu, status_callback=status_callback)
    _matrix_diagnostics("Muu", Muu, status_callback=status_callback)
    _matrix_diagnostics("Kpp", Kpp, status_callback=status_callback)
    _matrix_diagnostics("Mpp", Mpp, status_callback=status_callback)
    _matrix_diagnostics("K (coupled)", K, status_callback=status_callback)
    _matrix_diagnostics("M (coupled)", M, status_callback=status_callback)

    # Material unit sanity prints (SI expected: Pa, kg/m^3).
    _emit(
        "[diag] material units check: "
        f"E_top={E_top:.6e} Pa, E_back={E_back:.6e} Pa, "
        f"rho_top={rho_top:.3f} kg/m^3, rho_back={rho_back:.3f} kg/m^3, "
        f"rho_air={rho_air:.6f} kg/m^3, c0={c0:.3f} m/s",
        status_callback=status_callback,
    )
    if (E_top < 1e6 or E_back < 1e6 or E_top > 1e12 or E_back > 1e12):
        _emit(
            "[diag][warn] Young's modulus seems out of typical SI wood range (1e6..1e12 Pa).",
            status_callback=status_callback,
            level="warning",
        )
    if (rho_top < 50 or rho_back < 50 or rho_top > 5000 or rho_back > 5000):
        _emit(
            "[diag][warn] Wood density seems out of typical SI range (50..5000 kg/m^3).",
            status_callback=status_callback,
            level="warning",
        )

    return K, M, mesh_air, n_u, n_p


def export_mode_shapes(base_mesh_file, out_dir, eigvecs, n_u, n_p):
    """
    Export coupled modal vectors to files readable by PyVista:
    - .npz for raw modal data
    - .vtk (if meshio is available) with per-mode scalar magnitude fields.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "coupled_modes_raw.npz", eigvecs=eigvecs, n_u=n_u, n_p=n_p)

    if meshio is None:
        return []

    m = meshio.read(str(base_mesh_file))
    written = []
    for i in range(eigvecs.shape[1]):
        vec = eigvecs[:, i]
        u_mode = vec[:n_u]
        p_mode = vec[n_u:n_u + n_p]
        # Compact visualization data: modal amplitudes in two channels.
        # (SfePy dof -> mesh points mapping can vary; we publish normalized global indicators.)
        u_amp = np.full((m.points.shape[0],), float(np.linalg.norm(u_mode)), dtype=np.float64)
        p_amp = np.full((m.points.shape[0],), float(np.linalg.norm(p_mode)), dtype=np.float64)
        out_file = out_dir / f"mode_{i+1:02d}.vtk"
        meshio.write_points_cells(
            str(out_file),
            points=m.points,
            cells=m.cells,
            point_data={
                "u_mode_amp": u_amp,
                "p_mode_amp": p_amp,
            },
            cell_data=m.cell_data if m.cell_data else None,
        )
        written.append(str(out_file))

    return written


def solve_3d_coupled_eigenmodes(mesh_file, config, num_modes=10, status_callback=None):
    K, M, mesh, n_u, n_p = build_coupled_matrices(mesh_file, config, status_callback=status_callback)

    n = K.shape[0]
    req_modes = max(1, min(int(num_modes), max(1, n - 2)))

    sigma_shift = float(config.get("solver", {}).get("eigs_sigma", 1e-2))
    eig_tol = float(config.get("solver", {}).get("eigs_tol", 1e-6))
    eig_maxiter = int(config.get("solver", {}).get("eigs_maxiter", 2000))
    _emit(
        "Step 3/4: Solving eigenvalue problem...",
        status_callback=status_callback,
    )
    _emit(
        f"[solver] Starting eigsh: k={req_modes}, sigma={sigma_shift:.6e}, "
        f"tol={eig_tol:.2e}, maxiter={eig_maxiter}"
        ,
        status_callback=status_callback,
    )
    vals, vecs = eigsh(K, k=req_modes, M=M, sigma=sigma_shift, which="LM", tol=eig_tol, maxiter=eig_maxiter)
    _emit("[solver] eigsh finished.", status_callback=status_callback)
    vals = np.real(vals)
    vecs = np.real(vecs)

    pos = vals > 1e-12
    vals = vals[pos]
    vecs = vecs[:, pos]

    freqs = np.sqrt(vals) / (2.0 * np.pi)
    order = np.argsort(freqs)
    freqs = freqs[order]
    vecs = vecs[:, order]

    return freqs.tolist(), vecs, n_u, n_p, bool(config.get("solver", {}).get("acoustic_only", False))


def run_fem_3d_simulation(config_path, status_callback=None):
    _emit(">>> FEM 3D entrypoint reached (run_fem_3d_simulation).", status_callback=status_callback)
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    mesh_file = Path(config["solver"]["mesh_file"])
    num_modes = int(config.get("solver", {}).get("num_modes", 10))

    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    freqs, eigvecs, n_u, n_p, acoustic_only = solve_3d_coupled_eigenmodes(
        str(mesh_file), config, num_modes=num_modes, status_callback=status_callback
    )

    out_dir = config_path.parents[1] / "outputs"
    mode_dir = out_dir / "modes_3d"
    _emit("Step 4/4: Exporting mode shapes and writing outputs...", status_callback=status_callback)
    vtk_files = export_mode_shapes(mesh_file, mode_dir, eigvecs, n_u, n_p)

    output_data = {
        "analysis": "acoustic_only_eigen" if acoustic_only else "acoustic_structural_coupled_eigen",
        "modes_hz": freqs,
        "num_modes": len(freqs),
        "mode_vectors_file": str((mode_dir / "coupled_modes_raw.npz").resolve()),
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
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    # Mirror key solver output into main config for App/Solver synchronization.
    config.setdefault("results", {})
    config["results"]["modes_hz"] = freqs
    config["results"]["mode_vectors_file"] = output_data["mode_vectors_file"]
    config["results"]["vtk_mode_files"] = vtk_files
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    _emit(f"SUCCESS: Coupled 3D frequencies saved to {output_path}", status_callback=status_callback)
    return output_path


if __name__ == "__main__":
    test_config = Path(__file__).resolve().parents[1] / "configs" / "guitar_3d.json"
    run_fem_3d_simulation(str(test_config))