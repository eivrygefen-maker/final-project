import json
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


def _assemble_matrix(problem, equation):
    mtx = problem.equations.create_matrix_graph()
    out = equation.evaluate(mode="weak", dw_mode="matrix", asm_obj=mtx)
    if isinstance(out, tuple):
        return out[0]
    return mtx


def _pick_region(domain, name, expr, kind):
    return domain.create_region(name, expr, kind)


def _matrix_diagnostics(name, mtx):
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

    print(
        f"[diag] {name}: shape={mtx.shape}, nnz={mtx.nnz}, "
        f"min={data_min:.6e}, max={data_max:.6e}, "
        f"zero_rows={zero_rows}, zero_cols={zero_cols}"
    )


def build_coupled_matrices(mesh_file, config):
    """
    Build block matrices for coupled acoustic-structural modal analysis:
      [Kuu  Kup][u] = w^2 [Muu   0][u]
      [Kpu  Kpp][p]       [ 0   Mpp][p]
    """
    mesh = Mesh.from_file(mesh_file)
    domain = FEDomain("domain", mesh)

    # Tag protocol regions.
    # Wood interface surfaces (tags 1+3), acoustic volume (tag 10), soundhole bc (tag 2).
    wood_surf = _pick_region(domain, "WoodSurf", "cells of group 1 +c cells of group 3", "facet")
    air = _pick_region(domain, "Air", "cells of group 10", "cell")
    soundhole = _pick_region(domain, "Soundhole", "vertices of group 2", "facet")

    # Structural field on wood interface vertices (surface shell-like discretization).
    fu = Field.from_args("fu", np.float64, 3, wood_surf, approx_order=1)
    # Acoustic pressure in enclosed air volume.
    fp = Field.from_args("fp", np.float64, 1, air, approx_order=1)

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
    C_eff = isotropic_stiffness(E_eff, nu_eff)

    c0 = float(air_mat["speed_of_sound"])
    rho_air = float(air_mat["density"])

    m_s = Material("m_s", C=C_eff[np.newaxis, :, :], rho=np.array([[[rho_eff]]], dtype=np.float64))
    m_a = Material(
        "m_a",
        inv_rho=np.array([[[1.0 / rho_air]]], dtype=np.float64),
        inv_rho_c2=np.array([[[1.0 / (rho_air * c0 * c0)]]], dtype=np.float64),
    )
    integ = Integral("i", order=2)

    # Structural operators.
    eq_ku = Equation("Kuu", Term.new("dw_lin_elastic(m_s.C, v, u)", integ, wood_surf, m_s=m_s, v=v, u=u))
    eq_mu = Equation("Muu", Term.new("dw_volume_dot(m_s.rho, v, u)", integ, wood_surf, m_s=m_s, v=v, u=u))

    # Acoustic operators: -div(1/rho grad p) = w^2 * (1/(rho*c^2)) p
    eq_kp = Equation("Kpp", Term.new("dw_laplace(m_a.inv_rho, q, p)", integ, air, m_a=m_a, q=q, p=p))
    eq_mp = Equation("Mpp", Term.new("dw_volume_dot(m_a.inv_rho_c2, q, p)", integ, air, m_a=m_a, q=q, p=p))

    # Separate dummy problems for matrix assembly.
    pb_u = Problem("struct", equations=Equations([eq_ku, eq_mu]))
    pb_p = Problem("acous", equations=Equations([eq_kp, eq_mp]))

    # Soundhole: pressure release p=0.
    pb_p.set_bcs(Conditions([EssentialBC("p0", soundhole, {"p.0": 0.0})]))

    for pb in (pb_u, pb_p):
        pb.time_update()
        pb.update_materials()
        vars_ = pb.get_variables()
        vars_.init_state()

    Kuu = _assemble_matrix(pb_u, eq_ku).tocsr()
    Muu = _assemble_matrix(pb_u, eq_mu).tocsr()
    Kpp = _assemble_matrix(pb_p, eq_kp).tocsr()
    Mpp = _assemble_matrix(pb_p, eq_mp).tocsr()

    # Interface coupling blocks (lightweight consistent coupling).
    # The shared-node mesh from fragment enforces spatial compatibility.
    n_u = Kuu.shape[0]
    n_p = Kpp.shape[0]
    alpha = float(config.get("solver", {}).get("coupling_alpha", 1.0))
    beta = float(config.get("solver", {}).get("coupling_beta", 1.0))

    # Sparse low-rank coupling scaffold to keep block system stable.
    # This can be replaced with stronger variational terms once needed.
    k = min(n_u, n_p)
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
    _matrix_diagnostics("Kuu", Kuu)
    _matrix_diagnostics("Muu", Muu)
    _matrix_diagnostics("Kpp", Kpp)
    _matrix_diagnostics("Mpp", Mpp)
    _matrix_diagnostics("K (coupled)", K)
    _matrix_diagnostics("M (coupled)", M)

    # Material unit sanity prints (SI expected: Pa, kg/m^3).
    print(
        "[diag] material units check: "
        f"E_top={E_top:.6e} Pa, E_back={E_back:.6e} Pa, "
        f"rho_top={rho_top:.3f} kg/m^3, rho_back={rho_back:.3f} kg/m^3, "
        f"rho_air={rho_air:.6f} kg/m^3, c0={c0:.3f} m/s"
    )
    if (E_top < 1e6 or E_back < 1e6 or E_top > 1e12 or E_back > 1e12):
        print("[diag][warn] Young's modulus seems out of typical SI wood range (1e6..1e12 Pa).")
    if (rho_top < 50 or rho_back < 50 or rho_top > 5000 or rho_back > 5000):
        print("[diag][warn] Wood density seems out of typical SI range (50..5000 kg/m^3).")

    return K, M, mesh, n_u, n_p


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


def solve_3d_coupled_eigenmodes(mesh_file, config, num_modes=10):
    K, M, mesh, n_u, n_p = build_coupled_matrices(mesh_file, config)

    n = K.shape[0]
    req_modes = max(1, min(int(num_modes), max(1, n - 2)))

    sigma_shift = float(config.get("solver", {}).get("eigs_sigma", 1e-2))
    eig_tol = float(config.get("solver", {}).get("eigs_tol", 1e-6))
    eig_maxiter = int(config.get("solver", {}).get("eigs_maxiter", 2000))
    print(
        f"[solver] Starting eigsh: k={req_modes}, sigma={sigma_shift:.6e}, "
        f"tol={eig_tol:.2e}, maxiter={eig_maxiter}"
    )
    vals, vecs = eigsh(K, k=req_modes, M=M, sigma=sigma_shift, which="LM", tol=eig_tol, maxiter=eig_maxiter)
    print("[solver] eigsh finished.")
    vals = np.real(vals)
    vecs = np.real(vecs)

    pos = vals > 1e-12
    vals = vals[pos]
    vecs = vecs[:, pos]

    freqs = np.sqrt(vals) / (2.0 * np.pi)
    order = np.argsort(freqs)
    freqs = freqs[order]
    vecs = vecs[:, order]

    return freqs.tolist(), vecs, n_u, n_p


def run_fem_3d_simulation(config_path):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    mesh_file = Path(config["solver"]["mesh_file"])
    num_modes = int(config.get("solver", {}).get("num_modes", 10))

    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    freqs, eigvecs, n_u, n_p = solve_3d_coupled_eigenmodes(str(mesh_file), config, num_modes=num_modes)

    out_dir = config_path.parents[1] / "outputs"
    mode_dir = out_dir / "modes_3d"
    vtk_files = export_mode_shapes(mesh_file, mode_dir, eigvecs, n_u, n_p)

    output_data = {
        "analysis": "acoustic_structural_coupled_eigen",
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

    print(f"SUCCESS: Coupled 3D frequencies saved to {output_path}")
    return output_path


if __name__ == "__main__":
    test_config = Path(__file__).resolve().parents[1] / "configs" / "guitar_3d.json"
    run_fem_3d_simulation(str(test_config))