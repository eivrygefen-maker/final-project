from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Import dataclasses used by fem_api dispatcher.
try:
    from fem_api import RectPlateGeometry2D, OrthotropicPlateMaterial, BoundaryCondition, FemResult
except Exception:  # pragma: no cover
    from .fem_api import RectPlateGeometry2D, OrthotropicPlateMaterial, BoundaryCondition, FemResult  # type: ignore


def _compute_bending_stiffness_from_engineering_constants(
    *,
    E_L: float,
    E_T: float,
    G_LT: float,
    nu_LT: float,
    h: float,
) -> tuple[float, float, float, float]:
    """
    Compute bending stiffness constants D11, D22, D12, D66 for an orthotropic plate
    from engineering constants (classical orthotropic thin-plate relations).
    """
    nu_TL = nu_LT * (E_T / E_L)

    denom = 1.0 - nu_LT * nu_TL
    if abs(denom) < 1e-12:
        raise ValueError("Invalid material constants: 1 - nu_LT*nu_TL is ~0.")

    h3_over_12 = (h**3) / 12.0

    D11 = (E_L * h3_over_12) / denom
    D22 = (E_T * h3_over_12) / denom
    D12 = (nu_TL * E_L * h3_over_12) / denom
    D66 = G_LT * h3_over_12

    return D11, D22, D12, D66

def _d4_clamped(n: int, dx: float) -> sp.csr_matrix:
    """
    1D fourth-derivative matrix on n interior points with CLAMPED BC:
      w=0 at boundaries + dw/dn=0 at boundaries (implemented via ghost nodes).
    Interior points correspond to i=1..n in a full grid 0..n+1.
    """
    if n < 3:
        raise ValueError("Need at least 3 interior points for clamped D4.")

    inv = 1.0 / (dx**4)
    A = sp.lil_matrix((n, n), dtype=float)

    # Helper to add coefficient safely
    def add(r: int, c: int, val: float) -> None:
        if 0 <= c < n:
            A[r, c] += val

    # Row 0 (i=1): (w_-1 -4w0 +6w1 -4w2 +w3)/dx^4, with w0=0, w_-1=w1  => (7w1 -4w2 +w3)/dx^4
    add(0, 0, 7.0)
    add(0, 1, -4.0)
    add(0, 2, 1.0)

    # Row 1 (i=2): (w0 -4w1 +6w2 -4w3 +w4)/dx^4, with w0=0
    if n >= 4:
        add(1, 0, -4.0)
        add(1, 1, 6.0)
        add(1, 2, -4.0)
        add(1, 3, 1.0)
    else:
        # n==3: i=2 is also near the right boundary; handle symmetrically by direct stencil with w0=0 and w4=0
        add(1, 0, -4.0)
        add(1, 1, 6.0)
        add(1, 2, -4.0)

    # Middle rows: standard [1, -4, 6, -4, 1]
    for r in range(2, n - 2):
        add(r, r - 2, 1.0)
        add(r, r - 1, -4.0)
        add(r, r, 6.0)
        add(r, r + 1, -4.0)
        add(r, r + 2, 1.0)

    # Row n-2 (i=n-1): symmetric to i=2 with right boundary w_{n+1}=0
    if n >= 4:
        r = n - 2
        add(r, n - 4, 1.0)
        add(r, n - 3, -4.0)
        add(r, n - 2, 6.0)
        add(r, n - 1, -4.0)

    # Row n-1 (i=n): (w_{n-2} -4w_{n-1} +6w_n -4w_{n+1} +w_{n+2})/dx^4
    # with w_{n+1}=0 and w_{n+2}=w_n  => (w_{n-2} -4w_{n-1} +7w_n)/dx^4
    r = n - 1
    add(r, n - 3, 1.0)
    add(r, n - 2, -4.0)
    add(r, n - 1, 7.0)

    return (A.tocsr()) * inv

def _d2_dirichlet(n: int, dx: float) -> sp.csr_matrix:
    """
    1D second-derivative matrix on n interior points with Dirichlet boundary w=0.
    Central differences: (w_{i-1} - 2w_i + w_{i+1})/dx^2
    """
    main = (-2.0 / (dx * dx)) * np.ones(n)
    off = (1.0 / (dx * dx)) * np.ones(n - 1)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _pickup_index_on_interior_grid(
    *,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    nix: int,
    niy: int,
) -> Tuple[int, Tuple[float, float]]:
    """
    Map pickup point (x0,y0) in meters to nearest interior DOF index for our flattening.
    Flattening convention (matches kron(Iy, D4x)): x varies fastest, then y blocks.
      idx = iy*nix + ix
    """
    i_full = int(round(x0 / dx))
    j_full = int(round(y0 / dy))

    # Clamp to interior nodes in full grid: 1..nx-2, 1..ny-2
    i_full = _clamp(i_full, 1, nx - 2)
    j_full = _clamp(j_full, 1, ny - 2)

    ix = i_full - 1  # 0..nix-1
    iy = j_full - 1  # 0..niy-1

    ix = _clamp(ix, 0, nix - 1)
    iy = _clamp(iy, 0, niy - 1)

    pickup_idx = iy * nix + ix
    pickup_xy_used = (i_full * dx, j_full * dy)
    return pickup_idx, pickup_xy_used


def solve_plate_ortho_kl_2d(
    *,
    geom: RectPlateGeometry2D,
    mat: OrthotropicPlateMaterial,
    bc: BoundaryCondition,
    n_modes: int = 20,
    mesh_size: float = 0.01,
) -> FemResult:
    """
    Numerical eigenfrequency solver for a rectangular orthotropic thin plate.

    New addition:
    - Computes per-mode weights A_n from the eigenvector value at a chosen pickup point.
      These are stored in FemResult.mode_weights so STK can use them later.
    """
    kind = bc.kind.upper()
    if kind not in ("SSSS", "CCCC"):
        raise NotImplementedError(f"Only bc.kind='SSSS' or 'CCCC' supported for now, got: {bc.kind}")

    a, b, h = geom.a, geom.b, geom.h

    # Build grid sizes from mesh_size: use interior nodes only (Dirichlet boundary w=0)
    nx = max(6, int(round(a / mesh_size)) + 1)  # total points incl boundaries
    ny = max(6, int(round(b / mesh_size)) + 1)

    dx = a / (nx - 1)
    dy = b / (ny - 1)

    # interior node counts
    nix = nx - 2
    niy = ny - 2
    if nix < 2 or niy < 2:
        raise ValueError("Mesh too coarse. Increase resolution (smaller mesh_size).")

    # Compute bending stiffness constants internally from material constants + thickness.
    D11, D22, D12, D66 = _compute_bending_stiffness_from_engineering_constants(
        E_L=mat.E_L, E_T=mat.E_T, G_LT=mat.G_LT, nu_LT=mat.nu_LT, h=h
    )
    Dxy = (D12 + 2.0 * D66)

    # Discrete operators on interior grid:
    D2x = _d2_dirichlet(nix, dx)
    D2y = _d2_dirichlet(niy, dy)

    # Fourth derivatives via squaring second derivative matrices
    if kind == "SSSS":
        D4x = (D2x @ D2x).tocsr()
        D4y = (D2y @ D2y).tocsr()
    else:  # CCCC (clamped)
        D4x = _d4_clamped(nix, dx)
        D4y = _d4_clamped(niy, dy)


    Ix = sp.identity(nix, format="csr")
    Iy = sp.identity(niy, format="csr")

    # Build plate operator:
    L = (D11 * sp.kron(Iy, D4x, format="csr") +
         2.0 * Dxy * sp.kron(D2y, D2x, format="csr") +
         D22 * sp.kron(D4y, Ix, format="csr"))

    # Mass term: (rho * h) * w  (lumped as identity for interior DOFs)
    mass_scale = mat.rho * h
    M = mass_scale * sp.identity(L.shape[0], format="csr")

    # Solve generalized eigenproblem
    k = max(1, min(n_modes, L.shape[0] - 2))
    vals, vecs = spla.eigsh(L, k=k, M=M, which="SM")  # smallest magnitude

    # Convert to frequencies + sort eigenpairs together
    vals = np.real(vals)
    vals = np.clip(vals, 0.0, None)

    omegas = np.sqrt(vals)
    freqs = omegas / (2.0 * math.pi)

    order = np.argsort(freqs)
    freqs = freqs[order]
    vecs = vecs[:, order]

    modes_hz: List[float] = [float(f) for f in freqs.tolist()]
    f0 = float(modes_hz[0]) if modes_hz else 0.0

    # -----------------------------
    # NEW: mode weights A_n at pickup
    # -----------------------------
    # Fixed pickup point for now (meters). Later: move to config.
    x0, y0 = 0.22, 0.17

    pickup_idx, pickup_xy_used = _pickup_index_on_interior_grid(
        x0=x0, y0=y0, dx=dx, dy=dy, nx=nx, ny=ny, nix=nix, niy=niy
    )

    mode_weights: List[float] = []
    for i_mode in range(vecs.shape[1]):
        v = vecs[:, i_mode]

        # Mass-normalize: v^T M v = 1
        # Here M = mass_scale * I  => v^T M v = mass_scale * ||v||^2
        vn2 = float(v @ v)
        mn = math.sqrt(mass_scale * vn2) if vn2 > 0 else 0.0
        if mn > 0:
            v = v / mn

        mode_weights.append(float(abs(v[pickup_idx])))

    # Normalize weights to max=1 for synthesis convenience
    if mode_weights:
        wmax = max(mode_weights)
        if wmax > 0:
            mode_weights = [w / wmax for w in mode_weights]

    return FemResult(
        f0=f0,
        modes=modes_hz,
        mode_weights=mode_weights,
        pickup_xy=pickup_xy_used,
    )
