# FEM/scripts/fem_api.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple


# -----------------------------
# Generic FEM result
# -----------------------------
@dataclass
class FemResult:
    """
    FEM output (modal):
    - f0: first (lowest) natural frequency [Hz]
    - modes: list of natural frequencies [Hz]

    New (optional) fields for audio realism:
    - mode_weights: A_n weights per mode (same length/order as modes)
      Typically extracted from mode shape at a chosen "pickup" point.
    - pickup_xy: (x0, y0) in meters used to compute mode_weights
    """
    f0: float
    modes: List[float]

    # Optional: modal weights (A_n) for synthesis
    mode_weights: Optional[List[float]] = None

    # Optional: where we sampled mode shapes (meters)
    pickup_xy: Optional[Tuple[float, float]] = None


# -----------------------------
# 2D rectangular orthotropic plate (Kirchhoff–Love) inputs
# -----------------------------
@dataclass
class RectPlateGeometry2D:
    a: float  # length in x [m]
    b: float  # length in y [m]
    h: float  # thickness [m]


@dataclass
class OrthotropicPlateMaterial:
    """
    Orthotropic (wood-like) plate material parameters.
    We store engineering constants; solver may internally compute D_ij.

    Units:
    - rho: kg/m^3
    - E_L, E_T, G_LT: Pa
    - nu_LT: dimensionless
    """
    rho: float
    E_L: float
    E_T: float
    G_LT: float
    nu_LT: float


@dataclass
class BoundaryCondition:
    """
    For now we only support a simple label; later we can extend with regions.
    Examples: "SSSS", "CCCC", etc.
    """
    kind: str  # e.g. "SSSS"


@dataclass
class SolverSettings:
    """
    Solver control parameters (numeric FEM).
    - dimension: 2 now, 3 later
    - model: selects which solver implementation to call
    """
    dimension: Literal[2, 3]
    model: str                 # e.g. "plate_ortho_kl_2d"
    n_modes: int = 20
    mesh_size: float = 0.01    # target element size [m]


@dataclass
class FemCase:
    """
    A complete case description that fem_main/config will build.
    """
    geometry_2d: Optional[RectPlateGeometry2D] = None
    material_ortho: Optional[OrthotropicPlateMaterial] = None
    bc: Optional[BoundaryCondition] = None
    solver: Optional[SolverSettings] = None


# -----------------------------
# Dispatcher entry point
# -----------------------------
def run_fem_case(case: FemCase) -> FemResult:
    """
    Main FEM entry point (dispatcher).
    Chooses a numeric solver based on case.solver.dimension + case.solver.model.
    """
    if case.solver is None:
        raise ValueError("FemCase.solver is missing.")

    if case.solver.dimension == 2 and case.solver.model == "plate_ortho_kl_2d":
        if case.geometry_2d is None:
            raise ValueError("FemCase.geometry_2d is missing for 2D plate solver.")
        if case.material_ortho is None:
            raise ValueError("FemCase.material_ortho is missing for 2D plate solver.")
        if case.bc is None:
            raise ValueError("FemCase.bc is missing for 2D plate solver.")

        # Call the real 2D solver
        try:
            from . import solver_2d_plate  # if scripts is a package later
        except Exception:
            # fallback import for "run as script" style
            import solver_2d_plate  # type: ignore

        if not hasattr(solver_2d_plate, "solve_plate_ortho_kl_2d"):
            raise NotImplementedError(
                "solver_2d_plate.solve_plate_ortho_kl_2d() is not implemented yet."
            )

        return solver_2d_plate.solve_plate_ortho_kl_2d(
            geom=case.geometry_2d,
            mat=case.material_ortho,
            bc=case.bc,
            n_modes=case.solver.n_modes,
            mesh_size=case.solver.mesh_size,
        )

    if case.solver.dimension == 3:
        raise NotImplementedError("3D solver not implemented yet (planned next).")

    raise NotImplementedError(
        f"Unsupported solver selection: dimension={case.solver.dimension}, model={case.solver.model}"
    )


# -----------------------------
# Backward-compatible wrapper (temporary)
# -----------------------------
@dataclass
class GuitarGeometry:
    length: float
    width: float
    depth: float
    top_thickness: float


@dataclass
class MaterialProps:
    young: float
    density: float
    poisson: float


def run_guitar_fem(geom: GuitarGeometry, mat: MaterialProps) -> FemResult:
    raise NotImplementedError(
        "Deprecated API: run_guitar_fem() was a placeholder. "
        "Use FemCase + run_fem_case() instead (fem_main will be updated in step 3)."
    )
