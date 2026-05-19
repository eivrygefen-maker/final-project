import json
import logging
import math
import gc
import shutil
import subprocess
import sys
import builtins
import faulthandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import petsc4py
from scipy import sparse as sp_sparse

from fem_mode_array_utils import (
    MODE_VECTOR_FILE_SUFFIX,
    csr_col_norm,
    csr_normalized_overlap,
    csr_u_slice,
    dense_to_csr_f32_column,
    load_mode_column_any,
    save_mode_csr,
)
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
# Explicit PETSc initialization on all ranks.
petsc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc

# Start PETSc logging on all ranks for collective-call tracing.
try:
    PETSc.Log.begin()
except Exception:
    pass

# Print Python-level tracebacks on fatal C-level crashes (e.g., segfaults).
faulthandler.enable()

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


WOOD_SURFACE_TAGS = (1, 3, 4)  # top plate, back plate, ribs/sides (FSI shell)
RIBS_SURFACE_TAG = 4
WOOD_FIX_SURFACE_TAG = 5
AIR_VOLUME_TAG = 10
ROOT_RANK = 0
SORTING_ROOT = Path(__file__).resolve().parents[1] / "SORTING"
SORTING_LOG = SORTING_ROOT / "candidates_log.json"
SORTING_TEMP_MODES = SORTING_ROOT / "temp_modes"


def set_sorting_root(path: Path) -> None:
    """
    Redirect ``SORTING_ROOT`` / ``SORTING_LOG`` / ``SORTING_TEMP_MODES`` (e.g. worker subprocess
    when ``fem_master_dynamic`` uses ``--sorting-root`` or a LAB layout). Must be called on all
    MPI ranks before any sorting-path I/O if workers are not using the default ``FEM/SORTING``.
    """
    global SORTING_ROOT, SORTING_LOG, SORTING_TEMP_MODES
    root = Path(path).expanduser().resolve()
    SORTING_ROOT = root
    SORTING_LOG = root / "candidates_log.json"
    SORTING_TEMP_MODES = root / "temp_modes"


def _root_print(*args, **kwargs):
    """Hard silence for worker ranks to avoid MPI stdio contention/spin-waits."""
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        builtins.print(*args, **kwargs)
        sys.stdout.flush()
        sys.stderr.flush()


# Override module-local print usage to root-only.
print = _root_print


def _emit(message: str, status_callback=None, level: str = "info") -> None:
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        if level == "error":
            LOGGER.error(message)
        elif level == "warning":
            LOGGER.warning(message)
        else:
            LOGGER.info(message)
    rank = MPI.COMM_WORLD.rank
    is_root = rank == ROOT_RANK
    should_print = is_root or level == "error"
    if should_print:
        print(message)
    if status_callback is not None and is_root:
        status_callback(message)


def _prepare_sorting_workspace() -> None:
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    SORTING_ROOT.mkdir(parents=True, exist_ok=True)
    SORTING_TEMP_MODES.mkdir(parents=True, exist_ok=True)
    with open(SORTING_LOG, "w", encoding="utf-8") as f:
        json.dump({"candidates": []}, f, indent=2)


def _append_candidate_metadata(entry: Dict) -> None:
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    if SORTING_LOG.exists():
        with open(SORTING_LOG, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"candidates": []}
    payload.setdefault("candidates", []).append(entry)
    with open(SORTING_LOG, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def cleanup_sorting_workspace() -> None:
    """Called after snapshot packing succeeds, to keep SORTING clean."""
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    if SORTING_TEMP_MODES.exists():
        shutil.rmtree(SORTING_TEMP_MODES, ignore_errors=True)
    SORTING_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SORTING_LOG, "w", encoding="utf-8") as f:
        json.dump({"candidates": []}, f, indent=2)


def _debug_rank(message: str) -> None:
    rank = MPI.COMM_WORLD.rank
    if rank == 0:
        print(f"[DEBUG] Rank {rank}: {message}")
        sys.stdout.flush()


def _debug_petsc_comm(name: str, obj) -> None:
    try:
        comm = obj.getComm()
        _debug_rank(f"{name} comm rank={comm.getRank()} size={comm.getSize()}")
    except Exception:
        _debug_rank(f"{name} comm unavailable")


def _phase_sync(phase_id: int, label: str, status_callback=None) -> None:
    """Collective phase checksum; hangs exactly where ranks diverge."""
    comm = MPI.COMM_WORLD
    try:
        sys.stderr.write(f"[MPI-TRACE] Rank {comm.rank} ENTERING phase {phase_id:04d} ({label})\n")
        sys.stderr.flush()
    except Exception:
        pass
    checksum = comm.allreduce(int(phase_id), op=MPI.SUM)
    expected = int(phase_id) * int(comm.size)
    try:
        sys.stderr.write(f"[MPI-TRACE] Rank {comm.rank} EXITING phase {phase_id:04d} ({label})\n")
        sys.stderr.flush()
    except Exception:
        pass
    if comm.rank == 0:
        msg = f"[PHASE] {phase_id:04d} {label} checksum={checksum}/{expected}"
        print(msg)
        sys.stdout.flush()
        sys.stderr.flush()
        _emit(msg, status_callback=status_callback)
    if checksum != expected:
        raise RuntimeError(f"Phase checksum mismatch at {phase_id} ({label}): {checksum} != {expected}")


def _sync_all_connectivity(msh: mesh.Mesh) -> None:
    """Build all topology connectivities collectively and synchronize with one collective reduction."""
    tdim = msh.topology.dim
    for d0 in range(tdim + 1):
        for d1 in range(tdim + 1):
            try:
                msh.topology.create_connectivity(d0, d1)
            except Exception:
                pass
    # Collective sync without explicit barrier (safer for uneven rank progress diagnostics).
    _ = MPI.COMM_WORLD.allreduce(1, op=MPI.SUM)


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


def _generate_mesh_with_gmsh(
    config_path: Optional[Path] = None,
    status_callback=None,
) -> None:
    from mesh_sync import build_mesh_for_config

    repo = Path(__file__).resolve().parents[2]
    cfg = Path(config_path).resolve() if config_path is not None else repo / "FEM" / "configs" / "guitar_3d.json"
    # In MPI runs, only rank 0 should invoke external gmsh process.
    comm = MPI.COMM_WORLD
    root_ok = 1
    root_err = ""
    if MPI.COMM_WORLD.rank == 0:
        try:
            build_mesh_for_config(cfg, repo)
        except Exception as exc:
            root_ok = 0
            root_err = str(exc)
    root_ok = comm.bcast(root_ok, root=0)
    root_err = comm.bcast(root_err, root=0)
    if int(root_ok) != 1:
        raise RuntimeError(f"Rank0 mesh generation failure broadcast: {root_err}")
    _emit(f"[mesh] built from config {cfg}", status_callback=status_callback)


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
    # Direct Gmsh->FEniCSx import:
    # Use dolfinx.io.gmsh.model_to_mesh to construct the mesh and tagged meshtags
    # directly from the .msh file (no meshio conversion / XDMF loading).
    _emit(f"[mesh] loading via dolfinx.io.gmsh.model_to_mesh: {mesh_file}", status_callback=status_callback)

    # Import gmsh and dolfinx gmsh helpers in a version-tolerant way.
    try:
        import gmsh  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gmsh Python module is required for dolfinx.io.gmsh.model_to_mesh mesh loading.",
            name="gmsh",
        ) from exc

    try:
        from dolfinx.io import gmsh as gmshio  # dolfinx >= 0.10
    except Exception:
        # Older dolfinx variants may expose gmshio under a different name.
        from dolfinx.io import gmshio as gmshio  # type: ignore

    rank = 0
    gdim = 3
    comm = MPI.COMM_WORLD

    # model_to_mesh only processes the gmsh model on `rank`; other ranks can pass `None`.
    gmsh_model = gmsh.model if MPI.COMM_WORLD.rank == rank else None
    root_ok = 1
    root_err = ""
    if MPI.COMM_WORLD.rank == rank:
        try:
            gmsh.initialize()
            gmsh.open(str(mesh_file))
            _emit("[mesh] gmsh file opened successfully on rank 0.", status_callback=status_callback)
        except Exception as exc:
            root_ok = 0
            root_err = str(exc)
    root_ok = comm.bcast(root_ok, root=0)
    root_err = comm.bcast(root_err, root=0)
    if int(root_ok) != 1:
        raise RuntimeError(f"Rank0 gmsh open failure broadcast: {root_err}")

    _phase_sync(1100, "before model_to_mesh", status_callback=status_callback)
    mesh_data = gmshio.model_to_mesh(gmsh_model, MPI.COMM_WORLD, rank, gdim=gdim)
    _phase_sync(1101, "after model_to_mesh", status_callback=status_callback)

    if MPI.COMM_WORLD.rank == rank:
        gmsh.finalize()

    msh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    # Ensure all topology connectivities (0..tdim) are built on all ranks synchronously.
    _sync_all_connectivity(msh)
    _phase_sync(1102, "after connectivity sync", status_callback=status_callback)
    return msh, cell_tags, facet_tags


def _load_mesh_and_tags(mesh_file: Path, status_callback=None):
    _emit("Step 1/5: Loading mesh and physical tags...", status_callback=status_callback)
    msh, cell_tags, facet_tags = _load_mesh_with_fallback(mesh_file, status_callback=status_callback)
    if cell_tags is None:
        raise RuntimeError("No 3D physical tags detected in mesh. Expected Air_Internal=10.")
    if facet_tags is None:
        raise RuntimeError("No 2D physical tags detected in mesh. Expected Top/Back/Ribs facet tags.")

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
    try:
        fdim = msh.topology.dim - 1
        n_facets_local = int(msh.topology.index_map(fdim).size_local)
        fidx = np.asarray(facet_tags.indices, dtype=np.int32)
        in_range = int(np.sum((fidx >= 0) & (fidx < n_facets_local)))
        _emit(
            f"[diag] facet_tags map check: dim={facet_tags.dim}, n_local_facets={n_facets_local}, "
            f"tagged_facets={fidx.size}, in_local_range={in_range}, "
            f"find1={facet_tags.find(1).size}, find2={facet_tags.find(2).size}, "
            f"find3={facet_tags.find(3).size}, find4={facet_tags.find(4).size}, "
            f"find5={facet_tags.find(5).size}",
            status_callback=status_callback,
        )
    except Exception as exc:
        _emit(f"[diag][warn] facet_tags map check failed: {exc}", status_callback=status_callback, level="warning")
    # Explicit per-tag sanity counts requested for fallback validation.
    vol_counts = {1: int(np.sum(cell_tags.values == 1)), 2: int(np.sum(cell_tags.values == 2)),
                  3: int(np.sum(cell_tags.values == 3)), 10: int(np.sum(cell_tags.values == 10))}
    fac_counts = {
        1: int(np.sum(facet_tags.values == 1)),
        2: int(np.sum(facet_tags.values == 2)),
        3: int(np.sum(facet_tags.values == 3)),
        4: int(np.sum(facet_tags.values == 4)),
        5: int(np.sum(facet_tags.values == 5)),
        10: int(np.sum(facet_tags.values == 10)),
    }
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


def _orthotropic_plate_stiffness(mat: Dict, thickness: float) -> Dict[str, float]:
    """
    Kirchhoff–Love orthotropic plate stiffnesses (membrane A_ij and bending D_ij).

    Grain direction (1) aligns with local e1 on each facet (≈ guitar body +x).
    Matches ``solver_2d_plate._compute_bending_stiffness_from_engineering_constants``.
    """
    E_L = float(mat.get("E_L", 1.0e9))
    E_T = float(mat.get("E_T", E_L))
    G_LT = float(mat.get("G_LT", E_L / 10.0))
    nu_LT = float(mat.get("nu_LT", 0.3))
    rho = float(mat["density"])
    h = float(thickness)
    nu_TL = nu_LT * (E_T / E_L)
    denom = 1.0 - nu_LT * nu_TL
    if abs(denom) < 1e-12:
        raise ValueError(f"Invalid orthotropic constants: 1 - nu_LT*nu_TL ≈ 0 (E_L={E_L}, E_T={E_T})")
    h3_over_12 = (h**3) / 12.0
    D11 = (E_L * h3_over_12) / denom
    D22 = (E_T * h3_over_12) / denom
    D12 = (nu_TL * E_L * h3_over_12) / denom
    D66 = G_LT * h3_over_12
    A11 = (E_L * h) / denom
    A22 = (E_T * h) / denom
    A12 = (nu_TL * E_L * h) / denom
    A66 = G_LT * h
    return {
        "rho": rho,
        "E_L": E_L,
        "E_T": E_T,
        "G_LT": G_LT,
        "nu_LT": nu_LT,
        "A11": A11,
        "A22": A22,
        "A12": A12,
        "A66": A66,
        "D11": D11,
        "D22": D22,
        "D12": D12,
        "D66": D66,
    }


def _split_wood_materials(config: Dict) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """
    Per-region orthotropic shell properties for Top Plate (facet tag 1) and Back/Sides (tag 3).

    Returns (top_dict, back_dict, top_thickness_m, back_thickness_m).
    """
    from wood_library import resolve_plate_thicknesses

    top = config["materials"]["top"]
    back = config["materials"]["back"]
    t_top, t_back = resolve_plate_thicknesses(config)
    top_out = _orthotropic_plate_stiffness(top, t_top)
    back_out = _orthotropic_plate_stiffness(back, t_back)
    return top_out, back_out, t_top, t_back


def _plate_local_frame(n, P):
    """Local orthotropic axes on a curved facet: e1 ≈ projected body +x, e2 = n × e1."""
    ex = ufl.as_vector((1.0, 0.0, 0.0))
    ey = ufl.as_vector((0.0, 1.0, 0.0))
    e1x = P * ex
    e1y = P * ey
    n1 = ufl.sqrt(ufl.dot(e1x, e1x))
    n2 = ufl.sqrt(ufl.dot(e1y, e1y))
    e1 = e1x / (n1 + 1e-30) * ufl.conditional(ufl.gt(n1, 1e-8), 1.0, 0.0) + e1y / (n2 + 1e-30) * ufl.conditional(
        ufl.le(n1, 1e-8), 1.0, 0.0
    )
    e2 = ufl.cross(n, e1)
    e2 = e2 / (ufl.sqrt(ufl.dot(e2, e2)) + 1e-30)
    return e1, e2


def _membrane_strain_voigt(eps, e1, e2):
    e11 = ufl.dot(e1, eps * e1)
    e22 = ufl.dot(e2, eps * e2)
    e12 = ufl.dot(e1, eps * e2) + ufl.dot(e2, eps * e1)
    return e11, e22, e12


def _curvature_voigt(w_n, e1, e2, P):
    """Surface curvatures κ_ij from normal displacement (Kirchhoff–Love, facet-local)."""
    gw = P * ufl.grad(w_n)
    k11 = ufl.dot(e1, P * ufl.grad(ufl.dot(e1, gw)))
    k22 = ufl.dot(e2, P * ufl.grad(ufl.dot(e2, gw)))
    k12 = 0.5 * (
        ufl.dot(e1, P * ufl.grad(ufl.dot(e2, gw))) + ufl.dot(e2, P * ufl.grad(ufl.dot(e1, gw)))
    )
    return k11, k22, k12


def _orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, ortho: Dict[str, float]):
    """Membrane + bending bilinear form for one orthotropic plate region."""
    eu11, eu22, eu12 = _membrane_strain_voigt(eps_u, e1, e2)
    ev11, ev22, ev12 = _membrane_strain_voigt(eps_v, e1, e2)
    mem = (
        ortho["A11"] * eu11 * ev11
        + ortho["A22"] * eu22 * ev22
        + ortho["A12"] * (eu11 * ev22 + eu22 * ev11)
        + ortho["A66"] * eu12 * ev12
    )
    ku11, ku22, ku12 = _curvature_voigt(w_n, e1, e2, P)
    kv11, kv22, kv12 = _curvature_voigt(v_n, e1, e2, P)
    bend = (
        ortho["D11"] * ku11 * kv11
        + ortho["D22"] * ku22 * kv22
        + ortho["D12"] * (ku11 * kv22 + ku22 * kv11)
        + 4.0 * ortho["D66"] * ku12 * kv12
    )
    return mem + bend


def _audit_and_scale_mesh_units(msh: mesh.Mesh, config: Dict, status_callback=None) -> None:
    """Audit mesh coordinate units and apply optional mm->m scaling."""
    solver_cfg = config.get("solver", {})
    mode = str(solver_cfg.get("mesh_unit_mode", "auto")).strip().lower()
    coords = msh.geometry.x
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    span = maxs - mins
    span_max = float(np.max(span))

    scale = 1.0
    # Guitar body characteristic length should be O(1e-1) meters.
    if mode in ("millimeter", "millimeters", "mm"):
        scale = 1.0e-3
    elif mode in ("meter", "meters", "m"):
        scale = 1.0
    elif mode == "auto":
        # If span looks like hundreds (e.g. ~480), assume mm and convert to meters.
        if span_max > 5.0:
            scale = 1.0e-3
    else:
        _emit(f"[diag][warn] unknown mesh_unit_mode='{mode}', using auto.", status_callback=status_callback, level="warning")
        if span_max > 5.0:
            scale = 1.0e-3

    if scale != 1.0:
        msh.geometry.x[:, :] *= scale
        coords = msh.geometry.x
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        span = maxs - mins
        span_max = float(np.max(span))
        _emit(
            f"[diag] mesh coordinate scaling applied: factor={scale:.3e} (mode={mode})",
            status_callback=status_callback,
        )

    print(f"[DIAG] Mesh Bounds - X: {mins[0]} to {maxs[0]}")
    print(f"[DIAG] Mesh Bounds - Y: {mins[1]} to {maxs[1]}, Z: {mins[2]} to {maxs[2]}")
    print(f"[DIAG] Mesh Span (m): dx={span[0]:.6f}, dy={span[1]:.6f}, dz={span[2]:.6f}")
    if span_max < 0.05 or span_max > 2.0:
        _emit(
            f"[diag][warn] unusual guitar size after unit audit (max span={span_max:.6f} m).",
            status_callback=status_callback,
            level="warning",
        )
    sys.stdout.flush()


def _mesh_interface_diagnostic(msh: mesh.Mesh, cell_tags, facet_tags, status_callback=None) -> None:
    """Report node sharing between wood (1/2/3) and air (10)."""
    try:
        tdim = msh.topology.dim
        fdim = tdim - 1
        msh.topology.create_connectivity(tdim, 0)
        msh.topology.create_connectivity(tdim, fdim)
        msh.topology.create_connectivity(fdim, tdim)
        msh.topology.create_connectivity(fdim, 0)
        c2v = msh.topology.connectivity(tdim, 0)
        c2f = msh.topology.connectivity(tdim, fdim)
        f2c = msh.topology.connectivity(fdim, tdim)
        f2v = msh.topology.connectivity(fdim, 0)

        wood_cells = np.array([], dtype=np.int32)
        for tag in (1, 2, 3):
            c = np.asarray(cell_tags.find(tag), dtype=np.int32)
            if c.size > 0:
                wood_cells = np.unique(np.concatenate([wood_cells, c]).astype(np.int32))
        air_cells = np.asarray(cell_tags.find(AIR_VOLUME_TAG), dtype=np.int32)

        def _nodes_from_cells(cells: np.ndarray) -> np.ndarray:
            if cells.size == 0:
                return np.array([], dtype=np.int32)
            blocks = [c2v.links(int(ci)) for ci in cells]
            if not blocks:
                return np.array([], dtype=np.int32)
            return np.unique(np.concatenate(blocks)).astype(np.int32)

        wood_nodes = _nodes_from_cells(wood_cells)
        air_nodes = _nodes_from_cells(air_cells)
        shared_nodes = np.intersect1d(wood_nodes, air_nodes).astype(np.int32)
        wood_only_nodes = np.setdiff1d(wood_nodes, air_nodes, assume_unique=False).astype(np.int32)
        air_only_nodes = np.setdiff1d(air_nodes, wood_nodes, assume_unique=False).astype(np.int32)

        wood_surface_facets = np.unique(
            np.concatenate(
                [
                    np.asarray(facet_tags.find(1), dtype=np.int32),
                    np.asarray(facet_tags.find(3), dtype=np.int32),
                    np.asarray(facet_tags.find(4), dtype=np.int32),
                ]
            )
        ).astype(np.int32)
        wood_surface_nodes = (
            np.unique(np.concatenate([f2v.links(int(fi)) for fi in wood_surface_facets])).astype(np.int32)
            if wood_surface_facets.size > 0
            else np.array([], dtype=np.int32)
        )

        interface_facets = []
        wood_tag_set = {1, 2, 3, 4}
        for ci in air_cells:
            for fi in c2f.links(int(ci)):
                nbr_cells = np.asarray(f2c.links(int(fi)), dtype=np.int32)
                if nbr_cells.size == 0:
                    continue
                nbr_tags = set(int(cell_tags.values[int(nc)]) for nc in nbr_cells)
                if (AIR_VOLUME_TAG in nbr_tags) and any(t in wood_tag_set for t in nbr_tags):
                    interface_facets.append(int(fi))
        interface_facets = np.unique(np.asarray(interface_facets, dtype=np.int32))
        interface_nodes = (
            np.unique(np.concatenate([f2v.links(int(fi)) for fi in interface_facets])).astype(np.int32)
            if interface_facets.size > 0
            else np.array([], dtype=np.int32)
        )

        boundary_shared = np.intersect1d(wood_surface_nodes, interface_nodes).astype(np.int32)

        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                f"[DIAG][IFACE] volume nodes: wood_only={wood_only_nodes.size}, "
                f"air_only={air_only_nodes.size}, shared={shared_nodes.size}"
            )
            print(
                f"[DIAG][IFACE] boundary nodes: wood_surface={wood_surface_nodes.size}, "
                f"air_inner={interface_nodes.size}, shared_boundary={boundary_shared.size}"
            )
            sys.stdout.flush()
            if boundary_shared.size == 0:
                _emit(
                    "[DIAG][IFACE][WARN] shared boundary nodes == 0 (possible unstitched wood-air interface).",
                    status_callback=status_callback,
                    level="warning",
                )
    except Exception as exc:
        _emit(f"[diag][warn] mesh interface diagnostic failed: {exc}", status_callback=status_callback, level="warning")


def _mat_frobenius_norm(a_form) -> float:
    """Assemble a bilinear form and return its Frobenius norm (PETSc default norm)."""
    K = assemble_matrix(fem.form(a_form))
    K.assemble()
    try:
        return float(K.norm())
    finally:
        K.destroy()


def _diagnose_shell_stiffness_assembly(
    a_uu,
    shell_top,
    shell_back,
    ds_top,
    ds_back,
    tag_top: int,
    tag_back: int,
    n_facets_top: int,
    n_facets_back: int,
    *,
    status_callback=None,
) -> None:
    """
    Stiffness assembly diagnostic: verify shell forms integrate on tagged facets (1 top, 3 back).
    """
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    print(
        f"[DEBUG] Shell ds measures: ds_top → facet tag {tag_top} (n_local_facets={n_facets_top}), "
        f"ds_back → facet tag {tag_back} (n_local_facets={n_facets_back})"
    )
    a_uu_norm = float("nan")
    top_norm = float("nan")
    back_norm = float("nan")
    try:
        a_uu_norm = _mat_frobenius_norm(a_uu)
        print(f"[DEBUG] Matrix norm of a_uu: {a_uu_norm:.6e}")
    except Exception as exc:
        _emit(f"[DEBUG][warn] a_uu norm assembly failed: {exc}", status_callback=status_callback, level="warning")
    if n_facets_top > 0:
        try:
            top_norm = _mat_frobenius_norm(shell_top * ds_top)
            print(f"[DEBUG] Matrix norm of shell_top*ds(tag{tag_top}): {top_norm:.6e}")
        except Exception as exc:
            _emit(
                f"[DEBUG][warn] shell_top*ds(tag{tag_top}) norm failed: {exc}",
                status_callback=status_callback,
                level="warning",
            )
    else:
        print(f"[DEBUG][warn] tag {tag_top} has zero local facets — shell_top contributes nothing.")
    if n_facets_back > 0:
        try:
            back_norm = _mat_frobenius_norm(shell_back * ds_back)
            print(f"[DEBUG] Matrix norm of shell_back*ds(tag{tag_back}): {back_norm:.6e}")
        except Exception as exc:
            _emit(
                f"[DEBUG][warn] shell_back*ds(tag{tag_back}) norm failed: {exc}",
                status_callback=status_callback,
                level="warning",
            )
    else:
        print(f"[DEBUG][warn] tag {tag_back} has zero local facets — shell_back contributes nothing.")
    if (not math.isfinite(a_uu_norm)) or a_uu_norm < 1.0e-15:
        _emit(
            "[DEBUG][CRITICAL] ||a_uu|| ≈ 0 — structural shell stiffness is not assembled. "
            f"Check mesh facet tags ({tag_top} top, {tag_back} back) vs build_3d_guitar physical groups "
            "and that ds subdomain_data matches facet_tags.",
            status_callback=status_callback,
            level="error",
        )
    elif top_norm < 1.0e-15 and n_facets_top > 0:
        _emit(
            f"[DEBUG][warn] shell_top*ds(tag{tag_top}) norm ≈ 0 despite {n_facets_top} tagged facets.",
            status_callback=status_callback,
            level="warning",
        )
    elif back_norm < 1.0e-15 and n_facets_back > 0:
        _emit(
            f"[DEBUG][warn] shell_back*ds(tag{tag_back}) norm ≈ 0 despite {n_facets_back} tagged facets.",
            status_callback=status_callback,
            level="warning",
        )
    sys.stdout.flush()


def _plate_modal_energy_ratios(
    phi: PETSc.Vec,
    M_top: Optional[PETSc.Mat],
    M_back: Optional[PETSc.Mat],
    work: PETSc.Vec,
    mass_top: float,
    mass_back: float,
) -> Tuple[float, float]:
    """
    Relative shell participation on each plate mass block (scale-invariant to coupled SLEPc norm).

    tag1_ratio = phi^T M_top phi / (phi^T M_top phi + phi^T M_back phi)
    tag3_ratio = phi^T M_back phi / (phi^T M_top phi + phi^T M_back phi)

    ``mass_top`` / ``mass_back`` are retained for call-site compatibility (unused here).
    """
    e_top = 0.0
    e_back = 0.0
    if M_top is not None:
        M_top.mult(phi, work)
        e_top = float(np.real(phi.dot(work)))
    if M_back is not None:
        M_back.mult(phi, work)
        e_back = float(np.real(phi.dot(work)))
    total_wood_energy = e_top + e_back
    if total_wood_energy < 1.0e-18:
        return 0.0, 0.0
    return e_top / total_wood_energy, e_back / total_wood_energy


def _coupled_pressure_dof_scale(solver_cfg: Dict) -> float:
    """
    Similarity scale s on pressure DOFs (D = diag(I, s·I_p)); applied consistently to all
    pressure blocks so GNHEP eigenvalues (ω²) are unchanged but block magnitudes match u.
    """
    s = float(solver_cfg.get("pressure_dof_scale", 1.0e5))
    return s if s > 0.0 else 1.0


def _normalize_assembled_gnhep(A: PETSc.Mat, M: PETSc.Mat, solver_cfg: Dict) -> float:
    """Common Frobenius scaling of A and M (preserves eigenvalues; improves EPS conditioning)."""
    if not _solver_bool(solver_cfg, "gnhep_normalize_matrices", default=True):
        return 1.0
    a_n = float(A.norm())
    m_n = float(M.norm())
    scale = max(a_n, m_n, 1.0e-30)
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    A.scale(1.0 / scale)
    M.scale(1.0 / scale)
    return scale


def _slepc_spectrum_min_hz(solver_cfg: Dict, scheduler_hz: float) -> float:
    """Lower frequency bound (Hz) for physical modes; scheduler_hz is logging/scheduling only."""
    return float(
        solver_cfg.get(
            "min_valid_mode_hz",
            solver_cfg.get("eps_spectrum_min_hz", scheduler_hz),
        )
    )


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
    mass_top: float = 1.0,
    mass_back: float = 1.0,
    eps_max_it_cap: Optional[int] = None,
) -> Tuple[int, List[Tuple[float, np.ndarray, Optional[float], Optional[float]]]]:
    """
    Krylov-Schur GNHEP batch: shift-invert at ``shift_hz`` (TARGET_MAGNITUDE on λ = ω²).

    ``shift_hz`` is the worker/master target frequency; ST sinvert inverts around that band.
    """
    min_hz = _slepc_spectrum_min_hz(solver_cfg, shift_hz)
    target_hz = float(shift_hz)
    target_lambda = (2.0 * math.pi * target_hz) ** 2
    rigid_tol = float(solver_cfg.get("coupled_rigid_lambda_tol", 1.0e-10))
    rigid_buf = int(solver_cfg.get("eps_rigid_mode_buffer", 10))
    nev_request = int(batch) + max(rigid_buf, 0)

    # ST σ near target + jitter keeps inversion away from an exact eigenvalue on the shift.
    shift_jitter_hz = float(solver_cfg.get("shift_jitter_hz", 5.0))
    st_sigma_hz = max(1.0, target_hz + shift_jitter_hz)
    st_sigma = (2.0 * math.pi * st_sigma_hz) ** 2

    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    ks_restart = float(solver_cfg.get("krylov_schur_restart", 0.5))
    if ks_restart <= 0.0:
        ks_restart = 0.5
    eps.setKrylovSchurRestart(ks_restart)
    ks_pert = float(solver_cfg.get("eps_krylovschur_pertsize", 1.0e-8))
    if ks_pert > 0.0:
        try:
            eps.setKrylovSchurPertSize(ks_pert)
        except AttributeError:
            pass

    st = eps.getST()
    _st_name = str(solver_cfg.get("st_type", "sinvert")).strip().lower()
    if _st_name in ("shift", "stshift"):
        st.setType(SLEPc.ST.Type.SHIFT)
    else:
        st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(st_sigma)

    ksp = st.getKSP()
    pc = ksp.getPC()
    _debug_rank("Entering KSP Setup")
    # Shift-invert ST: direct LU + MUMPS (robust pivoting on coupled indefinite blocks; P1 keeps DOFs small).
    st_ksp_type = str(solver_cfg.get("st_ksp_type", "preonly"))
    st_pc_type = str(solver_cfg.get("st_pc_type", "lu"))
    ksp.setType(st_ksp_type)
    pc.setType(st_pc_type)
    _st_factor = str(
        solver_cfg.get(
            "st_pc_factor_mat_solver_type",
            solver_cfg.get("st_factor_solver_type", "mumps"),
        )
    )
    if st_pc_type.lower() == "lu":
        try:
            pc.setFactorSolverType(_st_factor)
        except Exception:
            pass
    try:
        ksp.setTolerances(rtol=1.0e-12, atol=1.0e-14, max_it=1)
    except Exception:
        pass

    # MUMPS options for the ST inner KSP/PC must use the SLEPc "st_" prefix (e.g. st_mat_mumps_icntl_*),
    # otherwise PETSc ignores them and MUMPS keeps factory workspace estimates.
    mumps_icntl_14 = int(solver_cfg.get("mat_mumps_icntl_14", 500))
    mumps_icntl_24 = int(solver_cfg.get("mat_mumps_icntl_24", 1))
    mumps_icntl_6 = int(solver_cfg.get("mat_mumps_icntl_6", 7))
    mumps_icntl_12 = int(solver_cfg.get("mat_mumps_icntl_12", 1))
    # ICNTL(4)=0: silent MUMPS (no statistics I/O); non-zero enables host printing and slows each factorization.
    mumps_icntl_4 = int(solver_cfg.get("mat_mumps_icntl_4", 0))
    petsc_opts = PETSc.Options()
    petsc_opts["st_mat_mumps_icntl_14"] = mumps_icntl_14
    petsc_opts["st_mat_mumps_icntl_24"] = mumps_icntl_24
    petsc_opts["st_mat_mumps_icntl_6"] = mumps_icntl_6
    petsc_opts["st_mat_mumps_icntl_12"] = mumps_icntl_12
    petsc_opts["st_mat_mumps_icntl_4"] = mumps_icntl_4
    if (
        MPI.COMM_WORLD.size > 1
        and str(st_pc_type).lower() == "lu"
        and MPI.COMM_WORLD.rank == ROOT_RANK
    ):
        _emit(
            "[solver][warn] ST shift-invert LU is most stable with a single MPI process "
            f"(e.g. `mpiexec -n 1` or plain `python3`); MPI_COMM_WORLD.size={MPI.COMM_WORLD.size}.",
            status_callback=status_callback,
            level="warning",
        )
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
    petsc_opts["eps_which"] = "target_magnitude"
    petsc_opts["eps_target"] = target_lambda
    if _st_name not in ("shift", "stshift"):
        petsc_opts["st_type"] = "sinvert"
    petsc_opts["st_ksp_type"] = st_ksp_type
    petsc_opts["st_pc_type"] = st_pc_type
    if st_pc_type.lower() == "lu":
        petsc_opts["st_pc_factor_mat_solver_type"] = _st_factor
    petsc_opts["st_ksp_norm_type"] = str(solver_cfg.get("st_ksp_norm_type", "none"))
    if ks_pert > 0.0:
        petsc_opts["eps_krylovschur_pertsize"] = ks_pert

    ncv_min_factor = float(solver_cfg.get("eps_ncv_min_factor", 3.0))
    ncv_floor = int(math.ceil(max(3.0, ncv_min_factor) * float(nev_request)))
    ncv_cfg = int(solver_cfg.get("target_ncv", 0))
    ncv = max(ncv_floor, ncv_cfg, 40)
    eps.setDimensions(int(nev_request), int(ncv))
    eps_max_it = int(solver_cfg.get("eigs_maxiter", solver_cfg.get("eps_max_it", 3000)))
    if eps_max_it_cap is not None:
        # Never cap below batch-scaled floor (legacy bug used sifter_batch_max_it ≈ 50 for nev=80).
        eps_max_it = max(int(eps_max_it_cap), int(batch) * 5, 200)
    eps_tol = float(solver_cfg.get("eps_tol", solver_cfg.get("eigs_tol", 1.0e-6)))
    eps.setTolerances(eps_tol, eps_max_it)

    diag_vec = A.getDiagonal()
    diag_arr = np.real(diag_vec.array)
    if diag_arr.size > 0:
        diag_min = float(np.min(diag_arr))
        diag_max = float(np.max(diag_arr))
    else:
        diag_min = float("nan")
        diag_max = float("nan")
    _emit(
        f"[solver] EPS spectrum batch (target={target_hz:.2f} Hz): "
        f"target_lambda={target_lambda:.6e}, min_hz={min_hz:.2f}, "
        f"ST={_st_name} sigma={st_sigma_hz:.2f} Hz (jitter={shift_jitter_hz:.2f}), "
        f"which=TARGET_MAGNITUDE, ks_restart={ks_restart:.3f}, ks_pert={ks_pert:.2e}, "
        f"nev_request={nev_request} (batch={batch}+rigid_buf={rigid_buf}), "
        f"ncv={ncv}, eps_tol={eps_tol:.1e}, eps_max_it={eps_max_it}, "
        f"KSP={ksp.getType()}, PC={pc.getType()}, factor={_st_factor}, "
        f"MUMPS via ST opts: ICNTL4={mumps_icntl_4}, ICNTL6={mumps_icntl_6}, ICNTL12={mumps_icntl_12}, "
        f"ICNTL14={mumps_icntl_14}, ICNTL24={mumps_icntl_24}), "
        f"diag_shift={diag_shift:.2e}, A_diag_min={diag_min:.6e}, A_diag_max={diag_max:.6e}",
        status_callback=status_callback,
    )
    eps.setFromOptions()
    _debug_petsc_comm("A", A)
    _debug_petsc_comm("M", M)
    _debug_petsc_comm("EPS", eps)
    try:
        acomm = A.getComm()
        if acomm.getSize() != MPI.COMM_WORLD.size:
            raise RuntimeError(f"A communicator size mismatch: {acomm.getSize()} vs world {MPI.COMM_WORLD.size}")
    except Exception as exc:
        _emit(f"[error] communicator audit failed before EPS solve: {exc}", status_callback=status_callback, level="error")
        raise
    opts = PETSc.Options()
    opts["eps_monitor"] = None
    opts["eps_converged_reason"] = None
    eps.setFromOptions()
    # Re-apply after setFromOptions so CLI/options cannot leave sinvert without eps_target.
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(target_lambda)
    st = eps.getST()
    if _st_name in ("shift", "stshift"):
        st.setType(SLEPc.ST.Type.SHIFT)
    else:
        st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(st_sigma)
    print(f"[HEARTBEAT] Rank {MPI.COMM_WORLD.rank} reached EPS Solve")
    sys.stdout.flush()
    _debug_rank("Entering EPS Solve")
    eps.solve()

    its = eps.getIterationNumber()
    nconv = eps.getConverged()
    reason = eps.getConvergedReason()
    _emit(
        f"[solver] EPS sweep (scheduler={shift_hz:.1f} Hz): iterations={its}, converged={nconv}, "
        f"requested={nev_request}, min_hz={min_hz:.2f}, reason={reason}",
        status_callback=status_callback,
    )

    out: List[Tuple[float, np.ndarray, Optional[float], Optional[float]]] = []
    rvec = A.createVecRight()
    skipped_rigid = 0
    skipped_below_min = 0
    for i in range(int(nconv)):
        eig = eps.getEigenpair(i, rvec)
        eig_r = float(np.real(eig))
        if eig_r <= rigid_tol:
            skipped_rigid += 1
            continue
        omega = math.sqrt(max(eig_r, 0.0))
        f_hz = omega / (2.0 * math.pi)
        if f_hz + 1e-9 < float(min_hz):
            skipped_below_min += 1
            continue
        rt: Optional[float] = None
        rb: Optional[float] = None
        if work is not None and (M_top is not None or M_back is not None):
            rt, rb = _plate_modal_energy_ratios(rvec, M_top, M_back, work, mass_top, mass_back)
        out.append((f_hz, rvec.array.copy(), rt, rb))
        if len(out) >= int(batch):
            break

    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(
            f"[solver] EPS harvest: kept={len(out)}/{batch} from nconv={nconv} "
            f"(skip rigid={skipped_rigid}, f<{min_hz:.1f}Hz={skipped_below_min})"
        )
        sys.stdout.flush()

    if len(out) < int(batch) and MPI.COMM_WORLD.rank == ROOT_RANK:
        _emit(
            f"[solver][warn] EPS harvested {len(out)}/{batch} modes with f>={min_hz:.1f} Hz "
            f"(nconv={nconv}); increase eigs_maxiter (now {eps_max_it}) or target_ncv (now {ncv}).",
            status_callback=status_callback,
            level="warning",
        )

    try:
        rvec.destroy()
    except Exception:
        pass
    try:
        diag_vec.destroy()
    except Exception:
        pass
    eps.destroy()
    return len(out), out


def _solve_structural_only_evp(
    msh: mesh.Mesh,
    cell_tags,
    facet_tags,
    config: Dict,
    num_modes: int,
    status_callback=None,
) -> Tuple[mesh.Mesh, fem.FunctionSpace, List[float], np.ndarray, int, int]:
    """Structural-only diagnostic EVP (displacement field only, no acoustic coupling)."""
    _phase_sync(2100, "structural-only enter", status_callback=status_callback)
    _emit("[diag] structural-only diagnosis enabled: solving u-field EVP only.", status_callback=status_callback)
    tdim = msh.topology.dim
    fdim = tdim - 1
    # Build all connectivity needed for facet- and vertex-based localization before creating V_u.
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(0, tdim)
    msh.topology.create_connectivity(tdim, 0)

    facets_t1 = np.array(facet_tags.find(1), dtype=np.int32)
    facets_t2 = np.array(facet_tags.find(2), dtype=np.int32)
    facets_t3 = np.array(facet_tags.find(3), dtype=np.int32)
    facets_t4 = np.array(facet_tags.find(RIBS_SURFACE_TAG), dtype=np.int32)
    facets_fix = np.array(facet_tags.find(WOOD_FIX_SURFACE_TAG), dtype=np.int32)

    print(
        f"[DIAG] facet tag counts: tag1={facets_t1.size}, "
        f"tag2={facets_t2.size}, tag3_back={facets_t3.size}, "
        f"tag4_ribs={facets_t4.size}, tag5_fix={facets_fix.size}"
    )
    sys.stdout.flush()

    # Hard-coded P1 displacement (same as coupled path; config cannot raise FE order here).
    _u_deg_struct = 1
    u_el = element("Lagrange", msh.basix_cell(), _u_deg_struct, shape=(3,))
    V_u = fem.functionspace(msh, u_el)
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    xdmf_dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)

    top_m, back_m, t_top, t_back = _split_wood_materials(config)
    top_mat = config["materials"]["top"]
    back_mat = config["materials"]["back"]
    if msh.comm.rank == ROOT_RANK:
        print(
            f"[DIAG] Material audit (per-tag, no averaging): "
            f"top tag1(D11={top_m['D11']:.3e}, E_L={top_m['E_L']:.3e}, rho={top_m['rho']:.1f}), "
            f"back tag3(D11={back_m['D11']:.3e}, E_L={back_m['E_L']:.3e}, rho={back_m['rho']:.1f}), "
            f"t_top={t_top:.4f} m t_back={t_back:.4f} m | "
            f"{top_mat.get('name', '')!r} / {back_mat.get('name', '')!r}"
        )
    # Volume diagnostic branch: isotropic 3D reduction from E_L (coupled path uses facet KL shell).
    nu_t = float(top_m["nu_LT"])
    nu_b = float(back_m["nu_LT"])
    mu_t = top_m["E_L"] / (2.0 * (1.0 + nu_t))
    lam_t = top_m["E_L"] * nu_t / ((1.0 + nu_t) * (1.0 - 2.0 * nu_t))
    mu_b = back_m["E_L"] / (2.0 * (1.0 + nu_b))
    lam_b = back_m["E_L"] * nu_b / ((1.0 + nu_b) * (1.0 - 2.0 * nu_b))
    # Volume tags: 1=top plate, 2=back plate, 3=ribs/sides (back wood).
    vol_top = xdmf_dx(1)
    vol_back_sides = xdmf_dx(2) + xdmf_dx(3)
    wood_dx = vol_top + vol_back_sides

    eps_u = ufl.sym(ufl.grad(u))
    eps_v = ufl.sym(ufl.grad(v))
    a_uu = (
        2.0 * mu_t * ufl.inner(eps_u, eps_v) + lam_t * ufl.div(u) * ufl.div(v)
    ) * vol_top + (
        2.0 * mu_b * ufl.inner(eps_u, eps_v) + lam_b * ufl.div(u) * ufl.div(v)
    ) * vol_back_sides
    m_uu = top_m["rho"] * ufl.dot(u, v) * vol_top + back_m["rho"] * ufl.dot(u, v) * vol_back_sides

    # Optional dummy penalty on air cells (vacuum / structural-only diagnostic only).
    if _solver_bool(config.get("solver", {}), "structural_vacuum_air_dummy", default=True):
        air_tags = [AIR_VOLUME_TAG]
        for tag in air_tags:
            a_uu += 1.0e11 * ufl.inner(u, v) * xdmf_dx(tag)
            m_uu += 1.0e-9 * ufl.inner(u, v) * xdmf_dx(tag)

    # V_u coverage diagnostic by tag.
    try:
        f_to_v = msh.topology.connectivity(fdim, 0)
        tag_u_counts = {}
        for tag, facets_tag in ((1, facets_t1), (2, facets_t2), (3, facets_t3)):
            if facets_tag.size == 0:
                tag_u_counts[tag] = 0
                continue
            verts_tag = np.unique(np.concatenate([f_to_v.links(int(f)) for f in facets_tag])).astype(np.int32)
            dofs_tag = np.array(fem.locate_dofs_topological(V_u, 0, verts_tag), dtype=np.int32)
            tag_u_counts[tag] = int(dofs_tag.size)
        _emit(
            f"[diag] structural-only V_u coverage (dofs on facets): "
            f"tag1={tag_u_counts.get(1, 0)}, tag2={tag_u_counts.get(2, 0)}, tag3={tag_u_counts.get(3, 0)}",
            status_callback=status_callback,
        )
    except Exception as exc:
        _emit(f"[diag][warn] V_u coverage diagnostic failed: {exc}", status_callback=status_callback, level="warning")

    # BC/topology block: strict empty-array guards and fail-fast MPI abort on any C++ backend failure.
    try:
        def _safe_locate_topo(space, entity_dim: int, entities: np.ndarray, label: str) -> np.ndarray:
            # MPI alignment: always call into FEniCSx C++ even for empty local arrays.
            if entities is None:
                entities = np.array([], dtype=np.int32)
            entities = np.asarray(entities, dtype=np.int32)
            if msh.comm.rank == ROOT_RANK:
                builtins.print(f"--> [DEBUG] ENTERING locate_dofs_topological ({label})", flush=True)
                sys.stdout.flush()
            out = np.array(fem.locate_dofs_topological(space, entity_dim, entities), dtype=np.int32)
            if msh.comm.rank == ROOT_RANK:
                builtins.print(f"--> [DEBUG] EXITING locate_dofs_topological ({label})", flush=True)
                sys.stdout.flush()
            return out

        # Free–free structural diagnostic: no displacement Dirichlet BCs.
        bcs_u: List = []
        if msh.comm.rank == ROOT_RANK:
            print("[DIAG] structural-only: free–free (no displacement constraints).")
    except Exception as e:
        try:
            rank = int(msh.comm.rank)
            sys.stderr.write(f"[FATAL][Rank {rank}] topology/BC block failure: {e}\n")
            sys.stderr.flush()
        except Exception:
            pass
        msh.comm.Abort(1)
        raise

    # Collective-safe JIT form compilation with explicit cache dir.
    try:
        sys.stderr.write(f"[MPI-TRACE] Rank {msh.comm.rank} ABOUT_TO_CALL phase_sync 2101\n")
        sys.stderr.flush()
    except Exception:
        pass
    _phase_sync(2101, "structural-only before form JIT", status_callback=status_callback)
    jit_cache_dir = Path(
        config.get("solver", {}).get(
            "structural_jit_cache_dir",
            str((Path(__file__).resolve().parents[2] / ".ffcx_cache").resolve()),
        )
    )
    if MPI.COMM_WORLD.rank == 0:
        jit_cache_dir.mkdir(parents=True, exist_ok=True)
        if bool(config.get("solver", {}).get("structural_jit_clear_stale_lock", True)):
            for lock in jit_cache_dir.rglob("*.lock"):
                try:
                    lock.unlink()
                except Exception:
                    pass
    _phase_sync(2102, "structural-only after jit cache prep", status_callback=status_callback)
    jit_options = {"cache_dir": str(jit_cache_dir)}
    try:
        sys.stderr.write(f"[MPI-TRACE] Rank {msh.comm.rank} ENTERING fem.form(a_uu)\n")
        sys.stderr.flush()
    except Exception:
        pass
    a_uu_form = fem.form(a_uu, jit_options=jit_options)
    try:
        sys.stderr.write(f"[MPI-TRACE] Rank {msh.comm.rank} EXITING fem.form(a_uu)\n")
        sys.stderr.flush()
    except Exception:
        pass
    try:
        sys.stderr.write(f"[MPI-TRACE] Rank {msh.comm.rank} ENTERING fem.form(m_uu)\n")
        sys.stderr.flush()
    except Exception:
        pass
    m_uu_form = fem.form(m_uu, jit_options=jit_options)
    try:
        sys.stderr.write(f"[MPI-TRACE] Rank {msh.comm.rank} EXITING fem.form(m_uu)\n")
        sys.stderr.flush()
    except Exception:
        pass
    _phase_sync(2103, "structural-only after form JIT", status_callback=status_callback)

    _phase_sync(2105, "structural-only before matrix assembly", status_callback=status_callback)
    _debug_rank("Entering Matrix Assembly")
    K = assemble_matrix(a_uu_form, bcs=bcs_u); K.assemble()
    M = assemble_matrix(m_uu_form, bcs=bcs_u); M.assemble()
    # Defensive: allow new nonzeros during diagnostic diagonal updates.
    # With the ghost term, this should rarely be needed, but it prevents PETSc
    # from hard-failing if some diagonal entries were initially structurally zero.
    try:
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        M.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    except Exception:
        pass

    # Vacuum test: no air-diagonal penalty (assembly restricted to wood cells only).

    A = K.copy()
    eps_diag = float(config.get("solver", {}).get("structural_diag_shift", 1.0e-6))
    if eps_diag > 0:
        A.axpy(eps_diag, M, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        A.assemble()
        print(f"[DIAG] structural shift: A = K + {eps_diag:.2e} M")
        sys.stdout.flush()

    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # Target audible-range fundamentals via shift-and-invert.
    shift_hz = float(config.get("solver", {}).get("structural_shift_target_hz", config.get("solver", {}).get("shift_invert_target_hz", 100.0)))
    target_lambda = (2.0 * math.pi * shift_hz) ** 2
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(target_lambda)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    # Robustness: use configurable ST KSP/PC (iterative by default in VM).
    try:
        _debug_rank("Entering KSP Setup")
        ksp = st.getKSP()
        st_ksp_type = str(config.get("solver", {}).get("st_ksp_type", "cg"))
        st_pc_type = str(config.get("solver", {}).get("st_pc_type", "gamg"))
        ksp.setType(st_ksp_type)
        pc = ksp.getPC()
        pc.setType(st_pc_type)
        if st_pc_type.lower() == "lu":
            try:
                pc.setFactorSolverType(str(config.get("solver", {}).get("st_factor_solver_type", "mumps")))
                petsc_opts = PETSc.Options()
                petsc_opts["st_mat_mumps_icntl_14"] = int(config.get("solver", {}).get("mat_mumps_icntl_14", 500))
                petsc_opts["st_mat_mumps_icntl_6"] = int(config.get("solver", {}).get("mat_mumps_icntl_6", 7))
                petsc_opts["st_mat_mumps_icntl_12"] = int(config.get("solver", {}).get("mat_mumps_icntl_12", 1))
                petsc_opts["st_mat_mumps_icntl_24"] = int(config.get("solver", {}).get("mat_mumps_icntl_24", 1))
                petsc_opts["st_mat_mumps_icntl_4"] = int(config.get("solver", {}).get("mat_mumps_icntl_4", 0))
            except Exception:
                pass
        # Make KSP convergence checks essentially irrelevant.
        try:
            ksp.setTolerances(rtol=1.0e-50, atol=1.0e-50, max_it=1, divtol=1.0e50)
        except Exception:
            pass
    except Exception:
        pass
    nev_target = int(max(1, num_modes))
    # Request extra modes to skip rigid-body cluster near 0 Hz.
    nev = int(max(nev_target + 6, nev_target))
    eps.setDimensions(nev, int(max(40, 4 * nev)))
    eps.setTolerances(1e-6, 2000)
    eps.setFromOptions()
    _debug_petsc_comm("K", K)
    _debug_petsc_comm("M", M)
    _debug_petsc_comm("A", A)
    _debug_petsc_comm("EPS", eps)
    try:
        acomm = A.getComm()
        if acomm.getSize() != MPI.COMM_WORLD.size:
            raise RuntimeError(f"A communicator size mismatch: {acomm.getSize()} vs world {MPI.COMM_WORLD.size}")
    except Exception as exc:
        _emit(f"[error] communicator audit failed before structural EPS solve: {exc}", status_callback=status_callback, level="error")
        raise
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        try:
            k_norm = float(K.norm())
            m_norm = float(M.norm())
            ratio = (k_norm / m_norm) if m_norm > 0.0 else float("inf")
            print(
                f"[DIAG] Structural-only target: shift_hz={shift_hz:.3f}, "
                f"target_lambda={target_lambda:.6e}"
            )
            print(
                f"[DIAG] Matrix norms before EPS: ||K||={k_norm:.6e}, ||M||={m_norm:.6e}, "
                f"||K||/||M||={ratio:.6e}"
            )
            sys.stdout.flush()
        except Exception as exc:
            _emit(f"[diag][warn] failed to compute matrix norms: {exc}", status_callback=status_callback, level="warning")
    _phase_sync(2106, "structural-only before eps.solve", status_callback=status_callback)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print("🚀 Starting Production Run: Silent MUMPS, 450Hz Sweep, 100 Mode Quota.")
        sys.stdout.flush()
    opts = PETSc.Options()
    opts["eps_monitor"] = None
    opts["eps_converged_reason"] = None
    eps.setFromOptions()
    print(f"[HEARTBEAT] Rank {MPI.COMM_WORLD.rank} reached EPS Solve")
    sys.stdout.flush()
    _debug_rank("Entering EPS Solve")
    eps.solve()
    _phase_sync(2107, "structural-only after eps.solve", status_callback=status_callback)
    nconv = eps.getConverged()
    if nconv <= 0:
        raise RuntimeError("Structural-only diagnosis: no converged eigenpairs.")

    rvec = A.createVecRight()
    all_eigs: List[float] = []
    rigid_tol = float(config.get("solver", {}).get("structural_rigid_lambda_tol", 1.0e-10))
    min_structural_hz = float(config.get("solver", {}).get("structural_min_mode_hz", 10.0))
    skipped_rigid = 0
    skipped_low_hz = 0
    freqs_hz: List[float] = []
    vectors: List[np.ndarray] = []
    for i in range(min(nev, nconv)):
        eig = eps.getEigenpair(i, rvec)
        eig_r = float(np.real(eig))
        all_eigs.append(eig_r)
        if eig_r <= rigid_tol:
            skipped_rigid += 1
            continue
        f_hz = math.sqrt(eig_r) / (2.0 * math.pi)
        if f_hz < min_structural_hz:
            skipped_low_hz += 1
            continue
        freqs_hz.append(f_hz)
        vectors.append(rvec.array.copy())
    try:
        rvec.destroy()
    except Exception:
        pass
    eps.destroy()

    print(f"[DIAG] Structural-only eigen search: WHICH=SMALLEST_MAGNITUDE")
    print(
        f"[DIAG] Structural-only rigid filter: lambda_tol={rigid_tol:.3e}, "
        f"skipped_rigid={skipped_rigid}, min_mode_hz={min_structural_hz:.2f}, skipped_low_hz={skipped_low_hz}"
    )
    print(f"[DIAG] Raw eigenvalues: {[float(x) for x in all_eigs[:10]]}")
    sys.stdout.flush()

    if not freqs_hz:
        raise RuntimeError("Structural-only diagnosis: no positive eigenvalues.")
    order = np.argsort(np.array(freqs_hz))
    freqs_hz = [freqs_hz[int(i)] for i in order][:nev_target]
    eigvecs = np.stack([vectors[int(i)] for i in order[: len(freqs_hz)]], axis=1)

    print(f"[DIAG] Structural-only first {len(freqs_hz)} mode(s): {[round(f, 3) for f in freqs_hz]}")
    if freqs_hz:
        print(f"[DIAG] Structural-only first mode: {freqs_hz[0]:.3f} Hz")
    sys.stdout.flush()
    n_u = int(V_u.dofmap.index_map.size_local * V_u.dofmap.index_map_bs)
    try:
        A.destroy()
        K.destroy()
        M.destroy()
    except Exception:
        pass
    return msh, V_u, freqs_hz, eigvecs, n_u, 0


def _solve_coupled_evp(
    mesh_file: Path,
    config: Dict,
    num_modes: int,
    status_callback=None,
    solve_evp: bool = True,
):
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print("🔴🔴🔴 FORCE-RUNNING FULL COUPLED MODEL - NO DIAGNOSIS 🔴🔴🔴")
        sys.stdout.flush()
    _phase_sync(2000, "coupled enter", status_callback=status_callback)
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
    _audit_and_scale_mesh_units(msh, config, status_callback=status_callback)
    _mesh_interface_diagnostic(msh, cell_tags, facet_tags, status_callback=status_callback)
    _phase_sync(2001, "coupled after mesh load", status_callback=status_callback)
    _sod_raw = config.get("solver", {}).get("structural_only_diagnosis", "<missing>")
    _sod_eff = _solver_bool(config.get("solver", {}), "structural_only_diagnosis", default=False)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(
            f"[diag] structural_only_diagnosis: effective={_sod_eff!r} raw={_sod_raw!r} "
            f"solve_evp={solve_evp} → {'structural-only branch' if (solve_evp and _sod_eff) else 'full coupled (mixed u,p)'}"
        )
        sys.stdout.flush()
    if solve_evp and _sod_eff:
        return _solve_structural_only_evp(
            msh=msh,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            config=config,
            num_modes=max(1, int(config.get("solver", {}).get("structural_only_num_modes", 30))),
            status_callback=status_callback,
        )
    coords = msh.geometry.x
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
    # Hard-coded P1+P1 (ignore any future config-based FE order): minimizes global DOFs on this mesh.
    _u_deg_coupled = 1
    _p_deg_coupled = 1
    u_el = element("Lagrange", msh.basix_cell(), _u_deg_coupled, shape=(3,))
    p_el = element("Lagrange", msh.basix_cell(), _p_deg_coupled)
    W_el = mixed_element([u_el, p_el])
    W = fem.functionspace(msh, W_el)
    n_p_global = int(W.sub(1).dofmap.index_map.size_global * W.sub(1).dofmap.index_map_bs)
    n_u_global = int(W.sub(0).dofmap.index_map.size_global * W.sub(0).dofmap.index_map_bs)
    _deg_u_dbg = getattr(u_el, "degree", None)
    _deg_p_dbg = getattr(p_el, "degree", None)
    try:
        _du = _deg_u_dbg() if callable(_deg_u_dbg) else int(_deg_u_dbg)
    except Exception:
        _du = int(W.sub(0).element.basix_element.degree)
    try:
        _dp = _deg_p_dbg() if callable(_deg_p_dbg) else int(_deg_p_dbg)
    except Exception:
        _dp = int(W.sub(1).element.basix_element.degree)
    try:
        print(f"DEBUG: Degree for u is {u_el.degree()} and for p is {p_el.degree()}")
    except Exception:
        print(f"DEBUG: Degree for u is {_du} and for p is {_dp} (u_el.degree()/p_el.degree() unavailable)")
    print(
        f"DEBUG: mesh geometry nodes={msh.geometry.x.shape[0]}, "
        f"global u DOFs={n_u_global}, global p DOFs={n_p_global}, "
        f"mixed W global={n_u_global + n_p_global}"
    )
    _emit(
        f"[form] mixed P{_u_deg_coupled}+P{_p_deg_coupled}: global u DOFs n_u={n_u_global}, p DOFs n_p={n_p_global} "
        f"(acoustic forms use dx on air cells only, tag={AIR_VOLUME_TAG}).",
        status_callback=status_callback,
    )

    # Sub-space arguments must use TrialFunctions/TestFunctions (not ufl.split on W)
    # or dolfinx assembles each block only on its diagonal — FSI off-diagonal nnz stays 0.
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

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

    top_m, back_m, t_top, t_back = _split_wood_materials(config)
    air_mat = config["materials"]["air"]
    rho_air = float(air_mat["density"])
    c_air = float(air_mat["speed_of_sound"])
    top_mat = config["materials"]["top"]
    back_mat = config["materials"]["back"]
    _emit(
        f"[diag] material sanity (tag1 top / tag3 back, orthotropic KL): "
        f"top D11={top_m['D11']:.6e} E_L={top_m['E_L']:.6e} rho={top_m['rho']:.6e} | "
        f"back D11={back_m['D11']:.6e} E_L={back_m['E_L']:.6e} rho={back_m['rho']:.6e} | "
        f"rho_air={rho_air:.6e} t_top={t_top:.6e} m t_back={t_back:.6e} m | "
        f"{top_mat.get('name', '')!r} / {back_mat.get('name', '')!r}",
        status_callback=status_callback,
    )

    tag_top = int(WOOD_SURFACE_TAGS[0])
    tag_back = int(WOOD_SURFACE_TAGS[1])
    tag_ribs = int(WOOD_SURFACE_TAGS[2])

    def eps_surface(uu):
        grad_u = ufl.grad(uu)
        grad_tan = P * grad_u * P
        return 0.5 * (grad_tan + ufl.transpose(grad_tan))

    wood_tag_top = int(np.sum(facet_tags.values == tag_top))
    wood_tag_back = int(np.sum(facet_tags.values == tag_back))
    wood_tag_ribs = int(np.sum(facet_tags.values == tag_ribs))
    ds_top = xdmf_ds(tag_top)
    ds_back = xdmf_ds(tag_back)
    ds_ribs = xdmf_ds(tag_ribs)
    if wood_tag_top + wood_tag_back + wood_tag_ribs > 0:
        wood_ds = ds_top + ds_back + ds_ribs
        _emit(
            f"[form] structural shell integration on tagged facets: "
            f"tag{tag_top}={wood_tag_top} (top), tag{tag_back}={wood_tag_back} (back), "
            f"tag{tag_ribs}={wood_tag_ribs} (ribs/sides)",
            status_callback=status_callback,
        )
    else:
        wood_ds = ufl.ds(domain=msh)
        ds_top = wood_ds
        ds_back = wood_ds
        ds_ribs = wood_ds
        _emit(
            "[form][warn] structural facet tags missing; falling back to all exterior facets (ds).",
            status_callback=status_callback,
            level="warning",
        )

    eps_u = eps_surface(u)
    eps_v = eps_surface(v)
    w_n = ufl.dot(u, n)
    v_n = ufl.dot(v, n)
    e1, e2 = _plate_local_frame(n, P)

    shell_top = _orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
    shell_back = _orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
    shell_ribs = _orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)

    if wood_tag_top + wood_tag_back + wood_tag_ribs > 0:
        a_uu = shell_top * ds_top + shell_back * ds_back + shell_ribs * ds_ribs
    else:
        a_uu = (shell_top + shell_back + shell_ribs) * wood_ds

    _diagnose_shell_stiffness_assembly(
        a_uu,
        shell_top,
        shell_back,
        ds_top,
        ds_back,
        tag_top,
        tag_back,
        wood_tag_top,
        wood_tag_back,
        status_callback=status_callback,
    )

    # Pressure DOF similarity scale: balances u (~1e9) and acoustic (~1e1) block magnitudes.
    p_scale = _coupled_pressure_dof_scale(config.get("solver", {}))
    p2 = p_scale * p_scale
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(f"[form] coupled pressure_dof_scale={p_scale:.4e} (similarity on all p-blocks)")
        sys.stdout.flush()

    # Acoustic stiffness in internal air volume.
    a_pp = p2 * (1.0 / rho_air) * ufl.inner(ufl.grad(p), ufl.grad(q)) * xdmf_dx(AIR_VOLUME_TAG)

    # FSI interface (wood_ds): trial/test must span different sub-spaces for off-diagonal nnz.
    # Stiffness — fluid pressure traction on structure: trial p, test v  → block (u, p).
    a_up = -p_scale * p * ufl.dot(n, v) * wood_ds

    # Acoustic mass and structure mass (per facet tag).
    if wood_tag_top + wood_tag_back + wood_tag_ribs > 0:
        m_uu = (
            (top_m["rho"] * t_top) * ufl.dot(u, v) * ds_top
            + (back_m["rho"] * t_back) * ufl.dot(u, v) * ds_back
            + (back_m["rho"] * t_back) * ufl.dot(u, v) * ds_ribs
        )
    else:
        m_uu = (top_m["rho"] * t_top + back_m["rho"] * t_back) * ufl.dot(u, v) * wood_ds
    m_pp = p2 * (1.0 / (rho_air * c_air * c_air)) * p * q * xdmf_dx(AIR_VOLUME_TAG)

    # Mass — structural normal acceleration drives acoustic: trial u, test q  → block (p, u).
    m_pu = p_scale * rho_air * ufl.dot(u, n) * q * wood_ds

    # Pressure-only regularization (optional); displacement is free–free (no reg_u).
    diag_shift = float(config.get("solver", {}).get("diag_shift", 0.0))
    reg_p = p2 * diag_shift * p * q * xdmf_dx(AIR_VOLUME_TAG)

    a_form = a_uu + a_pp + a_up + reg_p
    m_form = m_uu + m_pp + m_pu

    # Per-facet-group shell mass forms for plate-specific sifter (Top tag 1, Body tag 3).
    m_uu_top_plate = (top_m["rho"] * t_top) * ufl.dot(u, v) * xdmf_ds(tag_top)
    m_uu_back_shell = (back_m["rho"] * t_back) * ufl.dot(u, v) * xdmf_ds(tag_back)
    m_uu_ribs_shell = (back_m["rho"] * t_back) * ufl.dot(u, v) * xdmf_ds(tag_ribs)
    has_top_plate_facets = wood_tag_top > 0
    has_back_shell_facets = wood_tag_back > 0
    has_ribs_facets = wood_tag_ribs > 0

    # Lumped masses consistent with m_uu (surface shell) and air volume (tag 10), before EVP solve.
    _wood_mass_note = (
        "Top+Back+Ribs facet tags (1/3/4)"
        if (wood_tag_top + wood_tag_back + wood_tag_ribs) > 0
        else "WARNING: full exterior ds (wood facet tags missing)"
    )
    mass_top_kg = float("nan")
    mass_back_kg = float("nan")
    try:
        mass_air_kg = float(fem.assemble_scalar(fem.form(rho_air * xdmf_dx(AIR_VOLUME_TAG))))
    except Exception as exc:
        mass_air_kg = float("nan")
        _emit(f"[diag] air mass integral failed: {exc}", status_callback=status_callback, level="warning")
    try:
        mass_wood_kg = float(
            fem.assemble_scalar(
                fem.form(
                    top_m["rho"] * t_top * ds_top
                    + back_m["rho"] * t_back * ds_back
                    + back_m["rho"] * t_back * ds_ribs
                )
            )
        )
        mass_top_kg = float(fem.assemble_scalar(fem.form(top_m["rho"] * t_top * ds_top)))
        mass_back_kg = float(
            fem.assemble_scalar(fem.form(back_m["rho"] * t_back * (ds_back + ds_ribs)))
        )
    except Exception as exc:
        mass_wood_kg = float("nan")
        mass_top_kg = float("nan")
        mass_back_kg = float("nan")
        _emit(f"[diag] wood shell mass integral failed: {exc}", status_callback=status_callback, level="warning")
    print(
        f"[DIAG] Total wood mass (integral rho*thickness per tags 1/3/4 shell ds; {_wood_mass_note}): "
        f"{mass_wood_kg:.6e} kg | mass_top={mass_top_kg:.6e} kg mass_back={mass_back_kg:.6e} kg"
    )
    print(f"[DIAG] Total air mass (integral rho_air over air volume tag {AIR_VOLUME_TAG}): {mass_air_kg:.6e} kg")
    if math.isfinite(mass_air_kg) and math.isfinite(mass_wood_kg) and mass_wood_kg > 0:
        print(f"[DIAG] Air mass / wood mass ratio: {mass_air_kg / mass_wood_kg:.3e}")
    elif math.isfinite(mass_air_kg) and math.isfinite(mass_wood_kg) and mass_wood_kg <= 0:
        print("[DIAG] Air mass / wood mass ratio: undefined (wood mass <= 0)")
    sys.stdout.flush()

    # Release no-longer-needed symbolic temporaries once forms are finalized.
    del eps_u, eps_v, w_n, v_n, wood_tag_top, wood_tag_back, wood_tag_ribs

    # Dirichlet BCs: soundhole pressure gauge + clamped ribs (tag 4); top/back remain free.
    soundhole_facets = np.array(facet_tags.find(2), dtype=np.int32)
    pressure_gauge = str(config.get("solver", {}).get("pressure_gauge", "soundhole")).lower()
    bcs = []
    p_dofs = np.array([], dtype=np.int32)
    try:
        V_p, _ = W.sub(1).collapse()

        n_p_collapsed = int(V_p.dofmap.index_map.size_global * V_p.dofmap.index_map_bs)

        coords = msh.geometry.x
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        tol_p = max(1.0e-9, 1.0e-5 * max(1.0, diag))

        if pressure_gauge == "soundhole":
            p_dofs = np.array(
                fem.locate_dofs_topological(V_p, fdim, soundhole_facets),
                dtype=np.int32,
            )
            _emit(
                f"[bc] pressure gauge: P=0 on soundhole facets (count={p_dofs.size}).",
                status_callback=status_callback,
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

        if p_dofs.size == 0:
            p_anchor = coords[np.argmin(np.linalg.norm(coords - mins, axis=1))]
            tol = max(1.0e-12, 1.0e-8 * max(1.0, diag))

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

        if p_dofs.size == 0:
            raise RuntimeError("Failed to create pressure grounding dofs (p_dofs is empty).")

        V_u, _ = W.sub(0).collapse()
        facets_ribs = np.array(facet_tags.find(RIBS_SURFACE_TAG), dtype=np.int32)
        u_dofs_ribs = np.array([], dtype=np.int32)
        if facets_ribs.size > 0:
            u_dofs_ribs = np.array(
                fem.locate_dofs_topological(V_u, fdim, facets_ribs),
                dtype=np.int32,
            )
        _emit(
            "[bc][diag] pressure gauge + ribs clamp (tag 4). "
            f"pressure BC dof count={p_dofs.size} (full pressure FE unknowns=n_p_collapsed={n_p_collapsed}), "
            f"ribs facets={facets_ribs.size}, ribs u_dof count={u_dofs_ribs.size}, "
            f"soundhole_facets.shape={soundhole_facets.shape}",
            status_callback=status_callback,
        )

        p_zero = fem.Constant(msh, PETSc.ScalarType(0.0))
        bc_p = fem.dirichletbc(p_zero, p_dofs, V_p)
        bcs = [bc_p]
        if u_dofs_ribs.size > 0:
            u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
            bc_u = fem.dirichletbc(u_zero, u_dofs_ribs, V_u)
            bcs.append(bc_u)
            _emit(
                f"[bc] ribs clamp: u = 0 on tag {RIBS_SURFACE_TAG} "
                f"({facets_ribs.size} facets, {u_dofs_ribs.size} displacement DOFs).",
                status_callback=status_callback,
            )
        else:
            _emit(
                f"[bc][warn] no facets on tag {RIBS_SURFACE_TAG}; ribs not clamped.",
                status_callback=status_callback,
                level="warning",
            )
    except Exception as e:
        _emit(
            "[bc][error] dirichletbc creation failed. "
            f"p_dofs.dtype={p_dofs.dtype}, p_dofs.shape={p_dofs.shape}, "
            f"soundhole_facets.dtype={soundhole_facets.dtype}, "
            f"soundhole_facets.shape={soundhole_facets.shape}, "
            f"error={e}",
            status_callback=status_callback,
            level="error",
        )
        raise

    _phase_sync(2002, "coupled before matrix assembly", status_callback=status_callback)
    _debug_rank("Entering Matrix Assembly")
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print("PRINT: ENTERING FULL COUPLED ACOUSTIC-STRUCTURAL SOLVE")
        sys.stdout.flush()
    A = assemble_matrix(fem.form(a_form), bcs=bcs)
    A.assemble()
    M = assemble_matrix(fem.form(m_form), bcs=bcs)
    M.assemble()

    solver_cfg = config.get("solver", {})
    gnhep_scale = _normalize_assembled_gnhep(A, M, solver_cfg)
    if MPI.COMM_WORLD.rank == ROOT_RANK and gnhep_scale != 1.0:
        print(
            f"[form] GNHEP Frobenius normalization: scale={gnhep_scale:.6e} "
            f"(||A||_F={float(A.norm()):.6e}, ||M||_F={float(M.norm()):.6e})"
        )
        sys.stdout.flush()
    use_sifter = _solver_bool(solver_cfg, "adaptive_mode_sifter", default=True)
    M_top: Optional[PETSc.Mat] = None
    M_back: Optional[PETSc.Mat] = None
    if solve_evp and (use_sifter or config.get("_worker_target_hz") is not None):
        if has_top_plate_facets:
            M_top = assemble_matrix(fem.form(m_uu_top_plate), bcs=bcs)
            M_top.assemble()
        if has_back_shell_facets:
            M_back = assemble_matrix(fem.form(m_uu_back_shell), bcs=bcs)
            M_back.assemble()

    if not solve_evp:
        return msh, W, A, M

    # Release form objects before eigensolve; matrices are already assembled.
    del a_form, m_form, a_uu, a_pp, a_up, m_uu, m_pp, m_pu, m_uu_top_plate, m_uu_back_shell, reg_p
    gc.collect()

    _emit("Step 3/5: Solving generalized EVP with SLEPc...", status_callback=status_callback)
    n_dofs = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)
    n_u_fe = int(W.sub(0).dofmap.index_map.size_global * W.sub(0).dofmap.index_map_bs)
    n_p_fe = int(W.sub(1).dofmap.index_map.size_global * W.sub(1).dofmap.index_map_bs)
    print(
        f"[DIAG] Final u_dofs={n_u_fe} p_dofs={n_p_fe} "
        f"(free–free structure; pressure gauge p_bc={p_dofs.size})"
    )
    sys.stdout.flush()
    print(f"Starting solver with {n_dofs} DOFs and proactive memory cleanup...")
    sys.stdout.flush()
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print("🚀 Starting Production Run: Silent MUMPS, 450Hz Sweep, 100 Mode Quota.")
        sys.stdout.flush()

    min_valid_hz = float(solver_cfg.get("min_valid_mode_hz", 50.0))
    max_valid_hz = float(solver_cfg.get("max_valid_mode_hz", 1000.0))
    work = M.createVecRight()

    _worker_hz = config.get("_worker_target_hz")
    if _worker_hz is not None:
        batch_w = max(1, int(config.get("_worker_num_modes", 80)))
        max_it = int(
            config.get(
                "_worker_eps_max_it",
                int(solver_cfg.get("eigs_maxiter", solver_cfg.get("eps_max_it", 3000))),
            )
        )
        if M_top is None and M_back is None:
            raise RuntimeError(
                "Worker single-shift batch requires shell mass matrices M_top/M_back (facet tags 1 and 3). "
                "Check mesh facet tagging."
            )
        _phase_sync(2099, "worker single-shift before batch", status_callback=status_callback)
        nconv, rows = _slepc_shift_invert_batch(
            A,
            M,
            solver_cfg,
            float(_worker_hz),
            batch_w,
            diag_shift,
            status_callback,
            M_top=M_top,
            M_back=M_back,
            work=work,
            mass_top=mass_top_kg,
            mass_back=mass_back_kg,
            eps_max_it_cap=max_it,
        )
        _emit(
            f"[worker] shift @ {float(_worker_hz):.4f} Hz: nconv={nconv}, usable_rows={len(rows)}",
            status_callback=status_callback,
        )
        _phase_sync(2100, "worker single-shift after batch", status_callback=status_callback)
        row_meta: List[Tuple[float, np.ndarray, float, float]] = []
        for f_hz, vec, rt, rb in rows:
            if rt is None or rb is None:
                continue
            row_meta.append((float(f_hz), np.asarray(vec, dtype=np.float64), float(rt), float(rb)))
        if not row_meta:
            if M_top is not None:
                M_top.destroy()
            if M_back is not None:
                M_back.destroy()
            try:
                work.destroy()
            except Exception:
                pass
            try:
                A.destroy()
                M.destroy()
            except Exception:
                pass
            raise RuntimeError(
                f"[worker] No usable modes at {float(_worker_hz):.4f} Hz (convergence or plate-energy ratios missing)."
            )

        row_meta.sort(key=lambda t: t[0])
        freqs_hz = [t[0] for t in row_meta]
        vectors = [t[1] for t in row_meta]
        config["_worker_tag1"] = [t[2] for t in row_meta]
        config["_worker_tag3"] = [t[3] for t in row_meta]
        eigvecs = np.stack(vectors, axis=1)

        config["_fom_sifter_stats"] = {"worker_single_batch": True, "nconv": int(nconv), "rows": int(len(rows))}
        config["_fom_uniqueness_scores"] = []
        config["_fom_participation_ratios"] = []

        if M_top is not None:
            M_top.destroy()
        if M_back is not None:
            M_back.destroy()

        n_u = int(W.sub(0).dofmap.index_map.size_local * W.sub(0).dofmap.index_map_bs)
        n_p = int(W.sub(1).dofmap.index_map.size_local * W.sub(1).dofmap.index_map_bs)
        try:
            work.destroy()
        except Exception:
            pass
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass
        return msh, W, freqs_hz, eigvecs, n_u, n_p

    if use_sifter and (M_top is not None or M_back is not None):
        quota = int(solver_cfg.get("sifter_quota", 100))
        batch = int(solver_cfg.get("sifter_batch_modes", 50))
        f_center = float(solver_cfg.get("sifter_start_hz", 100.0))
        f_cap = float(solver_cfg.get("sifter_max_hz", 450.0))
        df_s = float(solver_cfg.get("sifter_step_hz", 10.0))
        low_f_min = float(solver_cfg.get("sifter_low_freq_min_hz", 90.0))
        low_f_max = float(solver_cfg.get("sifter_low_freq_max_hz", 160.0))
        low_step_hz = float(solver_cfg.get("sifter_low_step_hz", 15.0))
        high_step_hz = float(solver_cfg.get("sifter_high_step_hz", 15.0))
        if "sifter_low_target" not in solver_cfg or "sifter_high_target" not in solver_cfg:
            raise RuntimeError(
                "Missing required solver keys: 'sifter_low_target' and/or 'sifter_high_target'. "
                "Refusing to fall back to stale defaults."
            )
        low_batch = int(solver_cfg["sifter_low_target"])
        high_batch = int(solver_cfg["sifter_high_target"])
        max_iter_cap = int(solver_cfg.get("sifter_batch_max_it", 50))
        adaptive_break = float(solver_cfg.get("sifter_adaptive_break_hz", 200.0))
        dup_hz_low = float(solver_cfg.get("sifter_dup_hz", 2.0))
        dup_hz_high = float(solver_cfg.get("sifter_dup_hz_high_band", 0.8))
        uniq_min_low = float(solver_cfg.get("sifter_uniqueness_min", 0.1))
        uniq_min_high = float(solver_cfg.get("sifter_uniqueness_min_high_band", 0.06))
        near_hz = float(solver_cfg.get("sifter_energy_priority_hz", 2.0))
        wood_gate_low = float(solver_cfg.get("min_wood_participation", 0.08))
        wood_gate_high = float(solver_cfg.get("min_wood_participation_high_band", 0.08))
        max_acoustic_only_keep = int(solver_cfg.get("max_acoustic_only_modes", 3))
        profile_name = str(solver_cfg.get("solver_profile_name", "default"))
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
        harvested_freqs: List[float] = []
        sifter_stats = {
            "modes_discarded_by_dup_hz": 0,
            "modes_discarded_by_plate_energy": 0,
            "modes_discarded_by_band": 0,
            "modes_discarded_by_wood": 0,
            "modes_discarded_by_uniqueness": 0,
            "total_raw_candidates_seen": 0,
            "stage1_candidates_logged": 0,
            "stage2_after_wood_gate": 0,
            "stage2_after_cluster": 0,
            "stage2_after_uniqueness": 0,
            "accepted_modes": 0,
        }
        weak_step_hz = float(solver_cfg.get("sifter_stage1_step_hz", 10.0))
        weak_batch = int(solver_cfg.get("sifter_stage1_batch_size", 60))
        weak_min_wood = float(solver_cfg.get("sifter_stage1_min_wood_participation", 0.0001))
        weak_uniqueness_min = float(solver_cfg.get("sifter_stage1_uniqueness_min", 0.01))
        weak_dup_hz = float(solver_cfg.get("sifter_stage1_dup_hz", 0.5))
        global_dup_hz = float(solver_cfg.get("sifter_stage2_global_dup_hz", 2.0))
        global_uniqueness_min = float(solver_cfg.get("sifter_stage2_global_uniqueness_min", 0.10))
        global_wood_low = float(solver_cfg.get("sifter_stage2_min_wood_low", 0.01))
        global_wood_high = float(solver_cfg.get("sifter_stage2_min_wood_high", 0.08))
        candidate_id = 0
        harvested_meta: List[Dict] = []

        def _dup_weak(freq: float) -> bool:
            return any(abs(freq - fs) < weak_dup_hz for fs in harvested_freqs)

        def _weak_uniqueness_from_disk(vec: np.ndarray, freq_hz: float) -> float:
            """Compute weak-stage structural uniqueness against already harvested disk vectors."""
            if MPI.COMM_WORLD.rank != ROOT_RANK:
                return 1.0
            if not harvested_meta:
                return 1.0
            v = csr_u_slice(dense_to_csr_f32_column(vec), n_u_fe)
            nv = csr_col_norm(v)
            if nv <= 0.0:
                return 1.0
            # Compare to nearest prior harvested frequencies to keep disk IO bounded.
            nearest = sorted(harvested_meta, key=lambda m: abs(float(m["hz"]) - float(freq_hz)))[:24]
            max_ov = 0.0
            for m in nearest:
                try:
                    vp = Path(str(m["vector_path"]))
                    if not vp.is_absolute():
                        vp = SORTING_ROOT / vp
                    prev = load_mode_column_any(vp)
                except Exception:
                    continue
                p = csr_u_slice(prev, n_u_fe)
                max_ov = max(max_ov, csr_normalized_overlap(v, p))
            return 1.0 - max_ov

        def _mean_disp_energy(vec: np.ndarray) -> float:
            # Mixed eigenvector = [u, p]. "Mean Displacement" should use displacement block only.
            u = np.asarray(vec[:n_u_fe])
            return float(np.mean(np.abs(u))) if u.size else 0.0

        def _structural_uniqueness_vs_saved(vec: np.ndarray) -> float:
            """
            Double-gate spatial term: 1 - max_j |cos(u, u_j)| over all accepted structural blocks.
            Require >= uniq_min (e.g. 0.12) so max overlap <= 0.88 vs any saved mode.
            """
            v = np.asarray(vec[:n_u_fe], dtype=np.float64)
            nv = float(np.linalg.norm(v))
            if nv <= 0.0 or not saved_vecs:
                return 1.0
            max_ov = 0.0
            for prev in saved_vecs:
                p = np.asarray(prev[:n_u_fe], dtype=np.float64)
                np_ = float(np.linalg.norm(p))
                if np_ <= 0.0:
                    continue
                ov = abs(float(np.vdot(v, p))) / (nv * np_)
                max_ov = max(max_ov, float(np.clip(ov, 0.0, 1.0)))
            return 1.0 - max_ov

        def _participation(vec: np.ndarray) -> Tuple[float, float]:
            v = np.asarray(vec)
            u = np.asarray(v[:n_u_fe])
            p = np.asarray(v[n_u_fe:])
            nu = float(np.linalg.norm(u))
            np_ = float(np.linalg.norm(p))
            tot = max(nu + np_, 1e-30)
            return nu / tot, np_ / tot

        _prepare_sorting_workspace()
        _emit(
            f"[sifter] two-stage pipeline enabled: Stage-1 Harvest (step={weak_step_hz:.0f}Hz, batch={weak_batch}, "
            f"weak wood>={weak_min_wood:g}, weak uniq>={weak_uniqueness_min:.2f}, weak dup={weak_dup_hz:.2f}Hz) -> "
            f"Stage-2 Global Curation (wood low/high={global_wood_low:.2f}/{global_wood_high:.2f}, "
            f"dup cluster={global_dup_hz:.2f}Hz, uniq>={global_uniqueness_min:.2f}, quota={quota}).",
            status_callback=status_callback,
        )
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(f"Using Solver Profile: {profile_name} - Targets: {low_batch}/{high_batch}")
            print(
                f"AUDIT_LOG | Targets: {low_batch}/{high_batch} | "
                f"Adaptive@{adaptive_break:.0f}Hz: dup {dup_hz_low:.2f}/{dup_hz_high:.2f}Hz, "
                f"uniq {uniq_min_low:.2f}/{uniq_min_high:.2f}, wood {wood_gate_low:.2f}/{wood_gate_high:.2f} | "
                f"near_win={near_hz:.2f}Hz"
            )
            sys.stdout.flush()

        while f_center <= f_cap + 1e-9:
            in_low_gear = low_f_min <= f_center <= low_f_max
            batch = weak_batch
            df_s = weak_step_hz
            _phase_sync(2003, "coupled sifter before batch solve", status_callback=status_callback)
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
                mass_top=mass_top_kg,
                mass_back=mass_back_kg,
                eps_max_it_cap=max_iter_cap,
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
            print(
                f"[DIAG] Batch {f_center:.1f}Hz - stage1 weak thresholds: dup_hz={weak_dup_hz:.2f}, "
                f"uniq_min={weak_uniqueness_min:.2f}, wood_min={weak_min_wood:g} | "
                f"stage2 gates: wood_low/high={global_wood_low:.2f}/{global_wood_high:.2f}, "
                f"dup_cluster={global_dup_hz:.2f}, uniq_global={global_uniqueness_min:.2f} | "
                f"Top Ratios: [{top_str}]"
            )
            sys.stdout.flush()

            sifter_stats["total_raw_candidates_seen"] += int(len(rows))
            added = 0
            dropped_unique = 0
            dropped_participation = 0
            for f_hz, vec, rt, rb in rows:
                if rt is None or rb is None:
                    continue
                if not (rt > th_top or rb > th_back):
                    sifter_stats["modes_discarded_by_plate_energy"] += 1
                    continue
                if not (min_valid_hz <= f_hz <= max_valid_hz):
                    sifter_stats["modes_discarded_by_band"] += 1
                    continue
                if _dup_weak(f_hz):
                    sifter_stats["modes_discarded_by_dup_hz"] += 1
                    continue
                wood_part, air_part = _participation(vec)
                wood_participation_metric = max(0.0, float(rt) + float(rb))
                if wood_participation_metric < weak_min_wood:
                    dropped_participation += 1
                    sifter_stats["modes_discarded_by_wood"] += 1
                    continue
                cur_f = float(f_hz)
                uniq_cur = _weak_uniqueness_from_disk(vec, cur_f)
                if uniq_cur < weak_uniqueness_min:
                    dropped_unique += 1
                    sifter_stats["modes_discarded_by_uniqueness"] += 1
                    continue
                harvested_freqs.append(cur_f)
                if MPI.COMM_WORLD.rank == ROOT_RANK:
                    vec_path = SORTING_TEMP_MODES / f"mode_{candidate_id:06d}{MODE_VECTOR_FILE_SUFFIX}"
                    save_mode_csr(vec_path, dense_to_csr_f32_column(vec))
                    rec = {
                        "id": int(candidate_id),
                        "hz": float(cur_f),
                        "tag1_ratio": float(rt),
                        "tag3_ratio": float(rb),
                        "uniqueness": float(uniq_cur),
                        "wood_participation": float(wood_participation_metric),
                        "vector_path": str(vec_path),
                    }
                    _append_candidate_metadata(rec)
                    harvested_meta.append(rec)
                candidate_id += 1
                sifter_stats["stage1_candidates_logged"] += 1
                added += 1
            _emit(
                f"[sifter] center={f_center:.1f} Hz [{('low' if in_low_gear else 'high')} gear: "
                f"step={df_s:.0f}Hz, batch={batch}, max_it<={max_iter_cap}] "
                f"weak-thresholds(dup={weak_dup_hz:.2f}, uniq>={weak_uniqueness_min:.2f}, wood>={weak_min_wood:g}): "
                f"converged={nconv}, harvested+{added}, unique-dropped={dropped_unique}, "
                f"participation-dropped={dropped_participation} (total stage1={sifter_stats['stage1_candidates_logged']}).",
                status_callback=status_callback,
            )
            _phase_sync(2004, "coupled sifter after batch solve", status_callback=status_callback)
            f_center += df_s

        _phase_sync(2007, "coupled sifter global curation begin", status_callback=status_callback)
        selected_records: Optional[List[Dict]] = None
        selected_vectors: Optional[List[np.ndarray]] = None
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            with open(SORTING_LOG, "r", encoding="utf-8") as f:
                payload = json.load(f)
            candidates = list(payload.get("candidates", []))
            if not candidates:
                candidates = []
            stage2_wood = []
            for c in candidates:
                hz = float(c.get("hz", 0.0))
                wood = float(c.get("wood_participation", 0.0))
                wood_gate = global_wood_low if hz < adaptive_break else global_wood_high
                if wood >= wood_gate:
                    stage2_wood.append(c)
            sifter_stats["stage2_after_wood_gate"] = int(len(stage2_wood))
            stage2_wood.sort(key=lambda x: float(x.get("hz", 0.0)))
            clustered: List[List[Dict]] = []
            cur_cluster: List[Dict] = []
            for c in stage2_wood:
                hz = float(c.get("hz", 0.0))
                if not cur_cluster:
                    cur_cluster = [c]
                else:
                    prev_hz = float(cur_cluster[-1].get("hz", 0.0))
                    if abs(hz - prev_hz) <= global_dup_hz:
                        cur_cluster.append(c)
                    else:
                        clustered.append(cur_cluster)
                        cur_cluster = [c]
            if cur_cluster:
                clustered.append(cur_cluster)
            clustered_winners = []
            for cl in clustered:
                best = max(cl, key=lambda x: float(x.get("wood_participation", 0.0)))
                clustered_winners.append(best)
            sifter_stats["stage2_after_cluster"] = int(len(clustered_winners))
            clustered_winners.sort(
                key=lambda x: (
                    float(x.get("wood_participation", 0.0)),
                    max(float(x.get("tag1_ratio", 0.0)), float(x.get("tag3_ratio", 0.0))),
                ),
                reverse=True,
            )

            selected_records = []
            selected_vectors = []
            for c in clustered_winners:
                vp = Path(str(c["vector_path"]))
                if not vp.is_absolute():
                    vp = SORTING_ROOT / vp
                vec_csr = load_mode_column_any(vp)
                vu = csr_u_slice(vec_csr, n_u_fe)
                uniq_global = 1.0
                if selected_vectors:
                    uniq_global = 1.0 - max(
                        csr_normalized_overlap(vu, csr_u_slice(prev, n_u_fe)) for prev in selected_vectors
                    )
                if uniq_global < global_uniqueness_min:
                    continue
                rec = dict(c)
                rec["uniqueness"] = float(uniq_global)
                selected_records.append(rec)
                selected_vectors.append(vec_csr)
                if len(selected_records) >= quota:
                    break
            sifter_stats["stage2_after_uniqueness"] = int(len(selected_records))

        selected_records = MPI.COMM_WORLD.bcast(selected_records, root=ROOT_RANK)
        selected_vectors = MPI.COMM_WORLD.bcast(selected_vectors, root=ROOT_RANK)
        _phase_sync(2008, "coupled sifter global curation done", status_callback=status_callback)

        if not selected_records or not selected_vectors:
            if M_top is not None:
                M_top.destroy()
            if M_back is not None:
                M_back.destroy()
            raise RuntimeError(
                "Two-stage sifter found no globally curated modes. "
                "Try lowering stage2 wood/uniqueness gates or widening the sweep."
            )

        order = np.argsort(np.array([float(r["hz"]) for r in selected_records], dtype=np.float64))
        freqs_hz = [float(selected_records[int(i)]["hz"]) for i in order]
        ordered_csr = [selected_vectors[int(i)] for i in order]
        uniqueness_scores = [float(selected_records[int(i)].get("uniqueness", 1.0)) for i in order]
        participation_ratios = [float(selected_records[int(i)].get("wood_participation", 0.0)) for i in order]
        eigvecs = np.asarray(sp_sparse.hstack(ordered_csr, format="csr").toarray(), dtype=np.float64)
        sifter_stats["accepted_modes"] = int(eigvecs.shape[1])
        config["_fom_sifter_stats"] = dict(sifter_stats)
        config["_fom_uniqueness_scores"] = list(uniqueness_scores)
        config["_fom_participation_ratios"] = list(participation_ratios)
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
        _phase_sync(2005, "coupled legacy before batch solve", status_callback=status_callback)
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
        _phase_sync(2006, "coupled legacy after batch solve", status_callback=status_callback)

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
    if "_fom_sifter_stats" not in config:
        config["_fom_sifter_stats"] = {}
    if "_fom_uniqueness_scores" not in config:
        config["_fom_uniqueness_scores"] = []
    if "_fom_participation_ratios" not in config:
        config["_fom_participation_ratios"] = []
    # Release solver matrices at end of FOM solve (assemble-only path returns earlier).
    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass
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


def run_fom_for_rom(
    config: Dict,
    num_modes: int = 10,
    status_callback=None,
    *,
    regenerate_mesh: bool = True,
    mesh_config_path: Optional[Path] = None,
):
    _phase_sync(3000, "run_fom_for_rom enter", status_callback=status_callback)
    mesh_file = Path(config["solver"]["mesh_file"])
    if regenerate_mesh:
        if MPI.COMM_WORLD.rank == 0:
            _emit(f"[mesh] generating mesh for config: {mesh_config_path or 'default'}", status_callback=status_callback)
        _generate_mesh_with_gmsh(config_path=mesh_config_path, status_callback=status_callback)
    _phase_sync(3001, "run_fom_for_rom after mesh generation", status_callback=status_callback)
    MPI.COMM_WORLD.barrier()
    if not mesh_file.exists():
        raise FileNotFoundError(f"Fresh mesh generation did not create expected file: {mesh_file}")
    _phase_sync(3002, "run_fom_for_rom before coupled solve", status_callback=status_callback)
    msh, W, freqs, eigvecs, n_u, n_p = _solve_coupled_evp(
        mesh_file=mesh_file,
        config=config,
        num_modes=num_modes,
        status_callback=status_callback,
    )
    sifter_stats = dict(config.pop("_fom_sifter_stats", {}) or {})
    uniqueness_scores = list(config.pop("_fom_uniqueness_scores", []) or [])
    participation_ratios = list(config.pop("_fom_participation_ratios", []) or [])
    return {
        "mesh": msh,
        "space": W,
        "freqs_hz": freqs,
        "eigvecs": eigvecs,
        "n_u": n_u,
        "n_p": n_p,
        "sifter_stats": sifter_stats,
        "uniqueness_scores": uniqueness_scores,
        "participation_ratios": participation_ratios,
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
            "Back_Plate": 3,
            "Ribs_Sides": 4,
            "wood_fix": 5,
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

    print(f"[RESULT] First {len(freqs)} mode frequencies (Hz): {[round(float(f), 3) for f in freqs]}")

    # Keep only the latest cache artifacts after a successful run.
    _cleanup_xdmf_cache_keep_latest(cache_dir, keep_last=2, status_callback=status_callback)

    _emit(f"Step 5/5: SUCCESS -> {output_path}", status_callback=status_callback)
    return output_path


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parents[1] / "configs" / "guitar_3d.json"
    run_fem_3d_simulation(str(default_config))