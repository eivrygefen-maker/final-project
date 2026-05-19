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
from typing import Any, Dict, List, Optional, Tuple

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
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc
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
STRUCTURAL_DIAG_SURFACE_TAGS = (1, 3)  # facet shell diagnostic: top + back only (no ribs)
RIBS_SURFACE_TAG = 4
WOOD_FIX_SURFACE_TAG = 5
AIR_VOLUME_TAG = 10
FSI_INTERFACE_FACET_TAG = 20  # topology wood↔air facets (not gmsh physical tag)
WOOD_VOLUME_TAGS = (1, 2, 3)
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
        for tag in WOOD_VOLUME_TAGS:
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
        wood_tag_set = set(WOOD_VOLUME_TAGS)
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


def _find_air_wood_interface_facets(msh: mesh.Mesh, cell_tags) -> np.ndarray:
    """Facets with one adjacent air (tag 10) cell and one adjacent wood volume cell (1/2/3)."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    c2f = msh.topology.connectivity(tdim, fdim)
    air_cells = np.asarray(cell_tags.find(AIR_VOLUME_TAG), dtype=np.int32)
    wood_tags = set(WOOD_VOLUME_TAGS)
    iface: List[int] = []
    for ci in air_cells:
        for fi in c2f.links(int(ci)):
            nbrs = np.asarray(f2c.links(int(fi)), dtype=np.int32)
            if nbrs.size < 2:
                continue
            tags = {int(cell_tags.values[int(nc)]) for nc in nbrs}
            if AIR_VOLUME_TAG in tags and bool(tags & wood_tags):
                iface.append(int(fi))
    return np.unique(np.asarray(iface, dtype=np.int32))


def _find_air_exterior_pressure_facets(
    msh: mesh.Mesh,
    cell_tags,
    facet_tags,
    interface_facets: np.ndarray,
) -> np.ndarray:
    """
    Air-boundary facets for ``p = 0`` (exterior of the cavity), excluding the FSI interface
    and soundhole (tag 2, already in the pressure gauge).
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    c2f = msh.topology.connectivity(tdim, fdim)
    air_cells = np.asarray(cell_tags.find(AIR_VOLUME_TAG), dtype=np.int32)
    iface_set = set(int(f) for f in np.asarray(interface_facets, dtype=np.int32))
    soundhole_set = set(int(f) for f in np.asarray(facet_tags.find(2), dtype=np.int32))
    out: List[int] = []
    for ci in air_cells:
        for fi in c2f.links(int(ci)):
            fi = int(fi)
            if fi in iface_set or fi in soundhole_set:
                continue
            nbrs = np.asarray(f2c.links(fi), dtype=np.int32)
            if nbrs.size != 1:
                continue
            if int(cell_tags.values[int(nbrs[0])]) == AIR_VOLUME_TAG:
                out.append(fi)
    return np.unique(np.asarray(out, dtype=np.int32))


def _meshtags_on_facets(msh: mesh.Mesh, facet_indices: np.ndarray, tag_value: int):
    """Build a facet ``MeshTags`` for a subset of local facet indices."""
    fdim = msh.topology.dim - 1
    idx = np.asarray(facet_indices, dtype=np.int32)
    if idx.size == 0:
        vals = np.array([], dtype=np.int32)
    else:
        vals = np.full(idx.shape, int(tag_value), dtype=np.int32)
    return mesh.meshtags(msh, fdim, idx, vals)


def _mesh_characteristic_length(msh: mesh.Mesh) -> float:
    """Length scale for Nitsche penalty (domain diagonal)."""
    coords = msh.geometry.x
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    return float(max(np.linalg.norm(maxs - mins), 1.0e-6))


def _fsi_nitsche_gamma(
    solver_cfg: Dict,
    *,
    norm_uu_ref: float,
    h_char: float,
) -> float:
    """
    Nitsche penalty tied to shell stiffness scale (not ``E_L * p_scale / h``, which can reach 1e10+).

    ``γ = frac * ||A_uu||_F / h``, capped at ``cap_frac * ||A_uu||_F``.
    """
    frac = float(solver_cfg.get("fsi_nitsche_penalty_frac", 1.0e-5))
    cap_frac = float(solver_cfg.get("fsi_nitsche_penalty_cap_frac", 1.0e-3))
    ref = max(float(norm_uu_ref), 1.0e-30)
    h = max(float(h_char), 1.0e-6)
    gamma = frac * ref / h
    gamma_max = cap_frac * ref
    return min(gamma, gamma_max)


def _audit_assembled_mixed_coupling(
    W: fem.FunctionSpace,
    a_up_form,
    m_pu_form,
    bcs: List[fem.DirichletBC],
    *,
    status_callback=None,
) -> None:
    """Assemble FSI off-diagonal blocks on ``W`` and report Frobenius norms (post-BC)."""
    _audit_assembled_coupling_block_layout(
        W, a_up_form, m_pu_form, bcs, status_callback=status_callback
    )


def _mat_frobenius_norm(
    a_form,
    *,
    bcs: Optional[List[fem.DirichletBC]] = None,
    label: str = "",
) -> float:
    """Assemble a bilinear form and return its Frobenius norm (PETSc default norm)."""
    K: Optional[PETSc.Mat] = None
    tag = label or "form"
    try:
        K = assemble_matrix(fem.form(a_form), bcs=bcs or [])
        K.assemble()
        return float(K.norm())
    except Exception as exc:
        raise RuntimeError(
            f"Frobenius norm assembly failed ({tag}): {type(exc).__name__}: {exc!r}"
        ) from exc
    finally:
        if K is not None:
            try:
                K.destroy()
            except Exception:
                pass


def _audit_mixed_w_dof_maps(W: fem.FunctionSpace, *, status_callback=None) -> None:
    """Log mixed-space vs subspace DOF layout before FSI block assembly."""
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    W_u = W.sub(0)
    W_p = W.sub(1)
    n_w = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)
    n_u_sub = int(W_u.dofmap.index_map.size_global * W_u.dofmap.index_map_bs)
    n_p_sub = int(W_p.dofmap.index_map.size_global * W_p.dofmap.index_map_bs)
    V_u_col, _ = W_u.collapse()
    V_p_col, _ = W_p.collapse()
    n_u_col = _u_global_dof_count(V_u_col)
    n_p_col = int(V_p_col.dofmap.index_map.size_global * V_p_col.dofmap.index_map_bs)
    print(
        "[FSI-DOF] mixed W: "
        f"global={n_w}, W.sub(0) index_map={n_u_sub}, W.sub(1) index_map={n_p_sub}, "
        f"collapsed u={n_u_col}, collapsed p={n_p_col}"
    )
    if n_u_sub != n_p_sub:
        print(
            "[FSI-DOF] note: W.sub(0) and W.sub(1) index_map sizes differ in mixed layout "
            "(expected); collapsed u/p counts are the physical unknowns."
        )
    sys.stdout.flush()


def _audit_assembled_coupling_block_layout(
    W: fem.FunctionSpace,
    a_up_form,
    m_pu_form,
    bcs: List[fem.DirichletBC],
    *,
    status_callback=None,
) -> Tuple[float, float]:
    """
    Assemble coupling blocks on mixed ``W``, compare PETSc sizes to subspace DOF maps.
    Returns (||A_up||_F, ||M_pu||_F).
    """
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return float("nan"), float("nan")
    A_up: Optional[PETSc.Mat] = None
    M_pu: Optional[PETSc.Mat] = None
    try:
        A_up = assemble_matrix(fem.form(a_up_form), bcs=bcs)
        A_up.assemble()
        M_pu = assemble_matrix(fem.form(m_pu_form), bcs=bcs)
        M_pu.assemble()
        n_up = float(A_up.norm())
        n_pu = float(M_pu.norm())
        nrows, ncols = A_up.getSize()
        W_u = W.sub(0)
        W_p = W.sub(1)
        n_u_sub = int(W_u.dofmap.index_map.size_global * W_u.dofmap.index_map_bs)
        n_p_sub = int(W_p.dofmap.index_map.size_global * W_p.dofmap.index_map_bs)
        try:
            own_up = A_up.getOwnershipRange()
        except Exception:
            own_up = (0, 0)
        print(
            f"[FSI-AUDIT] assembled coupling on mixed W: ||A_up||_F={n_up:.6e}, "
            f"||M_pu||_F={n_pu:.6e}, A_up PETSc size=({nrows}, {ncols}), "
            f"ownership_range={own_up}, W.sub(0) dofs={n_u_sub}, W.sub(1) dofs={n_p_sub}"
        )
        n_w = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)
        if nrows != n_w:
            _emit(
                f"[FSI-AUDIT][warn] A_up row count {nrows} != mixed W global {n_w}",
                status_callback=status_callback,
                level="warning",
            )
        if n_up < 1.0e-20 and n_pu < 1.0e-20:
            _emit(
                "[FSI-AUDIT][CRITICAL] assembled FSI coupling blocks are numerically zero.",
                status_callback=status_callback,
                level="error",
            )
        sys.stdout.flush()
        return n_up, n_pu
    except Exception as exc:
        _emit(
            f"[FSI-AUDIT][warn] coupling layout audit failed: {type(exc).__name__}: {exc!r}",
            status_callback=status_callback,
            level="warning",
        )
        return float("nan"), float("nan")
    finally:
        for mat in (A_up, M_pu):
            if mat is not None:
                try:
                    mat.destroy()
                except Exception:
                    pass


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


def _displacement_element(msh: mesh.Mesh, degree: int = 1):
    """Shared P{degree} vector Lagrange element for structural-only V_u and coupled W.sub(0)."""
    return element("Lagrange", msh.basix_cell(), int(degree), shape=(3,))


def _dofmap_list_length(V) -> int:
    """Length of the dofmap adjacency list (local mesh entities → DOF indices)."""
    dm_list = V.dofmap.list
    if hasattr(dm_list, "num_nodes"):
        return int(dm_list.num_nodes)
    if hasattr(dm_list, "size"):
        return int(dm_list.size)
    return int(len(np.asarray(dm_list.array)))


def _u_global_dof_count(V_u) -> int:
    return int(V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs)


def _audit_u_subspace_global(
    V_u,
    *,
    label: str,
    V_u_collapsed=None,
    status_callback=None,
) -> None:
    """Confirm displacement space is the full mesh P1^3 field (collapse must not shrink DOF set)."""
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    n_direct = _u_global_dof_count(V_u)
    msg = f"[DIAG] {label}: global u DOFs (direct)={n_direct}"
    if V_u_collapsed is not None:
        n_coll = _u_global_dof_count(V_u_collapsed)
        msg += f", collapsed={n_coll}"
        if n_coll != n_direct:
            level = "warning"
            note = "unexpected — BC/dof maps may be inconsistent"
            if "coupled W.sub(0)" in label:
                level = "info"
                note = "expected for mixed W.sub(0) (use collapsed V_u for BCs)"
            _emit(
                f"{label}: collapse() n_u={n_coll} vs subspace index_map n_u={n_direct} ({note}).",
                status_callback=status_callback,
                level=level,
            )
    _emit(msg, status_callback=status_callback)


def _locate_facet_displacement_dofs(V_u, msh, facet_indices: np.ndarray) -> np.ndarray:
    """Locate all displacement DOFs supported on facet entities (vector P1 → use fdim)."""
    if facet_indices is None or facet_indices.size == 0:
        return np.array([], dtype=np.int32)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    return np.array(
        fem.locate_dofs_topological(V_u, fdim, np.asarray(facet_indices, dtype=np.int32)),
        dtype=np.int32,
    )


def _audit_shell_facet_dof_coverage(
    msh,
    facet_tags,
    V_u,
    *,
    tag_top: int,
    tag_back: int,
    tag_ribs: int = RIBS_SURFACE_TAG,
    label: str = "V_u",
    constrained_u_dofs: Optional[np.ndarray] = None,
    status_callback=None,
) -> Dict[int, int]:
    """
    Report facet counts and len(dofs) per physical tag on the displacement space.
    Uses fdim-based localization (correct for vector Lagrange on shell facets).
    """
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return {}

    facets_top = np.array(facet_tags.find(int(tag_top)), dtype=np.int32)
    facets_back = np.array(facet_tags.find(int(tag_back)), dtype=np.int32)
    facets_ribs = np.array(facet_tags.find(int(tag_ribs)), dtype=np.int32)
    facets_fix = np.array(facet_tags.find(WOOD_FIX_SURFACE_TAG), dtype=np.int32)

    dofs_top = _locate_facet_displacement_dofs(V_u, msh, facets_top)
    dofs_back = _locate_facet_displacement_dofs(V_u, msh, facets_back)
    dofs_ribs = _locate_facet_displacement_dofs(V_u, msh, facets_ribs)
    dofs_fix = _locate_facet_displacement_dofs(V_u, msh, facets_fix)

    print(
        f"[DIAG] {label} shell facet coverage: "
        f"tag{tag_top} facets={facets_top.size} len(dofs_top)={dofs_top.size}, "
        f"tag{tag_back} facets={facets_back.size} len(dofs_back)={dofs_back.size}, "
        f"tag{tag_ribs} facets={facets_ribs.size} len(dofs_ribs)={dofs_ribs.size}, "
        f"tag{WOOD_FIX_SURFACE_TAG}(fix) facets={facets_fix.size} len(dofs_fix)={dofs_fix.size}"
    )
    if dofs_top.size == 0 and facets_top.size > 0:
        _emit(
            f"[DIAG][CRITICAL] tag {tag_top} has {facets_top.size} facets but len(dofs_top)=0 — "
            "locate_dofs_topological(fdim, facets) found no displacement DOFs. "
            "Check facet_tags mapping vs build_3d_guitar physical groups.",
            status_callback=status_callback,
            level="error",
        )
    elif dofs_top.size == 0:
        _emit(
            f"[DIAG][warn] tag {tag_top}: zero facets and zero DOFs on displacement space.",
            status_callback=status_callback,
            level="warning",
        )

    if constrained_u_dofs is not None and constrained_u_dofs.size > 0:
        cset = np.unique(np.asarray(constrained_u_dofs, dtype=np.int32))
        for tname, tdofs in (
            ("top", dofs_top),
            ("back", dofs_back),
            ("ribs", dofs_ribs),
            ("wood_fix", dofs_fix),
        ):
            if tdofs.size == 0:
                continue
            overlap = np.intersect1d(tdofs, cset)
            if overlap.size > 0:
                print(
                    f"[DIAG] {label}: {overlap.size} displacement DOFs on {tname} shell "
                    f"overlap Dirichlet BC set (expected only at rib/top junction if ribs clamped)."
                )
        top_bc = np.intersect1d(dofs_top, cset)
        if top_bc.size > 0:
            _emit(
                f"[DIAG][CRITICAL] {top_bc.size} tag-{tag_top} (top plate) displacement DOFs are "
                "inside a Dirichlet BC — top plate is not free.",
                status_callback=status_callback,
                level="error",
            )
    sys.stdout.flush()
    return {
        int(tag_top): int(dofs_top.size),
        int(tag_back): int(dofs_back.size),
        int(tag_ribs): int(dofs_ribs.size),
        WOOD_FIX_SURFACE_TAG: int(dofs_fix.size),
    }


def _assert_no_top_plate_dirichlet_bc(
    facet_tags,
    V_u,
    msh,
    u_dofs_constrained: np.ndarray,
    *,
    tag_top: int,
    status_callback=None,
) -> None:
    """Coupled path guard: constrained displacement DOFs must not include free top-plate facets."""
    facets_top = np.array(facet_tags.find(int(tag_top)), dtype=np.int32)
    dofs_top = _locate_facet_displacement_dofs(V_u, msh, facets_top)
    if dofs_top.size == 0 or u_dofs_constrained.size == 0:
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            _emit(
                f"[bc] verified: no displacement BC overlap on tag-{tag_top} top "
                f"(len(dofs_top)={dofs_top.size}, constrained u dofs={u_dofs_constrained.size}).",
                status_callback=status_callback,
            )
        return
    overlap = np.intersect1d(dofs_top, np.asarray(u_dofs_constrained, dtype=np.int32))
    if overlap.size > 0:
        raise RuntimeError(
            f"Displacement BC constrains {overlap.size} top-plate (tag {tag_top}) DOFs; "
            "only ribs (tag 4) may be clamped. Set clamp_ribs=false or fix facet tagging."
        )
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        _emit(
            f"[bc] verified: no displacement BC on tag-{tag_top} top plate "
            f"(len(dofs_top)={dofs_top.size}, rib-constrained dofs={u_dofs_constrained.size}).",
            status_callback=status_callback,
        )


def _mixed_eigenvector_block_norms(
    arr: np.ndarray,
    n_u_dofs: int = 0,
    *,
    u_to_W: Optional[np.ndarray] = None,
    p_to_W: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """L2 norms of u and p on the mixed W global vector (use collapse maps when provided)."""
    if u_to_W is not None and p_to_W is not None:
        u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
        p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
        u_norm = float(np.linalg.norm(arr[u_idx])) if u_idx.size > 0 else 0.0
        p_norm = float(np.linalg.norm(arr[p_idx])) if p_idx.size > 0 else 0.0
        return u_norm, p_norm
    n = max(0, int(n_u_dofs))
    if arr.size < n:
        return 0.0, 0.0
    u_norm = float(np.linalg.norm(arr[:n]))
    p_norm = float(np.linalg.norm(arr[n:])) if arr.size > n else 0.0
    return u_norm, p_norm


def _max_pressure_block_abs(
    arr: np.ndarray,
    p_to_W: Optional[np.ndarray] = None,
) -> float:
    """Max absolute pressure coefficient on mixed W (collapse map when provided)."""
    if p_to_W is None:
        return 0.0
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    if p_idx.size == 0:
        return 0.0
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return float(np.max(np.abs(flat[p_idx])))


def _eps_use_target_real(solver_cfg: Dict) -> bool:
    which = str(solver_cfg.get("eps_which", "")).strip().upper()
    if which in ("TARGET_REAL", "TARGET_REAL_PART"):
        return True
    return _solver_bool(solver_cfg, "eps_use_target_real", default=False)


def _slepc_lambda_hz_bounds(solver_cfg: Optional[Dict] = None) -> Tuple[float, float]:
    """Plausible physical frequency band for mapped λ = ω² (rad/s)²."""
    f_lo = 50.0
    f_hi = 600.0
    if solver_cfg:
        try:
            f_lo = max(10.0, 0.5 * float(solver_cfg.get("min_valid_mode_hz", 90.0)))
        except (TypeError, ValueError):
            pass
        try:
            f_hi = 1.25 * float(solver_cfg.get("max_valid_mode_hz", 480.0))
        except (TypeError, ValueError):
            pass
    lam_lo = (2.0 * math.pi * f_lo) ** 2
    lam_hi = (2.0 * math.pi * f_hi) ** 2
    return lam_lo, lam_hi


def _slepc_physical_lambda(
    eig_r: float,
    st_sigma: float,
    st_name: str,
    solver_cfg: Optional[Dict] = None,
) -> Tuple[float, str]:
    """
    Map EPS/ST Ritz value μ to physical GNHEP eigenvalue λ = ω² (rad/s)².

    Returns ``(λ, tag)`` with ``tag`` in
    ``raw|shift|invert|reject``. Non-finite λ means discard the Ritz pair.

    SLEPc builds differ: μ may already be physical λ, or λ-σ (shift ST), or
  1/(λ-σ) (sinvert). Never apply σ+1/μ when |μ| is astronomical — that pins all
    modes to the shift. For sinvert+TARGET_MAGNITUDE many builds return physical λ
    directly when μ ≈ σ (ω² scale).
    """
    sigma = float(st_sigma)
    mu = float(np.real(eig_r))
    name = str(st_name).strip().lower()
    lam_lo, lam_hi = _slepc_lambda_hz_bounds(solver_cfg)
    # |1/(λ-σ)| for sinvert; |λ-σ| for shift — both are modest for band modes.
    mu_invert_max = float(
        solver_cfg.get("eps_map_invert_mu_max", 1.0e8) if solver_cfg else 1.0e8
    )
    sigma_sep = float(
        solver_cfg.get("eps_map_min_sigma_sep_frac", 0.02) if solver_cfg else 0.02
    )
    min_dlam = max(1.0, sigma_sep * max(sigma, 1.0))

    def _ok(lam: float) -> bool:
        return math.isfinite(lam) and lam_lo <= lam <= lam_hi

    if not math.isfinite(mu):
        return float("nan"), "reject"

    if name in ("ciss", "ciss_contour"):
        if _ok(mu):
            return mu, "raw"
        return float("nan"), "reject"

    if name in ("sinvert", "shift_invert"):
        if mu <= 0.0:
            return float("nan"), "reject"
        # sinvert Ritz θ ≈ 1/(λ-σ); map via invert when |θ| is moderate.
        if mu > 0.0 and abs(mu) <= mu_invert_max:
            lam_inv = sigma + (1.0 / mu)
            if _ok(lam_inv) and abs(lam_inv - sigma) >= min_dlam:
                return lam_inv, "invert"
        # Physical λ at σ (sinvert+TARGET_MAGNITUDE) — discard when too close to σ (spurious anchor).
        if _ok(mu) and abs(mu - sigma) >= min_dlam:
            return mu, "raw"
        return float("nan"), "reject"

    lam_shift = mu + sigma
    if _ok(lam_shift):
        return lam_shift, "shift"
    if _ok(mu):
        return mu, "raw"
    if mu > 0.0 and abs(mu) <= mu_invert_max:
        lam_inv = sigma + (1.0 / mu)
        if _ok(lam_inv) and abs(lam_inv - sigma) >= min_dlam:
            return lam_inv, "invert"
    return float("nan"), "reject"


def _slepc_eps_strategy(
    solver_cfg: Dict,
) -> Tuple[Any, str, bool, bool]:
    """
    Resolve SLEPc ``which`` + ST pairing from config.

    Returns ``(eps_which_enum, label, use_st_shift, use_broad_hz_window)``.

    Default band solve: ``st_type=sinvert`` + ``TARGET_MAGNITUDE`` at the band center
    (harvest must apply ``_slepc_physical_lambda``). Config ``eps_which=SHIFT`` is used
    when the SLEPc build exposes ``EPS.Which.SHIFT``; otherwise we fall back without aborting.
    """
    which = str(solver_cfg.get("eps_which", "TARGET_MAGNITUDE")).strip().upper()
    st_name = str(solver_cfg.get("st_type", "sinvert")).strip().lower()
    use_st_shift = st_name in ("shift", "stshift")
    broad_hz = float(solver_cfg.get("eps_broad_search_hz", 0.0))

    if which == "SHIFT":
        eps_shift = getattr(SLEPc.EPS.Which, "SHIFT", None)
        if eps_shift is not None:
            return eps_shift, "SHIFT", True, broad_hz > 0.0
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                "[solver][warn] SLEPc.EPS.Which.SHIFT unavailable in this build; "
                "using TARGET_REAL + ST shift (set eps_which=TARGET_REAL to silence).",
                flush=True,
            )
        which = "TARGET_REAL"
        if broad_hz <= 0.0:
            broad_hz = 1.5

    if which in ("TARGET_REAL", "TARGET_REAL_PART") or _eps_use_target_real(solver_cfg):
        if use_st_shift and broad_hz <= 0.0:
            broad_hz = 1.5
        return (
            SLEPc.EPS.Which.TARGET_REAL,
            "TARGET_REAL",
            use_st_shift,
            broad_hz > 0.0,
        )
    if which in ("SMALLEST_MAGNITUDE", "SMALLEST_MAG"):
        # SLEPc shift-invert only accepts *target* which (TARGET_MAGNITUDE / TARGET_REAL).
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                "[solver][warn] eps_which=SMALLEST_MAGNITUDE is incompatible with "
                "st_type=sinvert (SLEPc: 'Shift-and-invert requires a target which'); "
                "falling back to TARGET_MAGNITUDE. Use eps_broad_search_hz + per-shift "
                "ladder for band coverage.",
                flush=True,
            )
        which = "TARGET_MAGNITUDE"
    if which in ("TARGET_MAGNITUDE", "TARGET_MAG", ""):
        return (
            SLEPc.EPS.Which.TARGET_MAGNITUDE,
            "TARGET_MAGNITUDE",
            use_st_shift,
            broad_hz > 0.0,
        )
    raise ValueError(f"Unsupported solver.eps_which={which!r}")


def _print_raw_coupling_block_norms(
    a_up,
    m_pu,
    *,
    status_callback=None,
) -> None:
    """Ground-truth Frobenius norms of FSI blocks before GNHEP scaling."""
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    try:
        norm_a_up = _mat_frobenius_norm(a_up)
        norm_m_pu = _mat_frobenius_norm(m_pu)
    except Exception as exc:
        _emit(f"[GNHEP-raw][warn] block norm trace failed: {exc}", status_callback=status_callback, level="warning")
        return
    print(
        f"[GNHEP-raw] before any GNHEP scaling: ||A_up||_F={norm_a_up:.6e}, "
        f"||M_pu||_F={norm_m_pu:.6e} (u→p mass coupling; M_up ≡ M_pu^T in mixed layout)"
    )
    sys.stdout.flush()


def _collapsed_u_from_mixed_vec(
    phi_arr: np.ndarray,
    u_to_W: np.ndarray,
) -> np.ndarray:
    """Extract collapsed displacement coefficients (same indexing as XDMF export)."""
    idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    return np.asarray(phi_arr, dtype=np.float64)[idx]


def _plate_facet_dof_energy_ratios(
    u_collapsed: np.ndarray,
    dofs_top: np.ndarray,
    dofs_back: np.ndarray,
) -> Tuple[float, float]:
    """Kinetic proxy on facet displacement DOFs (tag 1 / tag 3); robust for shell-tagged modes."""
    e_top = 0.0
    e_back = 0.0
    if dofs_top.size > 0:
        ut = u_collapsed[np.asarray(dofs_top, dtype=np.int32)]
        e_top = float(np.dot(ut, ut))
    if dofs_back.size > 0:
        ub = u_collapsed[np.asarray(dofs_back, dtype=np.int32)]
        e_back = float(np.dot(ub, ub))
    total = e_top + e_back
    if total < 1.0e-30:
        return 0.0, 0.0
    return e_top / total, e_back / total


def _plate_modal_energy_ratios(
    phi: PETSc.Vec,
    M_top: Optional[PETSc.Mat],
    M_back: Optional[PETSc.Mat],
    work: PETSc.Vec,
    mass_top: float,
    mass_back: float,
    *,
    u_parent_indices: Optional[np.ndarray] = None,
    phi_u: Optional[PETSc.Vec] = None,
    work_u: Optional[PETSc.Vec] = None,
    u_to_W: Optional[np.ndarray] = None,
    dofs_top: Optional[np.ndarray] = None,
    dofs_back: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Relative shell participation on each plate (tag1 top / tag3 back).

    Prefer facet-DOF kinetic ratios (matches structural diagnostic); fall back to M_top/M_back
    quadratic forms when facet DOFs carry no energy.
    """
    map_u = u_to_W if u_to_W is not None else u_parent_indices
    if map_u is not None and dofs_top is not None and dofs_back is not None:
        u_collapsed = _collapsed_u_from_mixed_vec(phi.array, map_u)
        rt_dof, rb_dof = _plate_facet_dof_energy_ratios(
            u_collapsed,
            np.asarray(dofs_top, dtype=np.int32),
            np.asarray(dofs_back, dtype=np.int32),
        )
        if rt_dof + rb_dof > 1.0e-12:
            return rt_dof, rb_dof

    vec_u = phi
    work_out = work_u if work_u is not None else work
    if u_parent_indices is not None and phi_u is not None:
        parent = np.asarray(u_parent_indices, dtype=np.int32).ravel()
        phi_u.set(0.0)
        src = phi.array
        dst = phi_u.array
        n = min(parent.size, dst.size)
        if n > 0:
            dst[:n] = src[parent[:n]]
        vec_u = phi_u
        if work_u is None:
            work_out = phi_u.duplicate()
        else:
            work_out = work_u

    e_top = 0.0
    e_back = 0.0
    if M_top is not None:
        M_top.mult(vec_u, work_out)
        e_top = float(np.real(vec_u.dot(work_out)))
    if M_back is not None:
        M_back.mult(vec_u, work_out)
        e_back = float(np.real(vec_u.dot(work_out)))
    total_wood_energy = e_top + e_back
    if total_wood_energy < 1.0e-18:
        return 0.0, 0.0
    return e_top / total_wood_energy, e_back / total_wood_energy


def _is_structural_only_run(config: Dict, solve_evp: bool) -> bool:
    """True → displacement-only shell EVP (no acoustic DOFs / no FSI)."""
    if not solve_evp:
        return False
    solver = config.get("solver", {})
    if not _solver_bool(solver, "couple_fluid", default=True):
        return True
    return _solver_bool(solver, "structural_only_diagnosis", default=False)


def _structural_diag_facet_tags(solver_cfg: Dict) -> Tuple[int, ...]:
    raw = solver_cfg.get("structural_shell_facet_tags", STRUCTURAL_DIAG_SURFACE_TAGS)
    if isinstance(raw, (list, tuple)):
        return tuple(int(t) for t in raw)
    return STRUCTURAL_DIAG_SURFACE_TAGS


def _fsi_coupling_gain(solver_cfg: Dict) -> float:
    """Scalar boost on u–p interface blocks so SLEPc sees FSI coupling in the GNHEP."""
    g = float(solver_cfg.get("fsi_coupling_gain", 1.0e4))
    return g if g > 0.0 else 1.0


def _coupled_pressure_dof_scale(solver_cfg: Dict) -> float:
    """
    Similarity scale s on pressure DOFs (D = diag(I, s·I_p)); applied consistently to all
    pressure blocks so GNHEP eigenvalues (ω²) are unchanged but block magnitudes match u.
    """
    s = float(solver_cfg.get("pressure_dof_scale", 30.0))
    return s if s > 0.0 else 1.0


def _audit_coupled_displacement_space(
    msh: mesh.Mesh,
    W: fem.FunctionSpace,
    u_el,
    *,
    status_callback=None,
) -> Tuple[fem.FunctionSpace, fem.FunctionSpace, np.ndarray]:
    """
    Coupled-space audit for mixed W vs structural ``fem.functionspace(msh, u_el)``.

    ``W.sub(0)`` is not the same object as standalone ``V_u`` and its ``dofmap.list`` length is
  *not* comparable (mixed subspaces use parent dofmap layout).  The coupled path uses
    ``V_u_collapsed, map = W.sub(0).collapse()`` for BCs and mode export; standalone ``V_u``
    must match ``V_u_collapsed`` global DOF count for structural→W transfer.
    """
    V_u_standalone = fem.functionspace(msh, u_el)
    V_u_sub = W.sub(0)
    V_u_collapsed, u_parent_indices = V_u_sub.collapse()

    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return V_u_standalone, V_u_collapsed, u_parent_indices

    len_sub_list = _dofmap_list_length(V_u_sub)
    len_standalone = _dofmap_list_length(V_u_standalone)
    len_collapsed = _dofmap_list_length(V_u_collapsed)
    n_u_sub_reported = _u_global_dof_count(V_u_sub)
    n_u_standalone = _u_global_dof_count(V_u_standalone)
    n_u_collapsed = _u_global_dof_count(V_u_collapsed)
    n_p = _u_global_dof_count(W.sub(1))
    n_w_total = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)

    print(
        "[COUPLED-SPACE] dofmap.list lengths (informational): "
        f"len(W.sub(0).dofmap.list)={len_sub_list}, "
        f"len(V_u_standalone.dofmap.list)={len_standalone}, "
        f"len(V_u_collapsed.dofmap.list)={len_collapsed}"
    )
    if len_sub_list != len_collapsed:
        print(
            "[COUPLED-SPACE] note: W.sub(0).dofmap.list length differs from collapsed V_u — "
            "expected for mixed subspaces; do not compare these list lengths."
        )
    print(
        "[COUPLED-SPACE] global DOFs: "
        f"V_u_standalone={n_u_standalone}, V_u_collapsed={n_u_collapsed}, "
        f"W.sub(0) index_map reports={n_u_sub_reported} (may include mixed layout), "
        f"W.sub(1) p={n_p}, W total={n_w_total}"
    )
    print(
        f"[COUPLED-SPACE] W.sub(0) is V_u_standalone object? {V_u_sub is V_u_standalone} "
        "(expected False; coupled BCs use V_u_collapsed + collapse map)"
    )

    if n_u_standalone != n_u_collapsed:
        raise RuntimeError(
            "Coupled-space audit failed: standalone V_u and W.sub(0).collapse() global DOF "
            f"counts differ ({n_u_standalone} vs {n_u_collapsed})."
        )

    parent_idx = np.asarray(u_parent_indices, dtype=np.int32)
    n_collapsed_local = int(V_u_collapsed.dofmap.index_map.size_local * V_u_collapsed.dofmap.index_map_bs)
    n_w_local = int(W.dofmap.index_map.size_local * W.dofmap.index_map_bs)
    if parent_idx.size != n_collapsed_local:
        raise RuntimeError(
            f"collapse() map length {parent_idx.size} != collapsed V_u local size {n_collapsed_local}."
        )
    if parent_idx.size > 0 and (int(parent_idx.max()) >= n_w_local or int(parent_idx.min()) < 0):
        raise RuntimeError(
            f"collapse() parent indices out of range for local mixed W vector "
            f"(max={int(parent_idx.max())}, local_W_size={n_w_local})."
        )

    _emit(
        f"[COUPLED-SPACE] OK: standalone V_u matches collapsed subspace "
        f"(n_u={n_u_collapsed}, collapse_map_len={parent_idx.size}, W_total={n_w_total}).",
        status_callback=status_callback,
    )
    sys.stdout.flush()
    return V_u_standalone, V_u_collapsed, parent_idx


def _audit_pressure_scale_block_balance(
    a_uu,
    a_pp,
    a_up,
    *,
    p_scale: float,
    fsi_gain: float = 1.0,
    tag_top: int,
    n_facets_top: int,
    norm_uu_shell: Optional[float] = None,
    status_callback=None,
) -> None:
    """
    Verify pressure similarity scale does not swamp shell stiffness A_uu (tag-1 facets).

    ``pressure_dof_scale`` only enters p–p and u–p blocks; A_uu is unchanged, but the full
    GNHEP may be ill-conditioned if coupling blocks dominate after scaling.
    """
    if MPI.COMM_WORLD.rank != ROOT_RANK:
        return
    try:
        norm_uu = (
            float(norm_uu_shell)
            if norm_uu_shell is not None and math.isfinite(float(norm_uu_shell))
            else _mat_frobenius_norm(a_uu, label="a_uu")
        )
        norm_pp = _mat_frobenius_norm(a_pp, label="a_pp")
        norm_up = _mat_frobenius_norm(a_up, label="a_up")
    except Exception as exc:
        _emit(
            f"[COUPLED-SPACE][warn] block norm audit failed: {type(exc).__name__}: {exc!r}",
            status_callback=status_callback,
            level="warning",
        )
        return
    ratio_pu = norm_pp / max(norm_uu, 1.0e-30)
    ratio_couple = norm_up / max(norm_uu, 1.0e-30)
    print(
        f"[COUPLED-SPACE] block Frobenius norms "
        f"(pressure_dof_scale={p_scale:.4g}, fsi_coupling_gain={fsi_gain:.4g}): "
        f"||A_uu||={norm_uu:.6e} (tag{tag_top} facets={n_facets_top}), "
        f"||A_pp||={norm_pp:.6e}, ||A_up||={norm_up:.6e}"
    )
    print(
        f"[COUPLED-SPACE] ||A_pp||/||A_uu||={ratio_pu:.6e}, ||A_up||/||A_uu||={ratio_couple:.6e} "
        f"(A_uu independent of pressure_dof_scale; coupling scales ∝ s·gain and s²·gain)"
    )
    for probe_s in (5.0, p_scale):
        if probe_s <= 0.0:
            continue
        s_ratio = probe_s / max(p_scale, 1.0e-30)
        est_up = norm_up * abs(s_ratio)
        est_pp = norm_pp * (s_ratio * s_ratio)
        print(
            f"[COUPLED-SPACE] probe s={probe_s:.4g}: estimated ||A_up||≈{est_up:.6e}, "
            f"||A_pp||≈{est_pp:.6e} vs ||A_uu||={norm_uu:.6e}"
        )
    if norm_uu < 1.0e-15 and n_facets_top > 0:
        _emit(
            f"[COUPLED-SPACE][CRITICAL] ||A_uu||≈0 with {n_facets_top} tag-{tag_top} facets — "
            "shell stiffness missing before pressure scaling.",
            status_callback=status_callback,
            level="error",
        )
    elif ratio_couple > 1.0e6 or ratio_pu > 1.0e8:
        _emit(
            "[COUPLED-SPACE][warn] pressure blocks may dominate A_uu in Frobenius norm — "
            "consider adjusting pressure_dof_scale or enabling gnhep_normalize_matrices.",
            status_callback=status_callback,
            level="warning",
        )
    sys.stdout.flush()


def _block_frobenius_normalize_coupled_forms(
    a_uu,
    a_pp,
    a_up,
    m_uu,
    m_pp,
    m_pu,
    reg_p,
    solver_cfg: Dict,
    *,
    status_callback=None,
):
    """
    Scale ``A_uu`` and ``A_pp`` (and matching mass / coupling blocks) by separate Frobenius
    factors so ``M`` is not crushed by a single global ``max(||A||_F, ||M||_F)`` scale.
    """
    if not _solver_bool(solver_cfg, "gnhep_block_frobenius_normalize", default=True):
        return a_uu, a_pp, a_up, m_uu, m_pp, m_pu, reg_p, 1.0, 1.0
    if _solver_bool(solver_cfg, "gnhep_normalize_matrices", default=False):
        return a_uu, a_pp, a_up, m_uu, m_pp, m_pu, reg_p, 1.0, 1.0

    s_u = max(_mat_frobenius_norm(a_uu), 1.0e-30)
    s_p = max(_mat_frobenius_norm(a_pp), 1.0e-30)
    s_c = math.sqrt(s_u * s_p)
    inv_u = 1.0 / s_u
    inv_p = 1.0 / s_p
    inv_c = 1.0 / s_c
    # UFL Forms do not support ``form / float``; scale with scalar multiply.
    a_uu_s = inv_u * a_uu
    a_pp_s = inv_p * a_pp
    a_up_s = inv_c * a_up
    m_uu_s = inv_u * m_uu
    m_pp_s = inv_p * m_pp
    m_pu_s = inv_c * m_pu
    reg_p_s = inv_p * reg_p
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        m_uu_n = _mat_frobenius_norm(m_uu_s)
        m_pp_n = _mat_frobenius_norm(m_pp_s)
        print(
            f"[form] GNHEP block Frobenius scales: s_uu={s_u:.6e}, s_pp={s_p:.6e}, s_couple={s_c:.6e} "
            f"(||M_uu||_F={m_uu_n:.6e}, ||M_pp||_F={m_pp_n:.6e} after block scaling)"
        )
        sys.stdout.flush()
    return a_uu_s, a_pp_s, a_up_s, m_uu_s, m_pp_s, m_pu_s, reg_p_s, s_u, s_p


def _resolvent_probe_block_solver_cfg(solver_cfg: Dict) -> Dict:
    """Probe path: force block Frobenius scaling so ||A_uu|| and ||A_pp|| are O(1)."""
    if not _solver_bool(solver_cfg, "resolvent_block_frobenius_normalize", default=True):
        return solver_cfg
    merged = dict(solver_cfg)
    merged["gnhep_block_frobenius_normalize"] = True
    return merged


def _append_resolvent_probe_stabilization(
    a_uu,
    a_pp,
    reg_p,
    *,
    u,
    v,
    p,
    q,
    p2: float,
    wood_ds,
    air_dx,
    solver_cfg: Dict,
    norm_uu_ref: float,
    norm_pp_ref: float,
    status_callback=None,
) -> Tuple[Any, Any, float, float, float, float]:
    """
    Soft grounding for the harmonic resolvent probe (free–free shell + acoustic null modes).

    - Structural: ``k_u = frac_u * ||A_uu||_F`` on wood shell facets (default frac_u=1e-7).
    - Acoustic: ``k_p = frac_p * ||A_pp||_F`` on air volume (default frac_p=1e-4), in addition
      to the soundhole pressure gauge BC.
    """
    frac_u = float(solver_cfg.get("resolvent_struct_penalty_frac", 1.0e-7))
    frac_p = float(solver_cfg.get("resolvent_acoustic_penalty_frac", 1.0e-4))
    norm_uu = max(float(norm_uu_ref), 1.0e-30)
    norm_pp = max(float(norm_pp_ref), 1.0e-30)
    k_u = frac_u * norm_uu
    k_p = frac_p * norm_pp
    reg_u = k_u * ufl.dot(u, v) * wood_ds
    reg_p_extra = k_p * p2 * p * q * air_dx
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(
            f"[resolvent-probe] stabilization: k_u={k_u:.6e} ({frac_u:.1e}*||A_uu||), "
            f"k_p={k_p:.6e} ({frac_p:.1e}*||A_pp||) on wood shell + air volume"
        )
        sys.stdout.flush()
    _emit(
        f"[resolvent-probe] soft penalties: k_u={k_u:.6e}, k_p={k_p:.6e} "
        f"(soundhole gauge BC unchanged)",
        status_callback=status_callback,
    )
    return a_uu + reg_u, reg_p + reg_p_extra, k_u, k_p, norm_uu, norm_pp


def _resolvent_symmetric_equilibrate(
    K: PETSc.Mat, b: PETSc.Vec
) -> Tuple[PETSc.Vec, PETSc.Vec]:
    """Symmetric diagonal scaling ``K' = D K D``, ``b' = D b`` with ``D_ii = 1/sqrt(|K_ii|)``."""
    d = K.getDiagonal()
    arr_d = np.abs(np.asarray(d.array, dtype=np.float64))
    arr_s = 1.0 / np.sqrt(np.maximum(arr_d, 1.0e-30))
    arr_s[arr_d < 1.0e-30] = 1.0
    inv_sqrt = PETSc.Vec().createWithArray(
        arr_s.astype(PETSc.ScalarType, copy=False), comm=K.getComm()
    )
    K.diagonalScale(inv_sqrt, inv_sqrt)
    b_scaled = b.duplicate()
    b_scaled.copy(b)
    b_scaled.pointwiseMult(inv_sqrt)
    return inv_sqrt, b_scaled


def _resolvent_unscale_solution(x: PETSc.Vec, inv_sqrt: PETSc.Vec) -> None:
    """Recover physical solution from equilibrated unknowns (``x = D^{-1} x_hat``)."""
    arr_x = np.asarray(x.array, dtype=np.float64)
    arr_s = np.asarray(inv_sqrt.array, dtype=np.float64)
    arr_x /= np.maximum(arr_s, 1.0e-30)
    x.setArray(arr_x.astype(PETSc.ScalarType, copy=False))


def _normalize_assembled_gnhep(A: PETSc.Mat, M: PETSc.Mat, solver_cfg: Dict) -> float:
    """Common Frobenius scaling of A and M (preserves eigenvalues; improves EPS conditioning)."""
    if not _solver_bool(solver_cfg, "gnhep_normalize_matrices", default=False):
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


def _slepc_hz_to_lambda(hz: float) -> float:
    f = max(float(hz), 0.0)
    return (2.0 * math.pi * f) ** 2


def _slepc_band_solver_strategy(solver_cfg: Dict) -> str:
    """
    Band eigensolver for coupled GNHEP.

    ``ciss`` — contour integral (non-symmetric GNHEP; interval on real λ axis).
    ``shift_invert`` — Krylov-Schur + sinvert at band center (legacy).
    ``interval`` — deprecated alias; Krylov-Schur spectrum slicing requires symmetric
    GHEP and fails on FSI GNHEP (SLEPc error 56); remapped via ``eps_interval_fallback``.
    """
    mode = str(solver_cfg.get("eps_band_solver", "shift_invert")).strip().lower()
    if mode in ("interval", "spectrum_slicing", "slice", "band_interval"):
        fb = str(solver_cfg.get("eps_interval_fallback", "ciss")).strip().lower()
        return fb if fb in ("ciss", "shift_invert") else "ciss"
    if mode in ("ciss", "contour", "contour_integral", "ciss_contour"):
        return "ciss"
    return "shift_invert"


def _slepc_interval_hz_band(
    solver_cfg: Dict,
    target_hz: float,
    min_hz: float,
) -> Tuple[float, float, float, float, float]:
    """Physical Hz band and λ interval for spectrum slicing around ``target_hz``."""
    half = float(solver_cfg.get("eps_interval_half_width_hz", 5.0))
    max_hz = float(solver_cfg.get("max_valid_mode_hz", 600.0))
    f_lo = max(float(min_hz), float(target_hz) - half)
    f_hi = min(max_hz, float(target_hz) + half)
    if f_hi <= f_lo + 1.0e-9:
        f_hi = f_lo + max(0.5, half)
    lam_lo = _slepc_hz_to_lambda(f_lo)
    lam_hi = _slepc_hz_to_lambda(f_hi)
    f_mid = 0.5 * (f_lo + f_hi)
    lam_mid = 0.5 * (lam_lo + lam_hi)
    return f_lo, f_hi, lam_lo, lam_hi, f_mid, lam_mid


def _slepc_build_mixed_block_is(
    u_to_W: Optional[np.ndarray],
    p_to_W: Optional[np.ndarray],
    comm: PETSc.Comm,
) -> Optional[Tuple[PETSc.IS, PETSc.IS]]:
    """Global row/column index sets for displacement (u) and pressure (p) blocks."""
    if u_to_W is None or p_to_W is None:
        return None
    u_idx = np.unique(np.asarray(u_to_W, dtype=np.int32).ravel())
    p_idx = np.unique(np.asarray(p_to_W, dtype=np.int32).ravel())
    if u_idx.size == 0 or p_idx.size == 0:
        return None
    is_u = PETSc.IS().createGeneral(u_idx, comm=comm)
    is_p = PETSc.IS().createGeneral(p_idx, comm=comm)
    return is_u, is_p


def _slepc_st_allow_fieldsplit(solver_cfg: Dict, *, use_ciss: bool) -> bool:
    """FieldSplit on shifted CISS operators often hits zero pivots; default off for CISS."""
    if use_ciss:
        return _solver_bool(solver_cfg, "st_ciss_use_fieldsplit", default=False)
    return str(solver_cfg.get("st_pc_type", "lu")).strip().lower() in (
        "fieldsplit",
        "fs",
    ) or _solver_bool(solver_cfg, "st_use_fieldsplit", default=False)


def _slepc_eps_destroy_safe(eps: Any) -> None:
    try:
        eps.destroy()
    except Exception as exc:
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(f"[solver][warn] EPS destroy failed (ignored): {exc}", flush=True)


def _slepc_configure_st_ksp_pc(
    ksp: PETSc.KSP,
    pc: PETSc.PC,
    solver_cfg: Dict,
    *,
    block_is: Optional[Tuple[PETSc.IS, PETSc.IS]] = None,
    opts_prefix: str = "st_",
    allow_fieldsplit: Optional[bool] = None,
    use_ciss: bool = False,
) -> Tuple[str, str]:
    """
    ST inner linear solve: direct LU (MUMPS) or block FieldSplit on (u, p).

    Returns ``(ksp_type_label, pc_type_label)`` for logging.
    """
    st_ksp_type = str(solver_cfg.get("st_ksp_type", "preonly"))
    st_pc_type_cfg = str(solver_cfg.get("st_pc_type", "lu")).strip().lower()
    if allow_fieldsplit is None:
        allow_fieldsplit = _slepc_st_allow_fieldsplit(solver_cfg, use_ciss=use_ciss)
    use_fs = bool(allow_fieldsplit) and block_is is not None
    ksp.setType(st_ksp_type)
    petsc_opts = PETSc.Options()
    st_factor = str(
        solver_cfg.get(
            "st_pc_factor_mat_solver_type",
            solver_cfg.get("st_factor_solver_type", "mumps"),
        )
    )

    if use_fs and block_is is not None:
        is_u, is_p = block_is
        pc.setType("fieldsplit")
        fs_type = str(solver_cfg.get("st_fieldsplit_type", "additive")).strip().lower()
        if fs_type in ("schur", "schur_complement"):
            pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            schur_fact = str(solver_cfg.get("st_fieldsplit_schur_fact", "full")).lower()
            _fact_map = {
                "full": PETSc.PC.FieldSplitSchurFactType.FULL,
                "upper": PETSc.PC.FieldSplitSchurFactType.UPPER,
                "lower": PETSc.PC.FieldSplitSchurFactType.LOWER,
            }
            pc.setFieldSplitSchurFactType(
                _fact_map.get(schur_fact, PETSc.PC.FieldSplitSchurFactType.FULL)
            )
            schur_pre = str(solver_cfg.get("st_fieldsplit_schur_pre", "self")).lower()
            _pre_map = {
                "self": PETSc.PC.FieldSplitSchurPreType.SELF,
                "selfp0": PETSc.PC.FieldSplitSchurPreType.SELFP0,
                "a11": PETSc.PC.FieldSplitSchurPreType.A11,
                "user": PETSc.PC.FieldSplitSchurPreType.USER,
            }
            pc.setFieldSplitSchurPreType(
                _pre_map.get(schur_pre, PETSc.PC.FieldSplitSchurPreType.SELF)
            )
            pc_label = f"fieldsplit/schur({schur_fact})"
        else:
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
            pc_label = "fieldsplit/additive"
        pc.setFieldSplitIS(("u", is_u), ("p", is_p))
        for blk in ("u", "p"):
            petsc_opts[f"{opts_prefix}fieldsplit_{blk}_ksp_type"] = str(
                solver_cfg.get(f"st_fieldsplit_{blk}_ksp_type", "preonly")
            )
            petsc_opts[f"{opts_prefix}fieldsplit_{blk}_pc_type"] = str(
                solver_cfg.get(f"st_fieldsplit_{blk}_pc_type", "lu")
            )
            petsc_opts[f"{opts_prefix}fieldsplit_{blk}_pc_factor_mat_solver_type"] = st_factor
        return st_ksp_type, pc_label

    pc.setType(st_pc_type_cfg if st_pc_type_cfg not in ("fieldsplit", "fs") else "lu")
    if pc.getType().lower() == "lu":
        try:
            pc.setFactorSolverType(st_factor)
        except Exception:
            pass
        shift_type = str(
            solver_cfg.get(
                "st_pc_factor_shift_type",
                solver_cfg.get("pc_factor_shift_type", "nonzero"),
            )
        )
        shift_amt = float(
            solver_cfg.get(
                "st_pc_factor_shift_amount",
                solver_cfg.get("pc_factor_shift_amount", 1.0e-8),
            )
        )
        petsc_opts[f"{opts_prefix}pc_factor_shift_type"] = shift_type
        petsc_opts[f"{opts_prefix}pc_factor_shift_amount"] = shift_amt
        try:
            pc.setFactorShiftType(shift_type)
            pc.setFactorShiftAmount(shift_amt)
        except Exception:
            pass
    return st_ksp_type, pc.getType()


def _slepc_clear_ciss_petsc_options() -> None:
    """Remove CISS/RG keys so a Krylov-Schur shift-invert retry is not contaminated."""
    opts = PETSc.Options()
    opts["eps_type"] = "krylovschur"
    for key in (
        "rg_type",
        "rg_interval_endpoints",
        "eps_ciss_integration_points",
        "eps_ciss_blocksize",
        "eps_ciss_moments",
        "eps_ciss_realmats",
        "eps_ciss_usest",
    ):
        try:
            opts.delValue(key)
        except Exception:
            pass


def _slepc_configure_ciss_region(
    eps: SLEPc.EPS,
    lam_lo: float,
    lam_hi: float,
    solver_cfg: Dict,
) -> None:
    """Real-axis λ interval [lam_lo, lam_hi] via SLEPc region object (RGINTERVAL)."""
    rg = eps.getRG()
    rg_type = getattr(SLEPc.RG.Type, "INTERVAL", None)
    if rg_type is not None:
        rg.setType(rg_type)
    else:
        rg.setType("interval")
    try:
        rg.setIntervalEndpoints(float(lam_lo), float(lam_hi), 0.0, 0.0)
    except AttributeError:
        petsc_opts = PETSc.Options()
        petsc_opts["rg_type"] = "interval"
        petsc_opts["rg_interval_endpoints"] = f"{lam_lo},{lam_hi},0,0"
    pad = float(solver_cfg.get("eps_ciss_imag_pad", 0.0))
    if pad > 0.0:
        try:
            rg.setIntervalEndpoints(float(lam_lo), float(lam_hi), -pad, pad)
        except Exception:
            pass


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
    u_parent_indices: Optional[np.ndarray] = None,
    n_u_global: int = 0,
    u_to_W: Optional[np.ndarray] = None,
    p_to_W: Optional[np.ndarray] = None,
    dofs_top: Optional[np.ndarray] = None,
    dofs_back: Optional[np.ndarray] = None,
    _fallback_depth: int = 0,
) -> Tuple[int, List[Tuple[float, np.ndarray, Optional[float], Optional[float], float, float]]]:
    """
    GNHEP band batch at ``shift_hz`` (λ = ω²).

    ``eps_band_solver=ciss``: contour integral (CISS) + RGINTERVAL on [λ_lo, λ_hi]
    (non-symmetric GNHEP; no Krylov-Schur inertia slicing).

    ``eps_band_solver=shift_invert``: Krylov-Schur + sinvert at band center σ.
  """
    min_hz = _slepc_spectrum_min_hz(solver_cfg, shift_hz)
    target_hz = float(shift_hz)
    target_lambda = _slepc_hz_to_lambda(target_hz)
    strategy = _slepc_band_solver_strategy(solver_cfg)
    raw_mode = str(solver_cfg.get("eps_band_solver", "shift_invert")).strip().lower()
    if raw_mode in ("interval", "spectrum_slicing", "slice", "band_interval") and MPI.COMM_WORLD.rank == ROOT_RANK:
        print(
            f"[solver][warn] eps_band_solver={raw_mode!r} is unsupported for non-symmetric GNHEP "
            f"(Krylov-Schur spectrum slicing requires symmetric GHEP); using {strategy!r} instead.",
            flush=True,
        )
    use_ciss = strategy == "ciss"
    strategy_label = "ciss_contour" if use_ciss else "shift_invert"
    st_map_name = "ciss" if use_ciss else str(solver_cfg.get("st_type", "sinvert")).strip().lower()
    shift_jitter_hz = float(solver_cfg.get("shift_jitter_hz", 0.0))
    rigid_tol = float(solver_cfg.get("coupled_rigid_lambda_tol", 1.0e-10))
    rigid_buf = int(solver_cfg.get("eps_rigid_mode_buffer", 10))
    nev_request = int(batch) + max(rigid_buf, 0)

    f_window_lo, f_window_hi, lam_lo, lam_hi, st_sigma_hz, st_sigma = _slepc_interval_hz_band(
        solver_cfg, target_hz, min_hz
    )
    broad_hz = 0.5 * (float(f_window_hi) - float(f_window_lo))

    if use_ciss:
        eps_which_label = "CISS_REGION"
        use_st_shift = False
        use_broad_window = True
    else:
        eps_which, eps_which_label, use_st_shift, use_broad_window = _slepc_eps_strategy(solver_cfg)
        broad_hz = float(solver_cfg.get("eps_broad_search_hz", 0.0))
        if use_broad_window and broad_hz <= 0.0:
            broad_hz = 1.5
        if use_broad_window and broad_hz > 0.0:
            f_window_lo = target_hz - broad_hz
            f_window_hi = target_hz + broad_hz
        use_sigma_jitter = _solver_bool(solver_cfg, "eps_st_sigma_use_jitter", default=False)
        if use_sigma_jitter and abs(shift_jitter_hz) > 0.0:
            st_sigma_hz = max(1.0, target_hz + shift_jitter_hz)
        else:
            st_sigma_hz = max(1.0, target_hz)
        st_sigma = _slepc_hz_to_lambda(st_sigma_hz)

    block_is = _slepc_build_mixed_block_is(u_to_W, p_to_W, A.getComm())
    if not use_ciss:
        _slepc_clear_ciss_petsc_options()

    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    _ciss_eps_type = getattr(SLEPc.EPS.Type, "CISS", None)
    if use_ciss:
        if _ciss_eps_type is None:
            if MPI.COMM_WORLD.rank == ROOT_RANK:
                print(
                    "[solver][warn] SLEPc.EPS.Type.CISS unavailable; falling back to shift_invert.",
                    flush=True,
                )
            use_ciss = False
            strategy_label = "shift_invert"
            st_map_name = str(solver_cfg.get("st_type", "sinvert")).strip().lower()
            eps_which, eps_which_label, use_st_shift, use_broad_window = _slepc_eps_strategy(solver_cfg)
            broad_hz = float(solver_cfg.get("eps_broad_search_hz", 0.0))
            if use_broad_window and broad_hz <= 0.0:
                broad_hz = 1.5
            if use_broad_window and broad_hz > 0.0:
                f_window_lo = target_hz - broad_hz
                f_window_hi = target_hz + broad_hz
            use_sigma_jitter = _solver_bool(solver_cfg, "eps_st_sigma_use_jitter", default=False)
            if use_sigma_jitter and abs(shift_jitter_hz) > 0.0:
                st_sigma_hz = max(1.0, target_hz + shift_jitter_hz)
            else:
                st_sigma_hz = max(1.0, target_hz)
            st_sigma = _slepc_hz_to_lambda(st_sigma_hz)
            eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        else:
            eps.setType(_ciss_eps_type)
    if not use_ciss:
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    ks_restart = float(solver_cfg.get("krylov_schur_restart", 0.5))
    if ks_restart <= 0.0:
        ks_restart = 0.5
    ks_pert = float(solver_cfg.get("eps_krylovschur_pertsize", 1.0e-8))
    if not use_ciss:
        eps.setKrylovSchurRestart(ks_restart)
        if ks_pert > 0.0:
            try:
                eps.setKrylovSchurPertSize(ks_pert)
            except AttributeError:
                pass

    st = eps.getST()
    _st_name = str(solver_cfg.get("st_type", "shift" if use_st_shift else "sinvert")).strip().lower()
    if not use_ciss:
        if use_st_shift:
            _st_name = "shift"
        if _st_name in ("shift", "stshift"):
            st.setType(SLEPc.ST.Type.SHIFT)
        else:
            st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(st_sigma)

    ksp = st.getKSP()
    pc = ksp.getPC()
    _debug_rank("Entering KSP Setup")
    st_ksp_type, st_pc_label = _slepc_configure_st_ksp_pc(
        ksp, pc, solver_cfg, block_is=block_is, opts_prefix="st_", use_ciss=use_ciss
    )
    _st_factor = str(
        solver_cfg.get(
            "st_pc_factor_mat_solver_type",
            solver_cfg.get("st_factor_solver_type", "mumps"),
        )
    )
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
        and "lu" in str(st_pc_label).lower()
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
    if use_ciss:
        petsc_opts["eps_type"] = "ciss"
        petsc_opts["rg_type"] = "interval"
        petsc_opts["rg_interval_endpoints"] = f"{lam_lo},{lam_hi},0,0"
        ciss_ip = int(solver_cfg.get("eps_ciss_integration_points", 32))
        ciss_bs = int(solver_cfg.get("eps_ciss_blocksize", max(16, nev_request)))
        ciss_ms = int(solver_cfg.get("eps_ciss_moments", 8))
        petsc_opts["eps_ciss_integration_points"] = ciss_ip
        petsc_opts["eps_ciss_blocksize"] = ciss_bs
        petsc_opts["eps_ciss_moments"] = ciss_ms
        petsc_opts["eps_ciss_realmats"] = 1
        petsc_opts["eps_ciss_usest"] = 1
    else:
        petsc_opts["eps_which"] = eps_which_label.lower()
        petsc_opts["eps_target"] = target_lambda
        petsc_opts["st_type"] = "shift" if _st_name in ("shift", "stshift") else "sinvert"
        if ks_pert > 0.0:
            petsc_opts["eps_krylovschur_pertsize"] = ks_pert
    petsc_opts["st_ksp_type"] = st_ksp_type
    petsc_opts["st_pc_type"] = pc.getType()
    if "lu" in str(st_pc_label).lower() and "fieldsplit" not in str(st_pc_label).lower():
        petsc_opts["st_pc_factor_mat_solver_type"] = _st_factor
    petsc_opts["st_ksp_norm_type"] = str(solver_cfg.get("st_ksp_norm_type", "none"))

    ncv_min_factor = float(solver_cfg.get("eps_ncv_min_factor", 4.0))
    ncv_floor = int(math.ceil(max(4.0, ncv_min_factor) * float(nev_request)))
    ncv_cfg = int(solver_cfg.get("target_ncv", 0))
    ncv = max(ncv_floor, ncv_cfg, 40)
    eps.setDimensions(int(nev_request), int(ncv))
    eps_max_it = int(solver_cfg.get("eigs_maxiter", solver_cfg.get("eps_max_it", 3000)))
    if eps_max_it_cap is not None:
        # Never cap below batch-scaled floor (legacy bug used sifter_batch_max_it ≈ 50 for nev=80).
        eps_max_it = max(int(eps_max_it_cap), int(batch) * 5, 200)
    eps_tol = float(solver_cfg.get("eps_tol", solver_cfg.get("eigs_tol", 1.0e-3)))
    eps.setTolerances(eps_tol, eps_max_it)
    _eps_conv_rel = getattr(SLEPc.EPS.Conv, "REL", None)
    if _eps_conv_rel is not None:
        try:
            eps.setConvergenceTest(_eps_conv_rel)
        except Exception:
            pass

    diag_vec = A.getDiagonal()
    diag_arr = np.real(diag_vec.array)
    if diag_arr.size > 0:
        diag_min = float(np.min(diag_arr))
        diag_max = float(np.max(diag_arr))
    else:
        diag_min = float("nan")
        diag_max = float("nan")
    _win_str = (
        f", window=[{f_window_lo:.2f}, {f_window_hi:.2f}] Hz"
        if f_window_lo is not None and f_window_hi is not None
        else ""
    )
    _fs_note = " block_is=ok" if block_is is not None else " block_is=missing"
    if use_ciss:
        ciss_ip = int(solver_cfg.get("eps_ciss_integration_points", 32))
        ciss_bs = int(solver_cfg.get("eps_ciss_blocksize", max(16, nev_request)))
        ciss_ms = int(solver_cfg.get("eps_ciss_moments", 8))
        _emit(
            f"[solver] EPS spectrum batch (target={target_hz:.2f} Hz, strategy={strategy_label}): "
            f"band=[{f_window_lo:.2f}, {f_window_hi:.2f}] Hz "
            f"(λ=[{lam_lo:.6e}, {lam_hi:.6e}]), min_hz={min_hz:.2f}, "
            f"CISS ip={ciss_ip} bs={ciss_bs} ms={ciss_ms}, "
            f"nev_request={nev_request} (batch={batch}+rigid_buf={rigid_buf}), "
            f"ncv={ncv}, eps_tol={eps_tol:.1e}, eps_max_it={eps_max_it}, "
            f"ST-KSP={st_ksp_type}, ST-PC={st_pc_label}{_fs_note}, factor={_st_factor}, "
            f"MUMPS ICNTL4={mumps_icntl_4} ICNTL6={mumps_icntl_6} ICNTL12={mumps_icntl_12} "
            f"ICNTL14={mumps_icntl_14} ICNTL24={mumps_icntl_24}, "
            f"diag_shift={diag_shift:.2e}, A_diag_min={diag_min:.6e}, A_diag_max={diag_max:.6e}",
            status_callback=status_callback,
        )
    else:
        _emit(
            f"[solver] EPS spectrum batch (target={target_hz:.2f} Hz, strategy={strategy_label}): "
            f"target_lambda={target_lambda:.6e}, min_hz={min_hz:.2f}, "
            f"ST={_st_name} sigma={st_sigma_hz:.2f} Hz (jitter={shift_jitter_hz:.2f}), "
            f"which={eps_which_label}{_win_str}, conv=REL, ks_restart={ks_restart:.3f}, "
            f"ks_pert={ks_pert:.2e}, "
            f"nev_request={nev_request} (batch={batch}+rigid_buf={rigid_buf}), "
            f"ncv={ncv}, eps_tol={eps_tol:.1e}, eps_max_it={eps_max_it}, "
            f"ST-KSP={st_ksp_type}, ST-PC={st_pc_label}{_fs_note}, factor={_st_factor}, "
            f"MUMPS ICNTL4={mumps_icntl_4} ICNTL6={mumps_icntl_6} ICNTL12={mumps_icntl_12} "
            f"ICNTL14={mumps_icntl_14} ICNTL24={mumps_icntl_24}, "
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
    # Re-apply after setFromOptions so CLI/options cannot override GNHEP band strategy.
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    if use_ciss:
        _slepc_configure_ciss_region(eps, lam_lo, lam_hi, solver_cfg)
        ciss_ip = int(solver_cfg.get("eps_ciss_integration_points", 32))
        ciss_bs = int(solver_cfg.get("eps_ciss_blocksize", max(16, nev_request)))
        ciss_ms = int(solver_cfg.get("eps_ciss_moments", 8))
        try:
            eps.setCISSSizes(
                ip=ciss_ip,
                bs=ciss_bs,
                ms=ciss_ms,
                realmats=True,
            )
            eps.setCISSUseST(True)
        except AttributeError:
            pass
        st = eps.getST()
        ksp_st = st.getKSP()
        _slepc_configure_st_ksp_pc(
            ksp_st,
            ksp_st.getPC(),
            solver_cfg,
            block_is=block_is,
            opts_prefix="st_",
            use_ciss=True,
        )
    else:
        eps.setWhichEigenpairs(eps_which)
        eps.setTarget(target_lambda)
        st = eps.getST()
        if _st_name in ("shift", "stshift"):
            st.setType(SLEPc.ST.Type.SHIFT)
        else:
            st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(st_sigma)
    if _eps_conv_rel is not None:
        try:
            eps.setConvergenceTest(_eps_conv_rel)
        except Exception:
            pass
    print(f"[HEARTBEAT] Rank {MPI.COMM_WORLD.rank} reached EPS Solve")
    sys.stdout.flush()
    _debug_rank("Entering EPS Solve")
    try:
        eps.solve()
    except Exception as exc:
        _emit(
            f"[solver][warn] EPS solve() raised (will still harvest nconv>0 if any): {exc}",
            status_callback=status_callback,
            level="warning",
        )

    its = eps.getIterationNumber()
    nconv_marked = int(eps.getConverged())
    reason = eps.getConvergedReason()
    force_partial = _solver_bool(solver_cfg, "eps_force_harvest_partial", True)
    harvest_slots = nconv_marked
    if harvest_slots == 0 and force_partial:
        harvest_slots = int(nev_request)
    _emit(
        f"[solver] EPS sweep (scheduler={shift_hz:.1f} Hz): iterations={its}, "
        f"nconv_marked={nconv_marked}, harvest_slots={harvest_slots}, "
        f"requested={nev_request}, min_hz={min_hz:.2f}, reason={reason}",
        status_callback=status_callback,
    )
    if reason < 0 and nconv_marked > 0:
        _emit(
            f"[solver][warn] EPS reason={reason} (not CONVERGED_TOL) but harvesting {nconv_marked} "
            f"converged pair(s) anyway.",
            status_callback=status_callback,
            level="warning",
        )
    elif reason < 0 and harvest_slots > 0 and nconv_marked == 0:
        _emit(
            f"[solver][warn] EPS reason={reason}; attempting partial harvest up to "
            f"{harvest_slots} Ritz slots (force_partial).",
            status_callback=status_callback,
            level="warning",
        )

    phi_u: Optional[PETSc.Vec] = None
    work_u: Optional[PETSc.Vec] = None
    if work is not None and (M_top is not None or M_back is not None) and u_parent_indices is not None:
        phi_u = M_top.createVecRight() if M_top is not None else M_back.createVecRight()
        work_u = phi_u.duplicate()

    rank_by_wood = _solver_bool(solver_cfg, "eps_harvest_rank_by_wood", default=True)
    min_wood_harvest = float(solver_cfg.get("eps_harvest_min_wood", 0.01))
    reject_decoupled = _solver_bool(solver_cfg, "eps_reject_decoupled_u_only", default=True)
    min_pressure_frac = float(solver_cfg.get("eps_harvest_min_pressure_fraction", 0.02))
    reject_sigma_spurious = _solver_bool(solver_cfg, "eps_reject_sigma_spurious", default=True)
    if use_ciss:
        reject_sigma_spurious = False
    sigma_spurious_hz = float(solver_cfg.get("eps_sigma_spurious_tol_hz", 0.35))
    sigma_p_frac_max = float(solver_cfg.get("eps_sigma_spurious_max_p_frac", 1.0e-3))
    reject_target_locked = _solver_bool(solver_cfg, "eps_reject_target_locked", default=True)

    candidates: List[Tuple[float, float, np.ndarray, float, float, float, float, float]] = []
    rvec = A.createVecRight()
    skipped_rigid = 0
    skipped_below_min = 0
    skipped_window = 0
    skipped_unavailable = 0
    skipped_decoupled = 0
    skipped_sigma = 0
    allow_weak = _solver_bool(
        solver_cfg,
        "eps_harvest_allow_weak_coupling",
        default=not use_ciss,
    )
    map_tag_counts: Dict[str, int] = {}
    raw_eig_samples: List[float] = []
    for i in range(int(harvest_slots)):
        try:
            eig = eps.getEigenpair(i, rvec)
        except Exception:
            skipped_unavailable += 1
            if i < nconv_marked:
                raise
            continue
        eig_r = float(np.real(eig))
        lam_phys, lam_map_tag = _slepc_physical_lambda(
            eig_r, st_sigma, st_map_name, solver_cfg
        )
        map_tag_counts[lam_map_tag] = map_tag_counts.get(lam_map_tag, 0) + 1
        if len(raw_eig_samples) < 5:
            raw_eig_samples.append(eig_r)
        if (not math.isfinite(lam_phys)) or lam_phys <= rigid_tol:
            skipped_rigid += 1
            continue
        omega = math.sqrt(max(lam_phys, 0.0))
        f_hz = omega / (2.0 * math.pi)
        rt = 0.0
        rb = 0.0
        if work is not None and (M_top is not None or M_back is not None):
            rt, rb = _plate_modal_energy_ratios(
                rvec,
                M_top,
                M_back,
                work,
                mass_top,
                mass_back,
                u_parent_indices=u_parent_indices,
                phi_u=phi_u,
                work_u=work_u,
                u_to_W=u_to_W if u_to_W is not None else u_parent_indices,
                dofs_top=dofs_top,
                dofs_back=dofs_back,
            )
        wood = float(rt) + float(rb)
        arr = rvec.array.copy()
        u_n, p_n = _mixed_eigenvector_block_norms(
            arr, n_u_global, u_to_W=u_to_W, p_to_W=p_to_W
        )
        p_frac = p_n / max(u_n + p_n, 1.0e-30)
        p_block_max = _max_pressure_block_abs(arr, p_to_W)
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                f"[eps-p-diag] converged slot={i} raw_eig={eig_r:.6e} "
                f"lam_phys={lam_phys:.6e} map={lam_map_tag} f={f_hz:.6f} Hz "
                f"max|p|={p_block_max:.6e} p_frac={p_frac:.3e} wood={wood:.4f}",
                flush=True,
            )
        if f_hz + 1e-9 < float(min_hz):
            skipped_below_min += 1
            continue
        if f_window_lo is not None and f_window_hi is not None:
            if f_hz < f_window_lo - 1.0e-9 or f_hz > f_window_hi + 1.0e-9:
                skipped_window += 1
                continue
        if p_frac < 1.0e-5 and MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                f"[FSI-AUDIT][warn] Mode f={f_hz:.2f} Hz: p_frac={p_frac:.3e} — "
                "coupling effectively absent (||p|| negligible vs ||u||); "
                "check fsi_coupling_gain / pressure_dof_scale."
            )
        if (
            not allow_weak
            and reject_decoupled
            and p_frac < min_pressure_frac
        ):
            skipped_decoupled += 1
            continue
        if reject_sigma_spurious and p_frac < sigma_p_frac_max:
            if abs(f_hz - st_sigma_hz) <= sigma_spurious_hz + 1.0e-9:
                skipped_sigma += 1
                continue
            if (
                reject_target_locked
                and abs(f_hz - target_hz) <= sigma_spurious_hz + 1.0e-9
            ):
                skipped_sigma += 1
                continue
        rank_by_p = _solver_bool(solver_cfg, "eps_harvest_rank_by_p_frac", default=False)
        rank_score = (p_frac + 0.25 * wood) if rank_by_p else (wood + 0.25 * p_frac)
        candidates.append((rank_score, f_hz, arr, float(rt), float(rb), u_n, p_n, p_block_max))

    if rank_by_wood and candidates:
        candidates.sort(key=lambda t: (-t[0], abs(t[1] - target_hz)))
        strong = [c for c in candidates if (c[3] + c[4]) >= min_wood_harvest - 1.0e-15]
        weak = [c for c in candidates if (c[3] + c[4]) < min_wood_harvest - 1.0e-15]
        ordered = strong + weak
    else:
        ordered = candidates

    out: List[Tuple[float, np.ndarray, Optional[float], Optional[float], float, float]] = []
    for _score, f_hz, arr, rt, rb, u_n, p_n, p_block_max in ordered:
        p_frac = p_n / max(u_n + p_n, 1.0e-30)
        out.append((f_hz, arr, rt, rb, float(p_frac), float(p_block_max)))
        if len(out) >= int(batch):
            break

    if MPI.COMM_WORLD.rank == ROOT_RANK:
        n_woodish = sum(1 for c in candidates if (c[3] + c[4]) >= min_wood_harvest)
        max_wood = max((c[3] + c[4] for c in candidates), default=0.0)
        freqs = [c[1] for c in candidates]
        f_span = (max(freqs) - min(freqs)) if freqs else 0.0
        if len(freqs) >= 10 and f_span < 0.05:
            print(
                f"[solver][warn] EPS harvest: f_span={f_span:.6f} Hz at ST_sigma={st_sigma_hz:.2f} Hz "
                f"({len(freqs)} candidates) — σ-cluster; check spurious Ritz / rigid null-space "
                f"or widen eps_broad_search_hz (now {broad_hz:.2f}).",
                flush=True,
            )
        print(
            f"[solver] EPS harvest: kept={len(out)}/{batch} from nconv_marked={nconv_marked} "
            f"(skip rigid={skipped_rigid}, f<{min_hz:.1f}Hz={skipped_below_min}, "
            f"outside_window={skipped_window}, unavail={skipped_unavailable}, "
            f"skip_decoupled={skipped_decoupled}, skip_sigma={skipped_sigma}, "
            f"allow_weak_coupling={allow_weak}, rank_by_wood={rank_by_wood}, "
            f"reject_decoupled={reject_decoupled}, reject_sigma_spurious={reject_sigma_spurious})"
        )
        _sigma_lbl = (
            f"band=[{f_window_lo:.2f},{f_window_hi:.2f}] Hz"
            if use_ciss and f_window_lo is not None and f_window_hi is not None
            else f"ST_sigma={st_sigma_hz:.2f} Hz"
        )
        print(
            f"[solver] EPS wood scan: slots={len(candidates)}, wood>={min_wood_harvest:.3f}: "
            f"{n_woodish}, max_wood={max_wood:.4f}, f_span={f_span:.3f} Hz, {_sigma_lbl}"
        )
        if len(candidates) == 0 and nconv_marked > 0:
            _lam_lo, _lam_hi = _slepc_lambda_hz_bounds(solver_cfg)
            tags = ", ".join(f"{k}={v}" for k, v in sorted(map_tag_counts.items()))
            raw_s = ", ".join(f"{x:.3e}" for x in raw_eig_samples)
            print(
                f"[solver][warn] EPS harvest: 0 slots after filters from nconv_marked={nconv_marked} "
                f"(ST={_st_name}, σ={st_sigma:.3e}); map_tags: {tags or 'none'}; "
                f"skip_decoupled={skipped_decoupled}, skip_sigma={skipped_sigma}, "
                f"allow_weak_coupling={allow_weak}; sample raw_eig: [{raw_s}]. "
                f"Set eps_harvest_allow_weak_coupling=true or lower eps_harvest_min_pressure_fraction.",
                flush=True,
            )
        for j, (_score, f_hz, _arr, rt, rb, u_n, p_n, p_block_max) in enumerate(
            sorted(candidates, key=lambda t: -t[0])[: min(5, len(candidates))]
        ):
            p_frac = p_n / max(u_n + p_n, 1.0e-30)
            print(
                f"[solver]   wood-rank {j + 1}: f={f_hz:.2f} Hz wood={rt + rb:.4f} "
                f"tag1={rt:.4f} tag3={rb:.4f} ||u||={u_n:.3e} ||p||={p_n:.3e} "
                f"max|p|={p_block_max:.3e} p_frac={p_frac:.3e}"
            )
        sys.stdout.flush()

    for _v in (phi_u, work_u):
        if _v is not None:
            try:
                _v.destroy()
            except Exception:
                pass

    if len(out) < int(batch) and MPI.COMM_WORLD.rank == ROOT_RANK:
        _emit(
            f"[solver][warn] EPS harvested {len(out)}/{batch} modes with f>={min_hz:.1f} Hz "
            f"(nconv_marked={nconv_marked}); increase eigs_maxiter (now {eps_max_it}) or target_ncv (now {ncv}).",
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
    _slepc_eps_destroy_safe(eps)

    if (
        len(out) == 0
        and use_ciss
        and int(_fallback_depth) == 0
        and _solver_bool(solver_cfg, "eps_ciss_fallback_shift_invert", default=True)
    ):
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                "[solver][warn] CISS band solve returned 0 harvestable modes; "
                "retrying with shift_invert + monolithic MUMPS LU.",
                flush=True,
            )
        fb_cfg = dict(solver_cfg)
        fb_cfg["eps_band_solver"] = "shift_invert"
        fb_cfg["st_pc_type"] = "lu"
        fb_cfg["st_use_fieldsplit"] = False
        fb_cfg["st_ciss_use_fieldsplit"] = False
        _slepc_clear_ciss_petsc_options()
        n_fb, out_fb = _slepc_shift_invert_batch(
            A,
            M,
            fb_cfg,
            shift_hz,
            batch,
            diag_shift,
            status_callback,
            M_top=M_top,
            M_back=M_back,
            work=work,
            mass_top=mass_top,
            mass_back=mass_back,
            eps_max_it_cap=eps_max_it_cap,
            u_parent_indices=u_parent_indices,
            n_u_global=n_u_global,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            dofs_top=dofs_top,
            dofs_back=dofs_back,
            _fallback_depth=1,
        )
        return n_fb, out_fb

    return len(out), out


def _slepc_spectrum_batch(
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, List[Tuple[float, np.ndarray, Optional[float], Optional[float], float, float]]]:
    """Dispatch wrapper: CISS contour band solve or Krylov-Schur shift-invert batch."""
    return _slepc_shift_invert_batch(*args, **kwargs)


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

    solver_cfg = config.get("solver", {})
    diag_tags = _structural_diag_facet_tags(solver_cfg)
    tag_top = int(diag_tags[0]) if len(diag_tags) > 0 else 1
    tag_back = int(diag_tags[1]) if len(diag_tags) > 1 else 3

    facets_t1 = np.array(facet_tags.find(tag_top), dtype=np.int32)
    facets_t2 = np.array(facet_tags.find(2), dtype=np.int32)
    facets_t3 = np.array(facet_tags.find(tag_back), dtype=np.int32)
    facets_t4 = np.array(facet_tags.find(RIBS_SURFACE_TAG), dtype=np.int32)
    facets_fix = np.array(facet_tags.find(WOOD_FIX_SURFACE_TAG), dtype=np.int32)

    print(
        f"[DIAG] facet tag counts: top(tag{tag_top})={facets_t1.size}, "
        f"soundhole(tag2)={facets_t2.size}, back(tag{tag_back})={facets_t3.size}, "
        f"ribs(tag4)={facets_t4.size}, fix(tag5)={facets_fix.size}"
    )
    sys.stdout.flush()

    # Hard-coded P1 displacement (same element factory as coupled W.sub(0)).
    _u_deg_struct = 1
    u_el = _displacement_element(msh, _u_deg_struct)
    V_u = fem.functionspace(msh, u_el)
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)

    top_m, back_m, t_top, t_back = _split_wood_materials(config)
    top_mat = config["materials"]["top"]
    back_mat = config["materials"]["back"]
    if msh.comm.rank == ROOT_RANK:
        print(
            f"[DIAG] Structural-only shell diagnostic: facet tags {diag_tags} "
            f"(top={tag_top}, back={tag_back}); NO fluid/pressure DOFs."
        )
        print(
            f"[DIAG] Material audit: "
            f"top(D11={top_m['D11']:.3e}, E_L={top_m['E_L']:.3e}, rho={top_m['rho']:.1f}), "
            f"back(D11={back_m['D11']:.3e}, E_L={back_m['E_L']:.3e}, rho={back_m['rho']:.1f}), "
            f"t_top={t_top:.4f} m t_back={t_back:.4f} m | "
            f"{top_mat.get('name', '')!r} / {back_mat.get('name', '')!r}"
        )

    xdmf_ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    n = ufl.FacetNormal(msh)
    P = ufl.Identity(3) - ufl.outer(n, n)

    def eps_surface(uu):
        grad_u = ufl.grad(uu)
        grad_tan = P * grad_u * P
        return 0.5 * (grad_tan + ufl.transpose(grad_tan))

    ds_top = xdmf_ds(tag_top)
    ds_back = xdmf_ds(tag_back)
    eps_u = eps_surface(u)
    eps_v = eps_surface(v)
    w_n = ufl.dot(u, n)
    v_n = ufl.dot(v, n)
    e1, e2 = _plate_local_frame(n, P)
    shell_top = _orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
    shell_back = _orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
    shell_ribs = _orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
    a_uu = shell_top * ds_top + shell_back * ds_back
    m_uu = (top_m["rho"] * t_top) * ufl.dot(u, v) * ds_top + (back_m["rho"] * t_back) * ufl.dot(u, v) * ds_back
    include_ribs_shell = _solver_bool(solver_cfg, "structural_shell_include_ribs", default=True)
    if include_ribs_shell and facets_t4.size > 0:
        ds_ribs = xdmf_ds(RIBS_SURFACE_TAG)
        a_uu = a_uu + shell_ribs * ds_ribs
        m_uu = m_uu + (back_m["rho"] * t_back) * ufl.dot(u, v) * ds_ribs
        if msh.comm.rank == ROOT_RANK:
            print(
                f"[DIAG] structural-only: rib shell (tag {RIBS_SURFACE_TAG}) included for "
                f"top/back structural coupling ({facets_t4.size} facets)."
            )

    _audit_u_subspace_global(V_u, label="structural-only V_u")
    _audit_shell_facet_dof_coverage(
        msh,
        facet_tags,
        V_u,
        tag_top=tag_top,
        tag_back=tag_back,
        tag_ribs=RIBS_SURFACE_TAG,
        label="structural-only V_u",
        constrained_u_dofs=None,
        status_callback=status_callback,
    )

    _diagnose_shell_stiffness_assembly(
        a_uu,
        shell_top,
        shell_back,
        ds_top,
        ds_back,
        tag_top,
        tag_back,
        int(np.sum(facet_tags.values == tag_top)),
        int(np.sum(facet_tags.values == tag_back)),
        status_callback=status_callback,
    )

    m_uu_top_plate = (top_m["rho"] * t_top) * ufl.dot(u, v) * ds_top
    m_uu_back_shell = (back_m["rho"] * t_back) * ufl.dot(u, v) * ds_back

    # Structural diagnostic: free–free; no displacement BC on tags 1/3/4/5 (see coupled clamp_ribs).
    bcs_u: List = []
    if msh.comm.rank == ROOT_RANK:
        print(
            "[DIAG] structural-only: free–free (no displacement DirichletBC; "
            f"tag-{tag_top} top / tag-{tag_back} back / tag-{RIBS_SURFACE_TAG} ribs unconstrained)."
        )

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
    K = assemble_matrix(a_uu_form, bcs=bcs_u)
    K.assemble()
    M = assemble_matrix(m_uu_form, bcs=bcs_u)
    M.assemble()
    M_top = assemble_matrix(fem.form(m_uu_top_plate), bcs=bcs_u)
    M_top.assemble()
    M_back = assemble_matrix(fem.form(m_uu_back_shell), bcs=bcs_u)
    M_back.assemble()
    mass_top_kg = float(fem.assemble_scalar(fem.form(top_m["rho"] * t_top * ds_top)))
    mass_back_kg = float(fem.assemble_scalar(fem.form(back_m["rho"] * t_back * ds_back)))
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(
            f"[DIAG] Shell lumped mass: mass_top={mass_top_kg:.4e} kg, "
            f"mass_back={mass_back_kg:.4e} kg"
        )
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
    plate_ratios: List[Tuple[float, float]] = []
    work = M.createVecRight()
    for i in range(min(nev, nconv)):
        eig = eps.getEigenpair(i, rvec)
        eig_r = float(np.real(eig))
        all_eigs.append(eig_r)
        if eig_r <= rigid_tol:
            skipped_rigid += 1
            continue
        f_hz = math.sqrt(max(eig_r, 0.0)) / (2.0 * math.pi)
        if f_hz < min_structural_hz:
            skipped_low_hz += 1
            continue
        rt, rb = _plate_modal_energy_ratios(rvec, M_top, M_back, work, mass_top_kg, mass_back_kg)
        freqs_hz.append(f_hz)
        vectors.append(rvec.array.copy())
        plate_ratios.append((float(rt), float(rb)))
    try:
        work.destroy()
    except Exception:
        pass
    try:
        rvec.destroy()
    except Exception:
        pass
    try:
        M_top.destroy()
        M_back.destroy()
    except Exception:
        pass
    eps.destroy()

    print("[DIAG] Structural-only eigen search: TARGET_MAGNITUDE (shell facet tags 1+3)")
    print(
        f"[DIAG] Structural-only rigid filter: lambda_tol={rigid_tol:.3e}, "
        f"skipped_rigid={skipped_rigid}, min_mode_hz={min_structural_hz:.2f}, skipped_low_hz={skipped_low_hz}"
    )
    print(f"[DIAG] Raw eigenvalues: {[float(x) for x in all_eigs[:10]]}")
    sys.stdout.flush()

    if not freqs_hz:
        raise RuntimeError("Structural-only diagnosis: no positive eigenvalues.")
    order = np.argsort(np.array(freqs_hz))
    order_list = [int(i) for i in order][:nev_target]
    freqs_hz = [freqs_hz[i] for i in order_list]
    plate_ratios = [plate_ratios[i] for i in order_list]
    eigvecs = np.stack([vectors[i] for i in order_list], axis=1)

    print(f"[DIAG] Structural-only first {len(freqs_hz)} mode(s): {[round(f, 3) for f in freqs_hz]}")
    dofs_top_mode = _locate_facet_displacement_dofs(V_u, msh, facets_t1)
    dofs_back_mode = _locate_facet_displacement_dofs(V_u, msh, facets_t3)
    for idx, (f_hz, (rt, rb)) in enumerate(zip(freqs_hz[:5], plate_ratios[:5])):
        wood = rt + rb
        loc = "top-dominated" if rt > 0.6 else "back-dominated" if rb > 0.6 else "mixed/localized"
        print(
            f"[DIAG]   mode {idx + 1}: f={f_hz:.2f} Hz, tag1_ratio={rt:.4f}, tag3_ratio={rb:.4f}, "
            f"wood_participation={wood:.4f} ({loc})"
        )
        if idx == 0 and eigvecs.size > 0:
            u_mode = eigvecs[:, 0]
            def _dof_kinetic_energy(dof_idx: np.ndarray) -> float:
                if dof_idx.size == 0:
                    return 0.0
                vals = u_mode[np.asarray(dof_idx, dtype=np.int32)]
                return float(np.dot(vals, vals))

            e_top_dof = _dof_kinetic_energy(dofs_top_mode)
            e_back_dof = _dof_kinetic_energy(dofs_back_mode)
            print(
                f"[DIAG]   mode 1 dof kinetic proxy: top len(dofs_top)={dofs_top_mode.size} "
                f"E_top={e_top_dof:.6e}, back len(dofs_back)={dofs_back_mode.size} E_back={e_back_dof:.6e}"
            )
            if dofs_top_mode.size == 0:
                print(
                    "[DIAG][CRITICAL] mode 1: len(dofs_top)=0 — top plate facets are not coupled to V_u DOFs."
                )
            elif e_top_dof < 1.0e-24 and e_back_dof > 1.0e-24:
                print(
                    "[DIAG][warn] mode 1: back vibrates but top dof energy ≈ 0 — "
                    "check shell_top*ds(tag1) assembly or disconnected top/back without rib shell."
                )
    band_lo = float(solver_cfg.get("structural_expected_hz_min", 80.0))
    band_hi = float(solver_cfg.get("structural_expected_hz_max", 200.0))
    in_band = [f for f in freqs_hz[:5] if band_lo <= f <= band_hi]
    print(
        f"[DIAG] Modes in expected band [{band_lo:.0f}, {band_hi:.0f}] Hz (first 5): "
        f"{[round(f, 1) for f in in_band]} ({'OK' if in_band else 'CHECK mesh/materials'})"
    )
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
    probe_spec: Optional[Dict[str, Any]] = None,
):
    _phase_sync(2000, "coupled enter", status_callback=status_callback)
    solver_cfg = config.get("solver", {})
    _struct_only = _is_structural_only_run(config, solve_evp)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        if _struct_only:
            print("🟢 STRUCTURAL-ONLY SHELL DIAGNOSTIC (couple_fluid=False, facet tags 1+3, no FSI)")
        else:
            print("🔴 FULL COUPLED ACOUSTIC–STRUCTURAL MODEL (mixed u,p + FSI)")
        sys.stdout.flush()
    _ams = solver_cfg.get("adaptive_mode_sifter", "<missing>")
    _sihz = solver_cfg.get("shift_invert_target_hz", "<missing>")
    print(
        f"[DEBUG] Sifter status: adaptive_mode_sifter={_ams!r} "
        f"(effective={_solver_bool(solver_cfg, 'adaptive_mode_sifter', True)}), "
        f"shift_invert_target_hz={_sihz!r}, solve_evp={solve_evp}, "
        f"couple_fluid={_solver_bool(solver_cfg, 'couple_fluid', True)}, "
        f"structural_only={_struct_only}"
    )
    sys.stdout.flush()

    msh, cell_tags, facet_tags = _load_mesh_and_tags(mesh_file, status_callback=status_callback)
    _audit_and_scale_mesh_units(msh, config, status_callback=status_callback)
    _mesh_interface_diagnostic(msh, cell_tags, facet_tags, status_callback=status_callback)
    _phase_sync(2001, "coupled after mesh load", status_callback=status_callback)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(
            f"[diag] branch: {'structural-only shell (tags {STRUCTURAL_DIAG_SURFACE_TAGS})' if _struct_only else 'full coupled (mixed u,p)'}"
        )
        sys.stdout.flush()
    if _struct_only:
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
    u_el = _displacement_element(msh, _u_deg_coupled)
    p_el = element("Lagrange", msh.basix_cell(), _p_deg_coupled)
    W_el = mixed_element([u_el, p_el])
    W = fem.functionspace(msh, W_el)
    V_u_standalone, V_u_collapsed, u_parent_indices = _audit_coupled_displacement_space(
        msh, W, u_el, status_callback=status_callback
    )
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

    # Trial/test on W.sub(0) and W.sub(1) so coupling blocks map to the correct mixed DOFs.
    W_u = W.sub(0)
    W_p = W.sub(1)
    u = ufl.TrialFunction(W_u)
    p = ufl.TrialFunction(W_p)
    v = ufl.TestFunction(W_u)
    q = ufl.TestFunction(W_p)
    _audit_mixed_w_dof_maps(W, status_callback=status_callback)

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

    fsi_iface_facets = _find_air_wood_interface_facets(msh, cell_tags)
    air_ext_p_facets = _find_air_exterior_pressure_facets(
        msh, cell_tags, facet_tags, fsi_iface_facets
    )
    if fsi_iface_facets.size == 0:
        raise RuntimeError(
            "FSI interface: no facets found between air volume (tag 10) and wood volumes (1/2/3). "
            "Check mesh boolean/fragment and cell_tags."
        )
    iface_meshtags = _meshtags_on_facets(msh, fsi_iface_facets, FSI_INTERFACE_FACET_TAG)
    iface_ds = ufl.Measure("ds", domain=msh, subdomain_data=iface_meshtags)(
        FSI_INTERFACE_FACET_TAG
    )
    h_char = _mesh_characteristic_length(msh)
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        n_wood_tagged = int(
            np.sum(
                np.isin(
                    facet_tags.values,
                    np.asarray(WOOD_SURFACE_TAGS, dtype=np.int32),
                )
            )
        )
        print(
            f"[FSI-IFACE] topology interface facets={fsi_iface_facets.size} "
            f"(wood tagged shell facets={n_wood_tagged}, ribs excluded from iface measure); "
            f"air exterior p=0 facets={air_ext_p_facets.size}, h_char={h_char:.4e} m"
        )
        sys.stdout.flush()

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
    try:
        a_uu_shell_norm = _mat_frobenius_norm(a_uu, label="a_uu_shell_pre_nitsche")
    except Exception as exc:
        a_uu_shell_norm = float("nan")
        _emit(
            f"[form][warn] shell ||A_uu|| reference failed: {type(exc).__name__}: {exc!r}",
            status_callback=status_callback,
            level="warning",
        )

    # Pressure DOF similarity scale: balances u (~1e9) and acoustic (~1e1) block magnitudes.
    p_scale = _coupled_pressure_dof_scale(config.get("solver", {}))
    p2 = p_scale * p_scale
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print(f"[form] coupled pressure_dof_scale={p_scale:.4e} (similarity on all p-blocks)")
        sys.stdout.flush()

    # Acoustic stiffness in internal air volume.
    a_pp = p2 * (1.0 / rho_air) * ufl.inner(ufl.grad(p), ufl.grad(q)) * xdmf_dx(AIR_VOLUME_TAG)

    # FSI on wood↔air interface facets only (topology), not exterior ribs (tag 4).
    fsi_gain = _fsi_coupling_gain(config.get("solver", {}))
    norm_uu_for_nitsche = (
        a_uu_shell_norm
        if math.isfinite(a_uu_shell_norm) and a_uu_shell_norm > 0.0
        else 1.083155e10
    )
    gamma_n = _fsi_nitsche_gamma(
        solver_cfg,
        norm_uu_ref=norm_uu_for_nitsche,
        h_char=h_char,
    )
    use_nitsche = _solver_bool(solver_cfg, "fsi_nitsche_enable", default=True)
    traction_up = -p_scale * fsi_gain * p * ufl.dot(n, v) * iface_ds
    mass_pu = p_scale * fsi_gain * rho_air * ufl.dot(u, n) * q * iface_ds
    if use_nitsche:
        nit_uu = gamma_n * ufl.dot(u, n) * ufl.dot(v, n) * iface_ds
        nit_up = gamma_n * p * ufl.dot(v, n) * iface_ds
        nit_pu = gamma_n * ufl.dot(u, n) * q * iface_ds
        a_uu = a_uu + nit_uu
        a_up = traction_up + nit_up
        m_pu = mass_pu + nit_pu
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                f"[form] FSI Nitsche on iface: gamma_n={gamma_n:.6e}, "
                f"fsi_gain={fsi_gain:.4g}, iface_facets={fsi_iface_facets.size}"
            )
            sys.stdout.flush()
    else:
        a_up = traction_up
        m_pu = mass_pu

    _audit_pressure_scale_block_balance(
        a_uu,
        a_pp,
        a_up,
        p_scale=p_scale,
        fsi_gain=fsi_gain,
        tag_top=tag_top,
        n_facets_top=wood_tag_top,
        norm_uu_shell=a_uu_shell_norm if math.isfinite(a_uu_shell_norm) else None,
        status_callback=status_callback,
    )

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

    _print_raw_coupling_block_norms(a_up, m_pu, status_callback=status_callback)

    acoustic_mass_scale = float(
        solver_cfg.get("acoustic_mass_scale", solver_cfg.get("mass_scale", 1.0))
    )
    if acoustic_mass_scale != 1.0:
        m_pp = acoustic_mass_scale * m_pp
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                f"[form] acoustic_mass_scale={acoustic_mass_scale:.6e} applied to M_pp only "
                f"(air mass block; M_pu unchanged)"
            )
            sys.stdout.flush()

    # Pressure-only regularization (optional); displacement is free–free (no reg_u).
    diag_shift = float(config.get("solver", {}).get("diag_shift", 0.0))
    reg_p = p2 * diag_shift * p * q * xdmf_dx(AIR_VOLUME_TAG)

    try:
        norm_pp_ref = _mat_frobenius_norm(a_pp, label="a_pp_pre_probe")
    except Exception:
        norm_pp_ref = 2.524481e03
    norm_uu_ref = (
        a_uu_shell_norm
        if math.isfinite(a_uu_shell_norm) and a_uu_shell_norm > 0.0
        else norm_uu_for_nitsche
    )

    if probe_spec is not None:
        a_uu, reg_p, _k_u, _k_p, _nu, _np = _append_resolvent_probe_stabilization(
            a_uu,
            a_pp,
            reg_p,
            u=u,
            v=v,
            p=p,
            q=q,
            p2=p2,
            wood_ds=wood_ds,
            air_dx=xdmf_dx(AIR_VOLUME_TAG),
            solver_cfg=solver_cfg,
            norm_uu_ref=norm_uu_ref,
            norm_pp_ref=norm_pp_ref,
            status_callback=status_callback,
        )

    block_solver_cfg = (
        _resolvent_probe_block_solver_cfg(solver_cfg)
        if probe_spec is not None
        else solver_cfg
    )
    a_uu, a_pp, a_up, m_uu, m_pp, m_pu, reg_p, _s_uu, _s_pp = _block_frobenius_normalize_coupled_forms(
        a_uu,
        a_pp,
        a_up,
        m_uu,
        m_pp,
        m_pu,
        reg_p,
        block_solver_cfg,
        status_callback=status_callback,
    )

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
    bcs_u_only: List = []
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
            p_gauge_facets = soundhole_facets
            if air_ext_p_facets.size > 0:
                p_gauge_facets = np.unique(
                    np.concatenate([soundhole_facets, air_ext_p_facets]).astype(np.int32)
                )
            p_dofs = np.array(
                fem.locate_dofs_topological(V_p, fdim, p_gauge_facets),
                dtype=np.int32,
            )
            _emit(
                f"[bc] pressure gauge: P=0 on soundhole ({soundhole_facets.size} facets)"
                + (
                    f" + air exterior ({air_ext_p_facets.size} facets, iface excluded)"
                    if air_ext_p_facets.size > 0
                    else ""
                )
                + f" → {p_dofs.size} pressure DOFs.",
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

        V_u = V_u_collapsed
        _audit_u_subspace_global(
            W.sub(0),
            label="coupled W.sub(0)",
            V_u_collapsed=V_u,
            status_callback=status_callback,
        )
        if MPI.COMM_WORLD.rank == ROOT_RANK:
            print(
                "[COUPLED-SPACE] BC/displacement map uses V_u_collapsed; "
                f"global n_u={_u_global_dof_count(V_u)}, "
                f"collapse_map_len={len(np.asarray(u_parent_indices, dtype=np.int32))}, "
                f"standalone n_u={_u_global_dof_count(V_u_standalone)}"
            )
            sys.stdout.flush()
        facets_ribs = np.array(facet_tags.find(RIBS_SURFACE_TAG), dtype=np.int32)
        u_dofs_ribs = _locate_facet_displacement_dofs(V_u, msh, facets_ribs)
        _audit_shell_facet_dof_coverage(
            msh,
            facet_tags,
            V_u,
            tag_top=tag_top,
            tag_back=tag_back,
            tag_ribs=RIBS_SURFACE_TAG,
            label="coupled V_u (collapsed)",
            constrained_u_dofs=None,
            status_callback=status_callback,
        )
        _emit(
            "[bc][diag] pressure gauge + optional ribs clamp (tag 4 only). "
            f"pressure BC dof count={p_dofs.size} (full pressure FE unknowns=n_p_collapsed={n_p_collapsed}), "
            f"ribs facets={facets_ribs.size}, ribs u_dof count={u_dofs_ribs.size}, "
            f"soundhole_facets.shape={soundhole_facets.shape}",
            status_callback=status_callback,
        )

        p_zero = fem.Constant(msh, PETSc.ScalarType(0.0))
        bc_p = fem.dirichletbc(p_zero, p_dofs, V_p)
        bcs = [bc_p]
        clamp_ribs = _solver_bool(solver_cfg, "clamp_ribs", default=True)
        if clamp_ribs and u_dofs_ribs.size > 0:
            u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
            bc_u = fem.dirichletbc(u_zero, u_dofs_ribs, V_u)
            bcs.append(bc_u)
            bcs_u_only.append(bc_u)
            _emit(
                f"[bc] ribs clamp: u = 0 on tag {RIBS_SURFACE_TAG} only "
                f"({facets_ribs.size} facets, {u_dofs_ribs.size} displacement DOFs); "
                f"tags {tag_top} (top) and {tag_back} (back) remain free.",
                status_callback=status_callback,
            )
        elif not clamp_ribs:
            _emit(
                "[bc] ribs clamp DISABLED (clamp_ribs=false); top/back free, tag-4 u unconstrained.",
                status_callback=status_callback,
            )
        else:
            _emit(
                f"[bc][warn] no facets on tag {RIBS_SURFACE_TAG}; ribs not clamped.",
                status_callback=status_callback,
                level="warning",
            )
        _audit_shell_facet_dof_coverage(
            msh,
            facet_tags,
            V_u,
            tag_top=tag_top,
            tag_back=tag_back,
            tag_ribs=RIBS_SURFACE_TAG,
            label="coupled V_u (with BC overlap check)",
            constrained_u_dofs=u_dofs_ribs if clamp_ribs and u_dofs_ribs.size > 0 else None,
            status_callback=status_callback,
        )
        _assert_no_top_plate_dirichlet_bc(
            facet_tags,
            V_u,
            msh,
            u_dofs_ribs if clamp_ribs and u_dofs_ribs.size > 0 else np.array([], dtype=np.int32),
            tag_top=tag_top,
            status_callback=status_callback,
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

    _, p_to_W_map = W.sub(1).collapse()
    p_to_W_map = np.asarray(p_to_W_map, dtype=np.int32)
    u_to_W_map = np.asarray(u_parent_indices, dtype=np.int32)
    shell_dofs_top = _locate_facet_displacement_dofs(
        V_u_collapsed,
        msh,
        np.array(facet_tags.find(tag_top), dtype=np.int32),
    )
    shell_dofs_back = _locate_facet_displacement_dofs(
        V_u_collapsed,
        msh,
        np.array(facet_tags.find(tag_back), dtype=np.int32),
    )

    _phase_sync(2002, "coupled before matrix assembly", status_callback=status_callback)
    _debug_rank("Entering Matrix Assembly")
    if MPI.COMM_WORLD.rank == ROOT_RANK:
        print("PRINT: ENTERING FULL COUPLED ACOUSTIC-STRUCTURAL SOLVE")
        sys.stdout.flush()
    A = assemble_matrix(fem.form(a_form), bcs=bcs)
    A.assemble()
    M = assemble_matrix(fem.form(m_form), bcs=bcs)
    M.assemble()

    _audit_assembled_mixed_coupling(
        W, a_up, m_pu, bcs, status_callback=status_callback
    )

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
        u_shell = ufl.TrialFunction(V_u)
        v_shell = ufl.TestFunction(V_u)
        m_shell_top_u = (top_m["rho"] * t_top) * ufl.dot(u_shell, v_shell) * xdmf_ds(tag_top)
        m_shell_back_u = (back_m["rho"] * t_back) * ufl.dot(u_shell, v_shell) * xdmf_ds(tag_back)
        if has_top_plate_facets:
            M_top = assemble_matrix(fem.form(m_shell_top_u), bcs=bcs_u_only)
            M_top.assemble()
        if has_back_shell_facets:
            M_back = assemble_matrix(fem.form(m_shell_back_u), bcs=bcs_u_only)
            M_back.assemble()

    if not solve_evp:
        if probe_spec is not None:
            return _coupled_resolvent_solve(
                msh,
                W,
                A,
                M,
                facet_tags=facet_tags,
                bcs=bcs,
                u_to_W_map=u_to_W_map,
                p_to_W_map=p_to_W_map,
                solver_cfg=solver_cfg,
                xdmf_ds=xdmf_ds,
                frequency_hz=float(probe_spec.get("frequency_hz", 102.0)),
                force_facet_tag=int(probe_spec.get("force_facet_tag", WOOD_SURFACE_TAGS[1])),
                force_scale=float(probe_spec.get("force_scale", 1.0)),
                status_callback=status_callback,
            )
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
            u_parent_indices=u_to_W_map,
            n_u_global=n_u_fe,
            u_to_W=u_to_W_map,
            p_to_W=p_to_W_map,
            dofs_top=shell_dofs_top,
            dofs_back=shell_dofs_back,
        )
        _emit(
            f"[worker] shift @ {float(_worker_hz):.4f} Hz: nconv={nconv}, usable_rows={len(rows)}",
            status_callback=status_callback,
        )
        _phase_sync(2100, "worker single-shift after batch", status_callback=status_callback)
        row_meta: List[Tuple[float, np.ndarray, float, float, float]] = []
        for f_hz, vec, rt, rb, p_frac, p_block_max in rows:
            if rt is None or rb is None:
                continue
            row_meta.append(
                (
                    float(f_hz),
                    np.asarray(vec, dtype=np.float64),
                    float(rt),
                    float(rb),
                    float(p_frac),
                    float(p_block_max),
                )
            )
        if not row_meta:
            _emit(
                f"[worker][warn] No usable modes at {float(_worker_hz):.4f} Hz "
                f"(nconv returned={nconv}, rows after filter=0).",
                status_callback=status_callback,
                level="warning",
            )
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
            n_u_empty = int(W.sub(0).dofmap.index_map.size_local * W.sub(0).dofmap.index_map_bs)
            n_p_empty = int(W.sub(1).dofmap.index_map.size_local * W.sub(1).dofmap.index_map_bs)
            return msh, W, [], np.zeros((0, 0), dtype=np.float64), n_u_empty, n_p_empty

        row_meta.sort(key=lambda t: t[0])
        freqs_hz = [t[0] for t in row_meta]
        vectors = [t[1] for t in row_meta]
        config["_worker_tag1"] = [t[2] for t in row_meta]
        config["_worker_tag3"] = [t[3] for t in row_meta]
        config["_worker_p_frac"] = [t[4] for t in row_meta]
        config["_worker_p_block_max"] = [t[5] for t in row_meta]
        _shift_jitter = float(solver_cfg.get("shift_jitter_hz", 0.0))
        _, _, _use_st_shift, _ = _slepc_eps_strategy(solver_cfg)
        if _use_st_shift:
            config["_worker_st_sigma_hz"] = max(1.0, float(_worker_hz) + _shift_jitter)
        elif _eps_use_target_real(solver_cfg) and float(solver_cfg.get("eps_broad_search_hz", 0.0)) > 0.0:
            config["_worker_st_sigma_hz"] = max(1.0, float(_worker_hz))
        else:
            config["_worker_st_sigma_hz"] = max(1.0, float(_worker_hz) + _shift_jitter)
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
                u_parent_indices=u_to_W_map,
                n_u_global=n_u_fe,
                u_to_W=u_to_W_map,
                p_to_W=p_to_W_map,
                dofs_top=shell_dofs_top,
                dofs_back=shell_dofs_back,
            )
            scored: List[Tuple[float, float, float]] = []
            for _f, _v, rt, rb, _pf, _pbm in rows:
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
            for f_hz, vec, rt, rb, _pf, _pbm in rows:
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
        if nconv <= 0 and not rows:
            _emit(
                "[solver][warn] Legacy coupled batch: no converged pairs and no harvested rows.",
                status_callback=status_callback,
                level="warning",
            )
        _phase_sync(2006, "coupled legacy after batch solve", status_callback=status_callback)

        freqs_hz = []
        vectors = []
        for f_hz, vec, _rt, _rb, _pf, _pbm in rows:
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


def _configure_probe_direct_ksp(ksp: PETSc.KSP, pc: PETSc.PC, solver_cfg: Dict) -> None:
    """Monolithic MUMPS LU for the resolvent probe (matches production ST defaults)."""
    prefix = "probe_"
    ksp.setOptionsPrefix(prefix)
    pc.setOptionsPrefix(prefix)

    st_ksp_type = str(solver_cfg.get("st_ksp_type", "preonly"))
    st_factor = str(
        solver_cfg.get(
            "st_pc_factor_mat_solver_type",
            solver_cfg.get("st_factor_solver_type", "mumps"),
        )
    )
    ksp.setType(st_ksp_type)
    pc.setType(str(solver_cfg.get("st_pc_type", "lu")))
    try:
        pc.setFactorSolverType(st_factor)
    except Exception:
        pass
    shift_type = str(
        solver_cfg.get(
            "resolvent_pc_factor_shift_type",
            solver_cfg.get(
                "st_pc_factor_shift_type",
                solver_cfg.get("pc_factor_shift_type", "nonzero"),
            ),
        )
    )
    shift_amt = float(
        solver_cfg.get(
            "resolvent_pc_factor_shift_amount",
            solver_cfg.get(
                "st_pc_factor_shift_amount",
                solver_cfg.get("pc_factor_shift_amount", 1.0e-2),
            ),
        )
    )
    diag_scale = _solver_bool(solver_cfg, "resolvent_pc_factor_diagonal_scaling", default=True)
    opts = PETSc.Options()
    opts[f"{prefix}ksp_type"] = st_ksp_type
    opts[f"{prefix}pc_type"] = str(solver_cfg.get("st_pc_type", "lu"))
    opts[f"{prefix}pc_factor_mat_solver_type"] = st_factor
    opts[f"{prefix}pc_factor_shift_type"] = shift_type
    opts[f"{prefix}pc_factor_shift_amount"] = shift_amt
    if diag_scale:
        opts[f"{prefix}pc_factor_diagonal_scaling"] = True
    opts[f"{prefix}mat_mumps_icntl_14"] = int(solver_cfg.get("mat_mumps_icntl_14", 500))
    opts[f"{prefix}mat_mumps_icntl_24"] = int(solver_cfg.get("mat_mumps_icntl_24", 1))
    opts[f"{prefix}mat_mumps_icntl_6"] = int(solver_cfg.get("mat_mumps_icntl_6", 7))
    opts[f"{prefix}mat_mumps_icntl_12"] = int(solver_cfg.get("mat_mumps_icntl_12", 1))
    opts[f"{prefix}mat_mumps_icntl_4"] = int(solver_cfg.get("mat_mumps_icntl_4", 0))
    try:
        pc.setFactorShiftType(shift_type)
        pc.setFactorShiftAmount(shift_amt)
    except Exception:
        pass
    try:
        ksp.setTolerances(rtol=1.0e-12, atol=1.0e-14, max_it=1)
    except Exception:
        pass
    ksp.setFromOptions()


def _coupled_resolvent_solve(
    msh: mesh.Mesh,
    W: fem.FunctionSpace,
    A: PETSc.Mat,
    M: PETSc.Mat,
    *,
    facet_tags,
    bcs: List[fem.DirichletBC],
    u_to_W_map: np.ndarray,
    p_to_W_map: np.ndarray,
    solver_cfg: Dict,
    xdmf_ds,
    frequency_hz: float,
    force_facet_tag: int,
    force_scale: float = 1.0,
    status_callback=None,
) -> Dict[str, Any]:
    """
    Harmonic resolvent probe: solve ``(A - ω² M) x = F`` with facet traction on structure.

    Returns norms and coupling diagnostics for ``||p||/||u||``.
    """
    omega = 2.0 * math.pi * float(frequency_hz)
    lam_shift = omega * omega
    v, _q = ufl.TestFunctions(W)
    n = ufl.FacetNormal(msh)
    f_amp = fem.Constant(msh, PETSc.ScalarType(float(force_scale)))
    tag = int(force_facet_tag)
    n_facets = int(np.sum(facet_tags.values == tag))
    if n_facets <= 0:
        raise ValueError(
            f"Resolvent probe: force facet tag {tag} has no facets on this mesh "
            f"(valid wood tags: {WOOD_SURFACE_TAGS})."
        )
    L = f_amp * ufl.dot(v, n) * xdmf_ds(tag)
    L_form = fem.form(L)

    b = assemble_vector(L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    b_norm = float(b.norm())
    if b_norm < 1.0e-30:
        raise RuntimeError(
            f"Resolvent probe: assembled load vector has ||F||≈0 (tag={tag}, facets={n_facets}). "
            "Check facet tagging and BC overlap on the loaded surface."
        )

    reg_base = float(solver_cfg.get("resolvent_mass_reg_frac", 1.0e-6))
    reg_retries = solver_cfg.get(
        "resolvent_mass_reg_retry_fracs",
        (1.0e-6, 1.0e-4, 1.0e-2, 0.1),
    )
    use_equilibrate = _solver_bool(solver_cfg, "resolvent_symmetric_equilibrate", default=True)
    if isinstance(reg_retries, (list, tuple)):
        reg_fracs = [float(reg_base)] + [float(r) for r in reg_retries if float(r) > float(reg_base)]
    else:
        reg_fracs = [reg_base, 1.0e-4, 1.0e-2]
    seen_reg: set = set()
    reg_fracs_unique: List[float] = []
    for r in reg_fracs:
        if r not in seen_reg:
            seen_reg.add(r)
            reg_fracs_unique.append(r)

    comm = A.getComm()
    reason = -9999
    its = 0
    reg_used = 0.0
    ksp: Optional[PETSc.KSP] = None
    K: Optional[PETSc.Mat] = None
    x: Optional[PETSc.Vec] = None

    a_norm_f = float(A.norm())
    m_norm_f = float(M.norm())
    if a_norm_f < 1.0e-30 and MPI.COMM_WORLD.rank == ROOT_RANK:
        try:
            d_a = A.getDiagonal()
            d_sum = float(np.sum(np.abs(d_a.array)))
            _emit(
                f"[resolvent-probe][warn] ||A||_F≈0 but sum|diag(A)|={d_sum:.6e} — "
                "check matrix assembly.",
                status_callback=status_callback,
                level="warning",
            )
        except Exception:
            pass

    for reg_frac in reg_fracs_unique:
        reg_lambda = float(reg_frac) * float(lam_shift)
        K_try = A.copy()
        K_try.assemble()
        K_try.axpy(
            -lam_shift,
            M,
            structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
        )
        if reg_lambda > 0.0:
            K_try.axpy(
                reg_lambda,
                M,
                structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
            )
        K_try.assemble()

        b_try = b.duplicate()
        b_try.copy(b)
        inv_sqrt: Optional[PETSc.Vec] = None
        if use_equilibrate:
            try:
                inv_sqrt, b_try = _resolvent_symmetric_equilibrate(K_try, b_try)
            except Exception as exc:
                if MPI.COMM_WORLD.rank == ROOT_RANK:
                    _emit(
                        f"[resolvent-probe][warn] symmetric equilibration failed: {exc}",
                        status_callback=status_callback,
                        level="warning",
                    )
                b_try.copy(b)

        x_try = K_try.createVecRight()
        x_try.set(0.0)
        ksp_try = PETSc.KSP().create(comm)
        ksp_try.setOperators(K_try)
        _configure_probe_direct_ksp(ksp_try, ksp_try.getPC(), solver_cfg)
        ksp_try.solve(b_try, x_try)
        if inv_sqrt is not None:
            _resolvent_unscale_solution(x_try, inv_sqrt)
        try:
            b_try.destroy()
            if inv_sqrt is not None:
                inv_sqrt.destroy()
        except Exception:
            pass
        reason_try = int(ksp_try.getConvergedReason())
        its_try = int(ksp_try.getIterationNumber())

        if reason_try > 0:
            if K is not None:
                try:
                    K.destroy()
                except Exception:
                    pass
            if x is not None:
                try:
                    x.destroy()
                except Exception:
                    pass
            if ksp is not None:
                try:
                    ksp.destroy()
                except Exception:
                    pass
            K, x, ksp = K_try, x_try, ksp_try
            reason, its, reg_used = reason_try, its_try, float(reg_frac)
            break

        if MPI.COMM_WORLD.rank == ROOT_RANK:
            try:
                k_norm_try = float(K_try.norm())
            except Exception:
                k_norm_try = float("nan")
            _emit(
                f"[resolvent-probe][warn] KSP failed (reason={reason_try}, its={its_try}) "
                f"with mass reg_frac={reg_frac:.2e} ||K||_F={k_norm_try:.6e} "
                f"(K = A - ({lam_shift:.3e} - {reg_lambda:.3e}) M); retrying.",
                status_callback=status_callback,
                level="warning",
            )
        try:
            ksp_try.destroy()
            x_try.destroy()
            K_try.destroy()
        except Exception:
            pass

    solve_ok = reason > 0 and x is not None and K is not None and ksp is not None
    u_norm = float("nan")
    p_norm = float("nan")
    ratio = float("nan")
    coupled_visible = False

    if solve_ok and x is not None:
        arr = x.array.copy()
        if np.all(np.isfinite(arr)):
            u_norm, p_norm = _mixed_eigenvector_block_norms(
                arr, u_to_W=u_to_W_map, p_to_W=p_to_W_map
            )
            ratio = p_norm / max(u_norm, 1.0e-30)
            p_floor = 1.0e-6 * max(u_norm, 1.0e-30)
            coupled_visible = bool(np.isfinite(p_norm) and np.isfinite(u_norm) and p_norm > p_floor)
        else:
            solve_ok = False
            reason = -8888

    if MPI.COMM_WORLD.rank == ROOT_RANK:
        if solve_ok:
            verdict = (
                "COUPLED (physics OK — prioritize solver strategy)"
                if coupled_visible
                else "DECOUPLED (physics/coupling issue — fix FSI formulation before eigensolver)"
            )
        else:
            verdict = (
                "SOLVE_FAILED (shifted operator singular/ill-conditioned — "
                "try another Hz or increase resolvent_mass_reg_frac)"
            )
        print(
            f"[resolvent-probe] f={frequency_hz:.4f} Hz omega={omega:.6e} rad/s "
            f"lambda_shift={lam_shift:.6e} mass_reg_frac={reg_used:.2e} "
            f"force_tag={tag} (n_facets={n_facets}) force_scale={force_scale:.4e}"
        )
        print(
            f"[resolvent-probe] ||A||_F={a_norm_f:.6e} ||M||_F={m_norm_f:.6e}"
        )
        print(f"[resolvent-probe] ||F||={b_norm:.6e} KSP its={its} reason={reason}")
        if solve_ok:
            print(f"[resolvent-probe] ||u||={u_norm:.6e} ||p||={p_norm:.6e} ||p||/||u||={ratio:.6e}")
            p_floor = 1.0e-6 * max(u_norm, 1.0e-30)
            print(
                f"[resolvent-probe] coupling_check (||p|| > 1e-6*||u||): "
                f"{coupled_visible} (threshold ||p|| > {p_floor:.6e})"
            )
        print(f"[resolvent-probe] VERDICT: {verdict}")
        sys.stdout.flush()

    try:
        if ksp is not None:
            ksp.destroy()
        if x is not None:
            x.destroy()
        if K is not None:
            K.destroy()
        b.destroy()
    except Exception:
        pass

    return {
        "frequency_hz": float(frequency_hz),
        "omega_rad_s": float(omega),
        "lambda_shift": float(lam_shift),
        "mass_reg_frac": float(reg_used),
        "force_facet_tag": int(tag),
        "force_facet_count": int(n_facets),
        "force_scale": float(force_scale),
        "load_norm": b_norm,
        "u_norm": u_norm,
        "p_norm": p_norm,
        "p_over_u": ratio,
        "coupling_check_pass": bool(coupled_visible),
        "solve_ok": bool(solve_ok),
        "ksp_iterations": int(its),
        "ksp_reason": int(reason),
    }


def run_coupled_resolvent_probe(
    config: Dict,
    frequency_hz: float = 102.0,
    force_facet_tag: int = 3,
    force_scale: float = 1.0,
    status_callback=None,
) -> Dict[str, Any]:
    """
    Assemble the coupled GNHEP operators and run one harmonic resolvent solve.

    Uses the same mesh, BCs, and ``A``/``M`` assembly path as the main EVP driver.
    """
    mesh_file = Path(config["solver"]["mesh_file"])
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    probe_spec = {
        "frequency_hz": float(frequency_hz),
        "force_facet_tag": int(force_facet_tag),
        "force_scale": float(force_scale),
    }
    result = _solve_coupled_evp(
        mesh_file=mesh_file,
        config=config,
        num_modes=0,
        status_callback=status_callback,
        solve_evp=False,
        probe_spec=probe_spec,
    )
    if not isinstance(result, dict):
        raise RuntimeError(
            "Resolvent probe did not return diagnostics (internal error: probe_spec ignored)."
        )
    return result


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

    analysis_name = (
        "structural_shell_diagnostic"
        if _is_structural_only_run(config, True)
        else "acoustic_structural_coupled_eigen"
    )
    output_data = {
        "analysis": analysis_name,
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