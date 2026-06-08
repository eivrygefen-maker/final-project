#!/usr/bin/env python3
"""Component-level physics identity: mesh submeshes, operator blocks, masks, contamination scans."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PHYSICS_IDENTITY_SCHEMA = "m4_physics_identity_v1"
MODE_PROVENANCE_SIDECAR = "aggregation/mode_provenance.jsonl"
PHYSICS_IDENTITY_MANIFEST = "freeze/physics_identity_manifest.json"

FORBIDDEN_HEAVY_REL_DIRS = (
    "lprod/checkpoint",
    "lprod/mesh",
    "scout/mesh",
    "scout/checkpoint",
    "scout/discovery",
    "worker_results",
)

WOOD_VOLUME_TAGS = (1, 2, 3)
CAVITY_AIR_TAG = 10

OTHER_SAMPLE_PATH_RE = re.compile(
    r"sample_\d{3}(?:/|\\)|sample_\d{3}_",
    re.IGNORECASE,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_indices(indices: Sequence[int]) -> Optional[str]:
    if not indices:
        return None
    payload = ",".join(str(int(i)) for i in sorted(int(x) for x in indices))
    return _sha256_bytes(payload.encode("utf-8"))


def csr_structure_and_values_hashes(npz_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": npz_path.is_file()}
    if not npz_path.is_file():
        return out
    with np.load(npz_path, allow_pickle=False) as z:
        shape = tuple(int(x) for x in np.asarray(z["shape"]).ravel())
        indptr = np.asarray(z["indptr"], dtype=np.int64).ravel()
        indices = np.asarray(z["indices"], dtype=np.int64).ravel()
        data = np.asarray(z["data"], dtype=np.float64).ravel()
        structure_bytes = (
            json.dumps(shape, separators=(",", ":")).encode("utf-8")
            + indptr.tobytes()
            + indices.tobytes()
        )
        out.update(
            {
                "shape": list(shape),
                "nnz": int(data.size),
                "structure_sha256": _sha256_bytes(structure_bytes),
                "values_sha256": _sha256_bytes(data.tobytes()),
            }
        )
    return out


def _load_csr(npz_path: Path):
    from scipy.sparse import csr_matrix  # noqa: WPS433

    with np.load(npz_path, allow_pickle=False) as z:
        shape = tuple(int(x) for x in np.asarray(z["shape"]).ravel())
        return csr_matrix(
            (np.asarray(z["data"]), np.asarray(z["indices"]), np.asarray(z["indptr"])),
            shape=shape,
        )


def _active_block_labels(built_meta: Mapping[str, Any]) -> Optional[np.ndarray]:
    try:
        active_local = np.asarray(built_meta["active_local"], dtype=np.int32).ravel()
        free_rows = np.asarray(built_meta["free_rows"], dtype=np.int32).ravel()
        n_u_b3 = int(built_meta.get("n_u_b3") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if n_u_b3 <= 0 or active_local.size == 0:
        return None
    w_rows = free_rows[active_local]
    labels = np.where(w_rows < n_u_b3, 0, 1).astype(np.int8)  # 0=u, 1=p
    return labels


def extract_active_block_hashes(checkpoint_dir: Path, built_meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Hash A/M sub-blocks uu, pp, up, pu from active CSR when metadata present."""
    labels = _active_block_labels(built_meta)
    out: Dict[str, Any] = {"status": "missing_labels"}
    if labels is None:
        return out
    a_path = checkpoint_dir / "A_active_csr.npz"
    m_path = checkpoint_dir / "M_active_csr.npz"
    if not a_path.is_file() or not m_path.is_file():
        out["status"] = "missing_csr"
        return out
    try:
        a_csr = _load_csr(a_path)
        m_csr = _load_csr(m_path)
        u_idx = np.flatnonzero(labels == 0)
        p_idx = np.flatnonzero(labels == 1)
        blocks: Dict[str, Any] = {}
        for name, rows, cols in (
            ("A_uu", u_idx, u_idx),
            ("A_pp", p_idx, p_idx),
            ("A_up", u_idx, p_idx),
            ("A_pu", p_idx, u_idx),
            ("M_uu", u_idx, u_idx),
            ("M_pp", p_idx, p_idx),
        ):
            if rows.size == 0 or cols.size == 0:
                blocks[name] = {"present": False}
                continue
            sub = a_csr if name.startswith("A_") else m_csr
            subm = sub[rows, :][:, cols]
            blocks[name] = {
                "present": True,
                "shape": list(subm.shape),
                "nnz": int(subm.nnz),
                "structure_sha256": _sha256_bytes(
                    json.dumps(subm.shape).encode()
                    + subm.indptr.tobytes()
                    + subm.indices.tobytes()
                ),
                "values_sha256": _sha256_bytes(np.asarray(subm.data, dtype=np.float64).tobytes()),
            }
        out = {"status": "ok", "blocks": blocks, "n_u_active": int(u_idx.size), "n_p_active": int(p_idx.size)}
    except Exception as exc:  # noqa: BLE001
        out = {"status": f"error:{type(exc).__name__}", "detail": str(exc)}
    return out


def mesh_component_hashes(mesh_path: Path, *, built_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Topology/coordinate hashes and submesh hashes for wood, cavity air, exterior pressure DOFs."""
    out: Dict[str, Any] = {"mesh_path": str(mesh_path), "mesh_file_sha256": _sha256_file(mesh_path)}
    if not mesh_path.is_file():
        out["status"] = "mesh_missing"
        return out
    try:
        from v2_b3_synthesis_export import import_fem_main_3d  # noqa: WPS433

        fem3d, _ = import_fem_main_3d(start=Path(__file__).resolve().parent)
        msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_path)
        coords = np.asarray(msh.geometry.x, dtype=np.float64)
        tdim = msh.topology.dim
        msh.topology.create_connectivity(tdim, 0)
        n_cells = int(msh.topology.index_map(tdim).size_local)
        conn = msh.geometry.dofmap
        cells = []
        cell_tag_vals = []
        for c in range(n_cells):
            nodes = tuple(sorted(int(n) for n in conn.links(c)))
            cells.append(nodes)
            cell_tag_vals.append(int(cell_tags.values[c]))
        topo_payload = json.dumps(
            [{"tag": cell_tag_vals[i], "nodes": cells[i]} for i in range(len(cells))],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        out["full_mesh_topology_sha256"] = _sha256_bytes(topo_payload)
        out["full_mesh_coordinate_sha256"] = _sha256_bytes(coords.tobytes())
        out["n_nodes"] = int(coords.shape[0])
        out["n_cells"] = n_cells

        def _submesh_hash(tag_set: Sequence[int], *, kind: str) -> Dict[str, Any]:
            idx = [i for i, t in enumerate(cell_tag_vals) if t in tag_set]
            if not idx:
                return {"present": False, "kind": kind}
            sub_cells = [cells[i] for i in idx]
            sub_nodes = sorted({n for c in sub_cells for n in c})
            sub_coords = coords[sub_nodes]
            payload = json.dumps(sub_cells, separators=(",", ":")).encode("utf-8")
            return {
                "present": True,
                "kind": kind,
                "n_cells": len(sub_cells),
                "n_nodes": len(sub_nodes),
                "topology_sha256": _sha256_bytes(payload),
                "coordinate_sha256": _sha256_bytes(sub_coords.tobytes()),
            }

        out["wood_shell_submesh"] = _submesh_hash(WOOD_VOLUME_TAGS, kind="wood_shell")
        out["cavity_air_submesh"] = _submesh_hash((CAVITY_AIR_TAG,), kind="cavity_air")

        p_idx_all = np.asarray((built_meta or {}).get("p_idx") or [], dtype=np.int32).ravel()
        cavity_p: np.ndarray = np.asarray([], dtype=np.int32)
        exterior_p: np.ndarray = np.asarray([], dtype=np.int32)
        try:
            cavity_p = np.asarray(
                fem3d._locate_air_volume_pressure_dofs(msh, cell_tags), dtype=np.int32
            ).ravel()
            if p_idx_all.size:
                cavity_set = set(int(x) for x in cavity_p)
                exterior_p = np.asarray(
                    [int(x) for x in p_idx_all if int(x) not in cavity_set], dtype=np.int32
                )
        except Exception:
            pass
        out["pressure_dof"] = {
            "p_idx_cavity_count": int(cavity_p.size),
            "p_idx_exterior_count": int(exterior_p.size),
            "p_idx_cavity_sha256": _sha256_indices(cavity_p.tolist()),
            "p_idx_exterior_sha256": _sha256_indices(exterior_p.tolist()),
            "pressure_dof_coordinate_note": "p_idx are global W pressure rows, not mesh node indices",
        }
        u_n = int((built_meta or {}).get("n_u_b3") or 0)
        if u_n > 0 and coords.shape[0] >= u_n:
            out["structural_dof_coordinate_sha256"] = _sha256_bytes(coords[:u_n].tobytes())
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"error:{type(exc).__name__}"
        out["error"] = str(exc)
    return out


def scan_cross_sample_path_contamination(
    run_root: Path,
    *,
    sample_id: str,
    max_files: int = 200,
) -> Dict[str, Any]:
    """Detect path references to other sample_ids under run_root."""
    hits: List[Dict[str, str]] = []
    own = sample_id.lower()
    checked = 0
    for path in run_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in (".npz", ".bin", ".msh") and path.stat().st_size > 5_000_000:
            continue
        checked += 1
        if checked > max_files:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in OTHER_SAMPLE_PATH_RE.finditer(text):
            token = m.group(0).rstrip("/\\").lower()
            other_sid = token.split("/")[0].split("\\")[0].replace("_chunk", "")
            if other_sid.startswith("sample_") and other_sid != own:
                hits.append(
                    {
                        "file": str(path.relative_to(run_root)).replace("\\", "/"),
                        "reference": token,
                        "other_sample_id": other_sid.split("_chunk")[0],
                    }
                )
    return {
        "sample_id": sample_id,
        "files_scanned": checked,
        "contamination_hits": hits,
        "contamination_detected": len(hits) > 0,
    }


def collect_mask_identity(checkpoint_dir: Path, built_meta: Mapping[str, Any]) -> Dict[str, Any]:
    from v2_b3_rich_modal_lib import load_region_dof_bundle  # noqa: WPS433

    out: Dict[str, Any] = {}
    try:
        ctx = load_region_dof_bundle(checkpoint_dir, built_meta, validate_aperture=False)
        region = ctx.get("region") or {}
        p_ap = np.asarray(region.get("p_idx_aperture", []), dtype=np.int32).ravel()
        bridge = np.asarray(region.get("u_idx_bridge", region.get("u_idx_top", [])), dtype=np.int32).ravel()
        out["p_idx_aperture_count"] = int(p_ap.size)
        out["p_idx_aperture_sha256"] = _sha256_indices(p_ap.tolist())
        out["p_idx_aperture_min"] = int(p_ap.min()) if p_ap.size else None
        out["p_idx_aperture_max"] = int(p_ap.max()) if p_ap.size else None
        out["bridge_dof_count"] = int(bridge.size)
        out["bridge_mask_sha256"] = _sha256_indices(bridge.tolist())
        out["aperture_selection_method"] = str(
            region.get("aperture_selection_method") or "facet_adjacent_air_cell_dofs_v1"
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}:{exc}"
    return out


def forbidden_solver_fallback_keys(core_config: Mapping[str, Any]) -> List[str]:
    solver = core_config.get("solver") or {}
    active: List[str] = []
    for key in ("eps_interval_fallback", "eps_ciss_fallback_shift_invert", "resolvent_static_fallback"):
        if key in solver and solver.get(key) not in (None, False, "", 0):
            active.append(key)
    return active


def build_physics_identity_manifest(
    *,
    run_root: Path,
    repo_root: Path,
    sample_input: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble durable identity manifest from checkpoint (must exist pre-compaction)."""
    from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: WPS433
    from v2_b3_m4_production_contracts import DATASET_VERSION, PRODUCTION_MIC_METHOD  # noqa: WPS433

    run_root = run_root.resolve()
    sample_id = str(sample_input.get("sample_id") or run_root.parent.parent.name)
    run_id = run_root.name
    ckpt = run_root / "lprod" / "checkpoint"
    built_path = ckpt / "built_metadata.json"
    built: Dict[str, Any] = {}
    if built_path.is_file():
        built = json.loads(built_path.read_text(encoding="utf-8"))

    geom = extract_geometry_dict(sample_input)
    mesh_path = run_root / "lprod" / "mesh" / "L_prod" / f"{sample_id}.msh"
    if not mesh_path.is_file():
        op_mesh = str(built.get("operator_mesh_file_used") or built.get("generated_mesh_file") or "")
        if op_mesh:
            mesh_path = Path(op_mesh) if Path(op_mesh).is_absolute() else repo_root / op_mesh

    core_cfg_path = run_root / "lprod" / "resolved_core_config.json"
    core_cfg: Dict[str, Any] = {}
    if core_cfg_path.is_file():
        core_cfg = json.loads(core_cfg_path.read_text(encoding="utf-8"))

    manifest: Dict[str, Any] = {
        "schema": PHYSICS_IDENTITY_SCHEMA,
        "sample_id": sample_id,
        "run_id": run_id,
        "dataset_version": built.get("dataset_version") or DATASET_VERSION,
        "geometry_fingerprint": geometry_fingerprint(geom) if geom else None,
        "geometry_numeric_parameters": dict(geom) if geom else {},
        "generated_mesh_path": str(mesh_path),
        "generated_mesh_sha256": built.get("generated_mesh_sha256") or _sha256_file(mesh_path),
        "operator_mesh_sha256": built.get("operator_mesh_sha256"),
        "operator_mesh_matches_generated": bool(built.get("operator_mesh_matches_generated")),
        "operator_mesh_file_used": built.get("operator_mesh_file_used"),
        "active_dimension": built.get("active_dimension"),
        "n_u_b3": built.get("n_u_b3"),
        "n_w": built.get("n_w"),
        "mic_output_method": PRODUCTION_MIC_METHOD,
        "solver_backend": "mkl_pardiso",
        "production_acceptance_pass": bool(acceptance.get("acceptance_pass")),
        "production_acceptance_failures": list(acceptance.get("failures") or []),
        "source_built_metadata_sha256": _sha256_file(built_path),
        "fallback_flags": {
            "solver_fallback_used": False,
            "mic_method_fallback": False,
            "participation_fallback": False,
            "geometry_fallback": False,
            "baseline_mesh_fallback": False,
            "cross_sample_reuse": False,
            "forbidden_solver_config_keys": forbidden_solver_fallback_keys(core_cfg),
        },
        "A_active_csr": csr_structure_and_values_hashes(ckpt / "A_active_csr.npz"),
        "M_active_csr": csr_structure_and_values_hashes(ckpt / "M_active_csr.npz"),
        "operator_blocks": extract_active_block_hashes(ckpt, built),
        "mesh_components": mesh_component_hashes(mesh_path, built_meta=built),
        "masks": collect_mask_identity(ckpt, built),
        "path_contamination": scan_cross_sample_path_contamination(run_root, sample_id=sample_id),
    }
    fb = manifest["fallback_flags"]
    if fb.get("forbidden_solver_config_keys"):
        fb["solver_config_fallback"] = True
    if manifest["path_contamination"].get("contamination_detected"):
        fb["cross_sample_reuse"] = True
    return manifest


def validate_physics_identity_manifest(manifest: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if str(manifest.get("schema")) != PHYSICS_IDENTITY_SCHEMA:
        errors.append("schema_mismatch")
    if not manifest.get("sample_id") or not manifest.get("run_id"):
        errors.append("missing_sample_run_identity")
    if not manifest.get("generated_mesh_sha256"):
        errors.append("missing_generated_mesh_sha256")
    if not bool(manifest.get("operator_mesh_matches_generated")):
        errors.append("operator_mesh_matches_generated!=true")
    if int(manifest.get("active_dimension") or 0) <= 0:
        errors.append("active_dimension<=0")
    masks = manifest.get("masks") or {}
    if int(masks.get("p_idx_aperture_count") or 0) <= 0:
        errors.append("p_idx_aperture_count<=0")
    fb = manifest.get("fallback_flags") or {}
    for key, val in fb.items():
        if key == "forbidden_solver_config_keys":
            if val:
                errors.append(f"forbidden_solver_config:{val}")
            continue
        if val:
            errors.append(f"fallback_flag_true:{key}")
    if manifest.get("path_contamination", {}).get("contamination_detected"):
        errors.append("cross_sample_path_contamination")
    if not bool(manifest.get("production_acceptance_pass")):
        errors.append("production_acceptance_pass!=true")
    return len(errors) == 0, errors


def count_forbidden_heavy_artifacts(run_root: Path) -> Tuple[int, List[str]]:
    present: List[str] = []
    for rel in FORBIDDEN_HEAVY_REL_DIRS:
        p = run_root / rel
        if p.exists():
            present.append(rel)
    return len(present), present


def verify_post_compaction_contract(run_root: Path) -> Dict[str, Any]:
    """Verify run tree after blocking compaction."""
    run_root = run_root.resolve()
    manifest_path = run_root / PHYSICS_IDENTITY_MANIFEST
    out: Dict[str, Any] = {
        "run_root": str(run_root),
        "physics_identity_manifest_present": manifest_path.is_file(),
        "mode_provenance_present": (run_root / MODE_PROVENANCE_SIDECAR).is_file(),
        "freeze_manifest_present": (run_root / "freeze" / "freeze_manifest.json").is_file(),
        "catalog_present": (run_root / "aggregation" / "modes_catalog.jsonl").is_file(),
        "compaction_manifest_present": (run_root / "compaction" / "compaction_manifest.json").is_file(),
    }
    forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(run_root)
    out["forbidden_heavy_artifact_count"] = forbidden_count
    out["forbidden_heavy_artifacts_present"] = forbidden_paths
    errors: List[str] = []
    if not out["physics_identity_manifest_present"]:
        errors.append("missing_physics_identity_manifest")
    if not out["mode_provenance_present"]:
        errors.append("missing_mode_provenance_sidecar")
    if not out["catalog_present"]:
        errors.append("missing_modes_catalog")
    if forbidden_count > 0:
        errors.append(f"forbidden_heavy_artifacts:{forbidden_paths}")
    if manifest_path.is_file():
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            ok, man_errs = validate_physics_identity_manifest(man)
            out["physics_identity_valid"] = ok
            if not ok:
                errors.extend(man_errs)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"physics_identity_unreadable:{exc}")
    comp_path = run_root / "compaction" / "compaction_manifest.json"
    if comp_path.is_file():
        try:
            comp = json.loads(comp_path.read_text(encoding="utf-8"))
            out["compaction_status"] = comp.get("status")
            if str(comp.get("status")) != "completed":
                errors.append(f"compaction_status={comp.get('status')}")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("compaction_manifest_unreadable")
    else:
        errors.append("missing_compaction_manifest")
    out["pass"] = len(errors) == 0
    out["errors"] = errors
    return out
