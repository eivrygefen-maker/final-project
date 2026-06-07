#!/usr/bin/env python3
"""Shared helpers for isolated M4 geometry/audio validation (not production)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
GUITARS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars")
MESH_LEVEL = "L_prod"
VALIDATION_RUN_SUFFIX = "geometryfix_validation"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_aperture_pressure_mask import (  # noqa: E402
    aperture_mask_summary,
    build_aperture_pressure_mask,
    write_aperture_mask_npz,
)
from v2_b3_m4_lhs_pool_bridge import lhs_entry_index  # noqa: E402
from v2_b3_m4_lprod_interfaces import extract_geometry_dict  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import load_fom_modes_catalog_deduped  # noqa: E402
from v2_b3_m4_worker_run_lib import load_json, rel  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

NARROW_TARGETS_281 = [272.0, 275.0, 278.0, 281.5, 284.0, 287.0]
NARROW_TARGETS_390 = [382.0, 385.0, 388.0, 391.5, 394.0, 397.0]
BAND_281 = (270.0, 290.0)
BAND_390 = (380.0, 400.0)


def validation_run_id(sample_id: str) -> str:
    return f"{sample_id}_{VALIDATION_RUN_SUFFIX}"


def production_run_root(repo_root: Path, pool: Mapping[str, Any], sample_id: str, run_id_suffix: str) -> Path:
    idx = lhs_entry_index(pool, sample_id)
    entry = (pool.get("entries") or [])[idx] if idx is not None else {}
    run_id = str(entry.get("last_run_id") or f"{sample_id}_{run_id_suffix}")
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


def validation_run_root(repo_root: Path, sample_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / validation_run_id(sample_id)


def _sha256_file(path: Path, *, max_bytes: Optional[int] = None) -> Optional[str]:
    if not path.is_file():
        return None
    data = path.read_bytes() if max_bytes is None else path.read_bytes()[:max_bytes]
    return hashlib.sha256(data).hexdigest()


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
                "structure_sha256": hashlib.sha256(structure_bytes).hexdigest(),
                "values_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            }
        )
    return out


def dolfinx_mesh_stats(path: Path) -> Dict[str, Any]:
    try:
        fem_scripts = SCRIPT_DIR.parent.parent.parent / "scripts"
        if str(fem_scripts) not in sys.path:
            sys.path.insert(0, str(fem_scripts))
        import fem_main_3d as fem3d  # noqa: WPS433

        msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(path)
        coords = np.asarray(msh.geometry.x)
        tdim = msh.topology.dim
        return {
            "dolfinx_available": True,
            "n_nodes": int(coords.shape[0]),
            "n_cells": int(msh.topology.index_map(tdim).size_local),
            "bbox_min": coords.min(axis=0).tolist(),
            "bbox_max": coords.max(axis=0).tolist(),
            "coord_sha256": hashlib.sha256(coords.tobytes()).hexdigest(),
            "n_facet_tag2_soundhole": int(np.asarray(facet_tags.find(2)).size),
            "n_cell_tag10_air": int(np.asarray(cell_tags.find(10)).size),
        }
    except Exception as exc:  # noqa: BLE001
        return {"dolfinx_available": False, "error": f"{type(exc).__name__}:{exc}"}


def prepare_validation_run_tree(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
) -> Dict[str, Any]:
    """Copy mesh + resolved config from production run into isolated validation run tree."""
    prod_root = production_run_root(repo_root, pool, sample_id, run_id_suffix)
    val_root = validation_run_root(repo_root, sample_id)
    val_root.mkdir(parents=True, exist_ok=True)

    src_mesh = prod_root / "lprod" / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    dst_mesh = val_root / "lprod" / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    if not src_mesh.is_file():
        raise FileNotFoundError(f"production mesh missing: {src_mesh}")

    dst_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_mesh, dst_mesh)
    summary_src = src_mesh.parent / f"{sample_id}_mesh_build_summary.json"
    if summary_src.is_file():
        shutil.copy2(summary_src, dst_mesh.parent / f"{sample_id}_mesh_build_summary.json")

    for cfg_name in ("resolved_core_config.json",):
        src_cfg = prod_root / "lprod" / cfg_name
        if src_cfg.is_file():
            dst_cfg = val_root / "lprod" / cfg_name
            dst_cfg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_cfg, dst_cfg)

    manifest = {
        "schema": "m4_geometryfix_validation_run_v1",
        "sample_id": sample_id,
        "run_id": validation_run_id(sample_id),
        "source_production_run": rel(prod_root, repo_root=repo_root),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "is_validation_only": True,
        "must_not_update_lhs": True,
    }
    write_json_atomic(val_root / "validation_run_manifest.json", manifest)
    return {
        "validation_run_root": val_root,
        "production_run_root": prod_root,
        "generated_mesh_path": dst_mesh,
        "resolved_core_config": val_root / "lprod" / "resolved_core_config.json",
        "manifest": manifest,
    }


def build_narrow_band_chunk_targets(*, sample_id: str, run_id: str) -> Dict[str, Any]:
    targets: List[Dict[str, Any]] = []
    for thz in NARROW_TARGETS_281:
        targets.append(
            {
                "target_hz": float(thz),
                "window_hz": [float(BAND_281[0]), float(BAND_281[1])],
                "zone_id": "validation_281",
                "spacing_hz": 0.0,
                "source": "m4_geometryfix_validation",
            }
        )
    for thz in NARROW_TARGETS_390:
        targets.append(
            {
                "target_hz": float(thz),
                "window_hz": [float(BAND_390[0]), float(BAND_390[1])],
                "zone_id": "validation_390",
                "spacing_hz": 0.0,
                "source": "m4_geometryfix_validation",
            }
        )
    return {
        "schema": "m4_worker_chunk_targets_v1",
        "sample_id": sample_id,
        "run_id": run_id,
        "chunk_id": "validation_narrow_band",
        "freq_range_hz": [float(BAND_281[0]), float(BAND_390[1])],
        "targets": targets,
    }


def _modes_in_band(modes: Sequence[Mapping[str, Any]], lo: float, hi: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in modes:
        f = m.get("frequency_hz")
        if f is None:
            continue
        try:
            fv = float(f)
        except (TypeError, ValueError):
            continue
        if lo <= fv <= hi:
            row = dict(m)
            out.append(row)
    out.sort(key=lambda r: float(r["frequency_hz"]))
    return out


def _dedupe_modes(modes: Sequence[Mapping[str, Any]], *, tol_hz: float = 0.05) -> List[Dict[str, Any]]:
    from v2_b3_m4_aggregate_worker_results import _dedupe_catalog  # noqa: WPS433

    deduped, _ = _dedupe_catalog(list(modes), tol_hz=tol_hz)
    return deduped


def _production_legacy_peak(
    prod_root: Path,
    band: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    catalog = prod_root / "aggregation" / "modes_catalog.jsonl"
    if not catalog.is_file():
        return None
    _raw, deduped, _ = load_fom_modes_catalog_deduped(catalog)
    in_band = _modes_in_band(deduped, band[0], band[1])
    if not in_band:
        return None
    best = max(in_band, key=lambda m: float(m.get("mic_output_proxy") or 0.0))
    return {
        "frequency_hz": best.get("frequency_hz"),
        "mic_output_proxy": best.get("mic_output_proxy"),
        "mic_output_method": best.get("mic_output_method"),
    }


def collect_checkpoint_report(
    *,
    repo_root: Path,
    val_root: Path,
    generated_mesh: Path,
    load_dolfinx: bool,
) -> Dict[str, Any]:
    ckpt = val_root / "lprod" / "checkpoint"
    built_path = ckpt / "built_metadata.json"
    built = load_json(built_path) if built_path.is_file() else {}
    operator_mesh = Path(str(built.get("operator_mesh_file_used") or built.get("region_dof_mesh_file") or ""))
    if not operator_mesh.is_file():
        operator_mesh = generated_mesh

    gen_sha = _sha256_file(generated_mesh)
    op_sha = _sha256_file(operator_mesh)
    report: Dict[str, Any] = {
        "generated_mesh_path": rel(generated_mesh, repo_root=repo_root),
        "operator_mesh_path": rel(operator_mesh, repo_root=repo_root),
        "generated_mesh_sha256": gen_sha,
        "operator_mesh_sha256": op_sha,
        "operator_mesh_matches_generated": bool(gen_sha and op_sha and gen_sha == op_sha),
        "n_w": built.get("n_w"),
        "n_u_b3": built.get("n_u_b3"),
        "active_dimension": built.get("active_dimension"),
        "n_p_air": len(built.get("p_idx") or []),
        "operator_mesh_file_used": built.get("operator_mesh_file_used"),
    }
    if load_dolfinx:
        report["generated_mesh_dolfinx"] = dolfinx_mesh_stats(generated_mesh)
        report["operator_mesh_dolfinx"] = dolfinx_mesh_stats(operator_mesh)
        gdx = report.get("generated_mesh_dolfinx") or {}
        odx = report.get("operator_mesh_dolfinx") or {}
        report["operator_node_count"] = odx.get("n_nodes")
        report["operator_cell_count"] = odx.get("n_cells")
        report["generated_node_count"] = gdx.get("n_nodes")
        report["generated_cell_count"] = gdx.get("n_cells")
    a_hash = csr_structure_and_values_hashes(ckpt / "A_active_csr.npz")
    m_hash = csr_structure_and_values_hashes(ckpt / "M_active_csr.npz")
    report["A_active_csr"] = a_hash
    report["M_active_csr"] = m_hash
    return report


def attach_aperture_mask(
    *,
    val_root: Path,
    generated_mesh: Path,
    pool: Mapping[str, Any],
    sample_id: str,
) -> Dict[str, Any]:
    ckpt = val_root / "lprod" / "checkpoint"
    built = load_json(ckpt / "built_metadata.json")
    idx = lhs_entry_index(pool, sample_id)
    entry = (pool.get("entries") or [])[idx] if idx is not None else {}
    geom = extract_geometry_dict(entry)
    mask = build_aperture_pressure_mask(generated_mesh, geometry=geom, built_meta=built)
    mask_path = val_root / "validation" / "aperture_pressure_mask.npz"
    write_aperture_mask_npz(mask_path, mask)
    summary = aperture_mask_summary(mask)
    summary["mask_npz_path"] = str(mask_path)
    return summary


def collect_solve_band_results(solver_result_path: Path) -> Dict[str, Any]:
    if not solver_result_path.is_file():
        return {"status": "missing_solver_result"}
    result = load_json(solver_result_path)
    mode_records: List[Dict[str, Any]] = []
    for trow in result.get("targets") or []:
        for m in trow.get("accepted_modes") or []:
            if isinstance(m, dict) and m.get("frequency_hz") is not None:
                mode_records.append(dict(m))
    deduped = _dedupe_modes(mode_records)
    raw_281 = _modes_in_band(mode_records, *BAND_281)
    raw_390 = _modes_in_band(mode_records, *BAND_390)
    ded_281 = _modes_in_band(deduped, *BAND_281)
    ded_390 = _modes_in_band(deduped, *BAND_390)
    exact_dups_281 = len(raw_281) - len(ded_281)
    exact_dups_390 = len(raw_390) - len(ded_390)
    return {
        "solver_status": result.get("status"),
        "raw_mode_count": len(mode_records),
        "deduped_mode_count": len(deduped),
        "deduped_modes_270_290_hz": ded_281,
        "deduped_modes_380_400_hz": ded_390,
        "raw_duplicate_count_281_band": max(0, exact_dups_281),
        "raw_duplicate_count_390_band": max(0, exact_dups_390),
        "peak_mic_281": max(
            (float(m.get("mic_output_proxy") or 0.0) for m in ded_281),
            default=None,
        ),
        "peak_mic_390": max(
            (float(m.get("mic_output_proxy") or 0.0) for m in ded_390),
            default=None,
        ),
    }


def evaluate_validation_gates(
    test_b_results: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if len(test_b_results) < 2:
        failures.append("expected_two_sample_results")
        return False, failures

    struct_hashes: List[str] = []
    for row in test_b_results:
        sid = row.get("sample_id")
        if not row.get("operator_mesh_matches_generated"):
            failures.append(f"{sid}:operator_mesh_matches_generated=false")
        if int(row.get("p_idx_aperture_count") or 0) <= 0:
            failures.append(f"{sid}:p_idx_aperture_count=0")
        band_modes = (row.get("deduped_modes_270_290_hz") or []) + (
            row.get("deduped_modes_380_400_hz") or []
        )
        if row.get("solver_status") and band_modes and not all(
            str(m.get("mic_output_method")) == "aperture_pressure_rms_proxy_v1" for m in band_modes
        ):
            failures.append(f"{sid}:mic_output_method_not_aperture_pressure_rms_proxy_v1")
        if int(row.get("raw_duplicate_count_281_band") or 0) > 0:
            failures.append(f"{sid}:raw_duplicates_in_281_band")
        if int(row.get("raw_duplicate_count_390_band") or 0) > 0:
            failures.append(f"{sid}:raw_duplicates_in_390_band")
        a_struct = ((row.get("A_active_csr") or {}).get("structure_sha256") or "")
        if a_struct:
            struct_hashes.append(a_struct)

    if len(set(struct_hashes)) < 2 and len(struct_hashes) >= 2:
        failures.append("A_structure_hash_identical_across_extremes")

    freqs_281 = [
        float(m.get("frequency_hz"))
        for row in test_b_results
        for m in (row.get("deduped_modes_270_290_hz") or [])
        if m.get("frequency_hz") is not None
    ]
    if len(freqs_281) >= 2:
        span = max(freqs_281) - min(freqs_281)
        if span < 0.05:
            failures.append(f"281_band_frequency_span_too_small:{span:.6f}Hz")

    n_nodes = [row.get("operator_node_count") for row in test_b_results if row.get("operator_node_count")]
    if len(n_nodes) >= 2 and len(set(n_nodes)) < 2:
        failures.append("operator_node_counts_identical_across_extremes")

    return len(failures) == 0, failures
