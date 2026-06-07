#!/usr/bin/env python3
"""Read-only operator vs generated-mesh provenance audit for M4 full-retention samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import DEFAULT_RUN_ID_SUFFIX, lhs_entry_index, load_lhs_pool  # noqa: E402
from v2_b3_m4_lprod_interfaces import (  # noqa: E402
    BASELINE_L_PROD_MESH,
    extract_geometry_dict,
    geometry_fingerprint,
)
from v2_b3_rich_modal_lib import REGION_DOF_INDICES_NPZ, load_region_dof_bundle  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel  # noqa: E402
from v2_mesh_convergence_common import mesh_path  # noqa: E402

DEFAULT_LHS = "ROM/classic/lhs_pool.json"
GUITARS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars")
CASE_ID = "baseline_coupled_v2"
MESH_LEVEL = "L_prod"
DEFAULT_SAMPLES = ("sample_000", "sample_001", "sample_034", "sample_035")


def _parse_sample_ids(arg: str) -> List[str]:
    out: List[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            out.append(f"sample_{int(part):03d}")
        else:
            out.append(part)
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, max_bytes: Optional[int] = None) -> Optional[str]:
    if not path.is_file():
        return None
    data = path.read_bytes() if max_bytes is None else path.read_bytes()[:max_bytes]
    return _sha256_bytes(data)


def _csr_hashes(npz_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": str(npz_path), "present": npz_path.is_file()}
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
                "file_sha256_sample_1mb": _sha256_file(npz_path, max_bytes=1_048_576),
                "file_sha256_full": _sha256_file(npz_path),
            }
        )
    return out


def _mesh_counts_from_gmsh(path: Path) -> Dict[str, Any]:
    """Best-effort node/tet count without dolfinx (parse $Nodes / $Elements)."""
    if not path.is_file():
        return {"present": False}
    n_nodes = 0
    n_tets = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "$Nodes" in text:
            block = text.split("$Nodes", 1)[1].split("$EndNodes", 1)[0].strip().splitlines()
            if block:
                n_nodes = int(block[0].split()[0]) if block[0].split()[0].isdigit() else len(block)
        if "$Elements" in text:
            for line in text.split("$Elements", 1)[1].split("$EndElements", 1)[0].splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "4":
                    n_tets += 1
    except (OSError, ValueError, IndexError):
        pass
    return {
        "present": True,
        "path": str(path),
        "sha256_sample_2mb": _sha256_file(path, max_bytes=2_000_000),
        "sha256_full": _sha256_file(path),
        "gmsh_parsed_n_nodes": n_nodes or None,
        "gmsh_parsed_n_tets": n_tets or None,
        "size_bytes": path.stat().st_size,
    }


def _dolfinx_mesh_audit(path: Path) -> Dict[str, Any]:
    """Optional DOLFINx load for coordinate bbox and topology counts."""
    try:
        from v2_b3_synthesis_export import import_fem_main_3d  # noqa: WPS433

        fem3d, _diag = import_fem_main_3d(start=SCRIPT_DIR)

        msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(path)
        coords = np.asarray(msh.geometry.x)
        tdim = msh.topology.dim
        n_cells = msh.topology.index_map(tdim).size_local
        return {
            "dolfinx_available": True,
            "n_nodes": int(coords.shape[0]),
            "n_cells": int(n_cells),
            "coord_shape": list(coords.shape),
            "bbox_min": coords.min(axis=0).tolist(),
            "bbox_max": coords.max(axis=0).tolist(),
            "coord_sha256_sample": _sha256_bytes(coords[: min(1000, len(coords))].tobytes()),
            "coord_sha256_full": _sha256_bytes(coords.tobytes()),
            "n_facet_tag2_soundhole": int(np.asarray(facet_tags.find(2)).size),
            "n_cell_tag10_air": int(np.asarray(cell_tags.find(10)).size),
        }
    except Exception as exc:  # noqa: BLE001
        return {"dolfinx_available": False, "error": f"{type(exc).__name__}:{exc}"}


def _resolve_run(pool: Mapping[str, Any], sample_id: str, run_id_suffix: str) -> Tuple[str, Path, Dict[str, Any]]:
    idx = lhs_entry_index(pool, sample_id)
    entry = (pool.get("entries") or [])[idx] if idx is not None else {}
    run_id = str(entry.get("last_run_id") or f"{sample_id}_{run_id_suffix}")
    run_root = detect_repo_root(SCRIPT_DIR) / GUITARS_REL / sample_id / "runs" / run_id
    return run_id, run_root, entry


def audit_one_sample(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
    load_dolfinx: bool,
) -> Dict[str, Any]:
    run_id, run_root, entry = _resolve_run(pool, sample_id, run_id_suffix)
    geom = extract_geometry_dict(entry)
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": rel(run_root, repo_root=repo_root),
        "geometry_fingerprint": geometry_fingerprint(geom) if geom else None,
        "lhs_geometry": geom,
    }

    generated_msh = run_root / "lprod" / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    summary_path = generated_msh.parent / f"{sample_id}_mesh_build_summary.json"
    operator_baseline = mesh_path(MESH_LEVEL, CASE_ID)
    resolved_cfg = run_root / "lprod" / "resolved_core_config.json"

    row["generated_gmsh"] = _mesh_counts_from_gmsh(generated_msh)
    if summary_path.is_file():
        try:
            summary = load_json(summary_path)
            row["mesh_build_summary"] = {
                "n_nodes": summary.get("n_nodes"),
                "n_tetrahedra": summary.get("n_tetrahedra"),
                "geometry": summary.get("geometry"),
                "mesh_path": summary.get("mesh_path"),
            }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    cfg_mesh: Optional[str] = None
    if resolved_cfg.is_file():
        try:
            cfg = load_json(resolved_cfg)
            cfg_mesh = (cfg.get("solver") or {}).get("mesh_file")
            row["resolved_core_config_mesh_file"] = cfg_mesh
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    row["operator_mesh_canonical_path"] = str(operator_baseline.resolve())
    row["operator_mesh_canonical_sha256"] = _sha256_file(operator_baseline, max_bytes=2_000_000)
    row["baseline_l_prod_mesh_path"] = str(BASELINE_L_PROD_MESH.resolve())

    if load_dolfinx:
        row["generated_gmsh_dolfinx"] = _dolfinx_mesh_audit(generated_msh)
        row["operator_mesh_dolfinx"] = _dolfinx_mesh_audit(operator_baseline)
        if cfg_mesh:
            cfg_path = Path(cfg_mesh)
            if not cfg_path.is_absolute():
                cfg_path = repo_root / cfg_mesh
            if cfg_path.is_file():
                row["resolved_config_mesh_dolfinx"] = _dolfinx_mesh_audit(cfg_path)

    ckpt = run_root / "lprod" / "checkpoint"
    built_path = ckpt / "built_metadata.json"
    if built_path.is_file():
        built = load_json(built_path)
        row["checkpoint"] = {
            "built_metadata_sha256": _sha256_file(built_path),
            "mesh_level": built.get("mesh_level"),
            "n_w": built.get("n_w"),
            "n_u_b3": built.get("n_u_b3"),
            "active_dimension": built.get("active_dimension"),
            "n_p_air_estimate": len(built.get("p_idx") or []),
            "region_dof_mesh_file": built.get("region_dof_mesh_file"),
            "operator_build_mesh_file_expected": str(operator_baseline.resolve()),
        }
        region_ctx = load_region_dof_bundle(ckpt, built)
        region = region_ctx.get("region") or {}
        row["region_dof"] = {
            "npz_present": region_ctx.get("npz_present"),
            "region_dof_source": region_ctx.get("region_dof_source"),
            "n_u_idx_soundhole": int(np.asarray(region.get("u_idx_soundhole", [])).size),
            "n_p_idx_air": int(np.asarray(region.get("p_idx_air", [])).size),
            "soundhole_index_sha256": hashlib.sha256(
                ",".join(str(int(i)) for i in np.asarray(region.get("u_idx_soundhole", [])).ravel()).encode()
            ).hexdigest()
            if np.asarray(region.get("u_idx_soundhole", [])).size
            else hashlib.sha256(b"").hexdigest(),
        }
        for name in ("A_active_csr.npz", "M_active_csr.npz"):
            row["checkpoint"][name] = _csr_hashes(ckpt / name)
    else:
        row["checkpoint"] = {"status": "missing_or_compacted"}

    row["provenance_chain"] = [
        "LHS entry → sample_input.json / resolved_core_config.json (materials + geometry metadata)",
        f"Stage 4 mesh build → {rel(generated_msh, repo_root=repo_root)} (provenance artifact)",
        f"Stage A operator build → hard-coded {rel(operator_baseline, repo_root=repo_root)} via mesh_path(L_prod, baseline_coupled_v2)",
        "A/M assembly → _assemble_reduced_coupled_replay(mesh_file=baseline, core_config=resolved)",
        "Active restriction → identical sparsity; values vary with materials",
    ]
    row["mesh_mismatch_flag"] = (
        row.get("generated_gmsh", {}).get("gmsh_parsed_n_nodes") is not None
        and row.get("mesh_build_summary", {}).get("n_nodes") is not None
        and row.get("checkpoint", {}).get("n_w") is not None
        and row["generated_gmsh"]["gmsh_parsed_n_nodes"] != row["checkpoint"]["n_w"]
    )
    return row


def _cross_sample_flags(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def _group(key_fn) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = {}
        for r in rows:
            k = key_fn(r)
            if k:
                g.setdefault(str(k), []).append(str(r["sample_id"]))
        return {k: v for k, v in g.items() if len(v) > 1}

    a_struct: Dict[str, List[str]] = {}
    a_vals: Dict[str, List[str]] = {}
    dims: Dict[str, List[str]] = {}
    for r in rows:
        ck = r.get("checkpoint") or {}
        for mat in ("A_active_csr.npz", "M_active_csr.npz"):
            info = ck.get(mat) or {}
            s = info.get("structure_sha256")
            v = info.get("values_sha256")
            if s:
                a_struct.setdefault(s, []).append(f"{r['sample_id']}:{mat}")
            if v:
                a_vals.setdefault(v, []).append(f"{r['sample_id']}:{mat}")
        sig = (
            f"n_w={ck.get('n_w')}|active={ck.get('active_dimension')}|"
            f"n_p={ck.get('n_p_air_estimate')}|n_u={ck.get('n_u_b3')}"
        )
        if ck.get("n_w"):
            dims.setdefault(sig, []).append(str(r["sample_id"]))

    return {
        "identical_dimension_signature_across_all": len(dims) == 1 and len(rows) > 1,
        "dimension_signature_groups": dims,
        "identical_A_M_structure_hash_groups": _group(
            lambda r: ((r.get("checkpoint") or {}).get("A_active_csr.npz") or {}).get("structure_sha256")
        ),
        "identical_A_M_values_hash_groups": _group(
            lambda r: ((r.get("checkpoint") or {}).get("A_active_csr.npz") or {}).get("values_sha256")
        ),
        "all_soundhole_structural_dofs_empty": all(
            (r.get("region_dof") or {}).get("n_u_idx_soundhole", 0) == 0 for r in rows
        ),
        "generated_mesh_node_counts_differ": len(
            {((r.get("mesh_build_summary") or {}).get("n_nodes")) for r in rows if (r.get("mesh_build_summary") or {}).get("n_nodes")}
        )
        > 1,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4 operator mesh provenance audit (read-only).")
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--dolfinx", action="store_true", help="Load meshes with DOLFINx for coordinate audit.")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    sample_ids = _parse_sample_ids(str(args.samples))

    rows = [
        audit_one_sample(
            repo_root=repo_root,
            pool=pool,
            sample_id=sid,
            run_id_suffix=str(args.run_id_suffix),
            load_dolfinx=bool(args.dolfinx),
        )
        for sid in sample_ids
    ]
    report = {
        "schema": "m4_operator_provenance_audit_v1",
        "sample_ids": sample_ids,
        "per_sample": rows,
        "cross_sample": _cross_sample_flags(rows),
        "root_cause_summary": (
            "Stage A operator assembly uses canonical baseline_coupled_v2 mesh, not per-sample "
            "lprod/mesh/L_prod/sample_XXX.msh. Identical W/active_dimension/CSR structure across "
            "samples is expected under material-overlay parametric model; mesh_build_summary counts "
            "describe a different generated artifact."
        ),
    }

    print(f"audited_samples={len(rows)}")
    cs = report["cross_sample"]
    print(f"identical_dimension_signature={cs.get('identical_dimension_signature_across_all')}")
    print(f"generated_mesh_node_counts_differ={cs.get('generated_mesh_node_counts_differ')}")
    print(f"all_soundhole_structural_dofs_empty={cs.get('all_soundhole_structural_dofs_empty')}")
    for r in rows:
        gen_n = (r.get("mesh_build_summary") or {}).get("n_nodes")
        ck = r.get("checkpoint") or {}
        print(
            f"  {r['sample_id']}: gen_nodes={gen_n} n_w={ck.get('n_w')} "
            f"active={ck.get('active_dimension')} mesh_mismatch={r.get('mesh_mismatch_flag')}"
        )
        a = (ck.get("A_active_csr.npz") or {})
        if a.get("structure_sha256"):
            print(f"    A structure={a['structure_sha256'][:16]}... values={a.get('values_sha256','')[:16]}...")

    if args.json_out:
        out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"json_out={rel(out, repo_root=repo_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
