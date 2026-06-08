#!/usr/bin/env python3
"""Decisive component-level physics identity audit (read-only). Tests 10 contamination hypotheses."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent / "docs"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import load_lhs_pool  # noqa: E402
from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: E402
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    build_physics_identity_manifest,
    extract_active_block_hashes,
    extract_pressure_subblock_hashes,
    mesh_component_hashes,
    scan_cross_sample_path_contamination,
)
from v2_b3_m4_production_contracts import evaluate_production_acceptance  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel  # noqa: E402

GUITARS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars")
BAND_375 = (370.0, 385.0)
BAND_529 = (528.0, 533.0)
DEFAULT_OUT = DOCS_DIR / "M4_DECISIVE_PHYSICS_IDENTITY_AUDIT.json"


def _parse_samples(text: str) -> List[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _run_root(repo_root: Path, sample_id: str, run_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


def _modes_in_band(catalog_path: Path, band: Tuple[float, float]) -> List[Dict[str, Any]]:
    if not catalog_path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        f = m.get("frequency_hz")
        if f is None:
            continue
        if band[0] <= float(f) <= band[1]:
            out.append(m)
    return sorted(out, key=lambda r: float(r["frequency_hz"]))


def _bridge_outlier(catalog_path: Path) -> Optional[Dict[str, Any]]:
    modes: List[Dict[str, Any]] = []
    if not catalog_path.is_file():
        return None
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        b = m.get("bridge_excitation_abs")
        if b is not None:
            modes.append(m)
    if not modes:
        return None
    return max(modes, key=lambda r: float(r.get("bridge_excitation_abs") or 0.0))


def _hypothesis_verdicts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Classify each hypothesis: CONFIRMED / REJECTED / INCONCLUSIVE."""
    v: Dict[str, str] = {}
    mesh_groups = defaultdict(list)
    app_groups = defaultdict(list)
    fp_groups = defaultdict(list)
    contam = any(r.get("path_contamination", {}).get("contamination_detected") for r in rows)
    v["H1_cross_sample_first_batch_reuse"] = "CONFIRMED" if contam else "REJECTED"

    for r in rows:
        mc = r.get("mesh_components") or {}
        topo = mc.get("full_mesh_topology_sha256")
        if topo:
            mesh_groups[topo].append(r["sample_id"])
        blocks = (r.get("operator_blocks") or {}).get("blocks") or {}
        app = ((blocks.get("A_pp") or {}).get("structure_sha256"))
        if app:
            app_groups[app].append(r["sample_id"])
        for m in r.get("band_529_modes") or []:
            fp = m.get("eigenvector_fingerprint_sha256")
            if fp:
                fp_groups[fp].append((r["sample_id"], m.get("frequency_hz")))

    v["H2_same_sample_stale_resume"] = "INCONCLUSIVE"
    v["H3_identical_acoustic_submesh"] = (
        "CONFIRMED" if any(len(s) > 1 for s in mesh_groups.values()) else "REJECTED"
    )
    v["H4_identical_A_pp_M_pp"] = (
        "CONFIRMED" if any(len(s) > 1 for s in app_groups.values()) else "REJECTED"
    )
    v["H5_fixed_exterior_air_mode"] = "INCONCLUSIVE"
    v["H6_cavity_geometry_not_propagating"] = (
        "CONFIRMED"
        if len({r.get("geometry_fingerprint") for r in rows if r.get("geometry_fingerprint")}) > 1
        and any(len(s) > 1 for s in app_groups.values())
        else "INCONCLUSIVE"
    )
    v["H7_target_centre_artifact"] = "INCONCLUSIVE"
    aperture_hashes = [
        (r.get("masks") or {}).get("p_idx_aperture_sha256")
        for r in rows
        if (r.get("masks") or {}).get("p_idx_aperture_sha256")
    ]
    if len(aperture_hashes) < 2:
        v["H8_aperture_mask_reuse"] = "INCONCLUSIVE"
    elif len(set(aperture_hashes)) == 1:
        v["H8_aperture_mask_reuse"] = "CONFIRMED"
    else:
        v["H8_aperture_mask_reuse"] = "REJECTED"
    v["H9_bridge_mask_reuse"] = "INCONCLUSIVE"
    v["H10_duplicate_eigenmodes_overlapping_targets"] = (
        "CONFIRMED" if any(len(g) > 1 for g in fp_groups.values()) else "REJECTED"
    )
    return v


def _resolve_run_id(
    pool: Mapping[str, Any],
    sample_id: str,
    *,
    run_id_suffix: str,
    run_id_map: Optional[Mapping[str, str]] = None,
) -> str:
    if run_id_map and sample_id in run_id_map:
        return str(run_id_map[sample_id])
    entry = next((e for e in pool.get("entries") or [] if str(e.get("id")) == sample_id), {})
    return str(entry.get("last_run_id") or f"{sample_id}_{run_id_suffix}")


def audit_sample(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
    run_id_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    run_id = _resolve_run_id(pool, sample_id, run_id_suffix=run_id_suffix, run_id_map=run_id_map)
    run_root = _run_root(repo_root, sample_id, run_id)
    entry = next((e for e in pool.get("entries") or [] if str(e.get("id")) == sample_id), {})
    sample_input_path = run_root / "sample" / "sample_input.json"
    sample_input = load_json(sample_input_path) if sample_input_path.is_file() else {"sample_id": sample_id}
    geom = extract_geometry_dict(entry.get("parameters") or sample_input)

    ckpt = run_root / "lprod" / "checkpoint"
    built_path = ckpt / "built_metadata.json"
    built: Dict[str, Any] = {}
    if built_path.is_file():
        built = load_json(built_path)
    elif (run_root / "freeze" / "physics_identity_manifest.json").is_file():
        built = load_json(run_root / "freeze" / "physics_identity_manifest.json")

    mesh_path = run_root / "lprod" / "mesh" / "L_prod" / f"{sample_id}.msh"
    identity_manifest_path = run_root / "freeze" / "physics_identity_manifest.json"
    if identity_manifest_path.is_file():
        identity = load_json(identity_manifest_path)
    elif built_path.is_file():
        acceptance = evaluate_production_acceptance(run_root=run_root, sample_input=sample_input)
        identity = build_physics_identity_manifest(
            run_root=run_root,
            repo_root=repo_root,
            sample_input=sample_input,
            acceptance=acceptance,
        )
    else:
        identity = {"status": "checkpoint_compacted"}

    catalog = run_root / "aggregation" / "modes_catalog.jsonl"
    band_375_modes = _modes_in_band(catalog, BAND_375)
    band_529_modes = _modes_in_band(catalog, BAND_529)
    bridge_peak = _bridge_outlier(catalog)
    cavity_p = list((identity.get("masks") or {}).get("p_idx_cavity") or [])
    pressure_subblocks = extract_pressure_subblock_hashes(ckpt, built, cavity_p_idx=cavity_p)

    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": rel(run_root, repo_root=repo_root),
        "geometry_fingerprint": geometry_fingerprint(geom) if geom else None,
        "geometry": geom,
        "identity_manifest_present": identity_manifest_path.is_file(),
        "path_contamination": scan_cross_sample_path_contamination(run_root, sample_id=sample_id),
        "mesh_components": identity.get("mesh_components") or mesh_component_hashes(mesh_path, built_meta=built),
        "operator_blocks": identity.get("operator_blocks") or extract_active_block_hashes(ckpt, built),
        "masks": identity.get("masks") or {},
        "A_active_csr": identity.get("A_active_csr"),
        "M_active_csr": identity.get("M_active_csr"),
        "band_375_modes": band_375_modes,
        "band_529_modes": band_529_modes,
        "bridge_outlier_mode": bridge_peak,
        "pressure_subblocks": pressure_subblocks,
        "target_plan_frequencies_hz": _target_plan_frequencies(run_root),
    }
    for band_key, modes in (("band_375_summary", band_375_modes), ("band_529_summary", band_529_modes)):
        if not modes:
            continue
        freqs = [float(m["frequency_hz"]) for m in modes]
        mics = [float(m.get("mic_output_proxy") or 0) for m in modes]
        row[band_key] = {
            "count": len(modes),
            "freq_min": min(freqs),
            "freq_max": max(freqs),
            "freq_span_hz": max(freqs) - min(freqs),
            "mic_min": min(mics),
            "mic_max": max(mics),
            "cavity_air_shares": [m.get("cavity_air_share") for m in modes],
            "exterior_air_shares": [m.get("exterior_air_share") for m in modes],
            "coupling_classes": [m.get("coupling_class") for m in modes],
            "target_hz": [m.get("target_hz") for m in modes],
            "chunk_ids": [m.get("chunk_id") for m in modes],
            "fingerprints": [m.get("eigenvector_fingerprint_sha256") for m in modes],
            "bridge_excitation_abs": [m.get("bridge_excitation_abs") for m in modes],
            "lambda_reals": [m.get("lambda_real") for m in modes],
        }
    return row


def _target_plan_frequencies(run_root: Path) -> List[float]:
    plan_path = run_root / "lprod" / "lprod_target_plan.json"
    if not plan_path.is_file():
        return []
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return [float(x) for x in (plan.get("targets_hz") or [])]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Decisive M4 physics identity audit (read-only).")
    parser.add_argument("--sample-ids", default="sample_001,sample_002,sample_003")
    parser.add_argument("--run-id-suffix", default="m4prod2")
    parser.add_argument(
        "--run-id-map",
        default="",
        help='JSON object mapping sample_id to exact run_id, e.g. {"sample_001":"sample_001_m4prod2","sample_002":"sample_002_m4prod2_strict_val"}',
    )
    parser.add_argument("--lhs-json", type=Path, default=Path("ROM/classic/lhs_pool.json"))
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    sample_ids = _parse_samples(str(args.sample_ids))
    run_id_map: Dict[str, str] = {}
    if str(args.run_id_map or "").strip():
        run_id_map = json.loads(str(args.run_id_map))

    rows = [
        audit_sample(
            repo_root=repo_root,
            pool=pool,
            sample_id=sid,
            run_id_suffix=str(args.run_id_suffix),
            run_id_map=run_id_map or None,
        )
        for sid in sample_ids
    ]

    cross: Dict[str, Any] = {"A_pp_structure_groups": defaultdict(list), "mesh_topology_groups": defaultdict(list)}
    for r in rows:
        sid = r["sample_id"]
        blocks = (r.get("operator_blocks") or {}).get("blocks") or {}
        app = (blocks.get("A_pp") or {}).get("structure_sha256")
        if app:
            cross["A_pp_structure_groups"][app].append(sid)
        topo = (r.get("mesh_components") or {}).get("full_mesh_topology_sha256")
        if topo:
            cross["mesh_topology_groups"][topo].append(sid)
    cross["A_pp_structure_groups"] = {k: v for k, v in cross["A_pp_structure_groups"].items() if len(v) > 1}
    cross["mesh_topology_groups"] = {k: v for k, v in cross["mesh_topology_groups"].items() if len(v) > 1}

    band_freqs = [
        float(m["frequency_hz"])
        for r in rows
        for m in (r.get("band_529_modes") or [])
    ]
    fps = [
        m.get("eigenvector_fingerprint_sha256")
        for r in rows
        for m in (r.get("band_529_modes") or [])
        if m.get("eigenvector_fingerprint_sha256")
    ]
    hypotheses = _hypothesis_verdicts(rows)

    has_identity_data = any(
        (r.get("mesh_components") or {}).get("status") == "ok"
        or (r.get("operator_blocks") or {}).get("status") == "ok"
        for r in rows
    )
    if not has_identity_data or not band_freqs:
        root_cause_529 = "INCONCLUSIVE_REQUIRES_RUNTIME_PROOF"
    elif cross["A_pp_structure_groups"] or cross["mesh_topology_groups"]:
        root_cause_529 = "OPERATOR_OR_MESH_IDENTITY_SHARED"
    elif len(set(fps)) < len(fps) and fps:
        root_cause_529 = "EIGENVECTOR_REUSE"
    elif band_freqs and (max(band_freqs) - min(band_freqs)) < 0.05 and len(set(fps)) == len(fps):
        root_cause_529 = "INCONCLUSIVE_REQUIRES_RUNTIME_PROOF"
    else:
        root_cause_529 = "INDEPENDENT_SOLVES_SIMILAR_AIR_FAMILY"

    report = {
        "schema": "m4_decisive_physics_identity_audit_v1",
        "sample_ids": sample_ids,
        "per_sample": rows,
        "cross_sample": cross,
        "hypotheses": hypotheses,
        "band_529_cross_sample": {
            "freq_span_hz": (max(band_freqs) - min(band_freqs)) if band_freqs else None,
            "unique_fingerprints": len(set(fps)),
            "fingerprint_count": len(fps),
        },
        "root_cause_529_hz_family": root_cause_529,
        "verdict": "INCONCLUSIVE" if root_cause_529 == "INCONCLUSIVE_REQUIRES_RUNTIME_PROOF" else root_cause_529,
        "samples_000_003_decision": "HOLD pending VM audit with checkpoint or strict validation rerun",
    }

    out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"verdict={report['verdict']}")
    print(f"root_cause_529={root_cause_529}")
    print(f"hypotheses={json.dumps(hypotheses)}")
    print(f"wrote {rel(out, repo_root=repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
