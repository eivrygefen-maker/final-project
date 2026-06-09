#!/usr/bin/env python3
"""Durable scout/modal provenance for mesh-profile comparison (post-cleanup)."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    VALIDATION_INPUT_MANIFEST_NAME,
    VALIDATION_INPUT_PACKAGE_REL,
    sha256_file,
)
from v2_b3_m4_scout_intrinsic_coverage import (  # noqa: E402
    PRODUCTION_BAND_HI_HZ,
    PRODUCTION_BAND_LO_HZ,
    per_third_band_counts,
)
from v2_b3_m4_worker_run_lib import load_json  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SCOUT_INTRINSIC_SUMMARY_NAME = "scout_intrinsic_summary.json"
EXTERNAL_VALIDATION_INPUTS_REL = Path("pipeline_runs/validation_inputs")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def catalog_frequencies(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    return [float(r.get("frequency_hz") or 0.0) for r in rows if r.get("frequency_hz") is not None]


def derive_band_third_counts_from_catalog(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    freqs = catalog_frequencies(rows)
    return per_third_band_counts(freqs, band_lo=PRODUCTION_BAND_LO_HZ, band_hi=PRODUCTION_BAND_HI_HZ)


def mode_family_key(row: Mapping[str, Any], *, band_width_hz: float = 50.0) -> str:
    freq = float(row.get("frequency_hz") or 0.0)
    band = int(freq // band_width_hz) * int(band_width_hz)
    coupling = str(row.get("coupling_class") or "unknown")
    region = str(row.get("dominant_region") or "unknown")
    return f"{coupling}|{region}|{band}"


def major_mode_families(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_count: int = 2,
    top_bridge_rank: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Families with >=min_count modes or a top bridge-excitation representative."""
    counts: Counter[str] = Counter()
    reps: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = mode_family_key(row)
        counts[key] += 1
        if key not in reps:
            reps[key] = row
    bridge_top = sorted(
        rows,
        key=lambda r: float(r.get("bridge_excitation_abs") or 0.0),
        reverse=True,
    )[:top_bridge_rank]
    bridge_family_keys = {mode_family_key(r) for r in bridge_top}
    out: Dict[str, Dict[str, Any]] = {}
    for key, count in counts.items():
        if count >= min_count or key in bridge_family_keys:
            rep = reps[key]
            out[key] = {
                "family_key": key,
                "reference_mode_count": count,
                "representative_frequency_hz": rep.get("frequency_hz"),
                "coupling_class": rep.get("coupling_class"),
                "dominant_region": rep.get("dominant_region"),
                "bridge_excitation_abs": rep.get("bridge_excitation_abs"),
                "major_by_count": count >= min_count,
                "major_by_bridge_top10": key in bridge_family_keys,
            }
    return out


def compare_mode_family_survival(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ref_families = major_mode_families(ref_rows)
    cand_keys = {mode_family_key(r) for r in cand_rows}
    missing = [fam for key, fam in ref_families.items() if key not in cand_keys]
    new_keys = sorted(cand_keys - set(ref_families))
    return {
        "reference_major_family_count": len(ref_families),
        "candidate_family_count_observed": len(cand_keys),
        "reference_families_present": sorted(ref_families.keys()),
        "candidate_families_present": sorted(cand_keys),
        "missing_reference_families": missing,
        "new_candidate_families": new_keys,
        "unexplained_family_loss_count": len(missing),
        "family_survival_pass": len(missing) == 0,
    }


def compare_intrinsic_band_third_coverage(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    *,
    ref_scout: Optional[Mapping[str, Any]] = None,
    cand_scout: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ref_counts = (ref_scout or {}).get("per_zone_mode_counts") or derive_band_third_counts_from_catalog(ref_rows)
    cand_counts = (cand_scout or {}).get("per_zone_mode_counts") or derive_band_third_counts_from_catalog(cand_rows)
    missing_bands: List[str] = []
    population_changes: List[Dict[str, Any]] = []
    for band in ("low_third", "mid_third", "high_third"):
        ref_n = int(ref_counts.get(band) or 0)
        cand_n = int(cand_counts.get(band) or 0)
        if ref_n >= 2 and cand_n < ref_n:
            population_changes.append(
                {"band_third": band, "reference_count": ref_n, "candidate_count": cand_n, "material_decrease": True}
            )
        if ref_n >= 2 and cand_n == 0:
            missing_bands.append(band)
    return {
        "reference_band_third_counts": ref_counts,
        "candidate_band_third_counts": cand_counts,
        "missing_covered_band_thirds": missing_bands,
        "material_population_changes": population_changes,
        "intrinsic_band_third_no_loss_pass": len(missing_bands) == 0 and not population_changes,
    }


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def load_durable_scout_intrinsic_summary(run_root: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Post-cleanup scout intrinsic loader. Never reads scout/discovery or meshes."""
    errors: List[str] = []
    run_root = run_root.resolve()
    pkg = run_root / VALIDATION_INPUT_PACKAGE_REL / SCOUT_INTRINSIC_SUMMARY_NAME
    if pkg.is_file():
        body = _load_json_if_exists(pkg)
        return body, errors

    derived: Dict[str, Any] = {"source": "derived_from_durable_catalog", "per_zone_mode_counts": None}
    catalog_path = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    if not catalog_path.is_file():
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    if catalog_path.is_file():
        rows: List[Dict[str, Any]] = []
        for line in catalog_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        derived["per_zone_mode_counts"] = derive_band_third_counts_from_catalog(rows)
        derived["deduped_mode_count"] = len(rows)
    else:
        errors.append("missing_modes_catalog_for_intrinsic_derivation")

    manifest = _load_json_if_exists(run_root / "pipeline_run_manifest.json") or {}
    scout_stage = (manifest.get("stages") or {}).get("stage2_scout_discovery") or {}
    if scout_stage.get("intrinsic_coverage_pass") is not None:
        derived["intrinsic_coverage_pass"] = scout_stage.get("intrinsic_coverage_pass")
    density = _load_json_if_exists(run_root / "scout" / "density_zones.json") or {}
    for key in (
        "intrinsic_coverage_pass",
        "intrinsic_coverage_failures",
        "raw_unique_accepted_count",
        "raw_frequency_min_hz",
        "raw_frequency_max_hz",
        "raw_max_gap_hz",
        "per_zone_mode_counts",
        "target_success_count",
        "target_failure_count",
    ):
        if key in density and density.get(key) is not None:
            derived[key] = density.get(key)
    if derived.get("per_zone_mode_counts"):
        return derived, errors
    if errors:
        return None, errors
    return derived, errors


def preserve_comparison_provenance_before_cleanup(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
) -> Dict[str, Any]:
    """Copy scout intrinsic + chunk plan into durable validation package before cleanup."""
    run_root = run_root.resolve()
    pkg_dir = run_root / VALIDATION_INPUT_PACKAGE_REL
    pkg_dir.mkdir(parents=True, exist_ok=True)
    preserved: Dict[str, Any] = {"status": "ok", "artifacts": []}

    density_path = run_root / "scout" / "discovery" / "density_result.json"
    scout_summary: Dict[str, Any] = {"schema": "m4_scout_intrinsic_summary_v1", "sample_id": sample_id, "run_id": run_id}
    if density_path.is_file():
        density = _load_json_if_exists(density_path) or {}
        for key in (
            "coverage_policy",
            "coverage_policy_version",
            "intrinsic_coverage_pass",
            "intrinsic_coverage_failures",
            "raw_unique_accepted_count",
            "raw_frequency_min_hz",
            "raw_frequency_max_hz",
            "raw_max_gap_hz",
            "per_zone_mode_counts",
            "target_success_count",
            "target_failure_count",
        ):
            if key in density:
                scout_summary[key] = density.get(key)
        scout_summary["source_artifact"] = "scout/discovery/density_result.json"
        scout_summary["source_sha256"] = sha256_file(density_path)
    else:
        scout_summary["source_artifact"] = None
        scout_summary["note"] = "density_result_missing_pre_cleanup"

    chunk_plan_src = run_root / "lprod" / "worker_chunk_plan.preview.json"
    if not chunk_plan_src.is_file():
        chunk_plan_src = run_root / "lprod" / "worker_chunk_plan.json"
    if chunk_plan_src.is_file():
        chunk_body = _load_json_if_exists(chunk_plan_src) or {}
        chunk_dest = pkg_dir / "worker_chunk_plan.json"
        write_json_atomic(chunk_dest, chunk_body)
        preserved["artifacts"].append(
            {
                "name": "worker_chunk_plan",
                "package_path": f"{VALIDATION_INPUT_PACKAGE_REL}/worker_chunk_plan.json",
                "sha256": sha256_file(chunk_dest),
                "source_path": str(chunk_plan_src),
            }
        )

    scout_dest = pkg_dir / SCOUT_INTRINSIC_SUMMARY_NAME
    write_json_atomic(scout_dest, scout_summary)
    preserved["artifacts"].append(
        {
            "name": "scout_intrinsic_summary",
            "package_path": f"{VALIDATION_INPUT_PACKAGE_REL}/{SCOUT_INTRINSIC_SUMMARY_NAME}",
            "sha256": sha256_file(scout_dest),
            "source_path": scout_summary.get("source_artifact"),
        }
    )

    manifest_path = pkg_dir / VALIDATION_INPUT_MANIFEST_NAME
    prior = _load_json_if_exists(manifest_path) or {}
    inputs = [row for row in (prior.get("inputs") or []) if row.get("name") not in ("scout_intrinsic_summary", "worker_chunk_plan")]
    inputs.extend(preserved["artifacts"])
    write_json_atomic(
        manifest_path,
        {
            "schema": prior.get("schema") or "m4_mesh_validation_input_package_v1",
            "sample_id": sample_id,
            "run_id": run_id,
            "inputs": inputs,
        },
    )
    return preserved


def physics_identity_hash(run_root: Path) -> Optional[str]:
    path = run_root / "freeze" / "physics_identity_manifest.json"
    if not path.is_file():
        return None
    return sha256_file(path)


def material_fingerprint(sample_input: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "top_wood_id": sample_input.get("top_wood_id"),
            "back_wood_id": sample_input.get("back_wood_id"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reconstruct_target_plan_from_durable(
    run_root: Path,
) -> Tuple[Optional[Dict[str, Any]], str, List[str], List[Dict[str, str]]]:
    """
    Read-only exact reconstruction from durable artifacts only.
    Returns (plan_body, method, errors, source_artifacts).
    """
    errors: List[str] = []
    sources: List[Dict[str, str]] = []
    run_root = run_root.resolve()

    durable = run_root / VALIDATION_INPUT_PACKAGE_REL / "target_plan.json"
    if durable.is_file():
        body = _load_json_if_exists(durable) or {}
        sources.append({"path": str(durable.relative_to(run_root)), "sha256": sha256_file(durable) or ""})
        return body, "preserved_exact", errors, sources

    chunk_candidates = (
        run_root / VALIDATION_INPUT_PACKAGE_REL / "worker_chunk_plan.json",
        run_root / "lprod" / "worker_chunk_plan.preview.json",
        run_root / "worker_results" / "remaining_workers_m4_4_1b_4_manifest.json",
    )
    targets: List[float] = []
    chunks: List[Dict[str, Any]] = []
    for path in chunk_candidates:
        if not path.is_file():
            continue
        doc = _load_json_if_exists(path) or {}
        sources.append({"path": str(path.relative_to(run_root)), "sha256": sha256_file(path) or ""})
        for chunk in doc.get("chunks") or doc.get("chunk_results") or []:
            if not isinstance(chunk, dict):
                continue
            chunks.append(chunk)
            for t in chunk.get("targets_hz") or []:
                targets.append(float(t))
        if targets:
            break

    if not targets:
        manifest = _load_json_if_exists(run_root / "pipeline_run_manifest.json") or {}
        st = (manifest.get("stages") or {}).get("stage3_zones_plan") or {}
        for rel in st.get("artifact_paths") or []:
            p = run_root.parent.parent.parent / rel if rel.startswith("FEM/") else run_root / Path(rel).name
            if not p.is_file():
                p2 = run_root / Path(str(rel).replace("\\", "/").split("/")[-1])
                p = p2 if p2.is_file() else p
            if p.is_file() and "target_plan" in p.name:
                doc = _load_json_if_exists(p) or {}
                sources.append({"path": str(p), "sha256": sha256_file(p) or ""})
                targets = [float(x) for x in (doc.get("targets_hz") or [])]
                chunks = list(doc.get("chunks") or [])
                if targets:
                    break

    if not targets:
        errors.append("TARGET_PLAN_UNAVAILABLE")
        return None, "unavailable", errors, sources

    ordered = sorted(set(targets))
    body = {
        "schema": "m4_lprod_target_plan_v1",
        "targets_hz": ordered,
        "target_count": len(ordered),
        "frequency_range_hz": [min(ordered), max(ordered)],
        "chunks": chunks,
        "reconstruction_method": "reconstructed_exact_from_durable_provenance",
        "source_artifacts": sources,
    }
    return body, "reconstructed_exact_from_durable_provenance", errors, sources


def create_external_validation_input_package(
    *,
    repo_root: Path,
    reference_run_root: Path,
    sample_id: str,
    run_id: str,
    geometry_fingerprint: Optional[str] = None,
    material_fp: Optional[str] = None,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Create immutable external validation package; never writes into reference run."""
    plan, method, errors, sources = reconstruct_target_plan_from_durable(reference_run_root)
    report: Dict[str, Any] = {
        "TARGET_PLAN_READY": False,
        "creation_method": method,
        "errors": errors,
        "source_artifacts": sources,
    }
    if plan is None or errors:
        return None, report

    identity_hash = physics_identity_hash(reference_run_root) or "unknown"
    pkg_name = f"sample_{sample_id}_reference_{identity_hash[:16]}"
    pkg_root = (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/validation_inputs"
        / pkg_name
    )
    pkg_root.mkdir(parents=True, exist_ok=True)
    plan_path = pkg_root / "target_plan.json"
    write_json_atomic(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    manifest = {
        "schema": "m4_external_validation_input_package_v1",
        "sample_id": sample_id,
        "reference_run_id": run_id,
        "reference_run_root": str(reference_run_root),
        "geometry_fingerprint": geometry_fingerprint,
        "material_fingerprint": material_fp,
        "physics_identity_hash": identity_hash,
        "target_plan_sha256": plan_sha,
        "target_count": len(plan.get("targets_hz") or []),
        "frequency_range_hz": plan.get("frequency_range_hz"),
        "targets_hz": plan.get("targets_hz"),
        "chunk_count": len(plan.get("chunks") or []),
        "creation_method": method,
        "creation_reason": "mesh_profile_validation_input",
        "source_artifacts": sources,
        "created_utc": _utc_now(),
        "read_only": True,
    }
    write_json_atomic(pkg_root / "validation_input_manifest.json", manifest)
    report.update(
        {
            "TARGET_PLAN_READY": True,
            "package_root": str(pkg_root),
            "target_plan_sha256": plan_sha,
            "target_count": manifest["target_count"],
        }
    )
    return pkg_root, report
