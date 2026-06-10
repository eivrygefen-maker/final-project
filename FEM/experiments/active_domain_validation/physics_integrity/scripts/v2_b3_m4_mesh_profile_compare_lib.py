#!/usr/bin/env python3
"""Mesh profile reference vs ROM comparison logic (read-only, post-cleanup)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_ROM,
    ExternalValidationInputPackage,
    LEVEL_PROD_REFERENCE,
    LEVEL_ROM_PROD,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    REFERENCE_CONTROLS_M,
    ROM_CONTROLS_M,
    VALIDATION_INPUT_PACKAGE_REL,
    evaluate_legacy_reference_compatibility,
    load_durable_target_plan,
    load_external_validation_package,
    load_target_plan_file,
    sha256_file,
    validation_input_manifest_path,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    AGG_PASS,
    find_bookkeeping_reconciliation_report,
    is_run_usably_complete,
    read_run_production_summary,
)
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    PHYSICS_IDENTITY_MANIFEST,
    count_forbidden_heavy_artifacts,
    iter_physics_scan_files,
    iter_path_like_strings_from_json,
    validate_physics_identity_manifest,
)
from v2_b3_m4_sample_cleanup_barrier import (  # noqa: E402
    FAILURE_REPORT_REL,
    collect_shared_sample_artifact_paths,
    load_cleanup_barrier_manifest,
    require_cleanup_barrier_passed_for_validation,
    verify_success_durable_outputs,
)
from v2_b3_m4_mesh_profile_provenance_lib import (  # noqa: E402
    canonical_target_plan_semantic,
    compare_intrinsic_band_third_coverage,
    compare_mode_family_survival,
    compare_physical_identity_projections,
    compare_target_plan_semantic,
    load_durable_scout_intrinsic_summary,
    material_fingerprint,
    physics_identity_hash,
)
from v2_b3_m4_worker_run_lib import load_json  # noqa: E402

ACCEPTANCE_THRESHOLDS = {
    "global_median_rel_freq_error_max": 0.01,
    "global_p95_rel_freq_error_max": 0.025,
    "band_60_150_max_each": 0.01,
    "band_150_350_median_max": 0.015,
    "band_150_350_max_max": 0.03,
    "band_350_550_median_max": 0.02,
    "band_350_550_max_max": 0.04,
    "coupling_class_agreement_min": 0.90,
    "bridge_top10_overlap_min": 8,
    "mic_top10_overlap_min": 7,
    "recall_below_350_min": 0.95,
    "recall_350_550_min": 0.90,
    "runtime_reduction_min": 0.25,
    "peak_rss_per_worker_gib_max": 6.5,
}

FREQUENCY_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("60_150", 60.0, 150.0),
    ("150_250", 150.0, 250.0),
    ("250_350", 250.0, 350.0),
    ("350_450", 350.0, 450.0),
    ("450_550", 450.0, 550.0),
)

RECALL_BANDS: Tuple[Tuple[str, float, float, float], ...] = (
    ("below_350", 60.0, 350.0, 0.95),
    ("350_550", 350.0, 550.0, 0.90),
)

DURABLE_COMPARE_REL = (
    "aggregation/modes_catalog_deduped.jsonl",
    "aggregation/modes_catalog.jsonl",
    "aggregation/aggregation_result.json",
    "aggregation/modes_summary.json",
    "freeze/freeze_manifest.json",
    PHYSICS_IDENTITY_MANIFEST,
    "compaction/compaction_manifest.json",
    "cleanup/sample_cleanup_barrier.json",
    "pipeline_run_manifest.json",
    "sample/sample_input.json",
    f"{VALIDATION_INPUT_PACKAGE_REL}/validation_input_manifest.json",
    f"{VALIDATION_INPUT_PACKAGE_REL}/target_plan.json",
)

EXIT_PASS = 0
EXIT_ACCEPTANCE_FAIL = 1
EXIT_PRECONDITION_FAIL = 2
EXIT_INCOMPLETE = 3

RECOMMEND_ACCEPT_ROM_BALANCED = "ACCEPT_ROM_BALANCED"
RECOMMEND_ACCEPT_WITH_CAUTION = "ACCEPT_WITH_CAUTION"
RECOMMEND_REJECT_OR_TIGHTEN_MESH = "REJECT_OR_TIGHTEN_MESH"
RECOMMEND_INCOMPLETE = "INCOMPLETE"


def _mode_id(row: Mapping[str, Any]) -> str:
    return str(row.get("mode_id") or row.get("dedup_id") or row.get("frequency_hz"))


def _greedy_freq_pairs(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    *,
    max_distance_hz: float = 5.0,
) -> List[Tuple[int, int]]:
    ref_freqs = [float(r.get("frequency_hz") or 0.0) for r in ref_rows]
    cand_freqs = [float(r.get("frequency_hz") or 0.0) for r in cand_rows]
    used_c: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for i in sorted(range(len(ref_freqs)), key=lambda k: ref_freqs[k]):
        best_j = None
        best_d = float("inf")
        for j, cf in enumerate(cand_freqs):
            if j in used_c:
                continue
            d = abs(cf - ref_freqs[i])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= max_distance_hz:
            used_c.add(best_j)
            pairs.append((i, best_j))
    return pairs


def load_catalog(run_root: Path) -> List[Dict[str, Any]]:
    for rel in ("aggregation/modes_catalog_deduped.jsonl", "aggregation/modes_catalog.jsonl"):
        path = run_root / rel
        if not path.is_file():
            continue
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    return []


def _physics_identity(run_root: Path) -> Dict[str, Any]:
    p = run_root / PHYSICS_IDENTITY_MANIFEST
    return load_json(p) if p.is_file() else {}


def _pipeline_manifest(run_root: Path) -> Dict[str, Any]:
    p = run_root / "pipeline_run_manifest.json"
    return load_json(p) if p.is_file() else {}


def _runtime_prov(run_root: Path) -> Dict[str, Any]:
    for rel in ("m4_sample_runtime_provenance.json", "aggregation/runtime_summary.json"):
        p = run_root / rel
        if p.is_file():
            try:
                return load_json(p)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return {}


def _norm_path_key(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _all_strings_in_json(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for val in obj.values():
            out.extend(_all_strings_in_json(val))
    elif isinstance(obj, list):
        for val in obj:
            out.extend(_all_strings_in_json(val))
    return out


def scan_candidate_references_other_run(candidate_root: Path, *, forbidden_root: Path) -> List[Dict[str, Any]]:
    candidate_root = candidate_root.resolve()
    forbidden_key = _norm_path_key(forbidden_root.resolve())
    hits: List[Dict[str, Any]] = []
    for rel in DURABLE_COMPARE_REL:
        path = candidate_root / rel
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if forbidden_key in line.replace("\\", "/").lower():
                    hits.append({"file": rel, "line": i, "kind": "jsonl_path_reference"})
        else:
            try:
                doc = load_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            for ps in _all_strings_in_json(doc):
                if forbidden_key in ps.replace("\\", "/").lower():
                    hits.append({"file": rel, "path_field": ps, "kind": "json_string_reference"})
                    break
    for path in iter_physics_scan_files(candidate_root):
        text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
        if forbidden_key in text.replace("\\", "/").lower():
            hits.append({"file": path.relative_to(candidate_root).as_posix(), "kind": "physics_scan"})
    return hits


def find_recorded_target_plan_sha256(run_root: Path) -> Optional[str]:
    """Return recorded target-plan SHA256 from durable run provenance, if present."""
    keys = (
        "validation_input_sha256",
        "target_plan_sha256",
        "lprod_target_plan_sha256",
        "validation_input_package_sha256",
    )
    rel_paths = (
        "pipeline_run_manifest.json",
        "m4_sample_runtime_provenance.json",
        "sample/sample_resolved_config_manifest.json",
        "aggregation/runtime_summary.json",
        "freeze/freeze_manifest.json",
        "sample/sample_input.json",
    )
    for rel in rel_paths:
        path = run_root / rel
        if not path.is_file():
            continue
        try:
            doc = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        for key in keys:
            value = doc.get(key)
            if isinstance(value, str) and len(value) == 64:
                return value
        nested = doc.get("validation_input") or doc.get("target_plan") or {}
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                if isinstance(value, str) and len(value) == 64:
                    return value
    return None


def _sample_ids_match_runs(
    *,
    sample_id: str,
    ref_root: Path,
    cand_root: Path,
    external: ExternalValidationInputPackage,
) -> List[str]:
    errors: List[str] = []
    plan_sid = str(external.target_plan.get("sample_id") or external.manifest.get("sample_id") or "")
    entry_sid = str(external.manifest_entry.get("sample_id") or "")
    if plan_sid and plan_sid != sample_id:
        errors.append(f"external_target_plan_sample_id_mismatch:{plan_sid}!={sample_id}")
    if entry_sid and entry_sid != sample_id:
        errors.append(f"external_manifest_sample_id_mismatch:{entry_sid}!={sample_id}")
    ref_in = load_json(ref_root / "sample" / "sample_input.json") if (ref_root / "sample" / "sample_input.json").is_file() else {}
    cand_in = load_json(cand_root / "sample" / "sample_input.json") if (cand_root / "sample" / "sample_input.json").is_file() else {}
    ref_sid = str(ref_in.get("sample_id") or ref_root.parent.parent.name)
    cand_sid = str(cand_in.get("sample_id") or cand_root.parent.parent.name)
    if ref_sid != cand_sid:
        errors.append(f"run_tree_sample_id_mismatch:{ref_sid}!={cand_sid}")
    if ref_sid != sample_id:
        errors.append(f"reference_sample_id_mismatch:{ref_sid}!={sample_id}")
    if cand_sid != sample_id:
        errors.append(f"candidate_sample_id_mismatch:{cand_sid}!={sample_id}")
    return errors


def _identity_matches_external_package(
    *,
    ref_root: Path,
    cand_root: Path,
    external: ExternalValidationInputPackage,
) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    meta: Dict[str, Any] = {}
    ref_in = load_json(ref_root / "sample" / "sample_input.json") if (ref_root / "sample" / "sample_input.json").is_file() else {}
    cand_in = load_json(cand_root / "sample" / "sample_input.json") if (cand_root / "sample" / "sample_input.json").is_file() else {}
    geom_r = extract_geometry_dict(ref_in)
    geom_c = extract_geometry_dict(cand_in)
    if geometry_fingerprint(geom_r) != geometry_fingerprint(geom_c):
        errors.append("geometry_fingerprint_mismatch")

    manifest_geom = external.manifest_entry.get("geometry_fingerprint")
    if manifest_geom:
        if geometry_fingerprint(geom_r) != str(manifest_geom):
            errors.append("reference_geometry_fingerprint_mismatch_vs_external_manifest")
        if geometry_fingerprint(geom_c) != str(manifest_geom):
            errors.append("candidate_geometry_fingerprint_mismatch_vs_external_manifest")

    for key in ("top_wood_id", "back_wood_id"):
        if ref_in.get(key) != cand_in.get(key):
            errors.append(f"material_mismatch:{key}")

    manifest_mat = external.manifest_entry.get("material_fingerprint")
    if manifest_mat:
        ref_mat = material_fingerprint(ref_in)
        cand_mat = material_fingerprint(cand_in)
        if ref_mat != str(manifest_mat):
            errors.append("reference_material_fingerprint_mismatch_vs_external_manifest")
        if cand_mat != str(manifest_mat):
            errors.append("candidate_material_fingerprint_mismatch_vs_external_manifest")

    manifest_phys = external.manifest_entry.get("physics_identity_hash")
    if manifest_phys:
        ref_hash = physics_identity_hash(ref_root)
        if ref_hash and ref_hash != str(manifest_phys):
            errors.append("reference_physics_identity_hash_mismatch_vs_external_manifest")

    ext_chunk_count = external.manifest_entry.get("chunk_count") or external.manifest.get("chunk_count")
    target_semantic = canonical_target_plan_semantic(
        external.target_plan,
        manifest_chunk_count=ext_chunk_count,
    )
    identity_cmp = compare_physical_identity_projections(
        ref_root,
        cand_root,
        target_plan_semantic=target_semantic,
    )
    meta["physical_identity"] = identity_cmp
    if not identity_cmp.get("physical_identity_invariants_match"):
        for diff in identity_cmp.get("unexpected_identity_differences") or []:
            errors.append(f"physical_identity_invariant_mismatch:{diff}")
    return errors, meta


def verify_legacy_reference_lprod_target_plan(
    ref_root: Path,
    *,
    external: ExternalValidationInputPackage,
    barrier_meta: Mapping[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Legacy reference runs may retain lprod/lprod_target_plan.json as durable provenance.

    Accept only when cleanup verification is clean and content matches the authoritative package.
    """
    meta: Dict[str, Any] = {"legacy_lprod_target_plan_present": False}
    path = ref_root / "lprod" / "lprod_target_plan.json"
    if not path.is_file():
        return [], meta

    meta["legacy_lprod_target_plan_present"] = True
    forbidden = int(
        barrier_meta.get("forbidden_heavy_artifact_count")
        or barrier_meta.get("live_forbidden_heavy_artifact_count")
        or 0
    )
    shared = int(barrier_meta.get("shared_sample_artifact_count") or 0)
    if str(barrier_meta.get("barrier_status") or "") != "completed":
        return ["reference:legacy_lprod_target_plan_without_completed_cleanup_barrier"], meta
    if forbidden != 0 or shared != 0:
        return ["reference:legacy_lprod_target_plan_with_nonzero_cleanup_artifacts"], meta

    try:
        body, digest = load_target_plan_file(path)
    except Exception as exc:
        return [f"reference:legacy_lprod_target_plan_unreadable:{exc}"], meta

    errors: List[str] = []
    ext_chunk_count = external.manifest_entry.get("chunk_count") or external.manifest.get("chunk_count")
    plan_cmp = compare_target_plan_semantic(
        body,
        external.target_plan,
        left_sha256=digest,
        right_sha256=external.target_plan_sha256,
        left_manifest_chunk_count=ext_chunk_count,
        right_manifest_chunk_count=ext_chunk_count,
    )
    if not plan_cmp.get("semantic_match"):
        for diff in plan_cmp.get("differences") or []:
            errors.append(f"reference:legacy_lprod_target_plan_semantic_mismatch:{diff}")
    meta.update(
        {
            "classification": "legacy_durable_provenance",
            "sha256": digest,
            "target_count": len(body.get("targets_hz") or []),
            "target_plan_match_mode": plan_cmp.get("target_plan_match_mode"),
            "raw_sha_match": plan_cmp.get("raw_sha_match"),
            "semantic_differences": plan_cmp.get("differences") or [],
        }
    )
    return errors, meta


def verify_comparison_validation_input(
    *,
    ref_root: Path,
    cand_root: Path,
    external_package: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"validation_input_contract": "in_run_package"}
    if external_package is None:
        errors, cand_meta = verify_candidate_validation_package(cand_root)
        meta.update(cand_meta)
        return errors, meta

    external, load_errors = load_external_validation_package(external_package)
    if external is None:
        return [f"external_package:{e}" for e in load_errors], meta

    meta["validation_input_contract"] = "external_authoritative"
    meta["external_package_root"] = str(external.package_root)
    meta["authoritative_target_plan_sha256"] = external.target_plan_sha256
    meta["authoritative_target_count"] = len(external.target_plan.get("targets_hz") or [])

    sample_id = str(external.target_plan.get("sample_id") or external.manifest.get("sample_id") or ref_root.parent.parent.name)
    errors: List[str] = []
    errors.extend(_sample_ids_match_runs(sample_id=sample_id, ref_root=ref_root, cand_root=cand_root, external=external))
    identity_errors, identity_meta = _identity_matches_external_package(
        ref_root=ref_root,
        cand_root=cand_root,
        external=external,
    )
    errors.extend(identity_errors)
    meta["physical_identity"] = identity_meta.get("physical_identity") or {}

    cand_plan, cand_sha, cand_plan_errs = load_durable_target_plan(cand_root)
    meta["candidate_in_run_target_plan_available"] = cand_plan is not None
    meta["candidate_in_run_target_plan_sha256"] = cand_sha
    if cand_plan is not None:
        if cand_sha and cand_sha != external.target_plan_sha256:
            errors.append("candidate:in_run_target_plan_sha256_mismatch_vs_external")
        if (cand_plan.get("targets_hz") or []) != (external.target_plan.get("targets_hz") or []):
            errors.append("candidate:in_run_target_plan_targets_hz_mismatch_vs_external")
    else:
        if "TARGET_PLAN_UNAVAILABLE" not in cand_plan_errs and cand_plan_errs:
            errors.extend(f"candidate:{e}" for e in cand_plan_errs)

    recorded_sha = find_recorded_target_plan_sha256(cand_root)
    meta["candidate_recorded_target_plan_sha256"] = recorded_sha
    if recorded_sha and recorded_sha != external.target_plan_sha256:
        errors.append("candidate:recorded_target_plan_sha256_mismatch_vs_external")

    ref_plan, ref_sha, ref_plan_errs = load_durable_target_plan(ref_root)
    meta["reference_in_run_target_plan_available"] = ref_plan is not None
    meta["reference_in_run_target_plan_sha256"] = ref_sha
    if ref_plan is not None:
        if ref_sha and ref_sha != external.target_plan_sha256:
            errors.append("reference:in_run_target_plan_sha256_mismatch_vs_external")
        if (ref_plan.get("targets_hz") or []) != (external.target_plan.get("targets_hz") or []):
            errors.append("reference:in_run_target_plan_targets_hz_mismatch_vs_external")

    meta["authoritative_target_plan"] = external.target_plan
    return errors, meta


def verify_candidate_validation_package(cand_root: Path) -> Tuple[List[str], Dict[str, Any]]:
    """ROM candidate must have durable validation-input package (no live reference paths)."""
    errors: List[str] = []
    meta: Dict[str, Any] = {}
    plan, sha, plan_errs = load_durable_target_plan(cand_root)
    meta["durable_target_plan_available"] = plan is not None
    meta["durable_target_plan_sha256"] = sha
    if "TARGET_PLAN_UNAVAILABLE" in plan_errs:
        errors.append("candidate:TARGET_PLAN_UNAVAILABLE")
    elif plan_errs:
        errors.extend([f"candidate:{e}" for e in plan_errs])
    man_path = validation_input_manifest_path(cand_root)
    if not man_path.is_file():
        errors.append("candidate:missing_validation_input_manifest")
    else:
        try:
            man = load_json(man_path)
            entries = [r for r in (man.get("inputs") or []) if str(r.get("name")) == "target_plan"]
            if not entries:
                errors.append("candidate:validation_input_manifest_missing_target_plan")
            else:
                ent = entries[0]
                src = str(ent.get("source_path") or "")
                meta["validation_input_source_path"] = src
                if src and _norm_path_key(Path(src)) == _norm_path_key(cand_root):
                    errors.append("candidate:validation_input_source_is_candidate_run")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("candidate:validation_input_manifest_unreadable")
    return errors, meta


def _freeze_passed(run_root: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    freeze_path = run_root / "freeze" / "freeze_manifest.json"
    if not freeze_path.is_file():
        for alt in ("freeze/sample_e2e_run_manifest.json", "freeze/first_end_to_end_run_manifest.json"):
            if (run_root / alt).is_file():
                freeze_path = run_root / alt
                break
    if not freeze_path.is_file():
        return False, ["missing_freeze_manifest"]
    try:
        doc = load_json(freeze_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, ["freeze_manifest_unreadable"]
    if doc.get("production_acceptance_pass") is False:
        failures = list(doc.get("production_acceptance_failures") or [])
        if failures:
            errors.append(f"freeze_production_acceptance_failures={failures}")
    status = str(doc.get("status") or doc.get("terminal_status") or "")
    if status and status.lower() not in ("ok", "completed", "pass"):
        errors.append(f"freeze_status={status}")
    return len(errors) == 0, errors


def verify_reconciled_historical_compare_precondition(
    *,
    repo_root: Path,
    run_root: Path,
    label: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Alternate compare gate for historically misclassified runs reconciled via bookkeeping.

    Does not rewrite cleanup manifests; requires external bookkeeping report + live evidence.
    """
    errors: List[str] = []
    run_root = run_root.resolve()
    sample_id = run_root.parent.parent.name
    run_id = run_root.name
    meta: Dict[str, Any] = {
        "label": label,
        "sample_id": sample_id,
        "run_id": run_id,
        "precondition_contract": "reconciled_historical",
    }

    barrier = load_cleanup_barrier_manifest(run_root)
    if barrier is None:
        errors.append(f"{label}:missing_cleanup_barrier_manifest")
        meta["barrier_manifest_present"] = False
        return False, meta, errors
    meta["barrier_manifest_present"] = True
    meta["barrier_status"] = barrier.get("status")
    meta["sample_success"] = bool(barrier.get("sample_success"))
    if str(barrier.get("status") or "") != "completed":
        errors.append(f"{label}:cleanup_barrier_status={barrier.get('status')!r}")

    recon_doc, recon_path = find_bookkeeping_reconciliation_report(
        repo_root, sample_id=sample_id, run_id=run_id,
    )
    if recon_doc is None:
        errors.append(f"{label}:missing_bookkeeping_reconciliation_report")
    else:
        meta["bookkeeping_reconciliation"] = {
            "report_path": str(recon_path) if recon_path else None,
            "outcome": recon_doc.get("outcome"),
            "generated_utc": recon_doc.get("generated_utc"),
        }

    summary = read_run_production_summary(run_root)
    meta["terminal_status"] = summary.get("terminal_status")
    meta["aggregation_status"] = summary.get("aggregation_status")
    if str(summary.get("terminal_status") or "") != "COMPLETED":
        errors.append(f"{label}:terminal_status={summary.get('terminal_status') or 'missing'}")
    if str(summary.get("aggregation_status") or "") != AGG_PASS:
        errors.append(f"{label}:aggregation_status={summary.get('aggregation_status') or 'missing'}")
    if not is_run_usably_complete(summary):
        errors.append(f"{label}:aggregation_not_usably_complete")

    freeze_ok, freeze_errors = _freeze_passed(run_root)
    meta["freeze_passed"] = freeze_ok
    errors.extend(f"{label}:{e}" for e in freeze_errors)

    phys_path = run_root / PHYSICS_IDENTITY_MANIFEST
    if not phys_path.is_file():
        errors.append(f"{label}:missing_physics_identity_manifest")
    else:
        try:
            phys = load_json(phys_path)
            ok, man_errs = validate_physics_identity_manifest(phys)
            meta["physics_identity_valid"] = ok
            if not ok:
                errors.extend(f"{label}:physics_identity:{e}" for e in man_errs)
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append(f"{label}:physics_identity_unreadable")

    forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(run_root)
    shared_paths = [
        p for p in collect_shared_sample_artifact_paths(
            repo_root=repo_root, sample_id=sample_id, run_id=run_id,
        )
        if p.exists()
    ]
    meta["forbidden_heavy_artifact_count"] = forbidden_count
    meta["shared_sample_artifact_count"] = len(shared_paths)
    if forbidden_count != 0:
        errors.append(f"{label}:forbidden_heavy_artifact_count={forbidden_count}")
    if shared_paths:
        errors.append(f"{label}:shared_sample_artifact_count={len(shared_paths)}")

    durable_ok, durable_errors = verify_success_durable_outputs(
        run_root,
        require_compaction_manifest=False,
    )
    meta["durable_outputs_ok"] = durable_ok
    if not durable_ok:
        errors.extend(f"{label}:{e}" for e in durable_errors)

    ok = len(errors) == 0
    meta["cleanup_barrier_passed"] = ok
    return ok, meta, errors


def verify_run_compare_barrier_precondition(
    *,
    repo_root: Path,
    run_root: Path,
    label: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    ok, meta, errors = require_cleanup_barrier_passed_for_validation(
        repo_root=repo_root, run_root=run_root, label=label,
    )
    barrier = load_cleanup_barrier_manifest(run_root)
    sample_success = bool((barrier or {}).get("sample_success"))

    if ok and sample_success:
        meta["precondition_contract"] = "standard"
        return True, meta, []

    if barrier is not None and not sample_success:
        recon_ok, recon_meta, recon_errors = verify_reconciled_historical_compare_precondition(
            repo_root=repo_root, run_root=run_root, label=label,
        )
        if recon_ok:
            return True, recon_meta, []
        return False, {**meta, **recon_meta}, errors + recon_errors

    if ok and not sample_success:
        errors.append(f"{label}:sample_success_false_without_reconciliation")
    return False, meta, errors


def verify_cleanup_preconditions(
    *,
    repo_root: Path,
    ref_root: Path,
    cand_root: Path,
    validation_input_package: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    meta: Dict[str, Any] = {"reference": {}, "candidate": {}}
    for label, root in (("reference", ref_root), ("candidate", cand_root)):
        ok, barrier_meta, barrier_errors = verify_run_compare_barrier_precondition(
            repo_root=repo_root, run_root=root, label=label,
        )
        meta[label] = barrier_meta
        errors.extend(barrier_errors)
        forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(root)
        meta[label]["live_forbidden_heavy_artifact_count"] = forbidden_count
        if forbidden_count != 0:
            errors.append(f"{label}:live_forbidden_heavy_artifacts={forbidden_paths}")
        if (root / FAILURE_REPORT_REL).is_file():
            errors.append(f"{label}:cleanup_failure_report_present")

    if validation_input_package is None:
        for label, root in (("reference", ref_root), ("candidate", cand_root)):
            if (root / "lprod" / "lprod_target_plan.json").is_file():
                errors.append(f"{label}:transient_lprod_target_plan_still_present_post_cleanup")
    else:
        external, ext_errors = load_external_validation_package(validation_input_package)
        if external is None:
            errors.extend(f"external_package:{e}" for e in ext_errors)
        else:
            legacy_errors, legacy_meta = verify_legacy_reference_lprod_target_plan(
                ref_root,
                external=external,
                barrier_meta=meta.get("reference") or {},
            )
            errors.extend(legacy_errors)
            meta["reference_legacy_lprod_target_plan"] = legacy_meta

    cross = scan_candidate_references_other_run(cand_root, forbidden_root=ref_root)
    meta["candidate_reference_run_path_hits"] = cross
    if cross:
        errors.append(f"candidate:references_reference_run_paths:{len(cross)}_hits")
    val_errors, val_meta = verify_comparison_validation_input(
        ref_root=ref_root,
        cand_root=cand_root,
        external_package=validation_input_package,
    )
    meta["validation_input"] = val_meta
    errors.extend(val_errors)
    meta["cleanup_barrier_precondition_pass"] = len(errors) == 0
    return errors, meta


def resolve_reference_profile(
    ref_root: Path,
    *,
    repo_root: Path,
) -> Tuple[Optional[str], Dict[str, Any], List[str]]:
    ref_in_path = ref_root / "sample" / "sample_input.json"
    ref_in = load_json(ref_in_path) if ref_in_path.is_file() else {}
    meta: Dict[str, Any] = {}
    errors: List[str] = []
    prof = str(ref_in.get("mesh_profile") or "")
    if prof == MESH_PROFILE_REFERENCE:
        meta["reference_resolution"] = "explicit_profile"
        return MESH_PROFILE_REFERENCE, meta, []
    if prof and prof != MESH_PROFILE_REFERENCE:
        errors.append(f"reference mesh_profile={prof!r}")
        return None, meta, errors
    ok, legacy_meta, legacy_errors = evaluate_legacy_reference_compatibility(
        run_root=ref_root, repo_root=repo_root,
    )
    meta.update(legacy_meta)
    if not ok:
        errors.extend(legacy_errors)
        return None, meta, errors
    meta["reference_resolution"] = "legacy_compatibility"
    return MESH_PROFILE_REFERENCE, meta, []


def verify_physics_identity_equivalence(
    *,
    ref_root: Path,
    cand_root: Path,
    ref_legacy_meta: Mapping[str, Any],
    authoritative_target_plan: Optional[Mapping[str, Any]] = None,
    authoritative_target_plan_sha256: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    meta: Dict[str, Any] = {"reference": {}, "candidate": {}, "allowed_differences": []}
    ref_in = load_json(ref_root / "sample" / "sample_input.json") if (ref_root / "sample" / "sample_input.json").is_file() else {}
    cand_in = load_json(cand_root / "sample" / "sample_input.json") if (cand_root / "sample" / "sample_input.json").is_file() else {}
    sid_r = str(ref_in.get("sample_id") or ref_root.parent.parent.name)
    sid_c = str(cand_in.get("sample_id") or cand_root.parent.parent.name)
    if sid_r != sid_c:
        errors.append(f"sample_id mismatch: {sid_r} vs {sid_c}")

    geom_r = extract_geometry_dict(ref_in)
    geom_c = extract_geometry_dict(cand_in)
    if geometry_fingerprint(geom_r) != geometry_fingerprint(geom_c):
        errors.append("geometry_fingerprint mismatch")

    ref_id = _physics_identity(ref_root)
    cand_id = _physics_identity(cand_root)
    ref_pm = _pipeline_manifest(ref_root)
    cand_pm = _pipeline_manifest(cand_root)

    for key in ("top_wood_id", "back_wood_id"):
        if ref_in.get(key) != cand_in.get(key):
            errors.append(f"material_mismatch:{key}")

    ref_fp = ref_pm.get("frequency_policy") or {}
    cand_fp = cand_pm.get("frequency_policy") or {}
    ref_band = ref_fp.get("band_hz") or [60.0, 550.0]
    cand_band = cand_fp.get("band_hz") or [60.0, 550.0]
    if list(ref_band) != list(cand_band):
        errors.append(f"frequency_range_mismatch:{ref_band}!={cand_band}")

    if str(cand_in.get("mesh_profile") or "") != MESH_PROFILE_ROM:
        errors.append(f"candidate mesh_profile={cand_in.get('mesh_profile')!r}")
    if str(cand_id.get("mesh_level_id") or cand_id.get("mesh_level") or "") not in (LEVEL_ROM_PROD, ""):
        if str(cand_id.get("mesh_level_id") or "") != LEVEL_ROM_PROD:
            errors.append(f"candidate mesh_level_id={cand_id.get('mesh_level_id')!r}")

    ref_plan, ref_sha, ref_plan_errs = load_durable_target_plan(ref_root)
    cand_plan, cand_sha, cand_plan_errs = load_durable_target_plan(cand_root)
    meta["target_plan"] = {
        "reference_sha256": ref_sha,
        "candidate_sha256": cand_sha,
        "reference_available": ref_plan is not None,
        "candidate_available": cand_plan is not None,
        "authoritative_sha256": authoritative_target_plan_sha256,
    }
    if authoritative_target_plan is not None and authoritative_target_plan_sha256:
        auth_targets = list(authoritative_target_plan.get("targets_hz") or [])
        meta["target_plan"]["authoritative_target_count"] = len(auth_targets)
        for label, plan, sha in (
            ("reference", ref_plan, ref_sha),
            ("candidate", cand_plan, cand_sha),
        ):
            if plan is not None:
                if sha and sha != authoritative_target_plan_sha256:
                    errors.append(f"{label}:target_plan_sha256_mismatch_vs_authoritative")
                if list(plan.get("targets_hz") or []) != auth_targets:
                    errors.append(f"{label}:target_plan_targets_hz_mismatch_vs_authoritative")
        if ref_plan is None and (ref_root / "lprod" / "lprod_target_plan.json").is_file():
            try:
                legacy_body, legacy_sha = load_target_plan_file(ref_root / "lprod" / "lprod_target_plan.json")
                legacy_cmp = compare_target_plan_semantic(
                    legacy_body,
                    authoritative_target_plan,
                    left_sha256=legacy_sha,
                    right_sha256=authoritative_target_plan_sha256,
                )
                meta.setdefault("legacy_lprod_target_plan", {}).update(legacy_cmp)
                if not legacy_cmp.get("semantic_match"):
                    for diff in legacy_cmp.get("differences") or []:
                        errors.append(f"reference:legacy_lprod_target_plan_semantic_mismatch_vs_authoritative:{diff}")
            except Exception:
                errors.append("reference:legacy_lprod_target_plan_unreadable")
    elif "TARGET_PLAN_UNAVAILABLE" in ref_plan_errs or "TARGET_PLAN_UNAVAILABLE" in cand_plan_errs:
        errors.append("TARGET_PLAN_UNAVAILABLE")
    elif ref_plan_errs or cand_plan_errs:
        errors.extend(ref_plan_errs + cand_plan_errs)
    elif ref_plan and cand_plan:
        if (ref_plan.get("targets_hz") or []) != (cand_plan.get("targets_hz") or []):
            errors.append("durable_target_plan.targets_hz_mismatch")
        if ref_sha and cand_sha and ref_sha != cand_sha:
            errors.append("durable_target_plan_sha256_mismatch")
    meta["legacy_reference"] = ref_legacy_meta
    return errors, meta


def _band_error_stats(
    matched: Sequence[Mapping[str, Any]],
    lo: float,
    hi: float,
) -> Dict[str, Any]:
    sel = [p for p in matched if lo <= float(p["reference_hz"]) < hi]
    if not sel:
        return {"count": 0, "matched_count": 0}
    rel = [float(p["rel_error"]) for p in sel]
    abs_e = [float(p["abs_error_hz"]) for p in sel]
    arr = np.asarray(rel, dtype=float)
    return {
        "count": len(sel),
        "matched_count": len(sel),
        "median_rel_error": float(np.median(arr)),
        "p95_rel_error": float(np.percentile(arr, 95)),
        "max_rel_error": float(np.max(arr)),
        "median_abs_error_hz": float(np.median(abs_e)),
        "max_abs_error_hz": float(np.max(abs_e)),
    }


def _recall_in_band(ref_rows: Sequence[Mapping[str, Any]], matched_ref_idx: set[int], lo: float, hi: float) -> Dict[str, Any]:
    in_band = [i for i, r in enumerate(ref_rows) if lo <= float(r.get("frequency_hz") or 0) < hi]
    if not in_band:
        return {"reference_count": 0, "matched_count": 0, "recall": None}
    matched = sum(1 for i in in_band if i in matched_ref_idx)
    recall = matched / len(in_band)
    return {"reference_count": len(in_band), "matched_count": matched, "recall": recall}


def _top10_analysis(key: str, ref_rows: Sequence[Mapping[str, Any]], cand_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ref_top = sorted(ref_rows, key=lambda r: float(r.get(key) or 0.0), reverse=True)[:10]
    cand_top = sorted(cand_rows, key=lambda r: float(r.get(key) or 0.0), reverse=True)[:10]
    ref_ids = [_mode_id(r) for r in ref_top]
    cand_ids = [_mode_id(r) for r in cand_top]
    overlap = len(set(ref_ids) & set(cand_ids))
    rank_disp = 0
    for rid in ref_ids:
        if rid in cand_ids and ref_ids.index(rid) == cand_ids.index(rid):
            rank_disp += 1
    return {
        "overlap_count": overlap,
        "ranking_displacement_matches": rank_disp,
        "reference_top10_ids": ref_ids,
        "candidate_top10_ids": cand_ids,
    }


def _coupling_agreement(ref_rows: Sequence[Mapping[str, Any]], cand_rows: Sequence[Mapping[str, Any]], pairs: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    agree = 0
    total = 0
    dom_agree = 0
    for i, j in pairs:
        rc = ref_rows[i]
        cc = cand_rows[j]
        r_coupling = str(rc.get("coupling_class") or "unknown")
        c_coupling = str(cc.get("coupling_class") or "unknown")
        if r_coupling != "unknown" and c_coupling != "unknown":
            total += 1
            if r_coupling == c_coupling:
                agree += 1
        r_dom = str(rc.get("dominant_region") or "")
        c_dom = str(cc.get("dominant_region") or "")
        if r_dom and c_dom and r_dom == c_dom:
            dom_agree += 1
    return {
        "coupling_class_agreement": (agree / total) if total else None,
        "coupling_pairs_compared": total,
        "dominant_region_agreement_fraction": (dom_agree / len(pairs)) if pairs else None,
    }


def _mac_status(ref_rows: Sequence[Mapping[str, Any]], cand_rows: Sequence[Mapping[str, Any]], pairs: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    has_mac = any("mac" in r or "eigenvector_fingerprint" in r for r in ref_rows)
    if not has_mac:
        return {"MAC_STATUS": "UNAVAILABLE", "reason": "no_durable_eigenvector_data_post_cleanup"}
    macs: List[float] = []
    for i, j in pairs:
        m = ref_rows[i].get("mac") or cand_rows[j].get("mac")
        if m is not None:
            macs.append(float(m))
    if not macs:
        return {"MAC_STATUS": "UNAVAILABLE", "reason": "mac_field_empty"}
    arr = np.asarray(macs, dtype=float)
    below_300 = [m for (i, _), m in zip(pairs, macs) if float(ref_rows[i].get("frequency_hz") or 0) < 300]
    return {
        "MAC_STATUS": "AVAILABLE",
        "median_mac": float(np.median(arr)),
        "median_mac_below_300_hz": float(np.median(below_300)) if below_300 else None,
    }


def _count_catalog_rows(run_root: Path, rel: str) -> int:
    path = run_root / rel
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _spearman_rank_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    rx = np.argsort(np.argsort(arr_x))
    ry = np.argsort(np.argsort(arr_y))
    corr = np.corrcoef(rx, ry)
    return float(corr[0, 1]) if corr.shape == (2, 2) else None


def _matched_proxy_analysis(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Tuple[int, int]],
    *,
    key: str,
    signed: bool = False,
) -> Dict[str, Any]:
    ref_vals: List[float] = []
    cand_vals: List[float] = []
    ratios: List[float] = []
    signed_diffs: List[float] = []
    outliers: List[Dict[str, Any]] = []
    for i, j in pairs:
        rv = float(ref_rows[i].get(key) or 0.0)
        cv = float(cand_rows[j].get(key) or 0.0)
        ref_vals.append(rv)
        cand_vals.append(cv)
        if rv != 0.0:
            ratios.append(cv / rv)
        if signed:
            signed_diffs.append(cv - rv)
        if rv != 0.0 and cv / rv < 0.5:
            outliers.append(
                {
                    "reference_hz": ref_rows[i].get("frequency_hz"),
                    "candidate_hz": cand_rows[j].get("frequency_hz"),
                    "reference_value": rv,
                    "candidate_value": cv,
                    "ratio": cv / rv if rv else None,
                    "mode_id": _mode_id(ref_rows[i]),
                }
            )

    abs_diffs = [abs(a - b) for a, b in zip(ref_vals, cand_vals)]
    ratio_arr = np.asarray(ratios, dtype=float) if ratios else np.asarray([], dtype=float)
    return {
        "matched_count": len(pairs),
        "median_abs_difference": float(np.median(abs_diffs)) if abs_diffs else None,
        "p95_abs_difference": float(np.percentile(abs_diffs, 95)) if abs_diffs else None,
        "median_signed_difference": float(np.median(signed_diffs)) if signed_diffs else None,
        "median_amplitude_ratio": float(np.median(ratio_arr)) if ratios else None,
        "p95_amplitude_ratio": float(np.percentile(ratio_arr, 95)) if ratios else None,
        "rank_correlation": _spearman_rank_corr(ref_vals, cand_vals),
        "top10": _top10_analysis(key, ref_rows, cand_rows),
        "large_negative_rom_outliers": sorted(
            outliers, key=lambda r: float(r.get("ratio") or 0.0),
        )[:10],
        "systematic_rom_larger_than_reference": (
            float(np.median(ratio_arr)) > 1.1 if ratios else None
        ),
    }


def _share_difference_stats(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for share in ("top_share", "back_share", "air_share"):
        diffs = [
            abs(float(cand_rows[j].get(share) or 0.0) - float(ref_rows[i].get(share) or 0.0))
            for i, j in pairs
        ]
        signed = [
            float(cand_rows[j].get(share) or 0.0) - float(ref_rows[i].get(share) or 0.0)
            for i, j in pairs
        ]
        out[share] = {
            "median_abs_difference": float(np.median(diffs)) if diffs else None,
            "p95_abs_difference": float(np.percentile(diffs, 95)) if diffs else None,
            "median_signed_difference": float(np.median(signed)) if signed else None,
        }
    air_signed = [
        float(cand_rows[j].get("air_share") or 0.0) - float(ref_rows[i].get("air_share") or 0.0)
        for i, j in pairs
    ]
    out["systematic_air_participation_shift"] = (
        float(np.median(air_signed)) if air_signed else None
    )
    return out


def _performance_metrics(ref_rt: Mapping[str, Any], cand_rt: Mapping[str, Any]) -> Dict[str, Any]:
    stages = ("stage1_scout", "stage2_lprod_checkpoint", "stage5_workers")
    ref_stage = ref_rt.get("stage_wall_times_s") or {}
    cand_stage = cand_rt.get("stage_wall_times_s") or {}
    ref_wall = float(ref_stage.get("stage5_workers") or ref_rt.get("workers_wall_time_s") or 0)
    cand_wall = float(cand_stage.get("stage5_workers") or cand_rt.get("workers_wall_time_s") or 0)
    ref_total = float(ref_rt.get("total_wall_time_s") or ref_rt.get("elapsed_s") or 0)
    cand_total = float(cand_rt.get("total_wall_time_s") or cand_rt.get("elapsed_s") or 0)
    if ref_total <= 0:
        ref_total = sum(float(ref_stage.get(s) or 0) for s in stages)
    if cand_total <= 0:
        cand_total = sum(float(cand_stage.get(s) or 0) for s in stages)
    runtime_reduction = (ref_wall - cand_wall) / ref_wall if ref_wall > 0 else None
    total_reduction = (ref_total - cand_total) / ref_total if ref_total > 0 else None
    worker_records = list(cand_rt.get("worker_resource_records") or [])
    ref_worker_records = list(ref_rt.get("worker_resource_records") or [])
    peaks = [
        int(r.get("peak_rss_bytes") or r.get("max_rss_bytes"))
        for r in worker_records
        if r.get("peak_rss_bytes") or r.get("max_rss_bytes")
    ]
    ref_peaks = [
        int(r.get("peak_rss_bytes") or r.get("max_rss_bytes"))
        for r in ref_worker_records
        if r.get("peak_rss_bytes") or r.get("max_rss_bytes")
    ]
    peak_max = max(peaks) if peaks else cand_rt.get("peak_rss_bytes_max_worker")
    ref_peak_max = max(ref_peaks) if ref_peaks else ref_rt.get("peak_rss_bytes_max_worker")
    ram_reduction = None
    if ref_peak_max and peak_max:
        ram_reduction = (float(ref_peak_max) - float(peak_max)) / float(ref_peak_max)
    return {
        "reference_total_wall_s": ref_total or None,
        "candidate_total_wall_s": cand_total or None,
        "total_runtime_reduction_fraction": total_reduction,
        "reference_scout_wall_s": float(ref_stage.get("stage1_scout") or 0) or None,
        "candidate_scout_wall_s": float(cand_stage.get("stage1_scout") or 0) or None,
        "reference_checkpoint_wall_s": float(ref_stage.get("stage2_lprod_checkpoint") or 0) or None,
        "candidate_checkpoint_wall_s": float(cand_stage.get("stage2_lprod_checkpoint") or 0) or None,
        "reference_worker_wall_s": ref_wall or None,
        "candidate_worker_wall_s": cand_wall or None,
        "runtime_reduction_fraction": runtime_reduction,
        "ram_reduction_fraction": ram_reduction,
        "reference_peak_rss_bytes_max_worker": ref_peak_max,
        "candidate_peak_rss_bytes_max_worker": peak_max,
        "candidate_sum_of_individual_worker_peaks_upper_bound": sum(peaks) if peaks else None,
        "candidate_workers_parallel_observed": cand_rt.get("workers_actual_parallel") or cand_rt.get("workers_parallel_observed"),
        "reference_workers_parallel_observed": ref_rt.get("workers_actual_parallel") or ref_rt.get("workers_parallel_observed"),
        "rss_measurement_note": cand_rt.get("rss_aggregate_note"),
    }


def _mesh_scale_metrics(ref_id: Mapping[str, Any], cand_id: Mapping[str, Any]) -> Dict[str, Any]:
    ref_nodes = ref_id.get("mesh_node_count") or ref_id.get("n_nodes")
    cand_nodes = cand_id.get("mesh_node_count") or cand_id.get("n_nodes")
    ref_tetra = ref_id.get("mesh_tetra_count") or ref_id.get("n_tetra")
    cand_tetra = cand_id.get("mesh_tetra_count") or cand_id.get("n_tetra")
    node_reduction = None
    tetra_reduction = None
    if ref_nodes and cand_nodes:
        node_reduction = (float(ref_nodes) - float(cand_nodes)) / float(ref_nodes)
    if ref_tetra and cand_tetra:
        tetra_reduction = (float(ref_tetra) - float(cand_tetra)) / float(ref_tetra)
    return {
        "reference": {
            "nodes": ref_nodes,
            "tetrahedra": ref_tetra,
            "active_dimension": ref_id.get("active_dimension"),
        },
        "candidate": {
            "nodes": cand_nodes,
            "tetrahedra": cand_tetra,
            "active_dimension": cand_id.get("active_dimension"),
        },
        "node_count_reduction_fraction": node_reduction,
        "tetra_count_reduction_fraction": tetra_reduction,
    }


def derive_comparison_recommendation(
    report: Mapping[str, Any],
    acceptance_evaluation: Mapping[str, Any],
) -> Dict[str, Any]:
    if not report.get("comparison_executed"):
        return {"recommendation": RECOMMEND_INCOMPLETE, "reason": "comparison_not_executed"}
    if report.get("status") == "INCOMPLETE":
        return {"recommendation": RECOMMEND_INCOMPLETE, "reason": "mandatory_metrics_missing"}

    ae = dict(acceptance_evaluation or {})
    hard_fails = [k for k, v in ae.items() if k.endswith("_pass") and v is False]
    proxy = report.get("proxy_comparison") or {}
    scale_warning = bool(proxy.get("normalization_scale_warning"))
    rank_preserved = all(
        (proxy.get(k) or {}).get("rank_correlation") is None
        or float((proxy.get(k) or {}).get("rank_correlation") or 0) >= 0.85
        for k in ("bridge", "mic", "radiation")
    )

    freq_pass = ae.get("global_median_rel_freq_error_pass") and ae.get("global_p95_rel_freq_error_pass")
    family_pass = ae.get("mode_family_survival_pass")
    recall_pass = ae.get("recall_below_350_pass") and ae.get("recall_350_550_pass")
    coupling_pass = ae.get("coupling_class_agreement_pass")

    if hard_fails and not (scale_warning and rank_preserved and freq_pass and family_pass):
        if any(k.startswith(("global_", "recall_", "mode_family", "intrinsic")) for k in hard_fails):
            return {
                "recommendation": RECOMMEND_REJECT_OR_TIGHTEN_MESH,
                "reason": "frequency_or_family_gate_failed",
                "failed_checks": hard_fails,
            }
    if scale_warning and rank_preserved and freq_pass and family_pass and recall_pass:
        return {
            "recommendation": RECOMMEND_ACCEPT_WITH_CAUTION,
            "reason": "proxy_scale_differs_rank_and_frequency_preserved",
            "failed_checks": hard_fails,
        }
    if not hard_fails and freq_pass and coupling_pass and recall_pass and family_pass:
        return {"recommendation": RECOMMEND_ACCEPT_ROM_BALANCED, "reason": "all_mandatory_gates_pass"}
    if hard_fails:
        return {
            "recommendation": RECOMMEND_ACCEPT_WITH_CAUTION if rank_preserved else RECOMMEND_REJECT_OR_TIGHTEN_MESH,
            "reason": "partial_gate_failure",
            "failed_checks": hard_fails,
        }
    return {"recommendation": RECOMMEND_ACCEPT_WITH_CAUTION, "reason": "mixed_signals"}


def evaluate_acceptance(report: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
    """Returns (evaluation, acceptance_pass, incomplete)."""
    ae: Dict[str, Any] = {}
    incomplete = False
    mandatory_missing: List[str] = []

    freq = report.get("frequencies") or {}
    if freq.get("global_median_rel_error") is None:
        mandatory_missing.append("global_median_rel_error")
    else:
        ae["global_median_rel_freq_error_pass"] = (
            float(freq["global_median_rel_error"]) <= ACCEPTANCE_THRESHOLDS["global_median_rel_freq_error_max"]
        )
    if freq.get("global_p95_rel_error") is None:
        mandatory_missing.append("global_p95_rel_error")
    else:
        ae["global_p95_rel_freq_error_pass"] = (
            float(freq["global_p95_rel_error"]) <= ACCEPTANCE_THRESHOLDS["global_p95_rel_freq_error_max"]
        )

    bands = freq.get("bands") or {}
    b150 = bands.get("150_350") or {}
    b550 = bands.get("350_550") or {}
    b60 = bands.get("60_150") or {}
    if b60.get("max_rel_error") is not None:
        ae["band_60_150_max_each_pass"] = float(b60["max_rel_error"]) <= ACCEPTANCE_THRESHOLDS["band_60_150_max_each"]
    elif b60.get("matched_count"):
        mandatory_missing.append("band_60_150_stats")
    if b150.get("median_rel_error") is not None:
        ae["band_150_350_median_pass"] = float(b150["median_rel_error"]) <= ACCEPTANCE_THRESHOLDS["band_150_350_median_max"]
        if b150.get("max_rel_error") is not None:
            ae["band_150_350_max_pass"] = float(b150["max_rel_error"]) <= ACCEPTANCE_THRESHOLDS["band_150_350_max_max"]
        else:
            mandatory_missing.append("band_150_350_max")
    else:
        mandatory_missing.append("band_150_350_stats")
    if b550.get("median_rel_error") is not None:
        ae["band_350_550_median_pass"] = float(b550["median_rel_error"]) <= ACCEPTANCE_THRESHOLDS["band_350_550_median_max"]
        if b550.get("max_rel_error") is not None:
            ae["band_350_550_max_pass"] = float(b550["max_rel_error"]) <= ACCEPTANCE_THRESHOLDS["band_350_550_max_max"]
        else:
            mandatory_missing.append("band_350_550_max")
    else:
        mandatory_missing.append("band_350_550_stats")

    recall = report.get("modal_retention") or {}
    for key, thresh_key in (("recall_below_350", "recall_below_350_min"), ("recall_350_550", "recall_350_550_min")):
        val = recall.get(key, {}).get("recall")
        if val is None:
            mandatory_missing.append(key)
        else:
            ae[f"{key}_pass"] = float(val) >= ACCEPTANCE_THRESHOLDS[thresh_key]

    coupling = report.get("coupling_output") or {}
    cagr = coupling.get("coupling_class_agreement")
    if cagr is None:
        mandatory_missing.append("coupling_class_agreement")
    else:
        ae["coupling_class_agreement_pass"] = float(cagr) >= ACCEPTANCE_THRESHOLDS["coupling_class_agreement_min"]

    bridge = coupling.get("bridge_top10") or {}
    mic = coupling.get("mic_top10") or {}
    if bridge.get("overlap_count") is None:
        mandatory_missing.append("bridge_top10")
    else:
        ae["bridge_top10_pass"] = int(bridge["overlap_count"]) >= ACCEPTANCE_THRESHOLDS["bridge_top10_overlap_min"]
    if mic.get("overlap_count") is None:
        mandatory_missing.append("mic_top10")
    else:
        ae["mic_top10_pass"] = int(mic["overlap_count"]) >= ACCEPTANCE_THRESHOLDS["mic_top10_overlap_min"]

    perf = report.get("performance") or {}
    if perf.get("runtime_reduction_fraction") is not None:
        ae["runtime_reduction_pass"] = float(perf["runtime_reduction_fraction"]) >= ACCEPTANCE_THRESHOLDS["runtime_reduction_min"]
    else:
        mandatory_missing.append("runtime_reduction")
    if perf.get("candidate_peak_rss_bytes_max_worker"):
        ae["peak_rss_pass"] = (
            float(perf["candidate_peak_rss_bytes_max_worker"]) / (1024**3)
            <= ACCEPTANCE_THRESHOLDS["peak_rss_per_worker_gib_max"]
        )
    else:
        mandatory_missing.append("peak_rss")

    intrinsic = report.get("intrinsic_coverage") or {}
    if intrinsic.get("intrinsic_band_third_no_loss_pass") is None:
        mandatory_missing.append("intrinsic_band_third_coverage")
    else:
        ae["intrinsic_band_third_no_loss_pass"] = bool(intrinsic["intrinsic_band_third_no_loss_pass"])

    families = report.get("mode_family_survival") or {}
    if families.get("family_survival_pass") is None:
        mandatory_missing.append("mode_family_survival")
    else:
        ae["mode_family_survival_pass"] = bool(families["family_survival_pass"])

    mac = report.get("mac") or {}
    ae["mac_advisory_only"] = mac.get("MAC_STATUS") == "UNAVAILABLE"

    if mandatory_missing:
        incomplete = True
        ae["mandatory_metrics_missing"] = mandatory_missing

    acceptance_pass = (not incomplete) and all(v is True for k, v in ae.items() if k.endswith("_pass"))
    return ae, acceptance_pass, incomplete


def compare_runs(
    *,
    reference_run: Path,
    candidate_run: Path,
    repo_root: Path,
    validation_input_package: Optional[Path] = None,
) -> Dict[str, Any]:
    ref_root = reference_run.resolve()
    cand_root = candidate_run.resolve()

    cleanup_errors, cleanup_meta = verify_cleanup_preconditions(
        repo_root=repo_root,
        ref_root=ref_root,
        cand_root=cand_root,
        validation_input_package=validation_input_package,
    )
    if cleanup_errors:
        return {
            "schema": "m4_mesh_profile_compare_v2",
            "status": "PRECONDITION_FAILED",
            "comparison_executed": False,
            "cleanup_barrier_precondition_pass": False,
            "precondition_errors": cleanup_errors,
            "cleanup_barrier": cleanup_meta,
            "acceptance_pass": False,
            "exit_code": EXIT_PRECONDITION_FAIL,
        }

    ref_prof, ref_legacy_meta, ref_prof_errors = resolve_reference_profile(ref_root, repo_root=repo_root)
    if ref_prof_errors:
        return {
            "schema": "m4_mesh_profile_compare_v2",
            "status": "PRECONDITION_FAILED",
            "comparison_executed": False,
            "cleanup_barrier_precondition_pass": True,
            "precondition_errors": ref_prof_errors,
            "cleanup_barrier": cleanup_meta,
            "reference_profile_resolution": ref_legacy_meta,
            "acceptance_pass": False,
            "exit_code": EXIT_PRECONDITION_FAIL,
        }

    val_meta = cleanup_meta.get("validation_input") or {}
    auth_plan = val_meta.get("authoritative_target_plan")
    auth_sha = val_meta.get("authoritative_target_plan_sha256")
    phys_errors, phys_meta = verify_physics_identity_equivalence(
        ref_root=ref_root,
        cand_root=cand_root,
        ref_legacy_meta=ref_legacy_meta,
        authoritative_target_plan=auth_plan,
        authoritative_target_plan_sha256=auth_sha,
    )
    if phys_errors:
        return {
            "schema": "m4_mesh_profile_compare_v2",
            "status": "PRECONDITION_FAILED",
            "comparison_executed": False,
            "cleanup_barrier_precondition_pass": True,
            "precondition_errors": phys_errors,
            "physics_identity": phys_meta,
            "cleanup_barrier": cleanup_meta,
            "acceptance_pass": False,
            "exit_code": EXIT_PRECONDITION_FAIL,
        }

    ref_rows = load_catalog(ref_root)
    cand_rows = load_catalog(cand_root)
    ref_raw_count = _count_catalog_rows(ref_root, "aggregation/modes_catalog.jsonl")
    cand_raw_count = _count_catalog_rows(cand_root, "aggregation/modes_catalog.jsonl")
    if not ref_rows or not cand_rows:
        return {
            "schema": "m4_mesh_profile_compare_v2",
            "status": "INCOMPLETE",
            "comparison_executed": False,
            "cleanup_barrier_precondition_pass": True,
            "precondition_errors": ["missing_modes_catalog"],
            "acceptance_pass": False,
            "exit_code": EXIT_INCOMPLETE,
        }

    pairs = _greedy_freq_pairs(ref_rows, cand_rows)
    matched: List[Dict[str, Any]] = []
    rel_errors: List[float] = []
    for i, j in pairs:
        rf = float(ref_rows[i].get("frequency_hz") or 0)
        cf = float(cand_rows[j].get("frequency_hz") or 0)
        if rf > 0:
            abs_e = abs(cf - rf)
            rel_e = abs_e / rf
            rel_errors.append(rel_e)
            matched.append({
                "reference_hz": rf,
                "candidate_hz": cf,
                "abs_error_hz": abs_e,
                "rel_error": rel_e,
                "reference_coupling_class": ref_rows[i].get("coupling_class"),
                "candidate_coupling_class": cand_rows[j].get("coupling_class"),
            })

    matched_ref_idx = {i for i, _ in pairs}
    unmatched_ref = [ref_rows[i] for i in range(len(ref_rows)) if i not in matched_ref_idx]
    unmatched_cand_idx = {j for _, j in pairs}
    unmatched_cand = [cand_rows[j] for j in range(len(cand_rows)) if j not in unmatched_cand_idx]

    band_stats = {name: _band_error_stats(matched, lo, hi) for name, lo, hi in FREQUENCY_BANDS}
    band_stats["150_350"] = _band_error_stats(matched, 150.0, 350.0)
    band_stats["350_550"] = _band_error_stats(matched, 350.0, 550.0)
    recall_stats = {
        name: _recall_in_band(ref_rows, matched_ref_idx, lo, hi) for name, lo, hi, _ in RECALL_BANDS
    }
    recall_below_350 = _recall_in_band(ref_rows, matched_ref_idx, 60.0, 350.0)
    recall_350_550 = _recall_in_band(ref_rows, matched_ref_idx, 350.0, 550.0)

    ref_id = _physics_identity(ref_root)
    cand_id = _physics_identity(cand_root)
    coupling = _coupling_agreement(ref_rows, cand_rows, pairs)
    coupling["bridge_top10"] = _top10_analysis("bridge_excitation_abs", ref_rows, cand_rows)
    coupling["mic_top10"] = _top10_analysis("mic_output_proxy", ref_rows, cand_rows)
    coupling["radiation_top10"] = _top10_analysis("radiation_proxy", ref_rows, cand_rows)
    share_diffs = _share_difference_stats(ref_rows, cand_rows, pairs)
    bridge_proxy = _matched_proxy_analysis(
        ref_rows, cand_rows, pairs, key="bridge_excitation_abs", signed=True,
    )
    mic_proxy = _matched_proxy_analysis(ref_rows, cand_rows, pairs, key="mic_output_proxy")
    radiation_proxy = _matched_proxy_analysis(ref_rows, cand_rows, pairs, key="radiation_proxy")
    bridge_signed = _matched_proxy_analysis(
        ref_rows, cand_rows, pairs, key="bridge_excitation_coupling", signed=True,
    )
    proxy_scale_warning = False
    for proxy_doc in (bridge_proxy, mic_proxy, radiation_proxy):
        ratio = proxy_doc.get("median_amplitude_ratio")
        rank = proxy_doc.get("rank_correlation")
        if ratio is not None and (ratio < 0.7 or ratio > 1.4) and (rank is None or rank >= 0.85):
            proxy_scale_warning = True
    mac = _mac_status(ref_rows, cand_rows, pairs)
    ref_scout, ref_scout_errs = load_durable_scout_intrinsic_summary(ref_root)
    cand_scout, cand_scout_errs = load_durable_scout_intrinsic_summary(cand_root)
    intrinsic_cov = compare_intrinsic_band_third_coverage(
        ref_rows, cand_rows, ref_scout=ref_scout, cand_scout=cand_scout,
    )
    family_survival = compare_mode_family_survival(ref_rows, cand_rows)
    scout_provenance_errors = list(ref_scout_errs + cand_scout_errs)

    report: Dict[str, Any] = {
        "schema": "m4_mesh_profile_compare_v2",
        "status": "COMPARED",
        "comparison_executed": True,
        "cleanup_barrier_precondition_pass": True,
        "cleanup_barrier": cleanup_meta,
        "reference_profile_resolution": ref_legacy_meta,
        "physics_identity": phys_meta,
        "precondition_errors": [],
        "mesh_operator_scale": {
            "reference": {
                "mesh_level_id": ref_id.get("mesh_level_id") or LEVEL_PROD_REFERENCE,
                "effective_controls_m": ref_id.get("effective_controls_m") or REFERENCE_CONTROLS_M,
                "active_dimension": ref_id.get("active_dimension"),
                "generated_mesh_sha256": ref_id.get("generated_mesh_sha256"),
                "operator_mesh_sha256": ref_id.get("operator_mesh_sha256"),
            },
            "candidate": {
                "mesh_level_id": cand_id.get("mesh_level_id") or LEVEL_ROM_PROD,
                "effective_controls_m": cand_id.get("effective_controls_m") or ROM_CONTROLS_M,
                "active_dimension": cand_id.get("active_dimension"),
                "generated_mesh_sha256": cand_id.get("generated_mesh_sha256"),
                "operator_mesh_sha256": cand_id.get("operator_mesh_sha256"),
            },
        },
        "modal_retention": {
            "reference_raw_mode_count": ref_raw_count,
            "candidate_raw_mode_count": cand_raw_count,
            "reference_deduped_mode_count": len(ref_rows),
            "candidate_deduped_mode_count": len(cand_rows),
            "matched_mode_count": len(pairs),
            "unmatched_reference_mode_count": len(unmatched_ref),
            "unmatched_candidate_mode_count": len(unmatched_cand),
            "unmatched_reference_modes": [{"frequency_hz": r.get("frequency_hz"), "mode_id": _mode_id(r)} for r in unmatched_ref],
            "unmatched_candidate_modes": [{"frequency_hz": r.get("frequency_hz"), "mode_id": _mode_id(r)} for r in unmatched_cand],
            "recall_below_350": recall_below_350,
            "recall_350_550": recall_350_550,
            "recall_by_band": recall_stats,
        },
        "frequencies": {
            "matched_pair_count": len(matched),
            "global_median_rel_error": float(np.median(rel_errors)) if rel_errors else None,
            "global_p95_rel_error": float(np.percentile(rel_errors, 95)) if rel_errors else None,
            "global_max_rel_error": float(np.max(rel_errors)) if rel_errors else None,
            "global_median_abs_error_hz": float(np.median([m["abs_error_hz"] for m in matched])) if matched else None,
            "global_max_abs_error_hz": float(np.max([m["abs_error_hz"] for m in matched])) if matched else None,
            "bands": band_stats,
        },
        "coupling_output": coupling,
        "participation_shares": share_diffs,
        "proxy_comparison": {
            "normalization_scale_warning": proxy_scale_warning,
            "bridge": bridge_proxy,
            "bridge_signed_coupling": bridge_signed,
            "mic": mic_proxy,
            "radiation": radiation_proxy,
        },
        "intrinsic_coverage": intrinsic_cov,
        "mode_family_survival": family_survival,
        "scout_provenance": {
            "reference": ref_scout,
            "candidate": cand_scout,
            "errors": scout_provenance_errors,
        },
        "mac": mac,
        "performance": _performance_metrics(_runtime_prov(ref_root), _runtime_prov(cand_root)),
        "mesh_scale": _mesh_scale_metrics(ref_id, cand_id),
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
    }

    ae, acceptance_pass, incomplete = evaluate_acceptance(report)
    report["acceptance_evaluation"] = ae
    report["acceptance_pass"] = acceptance_pass
    report["recommendation"] = derive_comparison_recommendation(report, ae)
    if incomplete:
        report["status"] = "INCOMPLETE"
        report["comparison_executed"] = True
        report["exit_code"] = EXIT_INCOMPLETE
    elif acceptance_pass:
        report["exit_code"] = EXIT_PASS
    else:
        report["status"] = "ACCEPTANCE_FAILED"
        report["exit_code"] = EXIT_ACCEPTANCE_FAIL
    return report


def compare_exit_code(report: Mapping[str, Any]) -> int:
    ec = report.get("exit_code")
    if ec is None:
        return EXIT_PRECONDITION_FAIL
    return int(ec)
