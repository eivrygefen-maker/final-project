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
    LEVEL_PROD_REFERENCE,
    LEVEL_ROM_PROD,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    REFERENCE_CONTROLS_M,
    ROM_CONTROLS_M,
    VALIDATION_INPUT_PACKAGE_REL,
    evaluate_legacy_reference_compatibility,
    load_durable_target_plan,
    sha256_file,
    validation_input_manifest_path,
)
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    PHYSICS_IDENTITY_MANIFEST,
    count_forbidden_heavy_artifacts,
    iter_physics_scan_files,
    iter_path_like_strings_from_json,
)
from v2_b3_m4_sample_cleanup_barrier import (  # noqa: E402
    FAILURE_REPORT_REL,
    require_cleanup_barrier_passed_for_validation,
)
from v2_b3_m4_mesh_profile_provenance_lib import (  # noqa: E402
    compare_intrinsic_band_third_coverage,
    compare_mode_family_survival,
    load_durable_scout_intrinsic_summary,
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


def verify_cleanup_preconditions(
    *,
    repo_root: Path,
    ref_root: Path,
    cand_root: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    meta: Dict[str, Any] = {"reference": {}, "candidate": {}}
    for label, root in (("reference", ref_root), ("candidate", cand_root)):
        ok, barrier_meta, barrier_errors = require_cleanup_barrier_passed_for_validation(
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
        if (root / "lprod" / "lprod_target_plan.json").is_file():
            errors.append(f"{label}:transient_lprod_target_plan_still_present_post_cleanup")
    cross = scan_candidate_references_other_run(cand_root, forbidden_root=ref_root)
    meta["candidate_reference_run_path_hits"] = cross
    if cross:
        errors.append(f"candidate:references_reference_run_paths:{len(cross)}_hits")
    val_errors, val_meta = verify_candidate_validation_package(cand_root)
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
    }
    if "TARGET_PLAN_UNAVAILABLE" in ref_plan_errs or "TARGET_PLAN_UNAVAILABLE" in cand_plan_errs:
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


def _performance_metrics(ref_rt: Mapping[str, Any], cand_rt: Mapping[str, Any]) -> Dict[str, Any]:
    ref_wall = float(ref_rt.get("stage_wall_times_s", {}).get("stage5_workers") or ref_rt.get("workers_wall_time_s") or 0)
    cand_wall = float(cand_rt.get("stage_wall_times_s", {}).get("stage5_workers") or cand_rt.get("workers_wall_time_s") or 0)
    runtime_reduction = (ref_wall - cand_wall) / ref_wall if ref_wall > 0 else None
    worker_records = list(cand_rt.get("worker_resource_records") or [])
    peaks = [
        int(r.get("peak_rss_bytes") or r.get("max_rss_bytes"))
        for r in worker_records
        if r.get("peak_rss_bytes") or r.get("max_rss_bytes")
    ]
    peak_max = max(peaks) if peaks else cand_rt.get("peak_rss_bytes_max_worker")
    sum_peaks = sum(peaks) if peaks else None
    return {
        "reference_worker_wall_s": ref_wall,
        "candidate_worker_wall_s": cand_wall,
        "runtime_reduction_fraction": runtime_reduction,
        "candidate_peak_rss_bytes_max_worker": peak_max,
        "candidate_sum_of_individual_worker_peaks_upper_bound": sum_peaks,
        "candidate_workers_parallel_observed": cand_rt.get("workers_actual_parallel") or cand_rt.get("workers_parallel_observed"),
        "rss_measurement_note": cand_rt.get("rss_aggregate_note"),
    }


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
) -> Dict[str, Any]:
    ref_root = reference_run.resolve()
    cand_root = candidate_run.resolve()

    cleanup_errors, cleanup_meta = verify_cleanup_preconditions(
        repo_root=repo_root, ref_root=ref_root, cand_root=cand_root,
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

    phys_errors, phys_meta = verify_physics_identity_equivalence(
        ref_root=ref_root, cand_root=cand_root, ref_legacy_meta=ref_legacy_meta,
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
            "reference_deduped_mode_count": len(ref_rows),
            "candidate_deduped_mode_count": len(cand_rows),
            "matched_mode_count": len(pairs),
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
            "bands": band_stats,
        },
        "coupling_output": coupling,
        "intrinsic_coverage": intrinsic_cov,
        "mode_family_survival": family_survival,
        "scout_provenance": {
            "reference": ref_scout,
            "candidate": cand_scout,
            "errors": scout_provenance_errors,
        },
        "mac": mac,
        "performance": _performance_metrics(_runtime_prov(ref_root), _runtime_prov(cand_root)),
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
    }

    ae, acceptance_pass, incomplete = evaluate_acceptance(report)
    report["acceptance_evaluation"] = ae
    report["acceptance_pass"] = acceptance_pass
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
