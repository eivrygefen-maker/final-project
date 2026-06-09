#!/usr/bin/env python3
"""Intrinsic Scout density contract for m4_geometry_corrected_v1 production."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

COVERAGE_POLICY_INTRINSIC = "intrinsic_discovered_modes_v1"
COVERAGE_POLICY_VERSION = "m4_scout_intrinsic_coverage_v1"

PRODUCTION_BAND_LO_HZ = 60.0
PRODUCTION_BAND_HI_HZ = 550.0
ENDPOINT_TOLERANCE_HZ = 5.0
MAX_RAW_GAP_HZ = 25.0
MIN_RAW_UNIQUE_ACCEPTED = 12
MIN_MODES_PER_BAND_THIRD = 2
DEDUPE_TOL_HZ = 0.05
MAX_DUPLICATE_RATE = 0.20

REFERENCE_CLASS_STUB = "discovery_stub"
REFERENCE_CLASS_EXPLICIT_VALIDATION = "explicit_validation_reference"
REFERENCE_CLASS_EXTERNAL_REAL = "external_real_reference"
REFERENCE_CLASS_UNCLASSIFIED = "unclassified_reference"

EXTERNAL_STATUS_NOT_APPLICABLE_STUB = "NOT_APPLICABLE_STUB"
EXTERNAL_STATUS_DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
EXTERNAL_STATUS_GATED = "GATED"

_STUB_NOTE_MARKERS = (
    "stub",
    "discovery-only",
    "discovery only",
    "not meaningful",
    "placeholder",
    "diagnostic only",
)


def _dedupe_sorted(freqs: Sequence[float], *, tol_hz: float = DEDUPE_TOL_HZ) -> List[float]:
    out: List[float] = []
    for f in sorted(float(x) for x in freqs):
        if not out or abs(f - out[-1]) > tol_hz:
            out.append(f)
    return out


def _max_gap_hz(freqs: Sequence[float]) -> float:
    if len(freqs) < 2:
        return float("inf") if freqs else float("inf")
    gaps = [float(freqs[i + 1]) - float(freqs[i]) for i in range(len(freqs) - 1)]
    return max(gaps) if gaps else float("inf")


def per_third_band_counts(freqs: Sequence[float], *, band_lo: float, band_hi: float) -> Dict[str, int]:
    """Public helper: mode counts in low/mid/high band-thirds."""
    return _per_third_counts(freqs, band_lo=band_lo, band_hi=band_hi)


def _per_third_counts(freqs: Sequence[float], *, band_lo: float, band_hi: float) -> Dict[str, int]:
    width = (float(band_hi) - float(band_lo)) / 3.0
    thirds = {
        "low_third": (band_lo, band_lo + width),
        "mid_third": (band_lo + width, band_lo + 2.0 * width),
        "high_third": (band_lo + 2.0 * width, band_hi),
    }
    counts = {name: 0 for name in thirds}
    for f in freqs:
        fv = float(f)
        for name, (lo, hi) in thirds.items():
            if lo - 1e-9 <= fv <= hi + 1e-9:
                counts[name] += 1
                break
    return counts


def classify_reference_json(
    reference_json: Mapping[str, Any],
    *,
    reference_path: Optional[str] = None,
    band_lo_hz: float = PRODUCTION_BAND_LO_HZ,
    band_hi_hz: float = PRODUCTION_BAND_HI_HZ,
) -> Dict[str, Any]:
    """Classify whether an external reference may gate production acceptance."""
    path_text = str(reference_path or "").replace("\\", "/").lower()
    note = str(reference_json.get("note") or reference_json.get("description") or "").lower()
    declared = str(
        reference_json.get("reference_classification")
        or reference_json.get("reference_kind")
        or ""
    ).lower()

    markers_hit = [m for m in _STUB_NOTE_MARKERS if m in note]
    path_stub = "scout_discovery_reference_stub" in path_text or path_text.endswith("/stub.json")
    declared_stub = declared in ("discovery_stub", "stub", "placeholder")

    freqs: List[float] = []
    agg = reference_json.get("aggregate") or {}
    if agg.get("unique_accepted_frequencies_hz"):
        freqs = [float(x) for x in agg["unique_accepted_frequencies_hz"]]
    elif reference_json.get("accepted_frequencies_hz"):
        freqs = [float(x) for x in reference_json["accepted_frequencies_hz"]]

    narrow_single_freq = len(freqs) <= 1 and (band_hi_hz - band_lo_hz) > 50.0
    explicit_gate = bool(reference_json.get("external_reference_gate_enabled"))
    explicit_validation = bool(reference_json.get("explicit_validation_reference"))
    same_geometry = reference_json.get("same_geometry_as_run")
    sample_id = str(reference_json.get("sample_id") or "")

    if declared_stub or path_stub or markers_hit or narrow_single_freq:
        classification = REFERENCE_CLASS_STUB
        gate_enabled = False
        external_status = EXTERNAL_STATUS_NOT_APPLICABLE_STUB
    elif explicit_validation and explicit_gate and same_geometry is True and sample_id:
        classification = REFERENCE_CLASS_EXPLICIT_VALIDATION
        gate_enabled = True
        external_status = EXTERNAL_STATUS_GATED
    elif explicit_gate:
        classification = REFERENCE_CLASS_EXTERNAL_REAL
        gate_enabled = True
        external_status = EXTERNAL_STATUS_GATED
    else:
        classification = REFERENCE_CLASS_UNCLASSIFIED
        gate_enabled = False
        external_status = EXTERNAL_STATUS_DIAGNOSTIC_ONLY

    return {
        "external_reference_path": str(reference_path or ""),
        "external_reference_classification": classification,
        "external_reference_gate_enabled": gate_enabled,
        "external_reference_status": external_status,
        "external_reference_stub_markers": markers_hit,
        "external_reference_frequency_count": len(freqs),
        "use_intrinsic_policy": not gate_enabled,
    }


def validate_external_reference_gate(
    reference_json: Mapping[str, Any],
    *,
    reference_path: Optional[str],
    run_sample_id: str,
    run_geometry_fingerprint: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Dedicated validation only: explicit non-stub same-geometry reference may gate."""
    meta = classify_reference_json(reference_json, reference_path=reference_path)
    failures: List[str] = []
    if not meta["external_reference_gate_enabled"]:
        failures.append("external_reference_gate_not_enabled")
        return False, failures
    if meta["external_reference_classification"] == REFERENCE_CLASS_STUB:
        failures.append("external_reference_is_stub")
    ref_sample = str(reference_json.get("sample_id") or "")
    if not ref_sample or ref_sample != run_sample_id:
        failures.append(f"external_reference_sample_mismatch:{ref_sample}!={run_sample_id}")
    ref_fp = str(reference_json.get("geometry_fingerprint") or "")
    if run_geometry_fingerprint and ref_fp and ref_fp != run_geometry_fingerprint:
        failures.append("external_reference_geometry_fingerprint_mismatch")
    if not bool(reference_json.get("non_stub")):
        failures.append("external_reference_not_marked_non_stub")
    return len(failures) == 0, failures


def evaluate_intrinsic_scout_coverage(
    *,
    spacing_rows: Sequence[Mapping[str, Any]],
    band_lo_hz: float,
    band_hi_hz: float,
    dedupe_tol_hz: float = DEDUPE_TOL_HZ,
) -> Dict[str, Any]:
    """Evaluate discovered accepted modes before any Stage-3 target-plan repair."""
    raw_freqs: List[float] = []
    target_success_count = 0
    target_failure_count = 0
    best_spacing_hz: Optional[float] = None

    for row in spacing_rows:
        target_success_count += int(row.get("targets_succeeded") or 0)
        target_failure_count += int(row.get("targets_failed") or 0)
        if int(row.get("targets_failed") or 0) != 0:
            continue
        spacing_hz = float(row.get("spacing_hz") or 0.0)
        freqs = [float(x) for x in (row.get("unique_accepted_frequencies_hz") or [])]
        raw_freqs.extend(freqs)
        if best_spacing_hz is None or spacing_hz > best_spacing_hz:
            best_spacing_hz = spacing_hz

    deduped = _dedupe_sorted(raw_freqs, tol_hz=dedupe_tol_hz)
    raw_count = len(raw_freqs)
    deduped_count = len(deduped)
    duplicate_rate = 0.0
    if raw_count > 0:
        duplicate_rate = max(0.0, (raw_count - deduped_count) / float(raw_count))

    failures: List[str] = []
    if target_failure_count > 0:
        failures.append(f"target_solver_failures:{target_failure_count}")
    if deduped_count == 0:
        failures.append("raw_unique_accepted_empty")
    if deduped_count < MIN_RAW_UNIQUE_ACCEPTED:
        failures.append(f"raw_unique_accepted_count<{MIN_RAW_UNIQUE_ACCEPTED}")

    freq_min = min(deduped) if deduped else None
    freq_max = max(deduped) if deduped else None
    if deduped:
        if float(freq_min) > float(band_lo_hz) + ENDPOINT_TOLERANCE_HZ:
            failures.append(f"low_band_endpoint_missing:min={freq_min}")
        if float(freq_max) < float(band_hi_hz) - ENDPOINT_TOLERANCE_HZ:
            failures.append(f"high_band_endpoint_missing:max={freq_max}")
        max_gap = _max_gap_hz(deduped)
        if max_gap > MAX_RAW_GAP_HZ:
            failures.append(f"raw_max_gap_hz>{MAX_RAW_GAP_HZ}:gap={max_gap:.3f}")
    else:
        max_gap = float("inf")

    per_zone = _per_third_counts(deduped, band_lo=band_lo_hz, band_hi=band_hi_hz)
    for zone, count in per_zone.items():
        if count < MIN_MODES_PER_BAND_THIRD:
            failures.append(f"{zone}_mode_count<{MIN_MODES_PER_BAND_THIRD}:{count}")

    if duplicate_rate > MAX_DUPLICATE_RATE:
        failures.append(f"duplicate_rate>{MAX_DUPLICATE_RATE}:{duplicate_rate:.3f}")

    return {
        "coverage_policy": COVERAGE_POLICY_INTRINSIC,
        "coverage_policy_version": COVERAGE_POLICY_VERSION,
        "raw_unique_accepted_count": deduped_count,
        "raw_frequency_min_hz": freq_min,
        "raw_frequency_max_hz": freq_max,
        "raw_max_gap_hz": None if not deduped else round(_max_gap_hz(deduped), 6),
        "per_zone_mode_counts": per_zone,
        "target_success_count": target_success_count,
        "target_failure_count": target_failure_count,
        "duplicate_rate": round(duplicate_rate, 6),
        "best_intrinsic_spacing_hz": best_spacing_hz,
        "intrinsic_coverage_pass": len(failures) == 0,
        "intrinsic_coverage_failures": failures,
    }


def build_density_provenance_fields(
    *,
    reference_meta: Mapping[str, Any],
    intrinsic: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "coverage_policy": intrinsic.get("coverage_policy"),
        "coverage_policy_version": intrinsic.get("coverage_policy_version"),
        "external_reference_path": reference_meta.get("external_reference_path"),
        "external_reference_classification": reference_meta.get("external_reference_classification"),
        "external_reference_gate_enabled": reference_meta.get("external_reference_gate_enabled"),
        "external_reference_status": reference_meta.get("external_reference_status"),
        "raw_unique_accepted_count": intrinsic.get("raw_unique_accepted_count"),
        "raw_frequency_min_hz": intrinsic.get("raw_frequency_min_hz"),
        "raw_frequency_max_hz": intrinsic.get("raw_frequency_max_hz"),
        "raw_max_gap_hz": intrinsic.get("raw_max_gap_hz"),
        "per_zone_mode_counts": intrinsic.get("per_zone_mode_counts"),
        "target_success_count": intrinsic.get("target_success_count"),
        "target_failure_count": intrinsic.get("target_failure_count"),
        "intrinsic_coverage_pass": intrinsic.get("intrinsic_coverage_pass"),
        "intrinsic_coverage_failures": list(intrinsic.get("intrinsic_coverage_failures") or []),
    }


def production_density_status(
    *,
    reference_meta: Mapping[str, Any],
    intrinsic: Mapping[str, Any],
    spacing_rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, Optional[float]]:
    """Production PASS/FAIL only; never PARTIAL."""
    if reference_meta.get("use_intrinsic_policy"):
        if intrinsic.get("intrinsic_coverage_pass"):
            return "PASS", intrinsic.get("best_intrinsic_spacing_hz")
        return "FAIL", None
    if not reference_meta.get("external_reference_gate_enabled"):
        if intrinsic.get("intrinsic_coverage_pass"):
            return "PASS", intrinsic.get("best_intrinsic_spacing_hz")
        return "FAIL", None
    passing = [
        float(r["spacing_hz"])
        for r in spacing_rows
        if bool(r.get("coverage_pass")) and int(r.get("targets_failed") or 0) == 0
    ]
    sparsest = max(passing) if passing else None
    if sparsest is not None:
        return "PASS", sparsest
    if all(str(r.get("status")) == "FAIL" for r in spacing_rows):
        return "FAIL", None
    return "FAIL", None


def intrinsic_density_result_ok(data: Mapping[str, Any]) -> Tuple[bool, str]:
    policy = str(data.get("coverage_policy") or "")
    if policy != COVERAGE_POLICY_INTRINSIC:
        return False, f"coverage_policy={policy or 'missing'}"
    if str(data.get("status") or "") != "PASS":
        return False, f"status={data.get('status') or 'missing'}"
    if not bool(data.get("intrinsic_coverage_pass")):
        failures = data.get("intrinsic_coverage_failures") or []
        return False, f"intrinsic_coverage_failures:{failures}"
    if bool(data.get("external_reference_gate_enabled")) and (
        data.get("external_reference_classification") == REFERENCE_CLASS_STUB
    ):
        return False, "external_reference_gate_enabled_with_stub"
    return True, "ok"
