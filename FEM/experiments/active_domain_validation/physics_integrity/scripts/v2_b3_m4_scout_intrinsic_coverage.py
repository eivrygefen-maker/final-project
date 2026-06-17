#!/usr/bin/env python3
"""Intrinsic Scout density contract for m4_geometry_corrected_v1 production."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

COVERAGE_POLICY_INTRINSIC = "intrinsic_discovered_modes_v1"
COVERAGE_POLICY_BOX_V1 = "box_discovered_modes_v1"
COVERAGE_POLICY_BOX = "box_discovered_modes_v2"
COVERAGE_POLICY_ACOUSTIC = "acoustic_discovered_modes_v1"
COVERAGE_POLICY_VERSION = "m4_scout_intrinsic_coverage_v1"
COVERAGE_POLICY_VERSION_SHAPE = "m4_scout_intrinsic_coverage_v2"
COVERAGE_POLICY_VERSION_BOX_V2 = "m4_scout_intrinsic_coverage_box_v2"

PRODUCTION_BAND_LO_HZ = 60.0
PRODUCTION_BAND_HI_HZ = 550.0
ENDPOINT_TOLERANCE_HZ = 5.0
MAX_RAW_GAP_HZ = 25.0
MIN_RAW_UNIQUE_ACCEPTED = 12
MIN_MODES_PER_BAND_THIRD = 2
DEDUPE_TOL_HZ = 0.05
MAX_DUPLICATE_RATE = 0.20
REVIEW_FREQ_LIST_CAP = 128

ENDPOINT_MODE_FULL_BAND = "full_production_band"
ENDPOINT_MODE_SOLVER_SWEEP_EVIDENCE = "solver_sweep_evidence"
GAP_MODE_ABSOLUTE = "absolute"
GAP_MODE_DISCOVERED_SPAN_RELATIVE = "discovered_span_relative"
BAND_THIRDS_PRODUCTION = "production_band"
BAND_THIRDS_DISCOVERED = "discovered_band"

REGISTERED_SCOUT_DENSITY_POLICIES: Tuple[str, ...] = (
    COVERAGE_POLICY_INTRINSIC,
    COVERAGE_POLICY_BOX_V1,
    COVERAGE_POLICY_BOX,
    COVERAGE_POLICY_ACOUSTIC,
)

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


@dataclass(frozen=True)
class ScoutIntrinsicPolicySpec:
    policy_id: str
    policy_version: str
    min_raw_unique_accepted: int
    min_modes_per_band_third: int
    endpoint_tolerance_hz: float
    high_endpoint_tolerance_hz: float
    max_raw_gap_hz: float
    max_duplicate_rate: float
    shape_key: str
    low_endpoint_requires_mode: bool = True
    high_endpoint_requires_mode: bool = True
    gap_mode: str = GAP_MODE_ABSOLUTE
    max_gap_relative_of_discovered_span: float = 0.0
    per_third_band_source: str = BAND_THIRDS_PRODUCTION
    min_discovered_span_hz: float = 0.0
    low_endpoint_mode: str = ENDPOINT_MODE_FULL_BAND
    min_accepted_mode_max_hz: float = 0.0
    shape_relative_warnings: bool = False


SCOUT_INTRINSIC_POLICY_SPECS: Dict[str, ScoutIntrinsicPolicySpec] = {
    COVERAGE_POLICY_INTRINSIC: ScoutIntrinsicPolicySpec(
        policy_id=COVERAGE_POLICY_INTRINSIC,
        policy_version=COVERAGE_POLICY_VERSION,
        min_raw_unique_accepted=MIN_RAW_UNIQUE_ACCEPTED,
        min_modes_per_band_third=MIN_MODES_PER_BAND_THIRD,
        endpoint_tolerance_hz=ENDPOINT_TOLERANCE_HZ,
        high_endpoint_tolerance_hz=ENDPOINT_TOLERANCE_HZ,
        max_raw_gap_hz=MAX_RAW_GAP_HZ,
        max_duplicate_rate=MAX_DUPLICATE_RATE,
        shape_key="classic",
    ),
  # v1 retained for reading historical failed runs; superseded by v2 for production.
    COVERAGE_POLICY_BOX_V1: ScoutIntrinsicPolicySpec(
        policy_id=COVERAGE_POLICY_BOX_V1,
        policy_version=COVERAGE_POLICY_VERSION_SHAPE,
        min_raw_unique_accepted=8,
        min_modes_per_band_third=1,
        endpoint_tolerance_hz=8.0,
        high_endpoint_tolerance_hz=8.0,
        max_raw_gap_hz=35.0,
        max_duplicate_rate=0.35,
        shape_key="box",
    ),
    COVERAGE_POLICY_BOX: ScoutIntrinsicPolicySpec(
        policy_id=COVERAGE_POLICY_BOX,
        policy_version=COVERAGE_POLICY_VERSION_BOX_V2,
        # Box cavity: gate on discovered-band sufficiency (count/span/max), not classical
        # reference matching or proximity to the production band endpoints.
        min_raw_unique_accepted=8,
        min_modes_per_band_third=1,
        endpoint_tolerance_hz=8.0,
        high_endpoint_tolerance_hz=20.0,
        max_raw_gap_hz=35.0,
        max_duplicate_rate=0.35,
        shape_key="box",
        low_endpoint_requires_mode=False,
        high_endpoint_requires_mode=False,
        low_endpoint_mode=ENDPOINT_MODE_SOLVER_SWEEP_EVIDENCE,
        gap_mode=GAP_MODE_DISCOVERED_SPAN_RELATIVE,
        max_gap_relative_of_discovered_span=0.42,
        per_third_band_source=BAND_THIRDS_DISCOVERED,
        min_discovered_span_hz=250.0,
        min_accepted_mode_max_hz=500.0,
        shape_relative_warnings=True,
    ),
    COVERAGE_POLICY_ACOUSTIC: ScoutIntrinsicPolicySpec(
        policy_id=COVERAGE_POLICY_ACOUSTIC,
        policy_version=COVERAGE_POLICY_VERSION_SHAPE,
        min_raw_unique_accepted=10,
        min_modes_per_band_third=2,
        endpoint_tolerance_hz=6.0,
        high_endpoint_tolerance_hz=6.0,
        max_raw_gap_hz=30.0,
        max_duplicate_rate=0.25,
        shape_key="acoustic",
    ),
}


def is_registered_scout_density_policy(policy: str) -> bool:
    return str(policy or "") in SCOUT_INTRINSIC_POLICY_SPECS


def resolve_scout_intrinsic_policy_spec(policy_id: Optional[str]) -> ScoutIntrinsicPolicySpec:
    key = str(policy_id or COVERAGE_POLICY_INTRINSIC)
    return SCOUT_INTRINSIC_POLICY_SPECS.get(key, SCOUT_INTRINSIC_POLICY_SPECS[COVERAGE_POLICY_INTRINSIC])


def registered_scout_density_policies() -> Tuple[str, ...]:
    return REGISTERED_SCOUT_DENSITY_POLICIES


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


def _cap_freq_list(freqs: Sequence[float], *, cap: int = REVIEW_FREQ_LIST_CAP) -> List[float]:
    items = [round(float(f), 6) for f in freqs]
    if len(items) <= cap:
        return items
    return items[:cap]


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


def _policy_thresholds_dict(spec: ScoutIntrinsicPolicySpec) -> Dict[str, Any]:
    return {
        "min_raw_unique_accepted": spec.min_raw_unique_accepted,
        "min_modes_per_band_third": spec.min_modes_per_band_third,
        "endpoint_tolerance_hz": spec.endpoint_tolerance_hz,
        "high_endpoint_tolerance_hz": spec.high_endpoint_tolerance_hz,
        "max_raw_gap_hz": spec.max_raw_gap_hz,
        "max_duplicate_rate": spec.max_duplicate_rate,
        "low_endpoint_requires_mode": spec.low_endpoint_requires_mode,
        "high_endpoint_requires_mode": spec.high_endpoint_requires_mode,
        "low_endpoint_mode": spec.low_endpoint_mode,
        "gap_mode": spec.gap_mode,
        "max_gap_relative_of_discovered_span": spec.max_gap_relative_of_discovered_span,
        "per_third_band_source": spec.per_third_band_source,
        "min_discovered_span_hz": spec.min_discovered_span_hz,
        "min_accepted_mode_max_hz": spec.min_accepted_mode_max_hz,
        "shape_relative_warnings": spec.shape_relative_warnings,
    }


def _effective_gap_limit_hz(
    spec: ScoutIntrinsicPolicySpec,
    *,
    discovered_span_hz: float,
) -> float:
    if spec.gap_mode == GAP_MODE_DISCOVERED_SPAN_RELATIVE and discovered_span_hz > 0.0:
        relative = float(spec.max_gap_relative_of_discovered_span) * discovered_span_hz
        return max(float(spec.max_raw_gap_hz), relative)
    return float(spec.max_raw_gap_hz)


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


def _append_gate(
    *,
    spec: ScoutIntrinsicPolicySpec,
    hard_failures: List[str],
    warnings: List[str],
    advisory: bool,
    message: str,
) -> None:
    if advisory and spec.shape_relative_warnings:
        warnings.append(message)
    else:
        hard_failures.append(message)


def evaluate_intrinsic_scout_coverage(
    *,
    spacing_rows: Sequence[Mapping[str, Any]],
    band_lo_hz: float,
    band_hi_hz: float,
    dedupe_tol_hz: float = DEDUPE_TOL_HZ,
    coverage_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate discovered accepted modes before any Stage-3 target-plan repair."""
    spec = resolve_scout_intrinsic_policy_spec(coverage_policy)
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
    raw_sorted = _dedupe_sorted(raw_freqs, tol_hz=dedupe_tol_hz)
    raw_count = len(raw_freqs)
    deduped_count = len(deduped)
    duplicate_rate = 0.0
    if raw_count > 0:
        duplicate_rate = max(0.0, (raw_count - deduped_count) / float(raw_count))

    hard_failures: List[str] = []
    warnings: List[str] = []
    endpoint_warnings: List[str] = []
    band_distribution_warnings: List[str] = []
    coverage_warnings: List[str] = []

    if target_failure_count > 0:
        hard_failures.append(f"target_solver_failures:{target_failure_count}")
    if deduped_count == 0:
        hard_failures.append("raw_unique_accepted_empty")
    if deduped_count < spec.min_raw_unique_accepted:
        hard_failures.append(
            f"raw_unique_accepted_count<{spec.min_raw_unique_accepted}:{deduped_count}"
        )

    freq_min = min(deduped) if deduped else None
    freq_max = max(deduped) if deduped else None
    discovered_span_hz = (
        float(freq_max) - float(freq_min) if freq_min is not None and freq_max is not None else 0.0
    )
    max_gap = _max_gap_hz(deduped) if deduped else float("inf")
    gap_limit_hz = _effective_gap_limit_hz(spec, discovered_span_hz=discovered_span_hz)

    low_endpoint_policy: Dict[str, Any] = {
        "mode": spec.low_endpoint_mode,
        "requires_mode_near_band_lo": spec.low_endpoint_requires_mode,
        "production_band_lo_hz": float(band_lo_hz),
        "tolerance_hz": spec.endpoint_tolerance_hz,
    }
    high_endpoint_policy: Dict[str, Any] = {
        "requires_mode_near_band_hi": spec.high_endpoint_requires_mode,
        "production_band_hi_hz": float(band_hi_hz),
        "tolerance_hz": spec.high_endpoint_tolerance_hz,
        "min_accepted_mode_max_hz": spec.min_accepted_mode_max_hz,
        "advisory_only_for_box": spec.shape_relative_warnings,
    }

    if deduped:
        if spec.min_discovered_span_hz > 0.0 and discovered_span_hz < spec.min_discovered_span_hz:
            hard_failures.append(
                f"discovered_span_hz<{spec.min_discovered_span_hz}:{discovered_span_hz:.3f}"
            )

        if spec.min_accepted_mode_max_hz > 0.0 and float(freq_max) < spec.min_accepted_mode_max_hz:
            hard_failures.append(
                f"accepted_mode_max_hz<{spec.min_accepted_mode_max_hz}:{float(freq_max):.3f}"
            )

        if spec.low_endpoint_requires_mode:
            if float(freq_min) > float(band_lo_hz) + spec.endpoint_tolerance_hz:
                _append_gate(
                    spec=spec,
                    hard_failures=hard_failures,
                    warnings=endpoint_warnings,
                    advisory=False,
                    message=f"low_band_endpoint_missing:min={freq_min}",
                )
        else:
            low_endpoint_policy["solver_sweep_evidence"] = {
                "targets_succeeded": target_success_count,
                "targets_failed": target_failure_count,
                "note": (
                    "Low-band ST targets may solve with empty acceptance windows when the "
                    "first physical mode is above the production band floor."
                ),
            }
            if target_success_count <= 0:
                hard_failures.append("low_band_solver_evidence_missing:no_targets_solved")

        if spec.high_endpoint_requires_mode:
            if float(freq_max) < float(band_hi_hz) - spec.high_endpoint_tolerance_hz:
                _append_gate(
                    spec=spec,
                    hard_failures=hard_failures,
                    warnings=endpoint_warnings,
                    advisory=spec.shape_relative_warnings,
                    message=f"high_band_endpoint_missing:max={freq_max}",
                )
        elif spec.shape_relative_warnings:
            hi_target = float(band_hi_hz) - spec.high_endpoint_tolerance_hz
            if float(freq_max) < hi_target:
                endpoint_warnings.append(
                    f"high_band_endpoint_proximity_miss:max={float(freq_max):.3f} "
                    f"target_ge={hi_target:.3f}"
                )

        if max_gap > gap_limit_hz:
            _append_gate(
                spec=spec,
                hard_failures=hard_failures,
                warnings=coverage_warnings,
                advisory=spec.shape_relative_warnings,
                message=f"raw_max_gap_hz>{gap_limit_hz:.3f}:gap={max_gap:.3f}",
            )

    if spec.per_third_band_source == BAND_THIRDS_DISCOVERED and deduped:
        third_lo = float(freq_min)
        third_hi = float(freq_max)
    else:
        third_lo = float(band_lo_hz)
        third_hi = float(band_hi_hz)
    per_zone = _per_third_counts(deduped, band_lo=third_lo, band_hi=third_hi)
    for zone, count in per_zone.items():
        if count < spec.min_modes_per_band_third:
            _append_gate(
                spec=spec,
                hard_failures=hard_failures,
                warnings=band_distribution_warnings,
                advisory=spec.shape_relative_warnings,
                message=f"{zone}_mode_count<{spec.min_modes_per_band_third}:{count}",
            )

    if duplicate_rate > spec.max_duplicate_rate:
        _append_gate(
            spec=spec,
            hard_failures=hard_failures,
            warnings=coverage_warnings,
            advisory=spec.shape_relative_warnings,
            message=f"duplicate_rate>{spec.max_duplicate_rate}:{duplicate_rate:.3f}",
        )

    intrinsic_failures = list(hard_failures)
    intrinsic_pass = len(hard_failures) == 0
    intrinsic_pass_with_warnings = intrinsic_pass and bool(
        warnings or endpoint_warnings or band_distribution_warnings or coverage_warnings
    )
    policy_decision_reason = (
        "intrinsic_coverage_pass"
        if intrinsic_pass and not intrinsic_pass_with_warnings
        else (
            "intrinsic_coverage_pass_with_warnings"
            if intrinsic_pass_with_warnings
            else ";".join(hard_failures)
        )
    )

    return {
        "coverage_policy": spec.policy_id,
        "coverage_policy_version": spec.policy_version,
        "shape_key": spec.shape_key,
        "policy_thresholds": _policy_thresholds_dict(spec),
        "raw_unique_accepted_count": deduped_count,
        "raw_frequency_min_hz": freq_min,
        "raw_frequency_max_hz": freq_max,
        "discovered_span_hz": round(discovered_span_hz, 6) if deduped else None,
        "raw_max_gap_hz": None if not deduped else round(max_gap, 6),
        "deduped_max_gap_hz": None if not deduped else round(max_gap, 6),
        "effective_gap_limit_hz": round(gap_limit_hz, 6) if deduped else None,
        "per_zone_mode_counts": per_zone,
        "per_third_band_source": spec.per_third_band_source,
        "target_success_count": target_success_count,
        "target_failure_count": target_failure_count,
        "targets_solved_count": target_success_count,
        "targets_failed_count": target_failure_count,
        "duplicate_rate": round(duplicate_rate, 6),
        "best_intrinsic_spacing_hz": best_spacing_hz,
        "intrinsic_coverage_pass": intrinsic_pass,
        "intrinsic_coverage_pass_with_warnings": intrinsic_pass_with_warnings,
        "intrinsic_coverage_failures": intrinsic_failures,
        "intrinsic_coverage_warnings": warnings,
        "coverage_warnings": coverage_warnings,
        "endpoint_warnings": endpoint_warnings,
        "band_distribution_warnings": band_distribution_warnings,
        "reference_coverage_warnings": [],
        "intrinsic_policy_thresholds": _policy_thresholds_dict(spec),
        "low_endpoint_policy": low_endpoint_policy,
        "high_endpoint_policy": high_endpoint_policy,
        "accepted_mode_count": deduped_count,
        "accepted_mode_min_hz": freq_min,
        "accepted_mode_max_hz": freq_max,
        "accepted_mode_frequencies_hz": _cap_freq_list(deduped),
        "raw_mode_count": raw_count,
        "raw_mode_frequencies_hz": _cap_freq_list(raw_sorted),
        "policy_decision_reason": policy_decision_reason,
    }


def build_density_provenance_fields(
    *,
    reference_meta: Mapping[str, Any],
    intrinsic: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "coverage_policy": intrinsic.get("coverage_policy"),
        "coverage_policy_version": intrinsic.get("coverage_policy_version"),
        "policy_thresholds": intrinsic.get("policy_thresholds"),
        "external_reference_path": reference_meta.get("external_reference_path"),
        "external_reference_classification": reference_meta.get("external_reference_classification"),
        "external_reference_gate_enabled": reference_meta.get("external_reference_gate_enabled"),
        "external_reference_status": reference_meta.get("external_reference_status"),
        "raw_unique_accepted_count": intrinsic.get("raw_unique_accepted_count"),
        "raw_frequency_min_hz": intrinsic.get("raw_frequency_min_hz"),
        "raw_frequency_max_hz": intrinsic.get("raw_frequency_max_hz"),
        "discovered_span_hz": intrinsic.get("discovered_span_hz"),
        "raw_max_gap_hz": intrinsic.get("raw_max_gap_hz"),
        "deduped_max_gap_hz": intrinsic.get("deduped_max_gap_hz"),
        "effective_gap_limit_hz": intrinsic.get("effective_gap_limit_hz"),
        "per_zone_mode_counts": intrinsic.get("per_zone_mode_counts"),
        "per_third_band_source": intrinsic.get("per_third_band_source"),
        "target_success_count": intrinsic.get("target_success_count"),
        "target_failure_count": intrinsic.get("target_failure_count"),
        "targets_solved_count": intrinsic.get("targets_solved_count"),
        "targets_failed_count": intrinsic.get("targets_failed_count"),
        "intrinsic_coverage_pass": intrinsic.get("intrinsic_coverage_pass"),
        "intrinsic_coverage_pass_with_warnings": intrinsic.get("intrinsic_coverage_pass_with_warnings"),
        "intrinsic_coverage_failures": list(intrinsic.get("intrinsic_coverage_failures") or []),
        "intrinsic_coverage_warnings": list(intrinsic.get("intrinsic_coverage_warnings") or []),
        "coverage_warnings": list(intrinsic.get("coverage_warnings") or []),
        "endpoint_warnings": list(intrinsic.get("endpoint_warnings") or []),
        "band_distribution_warnings": list(intrinsic.get("band_distribution_warnings") or []),
        "reference_coverage_warnings": list(intrinsic.get("reference_coverage_warnings") or []),
        "intrinsic_policy_thresholds": intrinsic.get("intrinsic_policy_thresholds"),
        "shape_key": intrinsic.get("shape_key"),
        "low_endpoint_policy": intrinsic.get("low_endpoint_policy"),
        "high_endpoint_policy": intrinsic.get("high_endpoint_policy"),
        "accepted_mode_count": intrinsic.get("accepted_mode_count"),
        "accepted_mode_min_hz": intrinsic.get("accepted_mode_min_hz"),
        "accepted_mode_max_hz": intrinsic.get("accepted_mode_max_hz"),
        "accepted_mode_frequencies_hz": intrinsic.get("accepted_mode_frequencies_hz"),
        "raw_mode_count": intrinsic.get("raw_mode_count"),
        "raw_mode_frequencies_hz": intrinsic.get("raw_mode_frequencies_hz"),
        "duplicate_rate": intrinsic.get("duplicate_rate"),
        "policy_decision_reason": intrinsic.get("policy_decision_reason"),
    }


def production_density_status(
    *,
    reference_meta: Mapping[str, Any],
    intrinsic: Mapping[str, Any],
    spacing_rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, Optional[float]]:
    """Production PASS / PASS_WITH_WARNING / FAIL; never PARTIAL."""
    def _spacing_from_intrinsic() -> Tuple[str, Optional[float]]:
        if intrinsic.get("intrinsic_coverage_pass"):
            spacing = intrinsic.get("best_intrinsic_spacing_hz")
            if intrinsic.get("intrinsic_coverage_pass_with_warnings"):
                return "PASS_WITH_WARNING", spacing
            return "PASS", spacing
        return "FAIL", None

    if reference_meta.get("use_intrinsic_policy"):
        return _spacing_from_intrinsic()
    if not reference_meta.get("external_reference_gate_enabled"):
        return _spacing_from_intrinsic()
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
    if not is_registered_scout_density_policy(policy):
        return False, f"coverage_policy={policy or 'missing'}"
    if str(data.get("status") or "") not in ("PASS", "PASS_WITH_WARNING"):
        return False, f"status={data.get('status') or 'missing'}"
    if not bool(data.get("intrinsic_coverage_pass")):
        failures = data.get("intrinsic_coverage_failures") or []
        return False, f"intrinsic_coverage_failures:{failures}"
    if bool(data.get("external_reference_gate_enabled")) and (
        data.get("external_reference_classification") == REFERENCE_CLASS_STUB
    ):
        return False, "external_reference_gate_enabled_with_stub"
    return True, "ok"
