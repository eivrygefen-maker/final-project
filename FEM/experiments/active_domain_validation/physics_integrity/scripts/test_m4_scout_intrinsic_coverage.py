#!/usr/bin/env python3
"""Regression tests for intrinsic Scout density contract."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_scout_intrinsic_coverage import (  # noqa: E402
    COVERAGE_POLICY_INTRINSIC,
    PRODUCTION_BAND_HI_HZ,
    PRODUCTION_BAND_LO_HZ,
    REFERENCE_CLASS_STUB,
    classify_reference_json,
    evaluate_intrinsic_scout_coverage,
    intrinsic_density_result_ok,
    production_density_status,
    validate_external_reference_gate,
)
from v2_b3_run_coarse_scout_lhs_batch import _verify_density_result  # noqa: E402

STUB_PATH = (
    SCRIPT_DIR.parent
    / "pipeline_runs/specs/scout_discovery_reference_stub.json"
)


def _rich_freqs(*, lo: float = PRODUCTION_BAND_LO_HZ, hi: float = PRODUCTION_BAND_HI_HZ, step: float = 20.0) -> list[float]:
    freqs = []
    f = lo
    while f <= hi + 1e-9:
        freqs.append(round(f, 3))
        f += step
    if not freqs or abs(freqs[-1] - hi) > 1.0:
        freqs.append(float(hi))
    return sorted(set(freqs))


def _spacing_row(
    freqs: list[float],
    *,
    targets_failed: int = 0,
    spacing_hz: float = 7.5,
) -> dict:
    return {
        "spacing_hz": spacing_hz,
        "unique_accepted_frequencies_hz": freqs,
        "unique_accepted_count": len(freqs),
        "targets_failed": targets_failed,
        "targets_succeeded": max(0, 20 - targets_failed),
        "coverage_pass": False,
        "status": "PASS" if targets_failed == 0 else "FAIL",
    }


def _intrinsic_density_payload(*, freqs: list[float], status: str = "PASS") -> dict:
    rows = [_spacing_row(freqs)]
    intrinsic = evaluate_intrinsic_scout_coverage(
        spacing_rows=rows,
        band_lo_hz=PRODUCTION_BAND_LO_HZ,
        band_hi_hz=PRODUCTION_BAND_HI_HZ,
    )
    return {
        "status": status,
        "coverage_policy": COVERAGE_POLICY_INTRINSIC,
        "coverage_policy_version": "m4_scout_intrinsic_coverage_v1",
        "external_reference_classification": REFERENCE_CLASS_STUB,
        "external_reference_gate_enabled": False,
        "external_reference_status": "NOT_APPLICABLE_STUB",
        "intrinsic_coverage_pass": intrinsic["intrinsic_coverage_pass"],
        "intrinsic_coverage_failures": intrinsic["intrinsic_coverage_failures"],
        "raw_unique_accepted_count": intrinsic["raw_unique_accepted_count"],
        "spacings": rows,
        "sparsest_coverage_pass_spacing_hz": 7.5 if status == "PASS" else None,
    }


class ScoutIntrinsicCoverageTests(unittest.TestCase):
    def test_default_stub_classified_non_gating(self) -> None:
        stub = json.loads(STUB_PATH.read_text(encoding="utf-8"))
        meta = classify_reference_json(stub, reference_path=str(STUB_PATH))
        self.assertEqual(meta["external_reference_classification"], REFERENCE_CLASS_STUB)
        self.assertFalse(meta["external_reference_gate_enabled"])
        self.assertTrue(meta["use_intrinsic_policy"])
        self.assertEqual(meta["external_reference_status"], "NOT_APPLICABLE_STUB")

    def test_stub_mismatch_does_not_create_partial(self) -> None:
        stub = json.loads(STUB_PATH.read_text(encoding="utf-8"))
        meta = classify_reference_json(stub, reference_path=str(STUB_PATH))
        rows = [_spacing_row(_rich_freqs())]
        intrinsic = evaluate_intrinsic_scout_coverage(
            spacing_rows=rows,
            band_lo_hz=PRODUCTION_BAND_LO_HZ,
            band_hi_hz=PRODUCTION_BAND_HI_HZ,
        )
        status, _ = production_density_status(
            reference_meta=meta,
            intrinsic=intrinsic,
            spacing_rows=rows,
        )
        self.assertEqual(status, "PASS")
        self.assertNotEqual(status, "PARTIAL")
        self.assertFalse(rows[0]["coverage_pass"])

    def test_rich_intrinsic_coverage_passes(self) -> None:
        rows = [_spacing_row(_rich_freqs())]
        intrinsic = evaluate_intrinsic_scout_coverage(
            spacing_rows=rows,
            band_lo_hz=PRODUCTION_BAND_LO_HZ,
            band_hi_hz=PRODUCTION_BAND_HI_HZ,
        )
        self.assertTrue(intrinsic["intrinsic_coverage_pass"], intrinsic["intrinsic_coverage_failures"])

    def test_large_raw_gap_fails(self) -> None:
        freqs = [60.0, 120.0, 500.0]
        intrinsic = evaluate_intrinsic_scout_coverage(
            spacing_rows=[_spacing_row(freqs)],
            band_lo_hz=PRODUCTION_BAND_LO_HZ,
            band_hi_hz=PRODUCTION_BAND_HI_HZ,
        )
        self.assertFalse(intrinsic["intrinsic_coverage_pass"])
        self.assertTrue(any("raw_max_gap_hz" in f for f in intrinsic["intrinsic_coverage_failures"]))

    def test_missing_low_high_band_coverage_fails(self) -> None:
        freqs = [200.0 + i * 15.0 for i in range(15)]
        intrinsic = evaluate_intrinsic_scout_coverage(
            spacing_rows=[_spacing_row(freqs)],
            band_lo_hz=PRODUCTION_BAND_LO_HZ,
            band_hi_hz=PRODUCTION_BAND_HI_HZ,
        )
        self.assertFalse(intrinsic["intrinsic_coverage_pass"])
        self.assertTrue(
            any("low_band_endpoint_missing" in f or "high_band_endpoint_missing" in f for f in intrinsic["intrinsic_coverage_failures"])
        )

    def test_solver_failure_fails_intrinsic(self) -> None:
        intrinsic = evaluate_intrinsic_scout_coverage(
            spacing_rows=[_spacing_row(_rich_freqs(), targets_failed=2)],
            band_lo_hz=PRODUCTION_BAND_LO_HZ,
            band_hi_hz=PRODUCTION_BAND_HI_HZ,
        )
        self.assertFalse(intrinsic["intrinsic_coverage_pass"])
        self.assertIn("target_solver_failures:2", intrinsic["intrinsic_coverage_failures"])

    def test_repaired_only_coverage_fails_without_intrinsic_pass(self) -> None:
        rows = [_spacing_row([244.39])]
        intrinsic = evaluate_intrinsic_scout_coverage(
            spacing_rows=rows,
            band_lo_hz=PRODUCTION_BAND_LO_HZ,
            band_hi_hz=PRODUCTION_BAND_HI_HZ,
        )
        self.assertFalse(intrinsic["intrinsic_coverage_pass"])
        status, _ = production_density_status(
            reference_meta=classify_reference_json({}, reference_path="stub.json"),
            intrinsic=intrinsic,
            spacing_rows=rows,
        )
        self.assertEqual(status, "FAIL")

    def test_explicit_same_geometry_reference_can_gate(self) -> None:
        ref = {
            "non_stub": True,
            "explicit_validation_reference": True,
            "external_reference_gate_enabled": True,
            "same_geometry_as_run": True,
            "sample_id": "sample_001",
            "geometry_fingerprint": "fp_a",
            "aggregate": {"unique_accepted_frequencies_hz": [100.0, 200.0, 300.0]},
        }
        ok, failures = validate_external_reference_gate(
            ref,
            reference_path="validation.json",
            run_sample_id="sample_001",
            run_geometry_fingerprint="fp_a",
        )
        self.assertTrue(ok, failures)

    def test_foreign_geometry_reference_rejected(self) -> None:
        ref = {
            "non_stub": True,
            "explicit_validation_reference": True,
            "external_reference_gate_enabled": True,
            "same_geometry_as_run": True,
            "sample_id": "sample_001",
            "geometry_fingerprint": "fp_a",
            "aggregate": {"unique_accepted_frequencies_hz": [100.0, 200.0, 300.0]},
        }
        ok, failures = validate_external_reference_gate(
            ref,
            reference_path="validation.json",
            run_sample_id="sample_002",
            run_geometry_fingerprint="fp_b",
        )
        self.assertFalse(ok)
        self.assertTrue(any("sample_mismatch" in f for f in failures))

    def test_strict_verify_rejects_reference_only_density_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "density_result.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "sparsest_coverage_pass_spacing_hz": 7.5,
                        "spacings": [{"spacing_hz": 7.5, "coverage_pass": True, "unique_accepted_count": 3}],
                    }
                ),
                encoding="utf-8",
            )
            ok, detail, _ = _verify_density_result(path, strict=True)
            self.assertFalse(ok)
            self.assertIn("coverage_policy", detail)

    def test_strict_verify_accepts_intrinsic_density_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "density_result.json"
            path.write_text(json.dumps(_intrinsic_density_payload(freqs=_rich_freqs())), encoding="utf-8")
            ok, detail, _ = _verify_density_result(path, strict=True)
            self.assertTrue(ok, detail)
            self.assertTrue(intrinsic_density_result_ok(json.loads(path.read_text(encoding="utf-8")))[0])


if __name__ == "__main__":
    unittest.main()
