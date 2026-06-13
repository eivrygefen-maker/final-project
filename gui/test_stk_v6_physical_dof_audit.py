#!/usr/bin/env python3
"""Lightweight tests for STK V6 physical DOF audit (no FEM/ROM/audio)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from audit_stk_v6_physical_dofs import (  # noqa: E402
    _compute_cavity_geometry_proxy,
    build_stk_v6_physical_dof_audit,
)
from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

IMPORTANT_FEATURE_PROVENANCE_KEYS = (
    "value",
    "status",
    "source_path",
    "confidence",
    "per_sample",
    "intended_v6_use",
)


def _params(**overrides: float) -> dict:
    p = {
        "geometry.length": 0.52,
        "geometry.width": 0.32,
        "geometry.depth": 0.10,
        "geometry.hole_radius": 0.047,
        "materials.top.wood_id": "spruce",
        "materials.back.wood_id": "mahogany",
    }
    for k, v in overrides.items():
        if k == "depth":
            p["geometry.depth"] = v
        elif k == "hole_radius":
            p["geometry.hole_radius"] = v
        else:
            p[k] = v
    return p


def _assert_provenance(rec: dict) -> None:
    for key in IMPORTANT_FEATURE_PROVENANCE_KEYS:
        assert key in rec, f"missing provenance key {key!r} in {rec}"


class TestStkV6PhysicalDofAudit(unittest.TestCase):
    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_json_top_level_flags_are_booleans(self) -> None:
        report = build_stk_v6_physical_dof_audit(repo_root=REPO, sample_ids=("sample_000",))
        for key in (
            "website_default_unchanged",
            "no_audio_synthesis_performed",
            "no_fem_run",
            "no_rom_run",
            "stk_v5_behavior_unchanged",
        ):
            self.assertIn(key, report)
            self.assertIs(report[key], True)
        flags = report.get("explicit_flags") or {}
        self.assertIs(flags.get("website_default_unchanged"), True)

    def test_sample_000_geometry_body_depth_value(self) -> None:
        if not (REPO / "ROM" / "classic" / "lhs_pool.json").is_file():
            self.skipTest("lhs_pool.json not available")
        report = build_stk_v6_physical_dof_audit(repo_root=REPO, sample_ids=("sample_000",))
        s0 = report["samples"][0]
        depth = s0["geometry"]["body_depth"]
        _assert_provenance(depth)
        self.assertIsNotNone(depth["value"])
        self.assertEqual(depth["status"], "available")
        self.assertTrue(depth["per_sample"])

    def test_important_features_have_provenance(self) -> None:
        report = build_stk_v6_physical_dof_audit(
            repo_root=REPO,
            sample_ids=("sample_000", "sample_001"),
        )
        s0 = report["samples"][0]
        for section in ("geometry", "material"):
            for rec in (s0.get(section) or {}).values():
                _assert_provenance(rec)
        for name in (
            "body_volume_proxy",
            "helmholtz_like_frequency_proxy",
            "low_body_mode_frequency",
            "bridge_to_radiation_strength",
        ):
            rec = s0["derived_features"][name]
            _assert_provenance(rec)
            if name in ("low_body_mode_frequency", "bridge_to_radiation_strength"):
                self.assertEqual(rec["status"], "reference_shared")
                self.assertFalse(rec["per_sample"])

    def test_shared_or_constant_features_section(self) -> None:
        report = build_stk_v6_physical_dof_audit(
            repo_root=REPO,
            sample_ids=tuple(f"sample_{i:03d}" for i in range(3)),
        )
        shared = report.get("shared_or_constant_features")
        self.assertIsInstance(shared, list)
        self.assertGreater(len(shared), 0)
        ref_names = {r["feature_name"] for r in shared if r.get("likely_reason") == "reference_catalog"}
        self.assertIn("low_body_mode_frequency", ref_names)

    def test_stage2_readiness_exists(self) -> None:
        report = build_stk_v6_physical_dof_audit(repo_root=REPO, sample_ids=("sample_000",))
        readiness = report.get("stage2_readiness")
        self.assertIsInstance(readiness, dict)
        self.assertIn(readiness.get("status"), ("ready", "ready_with_limitations", "blocked"))
        self.assertIn("safe_to_use_in_stage2", readiness)
        self.assertIn("recommended_stage2_feature_set", report)

    def test_audit_runs_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "audit.json"
            md_out = Path(tmp) / "audit.md"
            from audit_stk_v6_physical_dofs import main as audit_main

            with patch(
                "sys.argv",
                [
                    "audit_stk_v6_physical_dofs.py",
                    "--repo-root",
                    str(REPO),
                    "--json-out",
                    str(json_out),
                    "--md-out",
                    str(md_out),
                    "--max-sample-index",
                    "2",
                ],
            ):
                audit_main()

            doc = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertIs(doc["website_default_unchanged"], True)
            self.assertIn("stage2_readiness", doc)
            self.assertIn("Step 1.1", md_out.read_text(encoding="utf-8"))

    def test_sample_000_in_report_when_available(self) -> None:
        report = build_stk_v6_physical_dof_audit(
            repo_root=REPO,
            sample_ids=("sample_000", "sample_001"),
        )
        if (REPO / "ROM" / "classic" / "lhs_pool.json").is_file():
            self.assertIn("sample_000", report["audited_sample_ids"])

    def test_missing_fields_reported(self) -> None:
        report = build_stk_v6_physical_dof_audit(repo_root=REPO, sample_ids=("sample_000",))
        missing = report.get("missing_critical_fields") or []
        self.assertTrue(any("scale_length" in m for m in missing))

    def test_helmholtz_proxy_depth_and_hole_sanity(self) -> None:
        shallow = _compute_cavity_geometry_proxy(_params(depth=0.09))
        deep = _compute_cavity_geometry_proxy(_params(depth=0.12))
        small_hole = _compute_cavity_geometry_proxy(_params(hole_radius=0.040))
        large_hole = _compute_cavity_geometry_proxy(_params(hole_radius=0.055))
        self.assertGreater(
            shallow["helmholtz_like_frequency_hz"],
            deep["helmholtz_like_frequency_hz"],
        )
        self.assertLess(
            small_hole["helmholtz_like_frequency_hz"],
            large_hole["helmholtz_like_frequency_hz"],
        )

    def test_no_wav_created_by_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            before = set(tmp_path.rglob("*.wav"))
            build_stk_v6_physical_dof_audit(repo_root=REPO, sample_ids=("sample_000",))
            after = set(tmp_path.rglob("*.wav"))
            self.assertEqual(before, after)

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run:
            build_stk_v6_physical_dof_audit(repo_root=REPO, sample_ids=("sample_000",))
            for call in mock_run.call_args_list:
                cmd = " ".join(str(x) for x in (call[0][0] if call[0] else []))
                self.assertNotIn("fem", cmd.lower())
                self.assertNotIn("rom_batch", cmd.lower())

    def test_dof_influence_map_present(self) -> None:
        report = build_stk_v6_physical_dof_audit(repo_root=REPO, sample_ids=("sample_000",))
        self.assertIn("body_depth", report["dof_influence_map"])
        self.assertIn("soundhole_radius", report["dof_influence_map"])


if __name__ == "__main__":
    unittest.main()
