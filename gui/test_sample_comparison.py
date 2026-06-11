#!/usr/bin/env python3
"""Sample comparison utility tests (no FEM, no Streamlit)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from build_sample_comparison import (  # noqa: E402
    build_sample_comparisons,
    load_lhs_sample_entries,
    synthetic_modal_for_sample,
)


class SampleComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manifest_ordering_deterministic(self) -> None:
        samples = [
            {"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": {}}
            for i in range(4)
        ]
        manifest = build_sample_comparisons(
            repo_root=REPO,
            out_dir=self.out_dir,
            samples=samples,
            notes=(("A2", 110.0),),
            duration_s=0.12,
            silence_s=0.05,
            use_surrogate=False,
        )
        self.assertEqual(manifest["sample_count"], 4)
        segs = manifest["notes"][0]["segments"]
        self.assertEqual([s["sample_id"] for s in segs], [f"sample_{i:03d}" for i in range(4)])
        self.assertEqual(segs[0]["segment_number"], 1)
        self.assertTrue((self.out_dir / "A2_26_guitars.wav").is_file())
        doc = json.loads((self.out_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], "sample_comparison_v1")

    def test_synthetic_modal_varies_by_sample(self) -> None:
        a = synthetic_modal_for_sample("sample_000")
        b = synthetic_modal_for_sample("sample_025")
        f0_a = float(a["predicted_modes"][0]["frequency_hz"])
        f0_b = float(b["predicted_modes"][0]["frequency_hz"])
        self.assertNotAlmostEqual(f0_a, f0_b, places=2)

    def test_load_lhs_entries_capped(self) -> None:
        rows = load_lhs_sample_entries(REPO, max_samples=26)
        if rows:
            self.assertLessEqual(len(rows), 26)
            self.assertEqual(rows[0]["sample_id"], "sample_000")


if __name__ == "__main__":
    unittest.main()
