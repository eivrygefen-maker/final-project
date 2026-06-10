#!/usr/bin/env python3
"""Tests for direct M4 run numerical comparison (no preconditions)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_direct_run_compare_lib import (  # noqa: E402
    compare_runs_direct,
    greedy_monotonic_freq_pairs,
    render_markdown_direct,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_catalog(run_root: Path, rows: list[dict]) -> None:
    path = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _minimal_run(run_root: Path, *, rows: list[dict], worker_wall_s: float, peak_rss: int) -> None:
    _write_catalog(run_root, rows)
    write_json_atomic(
        run_root / "m4_sample_runtime_provenance.json",
        {
            "stage_wall_times_s": {"stage5_workers": worker_wall_s},
            "peak_rss_bytes_max_worker": peak_rss,
            "worker_resource_records": [{"peak_rss_bytes": peak_rss}],
            "workers_parallel_observed": 3,
        },
    )
    write_json_atomic(
        run_root / "freeze" / "physics_identity_manifest.json",
        {
            "active_dimension": 1000,
            "mesh_components": {"n_nodes": 50000, "n_tetra": 200000},
        },
    )


class DirectRunCompareTests(unittest.TestCase):
    def test_monotonic_frequency_matching(self) -> None:
        ref = [{"frequency_hz": 100.0}, {"frequency_hz": 200.0}, {"frequency_hz": 300.0}]
        rom = [{"frequency_hz": 101.0}, {"frequency_hz": 205.0}]
        pairs = greedy_monotonic_freq_pairs(ref, rom, max_distance_hz=5.0)
        self.assertEqual(pairs, [(0, 0), (1, 1)])
        far = greedy_monotonic_freq_pairs(ref, rom, max_distance_hz=2.0)
        self.assertEqual(far, [(0, 0)])

    def test_missing_optional_fields_still_produces_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_root = root / "ref"
            rom_root = root / "rom"
            _minimal_run(
                ref_root,
                rows=[{"frequency_hz": 100.0, "coupling_class": "bridge"}],
                worker_wall_s=1200.0,
                peak_rss=6 * 1024**3,
            )
            _write_catalog(rom_root, [{"frequency_hz": 100.5}])
            report = compare_runs_direct(reference_run=ref_root, candidate_run=rom_root)
            self.assertTrue(report.get("comparison_executed"))
            self.assertEqual((report.get("frequency_matching") or {}).get("matched_count"), 1)
            self.assertEqual((report.get("performance") or {}).get("reference_worker_phase_s"), 1200.0)
            self.assertIsNone((report.get("performance") or {}).get("rom_worker_phase_s"))
            self.assertEqual((report.get("participation") or {}).get("status"), "UNAVAILABLE")
            md = render_markdown_direct(report)
            self.assertIn("Practical conclusion", md)
            self.assertIn("recommendation", md.lower())

    def test_frequency_order_mismatch_reduces_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_root = root / "ref"
            rom_root = root / "rom"
            _minimal_run(
                ref_root,
                rows=[{"frequency_hz": 100.0}, {"frequency_hz": 200.0}],
                worker_wall_s=1000.0,
                peak_rss=1,
            )
            _minimal_run(
                rom_root,
                rows=[{"frequency_hz": 102.0}, {"frequency_hz": 250.0}],
                worker_wall_s=500.0,
                peak_rss=1,
            )
            report = compare_runs_direct(reference_run=ref_root, candidate_run=rom_root, match_tolerance_hz=5.0)
            freq = report.get("frequency_matching") or {}
            self.assertEqual(freq.get("matched_count"), 1)
            self.assertEqual(freq.get("unmatched_reference_count"), 1)
            conclusion = report.get("practical_conclusion") or {}
            self.assertLess(conclusion.get("information_retention_per_sample") or 1.0, 1.0)
            self.assertGreater(conclusion.get("throughput_gain") or 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
