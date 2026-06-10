#!/usr/bin/env python3
"""Regression tests for M4 LHS batch sample outcome classification."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    AGG_PASS,
    TERMINAL_PRODUCTION_COMPLETED,
    classify_batch_sample_outcome,
)


def _complete_summary(*, terminal_status: str = TERMINAL_PRODUCTION_COMPLETED) -> dict:
    return {
        "terminal_status": terminal_status,
        "aggregation_status": AGG_PASS,
        "failed_chunks": 0,
        "missing_chunks": 0,
        "final_aggregation_ready": True,
    }


def _completed_cleanup_barrier() -> dict:
    return {
        "status": "completed",
        "forbidden_heavy_artifact_count": 0,
        "shared_sample_artifact_count": 0,
    }


class BatchSampleClassificationTests(unittest.TestCase):
    def test_rc0_completed_aggregation_pass_classifies_completed(self) -> None:
        outcome, err = classify_batch_sample_outcome(
            return_code=0,
            summary=_complete_summary(),
            cleanup_barrier=_completed_cleanup_barrier(),
            require_cleanup_barrier=True,
        )
        self.assertEqual(outcome, "pass")
        self.assertIsNone(err)

    def test_rc0_cleanup_failure_classifies_failed(self) -> None:
        outcome, err = classify_batch_sample_outcome(
            return_code=0,
            summary=_complete_summary(),
            cleanup_barrier={
                "status": "failed",
                "forbidden_heavy_artifact_count": 0,
                "shared_sample_artifact_count": 0,
            },
            require_cleanup_barrier=True,
        )
        self.assertEqual(outcome, "fail")
        self.assertIn("cleanup_barrier_status=failed", err or "")

    def test_nonzero_rc_classifies_failed(self) -> None:
        outcome, err = classify_batch_sample_outcome(
            return_code=1,
            summary=_complete_summary(),
            cleanup_barrier=_completed_cleanup_barrier(),
            require_cleanup_barrier=True,
        )
        self.assertEqual(outcome, "fail")
        self.assertIn("return_code=1", err or "")

    def test_rc0_incomplete_terminal_status_classifies_failed(self) -> None:
        outcome, err = classify_batch_sample_outcome(
            return_code=0,
            summary=_complete_summary(terminal_status="RUNNING"),
            cleanup_barrier=_completed_cleanup_barrier(),
            require_cleanup_barrier=True,
        )
        self.assertEqual(outcome, "fail")
        self.assertIn("terminal_status=RUNNING", err or "")


if __name__ == "__main__":
    unittest.main()
