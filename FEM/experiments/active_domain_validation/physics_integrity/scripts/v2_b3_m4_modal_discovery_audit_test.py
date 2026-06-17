#!/usr/bin/env python3
"""Tests for modal discovery audit (advisory; no heavy FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = SCRIPT_DIR.parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))

from evaluate_modal_discovery_audit import main as evaluate_main  # noqa: E402
from v2_b3_m4_freeze_first_e2e_run import AGG_STATUS_PASS  # noqa: E402
from v2_b3_m4_modal_discovery_audit_lib import (  # noqa: E402
    AUDIT_JSON_REL,
    AUDIT_MD_REL,
    TARGETS_PASSED_NOTE,
    build_modal_discovery_audit,
    classify_modal_discovery_issue,
    write_modal_discovery_audit,
)
from v2_b3_m4_target_candidate_audit_lib import (  # noqa: E402
    build_target_candidate_audit_row,
    write_target_candidate_audit_jsonl,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_box_like_run(
    run_root: Path,
    *,
    with_worker_results: bool = False,
    compacted: bool = False,
) -> None:
    sample_id = "box_sample_000"
    run_id = "box_sample_000_box_fom_v1"
    chunk_ids = [f"box_sample_000_chunk_{i:02d}" for i in range(1, 13)]
    chunk_details = []
    for i, cid in enumerate(chunk_ids):
        mode_count = 0 if i in (2, 5, 8) else 1
        chunk_details.append(
            {
                "chunk_id": cid,
                "classification": "completed",
                "worker_status": "PASS",
                "targets_attempted": 5,
                "targets_passed": 5,
                "mode_count": mode_count,
            }
        )

    write_json_atomic(
        run_root / "sample" / "sample_input.json",
        {"sample_id": sample_id, "shape_name": "box", "geometry_shape_type": "Box"},
    )
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {"sample_id": sample_id, "run_id": run_id},
    )
    write_json_atomic(
        run_root / "lprod" / "lprod_target_plan.json",
        {
            "target_count": 58,
            "frequency_range_hz": [80.0, 520.0],
            "target_metadata": [
                {"target_hz": 100.0 + i * 7.0, "zone_id": "ZONE_1_dense" if i % 3 == 0 else "ZONE_3_sparse"}
                for i in range(58)
            ],
        },
    )
    chunks = []
    for i, cid in enumerate(chunk_ids):
        chunks.append(
            {
                "chunk_id": cid,
                "targets_hz": [100.0 + i * 7.0 + j for j in range(5)],
                "target_windows_hz": [[95.0, 105.0]] * 5,
                "freq_range_hz": [90.0 + i * 7.0, 130.0 + i * 7.0],
            }
        )
    write_json_atomic(
        run_root / "lprod" / "worker_chunk_plan.preview.json",
        {"schema": "m4_worker_chunk_plan_v1", "sample_id": sample_id, "run_id": run_id, "chunks": chunks},
    )
    write_json_atomic(
        run_root / "aggregation" / "aggregation_result.json",
        {
            "status": AGG_STATUS_PASS,
            "final_aggregation_ready": True,
            "sample_id": sample_id,
            "run_id": run_id,
            "planned_chunk_count": 12,
            "completed_chunk_count": 12,
            "failed_chunk_count": 0,
            "missing_chunk_count": 0,
            "raw_mode_count": 10,
            "deduped_mode_count": 9,
            "total_targets_attempted": 58,
            "total_targets_passed": 58,
            "frequency_range_hz": [80.0, 520.0],
            "chunk_details": chunk_details,
        },
    )

    if with_worker_results:
        for detail in chunk_details:
            cid = detail["chunk_id"]
            chunk_dir = run_root / "worker_results" / cid
            chunk_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                chunk_dir / "worker_result.json",
                {
                    "chunk_id": cid,
                    "status": "PASS",
                    "targets_attempted": detail["targets_attempted"],
                    "targets_passed": detail["targets_passed"],
                    "accepted_mode_records": [],
                },
            )
            if not compacted:
                write_json_atomic(
                    chunk_dir / "solver_result.json",
                    {
                        "targets": [
                            {
                                "target_frequency_hz": 188.0,
                                "status": "PASS",
                                "factor_solver": "mkl_pardiso",
                                "nev": 32,
                                "converged_mode_count": 2,
                                "accepted_mode_count_in_interval": 0,
                                "candidate_rejection_tally": {
                                    "support_participation_fail": 2,
                                },
                            }
                        ]
                    },
                )

    if compacted:
        write_json_atomic(
            run_root / "compaction" / "compaction_manifest.json",
            {"schema": "m4_run_compaction_manifest_v1", "status": "PASS"},
        )


def test_audit_reads_aggregation_and_chunk_plan():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        _write_box_like_run(run_root)
        report = build_modal_discovery_audit(run_root=run_root, shape_name="box")
        assert report["target_count"] == 58
        assert report["raw_mode_count"] == 10
        assert report["deduped_mode_count"] == 9
        assert report["chunk_count"] == 12
        assert len(report["chunks"]) == 12


def test_audit_reports_empty_chunks():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        _write_box_like_run(run_root)
        report = build_modal_discovery_audit(run_root=run_root, shape_name="box")
        assert report["empty_chunk_count"] == 3
        assert "box_sample_000_chunk_03" in report["empty_chunks"]


def test_audit_distinguishes_target_pass_from_mode_discovery():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        _write_box_like_run(run_root, with_worker_results=True)
        report = build_modal_discovery_audit(run_root=run_root, shape_name="box")
        assert "numerical target solve pass" in report["targets_passed_semantics"].lower()
        zero_mode_chunks = [c for c in report["chunks"] if c.get("all_targets_passed_but_zero_modes")]
        assert len(zero_mode_chunks) == 3


def test_audit_handles_missing_worker_results_after_compaction():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        _write_box_like_run(run_root, with_worker_results=True, compacted=True)
        report = build_modal_discovery_audit(run_root=run_root, shape_name="box")
        assert report["candidate_level_diagnostics_available"] is False
        assert report["classification"] == "WORKER_DIAGNOSTICS_MISSING"
        assert "target_candidate_audit.jsonl" in " ".join(report["missing_diagnostics"])


def test_audit_computes_ratios():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        _write_box_like_run(run_root)
        report = build_modal_discovery_audit(run_root=run_root, shape_name="box")
        assert abs(report["modes_per_target"] - 10 / 58) < 1e-6
        assert abs(report["modes_per_chunk"] - 10 / 12) < 1e-6
        assert report["dedup_removed_count"] == 1


def test_audit_does_not_require_heavy_worker_artifacts():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        _write_box_like_run(run_root)
        json_path, md_path, _ = write_modal_discovery_audit(run_root=run_root, shape_name="box")
        assert json_path.is_file()
        assert md_path.is_file()


def test_evaluate_cli_writes_validation_outputs():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        _write_box_like_run(run_root)
        rc = evaluate_main(["--run-dir", str(run_root), "--shape", "box"])
        assert rc == 0
        assert (run_root / AUDIT_JSON_REL).is_file()
        assert (run_root / AUDIT_MD_REL).is_file()
        doc = json.loads((run_root / AUDIT_JSON_REL).read_text(encoding="utf-8"))
        assert doc["schema"] == "m4_modal_discovery_audit_v1"


def test_audit_does_not_touch_classic_outputs():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        sample_id = "sample_000"
        agg_path = run_root / "aggregation" / "aggregation_result.json"
        write_json_atomic(
            run_root / "aggregation" / "aggregation_result.json",
            {"sample_id": sample_id, "raw_mode_count": 40, "deduped_mode_count": 38, "chunk_details": []},
        )
        write_json_atomic(
            run_root / "lprod" / "worker_chunk_plan.preview.json",
            {"chunks": [], "sample_id": sample_id},
        )
        before = agg_path.read_text(encoding="utf-8")
        write_modal_discovery_audit(run_root=run_root, shape_name="classic")
        after = agg_path.read_text(encoding="utf-8")
        assert before == after
        assert (run_root / AUDIT_JSON_REL).is_file()


def test_target_candidate_audit_row_builder():
    row = build_target_candidate_audit_row(
        chunk_id="box_sample_000_chunk_01",
        target_row={
            "target_frequency_hz": 188.0,
            "status": "PASS",
            "factor_solver": "mkl_pardiso",
            "nev": 32,
            "converged_mode_count": 3,
            "accepted_mode_count_in_interval": 1,
            "candidate_rejection_tally": {"outside_acceptance_window": 2},
            "accepted_frequencies_hz": [187.5],
        },
        target_meta={"window_hz": [185.0, 191.0], "zone_id": "ZONE_1_dense"},
    )
    assert row["chunk_id"] == "box_sample_000_chunk_01"
    assert row["accepted_mode_count"] == 1
    assert row["rejection_reasons"]["outside_acceptance_window"] == 2


def test_classify_dedup_not_aggressive_for_box():
    cls = classify_modal_discovery_issue(
        target_count=58,
        raw_mode_count=10,
        deduped_mode_count=9,
        dedup_removed=1,
        modes_per_target=10 / 58,
        empty_chunk_count=3,
        chunk_count=12,
        all_targets_passed_zero_mode_chunks=3,
        candidate_level_diagnostics_available=False,
        frequency_range_hz=[80.0, 520.0],
        zone_contribution={"ZONE_1_dense": 20},
        aggregate_rejection_tally={},
        avg_converged_per_target=None,
    )
    assert cls == "WORKER_DIAGNOSTICS_MISSING"


def test_candidate_audit_jsonl_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        chunk_dir = Path(td)
        write_target_candidate_audit_jsonl(
            chunk_dir,
            [
                build_target_candidate_audit_row(
                    chunk_id="c1",
                    target_row={"target_frequency_hz": 100.0, "status": "PASS", "accepted_mode_count_in_interval": 0},
                )
            ],
        )
        report_root = Path(td) / "run"
        _write_box_like_run(report_root, with_worker_results=True)
        chunk_dest = report_root / "worker_results" / "box_sample_000_chunk_01"
        chunk_dest.mkdir(parents=True, exist_ok=True)
        (chunk_dest / "target_candidate_audit.jsonl").write_text(
            (chunk_dir / "target_candidate_audit.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        report = build_modal_discovery_audit(run_root=report_root, shape_name="box")
        assert report["candidate_level_diagnostics_available"] is True


def main() -> int:
    tests = [
        test_audit_reads_aggregation_and_chunk_plan,
        test_audit_reports_empty_chunks,
        test_audit_distinguishes_target_pass_from_mode_discovery,
        test_audit_handles_missing_worker_results_after_compaction,
        test_audit_computes_ratios,
        test_audit_does_not_require_heavy_worker_artifacts,
        test_evaluate_cli_writes_validation_outputs,
        test_audit_does_not_touch_classic_outputs,
        test_target_candidate_audit_row_builder,
        test_classify_dedup_not_aggressive_for_box,
        test_candidate_audit_jsonl_roundtrip,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
