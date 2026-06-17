#!/usr/bin/env python3
"""Tests for freeze eligibility when workers report PASS_WITH_WARNING."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_production_contracts import (  # noqa: E402
    PRODUCTION_MIC_METHOD,
    aggregation_authoritative_for_freeze,
    evaluate_production_acceptance,
)
from v2_b3_m4_production_freeze import (  # noqa: E402
    FREEZE_FAILURE_REPORT,
    PRODUCTION_FREEZE_MANIFEST,
    TERMINAL_PRODUCTION_COMPLETED,
    production_freeze_complete,
    replay_production_freeze,
)
from v2_b3_m4_mesh_profile_lib import DATASET_VERSION_REFERENCE  # noqa: E402
from v2_b3_m4_production_freeze_test import _write_minimal_production_aggregated_run  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _enrich_strict_production_fixtures(run_root: Path) -> None:
    """Minimal scout/catalog/built patches so strict production acceptance can pass in tests."""
    from test_m4_scout_intrinsic_coverage import _intrinsic_density_payload, _rich_freqs  # noqa: WPS433

    sample_id = "sample_001"
    built_path = run_root / "lprod" / "checkpoint" / "built_metadata.json"
    built_doc = json.loads(built_path.read_text(encoding="utf-8"))
    built_doc.update(
        {
            "dataset_version": DATASET_VERSION_REFERENCE,
            "operator_mesh_matches_generated": True,
            "generated_mesh_sha256": "abc123",
            "operator_mesh_file_used": (
                f"guitars/{sample_id}/runs/{run_root.name}/lprod/mesh/L_prod/{sample_id}.msh"
            ),
        }
    )
    write_json_atomic(built_path, built_doc)
    discovery = run_root / "scout" / "discovery"
    discovery.mkdir(parents=True, exist_ok=True)
    write_json_atomic(discovery / "density_result.json", _intrinsic_density_payload(freqs=_rich_freqs()))
    catalog_line = {
        "mic_output_method": PRODUCTION_MIC_METHOD,
        "frequency_hz": 100.0,
        "lambda_real": 39578.4,
        "convergence_status": "converged",
    }
    catalog = run_root / "aggregation" / "modes_catalog.jsonl"
    catalog.write_text(json.dumps(catalog_line) + "\n", encoding="utf-8")


def _write_warn_worker(run_root: Path, *, chunk_suffix: str = "02") -> str:
    sample_id = "sample_001"
    chunk_id = f"{sample_id}_chunk_{chunk_suffix}"
    chunk_dir = run_root / "worker_results" / chunk_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        chunk_dir / "worker_result.json",
        {
            "status": "PASS_WITH_WARNING",
            "exit_code": 0,
            "targets_attempted": 5,
            "targets_passed": 5,
            "warnings": ["nonfatal_worker_warning"],
        },
    )
    return chunk_id


def test_aggregation_pass_with_warn_workers_permits_freeze() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "sample_001_smoke"
        _write_minimal_production_aggregated_run(run_root)
        _enrich_strict_production_fixtures(run_root)
        warn_chunk = _write_warn_worker(run_root)
        agg_path = run_root / "aggregation" / "aggregation_result.json"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        agg["planned_chunk_count"] = 2
        agg["completed_chunk_count"] = 2
        write_json_atomic(agg_path, agg)
        write_json_atomic(
            run_root / "lprod" / "worker_chunk_plan.preview.json",
            {
                "chunks": [
                    {"chunk_id": "sample_001_chunk_01"},
                    {"chunk_id": warn_chunk},
                ]
            },
        )

        sample_input = json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8"))
        acceptance = evaluate_production_acceptance(run_root=run_root, sample_input=sample_input)
        assert acceptance["aggregation_freeze_eligible"] is True
        assert acceptance["acceptance_pass"] is True
        assert any("worker_pass_with_warning" in w for w in acceptance.get("acceptance_warnings") or [])

        rc, msg = replay_production_freeze(repo_root=Path(tmp), run_root=run_root, force=False)
        assert rc == 0, msg
        assert production_freeze_complete(run_root)
        freeze_doc = json.loads((run_root / "freeze" / PRODUCTION_FREEZE_MANIFEST).read_text(encoding="utf-8"))
        assert freeze_doc["production_acceptance_pass"] is True
        assert any("worker_pass_with_warning" in w for w in freeze_doc.get("acceptance_warnings") or [])
        assert (run_root / "freeze" / "physics_identity_manifest.json").is_file()
        pipeline = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        assert pipeline["terminal_status"] == TERMINAL_PRODUCTION_COMPLETED


def test_failed_chunks_block_freeze() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run"
        _write_minimal_production_aggregated_run(run_root)
        _enrich_strict_production_fixtures(run_root)
        agg_path = run_root / "aggregation" / "aggregation_result.json"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        agg["failed_chunk_count"] = 1
        agg["failed_chunks"] = ["sample_001_chunk_01"]
        write_json_atomic(agg_path, agg)

        ok, failures = aggregation_authoritative_for_freeze(run_root, agg=agg)
        assert ok is False
        assert any("aggregation_failed_chunks" in f for f in failures)

        sample_input = json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8"))
        acceptance = evaluate_production_acceptance(run_root=run_root, sample_input=sample_input)
        assert acceptance["acceptance_pass"] is False

        rc, _ = replay_production_freeze(repo_root=Path(tmp), run_root=run_root, force=False)
        assert rc == 2
        report = json.loads((run_root / "freeze" / FREEZE_FAILURE_REPORT).read_text(encoding="utf-8"))
        assert report["failure_reason"]
        assert not (run_root / "freeze" / PRODUCTION_FREEZE_MANIFEST).is_file()


def test_missing_chunks_block_freeze() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run"
        _write_minimal_production_aggregated_run(run_root)
        _enrich_strict_production_fixtures(run_root)
        agg_path = run_root / "aggregation" / "aggregation_result.json"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        agg["planned_chunk_count"] = 2
        agg["completed_chunk_count"] = 1
        agg["missing_chunk_count"] = 1
        agg["missing_chunks"] = ["sample_001_chunk_02"]
        write_json_atomic(agg_path, agg)

        ok, failures = aggregation_authoritative_for_freeze(run_root, agg=agg)
        assert ok is False
        assert any("aggregation_missing_chunks" in f or "aggregation_incomplete" in f for f in failures)


def test_warn_without_authoritative_aggregation_still_blocks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "run"
        _write_minimal_production_aggregated_run(run_root)
        _enrich_strict_production_fixtures(run_root)
        _write_warn_worker(run_root)
        agg_path = run_root / "aggregation" / "aggregation_result.json"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        agg["final_aggregation_ready"] = False
        write_json_atomic(agg_path, agg)

        sample_input = json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8"))
        acceptance = evaluate_production_acceptance(run_root=run_root, sample_input=sample_input)
        assert acceptance["acceptance_pass"] is False
        assert any("worker_pass_with_warning" in f for f in acceptance["failures"])


def main() -> int:
    tests = [
        test_aggregation_pass_with_warn_workers_permits_freeze,
        test_failed_chunks_block_freeze,
        test_missing_chunks_block_freeze,
        test_warn_without_authoritative_aggregation_still_blocks,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_freeze_gate] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
