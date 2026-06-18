#!/usr/bin/env python3
"""Tests for BOX raw modal discovery diagnostic mode (no heavy FEM)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_box_raw_modal_discovery_lib import (  # noqa: E402
    ACCEPTED_FILTERED_CATALOG_VAL,
    RAW_SOLVER_CATALOG_AGG,
    RAW_SOLVER_CATALOG_VAL,
    UNFILTERED_CATALOG_VAL,
    box_raw_modal_discovery_enabled,
    build_catalog_row,
    build_raw_vs_filtered_analysis,
    merge_box_raw_catalogs_for_run,
    resolve_worker_shape_name,
    write_worker_diagnostic_from_solver_targets,
)
from v2_b3_m4_worker_run_lib import production_worker_subprocess_env  # noqa: E402
from v2_b3_m4_minimal_rom_compaction import collect_minimal_rom_deletable_paths  # noqa: E402
from v2_b3_m4_modal_discovery_audit_lib import build_modal_discovery_audit  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _synthetic_candidate(
    *,
    freq: float,
    target_hz: float,
    pass_filters: bool,
    reasons: list,
) -> dict:
    return {
        "frequency_hz": freq,
        "residual": 1.0e-6 if pass_filters else 0.1,
        "residual_status": "PASS" if pass_filters else "WARNING",
        "inside_target_window": pass_filters,
        "target_window_hz": [target_hz - 5, target_hz + 5],
        "would_pass_normal_filters": pass_filters,
        "normal_filter_rejection_reasons": reasons,
        "passes_numerical_sanity": True,
        "bridge_excitation_coupling": None,
        "radiation_proxy": None,
        "mic_output_proxy": None,
        "dominant_region": None,
    }


def test_box_raw_enabled_only_for_box_with_env():
    old = os.environ.pop("BOX_RAW_MODAL_DISCOVERY", None)
    try:
        assert box_raw_modal_discovery_enabled(shape_name="box") is False
        os.environ["BOX_RAW_MODAL_DISCOVERY"] = "1"
        assert box_raw_modal_discovery_enabled(shape_name="box") is True
        assert box_raw_modal_discovery_enabled(shape_name="classic") is False
        assert box_raw_modal_discovery_enabled(shape_name="acoustic") is False
    finally:
        if old is not None:
            os.environ["BOX_RAW_MODAL_DISCOVERY"] = old
        else:
            os.environ.pop("BOX_RAW_MODAL_DISCOVERY", None)


def test_production_worker_env_propagates_box_raw_discovery_context():
    old_raw = os.environ.get("BOX_RAW_MODAL_DISCOVERY")
    old_shape = os.environ.get("SHAPE")
    try:
        os.environ["BOX_RAW_MODAL_DISCOVERY"] = "1"
        os.environ["SHAPE"] = "box"
        env = production_worker_subprocess_env(
            solver_python="/tmp/solver-mkl/bin/python",
            solver_venv="/tmp/solver-mkl",
        )
        assert env["BOX_RAW_MODAL_DISCOVERY"] == "1"
        assert env["SHAPE"] == "box"
    finally:
        if old_raw is not None:
            os.environ["BOX_RAW_MODAL_DISCOVERY"] = old_raw
        else:
            os.environ.pop("BOX_RAW_MODAL_DISCOVERY", None)
        if old_shape is not None:
            os.environ["SHAPE"] = old_shape
        else:
            os.environ.pop("SHAPE", None)


def test_resolve_worker_shape_from_sample_id():
    assert resolve_worker_shape_name({"sample_id": "box_sample_000"}) == "box"
    assert resolve_worker_shape_name({"sample_id": "sample_000"}) == "classic"


def test_catalog_row_includes_rejection_reasons():
    row = build_catalog_row(
        sample_id="box_sample_000",
        run_id="box_sample_000_box_fom_v1",
        shape="box",
        chunk_id="chunk_01",
        target_hz=240.0,
        target_row={"factor_solver": "mkl_pardiso", "nev": 12},
        candidate=_synthetic_candidate(
            freq=239.5,
            target_hz=240.0,
            pass_filters=False,
            reasons=["outside_acceptance_window", "support_participation_fail"],
        ),
        candidate_rank=0,
    )
    assert row["would_pass_normal_filters"] is False
    assert "outside_acceptance_window" in row["normal_filter_rejection_reasons"]
    assert row["bridge_excitation_proxy"] is None


def test_worker_diagnostic_writes_rejected_candidates():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        chunk_targets = {
            "sample_id": "box_sample_000",
            "run_id": "box_sample_000_box_fom_v1",
            "chunk_id": "box_sample_000_chunk_01",
        }
        solver_targets = [
            {
                "target_frequency_hz": 240.0,
                "factor_solver": "mkl_pardiso",
                "nev": 12,
                "diagnostic_candidates": [
                    _synthetic_candidate(
                        freq=239.5,
                        target_hz=240.0,
                        pass_filters=False,
                        reasons=["outside_acceptance_window"],
                    ),
                    _synthetic_candidate(
                        freq=240.1,
                        target_hz=240.0,
                        pass_filters=True,
                        reasons=[],
                    ),
                ],
            }
        ]
        n = write_worker_diagnostic_from_solver_targets(
            output_dir=out,
            chunk_targets=chunk_targets,
            solver_targets=solver_targets,
            shape_name="box",
        )
        assert n == 2
        lines = (out / "raw_modal_diagnostic.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        rejected = json.loads(lines[0])
        assert rejected["would_pass_normal_filters"] is False


def test_merge_creates_raw_and_unfiltered_catalogs():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        chunk_id = "box_sample_000_chunk_01"
        chunk_dir = run_root / "worker_results" / chunk_id
        write_worker_diagnostic_from_solver_targets(
            output_dir=chunk_dir,
            chunk_targets={
                "sample_id": "box_sample_000",
                "run_id": "r1",
                "chunk_id": chunk_id,
            },
            solver_targets=[
                {
                    "target_frequency_hz": 100.0,
                    "factor_solver": "mkl_pardiso",
                    "nev": 12,
                    "diagnostic_candidates": [
                        _synthetic_candidate(
                            freq=100.1,
                            target_hz=100.0,
                            pass_filters=False,
                            reasons=["residual_too_large"],
                        ),
                        _synthetic_candidate(
                            freq=100.2,
                            target_hz=100.0,
                            pass_filters=True,
                            reasons=[],
                        ),
                    ],
                }
            ],
            shape_name="box",
        )
        meta = merge_box_raw_catalogs_for_run(
            run_root,
            sample_id="box_sample_000",
            run_id="r1",
            shape_name="box",
            chunk_ids=[chunk_id],
            accepted_records=[{"chunk_id": chunk_id, "target_hz": 100.0, "frequency_hz": 100.2}],
        )
        assert meta["raw_solver_candidate_count"] == 2
        assert meta["unfiltered_mode_count"] == 2
        assert (run_root / RAW_SOLVER_CATALOG_AGG).is_file()
        assert (run_root / RAW_SOLVER_CATALOG_VAL).is_file()
        assert (run_root / UNFILTERED_CATALOG_VAL).is_file()
        assert (run_root / ACCEPTED_FILTERED_CATALOG_VAL).is_file()


def test_raw_catalogs_not_in_compaction_deletable():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        chunk_id = "c1"
        chunk_dir = run_root / "worker_results" / chunk_id
        write_worker_diagnostic_from_solver_targets(
            output_dir=chunk_dir,
            chunk_targets={"sample_id": "box_sample_000", "run_id": "r1", "chunk_id": chunk_id},
            solver_targets=[
                {
                    "target_frequency_hz": 200.0,
                    "nev": 12,
                    "diagnostic_candidates": [
                        _synthetic_candidate(
                            freq=200.1,
                            target_hz=200.0,
                            pass_filters=True,
                            reasons=[],
                        )
                    ],
                }
            ],
            shape_name="box",
        )
        merge_box_raw_catalogs_for_run(
            run_root,
            sample_id="box_sample_000",
            run_id="r1",
            shape_name="box",
            chunk_ids=[chunk_id],
            accepted_records=[],
        )
        deletable = {p.relative_to(run_root).as_posix() for p in collect_minimal_rom_deletable_paths(run_root)}
        assert RAW_SOLVER_CATALOG_VAL not in deletable
        assert UNFILTERED_CATALOG_VAL not in deletable


def test_raw_vs_filtered_classifies_filter_rejection():
    raw_rows = []
    for i in range(30):
        raw_rows.append(
            build_catalog_row(
                sample_id="box_sample_000",
                run_id="r1",
                shape="box",
                chunk_id="c1",
                target_hz=100.0 + i,
                target_row={"nev": 12},
                candidate=_synthetic_candidate(
                    freq=100.0 + i + 0.1,
                    target_hz=100.0 + i,
                    pass_filters=False,
                    reasons=["outside_acceptance_window"],
                ),
                candidate_rank=0,
            )
        )
    accepted = [r for r in raw_rows[:3]]
    for r in accepted:
        r["would_pass_normal_filters"] = True
        r["normal_filter_rejection_reasons"] = []
    analysis = build_raw_vs_filtered_analysis(
        raw_rows=raw_rows,
        unfiltered_rows=raw_rows,
        accepted_rows=accepted,
        deduped_mode_count=3,
    )
    assert analysis["loss_classification"] == "TARGET_WINDOW_TOO_STRICT"
    assert analysis["total_solver_candidates"] == 30
    assert analysis["total_rejected_by_normal_filters"] == 27


def test_raw_vs_filtered_classifies_solver_too_few():
    raw_rows = [
        build_catalog_row(
            sample_id="box_sample_000",
            run_id="r1",
            shape="box",
            chunk_id="c1",
            target_hz=100.0,
            target_row={"nev": 12},
            candidate=_synthetic_candidate(
                freq=100.1,
                target_hz=100.0,
                pass_filters=True,
                reasons=[],
            ),
            candidate_rank=0,
        )
    ]
    analysis = build_raw_vs_filtered_analysis(
        raw_rows=raw_rows,
        unfiltered_rows=raw_rows,
        accepted_rows=raw_rows,
        deduped_mode_count=1,
    )
    assert analysis["loss_classification"] == "SOLVER_RETURNS_TOO_FEW_CANDIDATES"


def test_modal_audit_includes_raw_vs_filtered_section():
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td)
        write_json_atomic(
            run_root / "aggregation" / "aggregation_result.json",
            {
                "status": "AGGREGATION_PASS",
                "sample_id": "box_sample_000",
                "run_id": "box_sample_000_box_fom_v1",
                "raw_mode_count": 3,
                "deduped_mode_count": 3,
                "chunk_details": [{"chunk_id": "c1", "mode_count": 3}],
            },
        )
        write_json_atomic(run_root / "lprod" / "worker_chunk_plan.preview.json", {"chunks": []})
        write_json_atomic(run_root / "lprod" / "lprod_target_plan.json", {"target_count": 58})
        raw_rows = []
        for i in range(20):
            raw_rows.append(
                {
                    "target_hz": 100.0 + i * 10,
                    "frequency_hz": 100.0 + i * 10 + 0.5,
                    "would_pass_normal_filters": i < 3,
                    "normal_filter_rejection_reasons": [] if i < 3 else ["outside_acceptance_window"],
                    "passes_numerical_sanity": True,
                }
            )
        write_json_atomic(
            run_root / RAW_SOLVER_CATALOG_VAL,
            {"schema": "m4_box_raw_modal_catalog_v1", "rows": raw_rows},
        )
        report = build_modal_discovery_audit(run_root=run_root, shape_name="box")
        assert report.get("raw_vs_filtered_analysis")
        assert report["raw_vs_filtered_analysis"]["total_solver_candidates"] == 20


def main() -> int:
    tests = [
        test_box_raw_enabled_only_for_box_with_env,
        test_resolve_worker_shape_from_sample_id,
        test_catalog_row_includes_rejection_reasons,
        test_worker_diagnostic_writes_rejected_candidates,
        test_merge_creates_raw_and_unfiltered_catalogs,
        test_raw_catalogs_not_in_compaction_deletable,
        test_raw_vs_filtered_classifies_filter_rejection,
        test_raw_vs_filtered_classifies_solver_too_few,
        test_modal_audit_includes_raw_vs_filtered_section,
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
