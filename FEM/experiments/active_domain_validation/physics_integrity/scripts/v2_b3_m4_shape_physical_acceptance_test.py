#!/usr/bin/env python3
"""Tests for shape-aware physical validation profiles and evaluator."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = SCRIPT_DIR.parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))

from m4_shape_context import resolve_shape_context  # noqa: E402
from m4_shape_validation_profile import (  # noqa: E402
    CLASSIC_LOCKED_PROFILE_ID,
    CLASSIC_LEGACY_PROFILE_ALIAS,
    CLASSIC_REFERENCE_BASELINE,
    classic_locked_profile_snapshot,
    register_custom_shape_validation_profile,
    resolve_shape_validation_profile,
)
from evaluate_shape_physical_acceptance import (  # noqa: E402
    ACCEPTANCE_JSON_REL,
    ACCEPTANCE_MD_REL,
    evaluate_shape_physical_acceptance,
    evaluate_numerical_acceptance,
    write_shape_physical_acceptance,
)
from collect_shape_validation_baseline import collect_shape_validation_baseline  # noqa: E402
from v2_b3_m4_freeze_first_e2e_run import AGG_STATUS_PASS  # noqa: E402
from v2_b3_m4_production_freeze import TERMINAL_PRODUCTION_COMPLETED  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_box_like_run(run_root: Path) -> None:
    sample_id = "box_sample_000"
    run_id = "box_sample_000_box_fom_v1"
    write_json_atomic(
        run_root / "sample" / "sample_input.json",
        {
            "sample_id": sample_id,
            "shape_name": "box",
            "geometry_shape_type": "Box",
            "shape_context": {
                "shape_name": "box",
                "geometry_shape_type": "Box",
                "shape_validation_profile_id": "box_body_plausibility_v1",
            },
        },
    )
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {"terminal_status": TERMINAL_PRODUCTION_COMPLETED, "sample_id": sample_id, "run_id": run_id},
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
        },
    )
    write_json_atomic(
        run_root / "aggregation" / "modes_summary.json",
        {
            "deduped_mode_count": 9,
            "raw_mode_count": 10,
            "participation_computed_count": 9,
            "bridge_coupling_available_count": 2,
            "mic_proxy_available_count": 9,
            "dominant_region_counts": {"air": 5, "back": 3, "top": 1},
            "coupling_class_counts": {"air_dominated": 5, "back_dominated": 3, "mixed": 1},
            "share_summary": {
                "top_share": {"count": 9, "median": 0.08, "mean": 0.1},
                "back_share": {"count": 9, "median": 0.35, "mean": 0.32},
                "air_share": {"count": 9, "median": 0.45, "mean": 0.42},
            },
            "radiation_proxy_summary": {"available_count": 9},
            "audio_coupling_summary": {"bridge_excitation_coupling_summary": {"median": 0.02}},
        },
    )
    catalog_lines = []
    for i in range(9):
        catalog_lines.append(
            {
                "mode_index": i,
                "frequency_hz": 120.0 + i * 15.0,
                "top_share": 0.08,
                "back_share": 0.35,
                "air_share": 0.45,
                "mic_output_proxy": 0.01,
                "radiation_proxy": 0.02,
                "participation_status": "computed",
            }
        )
    deduped = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    deduped.parent.mkdir(parents=True, exist_ok=True)
    deduped.write_text("\n".join(json.dumps(row) for row in catalog_lines) + "\n", encoding="utf-8")


    deduped.write_text("\n".join(json.dumps(row) for row in catalog_lines) + "\n", encoding="utf-8")


def _write_minimal_classic_run(run_root: Path) -> None:
    sample_id = "sample_000"
    run_id = "sample_000_smoke"
    write_json_atomic(
        run_root / "sample" / "sample_input.json",
        {
            "sample_id": sample_id,
            "shape_name": "classic",
            "geometry_shape_type": "Classical",
        },
    )
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {"terminal_status": TERMINAL_PRODUCTION_COMPLETED, "sample_id": sample_id, "run_id": run_id},
    )
    write_json_atomic(
        run_root / "aggregation" / "aggregation_result.json",
        {
            "status": AGG_STATUS_PASS,
            "final_aggregation_ready": True,
            "sample_id": sample_id,
            "run_id": run_id,
            "planned_chunk_count": 1,
            "completed_chunk_count": 1,
            "failed_chunk_count": 0,
            "missing_chunk_count": 0,
            "raw_mode_count": 10,
            "deduped_mode_count": 10,
        },
    )
    write_json_atomic(
        run_root / "aggregation" / "modes_summary.json",
        {
            "deduped_mode_count": 10,
            "participation_computed_count": 10,
            "bridge_coupling_available_count": 8,
            "mic_proxy_available_count": 10,
            "dominant_region_counts": {"top": 4, "back": 3, "air": 3},
            "share_summary": {
                "top_share": {"count": 10, "median": 0.22},
                "back_share": {"count": 10, "median": 0.28},
                "air_share": {"count": 10, "median": 0.25},
            },
        },
    )
    catalog = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "\n".join(
            json.dumps({"mode_index": i, "frequency_hz": 100.0 + i, "participation_status": "computed"})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )


def test_classic_resolves_locked_reference_profile() -> None:
    prof = resolve_shape_validation_profile("classic")
    assert prof.profile_id == CLASSIC_LOCKED_PROFILE_ID
    assert prof.locked is True
    assert prof.deduped_mode_count_min == 8
    ctx = resolve_shape_context("classic")
    assert ctx.geometry_shape_type == "Classical"
    assert ctx.shape_validation_profile_id == CLASSIC_LOCKED_PROFILE_ID
    assert ctx.physical_acceptance_profile["profile_type"] == "classical_reference"
    assert ctx.physical_acceptance_profile.get("locked") is True
    assert prof.to_dict().get("reference_baseline") == CLASSIC_REFERENCE_BASELINE


def test_classic_locked_profile_reports_67_sim_reference_baseline() -> None:
    snapshot = classic_locked_profile_snapshot()
    assert snapshot["reference_baseline"] == CLASSIC_REFERENCE_BASELINE
    assert snapshot["reference_baseline"] == "classic_lhs_67_simulations"
    prof = resolve_shape_validation_profile("classic")
    assert prof.to_dict().get("reference_baseline") == "classic_lhs_67_simulations"


def test_classic_legacy_alias_resolves_to_locked() -> None:
    prof = resolve_shape_validation_profile("classic", profile_id=CLASSIC_LEGACY_PROFILE_ALIAS)
    assert prof.profile_id == CLASSIC_LOCKED_PROFILE_ID
    assert prof.deduped_mode_count_min == 8


def test_classic_locked_profile_immune_to_config_drift() -> None:
    snapshot = classic_locked_profile_snapshot()
    import m4_shape_validation_profile as prof_mod  # noqa: WPS433

    poisoned = {
        "profiles": {
            CLASSIC_LOCKED_PROFILE_ID: {"mode_count_min": 2, "deduped_mode_count_min": 2},
        },
        "shape_defaults": {"classic": CLASSIC_LOCKED_PROFILE_ID},
    }
    with mock.patch.object(prof_mod, "_load_profiles_config", return_value=poisoned):
        prof = resolve_shape_validation_profile("classic")
    assert prof.deduped_mode_count_min == snapshot["deduped_mode_count_min"] == 8
    assert prof.profile_id == CLASSIC_LOCKED_PROFILE_ID


def test_box_profile_does_not_alter_classic_resolution() -> None:
    classic = resolve_shape_validation_profile("classic")
    box = resolve_shape_validation_profile("box")
    assert classic.profile_id != box.profile_id
    assert classic.deduped_mode_count_min == 8
    assert box.deduped_mode_count_min == 6
    assert box.profile_type == "shape_relative_body_validation"


def test_validator_writes_only_validation_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_classic_run(run_root)
        before = {
            rel: (run_root / rel).read_bytes()
            for rel in (
                "aggregation/aggregation_result.json",
                "aggregation/modes_summary.json",
                "aggregation/modes_catalog_deduped.jsonl",
                "pipeline_run_manifest.json",
                "sample/sample_input.json",
            )
        }
        report = evaluate_shape_physical_acceptance(run_root=run_root, shape_key="classic")
        write_shape_physical_acceptance(run_root, report)
        after = {
            rel: (run_root / rel).read_bytes()
            for rel in before
        }
        assert before == after
        assert (run_root / ACCEPTANCE_JSON_REL).is_file()
        assert (run_root / ACCEPTANCE_MD_REL).is_file()
        assert report["profile_id"] == CLASSIC_LOCKED_PROFILE_ID
        assert report["blocks_production"] is False


def test_dry_run_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_classic_run(run_root)
        from evaluate_shape_physical_acceptance import main as eval_main  # noqa: WPS433

        rc = eval_main(["--run-dir", str(run_root), "--shape", "classic", "--dry-run"])
        assert rc in (0, 1)
        assert not (run_root / "validation").exists() or not list((run_root / "validation").iterdir())


def test_box_resolves_box_profile() -> None:
    prof = resolve_shape_validation_profile("box")
    assert prof.profile_id == "box_body_plausibility_v1"
    assert prof.allow_worker_pass_with_warning_after_aggregation is True


def test_acoustic_resolves_acoustic_profile() -> None:
    prof = resolve_shape_validation_profile("acoustic")
    assert prof.profile_id == "acoustic_guitar_reference_v1"


def test_future_shape_custom_profile() -> None:
    prof = register_custom_shape_validation_profile(
        "wedge",
        {
            "profile_id": "wedge_body_v1",
            "profile_type": "shape_relative_body_validation",
            "deduped_mode_count_min": 4,
        },
    )
    assert prof.shape_name == "wedge"
    assert prof.profile_id == "wedge_body_v1"
    assert prof.deduped_mode_count_min == 4


def test_box_metrics_pass_or_warn_under_box_profile() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_box_like_run(run_root)
        report = evaluate_shape_physical_acceptance(run_root=run_root, shape_key="box")
        assert report["pipeline_integrity_pass"] is True
        assert report["numerical_acceptance_pass"] is True
        assert report["status"] in {"PASS", "PASS_WITH_WARNING"}
        assert report["musical_usefulness_status"] == "NOT_EVALUATED_SINGLE_SAMPLE"
        assert report["blocks_production"] is False


def test_same_box_metrics_stricter_under_classic_profile() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_box_like_run(run_root)
        box_report = evaluate_shape_physical_acceptance(
            run_root=run_root,
            profile=resolve_shape_validation_profile("box"),
        )
        classic_report = evaluate_shape_physical_acceptance(
            run_root=run_root,
            profile=resolve_shape_validation_profile("classic"),
        )
        assert box_report["shape_physical_plausibility_pass"] is True
        assert classic_report["shape_physical_plausibility_pass"] is False
        assert len(classic_report["layer_failures"]["shape_physical_plausibility"]) >= len(
            box_report["layer_failures"]["shape_physical_plausibility"]
        )


def test_missing_proxies_warn_not_always_fail_box() -> None:
    profile = resolve_shape_validation_profile("box")
    metrics = {
        "deduped_mode_count": 9,
        "participation_computed_count": 0,
        "mic_output_proxy_stats": {"mic_proxy_available_count": 0},
        "dominant_region_counts": {"air": 4, "top": 5},
        "top_share_stats": {"median": 0.2},
        "air_share_stats": {"median": 0.3},
        "bridge_excitation_coupling_stats": {"bridge_coupling_available_count": 3},
    }
    ok, failures, warnings = evaluate_numerical_acceptance(
        profile=profile,
        metrics=metrics,
        catalog_rows=[{"frequency_hz": 100.0}],
    )
    assert ok is True
    assert failures == []
    assert warnings


def test_baseline_insufficient_sample_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "runs"
        box_run = run_root / "guitars" / "box_sample_000" / "runs" / "box_sample_000_box_fom_v1"
        _write_box_like_run(box_run)
        from evaluate_shape_physical_acceptance import write_shape_physical_acceptance  # noqa: WPS433

        report = evaluate_shape_physical_acceptance(run_root=box_run, shape_key="box")
        write_shape_physical_acceptance(box_run, report)
        baseline = collect_shape_validation_baseline(
            shape="box",
            runs_root=run_root,
            shared_root=None,
        )
        assert baseline["baseline_status"] == "INSUFFICIENT_SAMPLE_COUNT"
        assert baseline["recommended_min_samples"] == 5
        assert baseline["sample_count"] == 1


def main() -> int:
    tests = [
        test_classic_resolves_locked_reference_profile,
        test_classic_locked_profile_reports_67_sim_reference_baseline,
        test_classic_legacy_alias_resolves_to_locked,
        test_classic_locked_profile_immune_to_config_drift,
        test_box_profile_does_not_alter_classic_resolution,
        test_validator_writes_only_validation_artifacts,
        test_dry_run_writes_nothing,
        test_box_resolves_box_profile,
        test_acoustic_resolves_acoustic_profile,
        test_future_shape_custom_profile,
        test_box_metrics_pass_or_warn_under_box_profile,
        test_same_box_metrics_stricter_under_classic_profile,
        test_missing_proxies_warn_not_always_fail_box,
        test_baseline_insufficient_sample_count,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_shape_physical_acceptance] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
