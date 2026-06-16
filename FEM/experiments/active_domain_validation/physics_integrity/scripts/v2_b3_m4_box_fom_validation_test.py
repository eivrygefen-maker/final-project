#!/usr/bin/env python3
"""Tests for terminal status promotion, mesh inspection, full-clean reset, residue audit."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "FEM" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "FEM" / "scripts"))

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    CHECKPOINT_TERMINAL_READY,
    TERMINAL_E2E,
)
from v2_b3_m4_post_run_residue_audit import format_post_run_residue_audit_line, run_post_run_residue_audit  # noqa: E402
from v2_b3_m4_reuse_integrity_lib import assess_stages_with_integrity  # noqa: E402
from v2_b3_m4_terminal_status_lib import (  # noqa: E402
    TERMINAL_STATUS_INCONSISTENCY,
    check_terminal_status_consistency,
    promote_after_aggregation_pass,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from reset_m4_sample_state import full_clean_sample_run, reset_sample_run_state  # noqa: E402


def _write_aggregate_pass(run_root: Path) -> None:
    agg = run_root / "aggregation"
    agg.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        agg / "aggregation_result.json",
        {
            "status": AGG_STATUS_PASS,
            "final_aggregation_ready": True,
            "planned_chunk_count": 1,
            "completed_chunk_count": 1,
        },
    )
    (agg / "modes_catalog.jsonl").write_text('{"mode_index":0,"frequency_hz":200.0}\n', encoding="utf-8")


def test_promote_after_aggregation_updates_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        write_json_atomic(
            run_root / "pipeline_run_manifest.json",
            {"terminal_status": CHECKPOINT_TERMINAL_READY, "stages": {}},
        )
        _write_aggregate_pass(run_root)
        result = promote_after_aggregation_pass(run_root)
        assert result["promoted"] is True
        manifest = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["terminal_status"] == TERMINAL_E2E
        ok, errors = check_terminal_status_consistency(run_root)
        assert ok is True
        assert errors == []
        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["aggregate"]["pass"] is True
        assert stages["workers"]["reuse_status"] != "resume_possible" or stages["workers"]["pass"] is False


def test_inconsistent_terminal_detected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        write_json_atomic(
            run_root / "pipeline_run_manifest.json",
            {"terminal_status": CHECKPOINT_TERMINAL_READY},
        )
        _write_aggregate_pass(run_root)
        write_json_atomic(
            run_root / "pipeline_run_manifest.m4_4_full_aggregation_preview.json",
            {"terminal_status": TERMINAL_E2E},
        )
        ok, errors = check_terminal_status_consistency(run_root)
        assert ok is False
        assert any(TERMINAL_STATUS_INCONSISTENCY in e for e in errors)


def test_full_clean_removes_stale_reuse_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_root = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
            / "box_sample_000"
            / "runs"
            / "box_sample_000_box_fom_v1"
        )
        run_root.mkdir(parents=True)
        write_json_atomic(
            run_root / "pipeline_run_manifest.json",
            {"terminal_status": CHECKPOINT_TERMINAL_READY},
        )
        (run_root / "worker_results" / "c1").mkdir(parents=True)
        write_json_atomic(run_root / "worker_results" / "c1" / "worker_result.json", {"status": "PASS"})
        _write_aggregate_pass(run_root)
        (run_root / "lprod").mkdir()
        write_json_atomic(run_root / "lprod" / "lprod_execution_plan.json", {"schema": "x"})

        report = reset_sample_run_state(
            repo_root=repo_root,
            run_root=run_root,
            sample_id="box_sample_000",
            run_id="box_sample_000_box_fom_v1",
            mode="full-clean",
            reset_pool_status=False,
        )
        assert report["status"] == "PASS"
        assert not (run_root / "worker_results").exists()
        assert not (run_root / "aggregation").exists()
        manifest = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["terminal_status"] == "PLANNED"
        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["scout"]["pass"] is False


def test_post_run_residue_audit_counts_forbidden() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_root = repo_root / "run"
        run_root.mkdir()
        (run_root / "lprod" / "checkpoint").mkdir(parents=True)
        report = run_post_run_residue_audit(
            repo_root=repo_root,
            run_root=run_root,
            sample_id="box_sample_000",
            run_id="box_sample_000_box_fom_v1",
        )
        assert report["forbidden_heavy_artifact_count"] > 0
        line = format_post_run_residue_audit_line(report)
        assert "POST_RUN_RESIDUE_AUDIT" in line
        assert "forbidden=" in line


def _write_synthetic_mesh(path: Path, *, with_aperture: bool) -> None:
    import meshio

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    air_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    if with_aperture:
        tri = np.array([[0, 1, 2]], dtype=np.int64)
        tri_tags = np.array([2], dtype=np.int32)
    else:
        tri = np.array([[0, 1, 4]], dtype=np.int64)
        tri_tags = np.array([1], dtype=np.int32)
    mesh = meshio.Mesh(
        points,
        [
            ("tetra", air_tet),
            ("triangle", tri),
        ],
        cell_data={
            "gmsh:physical": [
                np.array([10], dtype=np.int32),
                tri_tags,
            ]
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), mesh, file_format="gmsh22", binary=False)


def test_mesh_inspection_pass_and_fail() -> None:
    try:
        import meshio  # noqa: F401
    except ImportError:
        print("SKIP test_mesh_inspection_pass_and_fail (meshio not installed)", flush=True)
        return

    from inspect_shape_mesh_aperture import audit_mesh_aperture  # noqa: WPS433

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.msh"
        bad = Path(tmp) / "bad.msh"
        _write_synthetic_mesh(good, with_aperture=True)
        _write_synthetic_mesh(bad, with_aperture=False)
        good_audit = audit_mesh_aperture(good)
        bad_audit = audit_mesh_aperture(bad)
        assert good_audit["status"] == "PASS"
        assert good_audit["checks"]["air_path_connected"] is True
        assert bad_audit["status"] == "FAIL"


def test_inspect_script_locates_report_path() -> None:
    from inspect_shape_mesh_aperture import locate_mesh_path, render_gmsh_report_md  # noqa: WPS433
    from m4_shape_registry import resolve_shape_config  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        mesh, tried = locate_mesh_path(
            repo_root=repo_root,
            sample_id="box_sample_000",
            mesh_level="L_scout_coarse",
            run_root=None,
        )
        assert mesh is None
        assert tried
        cfg = resolve_shape_config("box")
        md = render_gmsh_report_md(
            shape_key="box",
            sample_id="box_sample_000",
            mesh_path=Path("dummy.msh"),
            audit={"status": "PASS", "checks": {"air_path_connected": True}, "failures": []},
            shape_cfg=cfg,
        )
        assert "gmsh" in md
        assert "Tag **2**" in md or "tag **2**" in md.lower() or "Tag **2**" in md


def main() -> int:
    tests = [
        test_promote_after_aggregation_updates_manifest,
        test_inconsistent_terminal_detected,
        test_full_clean_removes_stale_reuse_evidence,
        test_post_run_residue_audit_counts_forbidden,
        test_mesh_inspection_pass_and_fail,
        test_inspect_script_locates_report_path,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_box_fom_validation] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
