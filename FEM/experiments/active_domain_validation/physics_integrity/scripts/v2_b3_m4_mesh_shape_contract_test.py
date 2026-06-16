#!/usr/bin/env python3
"""Tests for shape-aware mesh manifests and scout/prod mesh consistency."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "FEM" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "FEM" / "scripts"))

from v2_b3_m4_mesh_manifest_lib import (  # noqa: E402
    MESH_REUSE_REJECTED,
    assert_scout_lprod_shape_consistency,
    assert_scout_mesh_shape_gate,
    build_mesh_manifest,
    collect_global_mesh_cache_paths_resolved,
    format_mesh_reuse_rejected,
    install_mesh_with_sidecars,
    invalidate_stale_mesh_files,
    mesh_manifest_path,
    resolve_mesh_validation_context,
    validate_mesh_reuse,
    write_mesh_manifest,
)
from reset_m4_sample_state import full_clean_sample_run  # noqa: E402


def test_scout_mesh_case_includes_box_shape_type() -> None:
    captured: Dict[str, Any] = {}

    def fake_build_level_mesh(case, level_id, level_def, *, config_dir):
        captured["case"] = dict(case)
        return {
            "n_nodes": 100,
            "n_tetrahedra": 200,
            "volume_tag_counts": {"1": 1, "2": 1, "3": 1, "10": 1},
            "triangle_tag_counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1},
        }

    import v2_b3_m4_scout_mesh_build as scout_mod

    with mock.patch.object(scout_mod, "build_level_mesh", fake_build_level_mesh), mock.patch.object(
        scout_mod, "load_manifest", return_value={"mesh_levels": {"L_scout_coarse": {"lc_scale": 1.0, "build_env": {"FEM_ALLOW_FOM": "1"}}}}
    ), mock.patch.object(scout_mod, "mesh_path", lambda level, sid: Path(tempfile.gettempdir()) / f"{sid}.msh"), mock.patch.object(
        scout_mod, "write_json", lambda path, body: None
    ):
        result = scout_mod.build_scout_mesh_for_case(
            sample_id="box_sample_000",
            geometry={"length": 0.46, "width": 0.36, "depth": 0.10, "top_thickness": 0.003, "hole_radius": 0.042},
            shape_name="box",
            geometry_shape_type="Box",
            gmsh_shape_type="Box",
        )
    assert captured["case"]["geometry_shape_type"] == "Box"
    assert captured["case"]["gmsh_shape_type"] == "Box"
    assert result["summary"]["geometry_shape_type"] == "Box"


def test_mismatched_manifest_rejects_reuse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / "box_sample_000.msh"
        mesh.write_text("fake", encoding="utf-8")
        write_mesh_manifest(
            mesh_manifest_path(mesh),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="classic",
                geometry_shape_type="Classical",
                gmsh_shape_type="Classical",
                mesh_level="L_scout_coarse",
                mesh_path=mesh,
                geometry={"length": 0.48},
            ),
        )
        ok, reason, _ = validate_mesh_reuse(
            mesh,
            sample_id="box_sample_000",
            mesh_level="L_scout_coarse",
            shape_name="box",
            geometry_shape_type="Box",
            gmsh_shape_type="Box",
            geometry={"length": 0.46},
        )
        assert ok is False
        rejected = format_mesh_reuse_rejected(
            reason=reason,
            existing_shape="Classical",
            expected_shape="Box",
            mesh_path=mesh,
        )
        assert MESH_REUSE_REJECTED in rejected


def test_missing_manifest_rejects_reuse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / "box_sample_000.msh"
        mesh.write_text("fake", encoding="utf-8")
        ok, reason, _ = validate_mesh_reuse(
            mesh,
            sample_id="box_sample_000",
            mesh_level="L_scout_coarse",
            shape_name="box",
            geometry_shape_type="Box",
            gmsh_shape_type="Box",
            geometry={"length": 0.46},
        )
        assert ok is False
        assert reason == "missing_manifest"
        rejected = format_mesh_reuse_rejected(
            reason=reason,
            existing_shape="missing",
            expected_shape="Box",
            mesh_path=mesh,
            manifest_path=mesh_manifest_path(mesh),
        )
        assert "expected_manifest=" in rejected


def test_install_mesh_with_sidecars_copies_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        global_dir = root / "global" / "L_scout_coarse"
        run_dir = root / "run" / "scout" / "mesh" / "L_scout_coarse"
        global_dir.mkdir(parents=True)
        src = global_dir / "box_sample_000.msh"
        dst = run_dir / "box_sample_000.msh"
        src.write_text("mesh-bytes", encoding="utf-8")
        geom = {
            "length": 0.46,
            "width": 0.36,
            "depth": 0.10,
            "top_thickness": 0.003,
            "hole_radius": 0.042,
        }
        write_mesh_manifest(
            mesh_manifest_path(src),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="box",
                geometry_shape_type="Box",
                gmsh_shape_type="Box",
                mesh_level="L_scout_coarse",
                mesh_path=src,
                geometry=geom,
            ),
        )
        (global_dir / "box_sample_000_mesh_build_summary.json").write_text("{}", encoding="utf-8")

        report = install_mesh_with_sidecars(src_msh=src, dst_msh=dst, sample_id="box_sample_000")
        assert dst.is_file()
        assert mesh_manifest_path(dst).is_file()
        assert "mesh_manifest" in report["copied"]
        manifest = json.loads(mesh_manifest_path(dst).read_text(encoding="utf-8"))
        assert manifest["run_dir_mesh_path"] == str(dst)
        assert manifest["canonical_mesh_path"] == str(src)
        assert manifest["geometry_shape_type"] == "Box"


def test_scout_mesh_shape_gate_passes_with_run_dir_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / "box_sample_000.msh"
        mesh.write_text("fake", encoding="utf-8")
        geom = {
            "length": 0.46,
            "width": 0.36,
            "depth": 0.10,
            "top_thickness": 0.003,
            "hole_radius": 0.042,
        }
        write_mesh_manifest(
            mesh_manifest_path(mesh),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="box",
                geometry_shape_type="Box",
                gmsh_shape_type="Box",
                mesh_level="L_scout_coarse",
                mesh_path=mesh,
                geometry=geom,
            ),
        )
        sample = {
            "sample_id": "box_sample_000",
            "shape_name": "box",
            "geometry_shape_type": "Box",
            "geometry": geom,
        }
        ok, detail = assert_scout_mesh_shape_gate(mesh_path=mesh, sample=sample)
        assert ok is True
        assert "SCOUT_MESH_SHAPE_ASSERT" in detail
        assert "status=PASS" in detail
        assert "SCOUT_MESH_VALIDATED_SOURCE run_dir" in detail


def test_scout_mesh_shape_gate_repairs_manifest_from_global() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        global_dir = root / "mesh" / "L_scout_coarse"
        global_dir.mkdir(parents=True)
        global_mesh = global_dir / "box_sample_000.msh"
        run_mesh = root / "run" / "box_sample_000.msh"
        global_mesh.write_text("global", encoding="utf-8")
        run_mesh.parent.mkdir(parents=True, exist_ok=True)
        run_mesh.write_text("run", encoding="utf-8")
        geom = {"length": 0.46, "width": 0.36, "depth": 0.10, "top_thickness": 0.003, "hole_radius": 0.042}
        write_mesh_manifest(
            mesh_manifest_path(global_mesh),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="box",
                geometry_shape_type="Box",
                gmsh_shape_type="Box",
                mesh_level="L_scout_coarse",
                mesh_path=global_mesh,
                geometry=geom,
            ),
        )
        sample = {"sample_id": "box_sample_000", "shape_name": "box", "geometry": geom}
        with mock.patch("v2_mesh_convergence_common.CONV_MESH", root / "mesh"):
            ok, detail = assert_scout_mesh_shape_gate(mesh_path=run_mesh, sample=sample)
        assert ok is True
        assert mesh_manifest_path(run_mesh).is_file()
        assert "run_dir_repaired_from_global" in detail


def test_lprod_install_matches_scout_manifest_convention() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "global" / "box_sample_000.msh"
        dst = root / "lprod" / "box_sample_000.msh"
        src.parent.mkdir(parents=True)
        src.write_text("mesh", encoding="utf-8")
        write_mesh_manifest(
            mesh_manifest_path(src),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="box",
                geometry_shape_type="Box",
                gmsh_shape_type="Box",
                mesh_level="L_rom_prod",
                mesh_path=src,
                geometry={"length": 0.46},
            ),
        )
        install_mesh_with_sidecars(src_msh=src, dst_msh=dst, sample_id="box_sample_000")
        assert mesh_manifest_path(dst).is_file()
        ok, detail = assert_scout_lprod_shape_consistency(
            scout_mesh_path=dst,
            lprod_mesh_path=dst,
        )
        assert ok is True
        assert "SCOUT_LPROD_SHAPE_CONSISTENCY_PASS" in detail


def test_full_clean_removes_run_dir_mesh_manifests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_root = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
            / "box_sample_000/runs/box_sample_000_box_fom_v1"
        )
        scout_mesh_dir = run_root / "scout/mesh/L_scout_coarse"
        scout_mesh_dir.mkdir(parents=True)
        mesh = scout_mesh_dir / "box_sample_000.msh"
        mesh.write_text("mesh", encoding="utf-8")
        write_mesh_manifest(
            mesh_manifest_path(mesh),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="box",
                geometry_shape_type="Box",
                gmsh_shape_type="Box",
                mesh_level="L_scout_coarse",
                mesh_path=mesh,
                geometry={"length": 0.46},
            ),
        )
        (run_root / "pipeline_run_manifest.json").write_text(
            json.dumps({"terminal_status": "RUNNING"}),
            encoding="utf-8",
        )
        report = full_clean_sample_run(
            repo_root=repo_root,
            run_root=run_root,
            sample_id="box_sample_000",
            run_id="box_sample_000_box_fom_v1",
        )
        assert report["status"] == "PASS"
        assert not scout_mesh_dir.exists()


def test_collect_global_mesh_cache_includes_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        level_dir = root / "mesh" / "L_scout_coarse"
        level_dir.mkdir(parents=True)
        mesh = level_dir / "box_sample_000.msh"
        mesh.write_text("mesh", encoding="utf-8")
        manifest = mesh_manifest_path(mesh)
        manifest.write_text("{}", encoding="utf-8")
        with mock.patch("v2_mesh_convergence_common.CONV_MESH", root / "mesh"):
            found = collect_global_mesh_cache_paths_resolved(root, "box_sample_000")
        assert mesh in found
        assert manifest in found


def test_scout_mesh_shape_gate_fails_with_missing_manifest_message() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / "box_sample_000.msh"
        mesh.write_text("fake", encoding="utf-8")
        sample = {
            "sample_id": "box_sample_000",
            "shape_name": "box",
            "geometry": {"length": 0.46},
        }
        ok, detail = assert_scout_mesh_shape_gate(mesh_path=mesh, sample=sample)
        assert ok is False
        assert "expected_manifest=" in detail
        assert "SCOUT_MESH_MANIFEST_EXISTS" in detail


def test_scout_mesh_shape_gate_fails_classical_for_box() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / "box_sample_000.msh"
        mesh.write_text("fake", encoding="utf-8")
        write_mesh_manifest(
            mesh_manifest_path(mesh),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="classic",
                geometry_shape_type="Classical",
                gmsh_shape_type="Classical",
                mesh_level="L_scout_coarse",
                mesh_path=mesh,
                geometry={"length": 0.48},
            ),
        )
        sample = {
            "sample_id": "box_sample_000",
            "shape_name": "box",
            "geometry_shape_type": "Box",
            "geometry": {
                "length": 0.46,
                "width": 0.36,
                "depth": 0.10,
                "top_thickness": 0.003,
                "hole_radius": 0.042,
            },
        }
        ok, detail = assert_scout_mesh_shape_gate(mesh_path=mesh, sample=sample)
        assert ok is False
        assert "SCOUT_MESH_SHAPE_ASSERT" in detail
        assert "FAIL" in detail
        assert "mesh_manifest_shape=Classical" in detail


def test_scout_lprod_shape_consistency_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        scout = Path(tmp) / "scout.msh"
        lprod = Path(tmp) / "lprod.msh"
        scout.write_text("s", encoding="utf-8")
        lprod.write_text("l", encoding="utf-8")
        geom = {
            "length": 0.46,
            "width": 0.36,
            "depth": 0.10,
            "top_thickness": 0.003,
            "hole_radius": 0.042,
        }
        for path, level in ((scout, "L_scout_coarse"), (lprod, "L_rom_prod")):
            write_mesh_manifest(
                mesh_manifest_path(path),
                build_mesh_manifest(
                    sample_id="box_sample_000",
                    shape_name="box",
                    geometry_shape_type="Box",
                    gmsh_shape_type="Box",
                    mesh_level=level,
                    mesh_path=path,
                    geometry=geom,
                ),
            )
        ok, detail = assert_scout_lprod_shape_consistency(
            scout_mesh_path=scout,
            lprod_mesh_path=lprod,
        )
        assert ok is True
        assert "SCOUT_LPROD_SHAPE_CONSISTENCY_PASS" in detail


def test_invalidate_stale_mesh_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / "box_sample_000.msh"
        mesh.write_text("mesh", encoding="utf-8")
        write_mesh_manifest(
            mesh_manifest_path(mesh),
            build_mesh_manifest(
                sample_id="box_sample_000",
                shape_name="box",
                geometry_shape_type="Box",
                gmsh_shape_type="Box",
                mesh_level="L_scout_coarse",
                mesh_path=mesh,
                geometry={"length": 0.46},
            ),
        )
        removed = invalidate_stale_mesh_files(mesh)
        assert not mesh.is_file()
        assert len(removed) >= 2


def main() -> int:
    tests = [
        test_scout_mesh_case_includes_box_shape_type,
        test_mismatched_manifest_rejects_reuse,
        test_missing_manifest_rejects_reuse,
        test_install_mesh_with_sidecars_copies_manifest,
        test_scout_mesh_shape_gate_passes_with_run_dir_manifest,
        test_scout_mesh_shape_gate_repairs_manifest_from_global,
        test_scout_mesh_shape_gate_fails_with_missing_manifest_message,
        test_scout_mesh_shape_gate_fails_classical_for_box,
        test_scout_lprod_shape_consistency_pass,
        test_lprod_install_matches_scout_manifest_convention,
        test_full_clean_removes_run_dir_mesh_manifests,
        test_collect_global_mesh_cache_includes_manifest,
        test_invalidate_stale_mesh_files,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_mesh_shape_contract] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
