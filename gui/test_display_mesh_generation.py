#!/usr/bin/env python3
"""Lightweight GUI/display-mesh checks (no FEM solve)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
GUI_APP = REPO / "gui" / "app.py"
GEOMETRY_SCRIPT = REPO / "FEM" / "geometry" / "build_3d_guitar.py"
CONFIG_PATH = REPO / "FEM" / "configs" / "guitar_3d.json"

try:
    import gmsh  # noqa: F401

    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

try:
    import pyvista  # noqa: F401

    HAS_GUI_DEPS = True
except ImportError:
    HAS_GUI_DEPS = False


class GeometryDisplayFixTests(unittest.TestCase):
    def test_shell_only_initializes_all_shell_surfs(self) -> None:
        src = GEOMETRY_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("all_shell_surfs: list = sorted(wood_boundary_surfs)", src)
        self.assertIn("elif shell_only:", src)
        self.assertIn("wood shell has no boundary surfaces", src)

    def test_display_fragment_skips_single_volume(self) -> None:
        src = GEOMETRY_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("if len(vols) < 2:", src)


class GuiPipelineWiringTests(unittest.TestCase):
    def test_app_declares_classic_rom_paths(self) -> None:
        src = GUI_APP.read_text(encoding="utf-8")
        self.assertIn('"mesh_profile": "rom"', src)
        self.assertIn('"mesh_level_id": "L_rom_prod"', src)
        self.assertIn("m4_geometry_corrected_rommesh_v1", src)
        self.assertIn("rom_model_manifest.json", src)
        self.assertIn("M4_ROM_MANIFEST", src)
        self.assertIn("official_rom_dataset.jsonl", src)
        self.assertIn("predict_m4_modal_frequencies", src)

    @unittest.skipUnless(HAS_GUI_DEPS, "pyvista/streamlit GUI deps not installed")
    def test_m4_parameters_from_ui_keys(self) -> None:
        sys.path.insert(0, str(REPO / "gui"))
        import app as gui_app  # noqa: WPS433

        params = gui_app.m4_parameters_from_ui(
            {
                "length": 0.48,
                "width": 0.325,
                "depth": 0.1,
                "top_thickness": 0.003,
                "hole_radius": 0.047,
            },
            top_wood="spruce",
            back_wood="rosewood",
        )
        self.assertIn("geometry.length", params)
        self.assertIn("top_wood_id", params)
        self.assertEqual(params["top_wood_id"], "spruce")

    @unittest.skipUnless(HAS_GUI_DEPS, "pyvista/streamlit GUI deps not installed")
    def test_save_changes_does_not_launch_fem(self) -> None:
        sys.path.insert(0, str(REPO / "gui"))
        import app as gui_app  # noqa: WPS433

        with mock.patch.object(gui_app, "run_gmsh_display") as mock_display, mock.patch.object(
            gui_app, "run_gmsh_fom", side_effect=AssertionError("FOM must not run on save")
        ), mock.patch("streamlit.status") as mock_status:
            mock_status.return_value.__enter__ = lambda s: s
            mock_status.return_value.__exit__ = lambda s, *a: None
            gui_app.regenerate_display_mesh(
                {
                    "shape_type": "Classical",
                    "length": 0.48,
                    "width": 0.325,
                    "depth": 0.1,
                    "top_thickness": 0.003,
                    "hole_radius": 0.047,
                },
                top_wood="spruce",
                back_wood="rosewood",
                clamp_ribs=True,
                pin_neck_fix=True,
                fixture_preset=gui_app.DEFAULT_FIXTURE_PRESET,
                geom_fp="test",
            )
        mock_display.assert_called_once()


@unittest.skipUnless(HAS_GMSH, "gmsh not installed")
class DisplayMeshGenerationTests(unittest.TestCase):
    def _run_display_mesh(self, config: dict, out_msh: Path) -> subprocess.CompletedProcess[str]:
        cfg_path = out_msh.parent / "guitar_test.json"
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        env = {
            **os.environ,
            "FEM_ALLOW_DISPLAY": "1",
            "FEM_ALLOW_PREVIEW": "0",
            "FEM_ALLOW_FOM": "0",
        }
        for key in ("FEM_ALLOW_PREVIEW", "FEM_ALLOW_FOM", "FEM_VALIDATION_MESH"):
            env.pop(key, None)
        return subprocess.run(
            [sys.executable, str(GEOMETRY_SCRIPT), "--config", str(cfg_path), "-nopopup"],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def _base_config(self) -> dict:
        base = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        base["geometry"] = {
            **base.get("geometry", {}),
            "shape_type": "Classical",
            "length": 0.48,
            "width": 0.325,
            "depth": 0.1,
            "top_thickness": 0.003,
            "hole_radius": 0.047,
            "soundhole_from_neck_ratio": 0.5,
        }
        return base

    def test_display_mesh_default_classical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "display_mesh.msh"
            result = self._run_display_mesh(self._base_config(), out)
            produced = REPO / "FEM" / "mesh" / "display_mesh.msh"
            self.assertTrue(
                produced.is_file() and produced.stat().st_size > 0,
                (result.stdout or "") + (result.stderr or ""),
            )
            combined = (result.stdout or "") + (result.stderr or "")
            self.assertNotIn("all_shell_surfs", combined)
            self.assertNotIn("referenced before assignment", combined)

    def test_display_mesh_after_parameter_change(self) -> None:
        cfg = self._base_config()
        cfg["geometry"]["length"] = 0.50
        cfg["geometry"]["hole_radius"] = 0.042
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "display_mesh.msh"
            result = self._run_display_mesh(cfg, out)
            produced = REPO / "FEM" / "mesh" / "display_mesh.msh"
            self.assertTrue(produced.is_file() and produced.stat().st_size > 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
