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
from typing import Any
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


def _streamlit_stub_modules() -> dict[str, mock.MagicMock]:
    status_cm = mock.MagicMock()
    status_cm.__enter__ = mock.Mock(return_value=status_cm)
    status_cm.__exit__ = mock.Mock(return_value=False)
    st_mod = mock.MagicMock()
    st_mod.session_state = mock.MagicMock()
    st_mod.status.return_value = status_cm
    components_v1 = mock.MagicMock()
    components_pkg = mock.MagicMock()
    components_pkg.v1 = components_v1
    st_mod.components = components_pkg
    return {
        "streamlit": st_mod,
        "streamlit.components": components_pkg,
        "streamlit.components.v1": components_v1,
    }


_GUI_APP_MODULE: Any = None


def get_gui_app_module():
    """Import gui.app once per process with a Streamlit stub (no stpyvista registration)."""
    global _GUI_APP_MODULE
    if _GUI_APP_MODULE is not None:
        return _GUI_APP_MODULE
    sys.path.insert(0, str(REPO / "gui"))
    stubs = _streamlit_stub_modules()
    stubs["fem_main_3d"] = mock.MagicMock()
    try:
        with mock.patch.dict(sys.modules, stubs):
            import app as gui_app  # noqa: WPS433

            _GUI_APP_MODULE = gui_app
    except Exception:
        sys.modules.pop("app", None)
        raise
    return _GUI_APP_MODULE

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
        self.assertIn("def _import_stpyvista", src)
        self.assertNotIn("from stpyvista import stpyvista", src.split("def _import_stpyvista")[0])
        self.assertIn("if int(nev) > 0", src)
        self.assertIn("M4 ROM: ready", src)
        self.assertIn("st.table(rows)", src)
        self.assertIn("rom_body_pending", src)
        self.assertIn("complete_rom_body_response", src)

    def test_app_importable_without_stpyvista_registration(self) -> None:
        gui_app = get_gui_app_module()
        self.assertTrue(callable(gui_app.m4_parameters_from_ui))
        self.assertIsNone(gui_app._stpyvista_callable)

    def test_m4_parameters_from_ui_keys(self) -> None:
        gui_app = get_gui_app_module()
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
        self.assertIn("geometry.back_thickness", params)
        self.assertIn("top_wood_id", params)
        self.assertEqual(params["top_wood_id"], "spruce")

    def test_m4_prediction_keeps_all_frequencies_when_nev_zero(self) -> None:
        gui_app = get_gui_app_module()
        prediction = {"frequencies_hz": [98.4, 132.1, 200.5]}
        nev = 0
        raw_freqs = [float(f) for f in (prediction.get("frequencies_hz") or [])]
        freqs = raw_freqs[: int(nev)] if int(nev) > 0 else raw_freqs
        self.assertEqual(freqs, [98.4, 132.1, 200.5])
        self.assertEqual(
            gui_app.m4_parameters_from_ui(
                {"length": 0.48, "width": 0.325, "depth": 0.1, "top_thickness": 0.003, "hole_radius": 0.047},
                top_wood="spruce",
                back_wood="rosewood",
            )["geometry.back_thickness"],
            0.003 * 1.1,
        )

    def test_save_changes_does_not_launch_fem(self) -> None:
        gui_app = get_gui_app_module()
        with mock.patch.object(gui_app, "run_gmsh_display") as mock_display, mock.patch.object(
            gui_app, "run_gmsh_fom", side_effect=AssertionError("FOM must not run on save")
        ):
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
