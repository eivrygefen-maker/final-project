#!/usr/bin/env python3
"""Stage 2: auto ROM after Save & Sync (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
ROM_STK_JSON = REPO / "FEM" / "outputs" / "rom_stk_body.json"
sys.path.insert(0, str(REPO / "gui"))

from test_display_mesh_generation import get_gui_app_module  # noqa: E402


class _SessionBox(dict):
    def __getattr__(self, key: str):
        return self.get(key)

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


def _studio_save_event() -> dict:
    return {
        "action": "save_sync",
        "_ts": 123456,
        "shape_type": "Classical",
        "length": 0.48,
        "width": 0.325,
        "depth": 0.1,
        "top_thickness": 0.003,
        "hole_radius": 0.047,
        "top_wood_id": "spruce",
        "back_wood_id": "rosewood",
    }


class Stage2RomAutoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gui = get_gui_app_module()
        self.ss = _SessionBox(
            developer_fom_mode=False,
            mesh_is_dirty=False,
            show_mesh_overlay=False,
            _mesh_overlay_rom_fp="",
            _studio_event_id="",
            rom_body_ready=False,
            rom_body_pending=False,
            rom_body_fingerprint="",
            rom_body_error="",
            sound_stale=True,
            stk_body_json="",
            physics_ready=False,
        )
        self.gui.st.session_state = self.ss

    def test_save_sync_sets_rom_pending_without_fem(self) -> None:
        with mock.patch.object(self.gui, "regenerate_display_mesh") as regen, mock.patch.object(
            self.gui, "run_gmsh_fom", side_effect=AssertionError("FOM must not run")
        ), mock.patch.object(self.gui, "fem_main_3d", create=True) as fem_mock:
            fem_mock.run_fem_3d_simulation.side_effect = AssertionError("FEM must not run")
            self.gui.process_fast_preview_event(
                _studio_save_event(),
                clamp_ribs=True,
                pin_neck=True,
                fixture_preset=self.gui.DEFAULT_FIXTURE_PRESET,
            )
        regen.assert_called_once()
        self.assertTrue(self.ss["rom_body_pending"])
        self.assertFalse(self.ss["rom_body_ready"])

    def test_param_change_marks_rom_stale(self) -> None:
        self.ss["rom_body_ready"] = True
        self.ss["rom_body_fingerprint"] = "old"
        self.ss["stk_body_json"] = "/tmp/body.json"
        self.gui.process_fast_preview_event(
            {"action": "param_change", "_ts": 1, "shape_type": "Classical", "length": 0.5},
            clamp_ribs=True,
            pin_neck=True,
            fixture_preset=self.gui.DEFAULT_FIXTURE_PRESET,
        )
        self.assertFalse(self.ss["rom_body_ready"])
        self.assertTrue(self.ss["sound_stale"])
        self.assertEqual(self.ss["stk_body_json"], "")

    def test_rom_body_response_ready_requires_matching_fingerprint(self) -> None:
        self.ss["rom_body_ready"] = True
        self.ss["rom_body_fingerprint"] = "fp-a"
        with tempfile.TemporaryDirectory() as tmp:
            body = Path(tmp) / "rom_stk_body.json"
            body.write_text(json.dumps({"modes_hz": [100.0]}), encoding="utf-8")
            self.ss["stk_body_json"] = str(body)
            with mock.patch.object(self.gui, "ROM_STK_JSON", body):
                self.assertTrue(self.gui.rom_body_response_ready("fp-a"))
                self.assertFalse(self.gui.rom_body_response_ready("fp-b"))

    def test_complete_rom_body_response_writes_json(self) -> None:
        prediction = {
            "frequencies_hz": [90.0, 150.0, 220.0],
            "predicted_modes": [{"frequency_hz": 90.0, "mic_output_proxy": 0.01}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rom_stk_body.json"
            with mock.patch.object(self.gui, "ROM_STK_JSON", out), mock.patch.object(
                self.gui,
                "run_rom_acoustics",
                return_value=out,
            ) as run_rom:
                path = self.gui.complete_rom_body_response(
                    {"geometry.length": 0.48},
                    "Classical",
                    rom_fp="fp-test",
                )
            run_rom.assert_called_once()
            self.assertEqual(path, out)
            self.assertTrue(self.ss["rom_body_ready"])
            self.assertEqual(self.ss["rom_body_fingerprint"], "fp-test")
            self.assertTrue(self.ss["sound_stale"])

    def test_run_rom_acoustics_prefers_m4_surrogate(self) -> None:
        self.ss["_geom"] = {
            "length": 0.48,
            "width": 0.325,
            "depth": 0.1,
            "top_thickness": 0.003,
            "hole_radius": 0.047,
        }
        self.ss["_top_wood"] = "spruce"
        self.ss["_back_wood"] = "rosewood"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rom_stk_body.json"
            with mock.patch.object(self.gui, "ROM_STK_JSON", out), mock.patch.object(
                self.gui, "m4_rom_available", return_value=True
            ), mock.patch.object(
                self.gui,
                "predict_m4_modal_frequencies",
                return_value=([100.0, 200.0], {"predicted_modes": []}),
            ) as predict, mock.patch.object(self.gui, "get_rom_manager") as legacy:
                result = self.gui.run_rom_acoustics({}, "Classical")
            predict.assert_called_once()
            legacy.assert_not_called()
            self.assertEqual(result, out)
            doc = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(doc["modes_hz"], [100.0, 200.0])
            self.assertEqual(self.ss["rom_backend"], "m4_modal_surrogate")

    def test_rom_failure_leaves_body_not_ready(self) -> None:
        with mock.patch.object(
            self.gui,
            "run_rom_acoustics",
            side_effect=RuntimeError("M4 modal surrogate returned no frequencies."),
        ):
            with self.assertRaises(RuntimeError):
                self.gui.complete_rom_body_response({}, "Classical", rom_fp="fp")
        self.assertFalse(self.ss["rom_body_ready"])

    def test_source_declares_stage2_flow(self) -> None:
        src = (REPO / "gui" / "app.py").read_text(encoding="utf-8")
        self.assertIn("rom_body_pending", src)
        self.assertIn("complete_rom_body_response", src)
        self.assertIn("rom_body_response_ready", src)
        self.assertIn('action == "save_sync"', src)


if __name__ == "__main__":
    unittest.main()
