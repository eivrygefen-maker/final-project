#!/usr/bin/env python3
"""Website STK note-library startup command tests."""
from __future__ import annotations

import sys
import shutil
import types
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.SimpleNamespace(
        session_state={},
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        success=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        caption=lambda *args, **kwargs: None,
    )

import stk_app_audio_service as audio_service  # noqa: E402
import stk_app_ui  # noqa: E402


class TestStkNoteLibraryStartupCommand(unittest.TestCase):
    def test_command_uses_supported_classic_only_shape_args(self) -> None:
        cmd = audio_service.build_note_library_startup_command(
            script=Path("tools/build_app_stk_note_library.py"),
            root=Path("."),
            sample_id="sample_000",
            shape_type="Classical",
            cache_dir=Path("cache"),
            parameter_hash="abc123",
            job_status_json=Path("job.json"),
            render_mode="parallel_batch",
            parallel_workers=2,
            priority_notes=["A2", "E4"],
        )

        self.assertNotIn("--instrument", cmd)
        self.assertIn("--shape-type", cmd)
        self.assertEqual(cmd[cmd.index("--shape-type") + 1], "Classical")
        self.assertEqual(cmd[cmd.index("--sample-id") + 1], "sample_000")
        self.assertIn("--contrast-preset", cmd)
        self.assertEqual(cmd[cmd.index("--contrast-preset") + 1], "strong")

    def test_parallel_staging_promotes_sharp_note_wavs_to_final_cache(self) -> None:
        root = REPO / f".tmp_stk_promotion_test_{uuid4().hex}"
        staging = root / ".render_tmp" / "hashsharp" / "staging"
        final = root / "audio" / "app_stk_note_cache" / "classical" / "current_preview_hashsharp"
        try:
            staging.mkdir(parents=True)
            for note in ("E2", "F#2", "A2", "C#5"):
                (staging / f"{note}.wav").write_bytes(b"RIFF....WAVEfmt ")

            result = audio_service.finalize_parallel_staging_cache(
                staging_dir=staging,
                target_dir=final,
                parameter_hash="hashsharp",
                cfg={"fret_count": 19, "render_mode": "parallel_batch", "parallel_workers": 3},
                required_notes=("E2", "F#2", "A2", "C#5"),
            )

            self.assertTrue((final / "E2.wav").is_file())
            self.assertTrue((final / "F#2.wav").is_file())
            self.assertTrue((final / "C#5.wav").is_file())
            self.assertGreaterEqual(result["generated_note_count"], 4)
            self.assertGreaterEqual(result["note_wav_count"], 4)
            self.assertEqual(result["missing_required_notes"], [])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_same_hash_running_job_does_not_launch_duplicate(self) -> None:
        existing = {"status": "running", "pid": 1234, "parameter_hash": "hash1"}
        with (
            patch.object(audio_service, "load_app_stk_config", return_value={"fret_count": 19}),
            patch.object(audio_service, "build_required_note_set_from_fretboard", return_value=["A2"]),
            patch.object(audio_service, "note_range_label_from_required", return_value="A2:A2"),
            patch.object(audio_service, "parallel_workers_from_config", return_value=3),
            patch.object(audio_service, "preview_cache_dir", return_value=Path("should_not_be_created")),
            patch.object(audio_service, "cache_is_ready_for_fretboard", return_value=False),
            patch.object(audio_service, "set_active_job"),
            patch.object(audio_service, "read_job_status", return_value=existing),
            patch.object(audio_service.subprocess, "Popen") as popen,
        ):
            result = audio_service.start_background_note_library_job(
                parameter_hash="hash1",
                repo_root=Path("."),
            )

        self.assertEqual(result, existing)
        popen.assert_not_called()

    def test_generate_display_request_does_not_schedule_stk_startup(self) -> None:
        with (
            patch.object(stk_app_ui, "compute_parameter_hash", return_value="hash2"),
            patch.object(stk_app_ui, "_stk_cache_is_loadable", return_value=False),
            patch.object(stk_app_ui, "resolve_preview_cache_ready_state", return_value={"status": "not_started"}),
        ):
            result = stk_app_ui.request_generate_guitar(
                repo_root=Path("."),
                rom_fp="rom",
                lhs_params={},
                geom={"shape_type": "Classical"},
                top_wood="spruce",
                back_wood="rosewood",
            )

        self.assertEqual(result["action"], "stk_running")
        self.assertFalse(hasattr(stk_app_ui, "schedule_stk_after_rom"))

    def test_failed_note_library_status_is_not_ready(self) -> None:
        with (
            patch.object(stk_app_ui, "compute_parameter_hash", return_value="hash3"),
            patch.object(stk_app_ui, "_stk_cache_is_loadable", return_value=False),
            patch.object(
                stk_app_ui,
                "resolve_preview_cache_ready_state",
                return_value={"status": "failed", "readiness": "generated_but_missing_notes"},
            ),
        ):
            result = stk_app_ui.request_generate_guitar(
                repo_root=Path("."),
                rom_fp="rom",
                lhs_params={},
                geom={"shape_type": "Classical"},
                top_wood="spruce",
                back_wood="rosewood",
            )

        self.assertEqual(result["action"], "stk_failed")
        self.assertFalse(hasattr(stk_app_ui, "schedule_stk_after_rom"))

    def test_activation_latches_stable_player_hash_and_cache_dir(self) -> None:
        sys.modules["streamlit"].session_state.clear()
        activation = {
            "cache_path": "audio/app_stk_note_cache/classical/current_preview_hash4",
            "parameter_hash": "hash4",
            "saved_guitar_id": "",
            "player_fingerprint": "stk_hash4",
            "player_payload": {"status": "ready", "positions": [{"wav": "S6_f0.wav"}]},
            "validation": {"ok": True},
        }

        stk_app_ui.apply_stk_activation_to_session(activation)
        ss = sys.modules["streamlit"].session_state

        self.assertEqual(ss["active_player_hash"], "hash4")
        self.assertEqual(ss["loaded_player_hash"], "hash4")
        self.assertTrue(ss["active_player_cache_dir"].endswith("current_preview_hash4"))
        self.assertTrue(ss["active_stk_player_key"])

    def test_generate_ignores_saved_stack_and_activates_current_preview(self) -> None:
        sys.modules["streamlit"].session_state.clear()
        preview = Path("audio/app_stk_note_cache/classical/current_preview_hash5")
        saved = Path("audio/app_stk_note_cache/classical/saved_guitar_hash5_123")
        captured = {}

        def fake_activate(*, repo_root, parameter_hash, cache_dir, saved_guitar_id=""):
            captured["cache_dir"] = str(cache_dir)
            return {
                "cache_path": str(cache_dir),
                "parameter_hash": parameter_hash,
                "saved_guitar_id": saved_guitar_id,
                "player_fingerprint": f"stk_{parameter_hash}",
                "player_payload": {"status": "ready", "positions": [{"wav": "S6_f0.wav"}]},
                "validation": {"ok": True},
            }

        with (
            patch.object(stk_app_ui, "compute_parameter_hash", return_value="hash5"),
            patch.object(
                stk_app_ui,
                "resolve_preview_cache_ready_state",
                return_value={"status": "ready", "preview_cache_ready": True, "preview_cache_path": str(preview)},
            ),
            patch.object(
                stk_app_ui,
                "find_stack_entry_by_hash",
                return_value={"saved_guitar_id": "saved_hash5", "note_cache_path": str(saved)},
            ),
            patch.object(stk_app_ui, "_activate_ready_preview_cache", side_effect=fake_activate),
        ):
            result = stk_app_ui.generate_or_load_ready_guitar(
                repo_root=Path("."),
                rom_fp="rom",
                lhs_params={},
                geom={"shape_type": "Classical"},
                top_wood="spruce",
                back_wood="rosewood",
            )

        self.assertEqual(result["action"], "activated_preview")
        self.assertEqual(captured["cache_dir"], str(preview))
        self.assertEqual(
            sys.modules["streamlit"].session_state["active_player_cache_dir"],
            str(preview).replace("\\", "/"),
        )


if __name__ == "__main__":
    unittest.main()
