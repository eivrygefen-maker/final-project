#!/usr/bin/env python3
"""Website STK note-library startup command tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from stk_app_audio_service import build_note_library_startup_command  # noqa: E402


class TestStkNoteLibraryStartupCommand(unittest.TestCase):
    def test_command_uses_supported_classic_only_shape_args(self) -> None:
        cmd = build_note_library_startup_command(
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


if __name__ == "__main__":
    unittest.main()
