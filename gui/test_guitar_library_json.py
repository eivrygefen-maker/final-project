#!/usr/bin/env python3
"""Lightweight static JSON validation for the Classical guitar player library."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from guitar_library import load_guitar_library  # noqa: E402


class TestGuitarLibraryJson(unittest.TestCase):
    def test_static_library_loads_and_validates(self) -> None:
        library = load_guitar_library()
        self.assertEqual(library["status"], "ready")
        self.assertEqual(len(library["chords"]["chords"]), 13)
        self.assertEqual(len(library["melodies"]["melodies"]), 10)


if __name__ == "__main__":
    unittest.main()
