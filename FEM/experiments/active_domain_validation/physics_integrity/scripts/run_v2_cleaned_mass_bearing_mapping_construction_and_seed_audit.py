#!/usr/bin/env python3
"""Deprecated wrapper — use run_v2_cleaned_mass_bearing_mapping_decision.py."""
from __future__ import annotations

import run_v2_cleaned_mass_bearing_mapping_decision as _decision

OUT_JSON = _decision.OUT_JSON
OUT_MD = _decision.OUT_MD


def main() -> int:
    return int(_decision.main())


if __name__ == "__main__":
    raise SystemExit(main())
