#!/usr/bin/env python3
"""Smoke test: validation scripts can import fem_main_3d (same bootstrap as Stage A)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_synthesis_export import fem_import_diagnostics, import_fem_main_3d  # noqa: E402


def main() -> int:
    diag = fem_import_diagnostics(start=SCRIPT_DIR)
    print(json.dumps(diag, indent=2, sort_keys=True))
    try:
        fem3d, import_diag = import_fem_main_3d(start=SCRIPT_DIR)
    except ModuleNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    mod_file = Path(fem3d.__file__).resolve()
    print(f"fem_main_3d = {mod_file}")
    print(f"imported_module_file = {import_diag.get('imported_module_file')}")
    print("PASS fem_main_3d import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
