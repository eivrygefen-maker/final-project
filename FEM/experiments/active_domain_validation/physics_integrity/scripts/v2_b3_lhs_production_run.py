#!/usr/bin/env python3
"""B3 production LHS entry point — delegates to the M4 production batch runner."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_production_batch import main as m4_batch_main  # noqa: E402

_LEGACY_MSG = (
    "DEPRECATED: use run_m4_production_pipeline.py for LHS pool production "
    "(or v2_b3_m4_lhs_production_batch.py with a hand-written batch spec). "
    "This wrapper remains for compatibility only."
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not any(a in ("-h", "--help") for a in args):
        print(_LEGACY_MSG, file=sys.stderr)
    return m4_batch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
