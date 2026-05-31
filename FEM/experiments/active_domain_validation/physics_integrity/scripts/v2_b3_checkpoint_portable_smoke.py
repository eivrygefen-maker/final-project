#!/usr/bin/env python3
"""Standalone portable checkpoint smoke (solver-mkl env; no DOLFINx/FEM imports)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_operator_checkpoint_portable import (  # noqa: E402
    B3_ST_CHECKPOINT_PORTABLE_SMOKE_ARG,
    B3_ST_REUSE_CHECKPOINT_ARG,
    CHECKPOINT_DIR_ARG,
    run_checkpoint_portable_smoke,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load checkpoint A/M (PETSc binary, then CSR fallback) and run mkl_pardiso smoke.",
    )
    parser.add_argument(
        CHECKPOINT_DIR_ARG,
        dest="checkpoint_dir",
        help="Checkpoint directory containing A_active/M_active exports.",
    )
    parser.add_argument(
        B3_ST_REUSE_CHECKPOINT_ARG,
        dest="checkpoint_dir_legacy",
        help="Legacy alias for --checkpoint-dir.",
    )
    args, _unknown = parser.parse_known_args(argv[1:] if argv is not None else None)
    checkpoint_dir = args.checkpoint_dir or args.checkpoint_dir_legacy
    if not checkpoint_dir:
        parser.error(f"one of {CHECKPOINT_DIR_ARG} or {B3_ST_REUSE_CHECKPOINT_ARG} is required")

    smoke_argv = [
        str(argv[0] if argv is not None else sys.argv[0]),
        B3_ST_CHECKPOINT_PORTABLE_SMOKE_ARG,
        CHECKPOINT_DIR_ARG,
        str(checkpoint_dir),
    ]
    return run_checkpoint_portable_smoke(smoke_argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
