#!/usr/bin/env python3
"""Repair one partial run stuck at terminal_status=RUNNING after L_prod checkpoint PASS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_run_status_repair import (  # noqa: E402
    STALE_RUNNING_REPAIR_REASON,
    assess_stale_running_repair,
    promote_checkpoint_ready_terminal,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair stale RUNNING terminal after checkpoint-ready partial run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="M4 run tree root.")
    parser.add_argument("--dry-run", action="store_true", help="Assess only; do not write manifest repair.")
    args = parser.parse_args(argv)
    run_root = args.run_dir.expanduser().resolve()
    assessment = assess_stale_running_repair(run_root)
    print(json.dumps(assessment, indent=2, sort_keys=True))
    if not assessment.get("eligible"):
        return 2
    result = promote_checkpoint_ready_terminal(
        run_root,
        repair_reason=STALE_RUNNING_REPAIR_REASON,
        previous_status=str(assessment.get("previous_status") or "RUNNING"),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in ("PASS", "DRY_RUN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
