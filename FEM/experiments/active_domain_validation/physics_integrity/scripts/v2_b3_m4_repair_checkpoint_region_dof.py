#!/usr/bin/env python3
"""CLI: repair region-DOF / aperture export on existing L_prod checkpoint (no A/M rebuild)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_checkpoint_region_dof_repair import (  # noqa: E402
    assess_checkpoint_region_dof_repair,
    repair_checkpoint_region_dof_export,
)
from v2_b3_m4_worker_run_lib import detect_repo_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair region-DOF/aperture mask on existing L_prod checkpoint (no A/M rebuild)."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="M4 run tree root.")
    parser.add_argument("--dry-run", action="store_true", help="Assess only; do not write artifacts.")
    parser.add_argument("--force", action="store_true", help="Repair even if contract already passes.")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    assessment = assess_checkpoint_region_dof_repair(run_root, repo_root=repo_root)
    print(json.dumps(assessment, indent=2, sort_keys=True))
    if args.dry_run:
        return 0 if assessment.get("eligible") else 2

    rc, msg = repair_checkpoint_region_dof_export(
        repo_root=repo_root,
        run_root=run_root,
        force=bool(args.force),
    )
    print(json.dumps({"rc": rc, "message": msg}, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
