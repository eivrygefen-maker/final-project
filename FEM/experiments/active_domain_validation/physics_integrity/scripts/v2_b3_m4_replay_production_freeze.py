#!/usr/bin/env python3
"""Replay production freeze + acceptance + terminal promotion (no workers / Stage A)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_production_freeze import (  # noqa: E402
    assess_production_completion,
    load_sample_input,
    replay_production_freeze,
)
from v2_b3_m4_worker_run_lib import detect_repo_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay production freeze from existing aggregation/checkpoint artifacts."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="M4 run tree root.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing freeze_manifest.json.")
    parser.add_argument("--dry-run", action="store_true", help="Assess only; do not write freeze outputs.")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    before = assess_production_completion(run_root)
    print(json.dumps({"before": before}, indent=2, sort_keys=True))
    if args.dry_run:
        return 0 if before["complete"] else 2

    rc, msg = replay_production_freeze(
        repo_root=repo_root,
        run_root=run_root,
        sample_input=load_sample_input(run_root),
        force=bool(args.force),
    )
    after = assess_production_completion(run_root)
    print(json.dumps({"replay": {"rc": rc, "message": msg}, "after": after}, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
