#!/usr/bin/env python3
"""
Run the full FEM→tuner→package pipeline for a range of LHS samples sequentially.

Each sample is fully isolated: ``package_rom --cleanup`` clears ``FEM/SORTING``
scratch (temp_modes, temp_results, candidates_log) before the next sample starts.

Example (overnight, 30 samples, 2 concurrent workers per sample)::

  python FEM/scripts/run_lhs_batch.py --first 1 --last 30 --max-workers 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PIPELINE = SCRIPT_DIR / "run_pipeline.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential LHS batch driver.")
    parser.add_argument("--first", type=int, default=1, help="First sample index (default 1).")
    parser.add_argument("--last", type=int, default=30, help="Last sample index inclusive.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Passed to run_pipeline / fem_master_dynamic (default 2).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "FEM" / "configs" / "guitar_3d.json",
        help="Base FEM JSON for run_pipeline.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one sample fails (default: stop on first failure).",
    )
    args = parser.parse_args()

    if not PIPELINE.is_file():
        print(f"Error: missing {PIPELINE}", file=sys.stderr)
        return 1
    if int(args.max_workers) < 1:
        print("Error: --max-workers must be >= 1", file=sys.stderr)
        return 1

    py = sys.executable
    failed: list[str] = []
    t0 = time.perf_counter()

    for n in range(int(args.first), int(args.last) + 1):
        sid = f"sample_{n:03d}"
        cmd = [
            py,
            str(PIPELINE),
            "--sample-id",
            sid,
            "--max-workers",
            str(int(args.max_workers)),
            "--config",
            str(args.config.resolve()),
        ]
        print(f"\n{'=' * 72}\n[batch] Starting {sid}\n{'=' * 72}", flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            failed.append(sid)
            print(f"[batch] {sid} FAILED (exit {rc})", file=sys.stderr)
            if not args.continue_on_error:
                break
        else:
            print(f"[batch] {sid} OK", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"\n[batch] Done in {elapsed / 3600.0:.2f} h. "
        f"failed={len(failed)} {failed if failed else ''}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
