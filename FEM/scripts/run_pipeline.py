#!/usr/bin/env python3
"""
Headless orchestrator: master FEM sweep → MMR selection → ROM packaging.

Stops on first failing subprocess and prints a pipeline summary on success.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"
DEFAULT_NPZ = REPO_ROOT / "FEM" / "SORTING" / "final_guitar_rom.npz"


def _run_step(name: str, cmd: list[str], cwd: Path) -> int:
    print(f"\n{'=' * 72}\n  {name}\n  $ {' '.join(cmd)}\n{'=' * 72}")
    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=str(cwd))
    code = int(completed.returncode)
    if code != 0:
        print(f"\n[PIPELINE ERROR] Step failed: {name} (exit code {code}).", file=sys.stderr)
        sys.stderr.flush()
    return code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fem_master_dynamic → dynamic_filter_tuner (--headless) → package_rom (--cleanup)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="FEM case JSON passed to fem_master_dynamic.py",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    py = sys.executable
    master = REPO_ROOT / "FEM" / "scripts" / "fem_master_dynamic.py"
    tuner = REPO_ROOT / "FEM" / "scripts" / "dynamic_filter_tuner.py"
    packer = REPO_ROOT / "FEM" / "scripts" / "package_rom.py"

    for script, label in ((master, "fem_master_dynamic.py"), (tuner, "dynamic_filter_tuner.py"), (packer, "package_rom.py")):
        if not script.is_file():
            print(f"Error: missing script {label} at {script}", file=sys.stderr)
            return 1

    t0 = time.perf_counter()

    step_a = [
        py,
        str(master),
        "--use-mpiexec",
        "--config",
        str(config_path),
    ]
    if _run_step("Step A — FEM master sweep (subprocess workers)", step_a, REPO_ROOT) != 0:
        return 1

    step_b = [
        py,
        str(tuner),
        "--headless",
    ]
    if _run_step("Step B — MMR selection (headless CSV + plot)", step_b, REPO_ROOT) != 0:
        return 1

    step_c = [
        py,
        str(packer),
        "--cleanup",
    ]
    if _run_step("Step C — Package ROM + workspace cleanup", step_c, REPO_ROOT) != 0:
        return 1

    elapsed = time.perf_counter() - t0
    npz = DEFAULT_NPZ.resolve()
    print(
        f"\n{'=' * 72}\n"
        f"  Pipeline Summary\n"
        f"{'=' * 72}\n"
        f"  Status:        SUCCESS\n"
        f"  Total time:    {elapsed:.1f} s\n"
        f"  Final ROM:     {npz}\n"
        f"{'=' * 72}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
