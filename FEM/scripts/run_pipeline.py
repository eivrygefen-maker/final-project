#!/usr/bin/env python3
"""
Headless orchestrator: master FEM sweep → MMR selection → ROM packaging → LHS pool update.

Step A uses ``fem_master_dynamic.py`` with at most **2** concurrent FEM workers (each
``mpiexec -n 1 --map-by core --bind-to core`` when ``--use-mpiexec``), leaving spare
cores for the OS and this orchestrator.

On success: writes snapshot NPZ under ROM/classic/snapshots/, selection plot under
FEM/results/plots/, runs package_rom --cleanup, then marks the sample completed in
the pool JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"
DEFAULT_POOL_PRIMARY = REPO_ROOT / "FEM" / "configs" / "lhs_pool.json"
DEFAULT_POOL_FALLBACK = REPO_ROOT / "ROM" / "classic" / "lhs_pool.json"
PLOTS_DIR = REPO_ROOT / "FEM" / "results" / "plots"
SNAPSHOTS_DIR = REPO_ROOT / "ROM" / "classic" / "snapshots"


def _default_pool_path() -> Path:
    if DEFAULT_POOL_PRIMARY.is_file():
        return DEFAULT_POOL_PRIMARY
    return DEFAULT_POOL_FALLBACK


def _parse_sample_index(raw: str) -> int:
    s = raw.strip()
    m = re.match(r"^sample_0*(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    if s.lower().startswith("sample_"):
        return int(s.split("_", 1)[-1])
    return int(s)


def _pool_sample_id(n: int) -> str:
    return f"sample_{n:03d}"


def _snapshot_basename(n: int) -> str:
    return f"snapshot_{n:04d}.npz"


def _plot_basename(n: int) -> str:
    return f"snapshot_{n:04d}.png"


def _snapshot_rel_npz(n: int) -> str:
    """Path as stored in lhs_pool.json (relative to ROM/classic/)."""
    return f"snapshots/{_snapshot_basename(n)}"


def _atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def _update_pool_entry(pool_path: Path, sample_id: str, snapshot_rel: str) -> None:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Pool file has no 'entries' list: {pool_path}")

    found = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id")) != sample_id:
            continue
        entry["status"] = "completed"
        entry["snapshot_file"] = snapshot_rel
        entry["error"] = None
        found = True
        break

    if not found:
        raise ValueError(f"No pool entry with id={sample_id!r} in {pool_path}")

    _atomic_write_json(pool_path, payload)


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
        description="Run fem_master_dynamic → dynamic_filter_tuner (--headless) → package_rom (--cleanup) → pool update."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="FEM case JSON passed to fem_master_dynamic.py",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        required=True,
        help='LHS sample key, e.g. "sample_011" or "11".',
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="lhs_pool.json path (default: FEM/configs/lhs_pool.json if present, else ROM/classic/lhs_pool.json).",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        n = _parse_sample_index(args.sample_id)
    except ValueError as exc:
        print(f"Error: invalid --sample-id {args.sample_id!r}: {exc}", file=sys.stderr)
        return 1

    sample_key = _pool_sample_id(n)
    pool_path = (args.pool.resolve() if args.pool is not None else _default_pool_path())
    if not pool_path.is_file():
        print(f"Error: pool file not found: {pool_path}", file=sys.stderr)
        return 1

    py = sys.executable
    master = REPO_ROOT / "FEM" / "scripts" / "fem_master_dynamic.py"
    tuner = REPO_ROOT / "FEM" / "scripts" / "dynamic_filter_tuner.py"
    packer = REPO_ROOT / "FEM" / "scripts" / "package_rom.py"

    for script, label in ((master, "fem_master_dynamic.py"), (tuner, "dynamic_filter_tuner.py"), (packer, "package_rom.py")):
        if not script.is_file():
            print(f"Error: missing script {label} at {script}", file=sys.stderr)
            return 1

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_path = PLOTS_DIR / _plot_basename(n)
    npz_path = SNAPSHOTS_DIR / _snapshot_basename(n)
    snapshot_rel = _snapshot_rel_npz(n)

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
        "--plot-out",
        str(plot_path.resolve()),
    ]
    if _run_step("Step B — MMR selection (headless CSV + plot)", step_b, REPO_ROOT) != 0:
        return 1

    step_c = [
        py,
        str(packer),
        "--out",
        str(npz_path.resolve()),
        "--cleanup",
    ]
    if _run_step("Step C — Package ROM + workspace cleanup", step_c, REPO_ROOT) != 0:
        return 1

    try:
        _update_pool_entry(pool_path, sample_key, snapshot_rel)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"\n[PIPELINE ERROR] Failed to update pool file: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    print(
        f"\n{'=' * 72}\n"
        f"  Pipeline Summary\n"
        f"{'=' * 72}\n"
        f"  Status:        SUCCESS\n"
        f"  Total time:    {elapsed:.1f} s\n"
        f"  Sample:        {sample_key}\n"
        f"  Final ROM:     {npz_path.resolve()}\n"
        f"  Selection plot:{plot_path.resolve()}\n"
        f"  Pool file:     {pool_path.resolve()}\n"
        f"{'=' * 72}\n"
    )
    print(
        f'Simulation [{sample_key}] completed successfully. '
        f"Status updated in {pool_path.as_posix()}. "
        f"ROM saved to {npz_path.as_posix()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
