#!/usr/bin/env python3
"""
Headless orchestrator: master FEM sweep → MMR selection → ROM packaging → LHS pool update.

Step A uses ``fem_master_dynamic.py`` with at most ``--max-workers`` concurrent FEM workers
(default **2**); on Linux each is pinned with ``taskset -c 1..N`` where ``N=max-workers``
before ``mpiexec --bind-to none -n 1``, with a
**10 s** pause before the second worker (mesh load / I/O stagger), plus a **5 s** minimum gap
between any two spawns; core 0 and other CPUs stay for the master and OS.
Candidate merge uses the **conditional adaptive manager** in ``fem_master_dynamic``
(zone wood floors 0.0008→0.0003, sparse overlap scoring, spectral shaping, HF quota).
LHS pool parameters are merged into a
per-sample config before the FEM master runs.

On success: writes snapshot NPZ under ROM/classic/snapshots/, selection plot under
FEM/results/plots/, runs package_rom --cleanup, then marks the sample completed in
the pool JSON. Mode vectors are CSR float32 (relative sparsification) on disk
(``*.smx.npz``); ``package_rom`` bundles the stacked CSR into one compressed NPZ
(``ev_*`` keys + metadata).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from paths import DEFAULT_SHAPE_NAME, FEM_RESULTS_PLOTS_DIR, infer_shape_from_pool_path, shared_plot_path
from wood_library import apply_lhs_parameters_to_config


def _repo_root() -> Path:
    """Walk parents from this file until a directory named ``final-project`` is found."""
    repo = Path(os.path.abspath(__file__)).resolve()
    while repo.name != "final-project" and repo.parent != repo:
        repo = repo.parent
    if repo.name != "final-project":
        raise RuntimeError(
            "Could not locate a parent directory named 'final-project' starting from "
            f"{Path(__file__).resolve()}"
        )
    return repo


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _repo_root()
DEFAULT_CONFIG = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"
DEFAULT_POOL_PRIMARY = REPO_ROOT / "FEM" / "configs" / "lhs_pool.json"
DEFAULT_POOL_FALLBACK = REPO_ROOT / "ROM" / "classic" / "lhs_pool.json"
PLOTS_DIR = FEM_RESULTS_PLOTS_DIR
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


def _find_pool_entry(pool_path: Path, sample_key: str) -> Dict[str, Any]:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Pool file has no 'entries' list: {pool_path}")
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id")) == sample_key:
            return entry
    raise ValueError(f"No pool entry with id={sample_key!r} in {pool_path}")


def _parameters_from_lhs_samples_file(path: Path, sample_key: str) -> Dict[str, Any]:
    """Load flat dotted keys (e.g. ``geometry.thickness``) for ``sample_key`` from auxiliary JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if sample_key in raw and isinstance(raw[sample_key], dict):
            return dict(raw[sample_key])
        entries = raw.get("entries")
        if isinstance(entries, list):
            for ent in entries:
                if isinstance(ent, dict) and str(ent.get("id")) == sample_key:
                    p = ent.get("parameters")
                    return dict(p) if isinstance(p, dict) else {}
    if isinstance(raw, list):
        for ent in raw:
            if isinstance(ent, dict) and str(ent.get("id")) == sample_key:
                p = ent.get("parameters")
                return dict(p) if isinstance(p, dict) else {}
    return {}


def _resolve_sample_parameters(
    sample_key: str,
    pool_entry: Dict[str, Any],
    lhs_samples_path: Optional[Path],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    pe = pool_entry.get("parameters")
    if isinstance(pe, dict):
        params = dict(pe)
    if lhs_samples_path is None:
        return params
    path = lhs_samples_path.resolve()
    if not path.is_file():
        print(
            f"Warning: --lhs-samples not found ({path}); using pool parameters only.",
            file=sys.stderr,
        )
        return params
    extra = _parameters_from_lhs_samples_file(path, sample_key)
    if extra:
        params.update(extra)
    return params


def _apply_dotted_parameters(cfg: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """Merge flat ``a.b.c`` keys into nested dict ``cfg`` (in-place)."""
    for dotted, value in parameters.items():
        if not isinstance(dotted, str) or not dotted.strip():
            continue
        parts = dotted.split(".")
        cur: Any = cfg
        for key in parts[:-1]:
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                cur[key] = {}
            cur = cur[key]
        cur[parts[-1]] = value


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
    parser.add_argument(
        "--lhs-samples",
        type=Path,
        default=None,
        help=(
            "Optional LHS JSON (e.g. lhs_samples.json): ``entries`` list with id/parameters, "
            "or top-level ``sample_XXX`` -> flat dotted keys. Values override the same keys from the pool."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help=(
            "Maximum concurrent workers passed to fem_master_dynamic "
            "(default: 2). Use 1 for debugging or higher values for lab stress tests."
        ),
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=DEFAULT_SHAPE_NAME,
        help="ROM shape name for shared-host export paths (default: classic).",
    )
    args = parser.parse_args()
    if int(args.max_workers) < 1:
        print(f"Error: --max-workers must be >= 1 (got {args.max_workers})", file=sys.stderr)
        return 1

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
    shape_name = str(args.shape).strip() or infer_shape_from_pool_path(pool_path)
    if not pool_path.is_file():
        print(f"Error: pool file not found: {pool_path}", file=sys.stderr)
        return 1

    try:
        pool_entry = _find_pool_entry(pool_path, sample_key)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: could not read pool entry for {sample_key}: {exc}", file=sys.stderr)
        return 1

    parameters = _resolve_sample_parameters(sample_key, pool_entry, args.lhs_samples)
    effective_config = config_path
    if parameters:
        try:
            merged_cfg = copy.deepcopy(json.loads(config_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: could not load base config {config_path}: {exc}", file=sys.stderr)
            return 1
        apply_lhs_parameters_to_config(merged_cfg, parameters)
        merged_dir = REPO_ROOT / "FEM" / "SORTING" / "pipeline_merged_configs"
        merged_dir.mkdir(parents=True, exist_ok=True)
        effective_config = merged_dir / f"{sample_key}.json"
        _atomic_write_json(effective_config, merged_cfg)
        geom = merged_cfg.get("geometry") if isinstance(merged_cfg.get("geometry"), dict) else {}
        th = geom.get("thickness")
        print(f"[pipeline] Wrote merged FEM config -> {effective_config.resolve()}")
        if isinstance(th, (int, float)):
            print(f"[pipeline] geometry.thickness = {float(th)} m ({float(th) * 1000.0:.4f} mm)")
        else:
            print(f"[pipeline] geometry.thickness = {th!r}")
        sys.stdout.flush()
    else:
        print(
            f"[pipeline] No LHS parameters dict for {sample_key}; using base config only: {config_path}",
        )
        sys.stdout.flush()

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

    plot_path = shared_plot_path(_plot_basename(n), shape_name=shape_name)
    local_plot_path = PLOTS_DIR / _plot_basename(n)
    npz_path = SNAPSHOTS_DIR / _snapshot_basename(n)
    snapshot_rel = _snapshot_rel_npz(n)

    t0 = time.perf_counter()

    sorting_root = (REPO_ROOT / "FEM" / "SORTING").resolve()
    step_a = [
        py,
        str(master),
        "--use-mpiexec",
        "--max-workers",
        str(int(args.max_workers)),
        "--config",
        str(effective_config.resolve()),
        "--sorting-root",
        str(sorting_root),
    ]
    if _run_step("Step A — FEM master sweep (subprocess workers)", step_a, REPO_ROOT) != 0:
        return 1

    step_b = [
        py,
        str(tuner),
        "--headless",
        "--plot-out",
        str(plot_path.resolve()),
        "--export",
        str((REPO_ROOT / "FEM" / "SORTING" / "selected_modes.csv").resolve()),
    ]
    if _run_step("Step B — MMR selection (headless CSV + plot)", step_b, REPO_ROOT) != 0:
        return 1
    try:
        import shutil

        local_plot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, local_plot_path)
    except OSError:
        pass

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
        f"  FEM config:    {effective_config.resolve()}\n"
        f"  Final ROM:     {npz_path.resolve()}\n"
        f"  Selection plot:{plot_path.resolve()} (local copy: {local_plot_path.resolve()})\n"
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
