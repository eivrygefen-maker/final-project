#!/usr/bin/env python3
"""
Single-band LAB sweep (60–480 Hz) for selected LHS samples.

Writes artifacts under ``FEM/results/LAB_RESULTS/sample_XXX/`` (no dual-window
subfolders). Pipeline: ``fem_master_dynamic`` → ``dynamic_filter_tuner`` (150 modes)
→ ``package_rom`` → optional SORTING purge.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from paths import FEM_LAB_RESULTS_DIR, shared_plot_path, shared_rom_csv_path
from wood_library import apply_lhs_parameters_to_config

HZ_MIN_DEFAULT = 60.0
HZ_MAX_DEFAULT = 480.0
DEFAULT_MAX_WORKERS = 2


def _repo_root() -> Path:
    repo = Path(os.path.abspath(__file__)).resolve()
    while repo.name != "final-project" and repo.parent != repo:
        repo = repo.parent
    if repo.name != "final-project":
        raise RuntimeError(
            "Could not locate a parent directory named 'final-project' starting from "
            f"{Path(__file__).resolve()}"
        )
    return repo


def _run_step(name: str, cmd: List[str], cwd: Path) -> int:
    print(f"\n{'=' * 72}\n  {name}\n  $ {' '.join(cmd)}\n{'=' * 72}")
    sys.stdout.flush()
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def _parse_samples(raw: str) -> List[int]:
    out: List[int] = []
    for part in (x.strip() for x in raw.split(",") if x.strip()):
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    dedup_sorted = sorted({int(x) for x in out if int(x) > 0})
    if not dedup_sorted:
        raise ValueError("No valid sample ids were parsed.")
    return dedup_sorted


def _pool_sample_id(n: int) -> str:
    return f"sample_{n:03d}"


def _default_pool_path(repo: Path) -> Path:
    p1 = repo / "FEM" / "configs" / "lhs_pool.json"
    if p1.is_file():
        return p1
    return repo / "ROM" / "classic" / "lhs_pool.json"


def _find_pool_entry(pool_path: Path, sample_key: str) -> Dict[str, Any]:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Pool missing 'entries' list: {pool_path}")
    for e in entries:
        if isinstance(e, dict) and str(e.get("id", "")) == sample_key:
            return e
    raise ValueError(f"Sample {sample_key} not found in pool: {pool_path}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _verify_rom_npz(npz_path: Path, expected_mode_count: int) -> Tuple[bool, str]:
    if not npz_path.is_file():
        return False, f"missing file: {npz_path}"
    need = ("ev_data", "ev_indices", "ev_indptr", "ev_shape", "frequencies", "wood_participations")
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if any(k not in z.files for k in need):
                return False, f"NPZ missing keys {need}"
            shape = tuple(int(x) for x in np.asarray(z["ev_shape"]).ravel())
            ncols = int(shape[1])
            if expected_mode_count > 0 and ncols != expected_mode_count:
                return False, f"NPZ columns {ncols} != CSV rows {expected_mode_count}"
    except OSError as exc:
        return False, str(exc)
    return True, "ok"


def _purge_sorting_workspace(sorting_root: Path, sdir: Path) -> Tuple[Optional[Path], Optional[str]]:
    err_parts: List[str] = []
    for name in ("temp_modes", "temp_results"):
        p = sorting_root / name
        if p.is_dir():
            try:
                shutil.rmtree(p)
            except OSError as exc:
                err_parts.append(f"{name}: {exc}")
    backup: Optional[Path] = None
    log_path = sorting_root / "candidates_log.json"
    if log_path.is_file():
        backup = sdir / "candidates_log.archived.json"
        try:
            shutil.copy2(log_path, backup)
            _write_json(log_path, {"candidates": [], "completed_shift_targets": []})
        except OSError as exc:
            err_parts.append(f"candidates_log: {exc}")
    if err_parts:
        return backup, "; ".join(err_parts)
    return backup, None


def _count_csv_rows(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    return max(0, len(lines) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LAB single-band sweep (60–480 Hz) → FEM/results/LAB_RESULTS/sample_XXX/"
    )
    parser.add_argument("--config", type=Path, default=None, help="Base FEM config (default: FEM/configs/guitar_3d.json).")
    parser.add_argument("--pool", type=Path, default=None, help="LHS pool JSON path.")
    parser.add_argument("--samples", type=str, default="1-7,12-20", help="Sample ids, e.g. 1-7,12-20.")
    parser.add_argument("--hz-min", type=float, default=HZ_MIN_DEFAULT)
    parser.add_argument("--hz-max", type=float, default=HZ_MAX_DEFAULT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--mpiexec", action="store_true")
    parser.add_argument("--no-purge", action="store_true", help="Keep SORTING temp_modes/temp_results after pack.")
    args = parser.parse_args()

    if int(args.max_workers) != 2:
        print("Error: LAB runner enforces --max-workers 2 for OOM safety.", file=sys.stderr)
        return 1

    repo = _repo_root()
    py = sys.executable
    master = repo / "FEM" / "scripts" / "fem_master_dynamic.py"
    tuner = repo / "FEM" / "scripts" / "dynamic_filter_tuner.py"
    packer = repo / "FEM" / "scripts" / "package_rom.py"
    for p in (master, tuner, packer):
        if not p.is_file():
            print(f"Error: missing {p}", file=sys.stderr)
            return 1

    base_config = args.config.resolve() if args.config else (repo / "FEM" / "configs" / "guitar_3d.json")
    pool_path = args.pool.resolve() if args.pool else _default_pool_path(repo)
    if not base_config.is_file() or not pool_path.is_file():
        print("Error: base config or pool not found.", file=sys.stderr)
        return 1

    lab_root = FEM_LAB_RESULTS_DIR.resolve()
    lab_root.mkdir(parents=True, exist_ok=True)
    hz_lo, hz_hi = float(args.hz_min), float(args.hz_max)

    _write_json(
        lab_root / "run_manifest.json",
        {
            "hz_min": hz_lo,
            "hz_max": hz_hi,
            "samples": _parse_samples(str(args.samples)),
            "max_workers": 2,
            "note": "Single contiguous band; no dual-window splits.",
        },
    )

    failures: List[str] = []
    for sid in _parse_samples(str(args.samples)):
        skey = _pool_sample_id(sid)
        sdir = lab_root / skey
        sdir.mkdir(parents=True, exist_ok=True)
        try:
            entry = _find_pool_entry(pool_path, skey)
        except Exception as exc:
            failures.append(f"{skey}: {exc}")
            continue

        merged_cfg = copy.deepcopy(json.loads(base_config.read_text(encoding="utf-8")))
        params = entry.get("parameters", {})
        if isinstance(params, dict):
            apply_lhs_parameters_to_config(merged_cfg, params)
        merged_path = sdir / "merged_config.json"
        _write_json(merged_path, merged_cfg)

        sorting_root = sdir / "SORTING"
        (sorting_root / "temp_modes").mkdir(parents=True, exist_ok=True)
        (sorting_root / "temp_results").mkdir(parents=True, exist_ok=True)
        _write_json(sorting_root / "candidates_log.json", {"candidates": [], "completed_shift_targets": []})

        master_cmd = [
            py,
            str(master),
            "--config",
            str(merged_path),
            "--hz-min",
            str(hz_lo),
            "--hz-max",
            str(hz_hi),
            "--max-workers",
            "2",
            "--sorting-root",
            str(sorting_root),
        ]
        if args.mpiexec:
            master_cmd.append("--use-mpiexec")

        if _run_step(f"{skey} | master {hz_lo:.0f}-{hz_hi:.0f} Hz", master_cmd, repo) != 0:
            failures.append(f"{skey}: master failed")
            continue

        selected_csv = sdir / "selected_modes.csv"
        shared_csv = shared_rom_csv_path(f"{skey}_selected_modes.csv")
        plot_out = shared_plot_path(f"{skey}_selection_plot.png")
        tuner_cmd = [
            py,
            str(tuner),
            "--headless",
            "--candidates",
            str(sorting_root / "candidates_log.json"),
            "--window-min",
            str(hz_lo),
            "--window-max",
            str(hz_hi),
            "--quota",
            "150",
            "--min-selected",
            "150",
            "--adaptive-veto",
            "--adaptive-steps",
            "12",
            "--export",
            str(selected_csv),
            "--metadata-out",
            str(sdir / "selection_metadata.json"),
            "--plot-out",
            str(plot_out),
        ]
        if _run_step(f"{skey} | tuner (150 modes)", tuner_cmd, repo) != 0:
            failures.append(f"{skey}: tuner failed")
            continue

        try:
            shutil.copy2(selected_csv, shared_csv)
        except OSError:
            pass

        rom_npz = sdir / "lab_window_rom.npz"
        if (
            _run_step(
                f"{skey} | package_rom",
                [
                    py,
                    str(packer),
                    "--csv",
                    str(selected_csv),
                    "--out",
                    str(rom_npz),
                    "--sorting-root",
                    str(sorting_root),
                ],
                repo,
            )
            != 0
        ):
            failures.append(f"{skey}: package failed")
            continue

        n_sel = _count_csv_rows(selected_csv)
        ok, msg = _verify_rom_npz(rom_npz, n_sel)
        if not ok:
            failures.append(f"{skey}: NPZ verify: {msg}")
            continue

        archived = None
        purge_err = None
        if not args.no_purge:
            archived, purge_err = _purge_sorting_workspace(sorting_root, sdir)
            if purge_err:
                failures.append(f"{skey}: purge: {purge_err}")
                continue

        _write_json(
            sdir / "sample_manifest.json",
            {
                "sample_id": skey,
                "hz_min": hz_lo,
                "hz_max": hz_hi,
                "selected_count": n_sel,
                "selected_csv": str(selected_csv),
                "shared_csv": str(shared_csv),
                "selection_plot": str(plot_out),
                "rom_npz": str(rom_npz),
                "npz_verify": msg,
                "candidates_log_archived": str(archived) if archived else None,
            },
        )

    status = {"ok": not failures, "failures": failures, "root": str(lab_root)}
    _write_json(lab_root / "run_status.json", status)
    if failures:
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"LAB sweep complete: {lab_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
