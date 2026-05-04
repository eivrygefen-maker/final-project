#!/usr/bin/env python3
"""
Regenerate MMR-style selection plots (Frequency vs. wood participation) from packaged
``snapshot_XXXX.npz`` files when the corresponding PNGs are missing.

The pipeline (``package_rom``) stores **MMR-selected** modes only: ``frequencies`` and
``wood_participations``. The live tuner plot also shows **rejected** candidates (red);
that set is not stored in the NPZ, so regenerated figures show **green selected points
only**—same marker style, axis labels, and colors as ``dynamic_filter_tuner._plot_selection``.

Some older FOM snapshots use ``freqs_hz`` + ``participation_ratios``; both are supported.

Example::

    py FEM/scripts/reproduce_plots.py --start 1 --end 12
    py FEM/scripts/reproduce_plots.py --start 1 --end 19 --skip-existing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _repo_root() -> Path:
    repo = Path(os.path.abspath(__file__)).resolve()
    while repo.name != "final-project" and repo.parent != repo:
        repo = repo.parent
    if repo.name != "final-project":
        raise RuntimeError(
            "Could not locate parent directory named 'final-project' starting from "
            f"{Path(__file__).resolve()}"
        )
    return repo


REPO_ROOT = _repo_root()

from dynamic_filter_tuner import (  # noqa: E402
    LAMBDA_VAL,
    SIGMA_HZ,
    U,
    UNIQUENESS_VETO_MIN,
    W,
    WOOD_FILTER_MIN,
    _plot_selection,
)


def _load_snapshot_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "frequencies" in files and "wood_participations" in files:
            hz = np.asarray(z["frequencies"], dtype=np.float64).ravel()
            wood = np.asarray(z["wood_participations"], dtype=np.float64).ravel()
            return hz, wood, "pipeline_package_rom"
        if "freqs_hz" in files and "participation_ratios" in files:
            hz = np.asarray(z["freqs_hz"], dtype=np.float64).ravel()
            wood = np.asarray(z["participation_ratios"], dtype=np.float64).ravel()
            return hz, wood, "rom_fom"
    raise ValueError(
        f"{path.name}: expected (frequencies, wood_participations) or "
        f"(freqs_hz, participation_ratios); got {sorted(files)}"
    )


def _to_selected_candidates(hz: np.ndarray, wood: np.ndarray) -> List[Dict[str, Any]]:
    if hz.shape != wood.shape:
        raise ValueError(f"hz shape {hz.shape} != wood shape {wood.shape}")
    out: List[Dict[str, Any]] = []
    for i in range(int(hz.size)):
        out.append(
            {
                "id": i + 1,
                "hz": float(hz[i]),
                "wood_participation": float(wood[i]),
                "uniqueness": 1.0,
                "tag1_ratio": 0.0,
                "tag3_ratio": 0.0,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate snapshot_XXXX.png from snapshot NPZ files.")
    ap.add_argument("--start", type=int, default=1, help="First snapshot index (default 1).")
    ap.add_argument("--end", type=int, default=12, help="Last snapshot index inclusive (default 12).")
    ap.add_argument(
        "--snapshots-dir",
        type=Path,
        default=REPO_ROOT / "ROM" / "classic" / "snapshots",
        help="Directory containing snapshot_XXXX.npz",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=REPO_ROOT / "FEM" / "results" / "plots",
        help="Output directory for snapshot_XXXX.png",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="If the target snapshot_XXXX.png already exists, do not overwrite it.",
    )
    args = ap.parse_args()

    snapshots_dir = args.snapshots_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    lo = int(args.start)
    hi = int(args.end)
    if hi < lo:
        print("error: --end must be >= --start", file=sys.stderr)
        return 2

    for idx in range(lo, hi + 1):
        snap_name = f"snapshot_{idx:04d}.npz"
        plot_name = f"snapshot_{idx:04d}.png"
        npz_path = snapshots_dir / snap_name
        out_path = plots_dir / plot_name

        if args.skip_existing and out_path.is_file():
            print(f"[skip] {plot_name} already exists")
            continue

        if not npz_path.is_file():
            print(f"[skip] missing {npz_path}")
            continue

        try:
            hz, wood, fmt = _load_snapshot_arrays(npz_path)
        except (OSError, ValueError) as exc:
            print(f"[error] {npz_path}: {exc}", file=sys.stderr)
            return 1

        selected = _to_selected_candidates(hz, wood)
        rejected: List[Dict[str, Any]] = []

        title = (
            f"MMR tuner | selected={len(selected)} rejected=0 | "
            f"W={W}, U={U}, λ={LAMBDA_VAL}, σ={SIGMA_HZ} Hz | "
            f"vetoes: wood≥{WOOD_FILTER_MIN}, uniqueness≥{UNIQUENESS_VETO_MIN} | "
            f"{plot_name} (regenerated from {snap_name}, {fmt})"
        )

        _plot_selection(selected, rejected, title, headless=True, save_path=out_path)
        print(f"[ok] {out_path}  (N={len(selected)}, {fmt})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
