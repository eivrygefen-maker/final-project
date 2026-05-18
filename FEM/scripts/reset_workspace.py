#!/usr/bin/env python3
"""One-shot workspace reset for pipeline refactor (Phase 1)."""
from __future__ import annotations

import shutil
from pathlib import Path

from paths import (
    FEM_RESULTS_PLOTS_DIR,
    FEM_SORTING_DIR,
    REPO_ROOT,
    ROM_CLASSIC_SNAPSHOTS_DIR,
    SHARED_HOST_DIR,
    normalize_shape_name,
)


def _clear_directory_files(directory: Path) -> int:
    """Remove all files under ``directory``; keep subdirectories."""
    removed = 0
    if not directory.is_dir():
        return 0
    for p in directory.rglob("*"):
        if p.is_file():
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _unlink_if_exists(path: Path) -> bool:
    if path.is_file():
        path.unlink()
        print(f"Deleted: {path}")
        return True
    return False


def _clean_sample_001_artifacts() -> None:
    """Remove snapshot_0001 / sample_001 outputs so a new LHS run starts clean."""
    shape = normalize_shape_name("classic")
    patterns = (
        "snapshot_0001*",
        "sample_001*",
        "simulation_v1*",
    )
    for base in (FEM_RESULTS_PLOTS_DIR, SHARED_HOST_DIR / shape / "plots"):
        if not base.is_dir():
            continue
        for pat in patterns:
            for p in base.glob(pat):
                if p.is_file():
                    p.unlink()
                    print(f"Deleted: {p}")

    snap = ROM_CLASSIC_SNAPSHOTS_DIR / "snapshot_0001.npz"
    _unlink_if_exists(snap)

    sample_cfg = FEM_SORTING_DIR / "pipeline_merged_configs" / "sample_001.json"
    _unlink_if_exists(sample_cfg)


def main() -> int:
    extra = REPO_ROOT / "FEM" / "results" / "EXTRA_RESULTS"
    if extra.is_dir():
        shutil.rmtree(extra)
        print(f"Removed: {extra}")

    sorting = FEM_SORTING_DIR
    sorting.mkdir(parents=True, exist_ok=True)
    (sorting / "pipeline_merged_configs").mkdir(parents=True, exist_ok=True)
    (sorting / "temp_modes").mkdir(parents=True, exist_ok=True)
    (sorting / "temp_results").mkdir(parents=True, exist_ok=True)
    n = _clear_directory_files(sorting)
    print(f"Cleared {n} file(s) under {sorting} (directories kept).")

    _clean_sample_001_artifacts()

    classic_snapshots = ROM_CLASSIC_SNAPSHOTS_DIR
    if classic_snapshots.is_dir():
        for npz in classic_snapshots.glob("snapshot_*.npz"):
            npz.unlink()
            print(f"Deleted: {npz}")

    for rel in (
        "ROM/classic/lhs_pool.json",
        "FEM/configs/lhs_samples.json",
    ):
        p = REPO_ROOT / rel
        if p.is_file():
            p.unlink()
            print(f"Deleted: {p}")
        else:
            print(f"Already absent: {p}")

    lab = REPO_ROOT / "FEM" / "results" / "LAB_RESULTS"
    lab.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {lab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
