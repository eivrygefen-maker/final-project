#!/usr/bin/env python3
"""One-shot workspace reset for pipeline refactor (Phase 1)."""
from __future__ import annotations

import shutil
from pathlib import Path

from paths import FEM_SORTING_DIR, REPO_ROOT


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
