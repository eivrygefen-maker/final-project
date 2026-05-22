#!/usr/bin/env python3
"""
Plot packaged modes for snapshot_0001 (sample_001) as mode index vs wood participation.

Data source: ``FEM/SORTING/selected_modes.csv`` — same rows written by the pipeline tuner
and packaged into ``ROM/classic/snapshots/snapshot_0001.npz`` (dynamic basis size).

Output (shared host only): ``{SHARED_HOST_DIR}/classic/plots/snapshot_0001_fixed_scale.png``
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paths import DEFAULT_SHAPE_NAME, shared_plot_path  # noqa: E402

# Shared export — override with: export SHARED_HOST_DIR=/media/sf_gmar
OUTPUT_NAME = "snapshot_0001_fixed_scale.png"
PHYSICAL_WOOD_MAX = 1.0
Y_LIMIT_PADDING = 1.08


def _repo_root() -> Path:
    return SCRIPT_DIR.parents[1]


def _load_snapshot_0001_modes(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mode_index, wood_participation, frequency_hz) for exactly the packaged modes."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    wood = np.array([float(r["wood_participation"]) for r in rows], dtype=np.float64)
    hz = np.array([float(r["hz"]) for r in rows], dtype=np.float64)
    mode_idx = np.arange(len(rows), dtype=np.int64)
    return mode_idx, wood, hz


def _y_upper_from_modes(wood: np.ndarray) -> float:
    """
    Y-axis top from the maximum *physical* wood value in this snapshot only.

    Ignores non-finite, negative, and non-physical values (> ``PHYSICAL_WOOD_MAX``).
    All modes in the CSV are used — no cross-run quantile cap.
    """
    valid = wood[np.isfinite(wood) & (wood >= 0.0) & (wood <= PHYSICAL_WOOD_MAX)]
    if valid.size == 0:
        return PHYSICAL_WOOD_MAX
    return float(np.max(valid)) * Y_LIMIT_PADDING


def main() -> int:
    repo = _repo_root()
    csv_path = repo / "FEM" / "SORTING" / "selected_modes.csv"
    if not csv_path.is_file():
        print(f"Error: missing {csv_path}", file=sys.stderr)
        return 1

    mode_idx, wood, hz = _load_snapshot_0001_modes(csv_path)
    n = int(mode_idx.size)
    print(f"Packaged modes in CSV: {n}")

    y_top = _y_upper_from_modes(wood)
    out_path = shared_plot_path(OUTPUT_NAME, shape_name=DEFAULT_SHAPE_NAME)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(mode_idx, wood, width=0.85, color="#2e7d32", edgecolor="#1b5e20", linewidth=0.35, alpha=0.92)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Wood participation")
    ax.set_title(f"snapshot_0001 — {n} packaged modes (sample_001)")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0.0, y_top)
    ax.grid(True, axis="y", alpha=0.3)
    fig.text(
        0.01,
        0.01,
        f"Source: {csv_path.name} | f ∈ [{hz.min():.1f}, {hz.max():.1f}] Hz | Y cap={y_top:.6f}",
        fontsize=8,
        color="gray",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path.resolve()), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Data file: {csv_path.resolve()}")
    print(f"Modes plotted: {n}")
    print(f"Wood range (raw): [{wood.min():.6f}, {wood.max():.6f}]")
    print(f"Y-axis limit (0, {y_top:.6f})")
    print(f"Saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
