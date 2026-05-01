#!/usr/bin/env python3
"""
Analyze MMR tuner CSV output for ROM / Master–Worker FEM parameter tuning.

Reads selected modes from CSV. If the CSV includes a ``status`` column, rows are
split into Selected vs Rejected; otherwise all rows are treated as Selected and
Rejected modes are inferred from ``candidates_log.json`` (IDs not present in CSV).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_csv_path() -> Path:
    return _project_root() / "FEM" / "SORTING" / "selected_modes.csv"


def _default_candidates_path() -> Path:
    return _project_root() / "FEM" / "SORTING" / "candidates_log.json"


def _lower_col_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def _load_candidates_json(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        raw = list(data.get("candidates", []))
    elif isinstance(data, list):
        raw = list(data)
    else:
        return []
    out: List[Dict] = []
    for c in raw:
        try:
            out.append(
                {
                    "id": int(c.get("id")),
                    "hz": float(c.get("hz")),
                    "wood_participation": float(c.get("wood_participation", 0.0)),
                }
            )
        except (TypeError, ValueError, KeyError):
            continue
    return out


def _split_selected_rejected(
    df: pd.DataFrame,
    candidates_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Returns (selected_df, rejected_df, note).
    rejected_df may be empty if no status column and no usable candidates file.
    """
    cmap = _lower_col_map(df)
    if "status" in cmap:
        stat_col = cmap["status"]
        s = df[stat_col].astype(str).str.strip().str.lower()
        sel_mask = s.isin(
            ("selected", "mmr", "mmr_selected", "mmr selected", "1", "yes", "true", "y")
        )
        selected = df.loc[sel_mask].copy()
        rejected = df.loc[~sel_mask].copy()
        return selected, rejected, "split using CSV ``status`` column"

    # Tuner default: CSV is selected-only; infer rejected from full candidate pool.
    if "id" not in cmap:
        raise ValueError("CSV must contain an ``id`` column (and optionally ``status``).")
    id_col = cmap["id"]
    if "hz" not in cmap:
        raise ValueError("CSV must contain an ``hz`` column.")
    hz_col = cmap["hz"]
    wood_col = cmap.get("wood_participation")
    if wood_col is None:
        raise ValueError("CSV must contain a ``wood_participation`` column.")
    selected = df.copy()
    sel_ids = set(pd.to_numeric(selected[id_col], errors="coerce").dropna().astype(int))

    pool = _load_candidates_json(candidates_path)
    if not pool:
        rej = pd.DataFrame(columns=df.columns)
        return selected, rej, f"no ``status`` column; candidates file missing or empty: {candidates_path}"

    rejected_rows: List[Dict] = []
    for c in pool:
        if int(c["id"]) not in sel_ids:
            rejected_rows.append(
                {
                    id_col: int(c["id"]),
                    hz_col: float(c["hz"]),
                    wood_col: float(c["wood_participation"]),
                }
            )
    rejected = pd.DataFrame(rejected_rows) if rejected_rows else pd.DataFrame(columns=df.columns)
    return (
        selected,
        rejected,
        f"no ``status`` column; rejected inferred from candidates not in CSV ({candidates_path.name})",
    )


def _hz_series(df: pd.DataFrame) -> pd.Series:
    cmap = _lower_col_map(df)
    if "hz" not in cmap:
        raise ValueError("CSV must contain an ``hz`` column.")
    return pd.to_numeric(df[cmap["hz"]], errors="coerce")


def _wood_series(df: pd.DataFrame) -> pd.Series:
    cmap = _lower_col_map(df)
    if "wood_participation" not in cmap:
        raise ValueError("CSV must contain a ``wood_participation`` column.")
    return pd.to_numeric(df[cmap["wood_participation"]], errors="coerce")


def _print_band_density_50hz(selected: pd.DataFrame) -> None:
    hz = _hz_series(selected).dropna().to_numpy(dtype=np.float64)
    if hz.size == 0:
        print("  (no frequency data for selected modes)")
        return
    bin_start = (np.floor(hz / 50.0) * 50.0).astype(int)
    unique, counts = np.unique(bin_start, return_counts=True)
    order = np.argsort(unique)
    for b, cnt in zip(unique[order], counts[order]):
        print(f"  {int(b):5d} – {int(b) + 50:5d} Hz: {int(cnt):3d} mode(s)")


def _print_gap_analysis(selected: pd.DataFrame) -> None:
    hz = _hz_series(selected).dropna().sort_values().to_numpy(dtype=np.float64)
    if hz.size < 2:
        print("  Need at least two selected modes for gap statistics.")
        return
    gaps = np.diff(hz)
    avg = float(np.mean(gaps))
    imax = int(np.argmax(gaps))
    max_gap = float(gaps[imax])
    f_lo, f_hi = float(hz[imax]), float(hz[imax + 1])
    print(f"  Average gap (Δf): {avg:.4f} Hz")
    print(f"  Maximum gap:    {max_gap:.4f} Hz, occurring between {f_lo:.4f} Hz and {f_hi:.4f} Hz")


def _print_quality_metrics(selected: pd.DataFrame, rejected: pd.DataFrame) -> None:
    w_sel = _wood_series(selected).dropna()
    n_sel = int(w_sel.size)
    mean_sel = float(w_sel.mean()) if n_sel else float("nan")

    rcmap = _lower_col_map(rejected)
    if rejected.empty or "wood_participation" not in rcmap:
        w_rej = pd.Series(dtype=float)
    else:
        w_rej = pd.to_numeric(rejected[rcmap["wood_participation"]], errors="coerce").dropna()

    n_rej = int(w_rej.size)
    mean_rej = float(w_rej.mean()) if n_rej else float("nan")

    print(f"  Selected modes count:   {n_sel}")
    print(f"  Rejected modes count:   {n_rej}")
    if n_sel:
        print(f"  Mean wood (selected):   {mean_sel:.6f}")
    else:
        print("  Mean wood (selected):   (no data)")
    if n_rej:
        print(f"  Mean wood (rejected):   {mean_rej:.6f}")
        if n_sel and not np.isnan(mean_sel) and not np.isnan(mean_rej):
            delta = mean_sel - mean_rej
            print(f"  Difference (sel − rej): {delta:+.6f}")
    else:
        print("  Mean wood (rejected):   (no rejected rows; use CSV ``status`` or --candidates)")


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of MMR tuner CSV for ROM / FEM band and step tuning."
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=_default_csv_path(),
        help="Path to tuner CSV (default: FEM/SORTING/selected_modes.csv)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=_default_candidates_path(),
        help="candidates_log.json used when CSV has no status column (default: FEM/SORTING/candidates_log.json)",
    )
    args = parser.parse_args()
    csv_path: Path = args.csv_path
    candidates_path: Path = args.candidates

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    if df.empty:
        print("CSV contains no rows.")
        return 1

    try:
        selected, rejected, split_note = _split_selected_rejected(df, candidates_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()
    print(_hr("═"))
    print("  ROM / MMR selection — statistical report")
    print(_hr("═"))
    print(f"  CSV:        {csv_path}")
    print(f"  Split:      {split_note}")
    print(f"  Selected:   {len(selected)} row(s)  |  Rejected: {len(rejected)} row(s)")
    print()

    print(_hr())
    print("  (a) Band density — selected modes in 50 Hz bins")
    print(_hr())
    _print_band_density_50hz(selected)
    print()

    print(_hr())
    print("  (b) Frequency gap analysis (consecutive selected modes, sorted by Hz)")
    print(_hr())
    _print_gap_analysis(selected)
    print()

    print(_hr())
    print("  (c) Quality — mean wood_participation (raw)")
    print(_hr())
    _print_quality_metrics(selected, rejected)
    print()
    print(_hr("═"))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
