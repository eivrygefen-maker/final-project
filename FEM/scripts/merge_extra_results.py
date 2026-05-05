#!/usr/bin/env python3
"""
Manual merge utility for EXTRA_RESULTS targeted LAB outputs.

This script is command-driven and does not run automatically from targeted sweeps.
It builds curated per-sample selections by taking top-K from each target zone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for r in reader:
            try:
                rr = dict(r)
                rr["id"] = int(float(rr.get("id", "0")))
                rr["hz"] = float(rr.get("hz", "0"))
                rr["wood_participation"] = float(rr.get("wood_participation", "0"))
                rr["uniqueness"] = float(rr.get("uniqueness", "0"))
                rr["Q_mmr_base"] = float(rr.get("Q_mmr_base", "0"))
                rows.append(rr)
            except Exception:
                continue
    return rows


def _zone_for_hz(hz: float) -> str:
    if 80.0 <= hz <= 100.0:
        return "low_080_100"
    if 400.0 <= hz <= 600.0:
        return "high_400_600"
    return "other"


def _score(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("Q_mmr_base", 0.0)),
        float(row.get("wood_participation", 0.0)),
        float(row.get("uniqueness", 0.0)),
    )


def _sample_dirs(extra_root: Path) -> List[Path]:
    return sorted([p for p in extra_root.glob("sample_*") if p.is_dir()])


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["sample_id", "zone", "id", "hz", "wood_participation", "uniqueness", "Q_mmr_base", "source_csv"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manually combine targeted EXTRA_RESULTS selections. "
            "Picks top-K per zone (default K=30) for each sample."
        )
    )
    parser.add_argument(
        "--extra-root",
        type=Path,
        default=Path("FEM/results/EXTRA_RESULTS"),
        help="Root directory containing sample_XXX targeted outputs (default: FEM/results/EXTRA_RESULTS).",
    )
    parser.add_argument(
        "--per-zone-top-k",
        type=int,
        default=30,
        help="Top K candidates per zone per sample (default: 30).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("FEM/results/EXTRA_RESULTS/merged_targeted_training_set.csv"),
        help="Output merged CSV path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("FEM/results/EXTRA_RESULTS/merged_targeted_summary.json"),
        help="Output summary JSON path.",
    )
    args = parser.parse_args()

    extra_root = args.extra_root.resolve()
    if not extra_root.is_dir():
        print(f"Error: extra-root not found: {extra_root}")
        return 1
    k = max(1, int(args.per_zone_top_k))

    merged_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"extra_root": str(extra_root), "per_zone_top_k": k, "samples": {}}

    for sdir in _sample_dirs(extra_root):
        sample_id = sdir.name
        csvs = sorted(sdir.glob("window_*/selected_modes.csv"))
        if not csvs:
            continue
        zone_rows: Dict[str, List[Dict[str, Any]]] = {"low_080_100": [], "high_400_600": []}
        for csv_path in csvs:
            rows = _read_csv_rows(csv_path)
            for r in rows:
                zone = _zone_for_hz(float(r["hz"]))
                if zone not in zone_rows:
                    continue
                rr = dict(r)
                rr["sample_id"] = sample_id
                rr["zone"] = zone
                rr["source_csv"] = str(csv_path.resolve())
                zone_rows[zone].append(rr)

        sample_kept = 0
        summary["samples"][sample_id] = {}
        for zone, rows in zone_rows.items():
            rows_sorted = sorted(rows, key=_score, reverse=True)
            top_rows = rows_sorted[:k]
            merged_rows.extend(top_rows)
            summary["samples"][sample_id][zone] = {
                "available": len(rows),
                "selected": len(top_rows),
            }
            sample_kept += len(top_rows)
        summary["samples"][sample_id]["total_selected"] = sample_kept

    _write_csv(args.out_csv.resolve(), merged_rows)
    args.out_json.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Merged rows: {len(merged_rows)}")
    print(f"CSV:  {args.out_csv.resolve()}")
    print(f"JSON: {args.out_json.resolve()}")
    print("This merge is manual/explicit and does not modify master datasets automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

