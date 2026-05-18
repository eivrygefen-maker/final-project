#!/usr/bin/env python3
"""
Merge per-sample LAB/ROM selection CSVs into one training table.

Reads ``FEM/results/LAB_RESULTS/sample_XXX/selected_modes.csv`` (single-band layout).
Legacy dual-window paths under ``EXTRA_RESULTS`` are no longer used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from paths import FEM_LAB_RESULTS_DIR, shared_rom_csv_path


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


def _score(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("Q_mmr_base", 0.0)),
        float(row.get("wood_participation", 0.0)),
        float(row.get("uniqueness", 0.0)),
    )


def _sample_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("sample_*") if p.is_dir()])


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "sample_id",
        "id",
        "hz",
        "wood_participation",
        "uniqueness",
        "Q_mmr_base",
        "source_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge LAB_RESULTS selected_modes.csv files.")
    parser.add_argument(
        "--lab-root",
        type=Path,
        default=FEM_LAB_RESULTS_DIR,
        help="Root with sample_XXX/selected_modes.csv (default: FEM/results/LAB_RESULTS).",
    )
    parser.add_argument(
        "--per-sample-top-k",
        type=int,
        default=0,
        help="If >0, keep only top K modes per sample by MMR score (0 = keep all).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV (default: shared host merged_targeted_training_set.csv).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Summary JSON alongside out-csv.",
    )
    args = parser.parse_args()

    lab_root = args.lab_root.resolve()
    if not lab_root.is_dir():
        print(f"Error: lab-root not found: {lab_root}")
        return 1

    out_csv = args.out_csv.resolve() if args.out_csv else shared_rom_csv_path("merged_lab_training_set.csv")
    out_json = args.out_json.resolve() if args.out_json else out_csv.with_suffix(".summary.json")
    k = max(0, int(args.per_sample_top_k))

    merged_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"lab_root": str(lab_root), "per_sample_top_k": k, "samples": {}}

    for sdir in _sample_dirs(lab_root):
        sample_id = sdir.name
        csv_path = sdir / "selected_modes.csv"
        if not csv_path.is_file():
            continue
        rows = _read_csv_rows(csv_path)
        n_avail = len(rows)
        for r in rows:
            r["sample_id"] = sample_id
            r["source_csv"] = str(csv_path.resolve())
        if k > 0:
            rows = sorted(rows, key=_score, reverse=True)[:k]
        merged_rows.extend(rows)
        summary["samples"][sample_id] = {"available": n_avail, "selected": len(rows)}

    _write_csv(out_csv, merged_rows)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Merged rows: {len(merged_rows)}")
    print(f"CSV:  {out_csv}")
    print(f"JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
