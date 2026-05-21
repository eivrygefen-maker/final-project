#!/usr/bin/env python3
"""
Post-solve ROM harvest filter for ``temp_results/result_*.json``.

Classifies modes (physical FSI vs σ-locked Ritz) and writes annotated JSON with
``rom_ready_candidates`` for downstream merge / packaging.

Examples::

  # Audit one shift result
  python FEM/scripts/filter_harvest_results.py \\
    --sorting-root FEM/SORTING \\
    --result FEM/SORTING/temp_results/result_155000.json

  # Batch-annotate all pending results (in-place *.rom.json sidecars)
  python FEM/scripts/filter_harvest_results.py \\
    --sorting-root FEM/SORTING --all --write-sidecars

  # Use thresholds from merged sample config
  python FEM/scripts/filter_harvest_results.py \\
    --config FEM/SORTING/pipeline_merged_configs/sample_001.json \\
    --sorting-root FEM/SORTING --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fem_harvest_filter import (
    HarvestFilterConfig,
    filter_result_file,
    filter_result_payload,
    load_filter_config_from_json,
)


def _print_summary(path: Path, payload: dict) -> None:
    sm = payload.get("harvest_filter_summary") or {}
    print(f"\n=== {path.name} ===")
    print(
        f"  target={sm.get('target_hz')} Hz  st_sigma={sm.get('st_sigma_hz')} Hz  "
        f"incoming={sm.get('n_incoming')}  rom_ready={sm.get('n_rom_ready')}"
    )
    counts = sm.get("class_counts") or {}
    if counts:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  classes: {parts}")
    for row in payload.get("rom_ready_candidates") or []:
        print(
            f"    ROM  f={float(row.get('hz', 0)):.4f} Hz  p_frac={float(row.get('p_frac', 0)):.3f}  "
            f"wood={float(row.get('wood_participation', 0)):.3f}  "
            f"class={row.get('harvest_class')}  uniq={float(row.get('uniqueness', 0)):.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter worker harvest JSON for ROM-ready modes.")
    parser.add_argument(
        "--sorting-root",
        type=Path,
        default=SCRIPT_DIR.parent / "SORTING",
        help="Workspace with temp_results/ (default: FEM/SORTING).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Merged FEM JSON; reads solver.harvest_filter block.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="Single result_*.json path.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every temp_results/result_*.json (skip *.rom.json).",
    )
    parser.add_argument(
        "--write-sidecars",
        action="store_true",
        help="Write result_<tag>.rom.json next to each result file.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite result_*.json with annotated payload (rom_ready fields).",
    )
    args = parser.parse_args()

    cfg = (
        load_filter_config_from_json(args.config.resolve())
        if args.config is not None
        else HarvestFilterConfig()
    )

    paths: List[Path] = []
    if args.result is not None:
        paths.append(args.result.resolve())
    if args.all:
        tr = args.sorting_root.resolve() / "temp_results"
        paths.extend(
            sorted(
                p
                for p in tr.glob("result_*.json")
                if ".rom." not in p.name
            )
        )
    if not paths:
        print("Error: specify --result PATH or --all", file=sys.stderr)
        return 1

    n_ok = 0
    for path in paths:
        if not path.is_file():
            print(f"[skip] missing {path}", file=sys.stderr)
            continue
        write_rom = None
        if args.write_sidecars:
            write_rom = path.with_name(path.stem + ".rom.json")
        filtered = filter_result_file(path, cfg=cfg, write_rom_path=write_rom)
        if args.in_place:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
            tmp.replace(path)
        _print_summary(path, filtered)
        n_ok += 1

    print(f"\n[filter_harvest] processed {n_ok} result file(s).")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
