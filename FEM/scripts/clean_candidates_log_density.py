#!/usr/bin/env python3
"""
Clean a poisoned ``candidates_log.json`` by removing dense numerical clusters in ``hz``.

Default rule (matches merge-time density ceiling in ``fem_master_dynamic``): if more than
``--max-modes`` candidates lie within a sliding ``--span-hz`` window, keep only the best
``--keep-per-window`` row(s) per window by ``uniqueness`` (then wood_participation), and drop
the rest (optionally deleting their ``vector_path`` files under ``--sorting-root``).

Usage (from repo root)::

    py FEM/scripts/clean_candidates_log_density.py --log FEM/SORTING/candidates_log.json --sorting-root FEM/SORTING --dry-run
    py FEM/scripts/clean_candidates_log_density.py --log path/to/candidates_log.json --sorting-root path/to/SORTING
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _hz(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _uniq(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("uniqueness", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _wood(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("wood_participation", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _score(row: Dict[str, Any]) -> Tuple[float, float]:
    return (_uniq(row), _wood(row))


def prune_dense_clusters(
    rows: List[Dict[str, Any]],
    *,
    span_hz: float,
    max_modes: int,
    keep_per_window: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kept, removed) where *removed* violated the density ceiling."""
    if not rows or max_modes < 1 or span_hz <= 0.0:
        return list(rows), []

    idxs = sorted(range(len(rows)), key=lambda i: _hz(rows[i]))
    removed_idx: set[int] = set()
    i = 0
    while i < len(idxs):
        f_start = _hz(rows[idxs[i]])
        j = i
        while j + 1 < len(idxs) and _hz(rows[idxs[j + 1]]) - f_start <= span_hz + 1e-12:
            j += 1
        block = idxs[i : j + 1]
        if len(block) > max_modes:
            ranked = sorted(block, key=lambda idx: _score(rows[idx]), reverse=True)
            keep_set = set(ranked[: max(1, keep_per_window)])
            for b in block:
                if b not in keep_set:
                    removed_idx.add(b)
        i = j + 1

    kept = [rows[i] for i in range(len(rows)) if i not in removed_idx]
    removed = [rows[i] for i in sorted(removed_idx)]
    return kept, removed


def main() -> int:
    p = argparse.ArgumentParser(description="Prune dense hz clusters in candidates_log.json")
    p.add_argument("--log", type=Path, required=True, help="Path to candidates_log.json")
    p.add_argument(
        "--sorting-root",
        type=Path,
        default=None,
        help="Workspace root for deleting orphan mode vectors (temp_modes/…).",
    )
    p.add_argument("--span-hz", type=float, default=1.0, help="Sliding window width in Hz.")
    p.add_argument(
        "--max-modes",
        type=int,
        default=20,
        help="If a window contains more than this many modes, prune the excess.",
    )
    p.add_argument(
        "--keep-per-window",
        type=int,
        default=3,
        help="How many best-scoring modes to retain per violating window.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions only; do not write files.")
    args = p.parse_args()
    log_path = args.log.resolve()
    if not log_path.is_file():
        print(f"Error: log not found: {log_path}", flush=True)
        return 1

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    cands = list(payload.get("candidates") or [])
    kept, removed = prune_dense_clusters(
        cands,
        span_hz=float(args.span_hz),
        max_modes=int(args.max_modes),
        keep_per_window=int(args.keep_per_window),
    )
    print(f"Candidates: {len(cands)} -> keep {len(kept)}, remove {len(removed)}", flush=True)
    if removed and len(removed) <= 20:
        for r in removed:
            print(f"  drop id={r.get('id')} hz={_hz(r):.6f} uniq={_uniq(r):.4f}", flush=True)
    elif removed:
        print(f"  (first of {len(removed)} drops) id={removed[0].get('id')} hz={_hz(removed[0]):.6f}", flush=True)

    if args.dry_run:
        print("Dry-run: no files modified.", flush=True)
        return 0

    root = args.sorting_root.resolve() if args.sorting_root is not None else None
    if root is not None:
        for r in removed:
            rel = Path(str(r.get("vector_path", "") or ""))
            if not rel.parts:
                continue
            vp = (root / rel).resolve()
            try:
                if vp.is_file() and root in vp.parents:
                    vp.unlink()
            except OSError as exc:
                print(f"Warning: could not unlink {vp}: {exc}", flush=True)

    payload["candidates"] = kept
    backup = log_path.with_suffix(log_path.suffix + ".bak")
    shutil.copy2(log_path, backup)
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(log_path)
    print(f"Wrote {log_path} (backup {backup})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
