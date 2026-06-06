#!/usr/bin/env python3
"""Standalone M4 ROM vs FOM comparison on completed LHS samples (no FOM rerun)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import DEFAULT_RUN_ID_SUFFIX, load_lhs_pool  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    DEFAULT_MAX_MATCH_DISTANCE_HZ,
    DEFAULT_ROM_NEV,
    maybe_run_rom_compare,
    maybe_run_rom_prepredict,
    resolve_sample_context,
    select_completed_lhs_for_rom_compare,
    sync_lhs_pool_rom_fields,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ROM pre-prediction and/or ROM/FOM comparison for M4 completed samples. "
            "Reads FOM frequencies from aggregation/modes_catalog.jsonl (not legacy FOM)."
        ),
        epilog=(
            "Examples:\n"
            "  python FEM/.../run_m4_rom_compare.py --lhs-json ROM/classic/lhs_pool.json "
            "--force-sample sample_005\n"
            "  python FEM/.../run_m4_rom_compare.py --lhs-json ROM/classic/lhs_pool.json "
            "--completed-only --max-samples 10 --write-csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--completed-only", action="store_true", default=True)
    parser.add_argument("--include-incomplete", action="store_false", dest="completed_only")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-sample", help="Run only this sample_id (e.g. sample_005).")
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX)
    parser.add_argument("--nev", type=int, default=DEFAULT_ROM_NEV, help="ROM modes (0 = all basis).")
    parser.add_argument(
        "--rom-max-match-distance-hz",
        type=float,
        default=DEFAULT_MAX_MATCH_DISTANCE_HZ,
        help="Greedy match tolerance in Hz.",
    )
    parser.add_argument(
        "--run-prepredict",
        action="store_true",
        help="Run ROM pre-prediction even when only comparing (refresh prediction).",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip explicit prepredict step (compare will rerun ROM if prediction missing).",
    )
    parser.add_argument("--write-csv", action="store_true", help="Also write per-mode CSV under ROM/.../comparisons/.")
    parser.add_argument("--no-project-copy", action="store_true", help="Keep comparison only under run_dir/rom/.")
    parser.add_argument("--dry-run", action="store_true", help="List selected samples only.")
    return parser.parse_args(argv)


def _resolve_lhs(repo_root: Path, arg: Path) -> Path:
    return arg if arg.is_absolute() else repo_root / arg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = _resolve_lhs(repo_root, args.lhs_json)
    if not lhs_path.is_file():
        print(f"error: missing --lhs-json: {lhs_path}", file=sys.stderr)
        return 2

    try:
        pool = load_lhs_pool(lhs_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selection = select_completed_lhs_for_rom_compare(
        pool,
        completed_only=bool(args.completed_only),
        max_samples=args.max_samples,
        force_sample=args.force_sample,
        run_id_suffix=str(args.run_id_suffix),
    )
    if not selection:
        print("no samples selected")
        return 0

    print(f"selected_count={len(selection)} lhs_json={rel(lhs_path, repo_root=repo_root)}")
    for row in selection:
        print(f"  {row['sample_id']} -> {row['run_id']} (lhs_index={row['lhs_row_index']})")

    if args.dry_run:
        print("dry_run=true")
        return 0

    ok = 0
    failed = 0
    for row in selection:
        sid = str(row["sample_id"])
        rid = str(row["run_id"])
        context = resolve_sample_context(
            pool=pool,
            sample_id=sid,
            run_id=rid,
            repo_root=repo_root,
        )
        run_root = context["run_root"]
        print(f"[rom] {sid} / {rid} ...", flush=True)

        if args.run_prepredict and not args.compare_only:
            prep = maybe_run_rom_prepredict(
                repo_root=repo_root,
                run_root=run_root,
                context=context,
                nev=int(args.nev),
                nonblocking=True,
            )
            print(
                f"  prepredict status={prep.get('status')} "
                f"modes={len(prep.get('frequencies_hz') or [])} "
                f"path={prep.get('path')}",
                flush=True,
            )

        cmp_result = maybe_run_rom_compare(
            repo_root=repo_root,
            run_root=run_root,
            context=context,
            nev=int(args.nev),
            max_match_distance_hz=float(args.rom_max_match_distance_hz),
            nonblocking=True,
            copy_to_project=not bool(args.no_project_copy),
            write_csv=bool(args.write_csv),
            rerun_rom_if_missing=True,
        )
        comparison = cmp_result.get("comparison")
        if comparison and comparison.get("status") == "COMPLETED":
            ok += 1
            print(
                f"  compare status=COMPLETED matched={comparison.get('matched_mode_count')} "
                f"mean_abs_hz={comparison.get('mean_abs_error_hz')} "
                f"path={comparison.get('last_rom_comparison_path')}",
                flush=True,
            )
        else:
            failed += 1
            print(f"  compare failed: {cmp_result.get('error')}", flush=True)

        lhs_patch = cmp_result.get("lhs_patch")
        if lhs_patch:
            sync_lhs_pool_rom_fields(pool, sample_id=sid, lhs_patch=lhs_patch, lhs_path=lhs_path)

    print(f"compared_ok={ok} compared_failed={failed}")
    print(f"lhs_pool={rel(lhs_path, repo_root=repo_root)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
