#!/usr/bin/env python3
"""Build lightweight M4 modal frequency surrogate from completed FOM aggregation outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import DEFAULT_RUN_ID_SUFFIX, load_lhs_pool  # noqa: E402
from v2_b3_m4_modal_surrogate_lib import (  # noqa: E402
    DEFAULT_K_NEIGHBORS,
    build_surrogate_from_training_rows,
    collect_completed_fom_training_rows,
    save_surrogate_model,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train M4 Phase-1 ROM surrogate from completed FOM modes_catalog.jsonl files. "
            "Writes ROM/<shape>/m4_modal_surrogate.{json,npz} — no eigenvectors required."
        ),
        epilog=(
            "Example:\n"
            "  python FEM/.../build_m4_rom_from_completed_fom.py "
            "--lhs-json ROM/classic/lhs_pool.json --shape-name classic "
            "--completed-only --max-samples 16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--shape-name", default="classic")
    parser.add_argument("--completed-only", action="store_true", default=True)
    parser.add_argument("--include-incomplete", action="store_false", dest="completed_only")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--min-mode-count", type=int, default=10)
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        help="Exclude sample_id(s) from training (repeatable). For manual holdout builds.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List training rows only.")
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

    pool_shape = str(pool.get("shape_name") or "classic")
    shape_name = str(args.shape_name or pool_shape)
    if shape_name != pool_shape:
        print(
            f"warning: --shape-name={shape_name} differs from pool shape_name={pool_shape}",
            flush=True,
        )

    exclude_ids = [str(s).strip() for s in (args.exclude_sample or []) if str(s).strip()]
    training, skipped = collect_completed_fom_training_rows(
        repo_root=repo_root,
        pool=pool,
        run_id_suffix=str(args.run_id_suffix),
        completed_only=bool(args.completed_only),
        max_samples=args.max_samples,
        exclude_sample_ids=exclude_ids or None,
        min_mode_count=int(args.min_mode_count),
    )

    print(f"shape_name={shape_name}")
    print(f"lhs_json={rel(lhs_path, repo_root=repo_root)}")
    print(f"training_rows={len(training)} skipped_rows={len(skipped)}")
    if exclude_ids:
        print(f"excluded_from_training={exclude_ids}")
    for row in training:
        print(
            f"  {row['sample_id']} modes={row['mode_count']} "
            f"catalog={row.get('catalog_path')}"
        )
    for row in skipped[:10]:
        print(f"  skip {row.get('sample_id')}: {row.get('reason')}")

    if not training:
        print("error: no training rows collected", file=sys.stderr)
        return 2

    if args.dry_run:
        print("dry_run=true")
        return 0

    model = build_surrogate_from_training_rows(
        shape_name=shape_name,
        training_rows=training,
        k_neighbors=int(args.k_neighbors),
    )
    paths = save_surrogate_model(repo_root, model)
    report = {
        "schema": "m4_rom_build_report_v1",
        "generated_utc": model["generated_utc"],
        "shape_name": shape_name,
        "training_sample_count": model["training_sample_count"],
        "mode_count_median": model["mode_count_median"],
        "method": model["method"],
        "outputs": {k: rel(v, repo_root=repo_root) for k, v in paths.items()},
        "training_samples": model["training_samples"],
        "skipped": skipped,
    }
    report_path = paths["json"].parent / "m4_rom_build_report.json"
    write_json_atomic(report_path, report)

    print(f"surrogate_json={rel(paths['json'], repo_root=repo_root)}")
    print(f"surrogate_npz={rel(paths['npz'], repo_root=repo_root)}")
    print(f"manifest={rel(paths['manifest'], repo_root=repo_root)}")
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print(
        "next: python FEM/.../run_m4_rom_compare.py "
        f"--lhs-json {rel(lhs_path, repo_root=repo_root)} --force-sample <sample_id>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
