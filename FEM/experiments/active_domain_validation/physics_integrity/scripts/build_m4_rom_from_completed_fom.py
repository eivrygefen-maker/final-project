#!/usr/bin/env python3
"""Build M4 Intensity ROM v2.1 surrogate (deduped catalogs + log/norm intensity) from FOM."""
from __future__ import annotations

import argparse
import json
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
from v2_b3_m4_official_rom_dataset_lib import (  # noqa: E402
    OFFICIAL_INITIAL_RUN_IDS,
    audit_official_rom_training_dataset,
    build_initial_five_run_dataset_report,
)
from v2_b3_m4_rom_shadow_pipeline_lib import build_official_rom_surrogate_from_runs  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, rel  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train M4 Intensity ROM v2.1 from completed FOM catalogs (ROM-side dedupe, log/norm intensity). "
            "Writes ROM/<shape>/m4_modal_surrogate.{json,npz} — no eigenvectors, no FOM changes."
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
    parser.add_argument(
        "--official-rom-mesh-only",
        action="store_true",
        help="Train only from official L_rom_prod accepted runs (initial allowlist by default).",
    )
    parser.add_argument(
        "--official-initial-only",
        action="store_true",
        help="With --official-rom-mesh-only, restrict to the five initial rom_official_v1 runs.",
    )
    parser.add_argument(
        "--audit-official-dataset",
        action="store_true",
        help="Read-only audit of eligible official ROM-mesh training samples (no model build).",
    )
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

    if bool(args.audit_official_dataset) or (
        bool(args.official_rom_mesh_only) and args.dry_run
    ):
        report = audit_official_rom_training_dataset(
            repo_root=repo_root,
            shape_name=shape_name,
            min_mode_count=int(args.min_mode_count),
            initial_only=bool(args.official_initial_only),
        )
        report_path = repo_root / "ROM" / shape_name / "m4_official_rom_full_dataset_report.json"
        write_json_atomic(report_path, report)
        print(json.dumps(report, indent=2))
        print(f"audit_report={rel(report_path, repo_root=repo_root)}")
        print(
            f"eligible_training_count={report.get('total_training_count')} "
            f"initial_official_count={report.get('initial_official_count')} "
            f"registered_shadow_count={report.get('registered_shadow_count')}",
            flush=True,
        )
        return 0 if int(report.get("total_training_count") or 0) > 0 else 2

    if bool(args.official_rom_mesh_only):
        _model, training, skipped, report = build_official_rom_surrogate_from_runs(
            repo_root=repo_root,
            shape_name=shape_name,
            require_initial_allowlist=bool(args.official_initial_only),
            allowed_run_ids=list(OFFICIAL_INITIAL_RUN_IDS) if args.official_initial_only else None,
            k_neighbors=int(args.k_neighbors),
            min_mode_count=int(args.min_mode_count),
        )
        report_name = (
            "m4_official_rom_initial_build_report.json"
            if args.official_initial_only
            else "m4_official_rom_full_dataset_report.json"
        )
        report_path = repo_root / "ROM" / shape_name / report_name
        write_json_atomic(report_path, report)
        manifest_path = repo_root / "ROM" / shape_name / "rom_model_manifest.json"
        print(f"official_rom_build_report={rel(report_path, repo_root=repo_root)}")
        print(f"rom_model_manifest={rel(manifest_path, repo_root=repo_root)}")
        print(
            f"training_rows={len(training)} skipped_rows={len(skipped)} "
            f"initial_official_count={report.get('initial_official_count')} "
            f"registered_shadow_count={report.get('registered_shadow_count')}",
            flush=True,
        )
        return 0

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
        "schema": "m4_rom_build_report_v2_1",
        "generated_utc": model["generated_utc"],
        "shape_name": shape_name,
        "model_version": model.get("model_version"),
        "surrogate_schema": model.get("schema"),
        "training_sample_count": model["training_sample_count"],
        "mode_count_median": model["mode_count_median"],
        "method": model["method"],
        "rom_training_catalog_source": model.get("rom_training_catalog_source"),
        "rom_training_dedupe_tolerance_hz": model.get("rom_training_dedupe_tolerance_hz"),
        "rom_training_raw_mode_count": model.get("rom_training_raw_mode_count"),
        "rom_training_deduped_mode_count": model.get("rom_training_deduped_mode_count"),
        "rom_training_raw_mode_count_median": model.get("rom_training_raw_mode_count_median"),
        "rom_training_deduped_mode_count_median": model.get("rom_training_deduped_mode_count_median"),
        "intensity_log_epsilon": model.get("intensity_log_epsilon"),
        "normalization_percentile": model.get("normalization_percentile"),
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
        f"--lhs-json {rel(lhs_path, repo_root=repo_root)} "
        "--force-sample <sample_id> --exclude-target-from-training"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
