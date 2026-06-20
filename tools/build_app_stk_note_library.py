#!/usr/bin/env python3
"""Build APP STK note library — parameter export + C++/STK render (batch preferred)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gui"))

from app_stk_config import load_app_stk_config, priority_notes_from_config  # noqa: E402
from app_stk_instrument import default_sample_id_for_shape  # noqa: E402
from app_stk_fretboard import (  # noqa: E402
    build_required_note_set_from_fretboard,
    note_range_label_from_required,
)
from stk_app_audio_service import (  # noqa: E402
    APP_STK_PARALLEL_WORKERS_ENABLED,
    CLASSIC_AUDIBLE_IDENTITY_CONTRAST_PRESETS,
    DEFAULT_APP_STK_WORKERS,
    build_note_library,
    list_available_samples,
    set_active_job,
    stk_binary_path,
)


def main(argv=None) -> int:
    cfg = load_app_stk_config(REPO_ROOT)
    parser = argparse.ArgumentParser(description="Build APP STK classical note cache.")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--shape-type",
        default="Classical",
        help="UI shape label (Classical, Box, …) — picks default LHS sample when --sample-id omitted",
    )
    parser.add_argument(
        "--note-range",
        default="",
        help="Legacy chromatic range (ignored; fretboard-derived set is used)",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "audio" / "app_stk_note_cache")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Flat preview/saved cache directory")
    parser.add_argument("--parameter-hash", default=None)
    parser.add_argument("--job-status-json", type=Path, default=None)
    parser.add_argument("--priority-notes", nargs="*", default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument(
        "--render-mode",
        choices=("batch", "parallel_batch", "per_note"),
        default=str(cfg.get("render_mode") or "parallel_batch"),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=int(cfg.get("parallel_workers") or DEFAULT_APP_STK_WORKERS),
        help="Parallel STK worker count (parallel_batch mode)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--contrast-preset",
        choices=sorted(CLASSIC_AUDIBLE_IDENTITY_CONTRAST_PRESETS.keys()),
        default="conservative",
        help="Classical STK audible identity contrast preset for VM listening experiments.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    sample_id = str(args.sample_id or default_sample_id_for_shape(args.shape_type))

    if sample_id not in list_available_samples(args.repo_root, shape_type=args.shape_type):
        print(f"WARNING: sample_id {sample_id!r} not in LHS pool; proceeding anyway")

    binary = stk_binary_path(args.repo_root)
    if not binary.is_file():
        print(f"ERROR: STK binary missing: {binary}", file=sys.stderr)
        print("Run tools/build_stk_pgsm_demo.sh on VM first.", file=sys.stderr)
        return 1

    required = build_required_note_set_from_fretboard(int(cfg.get("fret_count") or 19))
    note_range = args.note_range or note_range_label_from_required(required)
    prio = list(args.priority_notes or priority_notes_from_config(cfg))

    if args.parameter_hash:
        set_active_job(args.parameter_hash)

    report = build_note_library(
        sample_id,
        note_range=note_range,
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        duration_s=args.duration_s,
        force=args.force,
        repo_root=args.repo_root,
        binary=binary,
        parameter_hash=args.parameter_hash,
        job_status_json=args.job_status_json,
        priority_notes=prio,
        render_mode=args.render_mode,
        parallel_workers=args.parallel_workers,
        contrast_preset=args.contrast_preset,
    )
    print(f"Render mode: {report.get('render_mode')}")
    print(f"Parallel workers enabled: {APP_STK_PARALLEL_WORKERS_ENABLED}")
    print(f"Worker count: {report.get('worker_count', DEFAULT_APP_STK_WORKERS)}")
    print(f"Contrast preset: {report.get('classic_contrast_preset')}")
    print(f"Required notes: {report.get('fretboard_required_note_count')} ({note_range})")
    print(f"Readiness: {report['readiness']}")
    print(f"Status: {report.get('status', report['readiness'])}")
    print(f"Notes: {report['note_count']}  hits: {report['cache_hit_count']}  misses: {report['cache_miss_count']}")
    print(f"Total render time: {report['total_render_time_s']} s")
    print(f"Average per note: {report['average_time_per_note_s']} s")
    print(f"Target achieved: {report.get('achieved_target')}")
    print(f"Output: {report['output_dir']}")
    print(f"Report: {report.get('report_json')}")
    if report.get("missing_notes"):
        print(f"Missing: {report['missing_notes']}", file=sys.stderr)
        return 1
    if report.get("readiness") != "ready_for_app_playback":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
