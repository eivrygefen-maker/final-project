#!/usr/bin/env python3
"""Build APP STK note library — parameter export + C++/STK render per note."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gui"))

from stk_app_audio_service import (  # noqa: E402
    DEFAULT_PRIORITY_NOTES,
    build_note_library,
    list_available_samples,
    set_active_job,
    stk_binary_path,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build APP STK classical note cache.")
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--instrument", default="classical")
    parser.add_argument("--note-range", default="E2:E5")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "audio" / "app_stk_note_cache")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Flat preview/saved cache directory")
    parser.add_argument("--parameter-hash", default=None)
    parser.add_argument("--job-status-json", type=Path, default=None)
    parser.add_argument("--priority-notes", nargs="*", default=list(DEFAULT_PRIORITY_NOTES))
    parser.add_argument("--duration-s", type=float, default=2.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.sample_id not in list_available_samples(args.repo_root):
        print(f"WARNING: sample_id {args.sample_id!r} not in LHS pool; proceeding anyway")

    binary = stk_binary_path(args.repo_root)
    if not binary.is_file():
        print(f"ERROR: STK binary missing: {binary}", file=sys.stderr)
        print("Run tools/build_stk_pgsm_demo.sh on VM first.", file=sys.stderr)
        return 1

    if args.parameter_hash:
        set_active_job(args.parameter_hash)

    report = build_note_library(
        args.sample_id,
        instrument=args.instrument,
        note_range=args.note_range,
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        duration_s=args.duration_s,
        force=args.force,
        repo_root=args.repo_root,
        binary=binary,
        parameter_hash=args.parameter_hash,
        job_status_json=args.job_status_json,
        priority_notes=args.priority_notes,
    )
    print(f"Readiness: {report['readiness']}")
    print(f"Status: {report.get('status', report['readiness'])}")
    print(f"Notes: {report['note_count']}  hits: {report['cache_hit_count']}  misses: {report['cache_miss_count']}")
    print(f"Total render time: {report['total_render_time_s']} s")
    print(f"Average per note: {report['average_time_per_note_s']} s")
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
