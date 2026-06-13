#!/usr/bin/env python3
"""Stage 5.2 — batch Leave-One-Out ROM validation for audio-proxy assessment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage52_rom_audio_proxy_report import (  # noqa: E402
    DEFAULT_LOO_SAMPLES,
    OPTIONAL_LATEST_LOO_SAMPLES,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel  # noqa: E402

DEFAULT_LHS = "ROM/classic/lhs_pool.json"
COMPARE_SCRIPT = SCRIPT_DIR / "run_m4_rom_compare.py"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Sample IDs to validate (default: spread set sample_010..sample_065 step 5)",
    )
    parser.add_argument(
        "--latest-set",
        action="store_true",
        help="Use optional latest set sample_056..sample_065 instead of default spread",
    )
    parser.add_argument("--write-csv", action="store_true", help="Pass --write-csv to each compare run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="After LOO runs, write stage52_rom_audio_proxy_report.{json,md}",
    )
    return parser.parse_args(argv)


def resolve_sample_list(args: argparse.Namespace) -> Sequence[str]:
    if args.samples:
        return list(args.samples)
    if args.latest_set:
        return list(OPTIONAL_LATEST_LOO_SAMPLES)
    return list(DEFAULT_LOO_SAMPLES)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    samples = resolve_sample_list(args)

    print(f"stage52_loo sample_count={len(samples)} lhs={rel(lhs_path, repo_root=repo_root)}")
    ok = 0
    failed = 0
    for sid in samples:
        cmd = [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--lhs-json",
            str(lhs_path),
            "--force-sample",
            sid,
            "--leave-one-out",
            "--run-prepredict",
            "--debug",
        ]
        if args.write_csv:
            cmd.append("--write-csv")
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[stage52] {sid} ...", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd))
            ok += 1
            continue
        proc = subprocess.run(cmd, cwd=str(repo_root))
        if proc.returncode == 0:
            ok += 1
        else:
            failed += 1
            print(f"  failed returncode={proc.returncode}", flush=True)

    print(f"stage52_loo compared_ok={ok} compared_failed={failed}")

    if args.generate_report and not args.dry_run:
        from stage52_rom_audio_proxy_report import build_stage52_report  # noqa: WPS433

        out_dir = repo_root / "audio" / "debug_reports"
        build_stage52_report(
            repo_root,
            sample_filter=samples,
            out_json=out_dir / "stage52_rom_audio_proxy_report.json",
            out_md=out_dir / "stage52_rom_audio_proxy_report.md",
        )
        print(f"wrote {rel(out_dir / 'stage52_rom_audio_proxy_report.json', repo_root=repo_root)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
