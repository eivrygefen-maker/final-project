#!/usr/bin/env python3
"""Standalone read-only numerical comparison of two completed M4 runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_direct_run_compare_lib import compare_runs_direct, render_markdown_direct  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Direct numerical comparison of two completed M4 runs (no preconditions).",
    )
    parser.add_argument("--reference-run", type=Path, required=True, help="Reference run root directory")
    parser.add_argument("--candidate-run", type=Path, required=True, help="ROM/candidate run root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for direct_run_compare outputs")
    parser.add_argument(
        "--match-tolerance-hz",
        type=float,
        default=5.0,
        help="Maximum Hz distance for monotonic frequency matching (default: 5)",
    )
    args = parser.parse_args(argv)

    report = compare_runs_direct(
        reference_run=args.reference_run,
        candidate_run=args.candidate_run,
        match_tolerance_hz=float(args.match_tolerance_hz),
    )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "direct_run_compare.json"
    md_path = out_dir / "direct_run_compare.md"
    write_json_atomic(json_path, report)
    md_path.write_text(render_markdown_direct(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    conclusion = report.get("practical_conclusion") or {}
    print(f"recommendation={conclusion.get('recommendation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
