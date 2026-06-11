#!/usr/bin/env python3
"""
Generate multi-mode 26-guitar body-difference diagnostic comparison WAVs (no FEM).

Example:
  python gui/build_body_difference_diagnostics.py \\
    --out-dir audio/body_difference_diagnostics \\
    --max-samples 26
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from build_sample_comparison import (  # noqa: E402
    build_sample_comparisons,
    load_lhs_sample_entries,
)
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402

DEFAULT_MODES = ",".join(list_diagnostic_modes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Body-difference diagnostic comparison builder")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=REPO / "audio" / "body_difference_diagnostics")
    parser.add_argument("--max-samples", type=int, default=26)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--silence", type=float, default=0.35)
    parser.add_argument("--no-surrogate", action="store_true")
    parser.add_argument("--diagnostic-modes", type=str, default=DEFAULT_MODES)
    args = parser.parse_args()

    modes = [m.strip() for m in args.diagnostic_modes.split(",") if m.strip()]
    samples = load_lhs_sample_entries(args.repo_root, max_samples=args.max_samples)
    if not samples:
        samples = [
            {"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": {}}
            for i in range(args.max_samples)
        ]

    for mode in modes:
        mode_dir = args.out_dir / mode
        build_sample_comparisons(
            repo_root=args.repo_root,
            out_dir=mode_dir,
            samples=samples,
            duration_s=args.duration,
            silence_s=args.silence,
            use_surrogate=not args.no_surrogate,
            diagnostic_mode=mode,
        )
        print(f"mode {mode} -> {mode_dir}")

    print(f"Done: {len(modes)} modes under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
