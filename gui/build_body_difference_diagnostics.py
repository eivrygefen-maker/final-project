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
    m4_surrogate_model_available,
)
from build_sample_comparison import parse_notes_arg  # noqa: E402
from diagnostic_synthesis import compare_mode_summaries  # noqa: E402
from stage45_damping_dataflow import validate_diagnostic_evidence  # noqa: E402

DEFAULT_MODES = "baseline_current,modal_body_60_40_v1"
DEFAULT_NOTES = "A2,A4,E5"


def main() -> int:
    parser = argparse.ArgumentParser(description="Body-difference diagnostic comparison builder")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=REPO / "audio" / "body_difference_diagnostics")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--silence", type=float, default=0.2)
    parser.add_argument("--notes", type=str, default=DEFAULT_NOTES, help="Comma-separated note ids, e.g. A2,A4,E5")
    parser.add_argument("--no-surrogate", action="store_true", help="Offline only; disables evidence validation.")
    parser.add_argument(
        "--skip-evidence-validation",
        action="store_true",
        help="Do not fail when cross-sample Q/material spread is zero.",
    )
    parser.add_argument("--diagnostic-modes", type=str, default=DEFAULT_MODES)
    args = parser.parse_args()

    modes = [m.strip() for m in args.diagnostic_modes.split(",") if m.strip()]
    notes = parse_notes_arg(args.notes)
    use_surrogate = not args.no_surrogate
    if use_surrogate and not m4_surrogate_model_available(args.repo_root):
        raise RuntimeError(
            "M4 modal surrogate model files are missing "
            "(ROM/classic/m4_modal_surrogate.json + .npz). "
            "Build or copy the trained surrogate, or rerun with --no-surrogate for offline plumbing checks only."
        )

    samples = load_lhs_sample_entries(args.repo_root, max_samples=args.max_samples)
    if not samples:
        samples = [
            {"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": {}}
            for i in range(args.max_samples)
        ]

    mode_summaries = {}
    for mode in modes:
        mode_dir = args.out_dir / mode
        manifest = build_sample_comparisons(
            repo_root=args.repo_root,
            out_dir=mode_dir,
            samples=samples,
            notes=notes,
            duration_s=args.duration,
            silence_s=args.silence,
            use_surrogate=use_surrogate,
            diagnostic_mode=mode,
        )
        mode_summaries[mode] = manifest.get("mode_summary") or {}
        if not args.skip_evidence_validation and not args.no_surrogate:
            for note_row in manifest.get("notes") or []:
                validate_diagnostic_evidence(note_row.get("segments") or [], require_m4=True)
        print(f"mode {mode} -> {mode_dir}")

    contrast = compare_mode_summaries(mode_summaries)
    import json

    (args.out_dir / "mode_comparison_summary.json").write_text(
        json.dumps(
            {
                "sample_count": len(samples),
                "notes": [n for n, _ in notes],
                "modes": modes,
                "mode_summaries": mode_summaries,
                "contrast": contrast,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done: {len(modes)} modes, {len(notes)} notes under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
