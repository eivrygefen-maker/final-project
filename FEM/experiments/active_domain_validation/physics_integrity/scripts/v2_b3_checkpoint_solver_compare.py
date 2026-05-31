#!/usr/bin/env python3
"""Compare two checkpoint solver benchmark result.json files (no FEM imports)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_st_sinvert_solver_lib import (  # noqa: E402
    FREQ_PARITY_TOL_HZ,
    compare_checkpoint_summaries,
    extract_summary_view,
)


def _load_result(path: Path) -> Dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    body["_source_path"] = str(path.resolve())
    return body


def _format_speedup(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}x"


def _write_compare_md(path: Path, report: Dict[str, Any]) -> None:
    timing = report.get("timing") or {}
    lines = [
        "# Checkpoint solver benchmark comparison",
        "",
        f"- baseline: `{report.get('baseline_path')}` ({report.get('baseline_factor_solver')})",
        f"- candidate: `{report.get('candidate_path')}` ({report.get('candidate_factor_solver')})",
        f"- parity_pass: `{report.get('parity_pass')}`",
        f"- accepted_frequencies_match: `{report.get('accepted_frequencies_match')}`",
        "",
        "## Timing speedup (baseline / candidate)",
        "",
        f"- aggregate_wall: {_format_speedup(timing.get('aggregate_wall_speedup'))}",
        f"- total_st: {_format_speedup(timing.get('total_st_speedup'))}",
        f"- total_setup: {_format_speedup(timing.get('total_setup_speedup'))}",
        f"- total_solve: {_format_speedup(timing.get('total_solve_speedup'))}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_compare(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare checkpoint solver benchmark results.")
    parser.add_argument("--baseline", required=True, help="Baseline result.json path.")
    parser.add_argument("--candidate", required=True, help="Candidate result.json path.")
    parser.add_argument(
        "--tol-hz",
        type=float,
        default=FREQ_PARITY_TOL_HZ,
        help="Accepted-frequency match tolerance in Hz.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write comparison report JSON.",
    )
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    baseline = _load_result(baseline_path)
    candidate = _load_result(candidate_path)

    base_summary = extract_summary_view(baseline)
    cand_summary = extract_summary_view(candidate)
    comparison = compare_checkpoint_summaries(
        baseline=baseline,
        candidate=candidate,
        tol_hz=float(args.tol_hz),
    )
    report: Dict[str, Any] = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_summary": base_summary,
        "candidate_summary": cand_summary,
        **comparison,
    }

    timing = report.get("timing") or {}
    print("[B3_checkpoint_solver_compare]", flush=True)
    print(f"  baseline:  {baseline_path}", flush=True)
    print(f"  candidate: {candidate_path}", flush=True)
    print(f"  parity_pass: {report.get('parity_pass')}", flush=True)
    print(f"  accepted_frequencies_match: {report.get('accepted_frequencies_match')}", flush=True)
    print(f"  aggregate_wall_speedup: {_format_speedup(timing.get('aggregate_wall_speedup'))}", flush=True)
    print(f"  total_st_speedup: {_format_speedup(timing.get('total_st_speedup'))}", flush=True)
    print(f"  total_setup_speedup: {_format_speedup(timing.get('total_setup_speedup'))}", flush=True)
    print(f"  total_solve_speedup: {_format_speedup(timing.get('total_solve_speedup'))}", flush=True)

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        write_json_atomic(out, report)
        _write_compare_md(out.with_suffix(".md"), report)
        print(f"  wrote: {out}", flush=True)

    return 0 if report.get("parity_pass") else 1


def main() -> int:
    return run_compare()


if __name__ == "__main__":
    raise SystemExit(main())
