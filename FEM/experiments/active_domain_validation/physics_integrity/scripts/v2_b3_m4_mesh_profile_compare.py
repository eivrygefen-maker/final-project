#!/usr/bin/env python3
"""Read-only comparison of reference vs ROM mesh profile production runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_mesh_profile_compare_lib import (  # noqa: E402
    EXIT_ACCEPTANCE_FAIL,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_PRECONDITION_FAIL,
    compare_exit_code,
    compare_runs as _compare_runs,
    scan_candidate_references_other_run,
)
from v2_b3_m4_worker_run_lib import detect_repo_root  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def compare_runs(
    *,
    reference_run: Path,
    candidate_run: Path,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = repo_root or detect_repo_root(SCRIPT_DIR)
    return _compare_runs(
        reference_run=reference_run,
        candidate_run=candidate_run,
        repo_root=root,
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M4 mesh profile comparison",
        "",
        f"- status: **{report.get('status')}**",
        f"- comparison_executed: **{report.get('comparison_executed')}**",
        f"- cleanup_barrier_precondition_pass: **{report.get('cleanup_barrier_precondition_pass')}**",
        f"- acceptance_pass: **{report.get('acceptance_pass')}**",
        f"- exit_code: **{report.get('exit_code')}**",
        "",
    ]
    if report.get("precondition_errors"):
        lines.append("## Precondition failures")
        for e in report["precondition_errors"]:
            lines.append(f"- {e}")
        lines.append("")

    if not report.get("comparison_executed"):
        return "\n".join(lines)

    freq = report.get("frequencies") or {}
    lines.extend(
        [
            "## Frequency errors (matched modes)",
            "",
            f"- median rel error: {freq.get('global_median_rel_error')}",
            f"- p95 rel error: {freq.get('global_p95_rel_error')}",
            f"- max rel error: {freq.get('global_max_rel_error')}",
            "",
            "## Performance",
            "",
        ]
    )
    perf = report.get("performance") or {}
    lines.append(f"- runtime reduction: {perf.get('runtime_reduction_fraction')}")
    lines.append(f"- candidate peak RSS (max worker VmHWM): {perf.get('candidate_peak_rss_bytes_max_worker')}")
    mac = report.get("mac") or {}
    lines.append(f"- MAC status: {mac.get('MAC_STATUS')}")
    lines.append("")
    lines.append("## Acceptance checks")
    for k, v in sorted((report.get("acceptance_evaluation") or {}).items()):
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare reference vs ROM M4 production runs.")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = args.repo_root or detect_repo_root(SCRIPT_DIR)
    report = compare_runs(
        reference_run=args.reference_run,
        candidate_run=args.candidate_run,
        repo_root=repo_root,
    )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "mesh_profile_compare.json", report)
    (out_dir / "mesh_profile_compare.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"status={report.get('status')}")
    print(f"cleanup_barrier_precondition_pass={report.get('cleanup_barrier_precondition_pass')}")
    print(f"comparison_executed={report.get('comparison_executed')}")
    print(f"acceptance_pass={report.get('acceptance_pass')}")
    print(f"exit_code={report.get('exit_code')}")
    print(f"output={out_dir / 'mesh_profile_compare.json'}")
    return compare_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
