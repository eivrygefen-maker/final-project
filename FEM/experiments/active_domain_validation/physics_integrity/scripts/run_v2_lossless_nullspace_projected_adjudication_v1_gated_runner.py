#!/usr/bin/env python3
"""Gated runner: one authorized nullspace-projected lossless EPS in isolated tree."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_certified_null_projection_lib import (
    NULL_BASIS_PREFLIGHT_JSON,
    PROJECTED_AUTH_JSON,
    STRATEGY_PROJECTED_V1,
    validate_null_basis_preflight_gates,
)
from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1
from v2_mesh_convergence_common import CONV_DIAG

DIAG_SCRIPT = (
    SCRIPT_DIR / "run_v2_l_mid_mapping_fixed_lossless_nullspace_projected_adjudication_v1_diagnostic.py"
)
DIAG_JSON = (
    CONV_DIAG
    / "v2_l_mid_mapping_fixed_unregularized_lossless_nullspace_projected_adjudication_v1_diagnostic.json"
)


def _run(cmd: List[str]) -> int:
    print(f"[projected_gated_runner] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorize-single-projected-eps-run",
        action="store_true",
        help="Authorize exactly one projected EPS in the isolated projected tree.",
    )
    args = parser.parse_args()

    preflight = (
        json.loads(NULL_BASIS_PREFLIGHT_JSON.read_text(encoding="utf-8"))
        if NULL_BASIS_PREFLIGHT_JSON.is_file()
        else {}
    )
    ok, issues = validate_null_basis_preflight_gates(preflight)
    if not ok:
        print(f"[projected_gated_runner] certified_null_preflight_gate_failed issues={issues}", file=sys.stderr)
        return 2

    print("[projected_gated_runner] certified_null_preflight_gate_pass=True", flush=True)
    print(
        f"[projected_gated_runner] recommended_future_strategy={preflight.get('recommended_future_strategy')}",
        flush=True,
    )

    if not args.authorize_single_projected_eps_run:
        print(
            "[projected_gated_runner] Gate pass. No EPS without --authorize-single-projected-eps-run.",
            flush=True,
        )
        return 0

    if preflight.get("recommended_future_strategy") != STRATEGY_PROJECTED_V1:
        print("[projected_gated_runner] preflight does not authorize projected strategy.", file=sys.stderr)
        return 2

    if PROJECTED_AUTH_JSON.is_file():
        try:
            rec = json.loads(PROJECTED_AUTH_JSON.read_text(encoding="utf-8"))
            if int(rec.get("eps_run_count_for_projected_lane", 0)) >= 1:
                print(
                    "[projected_gated_runner] ABORT: projected EPS already consumed.",
                    file=sys.stderr,
                )
                return 2
        except Exception:
            pass

    rc = _run(
        [
            sys.executable,
            str(DIAG_SCRIPT),
            "--authorize-single-projected-eps-run",
        ]
    )
    if rc != 0:
        return rc

    diag = json.loads(DIAG_JSON.read_text(encoding="utf-8")) if DIAG_JSON.is_file() else {}
    ev = diag.get("evaluation") or {}
    pre = diag.get("pre_eps_projection_gate") or {}
    print(f"[projected_gated_summary] projection_basis_dimension={pre.get('projection_basis_dimension', 23)}", flush=True)
    print("[projected_gated_summary] preflight_gate_pass=True", flush=True)
    print(f"[projected_gated_summary] eps_run_count_for_projected_lane={diag.get('eps_run_count_for_projected_lane', 1)}", flush=True)
    print(
        f"[projected_gated_summary] final_projected_adjudication_verdict={ev.get('final_projected_adjudication_verdict')}",
        flush=True,
    )
    print(
        f"[projected_gated_summary] branch_recovery_pass_count={ev.get('branch_recovery_pass_count', 0)}",
        flush=True,
    )
    print(
        f"[projected_gated_summary] mass_null_candidate_count_after_projection="
        f"{ev.get('mass_null_candidate_count_after_projection', 0)}",
        flush=True,
    )
    print("[projected_gated_summary] no_additional_eps_run_authorized=True", flush=True)
    print(f"[projected_gated_summary] output_subdir={OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
