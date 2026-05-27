#!/usr/bin/env python3
"""
Clean adjudication lane v1 entry point (prepare/preflight only by default).

Does NOT run EPS unless explicitly passed --run-eps (not authorized in current step).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1, planned_clean_lane_policy
from v2_mesh_convergence_common import (
    CONV_DIAG,
    case_by_id,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_sensitivity_common import REPO_ROOT, hz_result_tag

CASE_ID = "baseline_coupled_v2"
SELF_TEST_JSON = CONV_DIAG / "v2_mapping_fixed_candidate_persistence_self_test.json"
PREFLIGHT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.json"
REPORT_JSON = CONV_DIAG / "v2_l_mid_mapping_fixed_lossless_adjudication_v1_prepare.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-eps",
        action="store_true",
        help="NOT AUTHORIZED in current validation step; fails closed.",
    )
    args = parser.parse_args()

    if args.run_eps:
        print(
            "[lossless_adjudication_v1] EPS run not authorized; use preflight bundle first.",
            file=sys.stderr,
        )
        return 2

    manifest = load_manifest()
    case = case_by_id(manifest, CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
    mesh_file = mesh_path("L_mid", CASE_ID)
    sample = sample_spec_from_case(case)
    preflight = {}
    if PREFLIGHT_JSON.is_file():
        preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "prepare_only_no_eps",
        "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "output_dir": str(out_dir),
        "mesh_file": str(mesh_file),
        "planned_policy": planned_clean_lane_policy(),
        "policy_equivalence_preflight": preflight,
        "persistence_self_test_required": str(SELF_TEST_JSON),
        "lossless_vector_suffix": ".smx.dense.npy",
        "sparse_comparison_suffix": ".smx.npz",
        "solver_flags": [
            "--seed-branch-recovery-diagnostic",
            "--seed-branch-lossless-adjudication-v1",
        ],
        "eps_run_authorized": False,
        "no_eigensolve_executed": True,
    }
    write_json(REPORT_JSON, report)
    print(f"[lossless_adjudication_v1] prepare-only wrote {REPORT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
