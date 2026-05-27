#!/usr/bin/env python3
"""
Gated isolated lossless adjudication v1 runner.

Default: report-only preflight regeneration (no EPS).
EPS requires explicit --authorize-single-eps-run AND full top-level gate contract pass.
"""
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

from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
from v2_clean_lane_preflight_gate import (
    print_clean_lane_preflight_lines,
    validate_gate_contract_for_eps_authorization,
)
from v2_mesh_convergence_common import CONV_DIAG

PREFLIGHT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.json"
BUNDLE_SCRIPT = SCRIPT_DIR / "run_v2_clean_lane_report_only_bundle.py"


def _run_report_only_preflight() -> int:
    return subprocess.call([sys.executable, str(BUNDLE_SCRIPT)], cwd=str(REPO_ROOT))


def _load_gate() -> Dict[str, Any]:
    if not PREFLIGHT_JSON.is_file():
        return {}
    return json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gated lossless adjudication v1 (preflight default; EPS fail-closed)."
    )
    parser.add_argument(
        "--authorize-single-eps-run",
        action="store_true",
        help="Explicit human authorization for one isolated EPS (requires gate pass).",
    )
    parser.add_argument(
        "--skip-preflight-regeneration",
        action="store_true",
        help="Use existing preflight JSON without re-running report-only bundle.",
    )
    args = parser.parse_args()

    if not args.skip_preflight_regeneration:
        rc = _run_report_only_preflight()
        if rc != 0:
            print(
                "[gated_runner] report-only preflight bundle failed; EPS blocked.",
                file=sys.stderr,
                flush=True,
            )
            return rc

    gate = _load_gate()
    if not gate:
        print("[gated_runner] preflight gate JSON missing; EPS blocked.", file=sys.stderr)
        return 2

    print_clean_lane_preflight_lines(gate)
    ok, issues = validate_gate_contract_for_eps_authorization(gate)
    if not ok:
        print(f"[gated_runner] gate_validation_failed issues={issues}", file=sys.stderr, flush=True)
        return 2

    if not args.authorize_single_eps_run:
        print(
            "[gated_runner] prepare-only: gate pass; EPS not requested "
            f"(output_subdir={OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1}).",
            flush=True,
        )
        return 0

    print(
        "[gated_runner] EPS path not enabled in this patch; "
        "re-run report-only preflight and await explicit authorization command.",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
