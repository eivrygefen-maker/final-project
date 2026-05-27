#!/usr/bin/env python3
"""
Gated isolated lossless adjudication v1 runner.

With --authorize-single-eps-run: rerun preflight gate, one EPS, lossless evaluation, audit.
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
DIAG_SCRIPT = SCRIPT_DIR / "run_v2_l_mid_mapping_fixed_lossless_adjudication_v1_diagnostic.py"
AUDIT_SCRIPT = SCRIPT_DIR / "run_v2_lossless_adjudication_v1_full_pipeline_audit.py"
DIAG_JSON = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_lossless_adjudication_v1_diagnostic.json"
AUDIT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_full_pipeline_audit.json"


def _run(cmd: List[str]) -> int:
    print(f"[gated_runner] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _load_gate() -> Dict[str, Any]:
    if not PREFLIGHT_JSON.is_file():
        return {}
    return json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))


def _print_compact_summary(diag: Dict[str, Any], audit: Dict[str, Any]) -> None:
    ev = diag.get("evaluation") or {}
    print("[lossless_adjudication_summary] preflight_gate_pass=True", flush=True)
    print("[lossless_adjudication_summary] single_lossless_adjudication_run_authorized=True", flush=True)
    print(f"[lossless_adjudication_summary] eps_run_count_for_this_lane={diag.get('eps_run_count_for_this_lane', 0)}", flush=True)
    print("[lossless_adjudication_summary] no_additional_eps_run_authorized=True", flush=True)
    print(f"[lossless_adjudication_summary] nconv_marked={ev.get('eps_nconv_marked')}", flush=True)
    print(f"[lossless_adjudication_summary] lossless_candidate_count={ev.get('lossless_candidate_count')}", flush=True)
    print(f"[lossless_adjudication_summary] lossless_vectors_saved={ev.get('lossless_vectors_saved')}", flush=True)
    print(f"[lossless_adjudication_summary] lossless_roundtrip_failures={ev.get('lossless_roundtrip_failures', 0)}", flush=True)
    print(f"[lossless_adjudication_summary] legacy_sparse_comparison_saved={ev.get('legacy_sparse_comparison_saved')}", flush=True)
    print(f"[lossless_adjudication_summary] st_type_authoritative={ev.get('st_type_authoritative_provenance')}", flush=True)
    print(f"[lossless_adjudication_summary] final_adjudication_verdict={ev.get('diagnostic_verdict')}", flush=True)
    print(f"[lossless_adjudication_summary] audit_verdict={audit.get('lossless_adjudication_verdict')}", flush=True)
    print(f"[lossless_adjudication_summary] any_branch_recovery_pass={ev.get('any_branch_recovery_pass')}", flush=True)
    print(f"[lossless_adjudication_summary] output_subdir={OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1}", flush=True)
    print("[lossless_adjudication_summary] production_promotion=BLOCKED", flush=True)
    print("[lossless_adjudication_summary] mesh_convergence_resume=BLOCKED", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gated lossless adjudication v1 (preflight default; one EPS with --authorize-single-eps-run)."
    )
    parser.add_argument(
        "--authorize-single-eps-run",
        action="store_true",
        help="Explicit authorization for one isolated EPS in lossless adjudication v1 tree.",
    )
    parser.add_argument(
        "--skip-preflight-regeneration",
        action="store_true",
        help="Use existing preflight JSON (not recommended before first authorized EPS).",
    )
    args = parser.parse_args()

    if not args.skip_preflight_regeneration:
        rc = _run([sys.executable, str(BUNDLE_SCRIPT)])
        if rc != 0:
            print("[gated_runner] preflight bundle failed; EPS blocked.", file=sys.stderr)
            return rc

    gate = _load_gate()
    if not gate:
        print("[gated_runner] preflight gate JSON missing; EPS blocked.", file=sys.stderr)
        return 2

    print_clean_lane_preflight_lines(gate)
    ok, issues = validate_gate_contract_for_eps_authorization(gate)
    if not ok:
        print(f"[gated_runner] gate_validation_failed issues={issues}", file=sys.stderr)
        return 2

    if not args.authorize_single_eps_run:
        print(
            "[gated_runner] Gate pass. No EPS without --authorize-single-eps-run.",
            flush=True,
        )
        return 0

    rc = _run(
        [
            sys.executable,
            str(DIAG_SCRIPT),
            "--authorize-single-eps-run",
        ]
    )
    if rc != 0:
        print("[gated_runner] EPS/diagnostic failed; see logs in isolated tree.", file=sys.stderr)
        return rc

    rc = _run([sys.executable, str(AUDIT_SCRIPT)])
    if rc != 0:
        return rc

    diag = json.loads(DIAG_JSON.read_text(encoding="utf-8")) if DIAG_JSON.is_file() else {}
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.is_file() else {}
    _print_compact_summary(diag, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
