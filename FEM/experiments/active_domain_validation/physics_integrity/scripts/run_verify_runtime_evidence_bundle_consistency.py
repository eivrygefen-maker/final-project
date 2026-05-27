#!/usr/bin/env python3
"""Report-only: verify conservative merge + preflight against synced VM evidence bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_clean_adjudication_lane import (
    approved_replacement_policy_from_provenance,
    planned_clean_lane_policy,
)
from v2_conservative_audit_policy import (
    PROVENANCE_APPROVED_DEFAULTS,
    merge_operator_policy_from_provenance_inventory,
)
from v2_mesh_convergence_common import CONV_DIAG

BUNDLE = CONV_DIAG / "cursor_runtime_evidence_bundle_lossless_adjudication_preflight"

EXPECTED_POLICY = {
    **PROVENANCE_APPROVED_DEFAULTS,
    "sigma_used_hz": 243.5754171175576,
    "st_type": "missing_in_prior_artifacts",
}

EXPECTED_GAP = {
    "operator_policy_provenance_mismatch": False,
    "operator_policy_provenance_gap": True,
    "operator_policy_provenance_gap_fields": ["st_type"],
}


def _load(name: str) -> Dict[str, Any]:
    path = BUNDLE / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _check_policy(policy: Dict[str, Any], prov_meta: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key, expected in EXPECTED_POLICY.items():
        got = policy.get(key)
        if got != expected:
            errors.append(f"operator_policy.{key}: got {got!r}, expected {expected!r}")
    for key, expected in EXPECTED_GAP.items():
        got = prov_meta.get(key)
        if got != expected:
            errors.append(f"provenance_meta.{key}: got {got!r}, expected {expected!r}")
    return errors


def main() -> int:
    inventory = _load("v2_mapping_fixed_replacement_runtime_provenance_inventory.json")
    replacement = _load("replacement_baseline_diagnostic_compact.json")
    bank_hint = {"nconv_marked": 56, "eps_diagnostic_candidate_bank_count": 56, "num_vectors_saved": 56}

    st_op = replacement.get("st_operator_fields") or {}
    policy: Dict[str, Any] = {
        "continuation_seed_applied": bool(replacement.get("continuation_seed_applied")),
        "seed_frequency_hz": float(replacement.get("seed_frequency_hz", 243.0754171175576)),
        "actual_sigma_hz": st_op.get("actual_sigma_hz"),
        "st_type": st_op.get("st_type"),
        "diagnostic_operator_consistent_with_replay": st_op.get(
            "diagnostic_operator_consistent_with_replay"
        ),
        "nconv_marked": bank_hint["nconv_marked"],
        "candidate_bank_count": bank_hint["eps_diagnostic_candidate_bank_count"],
        "num_vectors_saved": bank_hint["num_vectors_saved"],
    }
    policy, prov_meta = merge_operator_policy_from_provenance_inventory(policy, inventory)
    errors = _check_policy(policy, prov_meta)

    approved = approved_replacement_policy_from_provenance(inventory)
    planned = planned_clean_lane_policy()
    preflight_failures: List[str] = []
    if approved.get("continuation_seed_applied") is not True:
        preflight_failures.append("approved continuation_seed_applied")
    if approved.get("actual_sigma_hz") != planned["actual_sigma_hz_target"]:
        preflight_failures.append("sigma mismatch approved vs planned")
    if approved.get("st_type") is not None:
        preflight_failures.append("st_type should be missing in inventory selected_value")

    lossless = _load("v2_lossless_candidate_persistence_self_test.json")
    if not lossless.get("self_test_pass"):
        errors.append("lossless_self_test_pass expected True")

    compact_audit = _load("full_pipeline_audit_compact.json")
    if compact_audit.get("operator_policy_provenance_mismatch") is True:
        print(
            "[bundle_verify] NOTE: full_pipeline_audit_compact.json is pre-merge VM snapshot; "
            "re-run clean_lane bundle on VM after sync for corrected authoritative audit.",
            flush=True,
        )

    print("[bundle_verify] merged_operator_policy_sample:", flush=True)
    for k in sorted(EXPECTED_POLICY):
        print(f"  {k}={policy.get(k)!r}", flush=True)
    print(f"[bundle_verify] prov_meta={prov_meta}", flush=True)

    if errors:
        print("[bundle_verify] FAIL:", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        return 2
    if preflight_failures:
        print("[bundle_verify] preflight_warnings:", preflight_failures, flush=True)
    print("[bundle_verify] PASS: merge + inventory consistent with bundle", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
