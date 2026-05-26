#!/usr/bin/env python3
"""Report-only EPS eigenvalue-mapping exposure inventory (no eigensolve)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_eps_mapping_audit_lib import MAPPING_FIX_SUMMARY, discover_replay_targets, static_exposure_inventory
from v2_mesh_convergence_common import CONV_DIAG, write_json

OUT_JSON = CONV_DIAG / "v2_eps_mapping_impact_inventory.json"
OUT_MD = CONV_DIAG / "v2_eps_mapping_impact_inventory.md"
REPO_ROOT = Path(__file__).resolve().parents[5]


def _classify_row(row: Dict[str, Any], targets: List[Dict[str, Any]]) -> str:
    if row.get("status_after_mapping_fix") == "not_examined":
        return "not_examined"
    guess = str(row.get("artifact_root_guess") or "")
    if not guess or guess == "None":
        return "not_examined"
    prefix = guess.split("{")[0].rstrip("/")
    matching = [t for t in targets if prefix and prefix.replace("/", "\\") in t["case_dir"].replace("\\", "/")]
    if not matching:
        return "not_examined"
    any_replay = any(t["has_replay_inputs"] for t in matching)
    if any_replay:
        return "potentially_exposed_recertifiable_report_only"
    return "potentially_exposed_artifacts_insufficient"


def main() -> int:
    targets = discover_replay_targets(REPO_ROOT)
    rows = static_exposure_inventory()
    for row in rows:
        row["existing_outputs_potentially_affected"] = (
            "reported_frequency_hz and harvest selection when EPS mapping mislabeled λ"
        )
        row["status_after_mapping_fix"] = _classify_row(row, targets)
        row["discovered_case_dirs"] = [
            t["case_dir"]
            for t in targets
            if str(row.get("artifact_root_guess", "")).split("{")[0].rstrip("/")
            in t["case_dir"].replace("\\", "/")
        ][:8]

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "confirmed_from_local_code_plus_vm_artifact_discovery",
        "mapping_fix": MAPPING_FIX_SUMMARY,
        "control_flow": {
            "native_STSINVERT": [
                "fem_main_3d._solve_coupled_evp sets ST.Type.SINVERT",
                "eps.getEigenpair(i,rvec) -> mu (SLEPc back-transformed)",
                "_slepc_physical_lambda: lam_phys=mu, tag=eps_backtransformed",
            ],
            "legacy_disabled": [
                "lam_phys=mu+sigma only if eps_eigenvalue_semantics=manual_st_shift",
                "sigma+1/mu only if eps_eigenvalue_semantics=manual_sinvert_theta",
            ],
            "fail_fast": "_slepc_eigenvalue_semantics() rejects unknown semantics strings",
        },
        "exposure_rows": rows,
        "discovered_replay_targets_count": len(targets),
        "prior_PASS_auto_invalidated": False,
        "mesh_topology_gates_separate_from_eigenvalue_frequency": True,
    }
    write_json(OUT_JSON, report)

    lines = [
        "# EPS eigenvalue mapping impact inventory",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Mapping fix",
        "",
        f"- **New rule:** `{report['mapping_fix']['new_behavior']}`",
        f"- **Legacy disabled:** `{report['mapping_fix']['legacy_double_shift_mapping_disabled']}`",
        "",
        "## Exposure rows",
        "",
        "| path | native ST | legacy mapping | status after fix |",
        "|------|-----------|----------------|------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['path_id']} | {r['uses_native_STSINVERT_EPS']} | "
            f"{r['uses_legacy_mu_plus_sigma_mapping']} | {r['status_after_mapping_fix']} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[mapping_inventory] wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
