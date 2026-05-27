#!/usr/bin/env python3
"""Report-only clean adjudication lane bundle (no EPS)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_conservative_audit_policy import VERDICT_PERSISTED_CONTENT_UNRESOLVED
from v2_mesh_convergence_common import CONV_DIAG

PIPELINE_JSON = CONV_DIAG / "v2_mapping_fixed_persistence_fixed_full_pipeline_audit.json"
CLASS_JSON = CONV_DIAG / "v2_clean_adjudication_filter_and_policy_classification.json"
PREFLIGHT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.json"
LOSSLESS_JSON = CONV_DIAG / "v2_lossless_candidate_persistence_self_test.json"
PROV_JSON = CONV_DIAG / "v2_mapping_fixed_replacement_runtime_provenance_inventory.json"


def _run(script: str, *, mpi: bool = False) -> int:
    if mpi:
        import shutil

        mpiexec = shutil.which("mpiexec") or "mpiexec"
        cmd = [mpiexec, "-n", "1", sys.executable, str(SCRIPT_DIR / script)]
    else:
        cmd = [sys.executable, str(SCRIPT_DIR / script)]
    print(f"[clean_lane_bundle] running {script}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    steps = {
        "provenance_inventory": _run(
            "run_v2_mapping_fixed_replacement_runtime_provenance_inventory.py", mpi=True
        ),
        "pipeline_audit": _run(
            "run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit.py", mpi=True
        ),
        "architecture_audit": _run("run_v2_physical_core_architecture_audit.py"),
        "filter_classification": _run("run_v2_clean_adjudication_filter_classification.py"),
        "lossless_self_test": _run("run_v2_lossless_candidate_persistence_self_test.py", mpi=True),
        "policy_equivalence_preflight": _run(
            "run_v2_lossless_adjudication_v1_policy_equivalence_preflight.py"
        ),
        "adjudication_prepare": _run("run_v2_l_mid_mapping_fixed_lossless_adjudication_v1.py"),
    }

    pipe = _load(PIPELINE_JSON)
    if pipe:
        try:
            from run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit import (
                _refresh_status_reports,
            )

            refresh = _refresh_status_reports(pipe)
            status_refresh_pass = bool(refresh.get("status_refresh_pass"))
            status_refresh_failure = refresh.get("status_refresh_failure")
        except Exception as exc:
            status_refresh_pass = False
            status_refresh_failure = f"{type(exc).__name__}:{exc}"
    else:
        status_refresh_pass = False
        status_refresh_failure = "pipeline_audit_json_missing"

    clf = _load(CLASS_JSON) or {}
    pre = _load(PREFLIGHT_JSON) or {}
    loss = _load(LOSSLESS_JSON) or {}

    verdict = (
        pipe.get("audit_verdict", VERDICT_PERSISTED_CONTENT_UNRESOLVED)
        if pipe
        else VERDICT_PERSISTED_CONTENT_UNRESOLVED
    )
    print(f"[clean_lane_preflight] authoritative_current_verdict={verdict}", flush=True)
    print(
        f"[clean_lane_preflight] operator_policy_provenance_mismatch="
        f"{pipe.get('operator_policy_provenance_mismatch', False) if pipe else False}",
        flush=True,
    )
    print(
        f"[clean_lane_preflight] operator_policy_provenance_gap="
        f"{pipe.get('operator_policy_provenance_gap', True) if pipe else True}",
        flush=True,
    )
    gap_fields = pipe.get("operator_policy_provenance_gap_fields") if pipe else ["st_type"]
    print(f"[clean_lane_preflight] operator_policy_provenance_gap_fields={gap_fields}", flush=True)
    print(
        f"[clean_lane_preflight] serialization_fidelity_risk="
        f"{pipe.get('serialization_fidelity_risk', True) if pipe else True}",
        flush=True,
    )
    print("[clean_lane_preflight] production_vector_fidelity_exposure=OPEN", flush=True)
    print(
        f"[clean_lane_preflight] lossless_self_test_pass="
        f"{loss.get('self_test_pass', False)}",
        flush=True,
    )
    print(
        f"[clean_lane_preflight] policy_equivalence_pass="
        f"{pre.get('policy_equivalence_pass', False)}",
        flush=True,
    )
    print(
        f"[clean_lane_preflight] filter_classification_complete="
        f"{clf.get('summary', {}).get('filter_classification_complete', False)}",
        flush=True,
    )
    print(
        f"[clean_lane_preflight] single_lossless_adjudication_run_ready="
        f"{pre.get('single_lossless_adjudication_run_ready', False)}",
        flush=True,
    )
    print("[clean_lane_preflight] single_lossless_adjudication_run_authorized=False", flush=True)
    print("[clean_lane_preflight] no_new_eigensolve_executed=True", flush=True)
    print(f"[clean_lane_preflight] status_refresh_pass={status_refresh_pass}", flush=True)
    if status_refresh_failure:
        print(f"[clean_lane_preflight] status_refresh_failure={status_refresh_failure}", flush=True)
    print(f"[clean_lane_preflight] step_exit_codes={steps}", flush=True)
    return 0 if pipe else 2


if __name__ == "__main__":
    raise SystemExit(main())
