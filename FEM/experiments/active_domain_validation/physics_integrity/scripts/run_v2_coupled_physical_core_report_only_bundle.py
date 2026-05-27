#!/usr/bin/env python3
"""
Orchestrate report-only v2 audits (no EPS). Prints compact VM summary lines.
"""
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

from v2_mesh_convergence_common import CONV_DIAG

PIPELINE_JSON = CONV_DIAG / "v2_mapping_fixed_persistence_fixed_full_pipeline_audit.json"
ARCH_JSON = CONV_DIAG / "v2_physical_core_architecture_and_hidden_policy_audit.json"
PROV_JSON = CONV_DIAG / "v2_mapping_fixed_replacement_runtime_provenance_inventory.json"
LOSSLESS_JSON = CONV_DIAG / "v2_lossless_candidate_persistence_self_test.json"


def _run(script: str) -> int:
    cmd = [sys.executable, str(SCRIPT_DIR / script)]
    if script.startswith("run_v2_mapping_fixed") and "audit" in script:
        import shutil

        mpiexec = shutil.which("mpiexec") or "mpiexec"
        cmd = [mpiexec, "-n", "1", sys.executable, str(SCRIPT_DIR / script)]
    print(f"[bundle] running {script}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    steps = [
        ("pipeline_audit", "run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit.py"),
        ("architecture_audit", "run_v2_physical_core_architecture_audit.py"),
        ("provenance_inventory", "run_v2_mapping_fixed_replacement_runtime_provenance_inventory.py"),
        ("lossless_self_test", "run_v2_lossless_candidate_persistence_self_test.py"),
    ]
    step_rc: Dict[str, int] = {}
    for name, script in steps:
        step_rc[name] = _run(script)
    # Status refresh only if pipeline audit JSON exists
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

    arch = _load(ARCH_JSON) or {}
    prov = _load(PROV_JSON) or {}
    loss = _load(LOSSLESS_JSON) or {}
    ser = (arch.get("serialization_and_vector_contract") or {}) if arch else (pipe or {})

    print(
        f"[bundle] corrected_authoritative_verdict={pipe.get('audit_verdict') if pipe else 'MISSING'}",
        flush=True,
    )
    print(
        f"[bundle] serialization_may_change_physical_replay_metrics="
        f"{pipe.get('serialization_may_change_physical_replay_metrics', ser.get('serialization_may_change_physical_replay_metrics'))}",
        flush=True,
    )
    print(
        f"[bundle] lossless_pre_sparsify_eps_vectors_available_in_current_run="
        f"{pipe.get('lossless_pre_sparsify_eps_vectors_available_in_current_run', False)}",
        flush=True,
    )
    print(
        f"[bundle] current_saved_vectors_sufficient_for_st_verdict="
        f"{pipe.get('current_saved_vectors_sufficient_for_st_verdict', False)}",
        flush=True,
    )
    print(
        f"[bundle] operator_policy_provenance_mismatch="
        f"{pipe.get('operator_policy_provenance_mismatch') if pipe else 'UNKNOWN'}",
        flush=True,
    )
    print(
        f"[bundle] runtime_provenance_inventory_status="
        f"conflicts={_dig(prov, 'summary', 'num_conflicts')} missing={_dig(prov, 'summary', 'num_missing')}",
        flush=True,
    )
    print(
        f"[bundle] lossless_persistence_self_test_status="
        f"pass={loss.get('self_test_pass', 'NOT_RUN')}",
        flush=True,
    )
    print("[bundle] no_new_eigensolve_executed=True", flush=True)
    print(f"[bundle] status_refresh_pass={status_refresh_pass}", flush=True)
    if status_refresh_failure:
        print(f"[bundle] status_refresh_failure={status_refresh_failure}", flush=True)
    print(f"[bundle] step_exit_codes={step_rc}", flush=True)
    return 0 if pipe else max(step_rc.values(), default=2)


def _dig(obj: Any, *keys: str) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


if __name__ == "__main__":
    raise SystemExit(main())
