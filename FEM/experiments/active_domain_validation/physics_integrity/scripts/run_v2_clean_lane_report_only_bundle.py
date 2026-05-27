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

from v2_clean_lane_preflight_gate import print_clean_lane_preflight_lines
from v2_mesh_convergence_common import CONV_DIAG

PIPELINE_JSON = CONV_DIAG / "v2_mapping_fixed_persistence_fixed_full_pipeline_audit.json"
PREFLIGHT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.json"


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

    pre = _load(PREFLIGHT_JSON)
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

    if pre:
        print_clean_lane_preflight_lines(pre)
    else:
        print("[clean_lane_preflight] ERROR=preflight_json_missing", flush=True)

    print(f"[clean_lane_preflight] status_refresh_pass={status_refresh_pass}", flush=True)
    if status_refresh_failure:
        print(f"[clean_lane_preflight] status_refresh_failure={status_refresh_failure}", flush=True)
    print(f"[clean_lane_preflight] step_exit_codes={steps}", flush=True)

    gate_ok = bool(pre) and pre.get("failure_reasons") == []
    return 0 if gate_ok and pipe else 2


if __name__ == "__main__":
    raise SystemExit(main())
