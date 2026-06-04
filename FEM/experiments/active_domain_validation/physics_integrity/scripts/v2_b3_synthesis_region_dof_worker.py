#!/usr/bin/env python3
"""Isolated subprocess worker for Stage A region DOF index export (segfault boundary)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_synthesis_export import (  # noqa: E402
    REGION_DOF_STATUS_PASS,
    export_region_dof_indices_npz,
    region_dof_status_is_pass,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A region DOF export worker.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mesh-level", required=True)
    parser.add_argument("--built-meta-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--mesh-file", help="M4 per-sample L_prod mesh path.")
    parser.add_argument("--core-config", help="Resolved core config JSON path.")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    result_path = Path(args.result_json).expanduser().resolve()
    built_meta: Dict[str, Any] = json.loads(Path(args.built_meta_json).read_text(encoding="utf-8"))

    mesh_file = Path(args.mesh_file).expanduser().resolve() if args.mesh_file else None
    core_config = Path(args.core_config).expanduser().resolve() if args.core_config else None
    if core_config is None and built_meta.get("region_dof_core_config_path"):
        core_config = Path(str(built_meta["region_dof_core_config_path"]))

    status = "deferred_to_stage_c"
    error: str | None = None
    try:
        status, error = export_region_dof_indices_npz(
            checkpoint,
            mesh_level=str(args.mesh_level),
            built_meta=built_meta,
            mesh_file=mesh_file,
            core_config_path=core_config,
        )
    except Exception as exc:
        import traceback

        status = "deferred_to_stage_c"
        error = f"{type(exc).__name__}:{exc}; {traceback.format_exc()[-1200:]}"

    write_json_atomic(
        result_path,
        {
            "status": status,
            "error": error,
            "region_dof_indices_npz": region_dof_status_is_pass(status),
        },
    )
    return 0 if region_dof_status_is_pass(status) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
