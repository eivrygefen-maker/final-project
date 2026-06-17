#!/usr/bin/env python3
"""Evaluate modal discovery audit for an M4 FOM run (advisory; validation/ only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = SCRIPT_DIR.parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))

from m4_shape_registry import normalize_shape_key  # noqa: E402
from v2_b3_m4_modal_discovery_audit_lib import (  # noqa: E402
    AUDIT_JSON_REL,
    AUDIT_MD_REL,
    build_modal_discovery_audit,
    write_modal_discovery_audit,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modal discovery audit (advisory).")
    p.add_argument("--run-dir", required=True, help="Pipeline run root directory")
    p.add_argument("--shape", default=None, help="Shape key override (box/classic/acoustic)")
    p.add_argument("--dry-run", action="store_true", help="Build report but do not write files")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_root = Path(args.run_dir).expanduser().resolve()
    if not run_root.is_dir():
        print(f"error: run-dir not found: {run_root}", file=sys.stderr)
        return 2

    shape = normalize_shape_key(args.shape) if args.shape else args.shape
    if args.dry_run:
        report = build_modal_discovery_audit(run_root=run_root, shape_name=shape)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    json_path, md_path, report = write_modal_discovery_audit(run_root=run_root, shape_name=shape)
    print(f"wrote {json_path.relative_to(run_root) if json_path.is_relative_to(run_root) else json_path}")
    print(f"wrote {md_path.relative_to(run_root) if md_path.is_relative_to(run_root) else md_path}")
    print(f"classification={report.get('classification')}")
    print(f"candidate_level_diagnostics_available={report.get('candidate_level_diagnostics_available')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
