#!/usr/bin/env python3
"""Post-compaction verification gate for strict m4_geometry_corrected_v1 production."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import load_lhs_pool  # noqa: E402
from v2_b3_m4_physics_identity_lib import verify_post_compaction_contract  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, rel  # noqa: E402

GUITARS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars")


def _parse_samples(text: str) -> List[str]:
    out: List[str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            out.append(f"sample_{int(part):03d}")
        else:
            out.append(part)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify post-compaction contract for M4 completed runs.")
    parser.add_argument("--sample-ids", required=True, help="Comma-separated sample ids")
    parser.add_argument("--run-id-suffix", default="m4prod2")
    parser.add_argument("--lhs-json", type=Path, default=Path("ROM/classic/lhs_pool.json"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    entry_by_id = {str(e.get("id")): e for e in pool.get("entries") or []}

    reports: List[Dict[str, Any]] = []
    all_pass = True
    for sid in _parse_samples(str(args.sample_ids)):
        entry = entry_by_id.get(sid) or {}
        run_id = str(entry.get("last_run_id") or f"{sid}_{args.run_id_suffix}")
        run_root = repo_root / GUITARS_REL / sid / "runs" / run_id
        rep = verify_post_compaction_contract(run_root)
        rep["sample_id"] = sid
        rep["run_id"] = run_id
        rep["run_root_rel"] = rel(run_root, repo_root=repo_root)
        reports.append(rep)
        print(f"{sid}: pass={rep.get('pass')} forbidden_heavy={rep.get('forbidden_heavy_artifact_count')}")
        if not rep.get("pass"):
            all_pass = False
            for err in rep.get("errors") or []:
                print(f"  error: {err}")

    payload = {"all_pass": all_pass, "samples": reports}
    if args.json_out:
        out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {rel(out, repo_root=repo_root)}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
