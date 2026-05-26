#!/usr/bin/env python3
"""
Report-only: recompute material structural displacement MAC and refresh Phase-2 status.

No material eigen solves. No geometry locator/coupled reruns.
Optional operator assembly (solve_evp=False) only if reduced u_to_W map validation fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_common import (
    BASELINE_STRUCTURAL_MAC_REF_JSON,
    DIAG_DIR,
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_SUMMARY_JSON,
    get_validated_reduced_u_to_W_map,
    load_phase1_preserved_results,
    load_production_manifest,
    production_sample_by_id,
    row_from_existing_solve_artifacts,
    validate_reduced_u_to_W_map,
    write_json,
    write_phase2_incremental,
    write_phase2_markdown_summary,
)


def _load_full_summary() -> Dict[str, Any]:
    if PRODUCTION_SUMMARY_JSON.is_file():
        return json.loads(PRODUCTION_SUMMARY_JSON.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-2 material MAC report-only recovery")
    parser.add_argument(
        "--refresh-u-map",
        action="store_true",
        help="Re-validate u_to_W via assembly-only if catalog map invalid",
    )
    args = parser.parse_args()

    manifest = load_production_manifest()
    phase2_ids: List[str] = list(manifest.get("phase2_sample_ids") or [])
    material_ids = [s for s in phase2_ids if str(s).startswith("material_")]

    u_map, map_meta = get_validated_reduced_u_to_W_map(refresh_catalog=args.refresh_u_map)
    print(f"[material_mac] validated u_to_W map: {map_meta}", flush=True)
    if u_map is None or not map_meta.get("valid"):
        print("[material_mac] FATAL: cannot obtain valid reduced u_to_W map", file=sys.stderr)
        return 2

    if BASELINE_STRUCTURAL_MAC_REF_JSON.is_file():
        cat = json.loads(BASELINE_STRUCTURAL_MAC_REF_JSON.read_text(encoding="utf-8"))
        cat["u_to_W"] = u_map.ravel().tolist()
        cat["u_to_W_reduced_validation"] = map_meta
        cat["crc32_u_to_W_reduced"] = map_meta.get("crc32_u_to_W_reduced")
        write_json(BASELINE_STRUCTURAL_MAC_REF_JSON, cat)

    prior = _load_full_summary()
    results: Dict[str, Dict[str, Any]] = dict(prior.get("samples") or {})
    results.update(load_phase1_preserved_results(manifest))

    for sid in phase2_ids:
        if sid not in results and prior.get("samples", {}).get(sid):
            results[sid] = dict(prior["samples"][sid])

    for sid in material_ids:
        sample = production_sample_by_id(manifest, sid)
        print(f"[material_mac] refresh {sid}", flush=True)
        row = row_from_existing_solve_artifacts(sample, manifest)
        if row is None:
            row = {
                "sample_id": sid,
                "status": "missing_artifacts",
                "error": "material solve artifacts missing",
            }
        else:
            row["resume_action"] = "material_mac_report_only"
        results[sid] = row
        write_phase2_incremental(results, manifest, phase2_ids=phase2_ids)

    from v2_sensitivity_common import _phase2_staged_promotion

    promotion = _phase2_staged_promotion(results, phase2_ids)
    summary = {
        **{k: v for k, v in prior.items() if k != "samples"},
        "samples": {k: v for k, v in results.items() if not str(k).startswith("_")},
        "material_mac_recovery": True,
        "validated_u_to_W_map": map_meta,
        **promotion,
    }
    write_json(PRODUCTION_SUMMARY_JSON, summary)
    md_path = write_phase2_markdown_summary(summary)
    print(f"[material_mac] wrote {PRODUCTION_SUMMARY_JSON}")
    print(f"[material_mac] wrote {md_path}")
    failed = [
        sid
        for sid in material_ids
        if (results.get(sid) or {}).get("structural_mac_status") != "ok"
    ]
    if failed:
        print(f"[material_mac] structural MAC still pending/failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
