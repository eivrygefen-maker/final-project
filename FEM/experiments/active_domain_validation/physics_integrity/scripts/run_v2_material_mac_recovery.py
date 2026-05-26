#!/usr/bin/env python3
"""
Report-only: recompute material structural displacement MAC and refresh Phase-2 status.

No material eigen solves. No geometry locator/coupled reruns. No u_to_W rebuild when valid.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_common import (
    DIAG_DIR,
    PRODUCTION_SUMMARY_JSON,
    get_validated_reduced_u_to_W_map,
    load_phase1_preserved_results,
    load_production_manifest,
    production_sample_by_id,
    row_from_existing_solve_artifacts,
    write_json,
    write_phase2_markdown_summary,
    write_phase2_reports,
    write_validation_status,
    _phase2_staged_promotion,
)


def _load_full_summary() -> Dict[str, Any]:
    if PRODUCTION_SUMMARY_JSON.is_file():
        return json.loads(PRODUCTION_SUMMARY_JSON.read_text(encoding="utf-8"))
    return {}


def _incremental_material_save(
    results: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    phase2_ids: List[str],
) -> None:
    """Persist partial summary after each material row (MAC preserved if later step fails)."""
    phase1_ids = list(manifest.get("preserve_phase1_sample_ids") or [])
    promotion = _phase2_staged_promotion(results, phase2_ids)
    partial = {
        "suite": manifest.get("suite"),
        "phase": manifest.get("phase"),
        "samples": {k: v for k, v in results.items() if not str(k).startswith("_")},
        "incremental": True,
        **promotion,
    }
    write_json(DIAG_DIR / "v2_production_validation_summary.partial.json", partial)
    write_json(PRODUCTION_SUMMARY_JSON, partial)
    try:
        write_validation_status(
            {k: v for k, v in results.items() if k in phase1_ids},
            {k: v for k, v in results.items() if k in phase2_ids},
            production_manifest=manifest,
        )
    except Exception as exc:
        partial["validation_status_write_error"] = f"{type(exc).__name__}: {exc}"
        write_json(DIAG_DIR / "v2_production_validation_summary.partial.json", partial)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-2 material MAC report-only recovery")
    parser.add_argument(
        "--refresh-u-map",
        action="store_true",
        help="Re-validate u_to_W via assembly-only (not needed if catalog map already valid)",
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

    prior = _load_full_summary()
    results: Dict[str, Dict[str, Any]] = dict(prior.get("samples") or {})
    results.update(load_phase1_preserved_results(manifest))

    for sid in phase2_ids:
        if sid not in results and (prior.get("samples") or {}).get(sid):
            results[sid] = dict(prior["samples"][sid])

    mac_errors: Dict[str, str] = {}
    for sid in material_ids:
        sample = production_sample_by_id(manifest, sid)
        print(f"[material_mac] refresh {sid}", flush=True)
        try:
            row = row_from_existing_solve_artifacts(sample, manifest)
            if row is None:
                row = {
                    "sample_id": sid,
                    "status": "missing_artifacts",
                    "error": "material solve artifacts missing",
                    "structural_mac_status": "structural_mac_unavailable",
                }
            else:
                row["resume_action"] = "material_mac_report_only"
        except Exception as exc:
            row = {
                "sample_id": sid,
                "status": "ok",
                "structural_mac_status": "report_failed",
                "structural_mac_reason": f"{type(exc).__name__}: {exc}",
                "structural_mac_traceback": traceback.format_exc(),
                "material_assignment": sample.get("materials"),
            }
            mac_errors[sid] = str(exc)
        results[sid] = row
        _incremental_material_save(results, manifest, phase2_ids)

    prior_meta = {k: v for k, v in prior.items() if k not in ("samples",)}
    prior_meta["material_mac_recovery"] = True
    prior_meta["validated_u_to_W_map"] = map_meta
    if mac_errors:
        prior_meta["material_mac_row_errors"] = mac_errors

    try:
        summary = write_phase2_reports(
            results, manifest, phase2_ids=phase2_ids, prior_meta=prior_meta
        )
    except Exception as exc:
        print(f"[material_mac] final report write failed: {exc}", file=sys.stderr)
        promotion = _phase2_staged_promotion(results, phase2_ids)
        summary = {
            **prior_meta,
            "samples": {k: v for k, v in results.items() if not str(k).startswith("_")},
            **promotion,
            "final_report_write_error": f"{type(exc).__name__}: {exc}",
        }
        write_json(PRODUCTION_SUMMARY_JSON, summary)
        try:
            write_phase2_markdown_summary(summary)
        except Exception as md_exc:
            print(f"[material_mac] markdown write failed: {md_exc}", file=sys.stderr)
        return 1

    print(f"[material_mac] wrote {PRODUCTION_SUMMARY_JSON}")
    print(f"[material_mac] wrote {summary.get('markdown_report')}")

    failed = [
        sid
        for sid in material_ids
        if (results.get(sid) or {}).get("structural_mac_status") != "ok"
    ]
    if failed:
        print(f"[material_mac] structural MAC still pending/failed: {failed}", file=sys.stderr)
        return 1
    print("[material_mac] all material structural MAC validations passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
