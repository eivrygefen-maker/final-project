#!/usr/bin/env python3
"""
Report-only provenance inventory from existing replacement-baseline VM artifacts.

Does not run EPS or regenerate candidates.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_conservative_audit_policy import SERIALIZER_FUNCTION, SERIALIZER_THRESHOLD
from v2_mapping_fixed_baseline_evaluator import OUT_SUBDIR_PERSISTENCE_FIXED
from v2_mesh_convergence_common import CONV_DIAG, solve_case_dir

OUT_JSON = CONV_DIAG / "v2_mapping_fixed_replacement_runtime_provenance_inventory.json"
OUT_MD = CONV_DIAG / "v2_mapping_fixed_replacement_runtime_provenance_inventory.md"
CASE_ID = "baseline_coupled_v2"

REQUIRED_FIELDS = [
    "continuation_seed_applied",
    "seed_frequency_hz",
    "actual_sigma_hz",
    "sigma_used_hz",
    "st_type",
    "eps_eigenvalue_semantics",
    "legacy_double_shift_mapping_disabled",
    "diagnostic_operator_consistent_with_replay",
    "actual_st_a_shift_frac",
    "actual_st_mass_reg_frac",
    "preserve_all_enabled",
    "nconv_marked",
    "candidate_bank_count",
    "num_vectors_saved",
    "serializer_function",
    "serializer_threshold",
    "p_to_W_source",
    "p_to_W_length",
    "p_to_W_crc32",
]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dig(obj: Any, *keys: str) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _collect_sources(out_dir: Path) -> Dict[str, Any]:
    sources: Dict[str, Path] = {}
    diag = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_persistence_fixed_baseline_diagnostic.json"
    sources["replacement_diagnostic_json"] = diag
    sources["eps_candidate_bank"] = out_dir / "diagnostics" / "eps_candidate_bank.json"
    sources["mode_energy_summary"] = out_dir / "diagnostics" / "mode_energy_summary.json"
    results = sorted((out_dir / "results").glob("result_*.json"))
    if results:
        sources["latest_result_json"] = results[-1]
    return sources


def _values_for_field(field: str, loaded: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rep = loaded.get("replacement_diagnostic_json") or {}
    bank = loaded.get("eps_candidate_bank") or {}
    res = loaded.get("latest_result_json") or {}
    eps = rep.get("eps_batch_diagnostics") or res.get("eps_batch_diagnostics") or {}
    pbm = bank.get("pressure_block_mapping") or {}

    if field == "continuation_seed_applied":
        for src, obj in (("replacement", rep), ("result", res), ("eps_diag", eps)):
            v = obj.get("continuation_seed_applied") if isinstance(obj, dict) else None
            if v is None and isinstance(obj, dict):
                v = _dig(obj, "evaluation", "continuation_seed_applied")
            if v is not None:
                out[src] = v
    elif field in ("actual_sigma_hz", "sigma_used_hz"):
        for src, obj in (("eps_diag", eps), ("result", res), ("replacement", rep)):
            if isinstance(obj, dict):
                v = obj.get("st_sigma_hz_used") or obj.get("target_hz") or obj.get("sigma_hz")
                if v is not None:
                    out[src] = v
    elif field == "seed_frequency_hz":
        for src, obj in (("replacement", rep), ("result", res)):
            if isinstance(obj, dict) and obj.get("target_hz") is not None:
                out[src] = obj.get("target_hz")
    elif field == "st_type":
        # Gap semantics: only eps_batch_diagnostics / result top-level count as persisted policy.
        if eps.get("st_type") is not None:
            out["eps_diag"] = eps.get("st_type")
        elif isinstance(res, dict) and res.get("st_type") is not None:
            out["result"] = res.get("st_type")
    elif field == "eps_eigenvalue_semantics":
        for src, obj in (("eps_diag", eps), ("result", res)):
            if isinstance(obj, dict) and obj.get("eps_eigenvalue_semantics") is not None:
                out[src] = obj.get("eps_eigenvalue_semantics")
    elif field == "legacy_double_shift_mapping_disabled":
        for src, obj in (("eps_diag", eps), ("result", res)):
            if isinstance(obj, dict) and "legacy_double_shift_mapping_disabled" in obj:
                out[src] = obj.get("legacy_double_shift_mapping_disabled")
    elif field == "diagnostic_operator_consistent_with_replay":
        for src, obj in (("eps_diag", eps), ("result", res), ("replacement", rep)):
            if isinstance(obj, dict) and obj.get("diagnostic_operator_consistent_with_replay") is not None:
                out[src] = obj.get("diagnostic_operator_consistent_with_replay")
    elif field == "actual_st_a_shift_frac":
        for src, obj in (("eps_diag", eps), ("result", res)):
            if isinstance(obj, dict):
                v = obj.get("st_a_shift_frac_used")
                if v is not None:
                    out[src] = v
    elif field == "actual_st_mass_reg_frac":
        for src, obj in (("eps_diag", eps), ("result", res)):
            if isinstance(obj, dict):
                v = obj.get("st_mass_reg_frac_used")
                if v is not None:
                    out[src] = v
    elif field == "preserve_all_enabled":
        for src, obj in (("eps_diag", eps), ("result", res)):
            if isinstance(obj, dict) and obj.get("eps_diagnostic_preserve_all_nconv_candidates") is not None:
                out[src] = obj.get("eps_diagnostic_preserve_all_nconv_candidates")
    elif field == "nconv_marked":
        for src, obj in (("bank", bank), ("eps_diag", eps), ("result", res)):
            if isinstance(obj, dict) and obj.get("nconv_marked") is not None:
                out[src] = obj.get("nconv_marked")
    elif field == "candidate_bank_count":
        if bank.get("eps_diagnostic_candidate_bank_count") is not None:
            out["bank"] = bank.get("eps_diagnostic_candidate_bank_count")
    elif field == "num_vectors_saved":
        for src, obj in (("bank", bank), ("replacement", rep)):
            if isinstance(obj, dict):
                v = obj.get("num_vectors_saved") or _dig(obj, "evaluation", "eps_candidate_bank_summary", "num_vectors_saved")
                if v is not None:
                    out[src] = v
    elif field == "serializer_function":
        out["code_contract"] = SERIALIZER_FUNCTION
    elif field == "serializer_threshold":
        out["code_contract"] = SERIALIZER_THRESHOLD
    elif field == "p_to_W_source":
        if pbm.get("source"):
            out["bank_pressure_block_mapping"] = pbm.get("source")
        if res.get("p_to_W_source"):
            out["result"] = res.get("p_to_W_source")
    elif field == "p_to_W_length":
        if pbm.get("p_to_W_length") is not None:
            out["bank"] = pbm.get("p_to_W_length")
    elif field == "p_to_W_crc32":
        if pbm.get("p_to_W_crc32") is not None:
            out["bank"] = pbm.get("p_to_W_crc32")
    return out


def _inventory_field(field: str, values_by_source: Dict[str, Any]) -> Dict[str, Any]:
    unique_vals = {json.dumps(v, sort_keys=True): v for v in values_by_source.values()}
    conflicting = len(unique_vals) > 1
    selected = None
    basis = None
    if values_by_source:
        if "bank" in values_by_source:
            selected = values_by_source["bank"]
            basis = "eps_candidate_bank"
        elif "eps_diag" in values_by_source:
            selected = values_by_source["eps_diag"]
            basis = "eps_batch_diagnostics"
        elif "replacement" in values_by_source:
            selected = values_by_source["replacement"]
            basis = "replacement_diagnostic"
        elif "code_contract" in values_by_source:
            selected = values_by_source["code_contract"]
            basis = "code_contract"
        else:
            k = next(iter(values_by_source))
            selected = values_by_source[k]
            basis = k
    return {
        "field_name": field,
        "sources_checked": list(values_by_source.keys()),
        "values_found_by_source": values_by_source,
        "field_missing_in_sources": len(values_by_source) == 0,
        "conflicting_values": conflicting,
        "selected_value_if_any": selected,
        "selection_basis": basis,
        "provenance_confidence": "high" if not conflicting and selected is not None else (
            "low_conflict" if conflicting else "missing"
        ),
        "match_status": "conflict" if conflicting else ("found" if selected is not None else "missing"),
        "impact_on_verdict": (
            "operator_policy_provenance_mismatch_likely_reporting_gap"
            if conflicting or (selected is None and field not in ("serializer_function", "serializer_threshold"))
            else "ok"
        ),
    }


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[provenance_inventory] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    out_dir = solve_case_dir("L_mid", CASE_ID) / OUT_SUBDIR_PERSISTENCE_FIXED
    source_paths = _collect_sources(out_dir)
    loaded: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, path in source_paths.items():
        loaded[name] = _load_json(path)

    rows = []
    for field in REQUIRED_FIELDS:
        vals = _values_for_field(field, loaded)
        rows.append(_inventory_field(field, vals))

    conflicts = sum(1 for r in rows if r.get("conflicting_values"))
    missing = sum(1 for r in rows if r.get("field_missing_in_sources"))

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "VM_runtime_artifacts_report_only",
        "output_tree": str(out_dir),
        "sources": {k: str(v) for k, v in source_paths.items()},
        "fields": rows,
        "summary": {
            "num_fields": len(rows),
            "num_conflicts": conflicts,
            "num_missing": missing,
            "operator_policy_provenance_mismatch_likely": conflicts > 0 or missing > 3,
        },
        "no_eigensolve_executed": True,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Replacement runtime provenance inventory",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"**conflicts:** {conflicts} **missing:** {missing}",
        "",
    ]
    for row in rows:
        lines.append(
            f"- **{row['field_name']}**: selected=`{row.get('selected_value_if_any')}` "
            f"status={row.get('match_status')} basis={row.get('selection_basis')}"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[provenance_inventory] conflicts={conflicts} missing={missing} wrote {OUT_JSON}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
