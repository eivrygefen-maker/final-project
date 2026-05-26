#!/usr/bin/env python3
"""
Build coupled-W seeds from archived true acoustic locator pressure vectors (L_mid),
then rerun the no-eigensolve mixed-mode audit. Does not rerun locators or coupled EPS.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_sensitivity_common import COUPLED_BASELINE_F_HZ, REPO_ROOT

RECOVERY_JSON = CONV_DIAG / "v2_l_mid_true_acoustic_reference_recovery_report.json"
RECOVERY_MD = CONV_DIAG / "v2_l_mid_true_acoustic_reference_recovery_report.md"
AUDIT_JSON = CONV_DIAG / "v2_l_mid_coupled_mixed_mode_audit.json"
AUDIT_MD = CONV_DIAG / "v2_l_mid_coupled_mixed_mode_audit.md"

ACOUSTIC_CASES = ("baseline_coupled_v2", "hole_radius_large")
ACOUSTIC_REFERENCE_SOURCE = "acoustic_only_locator_eigenvector"


def _true_ref_paths(case_dir: Path) -> Dict[str, Path]:
    d = case_dir / "diagnostics" / "l_mid_true_ref"
    return {
        "dir": d,
        "pressure_npy": d / "acoustic_locator_pressure.npy",
        "pressure_meta": d / "acoustic_locator_pressure_meta.json",
        "locator_json": case_dir / "diagnostics" / "acoustic_locator_l_mid.json",
        "seed_npy": case_dir / "diagnostics" / "acoustic_coupled_seed.npy",
        "seed_meta": case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json",
        "build_log": case_dir / "logs" / "build_acoustic_seed.log",
    }


def _log_tail(path: Path, *, max_lines: int = 48) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _mark_stale_proxy_audit() -> None:
    """Previous proxy-based mixed-mode reports are not valid physical diagnostics."""
    stale = {
        "stale_proxy_report": True,
        "superseded_by": "v2_l_mid_true_acoustic_reference_recovery",
        "note": (
            "Prior proxy-reference mixed-mode verdicts are invalid (circular MAC/overlap). "
            "Awaiting true acoustic_only_locator_eigenvector seed and audit rerun."
        ),
        "mesh_convergence_may_resume": False,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(AUDIT_JSON, stale)
    AUDIT_MD.write_text(
        "# L_mid mixed-mode audit (STALE — proxy reference)\n\n"
        "This report was superseded before true acoustic coupled-W seeds were built. "
        "Do not use proxy-based continuation verdicts.\n",
        encoding="utf-8",
    )


def _inspect_archives(paths: Dict[str, Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pressure_npy_exists": paths["pressure_npy"].is_file(),
        "pressure_meta_exists": paths["pressure_meta"].is_file(),
        "seed_npy_exists": paths["seed_npy"].is_file(),
        "seed_meta_exists": paths["seed_meta"].is_file(),
        "locator_json_exists": paths["locator_json"].is_file(),
        "archived_pressure_vector_valid": False,
        "archived_seed_valid": False,
    }
    if paths["pressure_npy"].is_file():
        try:
            plen = int(np.load(str(paths["pressure_npy"])).size)
            out["archived_pressure_vector_length"] = plen
        except Exception as exc:
            out["archived_pressure_load_error"] = str(exc)
            plen = 0
    else:
        plen = 0

    if paths["pressure_meta"].is_file():
        try:
            pm = json.loads(paths["pressure_meta"].read_text(encoding="utf-8"))
            out["pressure_meta"] = pm
            out["locator_reported_n_p_active"] = int(
                pm.get("n_p_active", pm.get("n_p_active_locator_vector", 0))
            )
        except Exception as exc:
            out["pressure_meta_error"] = str(exc)
            pm = {}

    out["archived_pressure_vector_valid"] = bool(
        paths["pressure_npy"].is_file()
        and plen > 0
        and (
            not paths["pressure_meta"].is_file()
            or (pm or {}).get("locator_pressure_reference_source") == ACOUSTIC_REFERENCE_SOURCE
        )
    )

    if paths["seed_meta"].is_file() and paths["seed_npy"].is_file():
        try:
            sm = json.loads(paths["seed_meta"].read_text(encoding="utf-8"))
            arr = np.load(str(paths["seed_npy"]))
            out["seed_meta"] = sm
            out["archived_seed_valid"] = (
                bool(sm.get("seed_layout_valid"))
                and bool(sm.get("seed_build_success"))
                and sm.get("locator_pressure_reference_source") == ACOUSTIC_REFERENCE_SOURCE
                and int(arr.size) == int(sm.get("n_reduced_W", arr.size))
            )
        except Exception as exc:
            out["seed_meta_error"] = str(exc)

    if paths["locator_json"].is_file():
        try:
            out["locator_json"] = json.loads(paths["locator_json"].read_text(encoding="utf-8"))
        except Exception as exc:
            out["locator_json_error"] = str(exc)
    return out


def _l0_reference_hz(case: Dict[str, Any]) -> float:
    return float(case.get("reference_f_hz", COUPLED_BASELINE_F_HZ))


def _run_seed_build(
    mesh_file: Path,
    sample_json: Path,
    paths: Dict[str, Path],
    *,
    locator_hz: float,
    reference_hz: float,
) -> Tuple[int, Dict[str, Any], str]:
    log_path = paths["build_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(SCRIPT_DIR / "v2_build_coupled_acoustic_seed.py"),
        "--mesh",
        str(mesh_file.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--locator-hz",
        str(float(locator_hz)),
        "--locator-lo-hz",
        "150.0",
        "--locator-hi-hz",
        "350.0",
        "--reference-hz",
        str(reference_hz),
        "--archived-pressure-npy",
        str(paths["pressure_npy"].resolve()),
        "--out-npy",
        str(paths["seed_npy"].resolve()),
        "--out-meta-json",
        str(paths["seed_meta"].resolve()),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT).returncode
    meta = (
        json.loads(paths["seed_meta"].read_text(encoding="utf-8"))
        if paths["seed_meta"].is_file()
        else {}
    )
    return int(rc), meta, _log_tail(log_path)


def _validate_recovery(paths: Dict[str, Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "acoustic_locator_vector_saved": False,
        "locator_pressure_reference_source": None,
        "seed_build_success": False,
        "seed_vector_length": None,
        "seed_layout_valid": False,
        "seed_is_finite": None,
        "seed_norm": None,
        "eps_seed_applied": False,
        "eps_seed_available_for_later_retry": False,
        "seed_failure_reason": None,
    }
    if paths["pressure_npy"].is_file():
        out["acoustic_locator_vector_saved"] = True
        out["locator_pressure_full_length"] = int(np.load(str(paths["pressure_npy"])).size)

    if paths["seed_meta"].is_file() and paths["seed_npy"].is_file():
        sm = json.loads(paths["seed_meta"].read_text(encoding="utf-8"))
        arr = np.load(str(paths["seed_npy"]))
        out.update(
            {
                "locator_pressure_reference_source": sm.get("locator_pressure_reference_source"),
                "seed_layout_valid": bool(sm.get("seed_layout_valid")),
                "seed_vector_length": int(arr.size),
                "n_reduced_W": int(sm.get("n_reduced_W", arr.size)),
                "n_p_active": int(sm.get("n_p_active", 0)),
                "n_u_active": int(sm.get("n_u_active", 0)),
                "seed_norm": float(sm.get("seed_norm", float(np.linalg.norm(arr)))),
                "seed_is_finite": bool(sm.get("seed_is_finite", np.all(np.isfinite(arr)))),
                "layout_maps": sm.get("layout_maps"),
            }
        )
        out["seed_build_success"] = (
            sm.get("locator_pressure_reference_source") == ACOUSTIC_REFERENCE_SOURCE
            and bool(sm.get("seed_layout_valid"))
            and bool(sm.get("seed_build_success"))
            and int(arr.size) == int(sm.get("n_reduced_W", arr.size))
            and float(np.linalg.norm(arr)) > 0.0
        )
        if not out["seed_build_success"]:
            out["seed_failure_reason"] = "seed/meta validation failed after build"
    elif paths["build_log"].is_file():
        out["seed_failure_reason"] = "seed meta missing after build; see build log tail"

    out["eps_seed_available_for_later_retry"] = bool(out["seed_build_success"])
    return out


def _process_case(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    case = next(c for c in manifest["cases"] if str(c["id"]) == case_id)
    mesh_file = mesh_path("L_mid", case_id)
    case_dir = solve_case_dir("L_mid", case_id)
    paths = _true_ref_paths(case_dir)
    sample_json = case_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample_spec_from_case(case), indent=2), encoding="utf-8")

    row: Dict[str, Any] = {
        "sample_id": case_id,
        "mesh_file": str(mesh_file),
        "inspect": _inspect_archives(paths),
        "actions": [],
    }
    if not mesh_file.is_file():
        row["status"] = "failed"
        row["error"] = f"missing mesh {mesh_file}"
        return row

    insp = row["inspect"]
    if not bool(insp.get("archived_pressure_vector_valid")):
        row["status"] = "archived_pressure_vector_missing_or_invalid"
        row["error"] = (
            "archived acoustic locator pressure vector required; "
            "do not rerun locator in this stage"
        )
        return row

    row["actions"].append("reuse_archived_pressure_vector")
    loc_hz = float("nan")
    if insp.get("locator_json"):
        loc_hz = float(insp["locator_json"].get("locator_frequency_hz", float("nan")))
    if paths["pressure_meta"].is_file():
        pm = json.loads(paths["pressure_meta"].read_text(encoding="utf-8"))
        loc_hz = float(pm.get("locator_frequency_hz", loc_hz))

    if bool(insp.get("archived_seed_valid")):
        row["actions"].append("reuse_valid_coupled_W_seed")
    else:
        row["actions"].append("build_coupled_W_acoustic_seed_from_archived_pressure")
        if not math.isfinite(loc_hz):
            row["status"] = "failed"
            row["error"] = "no locator frequency for seed build"
            return row
        src, sm, log_tail = _run_seed_build(
            mesh_file,
            sample_json,
            paths,
            locator_hz=loc_hz,
            reference_hz=_l0_reference_hz(case),
        )
        row["seed_build"] = {
            "exit_code": src,
            "meta": sm,
            "log_tail": log_tail,
            "failure_reason": None if src == 0 and sm.get("seed_build_success") else log_tail,
        }

    row["validation"] = _validate_recovery(paths)
    ok = bool(row["validation"].get("seed_build_success")) and bool(
        row["validation"].get("acoustic_locator_vector_saved")
    )
    row["status"] = "ok" if ok else "validation_failed"
    row["locator_acoustic_frequency_hz"] = loc_hz
    return row


def _write_recovery_md(report: Dict[str, Any]) -> None:
    lines = [
        "# L_mid true acoustic reference recovery (seed build from archive)",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        "Uses archived `diagnostics/l_mid_true_ref/acoustic_locator_pressure.npy` only. "
        "No acoustic locator rerun. No coupled EPS.",
        "",
    ]
    for cid, row in (report.get("cases") or {}).items():
        lines.append(f"## {cid}")
        lines.append("")
        lines.append(f"- Status: **{row.get('status')}**")
        lines.append(f"- Actions: {', '.join(row.get('actions') or [])}")
        insp = row.get("inspect") or {}
        lines.append(
            f"- Archived pressure: valid={insp.get('archived_pressure_vector_valid')} "
            f"length={insp.get('archived_pressure_vector_length')}"
        )
        val = row.get("validation") or {}
        lines.append(
            f"- Seed validation: ok={val.get('seed_build_success')} "
            f"n_W={val.get('seed_vector_length')} "
            f"source={val.get('locator_pressure_reference_source')}"
        )
        sb = row.get("seed_build") or {}
        if sb.get("failure_reason"):
            lines.append("")
            lines.append("### Seed build log tail")
            lines.append("")
            lines.append("```text")
            lines.append(str(sb.get("failure_reason", ""))[-4000:])
            lines.append("```")
        lines.append("")
    audit = report.get("mixed_mode_audit") or {}
    lines.append("## Mixed-mode audit")
    lines.append("")
    if audit.get("skipped"):
        lines.append("- **Skipped** (seed recovery incomplete)")
    else:
        lines.append(f"- Exit code: {audit.get('exit_code')}")
        lines.append(f"- Report: `{audit.get('report_json')}`")
        if audit.get("stale_proxy_superseded"):
            lines.append("- Prior proxy audit marked stale before rerun")
    lines.append("")
    RECOVERY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest = load_manifest()
    CONV_DIAG.mkdir(parents=True, exist_ok=True)
    _mark_stale_proxy_audit()

    cases: Dict[str, Any] = {}
    for cid in ACOUSTIC_CASES:
        print(f"[true_ref_recovery] {cid}", flush=True)
        cases[cid] = _process_case(manifest, cid)

    all_ok = all(str((cases.get(c) or {}).get("status")) == "ok" for c in ACOUSTIC_CASES)
    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "acoustic_reference_source_required": ACOUSTIC_REFERENCE_SOURCE,
        "locator_rerun_performed": False,
        "cases": cases,
        "all_cases_recovered": all_ok,
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }

    audit_rc = 1
    if all_ok:
        print("[true_ref_recovery] rerunning no-eigensolve mixed-mode audit", flush=True)
        audit_cmd = [
            "mpiexec",
            "-n",
            "1",
            sys.executable,
            str(SCRIPT_DIR / "run_v2_l_mid_coupled_mixed_mode_audit.py"),
            "--require-true-acoustic-reference",
        ]
        proc = subprocess.run(audit_cmd, cwd=str(REPO_ROOT), check=False)
        audit_rc = int(proc.returncode)
    report["mixed_mode_audit"] = {
        "exit_code": audit_rc,
        "report_json": str(AUDIT_JSON),
        "report_md": str(AUDIT_MD),
        "skipped": not all_ok,
        "stale_proxy_superseded": True,
    }
    write_json(RECOVERY_JSON, report)
    _write_recovery_md(report)
    print(f"[true_ref_recovery] wrote {RECOVERY_JSON}", flush=True)
    return 0 if all_ok and audit_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
