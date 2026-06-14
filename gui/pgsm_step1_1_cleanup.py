#!/usr/bin/env python3
"""PGSM Step 1.1 — verify obsolete STK V5/V6 prototype cleanup and write report."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP1_1_VERSION = "pgsm_step1_1_cleanup_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = REPO_ROOT / "gui"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_1_cleanup_report.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_1_cleanup_report.md"

DELETED_CODE_FILES: Tuple[str, ...] = (
    "gui/build_stk_v5_diagnostic_audio.py",
    "gui/stk_v5_design_helpers.py",
    "gui/test_stk_v5_design_helpers.py",
    "gui/build_stk_v6_2_diagnostic_audio.py",
    "gui/build_stk_v6_2_1_diagnostic_audio.py",
    "gui/build_stk_v6_2_2_review_audio.py",
    "gui/build_stk_v6_3_review_audio.py",
    "gui/build_stk_v6_4_review_audio.py",
    "gui/stk_v6_2_physical_routing.py",
    "gui/stk_v6_2_2_onset_tail_repair.py",
    "gui/stk_v6_3_artifact_quarantine.py",
    "gui/stk_v6_4_current_anchor_repair.py",
    "gui/test_stk_v6_2_physical_routing.py",
    "gui/test_stk_v6_2_1_balance_repair.py",
    "gui/test_stk_v6_2_2_onset_tail_repair.py",
    "gui/test_stk_v6_3_artifact_quarantine.py",
    "gui/test_stk_v6_4_current_anchor_repair.py",
    "gui/audit_stk_v6_physical_dofs.py",
    "gui/test_stk_v6_physical_dof_audit.py",
    "gui/build_stk_final_candidate_diagnostics.py",
    "gui/stage51h_stk_final_candidate_report.py",
    "gui/test_stage51h_stk_final_candidate.py",
)

DELETED_AUDIO_DIRS: Tuple[str, ...] = (
    "audio/stk_v5_diagnostic_audio",
    "audio/stk_v5_alpha_cavity_diagnostics",
    "audio/stk_v6_2_diagnostic_audio",
    "audio/stk_v6_2_2_review_audio",
    "audio/stk_v6_3_review_audio",
    "audio/stk_v6_4_review_audio",
    "audio/stk_final_candidate_diagnostics",
    "audio/_test_stk_v5_cavity_build",
    "audio/_test_stk_v5_cavity_no_fem",
    "audio/_test_v62_no_fem",
    "audio/_test_stk_final",
)

DELETED_DEBUG_REPORT_PREFIXES: Tuple[str, ...] = (
    "stk_v5",
    "stk_v6_2",
    "stk_v6_3",
    "stk_v6_4",
    "stage51h",
)

KEPT_FILES: Tuple[Tuple[str, str], ...] = (
    ("gui/pgsm_physical_factor_registry.py", "PGSM Step 1 registry"),
    ("gui/test_pgsm_physical_factor_registry.py", "PGSM Step 1 tests"),
    ("gui/stk_v6_2_audit_features.py", "PGSM data mapping dependency"),
    ("audio/debug_reports/stk_v6_physical_dof_audit.json", "PGSM Step 1 audit input"),
    ("audio/debug_reports/stk_v6_physical_dof_audit.md", "PGSM Step 1 audit human report"),
    ("audio/debug_reports/pgsm_step1_physical_factor_registry.json", "PGSM Step 1 output"),
    ("audio/debug_reports/pgsm_step1_physical_factor_registry.md", "PGSM Step 1 output"),
)

REJECTED_DIAGNOSTIC_MODES: Tuple[str, ...] = (
    "stk_v5_alpha_body_dominant",
    "v5_alpha_s10_b90",
    "v5_alpha_s20_b80",
    "v5_alpha_s35_b65",
    "stk_v6_2_physical_routing_alpha",
    "stk_v6_2_1_balanced_tail_alpha",
    "stk_v6_2_1_soft_pluck_tail_alpha",
    "stk_v6_2_2_single_onset_soft_tail_alpha",
    "stk_v6_2_2_no_thump_body_tail_alpha",
    "stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha",
    "stk_v6_3_clean_pluck_body_alpha",
    "stk_v6_4_current_anchor_soft_attack_alpha",
    "stk_v6_4_current_anchor_sustain_smooth_alpha",
)

OBSOLETE_CODE_GLOBS: Tuple[str, ...] = (
    "build_stk_v5*.py",
    "stk_v5*.py",
    "test_stk_v5*.py",
    "build_stk_v6*.py",
    "stk_v6_2_physical_routing.py",
    "stk_v6_2_2_onset_tail_repair.py",
    "stk_v6_3_artifact_quarantine.py",
    "stk_v6_4_current_anchor_repair.py",
    "test_stk_v6*.py",
    "audit_stk_v6_physical_dofs.py",
    "build_stk_final_candidate_diagnostics.py",
    "stage51h_stk_final_candidate_report.py",
    "test_stage51h_stk_final_candidate.py",
)

WEBSITE_IMPORT_MODULES: Tuple[str, ...] = (
    "diagnostic_synthesis",
    "body_response_synth",
    "body_hybrid_v4_1_identity_space",
    "stk_pipeline_defaults",
    "build_note_cache",
    "pgsm_physical_factor_registry",
    "stk_v6_2_audit_features",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _glob_obsolete_code_remaining() -> List[str]:
    found: List[str] = []
    for pattern in OBSOLETE_CODE_GLOBS:
        for p in GUI_DIR.glob(pattern):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel == "gui/stk_v6_2_audit_features.py":
                continue
            found.append(rel)
    return sorted(found)


def _obsolete_debug_reports_remaining() -> List[str]:
    report_dir = REPO_ROOT / "audio" / "debug_reports"
    if not report_dir.is_dir():
        return []
    out: List[str] = []
    for p in report_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.startswith("stk_v6_physical_dof_audit"):
            continue
        if name.startswith("pgsm_step1"):
            continue
        if any(name.startswith(prefix) for prefix in DELETED_DEBUG_REPORT_PREFIXES):
            out.append(f"audio/debug_reports/{name}")
    return sorted(out)


def _scan_legacy_references() -> List[str]:
    """Ripgrep for legacy identifiers in gui/*.py (excluding kept audit loader)."""
    patterns = ("stk_v5", "stk_v6_2_physical", "stk_v6_3", "stk_v6_4", "stage51h", "build_stk_v6", "build_stk_v5")
    refs: List[str] = []
    for py in GUI_DIR.glob("*.py"):
        if py.name == "pgsm_step1_1_cleanup.py":
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for pat in patterns:
            if pat in text:
                refs.append(f"{py.relative_to(REPO_ROOT).as_posix()}: contains '{pat}'")
                break
    return sorted(refs)


def check_website_imports() -> Dict[str, Any]:
    if str(GUI_DIR) not in sys.path:
        sys.path.insert(0, str(GUI_DIR))
    results: Dict[str, str] = {}
    ok = True
    for mod in WEBSITE_IMPORT_MODULES:
        try:
            importlib.import_module(mod)
            results[mod] = "ok"
        except Exception as exc:  # noqa: BLE001
            results[mod] = f"fail: {exc}"
            ok = False
    return {"pass": ok, "modules": results}


def verify_cleanup_status(*, repo_root: Path | None = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    code_remaining = _glob_obsolete_code_remaining()
    audio_remaining = [d for d in DELETED_AUDIO_DIRS if (root / d).exists()]
    reports_remaining = _obsolete_debug_reports_remaining()
    legacy_refs = _scan_legacy_references()

    from diagnostic_synthesis import DIAGNOSTIC_MODES, list_diagnostic_modes  # noqa: WPS433

    registered_rejected = [m for m in REJECTED_DIAGNOSTIC_MODES if m in DIAGNOSTIC_MODES]
    website_default_ok = DEFAULT_WEBSITE_STK_MODE == "stk_body_transfer_final_v1"
    import_status = check_website_imports()

    kept_missing = [rel for rel, _ in KEPT_FILES if not (root / rel).is_file()]

    return {
        "code_obsolete_remaining": code_remaining,
        "audio_dirs_remaining": audio_remaining,
        "debug_reports_remaining": reports_remaining,
        "rejected_modes_still_registered": registered_rejected,
        "legacy_references": legacy_refs,
        "kept_files_missing": kept_missing,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_ok": website_default_ok,
        "import_check": import_status,
        "diagnostic_mode_count": len(list_diagnostic_modes()),
        "all_clean": (
            not code_remaining
            and not audio_remaining
            and not reports_remaining
            and not registered_rejected
            and not kept_missing
            and website_default_ok
            and import_status["pass"]
        ),
    }


def build_cleanup_report(*, repo_root: Path | None = None, tests_run: Sequence[str] | None = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    status = verify_cleanup_status(repo_root=root)
    return {
        "report_version": PGSM_STEP1_1_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step1_1_cleanup_complete" if status["all_clean"] else "pgsm_step1_1_cleanup_incomplete",
        "no_audio_generated": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "deleted_code_files": list(DELETED_CODE_FILES),
        "deleted_audio_artifact_dirs": list(DELETED_AUDIO_DIRS),
        "deleted_debug_report_prefixes": list(DELETED_DEBUG_REPORT_PREFIXES),
        "remaining_legacy_references": status["legacy_references"],
        "kept_files_and_reason": [{"path": p, "reason": r} for p, r in KEPT_FILES],
        "website_default_status": {
            "mode": DEFAULT_WEBSITE_STK_MODE,
            "unchanged": status["website_default_ok"],
        },
        "import_check_status": status["import_check"],
        "verification": status,
        "tests_run": list(tests_run or []),
        "explicit_statement": "Obsolete STK V1–V6 prototype code/artifacts were deleted, not archived.",
        "pgsm_note": "PGSM Step 2 not started. Website retains stk_body_transfer_final_v1 production path.",
    }


def write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    v = report.get("verification") or {}
    lines = [
        "# PGSM Step 1.1 — STK V5/V6 cleanup report",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"Website default: `{report.get('website_default_status', {}).get('mode')}` "
        f"(unchanged={report.get('website_default_status', {}).get('unchanged')})",
        "",
        "## Deleted code files",
        "",
    ]
    for f in report.get("deleted_code_files") or []:
        lines.append(f"- `{f}`")
    lines.extend(["", "## Deleted audio artifact directories", ""])
    for d in report.get("deleted_audio_artifact_dirs") or []:
        lines.append(f"- `{d}`")
    lines.extend(["", "## Deleted debug report prefixes", ""])
    for p in report.get("deleted_debug_report_prefixes") or []:
        lines.append(f"- `{p}*` (except `stk_v6_physical_dof_audit` kept for PGSM)")
    lines.extend(["", "## Kept files", ""])
    for item in report.get("kept_files_and_reason") or []:
        lines.append(f"- `{item['path']}` — {item['reason']}")
    lines.extend(["", "## Verification", ""])
    lines.append(f"- Obsolete code remaining: {len(v.get('code_obsolete_remaining') or [])}")
    lines.append(f"- Obsolete audio dirs remaining: {len(v.get('audio_dirs_remaining') or [])}")
    lines.append(f"- Obsolete debug reports remaining: {len(v.get('debug_reports_remaining') or [])}")
    lines.append(f"- Rejected modes still registered: {v.get('rejected_modes_still_registered') or []}")
    lines.append(f"- Import check pass: {v.get('import_check', {}).get('pass')}")
    if v.get("legacy_references"):
        lines.extend(["", "## Remaining legacy references", ""])
        for ref in v["legacy_references"]:
            lines.append(f"- {ref}")
    lines.extend(["", "## Tests run", ""])
    for t in report.get("tests_run") or []:
        lines.append(f"- {t}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cleanup_reports(
    *,
    repo_root: Path | None = None,
    json_path: Path | None = None,
    md_path: Path | None = None,
    tests_run: Sequence[str] | None = None,
) -> Dict[str, Any]:
    report = build_cleanup_report(repo_root=repo_root, tests_run=tests_run)
    jp = Path(json_path or REPORT_JSON)
    mp = Path(md_path or REPORT_MD)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mp)
    return report


if __name__ == "__main__":
    r = write_cleanup_reports()
    print(f"Wrote {REPORT_JSON}")
    print(f"Status: {r['status']}")
    print(f"All clean: {(r.get('verification') or {}).get('all_clean')}")
