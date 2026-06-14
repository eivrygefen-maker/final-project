#!/usr/bin/env python3
"""PGSM Step 1.1c — delete and verify legacy STK V3/V4 / Stage 42–52 cleanup."""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP1_1C_VERSION = "pgsm_step1_1c_legacy_stk_cleanup_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = REPO_ROOT / "gui"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_1c_legacy_stk_cleanup_report.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_1c_legacy_stk_cleanup_report.md"

DELETED_CODE_FILES: Tuple[str, ...] = (
    "gui/build_body_timbre_decomposition_stage48.py",
    "gui/build_stk_v3_diagnostics.py",
    "gui/build_stk_v4_diagnostics.py",
    "gui/build_stk_v4_1_diagnostics.py",
    "gui/build_stk_v4_1_transition_diagnostics.py",
    "gui/build_stk_v4_1_identity_space_diagnostics.py",
    "gui/build_stk_v4_1_identity_sweep_diagnostics.py",
    "gui/build_stk_v4_1_identity_contrast_diagnostics.py",
    "gui/build_stk_v4_1_identity_contrast_hybrid_diagnostics.py",
    "gui/build_stk_v4_1_identity_contrast_g_diagnostics.py",
    "gui/build_stk_v4_2_body_response_first_diagnostics.py",
    "gui/stage45_damping_dataflow.py",
    "gui/stage46_research_audit.py",
    "gui/stage47_reports.py",
    "gui/stage48_timbre_decomposition_report.py",
    "gui/stage49_stk_v3_report.py",
    "gui/stage50_stk_v4_report.py",
    "gui/stage51_stk_v4_1_report.py",
    "gui/stage51b_stk_v4_1_transition_report.py",
    "gui/stage51c_stk_v4_1_identity_space_report.py",
    "gui/stage51d_stk_v4_1_identity_sweep_report.py",
    "gui/stage51e_stk_v4_1_identity_contrast_report.py",
    "gui/stage51f_stk_v4_1_identity_contrast_hybrid_report.py",
    "gui/stage51g_stk_v4_1_identity_contrast_g_report.py",
    "gui/stage51_rom_readiness_report.py",
    "gui/stage52a_stk_v4_2_body_response_first_report.py",
    "gui/test_stage42_diagnostics.py",
    "gui/test_stage43_structural.py",
    "gui/test_stage44_audit.py",
    "gui/test_stage45_plumbing.py",
    "gui/test_stage46_reports.py",
    "gui/test_stage47_radiation_v2.py",
    "gui/test_stage48_timbre_decomposition.py",
    "gui/test_stage49_stk_v3.py",
    "gui/test_stage50_stk_v4.py",
    "gui/test_stage51_stk_v4_1.py",
    "gui/test_stage51b_stk_v4_1_transition.py",
    "gui/test_stage51c_stk_v4_1_identity_space.py",
    "gui/test_stage51d_stk_v4_1_identity_sweep.py",
    "gui/test_stage51e_stk_v4_1_identity_contrast.py",
    "gui/test_stage51f_stk_v4_1_identity_contrast_hybrid.py",
    "gui/test_stage51g_stk_v4_1_identity_contrast_g.py",
    "gui/test_stage52a_stk_v4_2_body_response_first.py",
)

OBSOLETE_CODE_GLOBS: Tuple[str, ...] = (
    "build_stk_v3*.py",
    "build_stk_v4*.py",
    "build_body_timbre_decomposition_stage48.py",
    "stage45*.py",
    "stage46*.py",
    "stage47*.py",
    "stage48*.py",
    "stage49*.py",
    "stage50*.py",
    "stage51*.py",
    "stage52*.py",
    "test_stage42*.py",
    "test_stage43*.py",
    "test_stage44*.py",
    "test_stage45*.py",
    "test_stage46*.py",
    "test_stage47*.py",
    "test_stage48*.py",
    "test_stage49*.py",
    "test_stage50*.py",
    "test_stage51*.py",
    "test_stage52*.py",
)

OBSOLETE_CODE_EXCEPTIONS: Tuple[str, ...] = (
    "gui/test_gui_stage2_rom_auto.py",
)

DEBUG_REPORT_PREFIXES: Tuple[str, ...] = (
    "stage42", "stage43", "stage44", "stage45", "stage46", "stage47",
    "stage48", "stage49", "stage50", "stage51", "stage52",
    "stk_v3", "stk_v4",
)

AUDIO_DIR_PREFIXES: Tuple[str, ...] = (
    "stage42", "stage43", "stage44", "stage45", "stage46", "stage47",
    "stage48", "stage49", "stage50", "stage51", "stage52",
    "stk_v3", "stk_v4", "body_difference_diagnostics_stage",
    "body_timbre_decomposition_stage48", "_test_stk_v3", "_test_stk_v4",
)

KEPT_PRODUCTION_FILES: Tuple[Tuple[str, str], ...] = (
    ("gui/stk_pipeline_defaults.py", "Website production STK mode defaults"),
    ("gui/stk_final_v1_precompute_cache.py", "Note cache precompute for production path"),
    ("gui/test_stk_pipeline_default_and_cache.py", "Production path regression tests"),
    ("gui/stk_v6_2_audit_features.py", "PGSM Step 1/2 data-mapping dependency"),
    ("gui/body_hybrid_v4_1_identity_space.py", "Production stk_body_transfer_final_v1 synthesis"),
)

WEBSITE_IMPORT_MODULES: Tuple[str, ...] = (
    "diagnostic_synthesis",
    "body_response_synth",
    "body_hybrid_v4_1_identity_space",
    "stk_pipeline_defaults",
    "stk_final_v1_precompute_cache",
    "build_note_cache",
    "pgsm_physical_factor_registry",
    "pgsm_physical_interaction_map",
    "stk_v6_2_audit_features",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rm_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _is_obsolete_report_name(name: str) -> bool:
    if name.startswith("pgsm_") or name.startswith("stk_v6_physical_dof_audit"):
        return False
    return any(name.startswith(p) for p in DEBUG_REPORT_PREFIXES)


def _is_obsolete_audio_dir_name(name: str) -> bool:
    if name == "note_cache":
        return False
    return any(name.startswith(p) or p in name for p in AUDIO_DIR_PREFIXES)


def delete_legacy_stk_artifacts(*, repo_root: Path | None = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    removed_code: List[str] = []
    removed_reports: List[str] = []
    removed_dirs: List[str] = []

    for rel in DELETED_CODE_FILES:
        p = root / rel
        if _rm_path(p):
            removed_code.append(rel)

    gui = root / "gui"
    for pattern in OBSOLETE_CODE_GLOBS:
        for p in gui.glob(pattern):
            rel = p.relative_to(root).as_posix()
            if rel in OBSOLETE_CODE_EXCEPTIONS:
                continue
            if _rm_path(p):
                removed_code.append(rel)

    report_dir = root / "audio" / "debug_reports"
    if report_dir.is_dir():
        for p in list(report_dir.iterdir()):
            if p.is_file() and _is_obsolete_report_name(p.name):
                rel = p.relative_to(root).as_posix()
                if _rm_path(p):
                    removed_reports.append(rel)

    audio_root = root / "audio"
    if audio_root.is_dir():
        for p in list(audio_root.iterdir()):
            if p.is_dir() and _is_obsolete_audio_dir_name(p.name):
                rel = p.relative_to(root).as_posix()
                if _rm_path(p):
                    removed_dirs.append(rel)

    return {
        "removed_code_files": sorted(set(removed_code)),
        "removed_debug_reports": sorted(set(removed_reports)),
        "removed_audio_dirs": sorted(set(removed_dirs)),
    }


def _glob_obsolete_code_remaining(*, repo_root: Path) -> List[str]:
    found: List[str] = []
    gui = repo_root / "gui"
    for pattern in OBSOLETE_CODE_GLOBS:
        for p in gui.glob(pattern):
            rel = p.relative_to(repo_root).as_posix()
            if rel in OBSOLETE_CODE_EXCEPTIONS:
                continue
            found.append(rel)
    return sorted(found)


def _obsolete_reports_remaining(*, repo_root: Path) -> List[str]:
    report_dir = repo_root / "audio" / "debug_reports"
    if not report_dir.is_dir():
        return []
    return sorted(
        f"audio/debug_reports/{p.name}"
        for p in report_dir.iterdir()
        if p.is_file() and _is_obsolete_report_name(p.name)
    )


def _obsolete_audio_dirs_remaining(*, repo_root: Path) -> List[str]:
    audio_root = repo_root / "audio"
    if not audio_root.is_dir():
        return []
    return sorted(
        p.relative_to(repo_root).as_posix()
        for p in audio_root.iterdir()
        if p.is_dir() and _is_obsolete_audio_dir_name(p.name)
    )


def _scan_disallowed_legacy_references(*, repo_root: Path) -> List[str]:
    patterns = (
        "build_stk_v3", "build_stk_v4", "stage45_damping", "stage46_research",
        "stage47_reports", "stage48_timbre", "stage49_stk", "stage50_stk",
        "stage51_stk", "stage51b_", "stage51c_", "stage51d_", "stage51e_",
        "stage51f_", "stage51g_", "stage51_rom", "stage52a_stk",
        "build_body_timbre_decomposition_stage48",
    )
    allowed_files = {
        "pgsm_step1_1c_legacy_stk_cleanup.py",
        "pgsm_step1_1_cleanup.py",
        "test_pgsm_step1_1c_legacy_stk_cleanup.py",
        "build_body_difference_diagnostics.py",
        "audit_synthesis_model.py",
    }
    refs: List[str] = []
    for py in (repo_root / "gui").glob("*.py"):
        if py.name in allowed_files:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for pat in patterns:
            if pat in text:
                refs.append(f"{py.relative_to(repo_root).as_posix()}: contains '{pat}'")
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
    code_remaining = _glob_obsolete_code_remaining(repo_root=root)
    reports_remaining = _obsolete_reports_remaining(repo_root=root)
    dirs_remaining = _obsolete_audio_dirs_remaining(repo_root=root)
    legacy_refs = _scan_disallowed_legacy_references(repo_root=root)
    kept_missing = [rel for rel, _ in KEPT_PRODUCTION_FILES if not (root / rel).is_file()]
    import_status = check_website_imports()
    website_ok = DEFAULT_WEBSITE_STK_MODE == "stk_body_transfer_final_v1"

    return {
        "code_obsolete_remaining": code_remaining,
        "debug_reports_remaining": reports_remaining,
        "audio_dirs_remaining": dirs_remaining,
        "legacy_references_in_gui": legacy_refs,
        "kept_production_files_missing": kept_missing,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_ok": website_ok,
        "import_check": import_status,
        "all_clean": (
            not code_remaining
            and not reports_remaining
            and not dirs_remaining
            and not kept_missing
            and website_ok
            and import_status["pass"]
        ),
    }


def build_cleanup_report(
    *,
    repo_root: Path | None = None,
    deletion_result: Dict[str, Any] | None = None,
    tests_run: Sequence[str] | None = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    status = verify_cleanup_status(repo_root=root)
    return {
        "report_version": PGSM_STEP1_1C_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step1_1c_legacy_stk_cleanup_complete" if status["all_clean"] else "pgsm_step1_1c_legacy_stk_cleanup_incomplete",
        "no_audio_generated": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": status["website_default_ok"],
        "deletion_run": deletion_result or {},
        "deleted_code_files": list(DELETED_CODE_FILES),
        "kept_production_files": [{"path": p, "reason": r} for p, r in KEPT_PRODUCTION_FILES],
        "remaining_legacy_references": status["legacy_references_in_gui"],
        "verification": status,
        "tests_run": list(tests_run or []),
        "explicit_statement": "Legacy STK V3/V4 and Stage 42–52 prototype code/artifacts were deleted, not archived.",
        "pgsm_note": "PGSM Step 3 not started. Production path stk_body_transfer_final_v1 retained.",
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    v = report.get("verification") or {}
    lines = [
        "# PGSM Step 1.1c — Legacy STK V3/V4 / Stage 42–52 cleanup",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"Website default: `{v.get('website_default')}`",
        "",
        "## Kept production / PGSM files",
        "",
    ]
    for item in report.get("kept_production_files") or []:
        lines.append(f"- `{item['path']}` — {item['reason']}")
    lines.extend(["", "## Verification", ""])
    lines.append(f"- Obsolete code remaining: {len(v.get('code_obsolete_remaining') or [])}")
    lines.append(f"- Obsolete debug reports remaining: {len(v.get('debug_reports_remaining') or [])}")
    lines.append(f"- Obsolete audio dirs remaining: {len(v.get('audio_dirs_remaining') or [])}")
    lines.append(f"- Import check pass: {v.get('import_check', {}).get('pass')}")
    if v.get("legacy_references_in_gui"):
        lines.extend(["", "## Remaining legacy string references (non-import)", ""])
        for ref in v["legacy_references_in_gui"]:
            lines.append(f"- {ref}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cleanup_reports(
    *,
    repo_root: Path | None = None,
    json_path: Path | None = None,
    md_path: Path | None = None,
    run_deletion: bool = False,
    tests_run: Sequence[str] | None = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    deletion = delete_legacy_stk_artifacts(repo_root=root) if run_deletion else None
    report = build_cleanup_report(repo_root=root, deletion_result=deletion, tests_run=tests_run)
    jp = Path(json_path or REPORT_JSON)
    mp = Path(md_path or REPORT_MD)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mp)
    return report


if __name__ == "__main__":
    r = write_cleanup_reports(run_deletion=True)
    dr = r.get("deletion_run") or {}
    if any(dr.get(k) for k in ("removed_code_files", "removed_debug_reports", "removed_audio_dirs")):
        print("Deleted legacy artifacts:")
        for k in ("removed_code_files", "removed_debug_reports", "removed_audio_dirs"):
            for item in dr.get(k) or []:
                print(f"  - {item}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Status: {r['status']}")
    v = r.get("verification") or {}
    print(f"All clean: {v.get('all_clean')}")
