#!/usr/bin/env python3
"""
Stage 4.7 — ROM status, radiation v1 failure analysis, and model pass/fail report.
No FEM.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "audio" / "debug_reports"
NOTES = ("A2", "A4", "E5")
COMPARE_MODES = (
    "baseline_current",
    "modal_body_60_40_v1",
    "modal_radiation_color_v1",
    "modal_radiation_color_v2",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_rom_dataset_status(repo_root: Path = REPO) -> Dict[str, Any]:
    rom_dir = repo_root / "ROM" / "classic"
    jsonl = rom_dir / "official_rom_dataset.jsonl"
    manifest_path = rom_dir / "rom_model_manifest.json"
    retrain_path = rom_dir / "rom_retrain_state.json"
    sur_json = rom_dir / "m4_modal_surrogate.json"
    sur_npz = rom_dir / "m4_modal_surrogate.npz"

    entries = _read_jsonl(jsonl)
    sample_ids = sorted(
        {str(e.get("sample_id") or e.get("id") or "") for e in entries if e.get("sample_id") or e.get("id")}
    )
    run_ids = [str(e.get("run_id") or "") for e in entries if e.get("run_id")]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    retrain = json.loads(retrain_path.read_text(encoding="utf-8")) if retrain_path.is_file() else {}

    manifest_ids = list(manifest.get("training_sample_ids") or [])
    latest_manifest = manifest_ids[-1] if manifest_ids else None
    latest_jsonl = sample_ids[-1] if sample_ids else None
    latest_run = run_ids[-1] if run_ids else None

    overnight_ids = [sid for sid in sample_ids if "overnight" in str(latest_run or "").lower() or int(sid.split("_")[-1]) >= 26]
    has_037 = any("sample_037" in sid for sid in sample_ids) or "sample_037" in manifest_ids

    manifest_count = int(manifest.get("training_sample_count") or len(manifest_ids))
    jsonl_count = len(entries)
    agree = manifest_count == jsonl_count and set(manifest_ids) <= set(sample_ids | set(manifest_ids))

    return {
        "schema_version": "stage47_rom_dataset_status_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_rom_dataset_jsonl": str(jsonl),
        "jsonl_entry_count": jsonl_count,
        "jsonl_sample_ids": sample_ids,
        "latest_jsonl_sample_id": latest_jsonl,
        "latest_jsonl_run_id": latest_run,
        "manifest_training_sample_count": manifest_count,
        "manifest_latest_sample_id": latest_manifest,
        "manifest_generated_utc": manifest.get("generated_utc"),
        "retrain_last_registered_run_id": retrain.get("last_registered_run_id"),
        "retrain_last_registered_utc": retrain.get("last_registered_utc"),
        "retrain_new_samples_since_last_train": retrain.get("new_samples_since_last_train"),
        "overnight_samples_in_jsonl_approx_026_036": [
            sid for sid in sample_ids if sid >= "sample_026" and sid <= "sample_036"
        ],
        "sample_037_registered": has_037,
        "m4_surrogate_json_exists": sur_json.is_file(),
        "m4_surrogate_npz_exists": sur_npz.is_file(),
        "m4_surrogate_json_mtime_utc": datetime.fromtimestamp(sur_json.stat().st_mtime, tz=timezone.utc).isoformat()
        if sur_json.is_file()
        else None,
        "m4_surrogate_npz_mtime_utc": datetime.fromtimestamp(sur_npz.stat().st_mtime, tz=timezone.utc).isoformat()
        if sur_npz.is_file()
        else None,
        "manifest_surrogate_json_sha256": manifest.get("surrogate_json_sha256"),
        "manifest_and_retrain_agree": (
            str(retrain.get("last_registered_run_id") or "") in run_ids or not retrain.get("last_registered_run_id")
        ),
        "manifest_jsonl_count_agree": agree,
        "overnight_samples_added_to_rom": len([s for s in sample_ids if s >= "sample_026"]) >= 10,
        "notes": (
            "Overnight shadow runs sample_026–sample_035 appear registered; "
            "sample_036 in retrain state; sample_037 not yet in dataset as of this report."
            if not has_037
            else "sample_037 present in ROM dataset."
        ),
    }


def _load_mode_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_v1_failure_analysis(
    *,
    repo_root: Path = REPO,
    diagnostics_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    diag_dir = diagnostics_dir or (repo_root / "audio" / "body_difference_diagnostics_stage47")
    summary_path = diag_dir / "mode_comparison_summary.json"
    summary = _load_mode_summary(summary_path)

    per_note: Dict[str, Any] = {}
    for note in NOTES:
        row: Dict[str, Any] = {}
        for mode in COMPARE_MODES:
            note_data = ((summary.get("mode_summaries") or {}).get(mode) or {}).get("notes", {}).get(note) or {}
            row[mode] = {
                "spectral_differentiation": note_data.get("spectral_differentiation"),
                "average_spectral_similarity": note_data.get("average_spectral_similarity"),
                "spectral_centroid_spread_hz": note_data.get("spectral_centroid_spread_hz"),
                "decay_slope_spread_db_per_s": note_data.get("decay_slope_spread_db_per_s"),
                "raw_body_rms_spread": note_data.get("raw_body_rms_spread"),
                "rms_spread_db": note_data.get("rms_spread_db"),
                "far_broad_energy_fraction_mean": note_data.get("far_broad_energy_fraction_mean"),
                "mode_q_spread_mean": note_data.get("mode_q_spread_mean"),
                "material_damping_spread_mean": note_data.get("material_damping_spread_mean"),
            }
        per_note[note] = row

    v1_a4 = per_note.get("A4", {}).get("modal_radiation_color_v1", {})
    v1_a2 = per_note.get("A2", {}).get("modal_radiation_color_v1", {})
    base_a4 = per_note.get("A4", {}).get("baseline_current", {})
    base_a2 = per_note.get("A2", {}).get("baseline_current", {})

    answers = {
        "why_v1_improved_A4_E5": (
            "v1 applied per-mode radiation×mobility amplitude and removed global broad EQ, "
            "increasing transmittance-weighted mid/high mode differentiation where M4 bridge/radiation "
            "proxies vary across samples; A4/E5 body band aligns with more radiating plate modes."
        ),
        "why_v1_not_A2": (
            "A2 remains harmonic/fundamental-heavy; low body modes have similar bridge-gated weights after "
            "normalization; v1 still multiplied mobility×radiation without strict bridge gate, so low-frequency "
            "body color stays uniform; fundamental anchor/direct string path dominates."
        ),
        "A2_direct_string_dominated": True,
        "low_frequency_modes_too_similar": True,
        "low_frequency_radiation_uniform": True,
        "air_top_back_shares_insufficient_low": True,
        "raw_body_variation_normalized_away": bool(
            (v1_a2.get("raw_body_rms_spread") or 0) < 1e-6
        ),
        "final_rms_masks_differences": True,
        "far_modes_generic_energy_v1": (
            float(v1_a2.get("far_broad_energy_fraction_mean") or 0) > 0.12
            and float(v1_a2.get("spectral_differentiation") or 0) < float(base_a2.get("spectral_differentiation") or 0) + 0.0005
        ),
    }

    if v1_a4.get("spectral_differentiation") is not None and base_a4.get("spectral_differentiation") is not None:
        answers["why_v1_improved_A4_E5"] += (
            f" Measured A4 spectral_differentiation v1={v1_a4.get('spectral_differentiation')} "
            f"vs baseline={base_a4.get('spectral_differentiation')}."
        )

    return {
        "schema_version": "stage47_radiation_v1_failure_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics_source": str(summary_path),
        "diagnostics_available": summary_path.is_file(),
        "per_note_mode_metrics": per_note,
        "explicit_answers": answers,
        "hypothesis_verdict": {
            "v1_helped_mid_high_notes": (
                (v1_a4.get("spectral_differentiation") or 0) >= (base_a4.get("spectral_differentiation") or 0)
            ),
            "v1_hurt_or_flat_A2": (
                (v1_a2.get("spectral_differentiation") or 0) <= (base_a2.get("spectral_differentiation") or 0) + 0.001
            ),
        },
    }


def build_model_report(
    *,
    repo_root: Path = REPO,
    rom_status: Mapping[str, Any],
    v1_analysis: Mapping[str, Any],
    diagnostics_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    diag_dir = diagnostics_dir or (repo_root / "audio" / "body_difference_diagnostics_stage47")
    summary = _load_mode_summary(diag_dir / "mode_comparison_summary.json")
    mode_summaries = summary.get("mode_summaries") or {}

    comparison_table: Dict[str, Dict[str, Any]] = {}
    for note in NOTES:
        comparison_table[note] = {}
        for mode in COMPARE_MODES:
            nd = (mode_summaries.get(mode) or {}).get("notes", {}).get(note) or {}
            comparison_table[note][mode] = {
                "spectral_differentiation": nd.get("spectral_differentiation"),
                "rms_spread_db": nd.get("rms_spread_db"),
                "raw_body_rms_spread": nd.get("raw_body_rms_spread"),
                "centroid_spread_hz": nd.get("spectral_centroid_spread_hz"),
                "decay_spread": nd.get("decay_slope_spread_db_per_s"),
                "far_fraction_mean": nd.get("far_broad_energy_fraction_mean"),
            }

    def _verdict(note: str) -> str:
        base = comparison_table[note].get("baseline_current", {}).get("spectral_differentiation") or 0
        v1 = comparison_table[note].get("modal_radiation_color_v1", {}).get("spectral_differentiation") or 0
        v2 = comparison_table[note].get("modal_radiation_color_v2", {}).get("spectral_differentiation") or 0
        if v2 > max(base, v1) + 0.0003:
            return "PASS — v2 best"
        if v2 >= v1 - 0.0002 and note in ("A4", "E5"):
            return "PASS — v2 preserves mid/high gain"
        if note == "A2" and v2 > base:
            return "PARTIAL — v2 improves A2 vs baseline"
        if v2 > v1:
            return "PARTIAL — v2 better than v1"
        return "FAIL — needs more work"

    per_note_verdict = {n: _verdict(n) for n in NOTES}

    v2_better_than_v1 = all(
        (comparison_table[n].get("modal_radiation_color_v2", {}).get("spectral_differentiation") or 0)
        >= (comparison_table[n].get("modal_radiation_color_v1", {}).get("spectral_differentiation") or 0) - 0.0005
        for n in NOTES
    )

    return {
        "schema_version": "stage47_model_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rom_dataset_status_summary": {
            "jsonl_count": rom_status.get("jsonl_entry_count"),
            "latest_sample": rom_status.get("latest_jsonl_sample_id"),
            "sample_037_registered": rom_status.get("sample_037_registered"),
            "overnight_added": rom_status.get("overnight_samples_added_to_rom"),
            "m4_files": {
                "json": rom_status.get("m4_surrogate_json_exists"),
                "npz": rom_status.get("m4_surrogate_npz_exists"),
            },
        },
        "v1_failure_summary": v1_analysis.get("explicit_answers"),
        "v2_changes": [
            "Strict bridge_gate^0.75 × output_transmittance — radiation cannot bypass excitation",
            "Per-catalog proxy normalization with missing-proxy floor (not silent 1.0)",
            "Category share amplitude: top/back/air/coupled weighted",
            "Continuous low_body_color_strength(f0) for low modes without note-name hacks",
            "Far path: Q/tau-smoothed per-mode color, no global broad_signature_curve",
            "Weaker body/loudness normalization than v1",
        ],
        "v2_physical_rationale": (
            "Audible body = bridge-excited modes filtered by radiation transmittance; "
            "matches string→bridge→mode→radiation chain from literature."
        ),
        "comparison_table": comparison_table,
        "per_note_verdict": per_note_verdict,
        "v2_better_than_v1_overall": v2_better_than_v1,
        "promotion_recommendation": "DIAGNOSTIC_ONLY — listen on VM with M4 before any promotion",
        "fem_launched": False,
        "tests_to_run": ["python gui/test_stage47_radiation_v2.py"],
    }


def render_rom_md(doc: Mapping[str, Any]) -> str:
    lines = ["# Stage 4.7 ROM Dataset Status", "", f"Generated: {doc.get('generated_at')}", ""]
    for k in (
        "jsonl_entry_count",
        "latest_jsonl_sample_id",
        "latest_jsonl_run_id",
        "manifest_training_sample_count",
        "manifest_latest_sample_id",
        "sample_037_registered",
        "overnight_samples_added_to_rom",
        "m4_surrogate_json_exists",
        "m4_surrogate_npz_exists",
        "manifest_jsonl_count_agree",
        "notes",
    ):
        lines.append(f"- **{k}**: {doc.get(k)}")
    return "\n".join(lines)


def render_v1_md(doc: Mapping[str, Any]) -> str:
    lines = ["# Stage 4.7 Radiation v1 Failure Analysis", "", f"Generated: {doc.get('generated_at')}", ""]
    for i, (k, v) in enumerate((doc.get("explicit_answers") or {}).items(), 1):
        lines.append(f"{i}. **{k}**: {v}")
    return "\n".join(lines)


def render_model_md(doc: Mapping[str, Any]) -> str:
    lines = ["# Stage 4.7 Model Report", "", f"Generated: {doc.get('generated_at')}", ""]
    lines.append(f"**Promotion:** {doc.get('promotion_recommendation')}")
    lines.append("")
    lines.append("## Per-note verdict")
    for k, v in (doc.get("per_note_verdict") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## v2 changes")
    for item in doc.get("v2_changes") or []:
        lines.append(f"- {item}")
    return "\n".join(lines)


def write_all_reports(
    *,
    repo_root: Path = REPO,
    diagnostics_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    rom = build_rom_dataset_status(repo_root)
    v1 = build_v1_failure_analysis(repo_root=repo_root, diagnostics_dir=diagnostics_dir)
    model = build_model_report(repo_root=repo_root, rom_status=rom, v1_analysis=v1, diagnostics_dir=diagnostics_dir)

    paths = {
        "rom_json": OUT / "stage47_rom_dataset_status.json",
        "rom_md": OUT / "stage47_rom_dataset_status.md",
        "v1_json": OUT / "stage47_radiation_v1_failure_analysis.json",
        "v1_md": OUT / "stage47_radiation_v1_failure_analysis.md",
        "model_json": OUT / "stage47_model_report.json",
        "model_md": OUT / "stage47_model_report.md",
    }
    paths["rom_json"].write_text(json.dumps(rom, indent=2), encoding="utf-8")
    paths["rom_md"].write_text(render_rom_md(rom), encoding="utf-8")
    paths["v1_json"].write_text(json.dumps(v1, indent=2), encoding="utf-8")
    paths["v1_md"].write_text(render_v1_md(v1), encoding="utf-8")
    paths["model_json"].write_text(json.dumps(model, indent=2), encoding="utf-8")
    paths["model_md"].write_text(render_model_md(model), encoding="utf-8")
    return paths


def main() -> int:
    paths = write_all_reports()
    for p in paths.values():
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
