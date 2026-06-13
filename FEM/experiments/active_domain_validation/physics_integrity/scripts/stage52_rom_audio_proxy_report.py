#!/usr/bin/env python3
"""Stage 5.2 — ROM audio-proxy LOO validation report (diagnostic only)."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage52_rom_audio_proxy_candidates import (  # noqa: E402
    diagnose_audio_proxy_weakness,
    evaluate_candidates_on_comparisons,
)
from v2_b3_m4_lhs_pool_bridge import load_lhs_pool  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    VALIDATION_LEAVE_ONE_OUT,
    comparisons_project_dir,
    project_comparison_copy_path,
    rom_comparison_path,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel  # noqa: E402

SCHEMA_VERSION = "stage52_rom_audio_proxy_v1"

DEFAULT_LOO_SAMPLES: Tuple[str, ...] = tuple(
    f"sample_{n:03d}"
    for n in (
        10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
    )
)

OPTIONAL_LATEST_LOO_SAMPLES: Tuple[str, ...] = tuple(f"sample_{n:03d}" for n in range(56, 66))

FREQUENCY_GATES = {
    "median_relative_error_max": 0.025,
    "p90_relative_error_max": 0.07,
}

AUDIO_GATES = {
    "dominant_region_acc_min": 0.85,
    "coupling_acc_min": 0.75,
    "top_k_radiation_overlap_min": 0.35,
    "top_k_radiation_overlap_target": 0.50,
    "mic_norm_mae_improvement_min_fraction": 0.10,
    "radiation_log_mae_improvement_min_fraction": 0.10,
}


def _median(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.median(vals), 8) if vals else None


def _mean(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.mean(vals), 8) if vals else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _is_valid_loo(comparison: Mapping[str, Any]) -> bool:
    return (
        str(comparison.get("status")) == "COMPLETED"
        and str(comparison.get("validation_mode")) == VALIDATION_LEAVE_ONE_OUT
        and comparison.get("training_includes_target") is False
        and comparison.get("accuracy_meaningful") is True
    )


def discover_comparison_files(
    repo_root: Path,
    *,
    shape_name: str = "classic",
) -> List[Path]:
    found: Dict[str, Path] = {}
    cmp_dir = comparisons_project_dir(repo_root, shape_name)
    if cmp_dir.is_dir():
        for path in sorted(cmp_dir.glob("*_rom_fom_comparison.json")):
            try:
                doc = load_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            sid = str(doc.get("sample_id") or path.name.split("__")[0])
            found[sid] = path

    pool_path = repo_root / "ROM" / shape_name / "lhs_pool.json"
    if pool_path.is_file():
        pool = load_lhs_pool(pool_path)
        for entry in pool.get("entries") or []:
            sid = str(entry.get("id") or entry.get("sample_id") or "")
            rid = str(entry.get("last_run_id") or "")
            if not sid or not rid:
                continue
            run_dir_raw = entry.get("last_run_dir")
            if run_dir_raw:
                run_root = Path(str(run_dir_raw))
                if not run_root.is_absolute():
                    run_root = repo_root / run_root
            else:
                run_root = (
                    repo_root
                    / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
                    / sid
                    / "runs"
                    / rid
                )
            run_cmp = rom_comparison_path(run_root)
            if run_cmp.is_file() and sid not in found:
                found[sid] = run_cmp
            proj = project_comparison_copy_path(
                repo_root, shape_name=shape_name, sample_id=sid, run_id=rid
            )
            if proj.is_file():
                found[sid] = proj
    return sorted(found.values(), key=lambda p: p.name)


def load_loo_comparisons(
    repo_root: Path,
    *,
    shape_name: str = "classic",
    sample_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    wanted = set(sample_filter) if sample_filter else None
    for path in discover_comparison_files(repo_root, shape_name=shape_name):
        try:
            doc = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        sid = str(doc.get("sample_id") or "")
        if wanted is not None and sid not in wanted:
            continue
        doc = dict(doc)
        doc["_source_path"] = str(path)
        if _is_valid_loo(doc):
            valid.append(doc)
        else:
            rejected.append(
                {
                    "sample_id": sid,
                    "path": str(path),
                    "validation_mode": doc.get("validation_mode"),
                    "training_includes_target": doc.get("training_includes_target"),
                    "accuracy_meaningful": doc.get("accuracy_meaningful"),
                    "status": doc.get("status"),
                }
            )
    valid.sort(key=lambda d: str(d.get("sample_id") or ""))
    return valid, rejected


def aggregate_metrics(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    freq_med: List[float] = []
    freq_mean: List[float] = []
    freq_p90: List[float] = []
    freq_abs: List[float] = []
    p2_keys = (
        "radiation_proxy_log_mae",
        "radiation_proxy_relative_error_median",
        "radiation_proxy_rank_correlation",
        "top_k_radiation_overlap",
        "mic_output_proxy_p95_norm_mae",
        "coupling_class_accuracy",
        "dominant_region_accuracy",
        "top_share_mae",
        "back_share_mae",
        "air_share_mae",
    )
    p2_buckets: Dict[str, List[float]] = {k: [] for k in p2_keys}

    per_sample: List[Dict[str, Any]] = []
    for comp in comparisons:
        sid = comp.get("sample_id")
        p2 = comp.get("phase2_scalar_metrics") or {}
        row = {
            "sample_id": sid,
            "matched_mode_count": comp.get("matched_mode_count"),
            "median_relative_error": comp.get("median_relative_error"),
            "mean_relative_error": comp.get("mean_relative_error"),
            "p90_relative_error": comp.get("p90_relative_error"),
            "median_abs_error_hz": comp.get("median_abs_error_hz"),
            "source_path": comp.get("_source_path"),
        }
        for key in p2_keys:
            val = p2.get(key)
            row[key] = val
            if val is not None and val == val:
                p2_buckets[key].append(float(val))
        per_sample.append(row)

        for key, bucket in (
            ("median_relative_error", freq_med),
            ("mean_relative_error", freq_mean),
            ("p90_relative_error", freq_p90),
            ("median_abs_error_hz", freq_abs),
        ):
            val = comp.get(key)
            if val is not None and val == val:
                bucket.append(float(val))

    return {
        "sample_count": len(comparisons),
        "sample_ids": [str(c.get("sample_id")) for c in comparisons],
        "per_sample": per_sample,
        "frequency": {
            "median_relative_error": _median(freq_med),
            "mean_relative_error": _mean(freq_mean),
            "p90_relative_error": _median(freq_p90),
            "median_abs_error_hz": _median(freq_abs),
        },
        "audio_proxy": {k: _median(p2_buckets[k]) for k in p2_keys},
    }


def _gate_status(value: Optional[float], *, max_val: Optional[float] = None, min_val: Optional[float] = None) -> str:
    if value is None or value != value:
        return "UNKNOWN"
    if max_val is not None and float(value) > float(max_val):
        return "FAIL"
    if min_val is not None and float(value) < float(min_val):
        return "FAIL"
    return "PASS"


def assess_verdicts(aggregated: Mapping[str, Any]) -> Dict[str, Any]:
    freq = aggregated.get("frequency") or {}
    audio = aggregated.get("audio_proxy") or {}
    sample_count = int(aggregated.get("sample_count") or 0)

    if sample_count == 0:
        return {
            "frequency_prediction_status": "UNKNOWN",
            "radiation_prediction_status": "UNKNOWN",
            "mic_prediction_status": "UNKNOWN",
            "coupling_prediction_status": "UNKNOWN",
            "overall_rom_readiness": "NOT_READY_NO_LOO_DATA",
            "recommendation": "A. Run Stage 5.2 LOO validation batch on VM with surrogate npz and FOM catalogs present",
            "gates": {
                "frequency": FREQUENCY_GATES,
                "audio": AUDIO_GATES,
            },
            "answers": {
                "rom_frequency_ready": False,
                "rom_radiation_audio_proxy_ready": False,
                "rom_ready_for_next_body_family_diagnostic": False,
                "rom_ready_for_next_body_family_production": False,
            },
        }

    freq_status = "PASS"
    if _gate_status(freq.get("median_relative_error"), max_val=FREQUENCY_GATES["median_relative_error_max"]) == "FAIL":
        freq_status = "WEAK"
    if _gate_status(freq.get("p90_relative_error"), max_val=FREQUENCY_GATES["p90_relative_error_max"]) == "FAIL":
        freq_status = "WEAK" if freq_status == "PASS" else "FAIL"

    rad_status = "PASS"
    for key, min_v in (
        ("dominant_region_accuracy", AUDIO_GATES["dominant_region_acc_min"]),
        ("coupling_class_accuracy", AUDIO_GATES["coupling_acc_min"]),
        ("top_k_radiation_overlap", AUDIO_GATES["top_k_radiation_overlap_min"]),
    ):
        if _gate_status(audio.get(key), min_val=min_v) == "FAIL":
            rad_status = "FAIL"
            break
    if rad_status == "PASS" and _gate_status(
        audio.get("top_k_radiation_overlap"),
        min_val=AUDIO_GATES["top_k_radiation_overlap_target"],
    ) == "FAIL":
        rad_status = "WEAK"

    mic_mae = audio.get("mic_output_proxy_p95_norm_mae")
    mic_status = "UNKNOWN" if mic_mae is None else ("WEAK" if float(mic_mae) > 0.30 else "PASS")

    coupling_status = _gate_status(
        audio.get("coupling_class_accuracy"),
        min_val=AUDIO_GATES["coupling_acc_min"],
    )
    if coupling_status == "UNKNOWN":
        coupling_status = "UNKNOWN"
    elif coupling_status == "FAIL":
        coupling_status = "FAIL"
    else:
        coupling_status = "PASS"

    freq_ready = freq_status in ("PASS", "WEAK")
    audio_ready = rad_status in ("PASS", "WEAK") and mic_status in ("PASS", "WEAK")
    production_ready = (
        freq_status == "PASS"
        and rad_status == "PASS"
        and mic_status == "PASS"
        and coupling_status == "PASS"
    )
    diagnostic_ready = freq_ready and aggregated.get("sample_count", 0) >= 3

    if production_ready:
        overall = "READY_FOR_NEXT_BODY_FAMILY_PRODUCTION"
        recommendation = "C. Start next body family diagnostic with conservative audio weighting"
    elif diagnostic_ready and freq_status in ("PASS", "WEAK"):
        overall = "READY_FOR_NEXT_BODY_DIAGNOSTIC"
        recommendation = "B. Improve audio proxy model; frequency adequate for diagnostic next-body work"
    elif aggregated.get("sample_count", 0) == 0:
        overall = "NOT_READY_NO_LOO_DATA"
        recommendation = "A. Run Stage 5.2 LOO validation batch on VM with surrogate npz present"
    else:
        overall = "NOT_READY_FOR_NEXT_BODY_PRODUCTION"
        recommendation = "B. Improve audio proxy model before next body family production"

    return {
        "frequency_prediction_status": freq_status,
        "radiation_prediction_status": rad_status,
        "mic_prediction_status": mic_status,
        "coupling_prediction_status": coupling_status,
        "overall_rom_readiness": overall,
        "recommendation": recommendation,
        "gates": {
            "frequency": FREQUENCY_GATES,
            "audio": AUDIO_GATES,
        },
        "answers": {
            "rom_frequency_ready": freq_status == "PASS",
            "rom_radiation_audio_proxy_ready": rad_status == "PASS",
            "rom_ready_for_next_body_family_diagnostic": diagnostic_ready,
            "rom_ready_for_next_body_family_production": production_ready,
        },
    }


def stk_integration_notes(verdicts: Mapping[str, Any], aggregated: Mapping[str, Any]) -> Dict[str, Any]:
    freq = aggregated.get("frequency") or {}
    audio = aggregated.get("audio_proxy") or {}
    notes: List[str] = []
    if verdicts.get("frequency_prediction_status") in ("PASS", "WEAK"):
        notes.append(
            "STK may use predicted modal frequencies; keep radiation shaping conservative when radiation status is not PASS."
        )
    top_k = audio.get("top_k_radiation_overlap")
    if top_k is not None and float(top_k) >= AUDIO_GATES["top_k_radiation_overlap_min"]:
        notes.append(
            "Top-k radiation overlap is usable for selecting important body modes even when absolute radiation values are imperfect."
        )
    else:
        notes.append(
            "Do not rely on raw radiation values as production weights; use bounded or rank-normalized weights only."
        )
    if verdicts.get("coupling_prediction_status") == "FAIL":
        notes.append(
            "Coupling / bridge excitation prediction is weak — avoid coupling-class-conditioned STK routing without manual bounds."
        )
    return {
        "guidance": notes,
        "use_predicted_frequencies": verdicts.get("frequency_prediction_status") in ("PASS", "WEAK"),
        "use_raw_radiation_weights": verdicts.get("radiation_prediction_status") == "PASS",
        "use_rank_normalized_weights": verdicts.get("radiation_prediction_status") != "PASS",
    }


def run_missing_loo_comparisons(
    repo_root: Path,
    *,
    samples: Sequence[str],
    lhs_json: Path,
    write_csv: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    script = SCRIPT_DIR / "run_m4_rom_compare.py"
    results: List[Dict[str, Any]] = []
    for sid in samples:
        cmd = [
            sys.executable,
            str(script),
            "--lhs-json",
            str(lhs_json if lhs_json.is_absolute() else repo_root / lhs_json),
            "--force-sample",
            sid,
            "--leave-one-out",
            "--run-prepredict",
            "--debug",
        ]
        if write_csv:
            cmd.append("--write-csv")
        if dry_run:
            cmd.append("--dry-run")
            results.append({"sample_id": sid, "command": cmd, "status": "dry_run"})
            continue
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        results.append(
            {
                "sample_id": sid,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
            }
        )
    return {"requested_samples": list(samples), "runs": results}


def build_stage52_report(
    repo_root: Path,
    *,
    shape_name: str = "classic",
    sample_filter: Optional[Sequence[str]] = None,
    run_missing_loo: bool = False,
    write_csv: bool = False,
    evaluate_candidates: bool = True,
    out_json: Optional[Path] = None,
    out_md: Optional[Path] = None,
) -> Dict[str, Any]:
    lhs_path = repo_root / "ROM" / shape_name / "lhs_pool.json"
    missing_run_result = None
    if run_missing_loo and sample_filter:
        existing, _ = load_loo_comparisons(repo_root, shape_name=shape_name)
        existing_ids = {str(c.get("sample_id")) for c in existing}
        missing = [s for s in sample_filter if s not in existing_ids]
        if missing:
            missing_run_result = run_missing_loo_comparisons(
                repo_root,
                samples=missing,
                lhs_json=lhs_path,
                write_csv=write_csv,
            )

    comparisons, rejected = load_loo_comparisons(
        repo_root,
        shape_name=shape_name,
        sample_filter=sample_filter,
    )
    aggregated = aggregate_metrics(comparisons)
    diagnosis = diagnose_audio_proxy_weakness(comparisons) if comparisons else {}
    verdicts = assess_verdicts(aggregated)
    stk_notes = stk_integration_notes(verdicts, aggregated)

    candidate_results: Dict[str, Any] = {}
    if evaluate_candidates and comparisons:
        candidate_results = evaluate_candidates_on_comparisons(comparisons)

    sur_npz = repo_root / "ROM" / shape_name / "m4_modal_surrogate.npz"
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "fem_launched": False,
        "production_surrogate_overwritten": False,
        "inputs": {
            "shape_name": shape_name,
            "comparison_search_dirs": [
                str(comparisons_project_dir(repo_root, shape_name)),
                "run_dir/rom/rom_fom_comparison.json",
            ],
            "sample_filter": list(sample_filter) if sample_filter else None,
            "m4_surrogate_npz_present": sur_npz.is_file(),
        },
        "loo_validation": {
            "valid_sample_count": len(comparisons),
            "sample_ids": aggregated.get("sample_ids") or [],
            "rejected_comparisons": rejected,
            "validation_checks": {
                "validation_mode": VALIDATION_LEAVE_ONE_OUT,
                "training_includes_target": False,
                "accuracy_meaningful": True,
            },
            "missing_loo_run": missing_run_result,
        },
        "metrics": aggregated,
        "diagnosis": diagnosis,
        "candidate_evaluation": candidate_results,
        "verdicts": verdicts,
        "stk_integration": stk_notes,
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if out_md:
        out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    loo = report.get("loo_validation") or {}
    metrics = report.get("metrics") or {}
    freq = metrics.get("frequency") or {}
    audio = metrics.get("audio_proxy") or {}
    verdicts = report.get("verdicts") or {}
    answers = verdicts.get("answers") or {}
    lines = [
        "# Stage 5.2 — ROM audio-proxy validation report",
        "",
        f"- **LOO samples found:** {loo.get('valid_sample_count')}",
        f"- **Sample IDs:** {', '.join(metrics.get('sample_ids') or []) or '(none)'}",
        f"- **validation_mode:** leave_one_out | **train_includes_target:** false | **meaningful:** true",
        f"- **Surrogate npz present:** {(report.get('inputs') or {}).get('m4_surrogate_npz_present')}",
        f"- **FEM launched:** {report.get('fem_launched')}",
        f"- **Production surrogate overwritten:** {report.get('production_surrogate_overwritten')}",
        "",
        "## Frequency metrics (median across LOO samples)",
        "",
        f"| metric | value |",
        f"|--------|-------|",
        f"| median_relative_error | {freq.get('median_relative_error')} |",
        f"| mean_relative_error | {freq.get('mean_relative_error')} |",
        f"| p90_relative_error | {freq.get('p90_relative_error')} |",
        f"| median_abs_error_hz | {freq.get('median_abs_error_hz')} |",
        "",
        "## Audio proxy metrics (median across LOO samples)",
        "",
        f"| metric | value |",
        f"|--------|-------|",
        f"| radiation_proxy_log_mae | {audio.get('radiation_proxy_log_mae')} |",
        f"| radiation_proxy_relative_error_median | {audio.get('radiation_proxy_relative_error_median')} |",
        f"| radiation_proxy_rank_correlation | {audio.get('radiation_proxy_rank_correlation')} |",
        f"| top_k_radiation_overlap | {audio.get('top_k_radiation_overlap')} |",
        f"| mic_output_proxy_p95_norm_mae | {audio.get('mic_output_proxy_p95_norm_mae')} |",
        f"| coupling_class_accuracy | {audio.get('coupling_class_accuracy')} |",
        f"| dominant_region_accuracy | {audio.get('dominant_region_accuracy')} |",
        f"| top_share_mae | {audio.get('top_share_mae')} |",
        f"| back_share_mae | {audio.get('back_share_mae')} |",
        f"| air_share_mae | {audio.get('air_share_mae')} |",
        "",
        "## Verdicts",
        "",
        f"- **frequency_prediction_status:** {verdicts.get('frequency_prediction_status')}",
        f"- **radiation_prediction_status:** {verdicts.get('radiation_prediction_status')}",
        f"- **mic_prediction_status:** {verdicts.get('mic_prediction_status')}",
        f"- **coupling_prediction_status:** {verdicts.get('coupling_prediction_status')}",
        f"- **overall_rom_readiness:** {verdicts.get('overall_rom_readiness')}",
        f"- **Recommendation:** {verdicts.get('recommendation')}",
        "",
        "## Explicit answers",
        "",
        f"- ROM frequency ready? **{answers.get('rom_frequency_ready')}**",
        f"- ROM radiation/audio proxy ready? **{answers.get('rom_radiation_audio_proxy_ready')}**",
        f"- ROM ready for next body family diagnostic? **{answers.get('rom_ready_for_next_body_family_diagnostic')}**",
        f"- ROM ready for next body family production? **{answers.get('rom_ready_for_next_body_family_production')}**",
        "",
        "## STK integration",
        "",
    ]
    for note in (report.get("stk_integration") or {}).get("guidance") or []:
        lines.append(f"- {note}")
    diag = report.get("diagnosis") or {}
    if diag:
        lines.extend(
            [
                "",
                "## Audio proxy diagnosis",
                "",
                f"- Likely contributors: {', '.join(diag.get('likely_contributors') or []) or '(insufficient data)'}",
                f"- Absolute radiation weak: {diag.get('absolute_radiation_prediction_weak')}",
                f"- Rank/top-k relatively better: {diag.get('rank_or_top_k_relatively_better')}",
            ]
        )
    if metrics.get("per_sample"):
        lines.extend(["", "## Per-sample table", ""])
        lines.append("| sample | median_rel | rad_log_mae | mic_norm_mae | top_k_rad | coupling_acc | dom_region_acc |")
        lines.append("|--------|------------|-------------|--------------|-----------|--------------|----------------|")
        for row in metrics["per_sample"]:
            lines.append(
                f"| {row.get('sample_id')} | {row.get('median_relative_error')} | "
                f"{row.get('radiation_proxy_log_mae')} | {row.get('mic_output_proxy_p95_norm_mae')} | "
                f"{row.get('top_k_radiation_overlap')} | {row.get('coupling_class_accuracy')} | "
                f"{row.get('dominant_region_accuracy')} |"
            )
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--shape", default="classic")
    parser.add_argument("--samples", nargs="*", default=None, help="Optional sample_id filter")
    parser.add_argument("--run-missing-loo", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root or detect_repo_root(SCRIPT_DIR)
    out_dir = repo_root / "audio" / "debug_reports"
    report = build_stage52_report(
        repo_root,
        shape_name=str(args.shape),
        sample_filter=args.samples,
        run_missing_loo=bool(args.run_missing_loo),
        write_csv=bool(args.write_csv),
        out_json=args.out_json or out_dir / "stage52_rom_audio_proxy_report.json",
        out_md=args.out_md or out_dir / "stage52_rom_audio_proxy_report.md",
    )
    print(f"loo_samples={report['loo_validation']['valid_sample_count']}")
    print(f"overall={report['verdicts']['overall_rom_readiness']}")
    print(f"wrote {rel(out_dir / 'stage52_rom_audio_proxy_report.json', repo_root=repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
