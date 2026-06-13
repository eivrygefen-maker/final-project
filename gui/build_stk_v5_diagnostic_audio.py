#!/usr/bin/env python3
"""
STK V5 diagnostic audio experiments — lightweight, no FEM/ROM batch.

Compares string/body/radiation layers and current modes vs V5 skeleton prototype.
Writes WAVs + JSON report under audio/stk_v5_diagnostic_audio/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import (  # noqa: E402
    STK_BODY_TRANSFER_FINAL_V1,
    STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
    build_batch_contrast_context,
    build_body_identity_vector,
    requires_identity_contrast_context,
)
from body_response_synth import DEFAULT_DURATION_S, DEFAULT_SAMPLE_RATE  # noqa: E402
from build_sample_comparison import (  # noqa: E402
    load_lhs_sample_entries,
    parse_notes_arg,
    resolve_modal_data_for_sample,
)
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_v5_design_helpers import (  # noqa: E402
    singleton_dz_body_quantification,
    synthesize_mode_to_wav,
)

DEFAULT_OUT = REPO / "audio" / "stk_v5_diagnostic_audio"
DEFAULT_NOTES = "A2,A3,A4,E5"
V41_MODE = "modal_body_hybrid_v4_1_full"

EXPERIMENTS: Tuple[Tuple[str, str], ...] = (
    ("string_only", ""),
    ("current_v4_1_full", V41_MODE),
    ("current_final_v1", STK_BODY_TRANSFER_FINAL_V1),
    ("current_de_thump", STK_BODY_TRANSFER_FINAL_V1_DE_THUMP),
    ("body_only_modal_response", ""),
    ("body_boost_test", ""),
    ("string_attenuated_test", ""),
    ("radiation_emphasized_test", ""),
    ("proposed_v5_skeleton", ""),
    ("stk_v5_alpha_body_dominant", "stk_v5_alpha_body_dominant"),
    ("v5_alpha_s10_b90", "v5_alpha_s10_b90"),
    ("v5_alpha_s20_b80", "v5_alpha_s20_b80"),
    ("v5_alpha_s35_b65", "v5_alpha_s35_b65"),
)

V5_ALPHA_EXPERIMENTS = (
    "stk_v5_alpha_body_dominant",
    "v5_alpha_s10_b90",
    "v5_alpha_s20_b80",
    "v5_alpha_s35_b65",
)
BASELINE_EXPERIMENT = "current_final_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aggregate_metric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [
        float(r["realism_metrics"][key])
        for r in rows
        if r.get("realism_metrics") and r["realism_metrics"].get(key) is not None
    ]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _build_ranking(experiment_summaries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    aggregates: Dict[str, Dict[str, Any]] = {}
    for exp_name, rows in experiment_summaries.items():
        aggregates[exp_name] = {
            "mean_string_dominance_ratio": _aggregate_metric(rows, "string_dominance_ratio"),
            "mean_body_to_string_energy_ratio": _aggregate_metric(rows, "body_to_string_energy_ratio"),
            "mean_body_audibility_index": _aggregate_metric(rows, "body_audibility_index"),
            "mean_metallicity_index": _aggregate_metric(rows, "metallicity_index"),
            "mean_spectral_centroid_hz": _aggregate_metric(rows, "spectral_centroid_hz"),
            "mean_attack_to_sustain_ratio": _aggregate_metric(rows, "attack_to_sustain_ratio"),
            "mean_guitar_realism_sanity_score": _aggregate_metric(rows, "guitar_realism_sanity_score"),
            "mean_radiation_contribution_proxy": _aggregate_metric(rows, "radiation_contribution_proxy"),
            "mean_peak_dbfs": _aggregate_metric(rows, "peak_dbfs"),
            "all_clipping_avoided": all(
                bool(r.get("realism_metrics", {}).get("clipping_avoided", True)) for r in rows
            )
            if rows
            else True,
        }

    alpha_aggs = {k: v for k, v in aggregates.items() if k in V5_ALPHA_EXPERIMENTS}
    if not alpha_aggs:
        return {"aggregates": aggregates}

    def _pick_best(key: str, *, higher: bool) -> Optional[str]:
        candidates = {
            k: v[key]
            for k, v in alpha_aggs.items()
            if v.get(key) is not None
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get) if higher else min(candidates, key=candidates.get)

    def _composite_score(name: str) -> float:
        agg = alpha_aggs[name]
        body_aud = float(agg.get("mean_body_audibility_index") or 0.0)
        str_dom = float(agg.get("mean_string_dominance_ratio") or 1.0)
        metallic = float(agg.get("mean_metallicity_index") or 0.0)
        realism = float(agg.get("mean_guitar_realism_sanity_score") or 0.0)
        b2s = float(agg.get("mean_body_to_string_energy_ratio") or 0.0)
        return (
            2.5 * body_aud
            + 1.5 * realism
            + 1.0 * min(b2s, 10.0)
            - 2.0 * str_dom
            - 0.5 * metallic
        )

    recommended = max(alpha_aggs.keys(), key=_composite_score)

    baseline = aggregates.get(BASELINE_EXPERIMENT, {})
    ranking_vs_baseline: Dict[str, Any] = {}
    for alpha_name in V5_ALPHA_EXPERIMENTS:
        a = alpha_aggs.get(alpha_name, {})
        ranking_vs_baseline[alpha_name] = {
            "string_dominance_lower_than_final_v1": (
                (a.get("mean_string_dominance_ratio") or 1.0)
                < (baseline.get("mean_string_dominance_ratio") or 1.0)
            ),
            "body_to_string_higher_than_final_v1": (
                (a.get("mean_body_to_string_energy_ratio") or 0.0)
                > (baseline.get("mean_body_to_string_energy_ratio") or 0.0)
            ),
        }

    return {
        "aggregates": aggregates,
        "best_body_audibility": _pick_best("mean_body_audibility_index", higher=True),
        "lowest_string_dominance": _pick_best("mean_string_dominance_ratio", higher=False),
        "lowest_metallicity": _pick_best("mean_metallicity_index", higher=False),
        "best_realism_score": _pick_best("mean_guitar_realism_sanity_score", higher=True),
        "recommended_alpha_variant": recommended,
        "recommended_alpha_composite_score": round(_composite_score(recommended), 6),
        "alpha_vs_baseline_final_v1": ranking_vs_baseline,
    }


def _params_for_mode(
    *,
    base_params: Mapping[str, Any],
    mode: str,
    contrast_ctx: Optional[Mapping[str, Any]],
    sample_id: str,
) -> Dict[str, Any]:
    params = dict(normalize_sample_parameters(base_params))
    params["sample_id"] = sample_id
    if contrast_ctx and requires_identity_contrast_context(mode):
        params["identity_contrast_context"] = dict(contrast_ctx)
    return params


def build_stk_v5_diagnostic_audio(
    *,
    repo_root: Path,
    out_dir: Path,
    sample_id: str = "sample_000",
    notes: Sequence[Tuple[str, float]] = (),
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    use_surrogate: bool = True,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_lhs_sample_entries(repo_root, max_samples=26)
    sample = next((s for s in samples if str(s["sample_id"]) == sample_id), None)
    if sample is None and samples:
        sample = samples[0]
        sid = str(sample["sample_id"])
    elif sample is None:
        raise RuntimeError(f"no LHS sample {sample_id}")
    else:
        sid = str(sample["sample_id"])
    base_params = normalize_sample_parameters(sample.get("parameters"))
    modal_data, modal_source = resolve_modal_data_for_sample(
        repo_root, sample, use_surrogate=use_surrogate
    )

    note_rows: List[Dict[str, Any]] = []
    experiment_summaries: Dict[str, List[Dict[str, Any]]] = {e[0]: [] for e in EXPERIMENTS}

    for note_name, frequency_hz in notes:
        z_body = build_body_identity_vector(
            parameters=base_params,
            modal_data=modal_data,
            frequency_hz=frequency_hz,
            repo_root=repo_root,
            sample_id=sid,
        )
        contrast_ctx_note = build_batch_contrast_context({sid: z_body}).get(sid)
        singleton = singleton_dz_body_quantification(
            sample_parameters=base_params,
            modal_data=modal_data,
            frequency_hz=frequency_hz,
            repo_root=repo_root,
            sample_id=sid,
        )

        exp_results: List[Dict[str, Any]] = []
        for exp_name, mode in EXPERIMENTS:
            wav = out_dir / f"{exp_name}_{note_name}_{sid}.wav"
            t0 = time.perf_counter()
            params = _params_for_mode(
                base_params=base_params,
                mode=mode,
                contrast_ctx=contrast_ctx_note,
                sample_id=sid,
            )
            meta = synthesize_mode_to_wav(
                mode=mode or exp_name,
                frequency_hz=frequency_hz,
                note_name=note_name,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                output_wav=wav,
                sample_parameters=params,
                repo_root=repo_root,
                sample_id=sid,
                experiment=exp_name,
            )
            row = {
                "experiment": exp_name,
                "mode": mode or exp_name,
                "note_name": note_name,
                "frequency_hz": frequency_hz,
                "wav_path": str(wav.relative_to(repo_root)).replace("\\", "/"),
                "render_time_sec": round(time.perf_counter() - t0, 4),
                "realism_metrics": meta.get("realism_metrics"),
                "diagnostic_mode": meta.get("diagnostic_mode"),
            }
            exp_results.append(row)
            experiment_summaries[exp_name].append(row)

        note_rows.append(
            {
                "note_name": note_name,
                "frequency_hz": frequency_hz,
                "singleton_contrast_quantification": singleton,
                "experiments": exp_results,
            }
        )

    ranking = _build_ranking(experiment_summaries)

    report: Dict[str, Any] = {
        "report_version": "stk_v5_diagnostic_audio_v2",
        "timestamp_utc": _utc_now(),
        "sample_id": sid,
        "modal_source": modal_source,
        "notes": [n for n, _ in notes],
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "output_dir": str(out_dir.relative_to(repo_root)).replace("\\", "/"),
        "experiments": [e[0] for e in EXPERIMENTS],
        "v5_alpha_experiments": list(V5_ALPHA_EXPERIMENTS),
        "note_results": note_rows,
        "experiment_aggregate": ranking.get("aggregates", {}),
        "ranking": {
            k: ranking[k]
            for k in (
                "best_body_audibility",
                "lowest_string_dominance",
                "lowest_metallicity",
                "best_realism_score",
                "recommended_alpha_variant",
                "recommended_alpha_composite_score",
                "alpha_vs_baseline_final_v1",
            )
            if k in ranking
        },
    }

    report_path = repo_root / "audio" / "debug_reports" / "stk_v5_diagnostic_audio_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path.relative_to(repo_root)).replace("\\", "/")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="STK V5 diagnostic audio (lightweight)")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--notes", default=DEFAULT_NOTES)
    parser.add_argument("--duration-s", type=float, default=0.85, help="Shorter clips for diagnostics")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--no-surrogate", action="store_true")
    args = parser.parse_args()

    notes = parse_notes_arg(args.notes)
    report = build_stk_v5_diagnostic_audio(
        repo_root=args.repo_root,
        out_dir=args.out_dir,
        sample_id=args.sample_id,
        notes=notes,
        duration_s=args.duration_s,
        sample_rate=args.sample_rate,
        use_surrogate=not args.no_surrogate,
    )
    print(json.dumps({"status": "ok", "report_path": report["report_path"], "output_dir": report["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
