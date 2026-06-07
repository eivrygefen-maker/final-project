#!/usr/bin/env python3
"""30-sample LOO comparison: v2.1 (A) vs v2.2 experimental methods B/C/D."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import load_lhs_pool  # noqa: E402
from v2_b3_m4_modal_surrogate_lib import (  # noqa: E402
    DEFAULT_K_NEIGHBORS,
    build_holdout_surrogate_model,
    guitars_root,
)
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    ACCURACY_BAND_HZ,
    greedy_nearest_hz_match,
    load_fom_modes_catalog_deduped,
    prepare_fom_modes_for_rom_compare,
    resolve_sample_context,
    resolve_validation_metadata,
)
from v2_b3_m4_rom_intensity_v22 import (  # noqa: E402
    INTENSITY_METHOD_V21_A,
    INTENSITY_METHOD_V22_B,
    INTENSITY_METHOD_V22_C,
    INTENSITY_METHOD_V22_D,
    MODEL_VERSION_V2_2,
    PREDICTION_METHOD_V2_2,
    band_label,
    compute_intensity_metrics_v22,
    predict_intensity_catalog_v22,
    predict_v21_baseline,
)
from v2_b3_m4_rom_scalar_fields import enrich_match_with_phase2_fields  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

METHODS = (
    INTENSITY_METHOD_V21_A,
    INTENSITY_METHOD_V22_B,
    INTENSITY_METHOD_V22_C,
    INTENSITY_METHOD_V22_D,
)
METHOD_LABELS = {
    INTENSITY_METHOD_V21_A: "v2.1_A",
    INTENSITY_METHOD_V22_B: "v2.2_B",
    INTENSITY_METHOD_V22_C: "v2.2_C",
    INTENSITY_METHOD_V22_D: "v2.2_D",
}


def _parse_sample_range(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            out.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _sample_id(i: int) -> str:
    return f"sample_{i:03d}"


def experimental_v22_dir(repo_root: Path, shape_name: str) -> Path:
    return repo_root / "ROM" / shape_name / "experimental_v22"


def _median(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.median(vals), 6) if vals else None


def _mean(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.mean(vals), 6) if vals else None


def _assert_no_leakage(
    *,
    target_sample_id: str,
    training_sample_ids: Sequence[str],
    neighbor_sample_ids: Sequence[str],
) -> None:
    tid = str(target_sample_id)
    if tid in training_sample_ids:
        raise AssertionError(f"leakage: target {tid} in training_sample_ids")
    if tid in neighbor_sample_ids:
        raise AssertionError(f"leakage: target {tid} in neighbor_sample_ids")


def _evaluate_prediction(
    *,
    prediction: Mapping[str, Any],
    fom_modes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rom_freqs = [float(f) for f in (prediction.get("frequencies_hz") or [])]
    rom_modes = list(prediction.get("predicted_modes") or [])
    matches, _ = greedy_nearest_hz_match(rom_frequencies_hz=rom_freqs, fom_modes=fom_modes)
    enriched = []
    for m in matches:
        rom_hz = float(m.get("rom_frequency_hz") or 0.0)
        fom_hz = float(m.get("fom_frequency_hz") or 0.0)
        rom_mode = min(rom_modes, key=lambda r: abs(float(r.get("frequency_hz") or 0.0) - rom_hz), default={})
        fom_mode = min(fom_modes, key=lambda r: abs(float(r.get("frequency_hz") or 0.0) - fom_hz), default={})
        enriched.append(enrich_match_with_phase2_fields(m, rom_mode=dict(rom_mode), fom_mode=dict(fom_mode)))
    freq_rel = [
        float(x["relative_error"])
        for x in enriched
        if x.get("relative_error") is not None and x["relative_error"] == x["relative_error"]
    ]
    metrics = compute_intensity_metrics_v22(enriched)
    p2 = metrics.get("phase2_scalar_metrics") or {}
    breakdown: Dict[str, Any] = {"by_coupling_class": {}, "by_band": {}}
    for cc in sorted({str(m.get("fom_coupling_class") or "") for m in enriched}):
        sub = [m for m in enriched if str(m.get("fom_coupling_class") or "") == cc]
        if sub:
            sm = compute_intensity_metrics_v22(sub).get("phase2_scalar_metrics") or {}
            breakdown["by_coupling_class"][cc] = {
                "count": len(sub),
                "mic_p95_norm_mae": sm.get("mic_output_proxy_p95_norm_mae"),
                "radiation_p95_norm_mae": sm.get("radiation_proxy_p95_norm_mae"),
            }
    for m in enriched:
        m["_band"] = band_label(float(m.get("fom_frequency_hz") or 0.0))
    for b in sorted({str(m.get("_band") or "") for m in enriched}):
        sub = [m for m in enriched if m.get("_band") == b]
        if sub:
            sm = compute_intensity_metrics_v22(sub).get("phase2_scalar_metrics") or {}
            breakdown["by_band"][b] = {
                "count": len(sub),
                "mic_p95_norm_mae": sm.get("mic_output_proxy_p95_norm_mae"),
                "radiation_p95_norm_mae": sm.get("radiation_proxy_p95_norm_mae"),
            }
    class_match = sum(
        1 for m in enriched if str(m.get("rom_coupling_class")) == str(m.get("fom_coupling_class"))
    )
    region_match = sum(
        1 for m in enriched if str(m.get("rom_dominant_region")) == str(m.get("fom_dominant_region"))
    )
    return {
        "matched_mode_count": len(enriched),
        "frequency_median_relative_error": _median(freq_rel),
        "phase2_scalar_metrics": p2,
        "breakdown": breakdown,
        "class_consistent_match_rate": round(class_match / len(enriched), 4) if enriched else None,
        "region_consistent_match_rate": round(region_match / len(enriched), 4) if enriched else None,
        "per_mode_matches": enriched,
    }


def _predict_for_method(
    *,
    method: str,
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    target_sample_id: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if method == INTENSITY_METHOD_V21_A:
        pred = predict_v21_baseline(model, parameters, nev=0)
    else:
        pred = predict_intensity_catalog_v22(
            model,
            parameters,
            intensity_method=method,
            training_rows=training_rows if method == INTENSITY_METHOD_V22_D else None,
            nev=0,
            excluded_sample_ids=[target_sample_id],
        )
    pred["runtime_s"] = round(time.perf_counter() - t0, 4)
    return pred


def run_loo_comparison(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_ids: Sequence[str],
    k_neighbors: int,
) -> Dict[str, Any]:
    shape = str(pool.get("shape_name") or "classic")
    per_method_samples: Dict[str, List[Dict[str, Any]]] = {m: [] for m in METHODS}
    per_sample_all: Dict[str, Dict[str, Any]] = {}

    for sid in sample_ids:
        train_ids = [s for s in sample_ids if s != sid]
        model, training_rows = build_holdout_surrogate_model(
            repo_root=repo_root,
            pool=pool,
            shape_name=shape,
            exclude_sample_ids=[sid],
            k_neighbors=k_neighbors,
        )
        train_sids = [str(r["sample_id"]) for r in training_rows]
        _assert_no_leakage(
            target_sample_id=sid,
            training_sample_ids=train_sids,
            neighbor_sample_ids=[],
        )
        entry = next(e for e in pool.get("entries") or [] if str(e.get("id")) == sid)
        run_id = str(entry.get("last_run_id") or f"{sid}_m4prod1")
        context = resolve_sample_context(
            pool=pool, sample_id=sid, run_id=run_id, repo_root=repo_root
        )
        catalog_path = Path(context["run_root"]) / "aggregation" / "modes_catalog.jsonl"
        _raw, deduped, _meta = load_fom_modes_catalog_deduped(catalog_path)
        fom_modes, _ = prepare_fom_modes_for_rom_compare(deduped, band=ACCURACY_BAND_HZ)
        vmeta = resolve_validation_metadata(
            target_sample_id=sid,
            training_sample_ids=train_sids,
            excluded_sample_ids=[sid],
            validation_mode="holdout",
        )

        sample_result: Dict[str, Any] = {
            "sample_id": sid,
            "validation": vmeta,
            "training_sample_count": len(train_sids),
        }
        for method in METHODS:
            pred = _predict_for_method(
                method=method,
                model=model,
                parameters=context["parameters"],
                training_rows=training_rows,
                target_sample_id=sid,
            )
            _assert_no_leakage(
                target_sample_id=sid,
                training_sample_ids=train_sids,
                neighbor_sample_ids=list(pred.get("neighbor_sample_ids") or []),
            )
            ev = _evaluate_prediction(prediction=pred, fom_modes=fom_modes)
            row = {
                "sample_id": sid,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "runtime_s": pred.get("runtime_s"),
                "fallback_rate": pred.get("fallback_rate"),
                "frequency_median_relative_error": ev["frequency_median_relative_error"],
                "phase2_scalar_metrics": ev["phase2_scalar_metrics"],
                "class_consistent_match_rate": ev["class_consistent_match_rate"],
                "region_consistent_match_rate": ev["region_consistent_match_rate"],
            }
            per_method_samples[method].append(row)
            sample_result[METHOD_LABELS[method]] = {
                "metrics": ev["phase2_scalar_metrics"],
                "breakdown": ev["breakdown"],
                "runtime_s": pred.get("runtime_s"),
                "neighbor_sample_ids": pred.get("neighbor_sample_ids"),
                "fallback_rate": pred.get("fallback_rate"),
            }
        per_sample_all[sid] = sample_result

    aggregate: Dict[str, Any] = {}
    for method in METHODS:
        rows = per_method_samples[method]
        p2s = [r["phase2_scalar_metrics"] for r in rows]
        aggregate[METHOD_LABELS[method]] = {
            "sample_count": len(rows),
            "frequency_median_relative_error_median": _median(
                [float(r["frequency_median_relative_error"]) for r in rows if r.get("frequency_median_relative_error") is not None]
            ),
            "mic_p95_norm_mae_median": _median(
                [float(p.get("mic_output_proxy_p95_norm_mae")) for p in p2s if p.get("mic_output_proxy_p95_norm_mae") is not None]
            ),
            "radiation_p95_norm_mae_median": _median(
                [float(p.get("radiation_proxy_p95_norm_mae")) for p in p2s if p.get("radiation_proxy_p95_norm_mae") is not None]
            ),
            "bridge_p95_norm_mae_median": _median(
                [float(p.get("bridge_excitation_abs_p95_norm_mae")) for p in p2s if p.get("bridge_excitation_abs_p95_norm_mae") is not None]
            ),
            "mic_rank_median": _median(
                [float(p.get("mic_output_proxy_rank_correlation")) for p in p2s if p.get("mic_output_proxy_rank_correlation") is not None]
            ),
            "radiation_rank_median": _median(
                [float(p.get("radiation_proxy_rank_correlation")) for p in p2s if p.get("radiation_proxy_rank_correlation") is not None]
            ),
            "top20_mic_overlap_median": _median(
                [float(p.get("mic_output_proxy_top_20pct_overlap")) for p in p2s if p.get("mic_output_proxy_top_20pct_overlap") is not None]
            ),
            "top20_radiation_overlap_median": _median(
                [float(p.get("radiation_proxy_top_20pct_overlap")) for p in p2s if p.get("radiation_proxy_top_20pct_overlap") is not None]
            ),
            "runtime_median_s": _median([float(r["runtime_s"]) for r in rows if r.get("runtime_s") is not None]),
            "fallback_rate_mean": _mean([float(r["fallback_rate"]) for r in rows if r.get("fallback_rate") is not None]),
            "class_match_rate_mean": _mean(
                [float(r["class_consistent_match_rate"]) for r in rows if r.get("class_consistent_match_rate") is not None]
            ),
            "region_match_rate_mean": _mean(
                [float(r["region_consistent_match_rate"]) for r in rows if r.get("region_consistent_match_rate") is not None]
            ),
        }

    return {
        "schema": "m4_rom_intensity_v21_v22_comparison_v1",
        "generated_utc": utc_now(),
        "model_version_v22": MODEL_VERSION_V2_2,
        "prediction_method_v22": PREDICTION_METHOD_V2_2,
        "sample_ids": list(sample_ids),
        "sample_count": len(sample_ids),
        "validation_mode": "holdout",
        "training_includes_target": False,
        "accuracy_meaningful": True,
        "leakage_count": 0,
        "aggregate_by_method": aggregate,
        "per_sample": per_sample_all,
    }


def _write_comparison_markdown(
    path: Path,
    report: Mapping[str, Any],
    *,
    repo_root: Path,
    json_out: Path,
) -> None:
    agg = report.get("aggregate_by_method") or {}
    a = agg.get("v2.1_A") or {}
    b = agg.get("v2.2_B") or {}
    c = agg.get("v2.2_C") or {}
    d = agg.get("v2.2_D") or {}

    def row(metric: str, key: str, fmt: str = ".4f", lower_better: bool = True) -> str:
        vals = {
            "v2.1": a.get(key),
            "B": b.get(key),
            "C": c.get(key),
            "D": d.get(key),
        }
        numeric = {k: v for k, v in vals.items() if v is not None}
        if not numeric:
            best = "n/a"
        else:
            best = min(numeric, key=lambda k: numeric[k]) if lower_better else max(numeric, key=lambda k: numeric[k])
        v21 = vals["v2.1"]
        def fmtv(v: Any) -> str:
            if v is None:
                return "n/a"
            if fmt == "pct":
                return f"{100*float(v):.2f}%"
            return f"{float(v):{fmt}}"
        imp = ""
        if v21 is not None and vals.get(best) is not None and best != "v2.1":
            delta = float(vals[best]) - float(v21)
            pct = 100.0 * delta / float(v21) if float(v21) != 0 else 0.0
            imp = f"{delta:+.4f} ({pct:+.1f}%)"
        return (
            f"| {metric} | {fmtv(v21)} | {fmtv(vals['B'])} | {fmtv(vals['C'])} | {fmtv(vals['D'])} | "
            f"**{best}** | {imp} |"
        )

    lines = [
        "# M4 ROM Intensity v2.1 vs v2.2 Comparison",
        "",
        f"**Generated:** {report.get('generated_utc')}",
        f"**Samples:** {report.get('sample_count')} LOO holdouts (`training_includes_target=false`)",
        "",
        "## Aggregate comparison",
        "",
        "| Metric | v2.1 | v2.2 B | v2.2 C | v2.2 D | Best | Δ vs v2.1 |",
        "|--------|------|--------|--------|--------|------|-----------|",
        row("Frequency median rel. error", "frequency_median_relative_error_median", ".4f"),
        row("Mic p95 norm MAE", "mic_p95_norm_mae_median"),
        row("Radiation p95 norm MAE", "radiation_p95_norm_mae_median"),
        row("Bridge p95 norm MAE", "bridge_p95_norm_mae_median"),
        row("Mic rank correlation", "mic_rank_median", ".4f", lower_better=False),
        row("Radiation rank correlation", "radiation_rank_median", ".4f", lower_better=False),
        row("Top-20% mic overlap", "top20_mic_overlap_median", ".4f", lower_better=False),
        row("Top-20% radiation overlap", "top20_radiation_overlap_median", ".4f", lower_better=False),
        row("Runtime median (s)", "runtime_median_s", ".2f"),
        row("Fallback rate mean", "fallback_rate_mean", ".4f"),
        row("Class match rate", "class_match_rate_mean", ".4f", lower_better=False),
        row("Region match rate", "region_match_rate_mean", ".4f", lower_better=False),
        "",
        "## Alignment 1.0 explanation",
        "",
        "Per-sample `overall_class_match_rate = 1.0` in diagnostics is **not self-comparison**.",
        "Holdout LOO excludes the target from training and neighbor pools. A rate of 1.0 means",
        "every FOM mode's Hz-nearest neighbor mode (from the 29 training guitars) had the same",
        "`coupling_class` — common for `top_back_mixed` guitars where most modes share one class.",
        "The compare script asserts `target not in neighbor_sample_ids`.",
        "",
        "## Recommendation",
        "",
        "See aggregate table. Promote v2.2 only if frequency stable and intensity metrics improve materially.",
        "",
        f"Full JSON: `{rel(json_out, repo_root=repo_root)}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs-json", type=Path, default=Path("ROM/classic/lhs_pool.json"))
    parser.add_argument("--samples", default="0-29")
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json.resolve() if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    sample_ids = [_sample_id(i) for i in _parse_sample_range(args.samples)]
    shape = str(pool.get("shape_name") or "classic")

    out_dir = experimental_v22_dir(repo_root, shape)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.json_out or out_dir / "comparison_summary.json"
    md_out = args.md_out or (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V21_V22_COMPARISON.md"
    )

    print(f"loo_samples={len(sample_ids)} methods={len(METHODS)}", flush=True)
    report = run_loo_comparison(
        repo_root=repo_root,
        pool=pool,
        sample_ids=sample_ids,
        k_neighbors=int(args.k_neighbors),
    )

    write_json_atomic(json_out, report)
    per_dir = out_dir / "per_sample_comparisons"
    per_dir.mkdir(parents=True, exist_ok=True)
    for sid, body in (report.get("per_sample") or {}).items():
        write_json_atomic(per_dir / f"{sid}_v21_v22.json", body)

    manifest = {
        "schema": "m4_rom_experimental_v22_manifest_v1",
        "generated_utc": utc_now(),
        "model_version": MODEL_VERSION_V2_2,
        "prediction_method": PREDICTION_METHOD_V2_2,
        "production_model_preserved": True,
        "comparison_summary": rel(json_out, repo_root=repo_root),
        "per_sample_dir": rel(per_dir, repo_root=repo_root),
    }
    write_json_atomic(out_dir / "model_manifest.json", manifest)
    _write_comparison_markdown(md_out, report, repo_root=repo_root, json_out=json_out)

    print(f"wrote {rel(json_out, repo_root=repo_root)}")
    print(f"wrote {rel(md_out, repo_root=repo_root)}")
    print(f"wrote {rel(out_dir / 'model_manifest.json', repo_root=repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
