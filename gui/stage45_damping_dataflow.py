#!/usr/bin/env python3
"""
Stage 4.5 — Hard verification of damping/material data flow for diagnostic runs.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    modes_in_validated_band,
    parse_modal_modes,
    synthesize_note_with_body_response,
)
from build_sample_comparison import (  # noqa: E402
    load_lhs_sample_entries,
    resolve_modal_data_for_sample,
)
from modal_damping import compute_per_mode_damping, summarize_mode_damping_records  # noqa: E402
from sample_parameters import normalize_sample_parameters, sample_has_real_woods  # noqa: E402


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(statistics.median(vals))


def analyze_sample_synthesis(
    *,
    sample: Mapping[str, Any],
    repo_root: Path,
    use_surrogate: bool,
    diagnostic_mode: Optional[str] = None,
    note_hz: float = 110.0,
) -> Dict[str, Any]:
    params = normalize_sample_parameters(sample.get("parameters"))
    modal_data, modal_source = resolve_modal_data_for_sample(
        repo_root, sample, use_surrogate=use_surrogate
    )
    all_modes, _ = parse_modal_modes(modal_data)
    band = modes_in_validated_band(all_modes)
    freqs = [float(m["frequency_hz"]) for m in band]

    tmp = Path("_stage45_tmp.wav")
    meta = synthesize_note_with_body_response(
        frequency_hz=note_hz,
        note_name="A2",
        duration_s=0.12,
        sample_rate=DEFAULT_SAMPLE_RATE,
        modal_data=modal_data,
        output_wav=tmp,
        diagnostic_mode=diagnostic_mode,
        sample_parameters=params,
        modal_source=modal_source,
    )
    if tmp.is_file():
        tmp.unlink()

    dqs = meta.get("damping_q_summary") or {}
    per_mode = list(meta.get("per_mode_damping") or [])
    if not per_mode and meta.get("per_mode_damping_count", 0) > 0:
        per_mode = []

    return {
        "sample_id": sample.get("sample_id"),
        "run_id": sample.get("run_id"),
        "top_wood_id": params.get("top_wood_id"),
        "back_wood_id": params.get("back_wood_id"),
        "geometry": {k.replace("geometry.", ""): v for k, v in params.items() if str(k).startswith("geometry.")},
        "modal_source": modal_source,
        "modes_loaded": len(all_modes),
        "modes_in_band": len(band),
        "frequency_min_hz": min(freqs) if freqs else None,
        "frequency_max_hz": max(freqs) if freqs else None,
        "mode_q_min": dqs.get("mode_q_min"),
        "mode_q_median": dqs.get("mode_q_median"),
        "mode_q_max": dqs.get("mode_q_max"),
        "mode_q_spread_within_sample": dqs.get("mode_q_spread"),
        "mode_tau_s_median": dqs.get("mode_tau_s_median"),
        "mode_bandwidth_hz_median": dqs.get("mode_bandwidth_hz_median"),
        "material_damping_min": dqs.get("material_damping_min"),
        "material_damping_median": dqs.get("material_damping_median"),
        "sample_material_damping_fingerprint": meta.get("sample_material_damping_fingerprint"),
        "sample_mode_q_fingerprint": meta.get("sample_mode_q_fingerprint"),
        "material_damping_max": dqs.get("material_damping_max"),
        "material_damping_spread_within_sample": dqs.get("material_damping_spread"),
        "avg_top_share": dqs.get("avg_top_share"),
        "avg_back_share": dqs.get("avg_back_share"),
        "avg_air_share": dqs.get("avg_air_share"),
        "per_mode_q_used_in_frequency_response": bool(meta.get("per_mode_q_used_in_frequency_response")),
        "per_mode_tau_used_in_time_decay": bool(meta.get("per_mode_tau_used_in_time_decay")),
        "far_mode_weights_sample_specific": bool(meta.get("far_mode_weights_sample_specific")),
        "material_damping_components_nonzero": any(
            float(r.get("top_wood_damping_component") or 0) > 0 for r in per_mode[:5]
        )
        if per_mode
        else False,
    }


def evaluate_pass_fail(
    rows: Sequence[Mapping[str, Any]],
    *,
    use_surrogate: bool,
) -> Dict[str, str]:
    woods_ok = (
        all(r.get("top_wood_id") and r.get("back_wood_id") for r in rows)
        if rows
        else False
    )
    m4_used = all(r.get("modal_source") == "m4_surrogate" for r in rows) if use_surrogate else False
    q_medians = [float(r.get("mode_q_median") or 0) for r in rows if r.get("mode_q_median") is not None]
    q_fps = [float(r.get("sample_mode_q_fingerprint") or 0) for r in rows if r.get("sample_mode_q_fingerprint")]
    mat_medians = [float(r.get("material_damping_median") or 0) for r in rows if r.get("material_damping_median")]
    mat_fps = [float(r.get("sample_material_damping_fingerprint") or 0) for r in rows if r.get("sample_material_damping_fingerprint")]
    q_vals = q_fps or q_medians
    q_varies = len(set(round(q, 4) for q in q_vals)) > 1 if len(q_vals) >= 2 else False
    mat_varies = (
        len(set(round(m, 6) for m in (mat_fps or mat_medians))) > 1 if len(mat_fps or mat_medians) >= 2 else False
    )
    freq_ok = all(r.get("per_mode_q_used_in_frequency_response") for r in rows) if rows else False
    tau_ok = all(r.get("per_mode_tau_used_in_time_decay") for r in rows) if rows else False
    far_ok = all(r.get("far_mode_weights_sample_specific") for r in rows) if rows else False

    def pf(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    return {
        "real_lhs_sample_parameters_present": pf(bool(rows)),
        "real_top_back_woods_present": pf(woods_ok),
        "m4_surrogate_modes_used": pf(m4_used if use_surrogate else bool(rows)),
        "per_mode_q_varies_with_sample": pf(q_varies),
        "material_damping_varies_with_sample": pf(mat_varies),
        "per_mode_q_used_in_frequency_response": pf(freq_ok),
        "per_mode_tau_used_in_time_decay": pf(tau_ok),
        "far_mode_weights_vary_by_sample": pf(far_ok),
    }


def build_dataflow_report(
    *,
    repo_root: Path,
    samples: Sequence[Mapping[str, Any]],
    use_surrogate: bool = True,
    diagnostic_mode: Optional[str] = None,
) -> Dict[str, Any]:
    per_sample = [
        analyze_sample_synthesis(
            sample=s,
            repo_root=repo_root,
            use_surrogate=use_surrogate,
            diagnostic_mode=diagnostic_mode,
        )
        for s in samples
    ]
    q_medians = [float(s["mode_q_median"]) for s in per_sample if s.get("mode_q_median") is not None]
    mat_medians = [float(s["material_damping_median"]) for s in per_sample if s.get("material_damping_median")]
    mat_fps = [float(s.get("sample_material_damping_fingerprint") or 0) for s in per_sample if s.get("sample_material_damping_fingerprint")]
    cross_q = max(q_medians) - min(q_medians) if len(q_medians) >= 2 else 0.0
    cross_mat = max(mat_fps or mat_medians) - min(mat_fps or mat_medians) if len(mat_fps or mat_medians) >= 2 else 0.0

    from build_sample_comparison import m4_surrogate_model_available

    m4_ready = m4_surrogate_model_available(repo_root)
    return {
        "schema_version": "stage45_damping_dataflow_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "use_surrogate": use_surrogate,
        "m4_surrogate_model_available": m4_ready,
        "diagnostic_mode": diagnostic_mode or "production",
        "sample_count": len(per_sample),
        "per_sample": per_sample,
        "cross_sample_mode_q_median_spread": round(cross_q, 6),
        "cross_sample_material_damping_median_spread": round(cross_mat, 6),
        "pass_fail": evaluate_pass_fail(per_sample, use_surrogate=use_surrogate),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 4.5 Damping Dataflow Verification",
        "",
        f"Generated: {report.get('generated_at')}",
        f"Samples: {report.get('sample_count')}",
        f"M4 surrogate requested: {report.get('use_surrogate')}",
        f"M4 model files present: {report.get('m4_surrogate_model_available')}",
        "",
        "## PASS/FAIL",
        "",
    ]
    for key, val in (report.get("pass_fail") or {}).items():
        lines.append(f"- **{key}**: {val}")
    lines.extend(
        [
            "",
            f"Cross-sample mode_q median spread: **{report.get('cross_sample_mode_q_median_spread')}**",
            f"Cross-sample material_damping median spread: **{report.get('cross_sample_material_damping_median_spread')}**",
            "",
            "## Per sample",
            "",
        ]
    )
    for row in report.get("per_sample") or []:
        lines.append(
            f"- `{row.get('sample_id')}` woods={row.get('top_wood_id')}/{row.get('back_wood_id')} "
            f"source={row.get('modal_source')} Q_med={row.get('mode_q_median')} "
            f"mat_med={row.get('material_damping_median')}"
        )
    return "\n".join(lines)


def write_dataflow_reports(
    out_dir: Path,
    *,
    repo_root: Path,
    max_samples: int = 10,
    use_surrogate: bool = True,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_lhs_sample_entries(repo_root, max_samples=max_samples)
    if not samples:
        raise RuntimeError("no LHS samples found for dataflow report")
    report = build_dataflow_report(repo_root=repo_root, samples=samples, use_surrogate=use_surrogate)
    json_path = out_dir / "stage45_damping_dataflow_report.json"
    md_path = out_dir / "stage45_damping_dataflow_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def validate_diagnostic_evidence(segments: Sequence[Mapping[str, Any]], *, require_m4: bool = True) -> None:
    """Fail loudly if cross-sample damping evidence is missing."""
    if len(segments) < 2:
        return
    if require_m4:
        sources = {str(s.get("modal_source") or "") for s in segments}
        if sources & {"synthetic_fallback", ""}:
            raise RuntimeError(
                "diagnostic evidence invalid: synthetic modal fallback detected "
                f"(sources={sources}). Run without --no-surrogate."
            )
    q_meds = [float(s.get("mode_q_median") or 0) for s in segments]
    q_fps = [float(s.get("sample_mode_q_fingerprint") or 0) for s in segments if s.get("sample_mode_q_fingerprint")]
    mat_meds = [float(s.get("material_damping_median") or 0) for s in segments]
    mat_fps = [float(s.get("sample_material_damping_fingerprint") or 0) for s in segments if s.get("sample_material_damping_fingerprint")]
    q_spread = max(q_fps or q_meds) - min(q_fps or q_meds) if (q_fps or q_meds) else 0.0
    mat_spread = max(mat_fps or mat_meds) - min(mat_fps or mat_meds) if (mat_fps or mat_meds) else 0.0
    if q_spread <= 1e-6:
        raise RuntimeError(
            f"diagnostic evidence invalid: cross-sample mode Q spread is zero "
            f"(q_fingerprints={q_fps}, q_medians={q_meds})"
        )
    if mat_spread <= 1e-8:
        raise RuntimeError(
            f"diagnostic evidence invalid: cross-sample material damping spread is zero "
            f"(mat_fingerprints={mat_fps}, mat_medians={mat_meds})"
        )


def m4_available(repo_root: Path) -> bool:
    from build_sample_comparison import m4_surrogate_model_available

    return m4_surrogate_model_available(repo_root)


def main() -> int:
    paths = write_dataflow_reports(
        REPO / "audio" / "debug_reports",
        repo_root=REPO,
        max_samples=10,
        use_surrogate=True,
    )
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    doc = json.loads(paths["json"].read_text(encoding="utf-8"))
    print("PASS/FAIL:", doc.get("pass_fail"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
