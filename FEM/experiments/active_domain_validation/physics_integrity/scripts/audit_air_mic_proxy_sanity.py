#!/usr/bin/env python3
"""Audit air/mic_output_proxy clusters across M4 LHS samples (read-only, no FOM changes)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import DEFAULT_RUN_ID_SUFFIX, lhs_entry_index, load_lhs_pool  # noqa: E402
from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import load_fom_modes_catalog_deduped  # noqa: E402
from v2_b3_m4_rom_scalar_fields import ROM_DEDUPE_TOLERANCE_HZ  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel  # noqa: E402

DEFAULT_LHS = "ROM/classic/lhs_pool.json"
MIC_CLUSTER_LO = 275.0
MIC_CLUSTER_HI = 290.0
MIC_CLUSTER2_LO = 385.0
MIC_CLUSTER2_HI = 400.0
PROXY_MATCH_TOL = 1.0e-7
FREQ_MATCH_TOL = 0.02


def _parse_sample_ids(arg: str) -> List[str]:
    out: List[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and part.replace("-", "").isdigit():
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                out.append(f"sample_{i:03d}")
        elif part.isdigit():
            out.append(f"sample_{int(part):03d}")
        else:
            out.append(part)
    return out


def _estimate_cavity_volume_m3(geom: Mapping[str, float]) -> Optional[float]:
    """Rough box cavity estimate: L×W×(D−2×top_thickness). Not CAD-exact."""
    try:
        L = float(geom["length"])
        W = float(geom["width"])
        D = float(geom["depth"])
        t = float(geom.get("top_thickness") or 0.003)
        inner = max(D - 2.0 * t, 1.0e-6)
        return L * W * inner
    except (KeyError, TypeError, ValueError):
        return None


def _top_air_mic_modes(
    modes: Sequence[Mapping[str, Any]],
    *,
    band_lo: float,
    band_hi: float,
    n: int = 3,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in modes:
        f = m.get("frequency_hz")
        mic = m.get("mic_output_proxy")
        if f is None or mic is None:
            continue
        try:
            f_hz = float(f)
            mic_v = float(mic)
        except (TypeError, ValueError):
            continue
        if not (band_lo <= f_hz <= band_hi):
            continue
        air = m.get("air_share")
        try:
            air_v = float(air) if air is not None else 0.0
        except (TypeError, ValueError):
            air_v = 0.0
        rows.append(
            {
                "frequency_hz": f_hz,
                "mic_output_proxy": mic_v,
                "radiation_proxy": m.get("radiation_proxy"),
                "coupling_class": m.get("coupling_class"),
                "dominant_region": m.get("dominant_region"),
                "top_share": m.get("top_share"),
                "back_share": m.get("back_share"),
                "air_share": air_v,
                "chunk_id": m.get("chunk_id"),
                "target_hz": m.get("target_hz"),
                "source": m.get("source"),
                "mic_output_method": m.get("mic_output_method"),
                "provenance_count": m.get("provenance_count"),
            }
        )
    rows.sort(key=lambda r: (-float(r["mic_output_proxy"]), float(r["frequency_hz"])))
    return rows[:n]


def _duplicate_groups(modes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, float], List[Dict[str, Any]]] = defaultdict(list)
    for m in modes:
        f = m.get("frequency_hz")
        mic = m.get("mic_output_proxy")
        if f is None or mic is None:
            continue
        try:
            key = (round(float(f), 6), round(float(mic), 10))
        except (TypeError, ValueError):
            continue
        groups[key].append(dict(m))
    out = []
    for (f, mic), items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0])):
        if len(items) < 2:
            continue
        out.append(
            {
                "frequency_hz": f,
                "mic_output_proxy": mic,
                "duplicate_count": len(items),
                "chunk_ids": sorted({str(x.get("chunk_id") or "") for x in items if x.get("chunk_id")}),
            }
        )
    return out


def _read_checkpoint_mesh_hint(run_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    built = run_root / "lprod" / "checkpoint" / "built_metadata.json"
    if built.is_file():
        try:
            meta = load_json(built)
            out["built_mesh_level"] = meta.get("mesh_level")
            out["region_dof_mesh_file"] = meta.get("region_dof_mesh_file")
            out["n_u_b3"] = meta.get("n_u_b3")
            out["n_p"] = meta.get("n_p") or (len(meta.get("p_idx") or []))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    mesh_summary = run_root / "lprod" / "mesh" / "L_prod"
    if mesh_summary.is_dir():
        for p in mesh_summary.glob("*_mesh_build_summary.json"):
            try:
                s = load_json(p)
                out["mesh_build_geometry"] = s.get("geometry")
                out["mesh_n_nodes"] = s.get("n_nodes")
                out["mesh_n_tetra"] = s.get("n_tetrahedra")
                out["mesh_path"] = s.get("mesh_path")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            break
    return out


def audit_sample(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
) -> Dict[str, Any]:
    idx = lhs_entry_index(pool, sample_id)
    entry = (pool.get("entries") or [])[idx] if idx is not None else {}
    geom = extract_geometry_dict(entry)
    run_id = str(entry.get("last_run_id") or f"{sample_id}_{run_id_suffix}")
    run_root = (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
        / sample_id
        / "runs"
        / run_id
    )
    catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "hole_radius": geom.get("hole_radius"),
        "depth": geom.get("depth"),
        "length": geom.get("length"),
        "width": geom.get("width"),
        "geometry_fingerprint": geometry_fingerprint(geom) if geom else None,
        "est_cavity_volume_m3": _estimate_cavity_volume_m3(geom),
        "catalog_path": rel(catalog_path, repo_root=repo_root) if catalog_path.is_file() else None,
        "catalog_status": "missing",
    }
    if not catalog_path.is_file():
        return row

    raw_modes, deduped, dedupe_meta = load_fom_modes_catalog_deduped(catalog_path)
    merge_groups = int(dedupe_meta.get("dedupe_merge_groups") or 0)
    row["raw_mode_count"] = len(raw_modes)
    row["deduped_mode_count"] = len(deduped)
    row["dedupe_merge_groups"] = len(merge_groups)
    row["catalog_status"] = "ok"

    peak_281 = _top_air_mic_modes(raw_modes, band_lo=MIC_CLUSTER_LO, band_hi=MIC_CLUSTER_HI, n=1)
    peak_390 = _top_air_mic_modes(raw_modes, band_lo=MIC_CLUSTER2_LO, band_hi=MIC_CLUSTER2_HI, n=1)
    row["peak_281hz_raw"] = peak_281[0] if peak_281 else None
    row["peak_390hz_raw"] = peak_390[0] if peak_390 else None
    peak_281_d = _top_air_mic_modes(deduped, band_lo=MIC_CLUSTER_LO, band_hi=MIC_CLUSTER_HI, n=1)
    row["peak_281hz_deduped"] = peak_281_d[0] if peak_281_d else None

    dups = _duplicate_groups(raw_modes)
    row["exact_duplicate_groups"] = len(dups)
    row["largest_duplicate_group"] = dups[0]["duplicate_count"] if dups else 0
    row["in_band_281_duplicate_groups"] = sum(
        1 for g in dups if MIC_CLUSTER_LO <= g["frequency_hz"] <= MIC_CLUSTER_HI
    )

    ckpt = _read_checkpoint_mesh_hint(run_root)
    row.update({f"ckpt_{k}": v for k, v in ckpt.items()})

    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if agg_path.is_file():
        try:
            agg = load_json(agg_path)
            row["agg_raw_mode_count"] = agg.get("raw_mode_count")
            row["agg_deduped_mode_count"] = agg.get("deduped_mode_count")
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return row


def _cross_sample_flags(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    peaks_281: List[Tuple[str, float, float]] = []
    peaks_390: List[Tuple[str, float, float]] = []
    fps: List[str] = []
    for r in rows:
        if r.get("geometry_fingerprint"):
            fps.append(str(r["geometry_fingerprint"]))
        p = r.get("peak_281hz_raw") or {}
        if p.get("frequency_hz") is not None and p.get("mic_output_proxy") is not None:
            peaks_281.append((str(r["sample_id"]), float(p["frequency_hz"]), float(p["mic_output_proxy"])))
        p390 = r.get("peak_390hz_raw") or {}
        if p390.get("frequency_hz") is not None and p390.get("mic_output_proxy") is not None:
            peaks_390.append((str(r["sample_id"]), float(p390["frequency_hz"]), float(p390["mic_output_proxy"])))

    def _cluster_stats(peaks: List[Tuple[str, float, float]]) -> Dict[str, Any]:
        if not peaks:
            return {}
        freqs = [p[1] for p in peaks]
        mics = [p[2] for p in peaks]
        freq_span = max(freqs) - min(freqs)
        mic_span = max(mics) - min(mics)
        mic_mean = sum(mics) / len(mics)
        mic_rel_span = mic_span / mic_mean if mic_mean > 0 else None
        exact_mic = len({round(m, 10) for m in mics}) == 1
        return {
            "sample_count": len(peaks),
            "freq_min_hz": min(freqs),
            "freq_max_hz": max(freqs),
            "freq_span_hz": freq_span,
            "mic_min": min(mics),
            "mic_max": max(mics),
            "mic_span": mic_span,
            "mic_rel_span": mic_rel_span,
            "all_mic_identical_10dp": exact_mic,
            "samples": [{"sample_id": s, "frequency_hz": f, "mic_output_proxy": m} for s, f, m in peaks],
        }

    return {
        "unique_geometry_fingerprints": len(set(fps)),
        "sample_count_with_geometry": len(fps),
        "cluster_281hz": _cluster_stats(peaks_281),
        "cluster_390hz": _cluster_stats(peaks_390),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit air/mic_output_proxy clusters (read-only).")
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument("--samples", default="0-22", help="e.g. 0-22 or sample_018,sample_019")
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX)
    parser.add_argument("--csv-out", type=Path, default=None, help="Write per-sample CSV report.")
    parser.add_argument("--json-out", type=Path, default=None, help="Write full JSON report.")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    pool = load_lhs_pool(lhs_path)
    sample_ids = _parse_sample_ids(str(args.samples))

    rows: List[Dict[str, Any]] = []
    for sid in sample_ids:
        rows.append(
            audit_sample(
                repo_root=repo_root,
                pool=pool,
                sample_id=sid,
                run_id_suffix=str(args.run_id_suffix),
            )
        )

    summary = _cross_sample_flags(rows)
    report = {
        "schema": "m4_air_mic_proxy_sanity_audit_v1",
        "lhs_json": rel(lhs_path, repo_root=repo_root),
        "sample_ids": sample_ids,
        "dedupe_tolerance_hz": ROM_DEDUPE_TOLERANCE_HZ,
        "per_sample": rows,
        "cross_sample": summary,
    }

    print(f"audited_samples={len(rows)}")
    print(f"unique_geometry_fingerprints={summary.get('unique_geometry_fingerprints')}")
    c281 = summary.get("cluster_281hz") or {}
    if c281:
        print(
            f"cluster_281hz: n={c281.get('sample_count')} "
            f"freq_span={c281.get('freq_span_hz'):.6f} Hz "
            f"mic_span={c281.get('mic_span'):.6e} "
            f"mic_rel_span={c281.get('mic_rel_span'):.6e} "
            f"all_mic_identical={c281.get('all_mic_identical_10dp')}"
        )
    c390 = summary.get("cluster_390hz") or {}
    if c390:
        print(
            f"cluster_390hz: n={c390.get('sample_count')} "
            f"freq_span={c390.get('freq_span_hz'):.6f} Hz "
            f"mic_span={c390.get('mic_span'):.6e}"
        )

    for r in rows:
        if r.get("catalog_status") != "ok":
            print(f"  {r['sample_id']}: catalog missing")
            continue
        p = r.get("peak_281hz_raw") or {}
        print(
            f"  {r['sample_id']}: hole_r={r.get('hole_radius')} depth={r.get('depth')} "
            f"raw={r.get('raw_mode_count')} dedup={r.get('deduped_mode_count')} "
            f"dups={r.get('exact_duplicate_groups')} "
            f"f281={p.get('frequency_hz')} mic={p.get('mic_output_proxy')} "
            f"air={p.get('air_share')} class={p.get('coupling_class')} "
            f"chunk={p.get('chunk_id')}"
        )

    if args.json_out:
        out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"json_out={rel(out, repo_root=repo_root)}")

    if args.csv_out:
        out = args.csv_out if args.csv_out.is_absolute() else repo_root / args.csv_out
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "sample_id",
            "hole_radius",
            "depth",
            "length",
            "width",
            "est_cavity_volume_m3",
            "geometry_fingerprint",
            "raw_mode_count",
            "deduped_mode_count",
            "exact_duplicate_groups",
            "peak_281_freq",
            "peak_281_mic",
            "peak_281_air_share",
            "peak_281_coupling_class",
            "peak_281_chunk_id",
            "peak_281_mic_method",
            "peak_390_freq",
            "peak_390_mic",
            "ckpt_region_dof_mesh_file",
            "ckpt_mesh_n_nodes",
        ]
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                p = r.get("peak_281hz_raw") or {}
                p390 = r.get("peak_390hz_raw") or {}
                w.writerow(
                    {
                        "sample_id": r.get("sample_id"),
                        "hole_radius": r.get("hole_radius"),
                        "depth": r.get("depth"),
                        "length": r.get("length"),
                        "width": r.get("width"),
                        "est_cavity_volume_m3": r.get("est_cavity_volume_m3"),
                        "geometry_fingerprint": r.get("geometry_fingerprint"),
                        "raw_mode_count": r.get("raw_mode_count"),
                        "deduped_mode_count": r.get("deduped_mode_count"),
                        "exact_duplicate_groups": r.get("exact_duplicate_groups"),
                        "peak_281_freq": p.get("frequency_hz"),
                        "peak_281_mic": p.get("mic_output_proxy"),
                        "peak_281_air_share": p.get("air_share"),
                        "peak_281_coupling_class": p.get("coupling_class"),
                        "peak_281_chunk_id": p.get("chunk_id"),
                        "peak_281_mic_method": p.get("mic_output_method"),
                        "peak_390_freq": p390.get("frequency_hz"),
                        "peak_390_mic": p390.get("mic_output_proxy"),
                        "ckpt_region_dof_mesh_file": r.get("ckpt_region_dof_mesh_file"),
                        "ckpt_mesh_n_nodes": r.get("ckpt_mesh_n_nodes"),
                    }
                )
        print(f"csv_out={rel(out, repo_root=repo_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
