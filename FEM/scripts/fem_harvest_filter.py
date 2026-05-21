"""
Post-solve harvest classification for coupled shift-invert FOM snapshots.

Distinguishes physical FSI modes (``physical_fsi``) from shift-locked Ritz artifacts
(``sigma_ritz``) using frequency proximity to ``st_sigma_hz``, not ``p_frac`` alone.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HarvestFilterConfig:
    """ROM-ready selection thresholds (override via JSON ``solver.harvest_filter``)."""

    min_hz: float = 90.0
    max_hz: float = 480.0
    min_p_frac_rom: float = 0.10
    min_wood: float = 0.01
    sigma_cluster_hz: float = 2.0
    min_sigma_sep_from_target_hz: float = 0.0
    keep_sigma_reference: bool = False
    max_sigma_reference_per_shift: int = 0
    near_dup_hz: float = 0.05
    min_uniqueness: float = 0.04

    @classmethod
    def from_solver_cfg(cls, solver_cfg: Optional[Dict[str, Any]]) -> "HarvestFilterConfig":
        raw = {}
        if isinstance(solver_cfg, dict):
            block = solver_cfg.get("harvest_filter")
            if isinstance(block, dict):
                raw = block
            else:
                raw = {
                    k: solver_cfg[k]
                    for k in (
                        "harvest_filter_min_hz",
                        "harvest_filter_max_hz",
                        "harvest_filter_min_p_frac",
                        "harvest_filter_sigma_cluster_hz",
                        "harvest_filter_min_uniqueness",
                    )
                    if k in solver_cfg
                }
        def _f(key: str, default: float) -> float:
            try:
                return float(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        def _b(key: str, default: bool) -> bool:
            v = raw.get(key, default)
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)

        def _i(key: str, default: int) -> int:
            try:
                return int(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            min_hz=_f("min_hz", cls.min_hz),
            max_hz=_f("max_hz", cls.max_hz),
            min_p_frac_rom=_f("min_p_frac_rom", _f("min_p_frac", cls.min_p_frac_rom)),
            min_wood=_f("min_wood", cls.min_wood),
            sigma_cluster_hz=_f("sigma_cluster_hz", cls.sigma_cluster_hz),
            min_sigma_sep_from_target_hz=_f(
                "min_sigma_sep_from_target_hz", cls.min_sigma_sep_from_target_hz
            ),
            keep_sigma_reference=_b("keep_sigma_reference", cls.keep_sigma_reference),
            max_sigma_reference_per_shift=_i(
                "max_sigma_reference_per_shift", cls.max_sigma_reference_per_shift
            ),
            near_dup_hz=_f("near_dup_hz", cls.near_dup_hz),
            min_uniqueness=_f("min_uniqueness", cls.min_uniqueness),
        )


def classify_mode_candidate(
    candidate: Dict[str, Any],
    *,
    target_hz: float,
    st_sigma_hz: float,
    cfg: HarvestFilterConfig,
) -> Tuple[str, bool, str]:
    """
    Classify one harvested mode.

    Returns ``(class_label, rom_ready, reason)``.
    """
    try:
        f_hz = float(candidate.get("hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        return "invalid", False, "non_finite_hz"
    try:
        p_frac = float(candidate.get("p_frac", 0.0) or 0.0)
    except (TypeError, ValueError):
        p_frac = 0.0
    try:
        wood = float(candidate.get("wood_participation", 0.0) or 0.0)
    except (TypeError, ValueError):
        wood = 0.0
    try:
        uniq = float(candidate.get("uniqueness", 1.0) or 1.0)
    except (TypeError, ValueError):
        uniq = 1.0

    if not math.isfinite(f_hz):
        return "invalid", False, "non_finite_hz"

    sigma = float(st_sigma_hz)
    target = float(target_hz)

    if abs(f_hz - sigma) <= float(cfg.sigma_cluster_hz):
        return "sigma_ritz", False, f"|f-st_sigma|={abs(f_hz - sigma):.3f}<={cfg.sigma_cluster_hz}"

    if f_hz + 1e-9 < float(cfg.min_hz) or f_hz > float(cfg.max_hz) + 1e-9:
        return "out_of_band", False, f"hz={f_hz:.2f} outside [{cfg.min_hz},{cfg.max_hz}]"

    if (
        float(cfg.min_sigma_sep_from_target_hz) > 0.0
        and abs(f_hz - target) < float(cfg.min_sigma_sep_from_target_hz)
        and abs(f_hz - sigma) > float(cfg.sigma_cluster_hz)
    ):
        pass

    if p_frac < 0.02:
        return "decoupled", False, f"p_frac={p_frac:.3e}<0.02"

    if wood < float(cfg.min_wood):
        return "low_wood", False, f"wood={wood:.4f}<{cfg.min_wood}"

    if p_frac < float(cfg.min_p_frac_rom):
        return "weak_coupling", False, f"p_frac={p_frac:.3f}<{cfg.min_p_frac_rom}"

    if uniq < float(cfg.min_uniqueness) - 1e-15:
        return "low_uniqueness", False, f"uniqueness={uniq:.3f}<{cfg.min_uniqueness}"

    return "physical_fsi", True, "coupled_fsi_ok"


def filter_result_payload(
    payload: Dict[str, Any],
    cfg: Optional[HarvestFilterConfig] = None,
) -> Dict[str, Any]:
    """
    Annotate and subset worker ``result_*.json`` payload with ROM-ready candidates.

    Adds per-candidate ``harvest_class``, ``rom_ready``, ``harvest_reason``.
    Top-level ``rom_ready_candidates`` lists ROM-ready rows (de-duplicated).
    """
    cfg = cfg or HarvestFilterConfig()
    try:
        target_hz = float(payload.get("target_hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        target_hz = 0.0
    try:
        st_sigma_hz = float(payload.get("st_sigma_hz", target_hz) or target_hz)
    except (TypeError, ValueError):
        st_sigma_hz = target_hz

    raw = list(payload.get("candidates") or [])
    annotated: List[Dict[str, Any]] = []
    sigma_refs_kept = 0

    for c in raw:
        row = dict(c)
        label, rom_ready, reason = classify_mode_candidate(
            row,
            target_hz=target_hz,
            st_sigma_hz=st_sigma_hz,
            cfg=cfg,
        )
        if (
            label == "sigma_ritz"
            and cfg.keep_sigma_reference
            and sigma_refs_kept < int(cfg.max_sigma_reference_per_shift)
        ):
            rom_ready = True
            reason = "sigma_reference_kept"
            sigma_refs_kept += 1
        row["harvest_class"] = label
        row["rom_ready"] = bool(rom_ready)
        row["harvest_reason"] = reason
        annotated.append(row)

    rom_ready = [r for r in annotated if r.get("rom_ready")]
    rom_ready.sort(
        key=lambda r: (
            -float(r.get("p_frac", 0.0) or 0.0),
            abs(float(r.get("hz", 0.0) or 0.0) - target_hz),
            -float(r.get("uniqueness", 0.0) or 0.0),
        )
    )

    deduped: List[Dict[str, Any]] = []
    for row in rom_ready:
        f_hz = float(row.get("hz", 0.0) or 0.0)
        if any(abs(f_hz - float(k.get("hz", 0.0))) < cfg.near_dup_hz for k in deduped):
            dup = next(
                k
                for k in deduped
                if abs(f_hz - float(k.get("hz", 0.0))) < cfg.near_dup_hz
            )
            if float(row.get("uniqueness", 0.0) or 0.0) > float(dup.get("uniqueness", 0.0) or 0.0):
                deduped.remove(dup)
                deduped.append(row)
            continue
        deduped.append(row)

    counts: Dict[str, int] = {}
    for r in annotated:
        k = str(r.get("harvest_class", "unknown"))
        counts[k] = counts.get(k, 0) + 1

    out = dict(payload)
    out["candidates"] = annotated
    out["rom_ready_candidates"] = deduped
    out["harvest_filter_summary"] = {
        "target_hz": target_hz,
        "st_sigma_hz": st_sigma_hz,
        "n_incoming": len(raw),
        "n_rom_ready": len(deduped),
        "class_counts": counts,
    }
    return out


def load_filter_config_from_json(config_path: Path) -> HarvestFilterConfig:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return HarvestFilterConfig()
    solver = data.get("solver") if isinstance(data.get("solver"), dict) else {}
    return HarvestFilterConfig.from_solver_cfg(solver)


def filter_result_file(
    result_path: Path,
    *,
    cfg: Optional[HarvestFilterConfig] = None,
    write_rom_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    filtered = filter_result_payload(payload, cfg=cfg)
    if write_rom_path is not None:
        write_rom_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = write_rom_path.with_suffix(write_rom_path.suffix + ".tmp")
        tmp.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
        tmp.replace(write_rom_path)
    return filtered
