"""
Post-solve harvest classification for coupled shift-invert FOM snapshots.

Distinguishes physical FSI modes (``physical_fsi``) from shift-locked Ritz artifacts
(``sigma_ritz``) using frequency proximity to ``st_sigma_hz``, not ``p_frac`` alone.

**Staged filtering** (``staged_filtering: true``): below ``staged_crossover_hz`` use
``min_p_frac_rom_low`` (default 0.02); at/above crossover use ``min_p_frac_rom_high``.
Wood-dominated body modes below ``wood_bypass_max_hz`` may pass when ``max|p|`` exceeds
``min_p_block_max_wood_bypass`` even if ``p_frac`` is far below the ROM gate. σ-cluster
width can tighten above ``staged_sigma_tighten_hz`` (default 400 Hz).

Policy version ``HARVEST_FILTER_POLICY_VERSION`` is stamped in ``candidates_log.json``
``pipeline_meta`` and per-mode rows.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HARVEST_FILTER_POLICY_VERSION = "staged_v2_unified_p02_wood_bypass"


@dataclass
class HarvestFilterConfig:
    """ROM-ready selection thresholds (override via JSON ``solver.harvest_filter``)."""

    min_hz: float = 60.0
    max_hz: float = 550.0
    min_p_frac_rom: float = 0.10
    min_wood: float = 0.01
    sigma_cluster_hz: float = 2.0
    min_sigma_sep_from_target_hz: float = 0.0
    keep_sigma_reference: bool = False
    max_sigma_reference_per_shift: int = 0
    near_dup_hz: float = 0.05
    min_uniqueness: float = 0.04
    # Staged FSI gate (60–550 Hz production sweep).
    staged_filtering: bool = True
    staged_crossover_hz: float = 350.0
    min_p_frac_rom_low: float = 0.02
    min_p_frac_rom_high: float = 0.02
    min_decoupled_p_frac: float = 0.02
    staged_sigma_tighten_hz: float = 400.0
    sigma_cluster_hz_high: float = 3.0
    min_wood_high_hz: float = 0.01
    # Wood-dominated body modes below crossover often have p_frac ~ 1e-10–1e-13
    # while max|p| remains finite; bypass weak p_frac when wood and |p|_max pass.
    wood_dominated_rom_bypass: bool = True
    min_wood_dominated_bypass: float = 0.90
    wood_bypass_max_hz: float = 350.0
    min_p_block_max_wood_bypass: float = 1.0e-14

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

        min_p_low = _f("min_p_frac_rom_low", _f("min_p_frac_rom", _f("min_p_frac", cls.min_p_frac_rom)))
        min_p_default = _f("min_p_frac_rom", _f("min_p_frac", min_p_low))

        return cls(
            min_hz=_f("min_hz", cls.min_hz),
            max_hz=_f("max_hz", cls.max_hz),
            min_p_frac_rom=min_p_default,
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
            staged_filtering=_b("staged_filtering", cls.staged_filtering),
            staged_crossover_hz=_f("staged_crossover_hz", cls.staged_crossover_hz),
            min_p_frac_rom_low=_f("min_p_frac_rom_low", min_p_low),
            min_p_frac_rom_high=_f("min_p_frac_rom_high", _f("min_p_frac_rom_high", 0.03)),
            min_decoupled_p_frac=_f("min_decoupled_p_frac", cls.min_decoupled_p_frac),
            staged_sigma_tighten_hz=_f("staged_sigma_tighten_hz", cls.staged_sigma_tighten_hz),
            sigma_cluster_hz_high=_f("sigma_cluster_hz_high", cls.sigma_cluster_hz_high),
            min_wood_high_hz=_f("min_wood_high_hz", _f("min_wood", cls.min_wood)),
            wood_dominated_rom_bypass=_b(
                "wood_dominated_rom_bypass", cls.wood_dominated_rom_bypass
            ),
            min_wood_dominated_bypass=_f(
                "min_wood_dominated_bypass", cls.min_wood_dominated_bypass
            ),
            wood_bypass_max_hz=_f("wood_bypass_max_hz", cls.wood_bypass_max_hz),
            min_p_block_max_wood_bypass=_f(
                "min_p_block_max_wood_bypass", cls.min_p_block_max_wood_bypass
            ),
        )


def sigma_cluster_hz_for_mode(f_hz: float, cfg: HarvestFilterConfig) -> float:
    """σ-cluster half-width (Hz); tighter above ``staged_sigma_tighten_hz`` when staged."""
    if cfg.staged_filtering and float(f_hz) >= float(cfg.staged_sigma_tighten_hz):
        return float(cfg.sigma_cluster_hz_high)
    return float(cfg.sigma_cluster_hz)


def min_p_frac_threshold_for_mode(f_hz: float, cfg: HarvestFilterConfig) -> float:
    """Required ``p_frac`` for ROM eligibility at mode frequency ``f_hz``."""
    if cfg.staged_filtering and float(f_hz) >= float(cfg.staged_crossover_hz):
        return float(cfg.min_p_frac_rom_high)
    if cfg.staged_filtering:
        return float(cfg.min_p_frac_rom_low)
    return float(cfg.min_p_frac_rom)


def min_wood_threshold_for_mode(f_hz: float, cfg: HarvestFilterConfig) -> float:
    if cfg.staged_filtering and float(f_hz) >= float(cfg.staged_crossover_hz):
        return float(cfg.min_wood_high_hz)
    return float(cfg.min_wood)


def _wood_dominated_bypass_ok(
    candidate: Dict[str, Any],
    *,
    f_hz: float,
    wood: float,
    cfg: HarvestFilterConfig,
) -> Tuple[bool, str]:
    """True when a low-band, wood-heavy mode may skip strict p_frac gates."""
    if not cfg.wood_dominated_rom_bypass:
        return False, "bypass_disabled"
    if float(f_hz) >= float(cfg.wood_bypass_max_hz) - 1e-9:
        return False, "above_wood_bypass_hz"
    if wood + 1e-15 < float(cfg.min_wood_dominated_bypass):
        return False, f"wood={wood:.4f}<{cfg.min_wood_dominated_bypass}"
    try:
        p_block_max = float(candidate.get("p_block_max", 0.0) or 0.0)
    except (TypeError, ValueError):
        p_block_max = 0.0
    if p_block_max + 1.0e-30 < float(cfg.min_p_block_max_wood_bypass):
        return (
            False,
            f"max|p|={p_block_max:.3e}<{cfg.min_p_block_max_wood_bypass:.3e}",
        )
    return True, f"wood_dominated_bypass(max|p|={p_block_max:.3e})"


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
    sig_tol = sigma_cluster_hz_for_mode(f_hz, cfg)

    if abs(f_hz - sigma) <= sig_tol:
        return "sigma_ritz", False, f"|f-st_sigma|={abs(f_hz - sigma):.3f}<={sig_tol}"

    if f_hz + 1e-9 < float(cfg.min_hz) or f_hz > float(cfg.max_hz) + 1e-9:
        return "out_of_band", False, f"hz={f_hz:.2f} outside [{cfg.min_hz},{cfg.max_hz}]"

    bypass_ok, bypass_detail = _wood_dominated_bypass_ok(
        candidate, f_hz=f_hz, wood=wood, cfg=cfg
    )

    if p_frac < float(cfg.min_decoupled_p_frac) and not bypass_ok:
        return "decoupled", False, f"p_frac={p_frac:.3e}<{cfg.min_decoupled_p_frac}"

    wood_min = min_wood_threshold_for_mode(f_hz, cfg)
    if wood < wood_min:
        return "low_wood", False, f"wood={wood:.4f}<{wood_min}"

    if uniq < float(cfg.min_uniqueness) - 1e-15:
        return "low_uniqueness", False, f"uniqueness={uniq:.3f}<{cfg.min_uniqueness}"

    p_need = min_p_frac_threshold_for_mode(f_hz, cfg)
    if p_frac < p_need:
        if bypass_ok:
            return (
                "physical_fsi",
                True,
                f"wood_dominated_low_band({bypass_detail},p_frac={p_frac:.3e})",
            )
        return "weak_coupling", False, f"p_frac={p_frac:.3f}<{p_need:.3f}"

    if cfg.staged_filtering and float(f_hz) >= float(cfg.staged_crossover_hz):
        if p_frac < float(cfg.min_p_frac_rom_low):
            return (
                "physical_fsi",
                True,
                f"staged_hf_ok(p_frac={p_frac:.3f}>={p_need:.3f},wood={wood:.3f})",
            )
        return "physical_fsi", True, "coupled_fsi_ok"

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
        "staged_filtering": bool(cfg.staged_filtering),
        "staged_crossover_hz": float(cfg.staged_crossover_hz),
        "min_p_frac_rom_low": float(cfg.min_p_frac_rom_low),
        "min_p_frac_rom_high": float(cfg.min_p_frac_rom_high),
        "wood_dominated_rom_bypass": bool(cfg.wood_dominated_rom_bypass),
        "min_wood_dominated_bypass": float(cfg.min_wood_dominated_bypass),
        "wood_bypass_max_hz": float(cfg.wood_bypass_max_hz),
        "min_p_block_max_wood_bypass": float(cfg.min_p_block_max_wood_bypass),
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
