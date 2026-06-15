#!/usr/bin/env python3
"""Streamlit UI panel for accepted STK classical guitar (background render + FIFO stack)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import streamlit as st

from stk_app_audio_service import (
    DEFAULT_SOURCE_SAMPLE_ID,
    compute_parameter_hash,
    get_note_wav,
    list_available_notes,
    list_guitar_stack,
    preview_cache_dir,
    promote_pending_stack_entries,
    refresh_stk_background_job_status,
    save_guitar_to_stack,
    stk_binary_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_NOTES = ("A2", "A4", "E5")


def _geometry_summary(geom: Mapping[str, Any], top_wood: str, back_wood: str) -> Dict[str, Any]:
    return {
        "length": geom.get("length"),
        "width": geom.get("width"),
        "depth": geom.get("depth"),
        "top_thickness": geom.get("top_thickness"),
        "hole_radius": geom.get("hole_radius"),
        "top_wood_id": top_wood,
        "back_wood_id": back_wood,
    }


def _status_badge(status: str) -> str:
    labels = {
        "not_started": "⏳ not started",
        "waiting_for_rom": "⏳ waiting for ROM",
        "running": "🔄 running",
        "partial_ready": "🎵 partial ready (A2/A4/E5)",
        "ready": "✅ ready",
        "failed": "❌ failed",
        "stale": "⚠️ stale",
        "pending_audio": "⏳ pending audio",
        "partial_audio": "🎵 partial audio",
        "failed_audio": "❌ failed audio",
    }
    return labels.get(status, status)


def _render_quick_play(
    cache_dir: Optional[Path],
    *,
    base_key: str,
    label_prefix: str = "",
) -> None:
    if cache_dir is None or not cache_dir.is_dir():
        return
    cols = st.columns(len(COMPARE_NOTES))
    for col, note_name in zip(cols, COMPARE_NOTES):
        with col:
            st.caption(f"{label_prefix}{note_name}")
            wav = get_note_wav(DEFAULT_SOURCE_SAMPLE_ID, note_name, cache_dir=cache_dir)
            if wav and wav.is_file():
                st.audio(wav.read_bytes(), format="audio/wav")
            else:
                st.caption("—")


def render_stk_classical_panel(
    *,
    repo_root: Optional[Path] = None,
    base_key: str = "stk_app",
    rom_fp: str = "",
    lhs_params: Optional[Mapping[str, Any]] = None,
    geom: Optional[Mapping[str, Any]] = None,
    top_wood: str = "",
    back_wood: str = "",
    rom_ready: bool = False,
    rom_pending: bool = False,
    rom_error: str = "",
) -> None:
    root = Path(repo_root or REPO_ROOT)
    st.subheader("STK Classical Guitar")
    st.caption(
        "Accepted STK/C++ renderer. Save runs ROM; STK note cache builds automatically in the background. "
        "**Generate Sound** saves the current guitar to the comparison stack (ready or pending)."
    )

    parameter_hash = compute_parameter_hash(rom_fp, lhs_params) if rom_fp else ""
    preview = preview_cache_dir(parameter_hash) if parameter_hash else None

    if rom_pending:
        rom_status = "running"
    elif rom_ready:
        rom_status = "ready"
    elif rom_error:
        rom_status = "failed"
    else:
        rom_status = "waiting"

    st.markdown("##### ROM status")
    st.write(f"ROM: **{rom_status}**" + (f" — {rom_error}" if rom_error else ""))

    job_doc: Dict[str, Any] = {}
    stk_status = "waiting_for_rom"
    if not rom_ready:
        stk_status = "waiting_for_rom"
    elif parameter_hash:
        promote_pending_stack_entries(parameter_hash)
        job_doc = refresh_stk_background_job_status(parameter_hash)
        stk_status = str(job_doc.get("status") or "not_started")

    st.markdown("##### STK background status")
    st.write(_status_badge(stk_status))
    if stk_status == "stale":
        st.warning("Previous STK job is stale because parameters changed.")
        st.caption("Save & Sync starts the correct cache for the current design.")

    if job_doc:
        actual = job_doc.get("actual_wav_count")
        if actual is None:
            actual = job_doc.get("wav_count")
        reported = job_doc.get("reported_rendered_notes")
        total = job_doc.get("total_notes")
        if actual is not None and total is not None:
            st.progress(min(1.0, float(actual) / max(float(total), 1.0)))
            if reported is not None and int(reported) != int(actual):
                st.caption(
                    f"Cached {actual} / {total} notes "
                    f"(progress file reports {reported})"
                )
            else:
                st.caption(f"Cached {actual} / {total} notes")
        elif job_doc.get("rendered_notes") is not None and total is not None:
            st.progress(min(1.0, float(job_doc.get("rendered_notes")) / max(float(total), 1.0)))
            st.caption(f"Rendered {job_doc.get('rendered_notes')} / {total} notes")
        elapsed = job_doc.get("elapsed_time_s")
        if elapsed is None:
            elapsed = job_doc.get("elapsed_s")
        if elapsed is not None:
            st.caption(
                f"Elapsed {elapsed} s · "
                f"hits {job_doc.get('cache_hit_count', 0)} · "
                f"misses {job_doc.get('cache_miss_count', 0)}"
            )

    if job_doc.get("preview_cache_ready"):
        st.success(
            f"STK cache ready — **{job_doc.get('wav_count', job_doc.get('note_count', 0))}** / "
            f"**{job_doc.get('total_notes', 37)}** notes"
        )
    elif stk_status == "partial_ready":
        st.info("Priority notes A2/A4/E5 are ready; full library still rendering.")

    if preview:
        st.caption(f"Preview cache: `{job_doc.get('preview_cache_path') or preview}`")
    if job_doc.get("latest_report_path"):
        st.caption(f"Latest report: `{job_doc['latest_report_path']}`")

    if not stk_binary_path(root).is_file():
        st.info("Build STK on VM: `tools/build_stk_pgsm_demo.sh`")

    cache_path = Path(str(job_doc.get("preview_cache_path") or preview or ""))
    notes = list_available_notes(DEFAULT_SOURCE_SAMPLE_ID, cache_dir=cache_path) if cache_path.is_dir() else []

    if notes:
        st.markdown("##### Preview notes")
        st.caption(f"{len(notes)} note WAV(s) in cache")
        if stk_status in ("ready", "partial_ready", "running"):
            _render_quick_play(cache_path, base_key=f"{base_key}_priority", label_prefix="")
        note_pick = st.selectbox("All cached notes", notes, key=f"{base_key}_note")
        wav = get_note_wav(DEFAULT_SOURCE_SAMPLE_ID, note_pick, cache_dir=cache_path)
        if wav:
            st.audio(wav.read_bytes(), format="audio/wav")
    elif stk_status == "running":
        st.caption("STK audio cache is still rendering. A2/A4/E5 appear first.")
    elif rom_ready:
        st.caption("STK preview cache not ready yet.")

    st.session_state[f"{base_key}_stk_ready"] = bool(job_doc.get("preview_cache_ready"))
    st.session_state[f"{base_key}_stk_status"] = stk_status
    st.session_state[f"{base_key}_parameter_hash"] = parameter_hash
    st.session_state[f"{base_key}_preview_cache_path"] = str(cache_path) if cache_path else ""
    st.session_state[f"{base_key}_note_count"] = len(notes)

    st.markdown("##### FIFO comparison stack (latest 3)")
    stack = list_guitar_stack()
    if not stack:
        st.caption("No saved guitars yet — use **Generate Sound** to save the current design.")
    for row in reversed(stack):
        entry_status = str(row.get("status") or "ready")
        st.caption(
            f"**{row.get('display_name')}** · {_status_badge(entry_status)} · "
            f"`{row.get('saved_guitar_id')}` · hash `{str(row.get('parameter_hash', ''))[:8]}…`"
        )
        st.caption(f"Saved {row.get('timestamp', '')}")
        if row.get("geometry_summary"):
            g = row["geometry_summary"]
            st.caption(
                f"L={g.get('length')} W={g.get('width')} D={g.get('depth')} · "
                f"{g.get('top_wood_id')}/{g.get('back_wood_id')}"
            )
        cache_entry = Path(str(row.get("note_cache_path") or ""))
        if entry_status == "pending_audio":
            st.caption("Audio pending — will attach when STK cache completes.")
        elif entry_status == "partial_audio":
            st.caption("Partial audio saved — full library still rendering.")
        elif entry_status == "failed_audio":
            st.caption(f"Audio failed: {row.get('error', 'see STK report')}")
        elif cache_entry.is_dir():
            st.caption(f"Cache: `{cache_entry}`")
            _render_quick_play(cache_entry, base_key=f"{base_key}_{row.get('saved_guitar_id')}")


def try_save_current_guitar_to_stack(
    *,
    repo_root: Path,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    geom: Mapping[str, Any],
    top_wood: str,
    back_wood: str,
    rom_physical_summary_path: str = "",
) -> Dict[str, Any]:
    parameter_hash = compute_parameter_hash(rom_fp, lhs_params)
    refresh_stk_background_job_status(parameter_hash)
    display_name = f"Guitar — {top_wood}/{back_wood}"
    return save_guitar_to_stack(
        parameter_hash=parameter_hash,
        display_name=display_name,
        geometry_summary=_geometry_summary(geom, top_wood, back_wood),
        rom_physical_summary_path=rom_physical_summary_path or None,
        repo_root=repo_root,
    )
