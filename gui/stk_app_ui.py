#!/usr/bin/env python3
"""Streamlit UI panel for accepted STK classical guitar (background render + FIFO stack)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import streamlit as st

from stk_app_audio_service import (
    DEFAULT_SOURCE_SAMPLE_ID,
    cache_is_ready,
    compute_parameter_hash,
    get_note_wav,
    list_available_notes,
    list_guitar_stack,
    poll_background_job,
    preview_cache_dir,
    read_job_status,
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
        "ready": "✅ ready",
        "failed": "❌ failed",
        "stale": "⚠️ stale",
    }
    return labels.get(status, status)


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
        "**Generate Sound** saves the current guitar to the comparison stack when the cache is ready."
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

    stk_status = "waiting_for_rom"
    job_doc: Dict[str, Any] = {}
    if not rom_ready:
        stk_status = "waiting_for_rom"
    elif parameter_hash:
        job_doc = poll_background_job(parameter_hash)
        stk_status = str(job_doc.get("status") or "not_started")
        if cache_is_ready(preview) and stk_status != "ready":
            stk_status = "ready"

    st.markdown("##### STK background status")
    st.write(_status_badge(stk_status))
    if job_doc:
        rendered = job_doc.get("rendered_notes")
        total = job_doc.get("total_notes")
        if rendered is not None and total is not None:
            st.progress(min(1.0, float(rendered) / max(float(total), 1.0)))
            st.caption(f"Rendered {rendered} / {total} notes")
        if job_doc.get("elapsed_s") is not None:
            st.caption(
                f"Elapsed {job_doc.get('elapsed_s')} s · "
                f"hits {job_doc.get('cache_hit_count', 0)} · "
                f"misses {job_doc.get('cache_miss_count', 0)}"
            )
    if preview:
        st.caption(f"Preview cache: `{preview}`")

    binary_ok = stk_binary_path(root).is_file()
    if not binary_ok:
        st.info("Build STK on VM: `tools/build_stk_pgsm_demo.sh`")

    cache_ready = bool(preview and cache_is_ready(preview))
    notes = list_available_notes(DEFAULT_SOURCE_SAMPLE_ID, cache_dir=preview) if preview else []

    if cache_ready and notes:
        note_pick = st.selectbox("Preview note", notes, key=f"{base_key}_note")
        wav = get_note_wav(DEFAULT_SOURCE_SAMPLE_ID, note_pick, cache_dir=preview)
        if wav:
            st.audio(wav.read_bytes(), format="audio/wav")
    elif stk_status == "running":
        st.caption("STK audio cache is still rendering. Please wait or continue editing.")
    elif rom_ready:
        st.caption("STK preview cache not ready yet.")

    st.session_state[f"{base_key}_stk_ready"] = cache_ready
    st.session_state[f"{base_key}_parameter_hash"] = parameter_hash

    st.markdown("##### FIFO comparison stack (latest 3)")
    stack = list_guitar_stack()
    if not stack:
        st.caption("No saved guitars yet — use **Generate Sound** when STK cache is ready.")
    for row in reversed(stack):
        cache_path = Path(str(row.get("note_cache_path") or ""))
        st.caption(
            f"**{row.get('display_name')}** · `{row.get('saved_guitar_id')}` · "
            f"hash `{row.get('parameter_hash', '')[:8]}…`"
        )
        if row.get("geometry_summary"):
            g = row["geometry_summary"]
            st.caption(
                f"L={g.get('length')} W={g.get('width')} D={g.get('depth')} · "
                f"{g.get('top_wood_id')}/{g.get('back_wood_id')}"
            )
        cols = st.columns(len(COMPARE_NOTES))
        for col, cn in zip(cols, COMPARE_NOTES):
            with col:
                st.write(cn)
                w = get_note_wav(DEFAULT_SOURCE_SAMPLE_ID, cn, cache_dir=cache_path) if cache_path.is_dir() else None
                if w and w.is_file():
                    st.audio(w.read_bytes(), format="audio/wav")
                else:
                    st.caption("—")


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
    job = poll_background_job(parameter_hash)
    if job.get("status") != "ready" and not cache_is_ready(preview_cache_dir(parameter_hash)):
        raise RuntimeError(
            "STK audio cache is still rendering. Please wait or continue editing."
        )
    display_name = f"Guitar — {top_wood}/{back_wood}"
    return save_guitar_to_stack(
        parameter_hash=parameter_hash,
        display_name=display_name,
        geometry_summary=_geometry_summary(geom, top_wood, back_wood),
        rom_physical_summary_path=rom_physical_summary_path or None,
        repo_root=repo_root,
    )
