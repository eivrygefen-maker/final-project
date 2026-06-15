#!/usr/bin/env python3
"""Streamlit UI panel for accepted STK classical guitar note library."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from stk_app_audio_service import (
    build_note_library,
    get_latest_note_library_report,
    get_melody_wav,
    get_note_wav,
    list_available_notes,
    list_available_samples,
    list_guitar_stack,
    list_melody_ids,
    load_melody_library,
    note_cache_dir,
    push_guitar_snapshot,
    stk_binary_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _profile_label(sample_id: str) -> str:
    labels = {
        "sample_000": "balanced/neutral",
        "sample_001": "bright/light",
        "sample_002": "warm/deep",
    }
    return labels.get(sample_id, "LHS classical")


def render_stk_classical_panel(*, repo_root: Optional[Path] = None, base_key: str = "stk_app") -> None:
    root = Path(repo_root or REPO_ROOT)
    st.subheader("STK Classical Guitar Library")
    st.caption(
        "Accepted STK/C++ renderer — Python exports parameters only. "
        "Build a chromatic note cache, play individual notes, or render simple melodies."
    )

    samples = list_available_samples(root)
    sample_id = st.selectbox(
        "Guitar sample",
        options=samples,
        index=0 if "sample_000" in samples else 0,
        key=f"{base_key}_sample",
    )
    cache_dir = note_cache_dir(sample_id, "classical")
    cached_notes = list_available_notes(sample_id)
    latest = get_latest_note_library_report(sample_id)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Cached notes", len(cached_notes))
    with col_b:
        if latest:
            st.metric("Last total render (s)", latest.get("total_render_time_s", "—"))
        else:
            st.metric("Last total render (s)", "—")

    force = st.checkbox("Force regenerate all notes", value=False, key=f"{base_key}_force")
    binary = stk_binary_path(root)
    binary_ok = binary.is_file()

    if st.button("Generate STK Note Library", type="primary", key=f"{base_key}_build", disabled=not binary_ok):
        if not binary_ok:
            st.error(f"STK binary missing: {binary}")
        else:
            with st.spinner("Exporting parameters and rendering notes via STK/C++…"):
                try:
                    report = build_note_library(
                        sample_id,
                        instrument="classical",
                        note_range="E2:E5",
                        force=force,
                        repo_root=root,
                        binary=binary,
                    )
                    st.session_state[f"{base_key}_last_report"] = report
                    push_guitar_snapshot(
                        sample_id=sample_id,
                        display_name=f"Guitar — {_profile_label(sample_id)}",
                        physical_summary=report.get("rom_physical_summary") or {},
                        note_cache_path=report.get("output_dir"),
                        timing_report_path=report.get("report_json"),
                    )
                    if report.get("readiness") == "ready_for_app_playback":
                        st.success(
                            f"Library ready — {report['note_count']} notes, "
                            f"{report['total_render_time_s']} s total "
                            f"({report['average_time_per_note_s']} s avg/note)"
                        )
                    else:
                        st.warning(f"Readiness: {report.get('readiness')}")
                except Exception as exc:
                    st.error(f"STK note library failed: {exc}")

    if not binary_ok:
        st.info("Build the STK renderer on VM: `tools/build_stk_pgsm_demo.sh`")

    report: Dict[str, Any] = st.session_state.get(f"{base_key}_last_report") or latest or {}
    if report:
        st.caption(
            f"Cache: `{cache_dir}` · hits {report.get('cache_hit_count', 0)} · "
            f"misses {report.get('cache_miss_count', 0)}"
        )

    notes = list_available_notes(sample_id)
    if notes:
        note_pick = st.selectbox("Note", notes, key=f"{base_key}_note")
        wav = get_note_wav(sample_id, note_pick)
        if wav and wav.is_file():
            st.audio(wav.read_bytes(), format="audio/wav")
    else:
        st.caption("No cached STK notes yet — click **Generate STK Note Library**.")

    st.markdown("##### Melodies (from cached notes)")
    try:
        melody_ids = list_melody_ids()
        lib = load_melody_library()
        id_to_name = {str(m["id"]): str(m.get("display_name") or m["id"]) for m in lib.get("melodies") or []}
    except FileNotFoundError:
        melody_ids = []
        id_to_name = {}

    if melody_ids:
        melody_id = st.selectbox(
            "Melody",
            melody_ids,
            format_func=lambda mid: id_to_name.get(mid, mid),
            key=f"{base_key}_melody",
        )
        mcol1, mcol2 = st.columns(2)
        melody_out = get_melody_wav(sample_id, melody_id)
        with mcol1:
            if st.button("Render melody", key=f"{base_key}_render_melody"):
                import subprocess
                import sys

                cmd = [
                    sys.executable,
                    str(root / "tools" / "render_app_stk_melody.py"),
                    "--sample-id",
                    sample_id,
                    "--melody-id",
                    melody_id,
                    "--output-dir",
                    str(root / "audio" / "app_stk_melody_cache" / "classical" / sample_id),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
                if proc.returncode == 0:
                    st.success("Melody rendered from cached STK notes.")
                else:
                    st.error(proc.stderr or proc.stdout or "Melody render failed")
        with mcol2:
            if melody_out and melody_out.is_file():
                st.caption("Cached melody")
            else:
                st.caption("Melody not cached")
        melody_out = get_melody_wav(sample_id, melody_id)
        if melody_out and melody_out.is_file():
            st.audio(melody_out.read_bytes(), format="audio/wav")

    stack = list_guitar_stack()
    if stack:
        st.markdown("##### Guitar comparison stack (latest 3)")
        for row in stack:
            st.caption(f"**{row.get('display_name')}** — `{row.get('sample_id')}` @ {row.get('timestamp')}")
            compare_notes = ["A2", "A4", "E5"]
            cols = st.columns(len(compare_notes))
            for col, cn in zip(cols, compare_notes):
                w = get_note_wav(str(row.get("sample_id")), cn)
                with col:
                    st.write(cn)
                    if w and w.is_file():
                        st.audio(w.read_bytes(), format="audio/wav")
                    else:
                        st.caption("—")
