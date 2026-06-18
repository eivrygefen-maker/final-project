#!/usr/bin/env python3
"""Streamlit UI — simplified STK UX (quiet background render, ready-only FIFO)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import streamlit as st

from app_stk_config import load_app_stk_config
from app_stk_instrument import default_sample_id_for_shape  # noqa: WPS433
from stk_app_audio_service import (
    activate_stk_guitar_for_player,
    compute_parameter_hash,
    find_stack_entry_by_hash,
    get_note_wav,
    list_available_notes,
    list_ready_guitar_stack,
    load_stack_guitar_for_player,
    preview_cache_dir,
    preview_cache_dir_has_required_notes,
    refresh_stk_background_job_status,
    resolve_preview_cache_ready_state,
    save_guitar_to_stack,
    stk_binary_path,
    user_facing_stk_status,
    AUDIT_INCOMPLETE_MSG,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_NOTES = ("A2", "A4", "E5")


def _session_set(key: str, value: Any) -> None:
    """Write session state (works with Streamlit SessionState and plain dict mocks)."""
    st.session_state[key] = value


def _norm_cache_path(path: Any) -> str:
    return str(path or "").replace("\\", "/")


def _player_key(parameter_hash: str, cache_path: str, player_fingerprint: str) -> str:
    raw = f"{parameter_hash}|{cache_path}|{player_fingerprint}"
    return "".join(ch if ch.isalnum() else "_" for ch in raw)[-120:]


def _log_once(key: str, message: str) -> None:
    seen_key = f"_logged_{key}"
    if st.session_state.get(seen_key):
        return
    _session_set(seen_key, True)
    print(message, flush=True)


def _active_player_matches(parameter_hash: str, cache_dir: Path) -> bool:
    cache_path = _norm_cache_path(cache_dir)
    payload = dict(st.session_state.get("active_stk_player_payload") or {})
    return (
        str(st.session_state.get("active_player_hash") or "") == str(parameter_hash)
        and _norm_cache_path(st.session_state.get("active_player_cache_dir")) == cache_path
        and payload.get("status") == "ready"
        and bool(payload.get("positions"))
    )


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


def apply_stk_activation_to_session(activation: Mapping[str, Any]) -> None:
    """Persist active STK guitar for the fretboard player."""
    validation = dict(activation.get("validation") or {})
    payload = dict(activation.get("player_payload") or {})
    if not validation.get("ok"):
        payload = {"status": "hidden", "positions": [], "fingerprint": ""}
        st.warning(AUDIT_INCOMPLETE_MSG)
    cache_path = _norm_cache_path(activation.get("cache_path"))
    parameter_hash = str(activation.get("parameter_hash") or "")
    player_fp = str(activation.get("player_fingerprint") or "")
    _session_set("active_stk_cache_path", cache_path)
    _session_set("active_stk_parameter_hash", parameter_hash)
    _session_set("active_stk_guitar_id", str(activation.get("saved_guitar_id") or ""))
    _session_set("active_stk_player_fp", player_fp)
    _session_set("active_player_hash", parameter_hash)
    _session_set("active_player_cache_dir", cache_path)
    _session_set("loaded_player_hash", parameter_hash)
    _session_set("loaded_player_cache_dir", cache_path)
    _session_set("active_stk_player_key", _player_key(parameter_hash, cache_path, player_fp))
    _session_set("active_stk_player_payload", payload)
    _session_set("active_stk_player_validation", validation)
    _session_set("sound_stale", False)


def render_user_stk_status_line(stk_status: str) -> None:
    st.caption(user_facing_stk_status(stk_status))


def render_saved_guitars_row(
    *,
    base_key: str = "stk_saved",
    on_load_key: str = "stk_load_guitar",
    instrument: str = "classical",
) -> Optional[str]:
    """Saved guitars row near the fretboard player. Returns saved_guitar_id if Load clicked."""
    if not load_app_stk_config().get("enable_ready_fifo_stack", True):
        return None
    stack = list_ready_guitar_stack(instrument)
    if not stack:
        return None

    st.markdown("##### Saved guitars")
    loaded_id: Optional[str] = None
    for row in reversed(stack):
        cols = st.columns([3, 1])
        with cols[0]:
            st.caption(
                f"**{row.get('display_name')}** · "
                f"{str(row.get('timestamp', ''))[:19].replace('T', ' ')}"
            )
        with cols[1]:
            sid = str(row.get("saved_guitar_id") or "")
            if st.button("Load", key=f"{base_key}_load_{sid}", use_container_width=True):
                loaded_id = sid
    return loaded_id


def _stk_cache_is_loadable(parameter_hash: str, repo_root: Path, instrument: str = "classical") -> bool:
    preview = preview_cache_dir(parameter_hash, instrument)
    return preview.is_dir() and preview_cache_dir_has_required_notes(preview, load_app_stk_config(repo_root))


def _activate_ready_preview_cache(
    *,
    repo_root: Path,
    parameter_hash: str,
    cache_dir: Path,
    saved_guitar_id: str = "",
) -> Dict[str, Any]:
    """Activate player from a ready preview cache; raise on validation failure."""
    _log_once(
        f"load_ready_cache_{parameter_hash}_{_player_key(parameter_hash, _norm_cache_path(cache_dir), saved_guitar_id)}",
        f"APP_STK_LOAD_READY_CACHE hash={parameter_hash} cache_dir={cache_dir}",
    )
    activation = activate_stk_guitar_for_player(
        cache_dir=cache_dir,
        parameter_hash=parameter_hash,
        saved_guitar_id=saved_guitar_id,
    )
    validation = dict(activation.get("validation") or {})
    if not validation.get("ok"):
        errors = validation.get("errors") or [AUDIT_INCOMPLETE_MSG]
        raise RuntimeError(f"Could not load guitar player: {'; '.join(errors)}")
    return activation


def generate_or_load_ready_guitar(
    *,
    repo_root: Path,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    geom: Mapping[str, Any],
    top_wood: str,
    back_wood: str,
    rom_physical_summary_path: str = "",
    instrument: str = "classical",
) -> Dict[str, Any]:
    """Save ready guitar to FIFO or load existing duplicate; activate player."""
    cfg = load_app_stk_config(repo_root)
    parameter_hash = compute_parameter_hash(rom_fp, lhs_params)
    state = resolve_preview_cache_ready_state(parameter_hash, instrument=instrument, repo_root=repo_root)

    if str(state.get("status")) != "ready" or not state.get("preview_cache_ready"):
        raise RuntimeError(
            "Guitar sound is still being prepared. Please wait a little longer."
        )

    cache_dir = Path(str(state.get("preview_cache_path") or preview_cache_dir(parameter_hash, instrument)))

    if not cfg.get("enable_ready_fifo_stack", True):
        if _active_player_matches(parameter_hash, cache_dir):
            _session_set("stk_generate_intent_hash", "")
            return {"action": "activated_preview", "activation": None, "already_loaded": True}
        activation = _activate_ready_preview_cache(
            repo_root=repo_root,
            parameter_hash=parameter_hash,
            cache_dir=cache_dir,
        )
        apply_stk_activation_to_session(activation)
        _session_set("stk_generate_intent_hash", "")
        return {"action": "activated_preview", "activation": activation}

    existing = find_stack_entry_by_hash(parameter_hash, instrument)
    if existing:
        if _active_player_matches(parameter_hash, cache_dir):
            _session_set("stk_generate_intent_hash", "")
            return {"action": "loaded_existing", "entry": existing, "activation": None, "already_loaded": True}
        activation = _activate_ready_preview_cache(
            repo_root=repo_root,
            parameter_hash=parameter_hash,
            cache_dir=cache_dir,
        )
        apply_stk_activation_to_session(activation)
        _session_set("stk_generate_intent_hash", "")
        return {"action": "loaded_existing", "entry": existing, "activation": activation}

    display_name = f"Guitar — {top_wood}/{back_wood}"
    entry = save_guitar_to_stack(
        parameter_hash=parameter_hash,
        display_name=display_name,
        geometry_summary=_geometry_summary(geom, top_wood, back_wood),
        rom_physical_summary_path=rom_physical_summary_path or None,
        repo_root=repo_root,
        instrument=instrument,
    )
    if entry.get("_duplicate"):
        if _active_player_matches(parameter_hash, cache_dir):
            _session_set("stk_generate_intent_hash", "")
            return {"action": "loaded_existing", "entry": entry, "activation": None, "already_loaded": True}
        activation = _activate_ready_preview_cache(
            repo_root=repo_root,
            parameter_hash=parameter_hash,
            cache_dir=cache_dir,
        )
        apply_stk_activation_to_session(activation)
        _session_set("stk_generate_intent_hash", "")
        return {"action": "loaded_existing", "entry": entry, "activation": activation}

    if _active_player_matches(parameter_hash, cache_dir):
        _session_set("stk_generate_intent_hash", "")
        return {"action": "saved_new", "entry": entry, "activation": None, "already_loaded": True}
    activation = _activate_ready_preview_cache(
        repo_root=repo_root,
        parameter_hash=parameter_hash,
        cache_dir=cache_dir,
    )
    apply_stk_activation_to_session(activation)
    _session_set("stk_generate_intent_hash", "")
    return {"action": "saved_new", "entry": entry, "activation": activation}


def _clear_stk_render_request() -> None:
    """Clear Generate-triggered STK watch flags only (not design/session keys)."""
    _session_set("stk_render_requested", False)
    _session_set("stk_render_requested_hash", "")
    _session_set("stk_generate_intent_hash", "")


def _set_stk_render_request(parameter_hash: str) -> None:
    """Record a single-click Generate request for auto-load when STK finishes."""
    import time

    _session_set("stk_render_requested", True)
    _session_set("stk_render_requested_hash", parameter_hash)
    _session_set("stk_render_started_at", time.time())
    _session_set("show_clickable_guitar_requested", True)
    _session_set(
        "generate_request_count",
        int(st.session_state.get("generate_request_count") or 0) + 1,
    )


def poll_stk_render_request(
    *,
    repo_root: Path,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    geom: Mapping[str, Any],
    top_wood: str,
    back_wood: str,
    rom_physical_summary_path: str = "",
    instrument: str = "classical",
) -> Dict[str, Any]:
    """Refresh STK job status; auto-load player when a Generate request completes."""
    out: Dict[str, Any] = {"polled": False, "result": None}
    if not st.session_state.get("stk_render_requested"):
        return out
    requested = str(st.session_state.get("stk_render_requested_hash") or "")
    if not requested:
        return out

    parameter_hash = compute_parameter_hash(rom_fp, lhs_params)
    out["polled"] = True

    if requested != parameter_hash:
        _clear_stk_render_request()
        out["result"] = {"action": "stk_request_stale"}
        return out

    state = resolve_preview_cache_ready_state(parameter_hash, instrument=instrument, repo_root=repo_root)
    status = str(state.get("status") or "not_started")
    preview_ready = bool(state.get("preview_cache_ready")) or _stk_cache_is_loadable(
        parameter_hash, repo_root, instrument
    )
    _session_set("stk_parameter_hash", parameter_hash)
    _session_set("stk_job_status", status if not preview_ready else "ready")
    _session_set("stk_preview_cache_ready", preview_ready)
    _session_set(
        "stk_preview_cache_path",
        str(state.get("preview_cache_path") or preview_cache_dir(parameter_hash, instrument)).replace("\\", "/"),
    )
    _session_set(
        "stk_note_count",
        int(
            state.get("actual_wav_count")
            or state.get("wav_count")
            or state.get("note_count")
            or 0
        ),
    )
    _session_set("stk_last_status", "ready" if preview_ready else status)

    if status == "failed":
        _clear_stk_render_request()
        out["result"] = {"action": "stk_failed", "status": status}
        return out

    if preview_ready:
        ready_cache = preview_cache_dir(parameter_hash, instrument)
        _log_once(
            f"auto_load_ready_{parameter_hash}_{_norm_cache_path(ready_cache)}",
            f"APP_STK_AUTO_LOAD_READY hash={parameter_hash} cache_dir={ready_cache}",
        )
        try:
            result = generate_or_load_ready_guitar(
                repo_root=repo_root,
                rom_fp=rom_fp,
                lhs_params=lhs_params,
                geom=geom,
                top_wood=top_wood,
                back_wood=back_wood,
                rom_physical_summary_path=rom_physical_summary_path,
                instrument=instrument,
            )
            _clear_stk_render_request()
            out["result"] = result
            return out
        except Exception as exc:
            out["result"] = {"action": "stk_load_failed", "error": str(exc)}
            return out

    out["result"] = {"action": "stk_running", "status": status}
    return out


def render_stk_render_watch_panel(
    *,
    repo_root: Path,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    geom: Mapping[str, Any],
    top_wood: str,
    back_wood: str,
    rom_physical_summary_path: str = "",
    instrument: str = "classical",
) -> Optional[Dict[str, Any]]:
    """Show STK build status and auto-load when ready (fragment-safe; no meta refresh)."""
    poll = poll_stk_render_request(
        repo_root=repo_root,
        rom_fp=rom_fp,
        lhs_params=lhs_params,
        geom=geom,
        top_wood=top_wood,
        back_wood=back_wood,
        rom_physical_summary_path=rom_physical_summary_path,
        instrument=instrument,
    )
    if not poll.get("polled"):
        return None
    result = poll.get("result") or {}
    action = str(result.get("action") or "")
    if action in ("saved_new", "loaded_existing", "activated_preview"):
        if action == "loaded_existing":
            st.success("Guitar sound is ready — loaded from comparison stack.")
        elif action == "activated_preview":
            st.success("Guitar sound is ready — player loaded.")
        else:
            name = str(result.get("entry", {}).get("display_name") or "guitar")
            st.success(f"Guitar sound is ready — saved **{name}**.")
    elif action == "stk_failed":
        st.error("STK rendering failed. Please try **Generate Sound** again.")
    elif action == "stk_load_failed":
        st.error(str(result.get("error") or "Could not load guitar player."))
    elif action == "stk_running":
        st.info(
            "Building guitar sound with STK… This may take a few minutes. "
            "The player will load automatically when ready."
        )
    return result if action else None


def request_generate_guitar(
    *,
    repo_root: Path,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    geom: Mapping[str, Any],
    top_wood: str,
    back_wood: str,
    rom_physical_summary_path: str = "",
    instrument: str = "classical",
) -> Dict[str, Any]:
    """Generate click: request display only; never start or restart STK."""
    parameter_hash = compute_parameter_hash(rom_fp, lhs_params)
    _set_stk_render_request(parameter_hash)

    ready_cache = preview_cache_dir(parameter_hash, instrument)
    if _active_player_matches(parameter_hash, ready_cache):
        _clear_stk_render_request()
        return {"action": "activated_preview", "activation": None, "already_loaded": True}

    if _stk_cache_is_loadable(parameter_hash, repo_root, instrument):
        return generate_or_load_ready_guitar(
            repo_root=repo_root,
            rom_fp=rom_fp,
            lhs_params=lhs_params,
            geom=geom,
            top_wood=top_wood,
            back_wood=back_wood,
            rom_physical_summary_path=rom_physical_summary_path,
            instrument=instrument,
        )

    state = resolve_preview_cache_ready_state(parameter_hash, instrument=instrument, repo_root=repo_root)
    status = str(state.get("status") or "not_started")
    preview_ready = bool(state.get("preview_cache_ready"))

    if preview_ready or status == "ready":
        return generate_or_load_ready_guitar(
            repo_root=repo_root,
            rom_fp=rom_fp,
            lhs_params=lhs_params,
            geom=geom,
            top_wood=top_wood,
            back_wood=back_wood,
            rom_physical_summary_path=rom_physical_summary_path,
            instrument=instrument,
        )

    if status == "failed":
        return {
            "action": "stk_failed",
            "parameter_hash": parameter_hash,
            "status": status,
            "error": state.get("error"),
        }

    if status in ("running", "partial_ready", "not_started", "waiting_for_rom", "stale"):
        return {
            "action": "stk_running",
            "parameter_hash": parameter_hash,
            "status": status,
        }

    return {
        "action": "stk_running",
        "parameter_hash": parameter_hash,
        "status": status,
    }


def fulfill_generate_intent_if_ready(
    *,
    repo_root: Path,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    geom: Mapping[str, Any],
    top_wood: str,
    back_wood: str,
    rom_physical_summary_path: str = "",
    instrument: str = "classical",
) -> Optional[Dict[str, Any]]:
    """Legacy alias — auto-load uses ``poll_stk_render_request`` / ``stk_render_requested_hash``."""
    poll = poll_stk_render_request(
        repo_root=repo_root,
        rom_fp=rom_fp,
        lhs_params=lhs_params,
        geom=geom,
        top_wood=top_wood,
        back_wood=back_wood,
        rom_physical_summary_path=rom_physical_summary_path,
        instrument=instrument,
    )
    return poll.get("result")


def render_stk_diagnostics_panel(
    *,
    repo_root: Optional[Path] = None,
    base_key: str = "stk_diag",
    rom_fp: str = "",
    lhs_params: Optional[Mapping[str, Any]] = None,
    rom_ready: bool = False,
    rom_pending: bool = False,
    rom_error: str = "",
    instrument: str = "classical",
) -> None:
    """Developer-only STK diagnostics (collapsed by default)."""
    root = Path(repo_root or REPO_ROOT)
    parameter_hash = compute_parameter_hash(rom_fp, lhs_params) if rom_fp else ""
    preview = preview_cache_dir(parameter_hash, instrument) if parameter_hash else None
    shape = "Classical"
    if lhs_params:
        shape = str(
            lhs_params.get("geometry.shape_type")
            or lhs_params.get("shape_type")
            or "Classical"
        )
    sample_id = default_sample_id_for_shape(shape)

    job_doc: Dict[str, Any] = {}
    stk_status = "waiting_for_rom"
    if rom_ready and parameter_hash:
        job_doc = refresh_stk_background_job_status(parameter_hash, instrument=instrument)
        stk_status = str(job_doc.get("status") or "not_started")

    st.caption(f"Internal status: `{stk_status}` · hash `{parameter_hash[:8]}…`" if parameter_hash else "")
    if job_doc:
        st.json(
            {
                k: job_doc.get(k)
                for k in (
                    "status",
                    "preview_cache_ready",
                    "actual_wav_count",
                    "reported_rendered_notes",
                    "total_notes",
                    "preview_cache_path",
                    "latest_report_path",
                    "stale_reason",
                )
                if job_doc.get(k) is not None
            }
        )
    if preview and preview.is_dir():
        notes = list_available_notes(sample_id, cache_dir=preview)
        st.caption(f"Preview WAV count: {len(notes)}")
    if not stk_binary_path(root).is_file():
        st.caption("STK binary not built on this machine.")

    stack = list_ready_guitar_stack(instrument)
    st.caption(f"Ready FIFO entries: {len(stack)}")
    for row in reversed(stack):
        cache_entry = Path(str(row.get("note_cache_path") or ""))
        st.caption(f"`{row.get('saved_guitar_id')}` → `{cache_entry}`")
        cols = st.columns(len(COMPARE_NOTES))
        for col, cn in zip(cols, COMPARE_NOTES):
            with col:
                st.caption(cn)
                w = get_note_wav(sample_id, cn, cache_dir=cache_entry) if cache_entry.is_dir() else None
                if w and w.is_file():
                    st.audio(w.read_bytes(), format="audio/wav")


def try_save_current_guitar_to_stack(
    *,
    repo_root: Path,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    geom: Mapping[str, Any],
    top_wood: str,
    back_wood: str,
    rom_physical_summary_path: str = "",
    instrument: str = "classical",
) -> Dict[str, Any]:
    """Backward-compatible alias — ready-only save/load with player activation."""
    return generate_or_load_ready_guitar(
        repo_root=repo_root,
        rom_fp=rom_fp,
        lhs_params=lhs_params,
        geom=geom,
        top_wood=top_wood,
        back_wood=back_wood,
        rom_physical_summary_path=rom_physical_summary_path,
        instrument=instrument,
    )
