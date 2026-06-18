# CODEX_HANDOFF.md

## Root cause of player not rendering
- Ready cache detection was not latched to one stable active player state.
- Generate/auto-load detected `current_preview_<hash>` as ready, then the ready-stack path could activate `saved_guitar_<hash>_<timestamp>` for the same hash.
- The active cache/fingerprint could therefore be reprocessed across reruns instead of settling into one player payload/key.

## Root cause of repeated logs
- Readiness checks and auto-load ran on repeated Streamlit reruns.
- Load logs were emitted before checking whether the same hash/cache was already active.
- Position WAV readiness was printed every time aliases were verified.

## Session-state / latch fix
- Added stable active player fields:
- `active_player_hash`
- `active_player_cache_dir`
- `loaded_player_hash`
- `loaded_player_cache_dir`
- `active_stk_player_key`
- `apply_stk_activation_to_session()` now sets these alongside the existing `active_stk_*` fields.
- Repeated Generate/auto-load for an already-active hash/cache returns without reactivating the player.

## Canonical cache_dir rule
- Current active Generate/auto-load now prefers `current_preview_<hash>`.
- `saved_guitar_<hash>_<timestamp>` remains valid for explicit Recent/stack loading only.
- The component key now uses `active_stk_player_key`, derived from hash/cache/fingerprint.

## Recent Guitars behavior
- FIFO behavior remains unchanged.
- Loading a Recent Guitar still activates its stored cache directly.
- Loading a Recent Guitar still clears `stk_render_requested` and does not start STK generation.

## Component / logging fix
- `guitar_player()` still receives the existing player payload.
- Its Streamlit key is now stable for the active hash/cache/fingerprint.
- `DEBUG fast_preview...` logs now require `APP_DEBUG_FAST_PREVIEW`.
- `APP_STK_POSITION_WAVS_READY`, `APP_STK_LOAD_READY_CACHE`, and `APP_STK_AUTO_LOAD_READY` are guarded per readiness/load transition.

## CLASSIC-only confirmation
- No BOX or ACOUSTIC UI choices were added.
- Shape payloads remain Classical-only.
- No FEM/ROM/STK physics, solver, synthesis, or WAV generation behavior changed.

## Files changed
- `gui/app.py`
- `gui/stk_app_ui.py`
- `gui/stk_app_audio_service.py`
- `gui/test_stk_note_library_startup_command.py`
- `CODEX_HANDOFF.md`

## Lightweight checks run
- `python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py gui/test_stk_note_library_startup_command.py`
- `python gui/test_stk_note_library_startup_command.py`

## VM commands to verify
```bash
git pull
python gui/test_stk_note_library_startup_command.py
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py
streamlit run gui/app.py
```

## Expected VM signs
- Pressing Generate with ready cache renders the clickable guitar.
- Auto-load after cache completion renders once.
- Active current playback uses `current_preview_<hash>`.
- Recent Load uses the selected stored cache and does not regenerate STK.
- Ready/load log lines stop repeating on every rerun.
