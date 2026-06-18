# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `CODEX_HANDOFF.md`

## Recent Guitars storage
- Stored in Streamlit `st.session_state["recent_guitars"]`.
- Active playable guitar snapshot stored in `st.session_state["active_recent_guitar_record"]`.
- Session-only persistence; no database or runtime output files.

## FIFO / swap behavior
- Capacity is 3.
- Save/Sync clears the active player state, but first pushes the previous active guitar only if its STK cache is playable.
- Loading a recent guitar removes it from its old slot.
- The previously active guitar moves to slot 1.
- Duplicate hashes/cache paths are removed.
- Oldest entries beyond 3 are dropped.

## Metadata shown
- Top/back woods.
- Length, width, depth.
- Soundhole radius.
- Short parameter hash.
- Each occupied slot has a `Load` button.

## Preview
- Uses a compact placeholder/metadata preview, not a real HTML/SVG capture yet.
- Record structure keeps the design payload so preview capture can be added later.

## Cache / regeneration behavior
- Recent entries require an existing cache with required playable note WAVs.
- Loading a recent guitar activates the existing STK cache directly.
- Loading clears `stk_render_requested`, so it does not start or poll generation.

## CLASSIC-only confirmation
- No BOX or ACOUSTIC UI choices were added.
- Shape payloads remain forced to `Classical`.
- No FEM/ROM/STK physics, solver, synthesis, WAV generation, or CLASSIC data behavior changed.

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

## Expected website signs
- Website remains Classical-only with no shape selector.
- Generate still opens the clickable guitar only when STK cache is ready.
- Recent guitars panel appears under the clickable guitar.
- Up to 3 playable previous guitars appear with Load buttons.
- Loading a recent guitar reuses its cache and does not restart STK generation.
