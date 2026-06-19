# CODEX_HANDOFF.md

## Root cause: geometry restore
- Recent records were not guaranteed to store the exact full studio payload used by the iframe.
- Recent Load updated `_fast_preview_geom`, but stale iframe events could still be accepted on the next rerun and overwrite the loaded geometry.
- Result: woods/materials could change while length/width/depth/top thickness/hole radius drifted back toward the previous active design.

## Root cause: player restore
- Recent Load could set display intent before confirming the activated player payload was actually ready.
- A hidden/stale activation payload could remain latched, so the clickable guitar did not auto-load.

## Exact fix
- `build_recent_guitar_record()` now stores a sanitized full `studio_payload` for every record.
- `set_active_recent_record_from_current_design()` passes the current `_fast_preview_geom` into the record.
- Added `force_active_studio_payload()` to replace `_fast_preview_geom`, `fast_preview_geom`, `_geom`, woods, and cached iframe props together.
- Added a short `_forced_studio_payload_fp` guard so stale component events cannot overwrite a just-loaded Recent Guitar.
- Recent Load still rebuilds the display Gmsh mesh from the loaded payload and only shows the player after cache/player validation passes.

## Recent Guitars status
- Recent Guitars remains enabled.
- No feature flag disable was needed.

## STK regeneration
- Recent Load does not start STK generation.
- It clears `stk_render_requested`, `stk_render_requested_hash`, and `stk_generate_intent_hash`.
- It reuses the saved cache path from the recent record.

## Full geometry restore
- The saved/restored payload includes:
- `length`
- `width`
- `depth`
- `top_thickness`
- `hole_radius`
- `top_wood_id`
- `back_wood_id`
- iframe metadata/bounds/colors/templates when present

## CLASSIC-only confirmation
- Website remains CLASSIC-only.
- No BOX/ACOUSTIC UI or behavior was reintroduced.
- No FEM/ROM/STK physics, solver behavior, ROM data, or WAV generation changed in this task.
- Sound contrast diagnostic was inspected only conceptually; no further audio tuning was made in this task.

## Files changed
- `gui/app.py`
- `CODEX_HANDOFF.md`

## Existing uncommitted files from prior tasks
- `.gitignore`
- `gui/pgsm_stk_parameter_export.py`
- `gui/test_pgsm_stk_parameter_export.py`

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py gui\stk_app_audio_service.py`
- `python gui\test_stk_note_library_startup_command.py`
- A direct app unit test was not possible in Codex because this local Python env lacks `streamlit`.

## VM verification commands
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py
python gui/test_stk_note_library_startup_command.py
streamlit run gui/app.py
```

## VM manual check
- Create/generate two playable guitars with visibly different L/W/D/hole radius.
- Load recent B, then recent A.
- Confirm fast preview and Gmsh proportions both change fully, not only woods.
- Confirm clickable guitar appears immediately after each Recent Load.
- Confirm terminal does not show a new STK note-library startup after Recent Load.
