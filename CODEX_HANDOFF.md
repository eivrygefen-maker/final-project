# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `CODEX_HANDOFF.md`

## Recent Load root cause
- Recent Load swapped FIFO records and activated the STK cache.
- It restored `_fast_preview_geom`, but did not force the saved display/Gmsh mesh view to the loaded design.
- It also did not restore the active recent preview image path/source into session state.
- Result: player/cache state and visual design state could drift after loading a recent guitar.

## Recent Load fix
- `apply_recent_record_to_design()` now restores:
- saved studio payload
- active geometry/material session state
- active parameter hash/cache path
- preview image path/source
- player-display intent
- `stk_render_requested=False`
- Added `restore_recent_display_mesh()` to regenerate only the display Gmsh mesh for the loaded design.
- FIFO dedupe now removes both matching hash and matching cache path before inserting the previous active guitar.

## Preview/player restore
- The fast-preview component receives the loaded `studio_payload` on rerun through `_fast_preview_geom`.
- The Gmsh display view is regenerated/pinned to the loaded guitar using default CLASSIC display settings.
- `activate_stk_guitar_for_player()` and `apply_stk_activation_to_session()` set the active player payload/key/cache.
- If the recent cache is playable, `show_clickable_guitar_requested=True` shows the player immediately.

## STK regeneration
- Recent Load does not call `schedule_stk_after_rom()`.
- Recent Load does not set `stk_render_requested`.
- Existing playable cache is reused directly.

## Sound contrast
- Plan-only; no sound generation changes implemented.
- Rationale: CLASSIC is frozen, and audible contrast changes need VM listening/cache validation.
- Safe first pass later:
- Increase existing body-modal/radiation/decay spreads by about 10-15%.
- Prefer bounded factors already present: `body_modal_gain`, `string_to_body_send_scale`, `radiation_brightness_factor`, modal `Q/tau` damping spread.
- Keep limiter/normalization intact.
- Use only existing geometry, wood/material, bridge mobility, modal participation, and identity-vector data.

## Avoid
- No random offsets.
- No fake detuning unrelated to modal data.
- No FEM/ROM solver or ROM data changes.
- No bypassing limiter/normalization.
- No BOX/ACOUSTIC UI reintroduction.

## CLASSIC risk
- Recent Load fix: LOW-MEDIUM UI risk because Load now refreshes display Gmsh mesh.
- Sound contrast: LOW, plan-only.
- No FEM/ROM/STK physics, solver, WAV generation, or cache-generation behavior changed.

## Lightweight checks run
- `python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py gui/test_stk_note_library_startup_command.py`
- `python gui/test_stk_note_library_startup_command.py`

## VM verification commands
```bash
git pull
python gui/test_stk_note_library_startup_command.py
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py
streamlit run gui/app.py
```

## Expected VM signs
- Load on a Recent Guitar updates the design studio preview after rerun.
- Load refreshes the displayed Gmsh mesh to the selected guitar.
- Clickable guitar appears immediately for a playable recent cache.
- Terminal should not show a new STK note-library generation for Recent Load.
- Recent FIFO remains max 3 with no duplicate hash/cache entries.
