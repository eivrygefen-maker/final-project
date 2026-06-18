# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `CODEX_HANDOFF.md`

## Mini guitar preview generation
- Recent Guitar records now store `preview_svg`.
- SVG is generated deterministically from saved session design metadata.
- No browser screenshot capture, `html2canvas`, external tools, or assets.
- Older session records without `preview_svg` generate it lazily at render time.

## Parameters affecting preview
- `top_wood` controls body/top color.
- `back_wood` controls side/neck/accent color.
- `length` affects body height.
- `width` affects body width.
- `depth` affects waist/body fullness.
- `hole_radius` affects soundhole/rosette size.

## Missing metadata handling
- Falls back to sanitized stored studio payload if metadata is absent.
- Falls back to default classical guitar dimensions/colors if fields are missing.

## Recent Guitars behavior
- FIFO/session-state behavior remains unchanged.
- `Load` still activates the existing ready STK cache directly.
- Loading a recent guitar still clears `stk_render_requested`, so it does not start generation.
- Recent entries still require an existing playable note cache.

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
- Recent Guitars cards appear under the clickable guitar.
- Occupied cards show a compact stylized guitar preview, metadata, short hash, and `Load`.
- Different woods/dimensions visibly affect the mini preview.
- Loading a recent guitar reuses its cache and does not restart STK generation.
