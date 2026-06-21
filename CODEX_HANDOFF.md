# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `gui/components/fast_preview/index.html`
- `gui/components/guitar_player/index.html`
- `CODEX_HANDOFF.md`

## Exact text removed/replaced
- Removed heading above the button:
  `GENERATE SOUND`
- Replaced fast-preview badge:
  `Live 3D · Luthier blueprint` -> `LIVE PREVIEW`
- Removed far-right fretboard text:
  `Nut / open`
- Kept the headstock spacer element so fretboard/open-string layout remains unchanged.

## Confirmations
- Generate Sound button text and behavior unchanged.
- Open-string note labels remain unchanged.
- Fretboard click/audio behavior unchanged.
- Gmsh, HTML body proportions, FEM/ROM/STK, cache logic, and player behavior unchanged.
- CLASSIC-only remains unchanged.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py`
- Targeted text scan for removed/replaced strings.
- `git diff --check`

## VM verification steps
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_ui.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Confirm no `GENERATE SOUND` heading appears above the button.
- Confirm fast-preview badge reads `LIVE PREVIEW`.
- Confirm `Nut / open` is gone from the fretboard right side.
- Confirm open-string labels and note clicks still work.
