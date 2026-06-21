# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `gui/components/guitar_player/index.html`
- `gui/components/fast_preview/index.html`
- `CODEX_HANDOFF.md`

## Gmsh display default
- Default view is now clean material view with mesh edges off.
- Added compact checkbox: `Show Engineering Mesh`.
- Checkbox only toggles PyVista `show_edges` and viewer key; it does not touch FEM/ROM/STK state.
- Current display geometry is unchanged.

## Display mesh density
- Display mesh density remained unchanged.
- No Gmsh/display/FEM mesh generation settings were modified.
- No solver mesh or physics mesh settings were touched.

## Seam/material coloring
- `display_mesh.msh` physical triangle tags are preserved into PyVista cell data.
- Flat per-cell colors now prefer physical groups:
  top, back, ribs/seam, and soundhole.
- If tags are unavailable, the previous spatial top/back fallback still applies.
- Clean view uses flat cell coloring and no triangle edge overlay.

## Generate area cleanup
- Removed the `GENERATE SOUND` heading above the button.
- Generate Sound button text and behavior are unchanged.

## Typography
- Main Streamlit step headings increased from `3.6rem` to `4.15rem`.
- Design Studio title increased from `1.25rem` to `1.44rem`.
- Guitar Player title increased from `2.75rem` to `3.16rem`.
- Body text and controls were not intentionally enlarged.

## STK loading spinner
- Added a local CSS-only spinner in `gui/components/guitar_player/index.html`.
- Spinner appears only for existing `player.status == "building"`.
- It displays beside `Building note cache...`.
- No Streamlit polling, rerun, audio, cache, or player behavior changed.

## Confirmations
- CLASSIC-only remains unchanged.
- No Recent Guitars were added.
- No FEM/ROM/STK physics, dimensions, cache generation, or player audio behavior changed.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py`
- Text scan for removed `GENERATE SOUND` heading and old player labels.
- `git diff --check`

## VM verification steps
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_ui.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Save & Sync a guitar and inspect the full model.
- Confirm default Gmsh view has no triangle edges.
- Toggle `Show Engineering Mesh` and confirm edges appear/disappear only visually.
- Confirm top/back/ribs/soundhole colors are clean and not blended by triangles.
- Confirm `GENERATE SOUND` heading is gone but the button remains.
- Click Generate while STK is building and confirm the small spinner appears with `Building note cache...`.
