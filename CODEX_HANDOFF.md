# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `gui/stk_app_ui.py`
- `gui/components/guitar_player/index.html`
- `gui/components/fast_preview/index.html`
- `CODEX_HANDOFF.md`

## Sound/Generate cleanup
- Kept one blue preparation message:
  `Building guitar sound with STK... This may take a few minutes.`
- Removed duplicate blue auto-load wording:
  `The player will load automatically when ready.`
- Removed:
  `Sound is preparing in the background. Click Generate Sound when you want to open the player.`
- Removed saving/comparison-stack wording from Generate responses and button help.
- Replaced Step 3 guidance with:
  `Click Generate Sound to open and play your guitar. If the sound is still preparing, the player will appear automatically when it is ready.`
- Generate Sound behavior is unchanged.

## Guitar Player cleanup
- Removed the `Ready · 44 notes` status badge from the player header.
- Removed the `Play all notes preview` button.
- Added null guards for the removed status/button elements.
- Clickable fretboard behavior, note aliases, cache paths, preloading, and audio playback logic were not changed.
- Space below/around the fretboard is now cleaner for future melody/chord controls.

## Design Studio cleanup
- Removed `User (ROM)` label.
- Removed subtitle:
  `ROM/LHS parameters only — instant Three.js preview. Save & Sync pushes to Gmsh/PyVista.`
- Kept the `Design Studio` title and all active design controls.
- No parameter sliders/selectors were removed.

## CLASSIC-only / Recent status
- CLASSIC-only behavior remains unchanged.
- No BOX/ACOUSTIC UI option was reintroduced.
- No Recent Guitars UI or load flow was added.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py`
- Removed-text scan for the requested labels/messages.
- `git diff --check`

## VM verification steps
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_ui.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Open the website and confirm Design Studio shows only title + controls.
- Save & Sync, then confirm no background preparation caption appears before Generate.
- Click Generate Sound while STK is preparing; confirm only one blue prep message appears.
- Confirm the player has no `Ready · 44 notes` label and no `Play all notes preview` button.
- Confirm clicking fretboard notes still plays from the current cache.
