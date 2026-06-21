# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `gui/stk_app_ui.py`
- `gui/stk_app_audio_service.py`
- `gui/test_stk_note_library_startup_command.py`
- `CODEX_HANDOFF.md`

## Exact statuses/messages removed
- Removed the misleading player-side error banner:
  `Guitar sound cache is incomplete. Please Save & Sync again before opening the player.`
- Replaced failed/running STK wording with one neutral waiting message while preparing:
  `Building guitar sound with STK... This may take a few minutes.`
- Replaced real failed wording with:
  `Sound preparation did not finish. Save & Sync to retry.`
- Removed automatic player opening after a background STK job becomes ready.

## Final user-facing flow
- Save & Sync still starts the background Classical STK note-cache work.
- If Generate Sound is clicked while STK is preparing, the UI shows only the neutral preparation message.
- When STK finishes, the player does not auto-open.
- Generate Sound opens the player only when a playable cache exists.
- A real error is shown only when the current hash has explicitly failed and no active job is still running.

## Status precedence
- Added UI precedence helper:
  `preparing > ready > explicit failed > idle`
- Active/running jobs now take priority over stale failed or incomplete report state.

## Confirmations
- No FEM/ROM solver changes.
- No STK rendering/cache-generation changes.
- No player audio changes.
- No Gmsh changes.
- CLASSIC-only remains unchanged.
- Recent Guitars remain removed.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py gui\stk_app_audio_service.py gui\test_stk_note_library_startup_command.py`
- `python gui\test_stk_note_library_startup_command.py`
- Targeted text scan for removed misleading messages and stale auto-load wording.
- `git diff --check`

## VM verification steps
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py gui/test_stk_note_library_startup_command.py
python gui/test_stk_note_library_startup_command.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Save & Sync a Classical guitar and click Generate Sound while STK is still rendering.
- Confirm only this message appears: `Building guitar sound with STK... This may take a few minutes.`
- Confirm no `failed`, `cache incomplete`, or retry banner appears during active rendering.
- After STK finishes, confirm the player does not auto-open.
- Click Generate Sound again and confirm the clickable guitar opens from the ready cache.
