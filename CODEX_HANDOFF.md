# CODEX_HANDOFF.md

## Root cause
- The previous waiting-state patch cleared the remembered Generate Sound request once STK became ready.
- That left the cache ready but the player closed, so the user had to click Generate Sound again.
- The UI also still had ready-state notifications in the Generate/watch flow.

## Files changed
- `gui/app.py`
- `gui/stk_app_ui.py`
- `gui/stk_app_audio_service.py`
- `gui/test_stk_note_library_startup_command.py`
- `CODEX_HANDOFF.md`

## One-click Generate behavior
- Generate Sound still does not start or restart STK generation.
- If the cache is already playable, Generate Sound opens the clickable guitar immediately.
- If the cache is not ready, Generate Sound records `stk_render_requested` for the current hash.
- When polling later sees that same hash become playable, it calls the existing ready-cache activation path and opens the player automatically.
- If the user never clicked Generate Sound, no player auto-opens.

## User-facing messages
- Removed user-facing `Guitar sound is ready` notifications from the active Generate/watch path.
- The shared user-facing status helper now returns no ready caption for `ready`.
- While STK is still preparing, the only waiting message remains:
  `Building guitar sound with STK... This may take a few minutes.`
- Explicit failure still uses:
  `Sound preparation did not finish. Save & Sync to retry.`

## Confirmations
- No GMSH, FEM, ROM, STK rendering, cache-generation, audio, strong preset, or HTML preview behavior changed.
- Generate Sound remains display-only and does not start computation.
- No Recent Guitars behavior was added.
- CLASSIC-only remains unchanged.
- No second Generate click is required for the same design after an early click.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py gui\stk_app_audio_service.py gui\test_stk_note_library_startup_command.py`
- `python gui\test_stk_note_library_startup_command.py`
- Targeted text scan for removed ready-state notification/action strings.
- `git diff --check`

## VM verification steps
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py gui/test_stk_note_library_startup_command.py
python gui/test_stk_note_library_startup_command.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Save & Sync a Classical guitar.
- Click Generate Sound while STK is still preparing.
- Confirm the neutral STK preparation message appears.
- Wait until STK finishes.
- Confirm the clickable guitar opens automatically without a second Generate Sound click.
- Confirm no `Guitar sound is ready` notification appears.
