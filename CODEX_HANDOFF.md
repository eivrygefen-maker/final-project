# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `gui/stk_app_ui.py`
- `gui/stk_app_audio_service.py`
- `gui/test_stk_note_library_startup_command.py`
- `CODEX_HANDOFF.md`

## Root cause of duplicate startup
- Generate still called `request_generate_guitar()`, which could schedule STK note-library generation.
- Save/ROM invalidation also marked the active STK job stale before the new hash was known.
- Same-design reruns could therefore make the existing job look restartable.

## Root cause of clickable guitar not showing
- Generate was coupled to generation startup instead of display intent.
- Failed/incomplete note-library reports (`generated_but_missing_notes`, `status=failed`) were not clearly separated from "ready to display".

## Generate behavior after fix
- Save & Sync starts ROM, then starts STK note-library preparation automatically.
- Generate does not start or restart STK.
- Generate only records display intent via:
- `show_clickable_guitar_requested=True`
- `generate_request_count += 1`
- If STK is still running, the UI waits and auto-loads the player when ready.
- If STK was ready before Generate, the player loads immediately after Generate.

## Debounce/idempotency guard
- Implemented in `gui/stk_app_audio_service.py::start_background_note_library_job()`.
- Same-hash `running`, `ready`, `partial_ready`, or `failed` status returns existing state and does not call `subprocess.Popen()`.
- The guard now runs before preview directory creation or process launch.

## Failed/missing notes handling
- `gui/stk_app_ui.py::request_generate_guitar()` returns `stk_failed` for failed readiness.
- The player remains hidden when status is failed or cache validation is incomplete.
- The UI shows an error instead of presenting the clickable guitar as ready.

## CLASSIC-only confirmation
- No BOX or ACOUSTIC UI choices were reintroduced.
- Website remains Classical/sample_000 only.
- No FEM/ROM/STK physics, solver settings, thresholds, or generated audio logic changed.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py gui\stk_app_audio_service.py gui\test_stk_note_library_startup_command.py`
- `python gui\test_stk_note_library_startup_command.py`
- `rg -n "Generate Sound.*build|build your guitar audio|rebuild audio|starts only from Generate|intentionally disabled|--instrument" gui\app.py gui\stk_app_ui.py gui\stk_app_audio_service.py gui\test_stk_note_library_startup_command.py tools\run_app_stk_note_library_classical_sample_000.sh tools\build_app_stk_note_library.py`
- `rg -n 'Box \(debug\)|Dreadnought / Acoustic|<option value="Box"|<option value="Dreadnought"' gui\app.py gui\components\fast_preview\index.html`

## VM commands to verify
```bash
git pull
python gui/test_stk_note_library_startup_command.py
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py
streamlit run gui/app.py
```

## Expected VM signs
- One `APP_STK_RENDER_MODE ... hash=...` line per saved design hash.
- Clicking Generate does not print another startup line for the same hash.
- If cache is incomplete, the clickable guitar stays hidden and an error is shown.
- If cache is ready, Generate opens the clickable guitar immediately.
