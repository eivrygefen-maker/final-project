# CODEX_HANDOFF.md

## Files changed
- `gui/stk_app_audio_service.py`
- `gui/test_stk_note_library_startup_command.py`
- `tools/run_app_stk_note_library_classical_sample_000.sh`
- `CODEX_HANDOFF.md`

## Where `--instrument` was found
- `gui/stk_app_audio_service.py`
  - background website STK note-library startup command.
- `tools/run_app_stk_note_library_classical_sample_000.sh`
  - classical helper script.

## Exact fix
- Removed unsupported `--instrument classical` from the note-library command.
- Added supported startup args instead:
- `--shape-type Classical`
- `--sample-id sample_000`
- Added `build_note_library_startup_command()` so the command can be tested without launching STK.

## CLASSIC-only confirmation
- No BOX or ACOUSTIC UI choices were reintroduced.
- Website startup still targets Classical/sample_000.
- No FEM/ROM/STK physics, generated audio behavior, solver settings, thresholds, or runtime outputs changed.

## Lightweight checks run
- `python -m py_compile gui\stk_app_audio_service.py gui\test_stk_note_library_startup_command.py tools\build_app_stk_note_library.py`
- `python gui\test_stk_note_library_startup_command.py`
- `rg -n -e "--instrument" gui tools\run_app_stk_note_library_classical_sample_000.sh tools\build_app_stk_note_library.py`

## VM commands to verify
```bash
git pull
python gui/test_stk_note_library_startup_command.py
python -m py_compile gui/stk_app_audio_service.py tools/build_app_stk_note_library.py
streamlit run gui/app.py
```

## Expected VM signs
- No `unrecognized arguments: --instrument classical` error.
- Interactive guitar note library starts loading again.
- Website remains Classical-only, with no BOX/ACOUSTIC shape selector.
