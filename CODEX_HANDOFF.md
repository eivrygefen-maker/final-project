# CODEX_HANDOFF.md

## Files changed
- `gui/stk_app_audio_service.py`
- `gui/test_stk_note_library_startup_command.py`
- `CODEX_HANDOFF.md`

## Root cause of final WAV count = 0
- `parallel_batch` rendered worker WAVs correctly into `.render_tmp/<hash>/worker_*`.
- Worker outputs were merged into `.render_tmp/<hash>/staging`.
- The code audited `staging` before creating per-position WAV aliases and before final cache promotion.
- That made validation fail early, so `current_preview_<hash>` kept only `.cache_spec.json`.

## Finalization/promotion fix
- Added `finalize_parallel_staging_cache()`.
- It promotes `staging/*.wav` into the final preview cache.
- It preserves sharp note filenames such as `F#2.wav`, `A#4.wav`, `C#5.wav`.
- It creates/refreshes per-position aliases in the final cache.
- It writes cache spec metadata with final-cache counts.
- `render_notes_parallel_batch()` now promotes first, then validates/counts the final output directory.

## Duplicate startup/logging
- The repeated `APP_STK_RENDER_MODE ...` line was duplicate logging:
- parent startup printed it before `subprocess.Popen()`;
- child builder printed the same line inside `build_note_library()`.
- Removed the parent print so the line now represents the actual builder process.
- Same-hash idempotency from the prior fix remains in place before process launch.

## CLASSIC-only confirmation
- No BOX or ACOUSTIC UI choices were reintroduced.
- STK note-library path remains Classical/sample_000.
- No FEM/ROM/STK physics or synthesis algorithm changed.

## Lightweight checks run
- `python -m py_compile gui\stk_app_audio_service.py gui\test_stk_note_library_startup_command.py tools\build_app_stk_note_library.py`
- `python gui\test_stk_note_library_startup_command.py`
- `rg -n -e "APP_STK_RENDER_MODE" -e "--instrument" gui\stk_app_audio_service.py gui\test_stk_note_library_startup_command.py tools\build_app_stk_note_library.py tools\run_app_stk_note_library_classical_sample_000.sh`

## VM commands to verify
```bash
git pull
python gui/test_stk_note_library_startup_command.py
python -m py_compile gui/stk_app_audio_service.py tools/build_app_stk_note_library.py
streamlit run gui/app.py
```

## Expected VM signs
- Worker WAVs are promoted into `audio/app_stk_note_cache/classical/current_preview_<hash>/`.
- Final cache contains note WAVs, including sharp-note filenames.
- `generated_note_count` and `note_wav_count` are nonzero.
- `missing_required_notes` is not all 44 when staging contains the required WAVs.
- Only one `APP_STK_RENDER_MODE ... hash=...` line appears per builder process.
