# CODEX_HANDOFF.md

## Root cause: missing/quiet contrast sample
- Strongest diagnosis: the contrast diagnostic accepted any readable WAV, so a repeated middle cache with a missing/zero/near-silent/corrupt note could enter the concatenated WAV as if playable.
- Exact bad cache still needs VM data, but invalid clips are now rejected before concatenation and reported with reasons.

## Invalid-clip handling added
- `tools/build_classic_stk_contrast_diagnostic.py` rejects missing/read-error, zero-length, invalid sample-rate, non-finite, low-RMS, and low-peak clips.
- New thresholds: `--min-valid-rms` default `1e-5`, `--min-valid-peak` default `1e-4`.
- JSON/Markdown now include `valid_clip_count`, `invalid_clip_thresholds`, and `rejected_clips`.
- Valid clips keep their original order after rejected clips are removed.

## Website default preset
- New website default STK contrast preset: `strong`.
- Internal presets still available: `off`, `conservative`, `strong`, `aggressive`.
- No website preset control was added.

## Default preset code path
- `gui/stk_app_audio_service.py`: `DEFAULT_WEBSITE_CLASSIC_CONTRAST_PRESET = "strong"`.
- Website flow: `gui/app.py` Save & Sync schedules STK -> `schedule_stk_after_rom()` -> `start_background_note_library_job()` -> `build_note_library_startup_command()`.
- Startup command now passes `--contrast-preset strong` to `tools/build_app_stk_note_library.py`.
- Cache spec/report metadata records `classic_contrast_preset=strong` and strength.
- Cache spec compatibility is strict, so older conservative caches are not silently relabeled as strong.

## No physics changes
- No FEM/ROM solver changes.
- No ROM data changes.
- No core STK synthesis algorithm changes.
- Existing limiter/normalization/clamps remain in place.

## Recent Guitar status
- No Recent Guitars UI hits remain in `gui/`.
- Active Generate flow now forces the old saved/FIFO stack branch disabled and activates only the current preview cache.
- Old helper functions remain as dead compatibility code; they are not called from `gui/app.py`.
- No BOX/ACOUSTIC UI selector was reintroduced.

## Normal-flow stability findings
- Save & Sync still resets player request/stale state and schedules background STK for the current hash.
- Generate still requests display only; it does not start STK generation.
- Ready cache activation uses the current `current_preview_<hash>` cache path.
- Player state latches through `active_player_hash`, `active_player_cache_dir`, `loaded_player_hash`, and `loaded_player_cache_dir`.
- Code inspection found no active Recent Load path.

## UI/design review
- Must fix before conference: none found by code inspection.
- Optional polish later: remove dead saved-stack helper/messages and manually check loading text spacing around Save & Sync / Generate.

## Files changed
- `tools/build_classic_stk_contrast_diagnostic.py`
- `gui/stk_app_audio_service.py`
- `tools/build_app_stk_note_library.py`
- `gui/stk_app_ui.py`
- `gui/test_stk_note_library_startup_command.py`
- `CODEX_HANDOFF.md`

## Lightweight checks run
- `python -m py_compile tools\build_classic_stk_contrast_diagnostic.py tools\build_app_stk_note_library.py gui\stk_app_audio_service.py gui\stk_app_ui.py gui\test_stk_note_library_startup_command.py`
- `python tools\build_classic_stk_contrast_diagnostic.py --help`
- `python tools\build_app_stk_note_library.py --help`
- `python gui\test_stk_note_library_startup_command.py`
- `git diff --check`

## VM verification commands
```bash
git pull
python -m py_compile tools/build_classic_stk_contrast_diagnostic.py tools/build_app_stk_note_library.py gui/stk_app_audio_service.py gui/stk_app_ui.py
python gui/test_stk_note_library_startup_command.py
python tools/build_classic_stk_contrast_diagnostic.py --note A2 --comparison-mode all --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
python tools/build_classic_stk_contrast_diagnostic.py --note A3 --comparison-mode all --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
python tools/build_classic_stk_contrast_diagnostic.py --note E5 --comparison-mode all --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```

## VM manual checks
- Open the website, confirm only Classical is visible.
- Save & Sync a design, then Generate after STK is ready.
- Confirm clickable guitar appears for the current design and no Recent Guitars panel appears.
- Check the new note-library report/cache spec for `classic_contrast_preset: strong`.
- In diagnostic JSON/Markdown, inspect `rejected_clips` for the repeated missing/quiet middle sample.
