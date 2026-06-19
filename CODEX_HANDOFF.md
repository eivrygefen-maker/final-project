# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `gui/pgsm_stk_parameter_export.py`
- `gui/test_pgsm_stk_parameter_export.py`
- `.gitignore`
- `CODEX_HANDOFF.md`

## Recent Load root cause/fix
- Root cause: repeated Recent Load swaps could restore design/Gmsh state while leaving player display intent latched to stale or hidden activation state.
- `apply_recent_record_to_design()` now accepts `cache_playable` and sets ready/show-player flags only when the saved cache has required WAVs.
- `load_recent_guitar_at_index()` now restores Gmsh/design first, validates required WAVs, activates the player cache, checks activation payload is ready, then latches player state.
- It clears `stk_render_requested`, `stk_render_requested_hash`, and `stk_generate_intent_hash`.

## STK regeneration
- Recent Load does not call STK startup or schedule generation.
- Recent Load reuses the saved cache path from the recent record.
- FIFO max 3 and hash/cache dedupe behavior remain unchanged.

## Sound contrast change
- Implemented a small bounded Classical STK parameter tuning pass.
- Constant: `CLASSIC_AUDIBLE_IDENTITY_CONTRAST = 0.12` in `gui/pgsm_stk_parameter_export.py`.
- Applies only to `demo_version == "v4_10_samples"` used by the Classical app note cache.
- Uses existing computed/design-driven factors only.

## Exact factors/ranges
- Direct string gain: reduced about 3% after modestly preserving existing spread; clamped `0.66..1.34`.
- Body modal gain: lifted about 5.4% and spread slightly; clamped `0.72..1.48`.
- String-to-body send scale: lifted about 4.2% and spread slightly; clamped `0.74..1.36`.
- `radiation_brightness_factor` spread increased by `1.072x`; clamped `0.78..1.28`.
- `top_damping_factor` spread increased by `1.054x`; clamped `0.78..1.32`.
- Metadata is written under `perceptual_calibration.classic_audible_identity_contrast`.

## Disable/revert
- Set `CLASSIC_AUDIBLE_IDENTITY_CONTRAST = 0.0` to disable the tuning layer.
- Existing generated caches are unchanged until rebuilt on VM.

## CLASSIC risk
- UI risk: LOW-MEDIUM, limited to Recent Load session/player restore.
- Audio risk: MEDIUM, future Classical STK cache rebuilds will sound slightly more body-forward.
- No FEM/ROM solver, ROM data, limiter, randomization, BOX, or ACOUSTIC changes.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_ui.py gui\stk_app_audio_service.py gui\pgsm_stk_parameter_export.py gui\test_pgsm_stk_parameter_export.py tools\build_classic_stk_contrast_diagnostic.py`
- `python -m unittest gui.test_pgsm_stk_parameter_export.TestPgsmStkParameterExport.test_v4_10_samples_export gui.test_pgsm_stk_parameter_export.TestPgsmStkParameterExport.test_v4_classic_audible_identity_contrast_metadata gui.test_pgsm_stk_parameter_export.TestPgsmStkParameterExport.test_no_python_wav_synthesis_in_module`
- `python gui\test_stk_note_library_startup_command.py`
- Full `python gui\test_pgsm_stk_parameter_export.py` hit sandbox temp-directory permission on an existing write-roundtrip test.

## VM commands
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py gui/pgsm_stk_parameter_export.py
python gui/test_stk_note_library_startup_command.py
python tools/build_classic_stk_contrast_diagnostic.py --note A3 --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
```
