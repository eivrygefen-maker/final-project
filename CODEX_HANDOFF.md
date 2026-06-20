# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `gui/pgsm_stk_parameter_export.py`
- `gui/stk_app_audio_service.py`
- `gui/test_pgsm_stk_parameter_export.py`
- `tools/build_app_stk_note_library.py`
- `CODEX_HANDOFF.md`

## Recent Guitars removal
- Recent Guitars UI section is removed.
- Recent Load buttons and FIFO/swap behavior are removed from active website flow.
- Recent-only session state, snapshot persistence, preview SVG helpers, load helpers, and FIFO push hooks were removed from `gui/app.py`.
- Existing local audio/cache/preview files are not deleted.

## Retained normal flow
- Save & Sync still drives current design, Gmsh display mesh, ROM/STK preparation.
- Generate still controls current clickable guitar display.
- Current active player session state remains intact.
- Website remains CLASSIC-only.

## Contrast preset design
- Website default remains `conservative`.
- `CLASSIC_AUDIBLE_IDENTITY_CONTRAST = 0.12`.
- Presets:
- `off = 0.0`
- `conservative = 0.12`
- `strong = 0.25`
- `aggressive = 0.35`
- Active preset is written into STK params/cache reports as `classic_contrast_preset`.

## Factors and clamps
- Direct string gain: lower direct dominance; clamped `0.58..1.40`.
- Body modal gain: stronger body contribution; clamped `0.70..1.62`.
- String-to-body send: stronger coupling/send; clamped `0.72..1.50`.
- Radiation brightness spread: stronger existing spread; clamped `0.74..1.34`.
- Top damping/Q-tau proxy spread: stronger existing spread; clamped `0.74..1.38`.
- No random offsets, fake detuning, unrelated EQ, limiter bypass, FEM/ROM solver, or ROM-data changes.

## How to select preset
- VM CLI:
```bash
python tools/build_app_stk_note_library.py --sample-id sample_000 --shape-type Classical --parameter-hash contrast_strong_sample_000 --cache-dir audio/app_stk_note_cache/classical/contrast_strong_sample_000 --contrast-preset strong --force
```
- Use `--contrast-preset conservative`, `strong`, `aggressive`, or `off`.
- Non-conservative presets are included in the cache spec hash to avoid accidental conservative-cache reuse.

## A3 comparison commands
```bash
git pull
python -m py_compile gui/app.py gui/stk_app_audio_service.py gui/pgsm_stk_parameter_export.py tools/build_app_stk_note_library.py
for s in sample_000 sample_001 sample_002; do python tools/build_app_stk_note_library.py --sample-id "$s" --shape-type Classical --parameter-hash "contrast_aggressive_$s" --cache-dir "audio/app_stk_note_cache/classical_contrast_aggressive/$s" --contrast-preset aggressive --force; done
python tools/build_classic_stk_contrast_diagnostic.py --note A3 --cache-root audio/app_stk_note_cache/classical_contrast_aggressive --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
```

## CLASSIC risk
- UI risk: LOW; Recent stack removed, current active guitar flow unchanged.
- Audio risk: LOW for website default; it remains conservative.
- Audio experiment risk: MEDIUM for explicit `strong/aggressive` VM cache rebuilds only.
- Rollback: set preset to `conservative` or `off`, or revert the STK preset patch.

## Lightweight checks run
- `python -m py_compile gui\app.py gui\stk_app_audio_service.py gui\pgsm_stk_parameter_export.py gui\test_pgsm_stk_parameter_export.py tools\build_app_stk_note_library.py tools\build_classic_stk_contrast_diagnostic.py`
- `python tools\build_app_stk_note_library.py --help`
- `python gui\test_stk_note_library_startup_command.py`
- `python -m unittest gui.test_pgsm_stk_parameter_export.TestPgsmStkParameterExport.test_v4_10_samples_export gui.test_pgsm_stk_parameter_export.TestPgsmStkParameterExport.test_v4_classic_audible_identity_contrast_metadata gui.test_pgsm_stk_parameter_export.TestPgsmStkParameterExport.test_v4_aggressive_contrast_is_explicit_and_bounded gui.test_pgsm_stk_parameter_export.TestPgsmStkParameterExport.test_no_python_wav_synthesis_in_module`
