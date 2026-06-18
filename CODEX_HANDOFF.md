# CODEX_HANDOFF.md

## Files changed
- `.gitignore`
- `gui/app.py`
- `gui/app_stk_instrument.py`
- `gui/stk_app_audio_service.py`
- `gui/components/fast_preview/index.html`
- `CODEX_HANDOFF.md`

## What was hidden/disabled
- Removed the visible shape selector from the website Design Studio.
- The app now opens directly as the validated Classical guitar design surface.
- BOX and ACOUSTIC/Dreadnought are not exposed as user-facing choices.
- Stale browser/component payloads are normalized back to `Classical`.

## CLASSIC-only guard
- `gui/app.py`
  - `SHAPE_OPTIONS = ("Classical",)` and `rom_namespace()` always returns `classic`.
  - `sanitize_studio_payload()` forces `shape_type` to `Classical`.
- `gui/app_stk_instrument.py`
  - Any website shape request maps to `ROM/classic`.
  - Any default/reference sample request maps to `sample_000`.
- `gui/stk_app_audio_service.py`
  - default website sample discovery lists Classical samples only.
- `gui/components/fast_preview/index.html`
  - shape input is hidden and fixed to `Classical`.

## Internal comment
- Added comments stating BOX and ACOUSTIC are experimental and frozen for now.

## CLASSIC runtime behavior
- CLASSIC solver, thresholds, simulation data, generated outputs, FEM/ROM/STK audio logic, and production runtime artifacts were not changed.
- This only changes website/UI shape exposure and website request guarding.

## .gitignore changes
- Added ignores for local shape batch runtime files:
- `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index/*/*_fom_overnight_batch_*.log`
- `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index/*/*_fom_overnight_batch_*_summary.json`
- `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index/*/*_fom_overnight_batch_*_results.jsonl`

## Lightweight checks run
- `python -m py_compile gui\app.py gui\app_stk_instrument.py gui\stk_app_audio_service.py gui\stk_app_ui.py`
- `python -c "import sys; sys.path.insert(0, 'gui'); import app_stk_instrument as m; assert m.rom_shape_namespace('Box') == 'classic'; assert m.rom_shape_namespace('Dreadnought') == 'classic'; assert m.default_sample_id_for_shape('Box') == 'sample_000'; assert m.reference_sample_id_for('box_sample_000') == 'sample_000'; assert m.shape_type_label_from_sample_id('box_sample_000') == 'classic'; print('PASS classic-only website STK guards')"`
- `rg -n 'Box \(debug\)|Dreadnought / Acoustic|<option value="Box"|<option value="Dreadnought"' gui\app.py gui\components\fast_preview\index.html`
- Local Streamlit launch was attempted but this Codex Python has no `streamlit` module installed.

## Quick website check commands
```bash
python -m py_compile gui/app.py gui/app_stk_instrument.py gui/stk_app_audio_service.py gui/stk_app_ui.py
streamlit run gui/app.py
```

## Expected quick check signs
- Website opens directly to the Classical guitar design surface.
- No shape dropdown is visible.
- Generate Sound uses the Classical/sample_000 path.
- No BOX or ACOUSTIC generation option is shown.
