# CODEX_HANDOFF.md

## Files changed
- `gui/components/fast_preview/index.html`
- `gui/app.py`
- `.gitignore`
- `docs/diagnostics/classical_sound_determination_report.json`
- `docs/diagnostics/classical_sound_determination_summary.md`
- `CODEX_HANDOFF.md`

## Files inspected
- `gui/app.py`
- `gui/components/fast_preview/index.html`
- `gui/stk_app_audio_service.py`
- `gui/stk_pipeline_defaults.py`
- `gui/body_response_synth.py`
- `gui/body_hybrid_v4_1_identity_space.py`
- `gui/pgsm_stk_parameter_export.py`
- `gui/pgsm_emergency_guitar_demo_engine.py`
- `gui/modal_damping.py`
- `gui/sample_parameters.py`

## Preview source used
- Preferred source is the actual Three.js fast-preview canvas.
- On Save & Sync, the component captures `canvas.toDataURL("image/png")`.
- Python decodes the PNG and stores it locally.

## Preview storage
- Stored under `gui/recent_guitar_previews/recent_guitar_<hash>.png`.
- Directory is ignored in `.gitignore` because previews are local UI/session artifacts.
- Recent Guitar records store `preview_image_path` and `preview_source`.

## Fallback behavior
- If canvas capture or PNG decode fails, no error is shown.
- Recent cards fall back to the existing synthetic SVG preview.
- FIFO, Load behavior, and STK cache reuse are unchanged.

## Sound report
- JSON: `docs/diagnostics/classical_sound_determination_report.json`
- Markdown: `docs/diagnostics/classical_sound_determination_summary.md`
- Report is diagnostic-only; no sound generation code was changed.

## What determines sound
- Direct inputs: woods, length, width, depth, top thickness, hole radius.
- Derived factors: damping/Q/tau, body vs string mix, radiation brightness, bridge mobility, modal gains/participation, identity contrast layer.
- Current website mode uses `stk_body_transfer_final_v1`, aliasing to v4.1 identity contrast `g_30_70`.

## What flattens contrast
- Conservative physical-factor clamps near 1.0.
- RMS/peak normalization.
- Body gain calibration toward a stable body/string ratio.
- Identity residual RMS/audibility guards.
- Fallback physical data if per-guitar audit data is missing.

## Safe later opportunities
- Slight bounded increases to existing body modal gain, string-to-body send, radiation brightness, and modal tau/Q spread.
- Prefer band-limited changes around physically relevant body bands.
- Use existing geometry/ROM/identity features only.

## Unsafe ideas
- Random differences.
- Unbounded EQ/gain or bypassed limiter.
- FEM/ROM solver changes for audio contrast.
- Synthetic detuning unrelated to modal/geometry data.
- Reintroducing BOX/ACOUSTIC.

## CLASSIC risk
- LOW.
- UI preview capture and diagnostic documentation only.
- No FEM/ROM/STK physics, solver, WAV generation, or CLASSIC behavior changed.

## Lightweight checks run
- `python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py gui/test_stk_note_library_startup_command.py`
- `python gui/test_stk_note_library_startup_command.py`
- `python -m json.tool docs/diagnostics/classical_sound_determination_report.json`

## VM verification steps
```bash
git pull
python gui/test_stk_note_library_startup_command.py
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py
python -m json.tool docs/diagnostics/classical_sound_determination_report.json
streamlit run gui/app.py
```

## Expected VM signs
- Save & Sync creates `gui/recent_guitar_previews/recent_guitar_<hash>.png`.
- Recent Guitar cards show the captured fast-preview image.
- If capture is unavailable, cards still show the synthetic fallback.
- Sound report files can be opened directly from `docs/diagnostics/`.
