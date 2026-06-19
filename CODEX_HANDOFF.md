# CODEX_HANDOFF.md

## Files changed
- `gui/app.py`
- `CODEX_HANDOFF.md`

## Files inspected
- `gui/app.py`
- `gui/stk_app_audio_service.py`
- `gui/stk_pipeline_defaults.py`
- `gui/body_response_synth.py`
- `gui/body_hybrid_v4_1_identity_space.py`
- `gui/pgsm_stk_parameter_export.py`
- `gui/pgsm_emergency_guitar_demo_engine.py`
- `gui/modal_damping.py`
- `gui/sample_parameters.py`
- `tools/write_stk_classical_final_acceptance.py`

## Recent guitar preview change
- Recent Guitar SVG previews now show body-only.
- Removed neck/headstock, bridge, strings, and fret details.
- Kept body outline/silhouette, top color, side/accent stroke, soundhole/rosette, and proportions.
- Length/width/depth/hole radius still drive the mini body shape.
- FIFO, Load behavior, cache reuse, and session-only persistence are unchanged.

## Current sound-determination flow
- Website design creates `lhs_params`: shape, length, width, depth, top thickness, hole radius, top wood, back wood.
- ROM/M4 readiness and STK cache hash are derived from ROM fingerprint plus `lhs_params`.
- `schedule_stk_note_library_after_rom()` starts the CLASSIC note-cache job after ROM.
- `stk_app_audio_service` builds per-note render JSON with note frequency, duration, physical factors, modal/mix data, and output path.
- STK/C++ renders note WAVs into worker/staging/final cache.
- Player uses final cache WAVs plus per-string/fret aliases.

## Parameters already influencing sound
- Geometry: length, width, depth, top thickness, hole radius.
- Materials: top/back wood IDs via damping, density/warmth, stiffness/mass proxies.
- Modal data: mode frequencies, gains, Q/tau, participation shares, radiation/mic/bridge proxies.
- Mix/coupling: direct string gain, body modal gain, string-to-body send, radiation weights.
- Per-note factors: note frequency, duration, harmonic/pluck behavior, high-note softening.
- Identity layer: final website mode is `stk_body_transfer_final_v1`, aliasing to v4.1 identity contrast g_30_70.

## What varies between guitars
- User-visible geometry and woods vary directly.
- Derived physical factors vary after conservative clamps around 1.0.
- Body identity vector varies from geometry, woods, modal statistics, mass proxies, radiation/mic/bridge ranks, and damping.
- Cache hash varies with the design, so saved guitars get separate note caches.

## Sound-control categories
- Decay/damping: wood damping coefficients, per-mode damping, Q/tau, top damping factor, note duration, identity decay layer.
- Body vs string mix: string/body send, direct string gain, body modal gain, body-to-string calibration.
- Brightness/warmth: radiation brightness, top/back weights, harmonic gains, high-frequency rolloff, wood/material proxies.
- Modal emphasis: mode frequencies/gains, participation shares, low/mid support, bridge/mic/radiation proxies.
- Coupling/resonance: bridge mobility, effective modal mass, soundhole/air/cavity factors.
- Attack/sustain: pluck position/gain, transient softening, decay tau, body residual shaping.

## Flattening points
- Physical factor clamps are conservative, often keeping changes near 1.0.
- Loudness/peak normalization reduces level differences by design.
- Body gain calibration targets a stable body/string ratio, preserving timbre more than loudness spread.
- Identity residual has RMS/audibility guards, avoiding extreme contrast.

## Safe contrast opportunities
- Slightly strengthen existing bounded factors only: body modal gain, string-to-body send, radiation brightness, modal tau spread.
- Use already-available design/ROM variables: cavity volume, hole/area ratio, bridge mobility, modal density/Q spread, top/back damping.
- Prefer band-limited changes, especially 120-450 Hz body modes and controlled harmonic bands.
- Keep loudness normalization; increase timbral contrast rather than raw volume.

## Unsafe ideas to avoid
- Random per-guitar differences.
- Unbounded EQ/gain or bypassing limiter/normalization.
- Changing FEM/ROM solver outputs to chase audio differences.
- Broad synthetic detuning unrelated to modal/geometry data.
- Reintroducing BOX/ACOUSTIC website choices.

## Recommended next step
- Add a diagnostic-only contrast audit comparing 2-3 cached Classical guitars: factor spread, identity-vector spread, body/string ratio, spectral centroid, decay slope.
- If weak, implement one small bounded tuning pass later: +10-15% spread on existing body modal/radiation/decay factors.

## CLASSIC risk
- LOW for this task.
- Code change is UI-only Recent preview SVG.
- Sound pipeline was inspected only; no solver, ROM, STK synthesis, or WAV generation logic changed.

## Lightweight checks run
- `python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py gui/test_stk_note_library_startup_command.py`
- `python gui/test_stk_note_library_startup_command.py`

## VM validation suggestions
```bash
git pull
python gui/test_stk_note_library_startup_command.py
python -m py_compile gui/app.py gui/stk_app_ui.py gui/stk_app_audio_service.py
streamlit run gui/app.py
```
