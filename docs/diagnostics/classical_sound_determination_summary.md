# Classical Sound Determination Summary

This is a diagnostic-only inspection of the Classical-only website sound path. No FEM, ROM, STK, solver, or WAV generation behavior was changed.

## Current Flow

The website design studio emits Classical geometry and material parameters. `gui/app.py` converts those into LHS parameters: shape, length, width, depth, top thickness, hole radius, top wood, and back wood. ROM readiness and the STK cache hash are derived from the ROM fingerprint plus those LHS parameters.

After Save/ROM, `gui/stk_app_audio_service.py` schedules the Classical note-library job. The note-library export creates one render entry per note with explicit note frequency, duration, physical factors, modal/mix data, and output path. STK/C++ renders WAVs into worker/staging/final cache, then aliases are created for string/fret playback.

## What Determines Sound

Direct design inputs:
- Woods: top/back wood IDs affect damping, material factors, identity vector, and exported physical factors.
- Length, width, depth: affect geometry proxies, mass/air/cavity/body identity features, damping scale, and the cache hash.
- Top thickness: affects stiffness/weight, damping, and identity features.
- Hole radius: affects soundhole/air/cavity/radiation factors and identity features.

Derived factors:
- Decay/damping: wood damping coefficients, per-mode damping, Q/tau, top damping factor, note duration.
- Body vs string mix: direct string gain, body modal gain, string-to-body send, body/string calibration.
- Brightness/warmth: radiation brightness, top/back/air weights, high-frequency rolloff, harmonic identity gains.
- Modal emphasis: mode frequencies, gains, Q/tau, participation shares, bridge/mic/radiation proxies.
- Coupling/resonance: bridge mobility, effective modal mass, soundhole/cavity/air factors.
- Identity layer: website default uses `stk_body_transfer_final_v1`, aliasing to v4.1 identity contrast `g_30_70`.

## What Flattens Contrast

- Physical factor clamps keep many values near 1.0.
- RMS and peak normalization reduce raw loudness differences.
- Body gain calibration targets a stable body/string ratio.
- Identity residual has audibility and RMS guards.
- If per-guitar physical audit data is missing, fallback values may reduce differentiation.

## Safe Later Opportunities

The safest future changes are bounded increases to factors that already exist: body modal gain, string-to-body send, radiation brightness, and modal tau/Q spread. Prefer band-limited changes in physically relevant body bands, especially 120-450 Hz, and preserve loudness normalization.

## Unsafe Ideas

Avoid random offsets, unbounded EQ/gain boosts, bypassing limiter/normalization, changing FEM/ROM solver behavior, synthetic detuning unrelated to modal data, or reintroducing BOX/ACOUSTIC website options.

## Cache Comparison

No local Classical `audio/app_stk_note_cache/classical` cache was present in the Codex workspace, so no 2-3 guitar cache comparison was possible here. Run this audit on the VM against existing `current_preview_*` and `saved_guitar_*` caches.
