

## Completion note - Yonatan HaKatan rhythm correction

- Corrected melody entry: `yonatan_hakatan_excerpt` in `gui/data/guitar_library/melodies.json`.
- Change type: rhythm/phrase-structure data correction only.
- Pitch sequence now follows `G E E | F D D | C D E F | G G G / G E E | F D D | C E G G | C`.
- Preserved current valid string/fret mappings for C4, D4, E4, F4, and G4.
- No player logic, playback code, chord JSON, STK, cache, GMSH, ROM, or UI behavior was changed.
- Lightweight check run: `python gui/test_guitar_library_json.py` passed.
- No STK, WAV generation, FEM, ROM, GMSH, simulations, or heavy validation was run.

---

## Completion note - Guitar Player resize/loading redraw fix

- Files changed: `gui/components/guitar_player/index.html`, `CODEX_HANDOFF.md`.
- Files inspected before editing: `CODEX_HANDOFF.md`, `gui/components/guitar_player/index.html`, `gui/components/guitar_player/__init__.py`.
- Exact issue fixed: the Guitar Player component now keeps a visible internal placeholder/loading panel with stable minimum height and redraws the existing fretboard layout after browser resize instead of remaining blank/disappearing.
- Resize implementation: stores the latest `player`/`library` payload, adds a debounced `resize` listener using `setTimeout` plus `requestAnimationFrame`, and rebuilds only the visual fretboard/library layout for ready payloads.
- State preservation: resize redraw does not call note preload, does not restart audio, and does not restart melody playback for a ready player.
- Generate Sound/STK/cache logic untouched: no Python runtime flow, request latching, background jobs, cache readiness, cache hashes, STK rendering, or note playback logic changed.
- Lightweight checks performed: reviewed component diff, ran `git diff --check`, and verified added resize/min-height symbols with `rg`.
- No GMSH, FEM, ROM, STK, WAV generation, simulations, website pipeline runs, production validation, or full test suites were run.

---

## Completion note - Final Guitar Player layout/latch fix

- Files changed: `gui/components/guitar_player/index.html`, `gui/stk_app_ui.py`, `CODEX_HANDOFF.md`.
- Files inspected before editing: `CODEX_HANDOFF.md`, `gui/components/guitar_player/index.html`, `gui/components/guitar_player/__init__.py`, `gui/app.py`, `gui/stk_app_ui.py`, and the current `git diff` of the previous resize patch.
- Regression cause fixed: the previous patch added oversized fixed/min-height layout and a destructive resize redraw from stored payload; `.idle` also received explicit display styling that could fight the `hidden` state.
- Exclusive state model: `showIdle()` hides `playerPanel`/library/orientation; `showPlayer()` hides the idle panel. Ready renders only fretboard plus Chords/Melodies; building/idle renders only the compact placeholder panel.
- Resize behavior: resize now only schedules a content-height update through `requestAnimationFrame`/`ResizeObserver`; it does not rebuild from cached payload, replace ready UI with a placeholder, restart audio, or restart melody playback.
- Layout behavior: removed oversized min-height/fixed-height behavior; Streamlit frame height is measured from the actual `.wrap` content so Chords and Melodies remain directly below the fretboard.
- Generate latch behavior: `request_generate_guitar()` still latches building requests by hash; immediate-ready paths now clear the pending request only after `generate_or_load_ready_guitar()` successfully issues/activates the ready player payload.
- FEM/GMSH/ROM/STK/WAV untouched: no solver, mesh, ROM, STK rendering, WAV generation, cache format/hash, chord/melody data, or note playback logic changed.
- Lightweight checks performed: `python -m py_compile gui/stk_app_ui.py`, `git diff --check`, `rg` source checks for removed stale resize/min-height symbols and new frame-height hooks, and manual diff inspection.
- No Streamlit, GMSH, FEM, ROM, STK, WAV generation, simulations, production validation, or full test suites were run.
