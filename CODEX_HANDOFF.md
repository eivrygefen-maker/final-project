# CODEX_HANDOFF.md

## Task completed
- Implemented the Classical Guitar Melody and Chord Library for the existing ready clickable Classical Guitar Player.
- This is player/UI-only integration using cached note WAVs and the existing string/fret mapping.

## Files created
- `gui/data/guitar_library/README.md`
- `gui/data/guitar_library/chords.json`
- `gui/data/guitar_library/melodies.json`
- `gui/guitar_library.py`
- `gui/test_guitar_library_json.py`

## Files changed
- `gui/app.py`
- `gui/components/guitar_player/__init__.py`
- `gui/components/guitar_player/index.html`
- `CODEX_HANDOFF.md`

## How JSON is loaded
- `gui/guitar_library.py` loads `chords.json` and `melodies.json`.
- The loader validates IDs, string/fret ranges, chord strums, melody timing, velocity, and note/string/fret agreement with the canonical Classical fretboard map.
- `gui/app.py` passes the validated library into the existing guitar player component.
- No STK, WAV rendering, FEM, ROM, GMSH, cache generation, or synthesis path is called.

## Controls added
- Chord buttons for all entries in `chords.json`.
- Melody dropdown.
- `Play Selected Melody`.
- `Play Random Melody`.
- Current/last melody title display.

## Playback behavior
- Chords and melodies resolve through the existing `player.positions` string/fret map.
- Playback uses the existing cached per-position WAV files and existing browser note playback path.
- Muted chord strings are ignored.
- Melody start beats are converted using each melody's `tempo_bpm`.
- Starting a melody cancels any pending previous melody timers.
- Random melody avoids immediate repeat when more than one melody exists.

## Confirmations
- Generate Sound behavior unchanged.
- GMSH -> ROM -> STK workflow unchanged.
- Cache generation unchanged.
- Audio synthesis, physical parameters, strong preset, and HTML preview unchanged.
- No Recent Guitar behavior added.
- Library controls appear only inside the ready player.

## Lightweight checks run
- `python -m py_compile gui\guitar_library.py gui\test_guitar_library_json.py gui\app.py gui\components\guitar_player\__init__.py`
- `python gui\test_guitar_library_json.py`
- Bundled Node syntax check for the inline `guitar_player/index.html` script.

## VM verification steps
```bash
git pull
python -m py_compile gui/guitar_library.py gui/test_guitar_library_json.py gui/app.py gui/components/guitar_player/__init__.py
python gui/test_guitar_library_json.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Open an already-ready clickable Classical guitar.
- Click every chord button once.
- Select and play two melodies.
- Use Play Random Melody twice and confirm no immediate repeat.
- Confirm no computation chain starts.
- Confirm no STK, GMSH, ROM, or cache activity is triggered.
