# CODEX_HANDOFF.md

## Task completed
- Adjusted the static Classical guitar chord playback data only.

## Files changed
- `gui/data/guitar_library/chords.json`
- `CODEX_HANDOFF.md`

## Playback-data change
- Previous value:
  `"inter_string_gap_ms": 18`
- New value:
  `"inter_string_gap_ms": 4`

## Confirmations
- Chord definitions unchanged.
- Melody playback data unchanged.
- Player logic unchanged.
- Generate Sound behavior unchanged.
- GMSH, ROM, STK, cache generation, synthesis, physical parameters, and HTML preview unchanged.
- No Recent Guitar behavior added.

## Lightweight checks run
- `python gui\test_guitar_library_json.py`

## VM verification steps
```bash
git pull
python gui/test_guitar_library_json.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Open an already-ready clickable Classical guitar.
- Click several chord buttons.
- Confirm chord notes sound almost simultaneous with only a subtle strum offset.
- Confirm no computation chain starts.
