# Guitar Library Data

This folder contains static JSON data for the interactive Classical Guitar Player.

## Files

- `chords.json`  
  Playable chord definitions.

- `melodies.json`  
  Predefined melody playback events.

## Scope

These files are player-library data only.

They must not start, restart, modify, or interfere with:

- GMSH
- ROM
- STK
- cache generation
- audio synthesis
- physical guitar parameters
- Generate Sound behavior

The library is used only after the existing clickable guitar player is ready.

## String Numbering

The existing guitar numbering is used:

- `string 6` = low E
- `string 5` = A
- `string 4` = D
- `string 3` = G
- `string 2` = B
- `string 1` = high E

For chord fingering:

- fret `0` = open string
- fret `-1` = muted string

## Melody Event Encoding

Each melody event is encoded as:

```text
[note, string, fret, start_beats, duration_beats, velocity]
```

Example:

```json
["C4", 2, 1, 0.0, 1.0, 0.78]
```

Meaning:

* play note `C4`
* on string `2`
* fret `1`
* at beat `0.0`
* for `1.0` beat
* with velocity `0.78`

The note, string, and fret must match the existing clickable-guitar mapping and available cached WAV files.

