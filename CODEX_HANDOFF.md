Read the repository and current `CODEX_HANDOFF.md` in full before doing anything.

Do not implement features, do not refactor, do not modify runtime behavior, and do not run GMSH, FEM, ROM, STK, WAV generation, simulations, or full test suites.

Your task is documentation and technical audit only.

Create a new detailed Markdown document:

`docs/CLASSICAL_GUITAR_CURRENT_TECHNICAL_AUDIT.md`

Also append a short note to `CODEX_HANDOFF.md` stating that the audit was created and where it is located.

The report must describe the CURRENT repository behavior only. Do not infer behavior from old code, unused code, comments, or filenames. Clearly distinguish:

* confirmed active runtime behavior;
* available but currently inactive/hidden code;
* offline/full-pipeline capability;
* assumptions or areas that could not be verified from source code.

For every major claim, provide exact source references in this format:

```text
file path
function/class name
relevant line range if available
```

# Required Report Structure

## 1. Executive technical summary

Explain, in 10–15 precise points:

* What the Classical Guitar website currently does.
* What data enters the system.
* What is computed.
* What is displayed.
* What is rendered as sound.
* What is precomputed and cached.
* What is physically modeled versus what is UI/playback behavior.

This section must be understandable by a technical examiner.

---

## 2. Current website flow — exact runtime sequence

Document the exact sequence from opening the website until an interactive guitar can be played.

Include:

1. Application startup.
2. Relevant Streamlit pages/components.
3. Design input controls.
4. Fast HTML visual preview.
5. Save & Sync.
6. GMSH stage.
7. ROM stage.
8. STK stage.
9. Note-cache readiness.
10. Generate Sound behavior.
11. Interactive fretboard loading.
12. Chord and melody library loading/playback.

For each stage, document:

* trigger;
* input files/data;
* output files/data;
* relevant Python function(s);
* relevant frontend component(s);
* whether it is synchronous, background, cached, or polling-based;
* what user-visible state appears during the stage.

Important:
Do not describe Save & Sync as directly starting STK unless the code proves that. Preserve the actual chain exactly as implemented.

---

## 3. Current Classical-only scope

Explain exactly:

* Which guitar shape is active in the website.
* Which shapes or old experimental paths exist but are not part of the active user flow.
* Whether BOX, Acoustic, or other historic branches are visible, hidden, disabled, or unused.
* Which outputs are considered frozen baselines.
* Which data/configuration is used by the currently active Classical Guitar pipeline.

Do not delete or alter anything. Report only.

---

## 4. Design parameters and their roles

Create a table of every user-configurable guitar parameter currently exposed in the website.

For each parameter include:

* UI label;
* internal variable name;
* units;
* accepted range/default;
* where it is stored;
* which downstream stage consumes it;
* physical meaning;
* whether it affects:

  * geometry;
  * mesh;
  * material;
  * ROM;
  * STK;
  * visual preview only.

Also identify parameters that are derived internally rather than directly editable.

---

## 5. Geometry and GMSH pipeline

Document the actual current geometry/GMSH process.

Include:

* source geometry builder files;
* body outline generation;
* top, back, ribs, bridge, soundhole, air cavity, and any other modeled components;
* physical groups/tags;
* display mesh versus full physics mesh;
* mesh profiles if more than one exists;
* which mesh is used in the website;
* which mesh is used for the full FEM pipeline;
* where mesh files are written;
* what artifacts are retained or reused.

Explain carefully the difference between:

```text
Fast visual preview
Display mesh
Full FEM physics mesh
```

Do not claim they are identical unless source code proves it.

---

## 6. Full FEM pipeline — detailed technical explanation

Document the full FEM capability, even if it is not executed for every website interaction.

Explain:

* governing problem type;
* structural displacement field;
* acoustic pressure field;
* structure–acoustic coupling;
* boundary conditions;
* physical domains;
* material model;
* solver path;
* eigenvalue/mode extraction;
* mode frequencies;
* modal shapes;
* mass/stiffness/coupling matrices;
* any shift-invert or alternative eigensolver strategy;
* mesh/solver validation or integrity checks;
* output artifacts.

Use equations only where they are genuinely used by the implementation or required to explain the pipeline accurately.

For every formula, explain each variable in plain language.

Do not present generic FEM theory as though it is necessarily implemented in this repository.

---

## 7. Meaning of V2, B3, T, and related internal model labels

This section is critical.

Find every occurrence of terms such as:

* V2
* B3
* T
* GMSH
* ROM
* FSI
* active domain
* physical integrity
* modal body
* body transfer
* identity contrast
* V4.1
* Stage 5.x

For each term:

* state its exact repository meaning;
* identify whether it is active, historical, experimental, or frozen;
* identify the files where it is defined or consumed;
* explain whether it is:

  * a physical-model version;
  * a boundary-condition formulation;
  * a geometry/material parameter;
  * a solver option;
  * a sound-mapping layer;
  * an experiment label;
  * a UI label;
  * or something else.

Do not use vague explanations. Do not state that V2/B3/T are user controls unless source code proves it.

---

## 8. ROM pipeline

Explain the active ROM process precisely.

Include:

* why ROM exists in this project;
* source inputs;
* reduction method or retained modal information;
* outputs;
* relationship to the full FEM model;
* what the website actually consumes;
* what information is passed forward to STK;
* what is cached;
* what has to be recomputed after a design change;
* what is reused when a matching design/cache already exists.

Clearly distinguish ROM that is physically grounded from any later artistic/aural mapping.

---

## 9. STK audio pipeline

Explain the current audio generation chain in detail.

Include:

* source files and functions;
* what starts STK;
* what inputs STK receives;
* whether Python only exports parameters or renders audio;
* how the current Classical contrast preset is selected;
* which data comes from body/ROM/modal information;
* how direct string, body/modal response, damping, radiation, decay, or other parameters are used if they are actually implemented;
* rendered note range;
* number of base notes;
* number of string/fret playback positions;
* cache directory format;
* hash/cache-spec behavior;
* background workers;
* worker staging/finalization behavior;
* readiness checks;
* error handling;
* what Generate Sound does and does not do.

Explicitly explain the distinction:

```text
STK render generation
versus
browser playback from cached WAV files
```

---

## 10. Interactive guitar player

Document the current ready-player behavior.

Include:

* fretboard orientation;
* string numbering;
* note mapping;
* open-string mapping;
* position aliases;
* same-string retrigger behavior;
* overlap behavior across different strings;
* voice limiting;
* cache file lookup;
* player component communication between Python and frontend;
* Generate Sound request behavior;
* readiness display behavior;
* conditions required before the player is shown.

Also document the current implemented chord/melody library:

* JSON file paths;
* JSON schema;
* chord playback;
* strum gap;
* melody timing;
* random melody behavior;
* cancellation/replacement behavior for a currently running melody;
* confirmation that these controls do not start GMSH, ROM, STK, or cache generation.

---

## 11. Data, hashes, caches, and reports

Create a practical file map covering:

* configuration files;
* geometry outputs;
* mesh outputs;
* FEM outputs;
* ROM outputs;
* STK parameters;
* note caches;
* debug reports;
* validation reports;
* player JSON data;
* shared/export audio paths.

For each path explain:

* what creates it;
* what reads it;
* whether it is source-of-truth, cache, diagnostic, or temporary;
* whether it is safe to delete or should be preserved.

Do not recommend deleting files unless source code/documentation proves it is safe.

---

## 12. Validation and reliability mechanisms

Document all current checks and gates, such as:

* mesh validation;
* physical-integrity checks;
* ROM checks;
* note-map validation;
* cache-readiness validation;
* WAV validation;
* JSON validation;
* current status precedence;
* worker completion checks;
* player readiness checks.

For each one explain:

* what failure it prevents;
* what artifact/report it creates;
* whether it is active in the website runtime or only in offline validation.

---

## 13. Current limitations and honest boundaries

Write a technically honest section suitable for an external presentation.

Include:

* what is physically simulated;
* what is reduced/approximated;
* what is heuristic or perceptual mapping;
* what is visual-only;
* what is precomputed;
* what is not a real-time full FEM solve;
* what depends on cached sound assets;
* what remains outside the current scope.

Do not overstate realism or claim direct measured validation unless code/reports prove it.

---

## 14. Presentation-ready explanation map

At the end, add two separate summaries:

### A. One-minute explanation

A concise explanation for a non-technical audience.

### B. Five-minute technical explanation

A structured explanation for examiners, including:

```text
Design input
→ geometry
→ mesh
→ FEM/ROM/modal response
→ STK sound rendering
→ cached note library
→ interactive guitar playback
```

Also include 10 likely examiner questions with accurate answers based only on the repository.

---

## Reporting style

* Write in clear English technical Markdown.
* Use tables where useful.
* Use simple diagrams in Mermaid only when they improve clarity.
* Be specific.
* Do not invent values, paths, or model behavior.
* Mark uncertainty clearly.
* No source-code changes except creating this documentation file and appending the short note to CODEX_HANDOFF.md.

At the end of CODEX_HANDOFF.md, add:

* files inspected;
* documentation file created;
* whether any code files changed;
* lightweight inspection commands used;
* confirmation that no FEM/ROM/STK/WAV jobs were run.

---

## Completion note - Classical Guitar technical audit

- Documentation file created: `docs/CLASSICAL_GUITAR_CURRENT_TECHNICAL_AUDIT.md`
- Repository files inspected: `gui/app.py`, `gui/stk_app_ui.py`, `gui/stk_app_audio_service.py`, `gui/pgsm_stk_parameter_export.py`, `gui/stk_pipeline_defaults.py`, `gui/guitar_library.py`, `gui/components/guitar_player/index.html`, `gui/components/guitar_player/__init__.py`, `config/app_stk_config.py`, `config/app_stk_config.json`, `config/classical_guitar_fretboard.json`, `tools/build_app_stk_note_library.py`, `FEM/geometry/build_3d_guitar.py`, `FEM/scripts/fem_main_3d.py`, `FEM/rom/rom_manager.py`, `FEM/scripts/m4_shape_registry.py`, `ROM/classic/rom_model_manifest.json`, `FEM/configs/guitar_3d.json`, and `CODEX_HANDOFF.md`.
- Code files changed: none.
- Documentation files changed: `docs/CLASSICAL_GUITAR_CURRENT_TECHNICAL_AUDIT.md`, `CODEX_HANDOFF.md`.
- Lightweight inspection commands used: `Get-Content`, `Select-String`, `rg`, `Test-Path`, and `git status --short`.
- No GMSH, FEM, ROM, STK, WAV generation, simulations, production validation, or full test suites were run.

---

## Completion note - GMSH demo script

- Files created: `tools/demo_classical_sample_gmsh.sh`, `docs/DEMO_GMSH_QUICK_START.md`.
- Purpose: safe presenter command for showing existing Classical `sample_000` geometry in the GMSH GUI.
- Safety behavior: script builds a temporary copy of the needed geometry/config assets under `/tmp`, writes the demo config and `display_mesh.msh` only there, opens GMSH on that temporary mesh, and cleans the temporary folder on exit/Ctrl+C where practical.
- Active website state untouched: script never writes `FEM/configs/guitar_3d.json` or `FEM/mesh/display_mesh.msh`.
- Code/runtime behavior changed: none outside the new demo script/documentation.
- Lightweight checks run: read/previewed the new script and quick-start doc with `Get-Content`; attempted `bash -n tools/demo_classical_sample_gmsh.sh`, but local Codex Windows environment does not have `bash` installed.
- No GMSH, FEM, ROM, STK, WAV generation, simulations, production validation, or full test suites were run in Codex.

---

## Completion note - Full-FOM GMSH demo script

- Files created: `tools/demo_classical_sample_fom_gmsh.sh`, `docs/DEMO_FULL_FOM_GMSH_QUICK_START.md`.
- Existing display demo left unchanged: `tools/demo_classical_sample_gmsh.sh`.
- Purpose: safe presenter command for opening existing Classical `sample_000` through the repository's real full-FOM GMSH path.
- FOM path used: script runs `FEM/geometry/build_3d_guitar.py` with `FEM_ALLOW_FOM=1` and a temp `FEM_MESH_OUT`, matching the real FOM branch rather than the display branch.
- Safety behavior: script works in a temporary `/tmp` demo folder, copies only needed geometry assets there, writes the demo config and full FOM mesh only there, opens GMSH on that temporary mesh, and cleans the temporary folder on exit/Ctrl+C where practical.
- Active website state untouched: script never writes `FEM/configs/guitar_3d.json`, `FEM/mesh/display_mesh.msh`, active caches, ROM/STK/audio outputs, or website runtime state.
- Code/runtime behavior changed: none outside the new demo script/documentation.
- Lightweight checks run: inspected `gui/app.py` `run_gmsh_fom`, inspected the geometry builder FOM branch/physical tags, read/previewed the new script and quick-start doc with `Get-Content`, and ran `git diff --check`.
- `bash -n` was not available in the local Codex Windows environment, so shell syntax validation should be run on the VM if desired.
- No GMSH, FEM solve, ROM, STK, WAV generation, simulations, production validation, or full test suites were run in Codex.

---

## Completion note - Yonatan HaKatan rhythm correction

- Corrected melody entry: `yonatan_hakatan_excerpt` in `gui/data/guitar_library/melodies.json`.
- Change type: rhythm/phrase-structure data correction only.
- Pitch sequence now follows `G E E | F D D | C D E F | G G G / G E E | F D D | C E G G | C`.
- Preserved current valid string/fret mappings for C4, D4, E4, F4, and G4.
- No player logic, playback code, chord JSON, STK, cache, GMSH, ROM, or UI behavior was changed.
- Lightweight check run: `python gui/test_guitar_library_json.py` passed.
- No STK, WAV generation, FEM, ROM, GMSH, simulations, or heavy validation was run.
