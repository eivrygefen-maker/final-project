# CODEX_HANDOFF.md

## Files changed
- `FEM/geometry/build_3d_guitar.py`
- `gui/app.py`
- `gui/components/guitar_player/index.html`
- `gui/components/fast_preview/index.html`
- `CODEX_HANDOFF.md`

## Rollback completed
- Reverted the latest visualization/loading patch completely.
- Removed Material View / physical-group flat coloring code.
- Removed `Show Engineering Mesh` checkbox and viewer key/toggle changes.
- Restored old Gmsh viewer behavior with mesh edges shown as before.
- Removed Guitar Player CSS spinner/loading-area changes.
- Restored heading sizes and `GENERATE SOUND` heading from the last stable UI.

## Only new change after rollback
- Increased only `display_mesh.msh` display density.
- `DISPLAY_GLOBAL_LC_M` changed from `0.012` to `0.010` in `FEM/geometry/build_3d_guitar.py`.
- This is a display-only `FEM_ALLOW_DISPLAY=1` shell mesh setting.
- Approximate triangle density increase: `(12 / 10)^2 = 1.44`, about 44% more triangles.

## Unchanged
- FEM/FOM physics mesh unchanged.
- ROM mesh/settings unchanged.
- Solver behavior unchanged.
- Geometry dimensions unchanged.
- STK/audio/cache/player logic unchanged.
- Website flow unchanged.
- CLASSIC-only unchanged.
- No Recent Guitars restored.

## Lightweight checks run
- `python -m py_compile FEM\geometry\build_3d_guitar.py gui\app.py gui\stk_app_ui.py`
- `git diff --check`
- Scan confirmed no `Show Engineering Mesh`, material-view coloring, or spinner code remains.

## VM verification steps
```bash
git pull
python -m py_compile FEM/geometry/build_3d_guitar.py gui/app.py gui/stk_app_ui.py
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- In the website, Save & Sync a Classical design to regenerate `FEM/mesh/display_mesh.msh`.
- Confirm the Gmsh/player UI looks like the previous stable version.
- Confirm display mesh log says `10 mm shell`.
- Confirm no `Show Engineering Mesh` toggle and no loading spinner are visible.
- Confirm no FEM/ROM/STK/full simulation was triggered beyond normal display mesh generation.
