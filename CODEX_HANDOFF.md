# CODEX_HANDOFF.md

## Root cause
- Fast preview used direct global width scaling:
  `outline_y *= width / nominal_width`.
- At the allowed extremes this made the Classical body cartoonishly wide or narrow.
- Camera/frame code was not the main cause; the distortion was in the outline mapping.

## Files changed
- `gui/components/fast_preview/index.html`
- `CODEX_HANDOFF.md`

## Exact preview-only mapping adjustment
- Added `visualWidthScaleForPoint()` inside the Three.js fast preview.
- Length still scales directly with selected `length`.
- Width now uses a bounded visual response by body region:
  lower bout response `0.82`, upper bout `0.68`, waist `0.52`.
- Visual width scale is clamped to `[0.70, 1.16]`.
- Nominal/default width remains visually unchanged.
- Actual `state.width` and emitted design payload are unchanged.

## Expected visual cases
- `W=0.200`: old global width scale about `0.55`; new visual scale is bounded around `0.70` at bouts and about `0.77` at waist.
- Nominal/default: visual scale stays `1.00`.
- `W=0.450`: old global width scale about `1.23`; new visual scale is about `1.16` lower/upper bout and `1.12` waist.
- Preview should still show narrow/wide differences, but with a more believable Classical silhouette.

## Scientific/Gmsh unchanged
- No Gmsh geometry generation changed.
- No FEM/ROM/solver/STK/physics changed.
- Width W still means the same thing in saved design data and scientific pipeline.
- Depth and soundhole controls are still separate from the visual width mapping.

## Lightweight checks run
- `python -m py_compile FEM\geometry\build_3d_guitar.py gui\app.py gui\stk_app_ui.py`
- `git diff --check`
- Code scan confirmed no material-view/toggle/spinner remnants were reintroduced.

## VM verification steps
```bash
git pull
python -m streamlit run gui/app.py --server.headless true --server.port 8501
```
- Compare HTML preview vs Gmsh for `W=0.200`, default/nominal, and `W=0.450`.
- Confirm HTML still reflects relative width changes.
- Confirm wide design is less bulged and narrow design less tall/needle-like.
- Confirm Save & Sync / Gmsh geometry remains scientifically unchanged.
