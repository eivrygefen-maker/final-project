# Design Studio — 3D ROM preview (Streamlit custom component)

Instant **Three.js** guitar body from luthier blueprints (Classical / Dreadnought). Sliders expose only the ROM/LHS parameters.

## Files

- `index.html` — Design Studio UI + Three.js extrusion + Streamlit bridge
- `__init__.py` — `declare_component("fast_preview", ...)`

## ROM parameters (component output)

| Field | Backend key |
|-------|----------------|
| Length L | `geometry.length` |
| Width W | `geometry.width` |
| Depth D | `geometry.depth` |
| Top thickness | `geometry.top_thickness` |
| Soundhole radius | `geometry.hole_radius` |
| Top / back wood | `materials.top/back.wood_id` |

## Actions

- **Save & Sync** → `app.py` writes `guitar_3d.json`, runs `build_3d_guitar.py`, refreshes PyVista
- **Run ROM Prediction** / **Run Full FEM** → sync + queue acoustics (User / Admin mode)

## Run

```bash
cd gui
streamlit run app.py
```
