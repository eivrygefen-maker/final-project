# Full-FOM GMSH Classical Sample Demo Quick Start

Run this from the VM:

```bash
cd ~/final-project
bash tools/demo_classical_sample_fom_gmsh.sh
```

This opens a real full-FOM Classical mesh for existing `sample_000` in the GMSH GUI.
It uses the repository geometry builder's `FEM_ALLOW_FOM=1` path, not the
display-mesh branch.

Notes for presenters:

- This is a real FOM mesh, not `FEM/mesh/display_mesh.msh`.
- It may take noticeably longer than the display demo because it builds the full
  FOM/FSI mesh with physical tags and internal air volume where supported.
- No FEM solve is run.
- No ROM, STK, WAV, audio, or website pipeline work is started.
- The active website config `FEM/configs/guitar_3d.json` is untouched.
- The active website display mesh `FEM/mesh/display_mesh.msh` is untouched.
- Temporary demo files are written under `/tmp` and removed after closing GMSH,
  including on Ctrl+C where practical.

In GMSH, use the Visibility/Physical Groups panels to inspect top, back,
ribs/sides, soundhole, support, and `Air_Internal` volume groups. Hiding the outer
wood surfaces or enabling clipping/section view helps reveal the internal air
volume.
