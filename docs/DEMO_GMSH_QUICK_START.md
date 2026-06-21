# GMSH Classical Sample Demo Quick Start

Run this from the VM:

```bash
cd ~/final-project
bash tools/demo_classical_sample_gmsh.sh
```

The script shows the existing Classical `sample_000` geometry in the GMSH GUI.

Verification note:

- The active website config `FEM/configs/guitar_3d.json` is untouched.
- The active website display mesh `FEM/mesh/display_mesh.msh` is untouched.
- No FEM, ROM, STK, WAV, website, or simulation work is started.
- Temporary demo files are written under `/tmp` and removed after closing GMSH, including on Ctrl+C where practical.
