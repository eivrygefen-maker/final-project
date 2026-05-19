import meshio
import numpy as np

# Read the .msh file
mesh = meshio.read("FEM/mesh/guitar_3d.msh")

print("--- Found Physical Tags ---")
found = False
for cell_block in mesh.cells:
    if "gmsh:physical" in cell_block.data:
        tags = cell_block.data["gmsh:physical"]
        unique_tags = np.unique(tags)
        print(f"Cell type: {cell_block.type}")
        for tag in unique_tags:
            print(f"  Tag {tag}: {np.sum(tags == tag)} elements")
        found = True

if not found:
    print("No 'gmsh:physical' tags found in the cell data!")
