import dolfinx
from mpi4py import MPI
import numpy as np

# Load mesh
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "FEM/mesh/guitar_3d.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    # Read the meshtags
    facet_tags = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "FEM/mesh/guitar_3d.xdmf", "r").read_meshtags(mesh, "facet_tags")
    
    # Print counts for each tag
    unique_tags = np.unique(facet_tags.values)
    for tag in unique_tags:
        count = np.sum(facet_tags.values == tag)
        print(f"Tag {tag} facet count: {count}")
