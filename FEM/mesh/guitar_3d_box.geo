// guitar_3d_box.geo — physical-tag reference for the 3D guitar pipeline.
// Production meshes are built by FEM/geometry/build_3d_guitar.py (Gmsh Python API),
// not by this placeholder file. Tags below must stay in sync with build_3d_guitar.py
// and fem_main_3d.py (WOOD_SURFACE_TAGS).
//
// 2D facet (surface) physical groups:
//   1  Top_Plate
//   2  Soundhole  (acoustic P=0)
//   3  Back_Plate
//   4  Ribs_Sides (optional Dirichlet clamp when clamp_ribs=true)
//   5  wood_fix   (optional stabilization patch)
//
// 3D volume physical groups:
//   1  Top_Plate_Volume
//   2  Back_Plate_Volume
//   3  Ribs_Sides_Volume
//  10  Air_Internal
