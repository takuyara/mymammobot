import os
import pyvista as pv
#path = "Airway_Phantom_AdjustSmooth.stl"
#path = "InnerAirwayMesh.stl"
path = "bronchiMesh.stl"
#path = "SiliconPhantomTrachea.stl"
#path = "MeshPhantom_20151113Smooth.stl"
surface = pv.read(os.path.join("meshes", path))
p = pv.Plotter()
p.add_mesh(surface)
p.show_bounds(grid = "front", location = "outer", all_edges = True)
p.show()
