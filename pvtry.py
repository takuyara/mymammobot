import numpy as np

import pyvista as pv

filename = "sphere-shrinking.mp4"

mesh = pv.Sphere()
mesh.cell_data["data"] = np.random.random(mesh.n_cells)

plotter = pv.Plotter()
# Open a movie file
plotter.open_movie(filename)

# Add initial mesh
plotter.add_mesh(mesh, scalars="data", clim=[0, 1])
# Add outline for shrinking reference
plotter.add_mesh(mesh.outline_corners())

plotter.show(auto_close=True)  # only necessary for an off-screen movie

# Run through each frame
plotter.write_frame()  # write initial data

# Update scalars on each frame
for i in range(100):
    random_points = np.random.random(mesh.points.shape)
    mesh.points = random_points * 0.01 + mesh.points * 0.99
    mesh.points -= mesh.points.mean(0)
    mesh.cell_data["data"] = np.random.random(mesh.n_cells)
    plotter.add_text(f"Iteration: {i}", name='time-label')
    plotter.write_frame()  # Write this frame
    print("Frame wrote ", i)

# Be sure to close the plotter when finished
plotter.close()
