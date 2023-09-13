import os
import csv
import numpy as np
import pyvista as pv

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
cl_base_path = "./CL"
cl_paths = [os.path.join(cl_base_path, t_cl) for t_cl in os.listdir(cl_base_path) if t_cl.endswith(".dat")]
#colours = ["0xFF0000", "0x7FFFD4", "0x000000", "0x0000FF", "0x8A2BE2", "0x654321", "0x7FFF00", "0xFF7F50", "0x008B8B", "0xA9A9A9", "0x006400", "0xFF8C00", "0x483D8B", "0xFF1493", "0xADFF2F", "0x808000", "0x008080"]
colours = ["red", "blue", "green", "yellow", "orange", "0x654321", "0x7FFF00", "0xFF7F50", "0x008B8B", "0xA9A9A9", "0x006400", "0xFF8C00", "0x483D8B", "0xFF1493", "0xADFF2F", "0x808000", "0x008080"]
#cl_paths = ["./CL/CL0.dat", "./CL/CL1.dat", "./CL/CL2.dat", "./CL/CL3.dat", "./CL/CL4.dat"]
#n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n  
def cl_to_poly(path):
	points = []
	scalars = []
	with open(path, newline = "") as f:
		reader = csv.DictReader(f, delimiter = " ")
		for row in reader:
			points.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
			scalars.append(float(row["MaximumInscribedSphereRadius"]))
	points = np.array(points)
	scalars = np.array(scalars)
	poly = pv.lines_from_points(points)
	poly["scalars"] = scalars
	tube = poly.tube(radius = 0.5)
	return tube

def main():
	print(len(cl_paths))
	airway = pv.read(airway_path)
	p = pv.Plotter()
	#p.add_mesh(airway, color = "grey", opacity = 0.2)
	p.add_mesh_clip_box(airway)
	p.show()
	print(p.box_clipped_meshes)
	exit()


	for i, path in enumerate(cl_paths):
		tube = cl_to_poly(path)
		p.add_mesh(tube, color = colours[i], label = f"CL_{i}")
	p.add_legend()
	p.show()

if __name__ == '__main__':
	main()
