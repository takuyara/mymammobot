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
	all_lens = []
	for i in range(len(points) - 1):
		t_len = np.linalg.norm(points[i] - points[i + 1])
		all_lens.append(t_len)
	poly = pv.lines_from_points(points)
	poly["scalars"] = scalars
	tube = poly.tube(radius = 0.5)
	return tube, np.array(all_lens), scalars

def main():
	print(len(cl_paths))
	airway = pv.read(airway_path)
	p = pv.Plotter()
	p.add_mesh(airway, color = "grey", opacity = 0.2)
	"""
	p.add_mesh_clip_box(airway)
	p.show()
	print(p.box_clipped_meshes)
	"""

	"""
	for i, path in enumerate(cl_paths):
		tube = cl_to_poly(path)
		p.add_mesh(tube, color = colours[i], label = f"CL_{i}")
	p.add_legend()
	p.show()
	"""
	all_lens, all_radii = [], []
	sum_lens = []
	for path in cl_paths:
		__, t_all_lens, t_all_radii = cl_to_poly(path)
		all_lens.append(t_all_lens)
		all_radii.append(t_all_radii)
		sum_lens.append(np.sum(t_all_lens))
	print("CL Len: {:.1f}, {:.1f}, BTPT Len: {:.4f}, {:.4f}".format(np.mean(sum_lens), np.std(sum_lens), np.mean(t_all_lens), np.std(t_all_lens)))

if __name__ == '__main__':
	main()
