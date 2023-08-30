import os
import csv
import sys
import numpy as np
import pyvista as pv

from utils.misc import str_to_arr
from ds_gen.dynamics_modelling import get_velocities, smoothing_poses

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
cl_path = "./CL1"

cl_paths = [os.path.join(cl_path, t_path) for t_path in os.listdir(cl_path) if t_path.endswith(".dat")]

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

def em_to_poly(path):
	all_res = [["idx", "x", "y", "z", "qw", "qx", "qy", "qz"]]
	points = []
	for i in range(len(os.listdir(path))):
		with open(os.path.join(path, f"{i}.txt")) as f:
			this_num = [float(x) for x in f.read().split()]
		points.append(this_num[ : 3])
		all_res.append([i] + this_num)
	points = np.array(points)
	poly = pv.lines_from_points(points)
	tube = poly.tube(radius = 0.5)
	"""
	with open("emcoords.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(all_res)
	"""
	return tube

def main():
	airway = pv.read(airway_path)
	p = pv.Plotter()
	p.add_mesh(airway, opacity = 0.5)
	em_idx = 0
	points = []
	orients = []
	ups = []


	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if int(row["em_idx"]) == em_idx:
				points.append(str_to_arr(row["position"]))
				orients.append(str_to_arr(row["orientation"]))
				ups.append(str_to_arr(row["up"]))
	points = np.stack(points, axis = 0)
	orients = np.stack(orients, axis = 0)
	ups = np.stack(ups, axis = 0)

	gen_points = []
	gen_orients = []
	path = "./virtual_dataset/single_image_reduced/val"
	for tp in os.listdir(path):
		if tp.endswith(".txt"):
			pose = np.loadtxt(os.path.join(path, tp))
			gen_points.append(pose[0, ...])
			gen_orients.append(pose[1, ...])
	gen_points = np.stack(gen_points, axis = 0)
	gen_orients = np.stack(gen_orients, axis = 0)

	gen_points_t = []
	gen_orients_t = []
	path = "./virtual_dataset/single_image_reduced/train"
	for tp in os.listdir(path):
		if tp.endswith(".txt"):
			pose = np.loadtxt(os.path.join(path, tp))
			gen_points_t.append(pose[0, ...])
			gen_orients_t.append(pose[1, ...])
	gen_points_t = np.stack(gen_points_t, axis = 0)
	gen_orients_t = np.stack(gen_orients_t, axis = 0)


	"""
	for i in range(len(points)):
		p.add_mesh(pv.Arrow(points[i, ...], orients[i, ...]), color = "red")

	for i in range(len(gen_points)):
		p.add_mesh(pv.Arrow(gen_points[i, ...], gen_orients[i, ...]), color = "green")
	"""

	for path in cl_paths:
		p.add_mesh(cl_to_poly(path), color = "blue")

	p.add_points(points, render_points_as_spheres = True, point_size = 5, color = "red")

	p.add_points(gen_points_t, render_points_as_spheres = True, point_size = 5, color = "yellow")
	p.add_points(gen_points, render_points_as_spheres = True, point_size = 5, color = "green")
	p.show()

if __name__ == '__main__':
	main()
