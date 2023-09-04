import os
import csv
import sys
import numpy as np
import pyvista as pv

from utils.misc import str_to_arr
from ds_gen.dynamics_modelling import get_velocities, smoothing_poses
from utils.geometry import arbitrary_perpendicular_vector
from ds_gen.depth_map_generation import get_depth_map

import matplotlib.pyplot as plt

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
	poly = pv.lines_from_points(points)
	tube = poly.tube(radius = 0.5)
	return tube


def get_points(p, path, idx):
	points = []
	with open(path, newline = "") as f:
		reader = csv.DictReader(f, delimiter = " ")
		for row in reader:
			points.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
	print(len(points))
	points = np.array(points)
	position = points[idx]
	print(position)
	orientation = points[idx - 10] - position
	orientation /= np.linalg.norm(orientation)
	up = arbitrary_perpendicular_vector(orientation)
	rgb, __ = get_depth_map(p, position, orientation, up, get_outputs = True)
	return rgb


def main():
	airway = pv.read(airway_path)
	#p = pv.Plotter(off_screen = True, window_size = (224, 224))
	p = pv.Plotter()
	p.add_mesh(airway, opacity = 0.5)
	#p.add_mesh(airway)

	"""
	rgb1 = get_points(p, "./CL1/CL0.dat", 800)
	rgb2 = get_points(p, "./CL1/CL4.dat", 800)
	plt.subplot(1, 2, 1)
	plt.imshow(rgb1)
	plt.subplot(1, 2, 2)
	plt.imshow(rgb2)
	plt.show()
	"""

	em_idx = 0
	points = []
	orients = []
	ups = []


	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if int(row["human_eval"]) == 1:
				points.append(str_to_arr(row["position"]))
				orients.append(str_to_arr(row["orientation"]))
				ups.append(str_to_arr(row["up"]))
	points = np.stack(points, axis = 0)
	orients = np.stack(orients, axis = 0)
	ups = np.stack(ups, axis = 0)

	gen_points = []
	gen_orients = []
	gen_ups = []
	path = "./virtual_dataset/pls-sm/val"
	for tp in os.listdir(path):
		if tp.endswith(".txt") and tp.find("clbase") == -1:
			pose = np.loadtxt(os.path.join(path, tp))
			gen_points.append(pose[0, ...])
			gen_orients.append(pose[1, ...])
			gen_ups.append(pose[2, ...])
	gen_points = np.stack(gen_points, axis = 0)
	gen_orients = np.stack(gen_orients, axis = 0)
	gen_ups = np.stack(gen_ups, axis = 0)

	"""
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
		p.add_mesh(pv.Arrow(points[i, ...], ups[i, ...]), color = "blue")

	for i in range(len(gen_points)):
		p.add_mesh(pv.Arrow(gen_points[i, ...], gen_orients[i, ...]), color = "green")
		p.add_mesh(pv.Arrow(gen_points[i, ...], gen_ups[i, ...]), color = "yellow")
	
	"""
	for path in cl_paths:
		p.add_mesh(cl_to_poly(path), color = "blue")
	"""

	#p.add_points(points, render_points_as_spheres = True, point_size = 5, color = "red")

	#p.add_points(gen_points_t, render_points_as_spheres = True, point_size = 5, color = "yellow")
	#p.add_points(gen_points, render_points_as_spheres = True, point_size = 5, color = "green")
	p.show()

if __name__ == '__main__':
	main()
