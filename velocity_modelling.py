import os
import csv
import sys
import numpy as np
import pyvista as pv

from utils.misc import str_to_arr
from ds_gen.dynamics_modelling import get_velocities, smoothing_poses

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
cl_path = "./CL"

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
	em_points = [[], [], []]
	em_orients = [[], [], []]
	em_ups = [[], [], []]
	all_res = [["radial_velocity", "axial_velocity", "total_velocity", "rotation_velocity", "lumen_radius"]]

	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			em_idx = int(row["em_idx"])
			em_points[em_idx].append(str_to_arr(row["position"]))
			em_orients[em_idx].append(str_to_arr(row["orientation"]))
			em_ups[em_idx].append(str_to_arr(row["up"]))

	for points, orients, ups in zip(em_points, em_orients, em_ups):
		points = np.stack(points, axis = 0)
		orients = np.stack(orients, axis = 0)
		ups = np.stack(ups, axis = 0)
		smoothed_poses = smoothing_poses(points, orients, ups, method = "none")
		velocity_res = get_velocities(smoothed_poses, cl_path)
		all_res.extend(velocity_res)

	with open("velocity_res.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(all_res)

	"""
	poly = pv.lines_from_points(points)
	tube = poly.tube(radius = 0.5)
	p.add_mesh(tube, color = "red")
	p.show()
	"""

if __name__ == '__main__':
	main()
