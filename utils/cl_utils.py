import os
import csv
import numpy as np
from utils.geometry import project_point_to_line

def load_cl(base_path, cl_idx):
	cl_path = os.path.join(base_path, f"CL{cl_idx}.dat")
	if not os.path.exists(cl_path):
		return None, None
	points = []
	radiuses = []
	with open(cl_path, newline = "") as f:
		reader = csv.DictReader(f, delimiter = " ")
		for row in reader:
			points.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
			radiuses.append(float(row["MaximumInscribedSphereRadius"]))
	points.reverse()
	radiuses.reverse()
	return np.array(points), np.array(radiuses)

def load_all_cls(base_path, n_cls = 100):
	all_cls = []
	for cl_idx in range(n_cls):
		points, radiuses = load_cl(base_path, cl_idx)
		if points is None:
			break
		all_cls.append((points, radiuses))
	return all_cls

def locate_on_cl(x, all_cls, n_candidates = 50):
	min_dist = 1e10
	for cl_idx, (cl_points, __) in enumerate(all_cls):
		cl_dists = np.sum((cl_points - x) ** 2, axis = 1)
		for i in np.argsort(cl_dists)[ : n_candidates]:
			if i + 1 >= len(cl_points):
				continue
			x_ = project_point_to_line(x, cl_points[i, ...], cl_points[i + 1, ...])
			if np.linalg.norm(x_ - x) < min_dist:
				min_dist, cl_pos = np.linalg.norm(x_ - x), (cl_idx, i)
	return cl_pos

def project_to_cl(x, all_cls, n_candidates = 50, return_cl_indices = False):
	cl_idx, on_line_idx = locate_on_cl(x, all_cls, n_candidates)
	points, radiuses = all_cls[cl_idx]
	proj = project_point_to_line(x, points[on_line_idx, ...], points[on_line_idx + 1, ...])
	if return_cl_indices:
		return proj, (cl_idx, on_line_idx)
	else:
		return proj

"""
def in_mesh_bounds(x, all_cls, n_candidates = 50):
	proj, (cl_idx, on_line_idx) = project_to_cl(x, all_cls, n_candidates, return_cl_indices = True)
	points, radiuses = all_cls[cl_idx]
	l_dist = np.linalg.norm(proj - points[on_line_idx, ...])
	r_dist = np.linalg.norm(proj - points[on_line_idx + 1, ...])
	radius = (radiuses[on_line_idx, ...] * l_dist + radiuses[on_line_idx + 1, ...] * r_dist) / (l_dist + r_dist)
	return np.linalg.norm(x - proj) <= radius
"""

def in_mesh_bounds(x, all_cls, cl_indices, tolerance = 0.5, n_candidates = 50):
	cl_idx, on_line_idx = cl_indices
	points, radiuses = all_cls[cl_idx]
	proj = project_point_to_line(x, points[on_line_idx, ...], points[on_line_idx + 1, ...], fix_oob = False)
	l_dist = np.linalg.norm(proj - points[on_line_idx, ...])
	r_dist = np.linalg.norm(proj - points[on_line_idx + 1, ...])
	radius = (radiuses[on_line_idx, ...] * l_dist + radiuses[on_line_idx + 1, ...] * r_dist) / (l_dist + r_dist)
	return np.linalg.norm(x - proj) <= radius + tolerance

def get_cl_direction(all_cls, cl_indices):
	cl_idx, on_line_idx = cl_indices
	points, radiuses = all_cls[cl_idx]
	res = points[on_line_idx + 1, ...] - points[on_line_idx, ...]
	res = res / np.linalg.norm(res)
	return res
