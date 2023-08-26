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

def find_all_possible_cls(x, all_cls, n_candidates = 50, tolerance = 2):
	res_list = []
	for cl_idx, (cl_points, cl_radiuses) in enumerate(all_cls):
		cl_dists = np.sum((cl_points - x) ** 2, axis = 1)
		this_cl_optim = 1e10
		for i in np.argsort(cl_dists)[ : n_candidates]:
			if i + 1 >= len(cl_points):
				continue
			x_ = project_point_to_line(x, cl_points[i, ...], cl_points[i + 1, ...])
			t_dst = np.linalg.norm(x - x_)
			if t_dst < this_cl_optim:
				this_cl_optim, this_cl_optim_index = t_dst, i
		res_list.append((this_cl_optim, cl_idx, this_cl_optim_index))
	res_list.sort()
	res_indices = []
	for t_dst, cl_idx, on_line_idx in res_list:
		if t_dst - res_list[0][0] < tolerance:
			res_indices.append((cl_idx, on_line_idx))
		else:
			break
	return res_indices

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

def get_cl_radius(x, all_cls, cl_indices):
	cl_idx, on_line_idx = cl_indices
	points, radiuses = all_cls[cl_idx]
	proj = project_point_to_line(x, points[on_line_idx, ...], points[on_line_idx + 1, ...], fix_oob = False)
	l_dist = np.linalg.norm(proj - points[on_line_idx, ...])
	r_dist = np.linalg.norm(proj - points[on_line_idx + 1, ...])
	radius = (radiuses[on_line_idx, ...] * l_dist + radiuses[on_line_idx + 1, ...] * r_dist) / (l_dist + r_dist)
	return radius

def in_mesh_bounds(x, all_cls, cl_indices, tolerance = 0.5):
	radius = get_cl_radius(x, all_cls, cl_indices)
	return np.linalg.norm(x - proj) <= radius + tolerance

def get_cl_direction(all_cls, cl_indices):
	cl_idx, on_line_idx = cl_indices
	points, radiuses = all_cls[cl_idx]
	res = points[on_line_idx + 1, ...] - points[on_line_idx, ...]
	res = res / np.linalg.norm(res)
	return res

def get_direction_dist_radius(all_cls, cl_indices, smoothing_dist = 5):
	cl_idx, on_line_idx = cl_indices
	cl_point_base = all_cls[cl_idx][0][on_line_idx, ...]
	cl_next_point = all_cls[cl_idx][0][on_line_idx + 1, ...]
	lumen_radius = min(all_cls[cl_idx][1][on_line_idx], all_cls[cl_idx][1][on_line_idx + 1])
	axial_len = np.linalg.norm(cl_next_point - cl_point_base)
	cl_next_point_smooth = all_cls[cl_idx][0][min(on_line_idx + smoothing_dist, len(all_cls[cl_idx][0]) - 1), ...]
	cl_orientation = cl_next_point_smooth - cl_point_base
	cl_orientation /= np.linalg.norm(cl_orientation)
	return cl_orientation, axial_len, lumen_radius

def index2point(all_cls, cl_indices):
	return all_cls[cl_indices[0]][0][cl_indices[1], ...]

def check_proj_on_cl_seqs(x, all_cls, cl_seqs_p, tolerance, n_candidates = 10):
	for cl_points in cl_seqs_p:
		cl_points = np.array(cl_points)
		cl_dists = np.sum((cl_points - x) ** 2, axis = 1)
		for i in np.argsort(cl_dists)[ : n_candidates]:
			if i + 1 >= len(cl_points):
				continue
			x_ = project_point_to_line(x, cl_points[i, ...], cl_points[i + 1, ...])
			if np.linalg.norm(x_ - x) < tolerance:
				return True
	return False

def get_unique_cl_indices(all_cls, tolerance = 0.1):
	cl_seqs = []
	cl_seqs_p = []
	current_buffer = []
	current_buffer_p = []
	num_total_points = 0
	for cl_idx, this_cl in enumerate(all_cls):
		this_cl_points = this_cl[0]
		for on_line_idx, point in enumerate(this_cl_points):
			num_total_points += 1
			if not check_proj_on_cl_seqs(point, all_cls, cl_seqs_p, tolerance):
				current_buffer.append((cl_idx, on_line_idx))
				current_buffer_p.append(point)
			else:
				if len(current_buffer) > 0:
					cl_seqs.append(current_buffer)
					cl_seqs_p.append(current_buffer_p)
					current_buffer = []
					current_buffer_p = []
		if len(current_buffer) > 0:
			cl_seqs.append(current_buffer)
			cl_seqs_p.append(current_buffer_p)
			current_buffer = []
			current_buffer_p = []
	cl_seq_flatten = []
	for cl_seq in cl_seqs:
		for cl_indices in cl_seq:
			cl_seq_flatten.append(cl_indices)
	print(f"Survived: {len(cl_seq_flatten)} / {num_total_points}.")
	return cl_seq_flatten
