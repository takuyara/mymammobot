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

def load_all_cls(base_path, n_cls = 50):
	all_cls = []
	for cl_idx in range(n_cls):
		points, radiuses = load_cl(base_path, cl_idx)
		if points is not None:
			all_cls.append((points, radiuses))
	return all_cls

def load_all_cls_npy(base_path, n_cls = 50):
	all_cls = []
	for cl_idx in range(n_cls):
		path = os.path.join(base_path, f"CL{cl_idx}.npy")
		if os.path.exists(path):
			x = np.load(path)
			all_cls.append((x[ : , : 3], x[ : , 3 : ].reshape(-1)))
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

def get_direction_radius_dynamics(points, radius, idx, smoothing_dist = 5):
	if idx == len(points) - 1:
		idx = idx - 1
	orient = points[min(idx + smoothing_dist, len(points) - 1), ...] - points[idx, ...]
	orient = orient / np.linalg.norm(orient)
	return orient, min(radius[idx], radius[idx + 1])

def index2point(all_cls, cl_indices):
	return all_cls[cl_indices[0]][0][cl_indices[1], ...]

def check_proj_on_cl_seqs(x, cl_seqs_p, tolerance, n_candidates = 10):
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

def get_unique_cl_indices(all_cls, tolerance = 0.1, flatten = True):
	cl_seqs = []
	cl_seqs_p = []
	current_buffer = []
	current_buffer_p = []
	num_total_points = 0
	for cl_idx, this_cl in enumerate(all_cls):
		this_cl_points = this_cl[0]
		for on_line_idx, point in enumerate(this_cl_points):
			num_total_points += 1
			if not check_proj_on_cl_seqs(point, cl_seqs_p, tolerance):
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
	return cl_seq_flatten if flatten else cl_seqs

def get_first_diff(pts1, pts2, tolerance = 0.1, n_candidates = 20, init_ignore = 100):
	pts2 = pts2[init_ignore : ]
	for on_line_idx_1, pt1 in enumerate(pts1[init_ignore : ]):
		cl_dists = np.sum((pt1 - pts2) ** 2, axis = 1)
		for i in np.argsort(cl_dists)[ : n_candidates]:
			if i + 1 >= len(pts2):
				continue
			x_ = project_point_to_line(pt1, pts2[i, ...], pts2[i + 1, ...])
			if np.linalg.norm(x_ - pt1) < tolerance:
				return on_line_idx_1 + init_ignore
	return None

def get_unique_between_pairs(all_cls, cl_idx_1, cl_idx_2, turning_buffer = 10):
	unq_idx_1 = get_first_diff(all_cls[cl_idx_1][0], all_cls[cl_idx_2][0])
	unq_idx_2 = get_first_diff(all_cls[cl_idx_2][0], all_cls[cl_idx_1][0])
	st_idx_1 = max(0, unq_idx_1 - turning_buffer)
	st_idx_2 = max(0, unq_idx_2 - turning_buffer)
	pt1, rd1 = all_cls[cl_idx_1][0][st_idx_1 : ], all_cls[cl_idx_1][1][st_idx_1 : ]
	pt2, rd2 = all_cls[cl_idx_2][0][st_idx_2 : ], all_cls[cl_idx_2][1][st_idx_2 : ]
	r_points = np.concatenate([np.flip(pt1), pt2], axis = 0)
	r_radius = np.concatenate([np.flip(rd1), rd2], axis = 0)
	return r_points, r_radius

def project_to_line_dynamics(points, radius, x, cur_idx, n_candidates = 30):
	min_dist = 1e10
	for i in range(max(cur_idx - n_candidates // 2, 0), min(cur_idx + n_candidates // 2, len(points) - 1)):
		x_ = project_point_to_line(x, points[i, ...], points[i + 1, ...])
		if np.linalg.norm(x_ - x) < min_dist:
			min_dist, ret_idx = np.linalg.norm(x_ - x), i
	return i, radius[i], min_dist

def find_max_intersection(points_1, points_2, tolerance = 0.6):
	flag1, flag2 = np.zeros(len(points_1)), np.zeros(len(points_2))
	for i, p1 in enumerate(points_1):
		if check_proj_on_cl_seqs(p1, [points_2], tolerance):
			flag1[i] = 1
	for i, p2 in enumerate(points_2):
		if check_proj_on_cl_seqs(p2, [points_1], tolerance):
			flag2[i] = 1
	segs1, segs2 = get_continuous_segs(flag1, target = 1), get_continuous_segs(flag2, target = 1)
	if len(segs1) == 0 or len(segs2) == 0:
		return (-1, -1), (-1, -1)
	range1 = max(segs1, key = lambda x : x[1] - x[0])
	range2 = max(segs2, key = lambda x : x[1] - x[0])
	return range1, range2

def get_according_splits(orig_seg, range_list):
	res = []
	for l, r in range_list:
		if r - l > 0:
			res.append((orig_seg[0][l : r, ...], orig_seg[1][l : r, ...]))
	return res

def get_continuous_segs(a, target):
	prev_idx = -1
	res = []
	for i in range(len(a)):
		if a[i] != target:
			if i - prev_idx - 1 > 0:
				res.append((prev_idx + 1, i))
			prev_idx = i
	if len(a) - prev_idx - 1 > 0:
		res.append((prev_idx + 1, len(a)))
	return res

def search_tree(p, segs, dist_threshold = 0.5, get_minimal = False):
	pass

def cut_cls(all_cls, min_len = 30, get_tree_struct = False):
	current_segs = []
	root_avgs = []
	for cl_idx in range(len(all_cls)):
		root_avgs.append(all_cls[cl_idx][0][0, ...])
		overlap_flags = np.zeros(len(all_cls[cl_idx][0]))
		for comp_idx in range(len(current_segs)):
			(int_comp_l, int_comp_r), (int_cur_l, int_cur_r) = find_max_intersection(current_segs[comp_idx][0], all_cls[cl_idx][0])
			overlap_flags[int_cur_l : int_cur_r] = 1
			if int_comp_r - int_comp_l > 0:
				split_comp = get_according_splits(current_segs[comp_idx], [(int_comp_l, int_comp_r), (0, int_comp_l), (int_comp_r, len(current_segs[comp_idx][0]))])
				current_segs[comp_idx] = split_comp[0]
				current_segs.extend(split_comp[1 : ])
		cur_unique_indices = get_continuous_segs(overlap_flags, target = 0)
		#print(cur_unique_indices)
		current_segs.extend(get_according_splits(all_cls[cl_idx], cur_unique_indices))
	filtered_segs = [cur_seg for cur_seg in current_segs if len(cur_seg[0]) >= min_len]
	if get_tree_struct:
		root_avgs = np.mean(np.stack(root_avgs, axis = 0), axis = 0)
		raise NotImplementedError

	#print(current_segs)
	return filtered_segs

def get_cl_dist_sum(all_cls):
	all_sums = []
	for cl_idx in range(len(all_cls)):
		sum_axials = []
		sum_axial = 0
		points = all_cls[cl_idx][0]
		for on_line_idx in range(len(points) - 1):
			axial_len = np.linalg.norm(points[on_line_idx, ...] - points[on_line_idx + 1, ...])
			sum_axial += axial_len
			sum_axials.append(sum_axial)
		all_sums.append(sum_axials)
	return all_sums

def axial_to_cl_point_ori(all_cls, all_sums, cl_idx, axial_len):
	on_line_idx = np.searchsorted(all_sums[cl_idx], axial_len, side = "right")[0]
	residual_len = axial_len - all_sums[cl_idx][on_line_idx]
	direction, __, ___ = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
	pt = all_cls[cl_idx][0][on_line_idx, ...] + direction * residual_len
	return pt, direction
