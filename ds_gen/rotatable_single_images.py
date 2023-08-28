import os
import cv2
import numpy as np
import pyvista as pv
from scipy import ndimage

import matplotlib.pyplot as plt

from utils.cl_utils import load_all_cls, get_unique_cl_indices, get_direction_dist_radius, index2point
from utils.geometry import arbitrary_perpendicular_vector, rotate_single_vector
from ds_gen.depth_map_generation import get_zoomed_plotter, get_depth_map
from ds_gen.camera_features import get_max_radial_offset, get_max_orient_offset, camera_params

def rotate_and_crop(img, deg, img_size):
	# Clockwise
	if not np.allclose(deg, 0):
		img = ndimage.rotate(img, -deg, reshape = False)
	st = (img.shape[0] - img_size) // 2
	return img[st : st + img_size, st : st + img_size]

def randu_gen(a, b):
	def randu():
		return np.random.rand() * (b - a) + a
	return randu

def generate_rotatable_images(mesh_path, cl_path, output_path, num_samples, img_size, out_pose_only = False, zoom_scale = 2 ** -0.5, axial_extend_rate = 0.1, radial_safe_rate = 0.9):	
	all_cls = load_all_cls(cl_path)
	unique_cl_indices = get_unique_cl_indices(all_cls)

	zoomed_plotter = get_zoomed_plotter(img_size, zoom_scale)
	surface = pv.read(mesh_path)
	zoomed_plotter.add_mesh(surface)

	total_volume = 0
	for cl_idx, on_line_idx in unique_cl_indices:
		if on_line_idx == len(all_cls[cl_idx][0]) - 1:
			continue
		cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
		total_volume += axial_len * lumen_radius ** 2

	print(f"Volume = {total_volume:.2f}.")

	#total_volume = 18000

	img_idx = 0
	for cl_idx, on_line_idx in unique_cl_indices:
		if on_line_idx == len(all_cls[cl_idx][0]) - 1:
			continue
		cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
		cl_point_base = index2point(all_cls, (cl_idx, on_line_idx))
		if abs(axial_len) < 1e-5:
			continue

		orient_perp = arbitrary_perpendicular_vector(cl_orientation)

		axial_norm_gen = randu_gen(- axial_extend_rate * axial_len, (1 + axial_extend_rate) * axial_len)
		radial_norm_gen = randu_gen(0, get_max_radial_offset(lumen_radius) * radial_safe_rate)
		focal_radial_norm_gen = randu_gen(0, get_max_orient_offset(lumen_radius))
		angle_gen = randu_gen(0, 360)

		num_samples_on_this_cl = int(round((axial_len * lumen_radius ** 2) / total_volume * num_samples))

		#print(cl_orientation)

		for i in range(num_samples_on_this_cl):
			axial_norm = axial_norm_gen()
			radial_norm = radial_norm_gen()
			focal_radial_norm = focal_radial_norm_gen()

			"""
			print(i, axial_norm, radial_norm, focal_radial_norm)

			axial_norm = 0
			radial_norm = 0
			focal_radial_norm = 10
			t = 56
			"""

			t_position = cl_point_base + cl_orientation * axial_norm + rotate_single_vector(orient_perp, cl_orientation, angle_gen()) * radial_norm
			t_focal_point = cl_point_base + cl_orientation * (camera_params["focal_length"] + axial_norm) + rotate_single_vector(orient_perp, cl_orientation, angle_gen()) * focal_radial_norm

			#t_focal_point = cl_point_base + cl_orientation * (camera_params["focal_length"] + axial_norm) + rotate_single_vector(orient_perp, cl_orientation, t) * focal_radial_norm
			#t_focal_point = cl_point_base + cl_orientation * (camera_params["focal_length"] + axial_norm)
			t_orientation = t_focal_point - t_position
			t_orientation = t_orientation / np.linalg.norm(t_orientation)


			t_up = arbitrary_perpendicular_vector(t_orientation)

			rgb, dep = get_depth_map(zoomed_plotter, t_position, t_orientation, t_up, get_outputs = True, zoom = zoom_scale)
			"""
			p.camera.position = t_position
			p.camera.focal_point = t_position + camera_params["focal_length"] * t_orientation

			plt.imshow(rgb)
			plt.show()
			exit()
			
			p.add_mesh(pv.Arrow(t_position, cl_orientation), color = "blue")
			p.add_mesh(pv.Arrow(t_position, rotate_single_vector(orient_perp, cl_orientation, t) * focal_radial_norm), color = "red")
			p.add_mesh(pv.Arrow(t_position, t_orientation), color = "green")
			p.show()

			"""

			if abs(np.max(np.min(rgb, axis = -1)) - 255) < 1e-2:
				continue

			out_pose = np.stack([t_position, t_orientation], axis = 0)
			np.savetxt(os.path.join(output_path, f"{img_idx:06d}.txt"), out_pose, fmt = "%.6f")
			if not out_pose_only:
				cv2.imwrite(os.path.join(output_path, f"{img_idx:06d}.png"), rgb)
				np.save(os.path.join(output_path, f"{img_idx:06d}.npy"), dep)
			
			img_idx += 1

