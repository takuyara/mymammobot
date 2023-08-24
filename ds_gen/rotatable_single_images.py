import cv2
import numpy as np

from utils.cl_utils import load_all_cls, get_unique_cl_indices, get_direction_dist_radius
from utils.geometry import arbitrary_perpendicular_vector, rotate_single_vector
from utils.pose_utils import get_output_array
from ds_gen.depth_map_generation import get_zoomed_plotter, get_depth_map
from ds_gen.camera_features import get_max_radial_offset, get_max_orient_offset

def randu_gen(a, b):
	def randu():
		return np.random.rand() * (b - a) + a
	return randu

def generate_rotatable_images(mesh_path, cl_path, output_path, num_samples, img_size, zoom_scale = 2 ** 0.5, axial_extend_rate = 0.1, radial_safe_rate = 0.9):
	all_cls = load_all_cls(cl_path)
	unique_cl_indices = get_unique_cl_indices(all_cls)

	zoomed_plotter = get_zoomed_plotter(img_size, zoom_scale)

	total_volume = 0
	for cl_idx, on_line_idx in unique_cl_indices:
		if on_line_idx == len(all_cls[cl_idx][0]) - 1:
			continue
		cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
		total_volume += axial_len * lumen_radius ** 2

	print(f"Volume = {total_volume:.2f}.")

	all_sample_data = []
	for cl_idx, on_line_idx in unique_cl_indices:
		if on_line_idx == len(all_cls[cl_idx][0]) - 1:
			continue
		cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
		orient_perp = arbitrary_perpendicular_vector(cl_orientation)

		axial_norm_gen = randu_gen(- axial_extend_rate * axial_len, (1 + axial_extend_rate) * axial_len)
		radial_norm_gen = randu_gen(0, get_max_radial_offset(lumen_radius) * radial_safe_rate)
		orient_norm_gen = randu_gen(0, get_max_orient_offset(lumen_radius))
		angle_gen = randu_gen(0, 360)

		num_samples_on_this_cl = int(round((axial_len * lumen_radius ** 2) / total_volume * num_samples))

		for i in range(num_samples_on_this_cl):
			axial_norm = axial_norm_gen()
			radial_norm = radial_norm_gen()
			orient_norm = orient_norm_gen()

			t_position = cl_point_base + cl_orientation * axial_norm + rotate_single_vector(orient_perp, cl_orientation, angle_gen()) * radial_norm
			t_orientation = cl_orientation + rotate_single_vector(orient_perp, cl_orientation, angle_gen()) * orient_norm
			t_orientation = t_orientation / np.linalg.norm(t_orientation)

			all_sample_data.append((t_position, t_orientation))

	for i, (t_position, t_orientation) in enumerate(all_sample_data):
		t_up = arbitrary_perpendicular_vector(t_orientation)
		rgb, dep = get_depth_map(zoomed_plotter, t_position, t_orientation, t_up, get_outputs = True)
		out_pose = get_output_array(t_position, t_orientation)
		cv2.imwrite(rgb, os.path.join(output_path, f"{i:06d}.png"))
		np.save(dep, os.path.join(output_path, f"{i:06d}.npy"))
		np.savetxt(out_pose, os.path.join(output_path, f"{i:06d}.txt"), fmt = "%.6f")
