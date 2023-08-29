import os
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from utils.misc import randu_gen
from utils.geometry import arbitrary_perpendicular_vector, rotate_single_vector
from pose_fixing.move_camera import get_radial_axial_offsets
from ds_gen.camera_features import camera_params, dynamic_total_velocity_gen, dynamic_radial_norm_velocity_gen, dynamic_axial_direction_gen, dynamic_rotation_velocity_gen, is_valid_focal_radial_norm, is_valid_radial_norm
from utils.cl_utils import load_all_cls, get_unique_between_pairs, get_direction_radius_dynamics, project_to_line_dynamics

def generate_video_sequence(mesh_path, cl_path, output_path, num_samples, img_size, turning_buffer = 10, smoothing_dist = 5, cut_ends = 5):
	p = pv.Plotter(off_screen = True, window_size = (img_size, img_size))
	p.add_mesh(pv.read(mesh_path))

	angle_gen = randu_gen(0, 360)
	axial_direction_gen = dynamic_axial_direction_gen()
	rotation_velocity_gen = dynamic_rotation_velocity_gen()
	all_cls = load_all_cls(cl_path)
	target_tubes = []
	for i in range(len(all_cls)):
		target_tubes.append((all_cls[i][0], all_cls[i][1], i, i))
	
	"""
	for i in range(len(all_cls) - 1):
		for j in range(i + 1, len(all_cls)):
			target_tubes.append((*get_unique_between_pairs(all_cls, i, j, turning_buffer), i, j))
	"""

	for this_sample in range(num_samples):
		for base_points, base_radius, cl_idx_1, cl_idx_2 in target_tubes:
			for direction in range(2):
				if direction == 1:
					points, radius = np.flip(base_points), np.flip(base_radius)
				else:
					points, radius = base_points, base_radius
				points, radius = points[cut_ends : ], radius[cut_ends : ]
				cur_idx = 0
				cur_position, cur_orientation = points[cur_idx, ...], get_direction_radius_dynamics(points, radius, cur_idx, smoothing_dist)[0]
				cur_up = rotate_single_vector(arbitrary_perpendicular_vector(cur_orientation), cur_orientation, angle_gen())
				poses = []
				while len(points) - cur_idx > cut_ends:
					cl_base_point, cl_orientation, lumen_radius = points[cur_idx, ...], *get_direction_radius_dynamics(points, radius, cur_idx, smoothing_dist)
					cl_orient_perp = arbitrary_perpendicular_vector(cl_orientation)
					velocity_gen = dynamic_total_velocity_gen(lumen_radius)
					radial_norm_gen = dynamic_radial_norm_velocity_gen(lumen_radius)
					radial_rot, radial_norm, axial_norm = get_radial_axial_offsets(cl_base_point, cl_orientation, cur_position)
					#print(cur_position, radial_norm, axial_norm)
					while True:
						cur_total_velocity = velocity_gen()
						cur_radial_norm = radial_norm_gen()
						if cur_radial_norm > cur_total_velocity:
							if cur_idx == 112:
								#print("Velocity failed.")
								pass
							continue
						cur_axial_norm = (cur_total_velocity ** 2 - cur_radial_norm ** 2) ** 0.5
						next_position = cur_position + cur_radial_norm * rotate_single_vector(cl_orient_perp, cl_orientation, angle_gen()) + cl_orientation * axial_direction_gen() * cur_axial_norm
						next_idx, next_radius, dist = project_to_line_dynamics(points, radius, next_position, cur_idx)
						if dist > next_radius:
							if cur_idx == 112:
								#print("OOB.")
								pass
							continue
						rotation_velocity = R.from_quat(rotation_velocity_gen())
						next_orientation, next_up = rotation_velocity.apply(cur_orientation), rotation_velocity.apply(cur_up)
						fixed_focal_length = camera_params["focal_length"] / np.dot(cl_orientation, next_orientation)
						fixed_focal_point = next_position + next_orientation * fixed_focal_length
						__, f_radial_norm, ___ = get_radial_axial_offsets(cl_base_point, cl_orientation, fixed_focal_point)
						__, t_radial_norm, ___ = get_radial_axial_offsets(cl_base_point, cl_orientation, cur_position)
						if is_valid_focal_radial_norm(f_radial_norm, lumen_radius) and is_valid_radial_norm(t_radial_norm, lumen_radius):
							#print("Orientation failed.")
							break
					poses.append(np.stack([next_position, next_orientation, next_up], axis = 0))
					cur_idx = next_idx
					cur_position, cur_orientation, cur_up = next_position, next_orientation, next_up
				poses = np.stack(poses, axis = 0)
				print(poses.shape)
				np.save(os.path.join(output_path, f"{this_sample}-{cl_idx_1}-{cl_idx_2}-{direction}.npy"), poses)
