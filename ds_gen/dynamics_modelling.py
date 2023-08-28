import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from pose_fixing.move_camera import get_radial_axial_offsets
from ds_gen.camera_features import camera_params
from utils.pose_utils import camera_pose_to_train_pose
from utils.geometry import arbitrary_perpendicular_vector, rotate_single_vector
from utils.cl_utils import load_all_cls, find_all_possible_cls, get_direction_dist_radius, index2point

def get_velocities(poses, cl_path):
	positions, quaternions = poses[ : , : 3], poses[ : , 3 : ]
	res_data = []

	all_cls = load_all_cls(cl_path)
	for i in range(len(positions) - 1):
		min_next_radial_norm = 1e10
		for cl_indices in find_all_possible_cls(positions[i], all_cls):
			cl_base_point = index2point(all_cls, cl_indices)
			cl_orientation, __, lumen_radius = get_direction_dist_radius(all_cls, cl_indices)
			next_radial_rot, next_radial_norm, next_axial_norm = get_radial_axial_offsets(cl_base_point, cl_orientation, positions[i + 1])
			if next_radial_norm < min_next_radial_norm:
				min_next_radial_norm = next_radial_norm
				best_cl_data = cl_indices, cl_base_point, cl_orientation, lumen_radius, next_radial_rot, next_radial_norm, next_axial_norm
		cl_indices, cl_base_point, cl_orientation, lumen_radius, next_radial_rot, next_radial_norm, next_axial_norm = best_cl_data
		radial_rot, radial_norm, axial_norm = get_radial_axial_offsets(cl_base_point, cl_orientation, positions[i])
		orient_perp = arbitrary_perpendicular_vector(cl_orientation)
		radial_offset = rotate_single_vector(orient_perp, cl_orientation, radial_rot) * radial_norm
		next_radial_offset = rotate_single_vector(orient_perp, cl_orientation, next_radial_rot) * next_radial_norm
		radial_velocity = np.linalg.norm(radial_offset - next_radial_offset)
		axial_velocity = next_axial_norm - axial_norm
		rotation_velocity = R.from_quat(quaternions[i + 1]) * R.from_quat(quaternions[i]).inv()
		total_velocity = np.linalg.norm(positions[i] - positions[i + 1])
		res_data.append((radial_velocity, axial_velocity, total_velocity, rotation_velocity.as_quat(), lumen_radius))

		"""
		print(np.allclose(total_velocity, (radial_velocity ** 2 + axial_velocity ** 2) ** 0.5))
		this_orient = R.from_quat(quaternions[i]).apply(camera_params["forward_direction"])
		next_orient = R.from_quat(quaternions[i + 1]).apply(camera_params["forward_direction"])
		print(np.allclose(rotation_velocity.apply(this_orient), next_orient))
		"""

	return res_data


def smoothing_poses(positions, orientations, ups, method = "savgol"):
	if method == "savgol":
		savgol_params = {"window_length": 5, "polyorder": 1, "axis": 0}
		positions = savgol_filter(positions, **savgol_params)
		orientations = savgol_filter(orientations, **savgol_params)
		ups = savgol_filter(ups, **savgol_params)
	elif method == "none":
		pass
	else:
		raise NotImplementedError

	poses = []
	for i in range(len(positions)):
		this_pose = camera_pose_to_train_pose(positions[i], orientations[i], ups[i])
		poses.append(this_pose)
	return np.stack(poses, axis = 0)

