import numpy as np
from utils.geometry import random_points_in_sphere, random_perpendicular_offsets, \
	random_perpendicular_vectors, arbitrary_perpendicular_vector, rotate_all_degrees, rotate_single_vector, get_rotate_angle

def randomise_params(
	focal_base,
	axial_scale, radial_scale, position_base,
	orientation_scale, orientation_base,
	num_samples, up_samples):
	all_sampled_params = []
	orientation_offsets = random_points_in_sphere(num_samples, orientation_scale)
	axial_offsets = np.random.rand(num_samples) * axial_scale - axial_scale / 2
	for orientation_offset, axial_offset in zip(orientation_offsets, axial_offsets):
		t_orientation = orientation_base + orientation_offset
		t_orientation = t_orientation / np.linalg.norm(t_orientation)
		radial_offset = random_perpendicular_offsets(1, t_orientation, radial_scale)[0, ...]
		up_vector_base = arbitrary_perpendicular_vector(t_orientation)
		t_position = position_base + axial_offset * t_orientation + radial_offset
		for t_up in rotate_all_degrees(up_vector_base, t_orientation, up_samples):
			all_sampled_params.append((focal_base, t_position, t_orientation, t_up))
	return all_sampled_params

def random_move_depr(axial_scale, radial_scale, position_base, orientation_scale, orientation_base, up_rot_scale, up_rot_base, distribution = "norm"):
	if distribution == "uniform":
		orientation_offset = random_points_in_sphere(1, orientation_scale)[0, ...]
		t_orientation = orientation_base + orientation_offset
		t_orientation = t_orientation / np.linalg.norm(t_orientation)
		
		axial_offset = np.random.rand() * axial_scale - axial_scale / 2
		radial_offset = random_perpendicular_offsets(1, t_orientation, radial_scale)[0, ...]
		t_position = position_base + axial_offset * t_orientation + radial_offset
		
		up_vector_base = arbitrary_perpendicular_vector(t_orientation)
		t_up_rot = up_rot_base + np.random.rand() * up_rot_scale - up_rot_scale / 2
		t_up = rotate_single_vector(up_vector_base, t_orientation, t_up_rot + up_rot_offset)
	
	elif distribution == "normal":
		orientation_offset = random_points_in_sphere(1, in_sphere = False)[0, ...]
		orientation_offset *= np.random.randn() * orientation_scale
		t_orientation = orientation_base + orientation_offset
		t_orientation = t_orientation / np.linalg.norm(t_orientation)

		axial_offset = np.random.randn() * axial_scale
		radial_offset = random_perpendicular_vectors(1, t_orientation)[0, ...]
		radial_offset *= np.random.randn() * radial_scale
		t_position = position_base + axial_offset * t_orientation + radial_offset

		up_vector_base = arbitrary_perpendicular_vector(t_orientation)
		t_up_rot = up_rot_base + np.random.randn() * up_rot_scale
		t_up = rotate_single_vector(up_vector_base, t_orientation, t_up_rot)

	else:
		raise NotImplementedError

	return t_position, t_orientation, t_up, t_up_rot

def uniform_norm(scale, num_samples):
	return np.arange(1, num_samples + 1) * scale / num_samples

def random_move(orient_rot, orient_norm, radial_rot, radial_norm, axial_norm, up_rot, rot_scale, axial_scale, radial_scale, orientation_scale, up_rot_scale):
	t_orient_rot = orient_rot + np.random.randn() * rot_scale
	t_orient_norm = orient_norm + np.random.randn() * orientation_scale
	t_radial_rot = radial_rot + np.random.randn() * rot_scale
	t_radial_norm = radial_norm + np.random.randn() * radial_scale
	t_axial_norm = axial_norm + np.random.randn() * axial_scale
	t_up_rot = up_rot + np.random.randn() * up_rot_scale
	return t_orient_rot, t_orient_norm, t_radial_rot, t_radial_norm, t_axial_norm, t_up_rot

def move_by_params(position, orientation, orient_rot, orient_norm, radial_rot, radial_norm, axial_norm, up_rot):
	orient_perp = arbitrary_perpendicular_vector(orientation)
	#print(f"Radial: {radial_norm:.4f}, Axial: {axial_norm:.4f}.")
	t_position = position + orientation * axial_norm + rotate_single_vector(orient_perp, orientation, radial_rot) * radial_norm
	t_orientation = orientation + rotate_single_vector(orient_perp, orientation, orient_rot) * orient_norm
	t_orientation = t_orientation / np.linalg.norm(t_orientation)
	orient_perp = arbitrary_perpendicular_vector(t_orientation)
	up = rotate_single_vector(orient_perp, t_orientation, up_rot)
	#print(np.linalg.norm(position - b_position))
	return t_position, t_orientation, up

def get_radial_axial_offsets(position, orientation, t_position):
	position_offset = t_position - position
	axial_norm = np.dot(position_offset, orientation)
	radial_offset = position_offset - axial_norm * orientation
	radial_norm = np.linalg.norm(radial_offset)
	orient_perp = arbitrary_perpendicular_vector(orientation)
	radial_rot = get_rotate_angle(orient_perp, radial_offset / radial_norm, orientation)
	return radial_rot, radial_norm, axial_norm

def reverse_to_geno(position, orientation, t_position, t_orientation, t_up):
	full_len_orientation = t_orientation / np.dot(t_orientation, orientation)
	orient_offset = full_len_orientation - orientation
	orient_norm = np.linalg.norm(orient_offset)
	orient_perp = arbitrary_perpendicular_vector(orientation)
	orient_rot = get_rotate_angle(orient_perp, orient_offset / orient_norm, orientation)

	up_rot = get_rotate_angle(arbitrary_perpendicular_vector(t_orientation), t_up, t_orientation)
	return orient_rot, orient_norm, *get_radial_axial_offsets(position, orientation, t_position), up_rot

def uniform_sampling(rot_scale, axial_scale, radial_scale, orientation_scale, norm_samples, rot_samples, up_rot_samples):
	orient_combs = [(0, 0)]
	radial_combs = [(0, 0)]
	for t_rot in np.arange(0, 360, 360 / rot_samples):
		for o_norm in uniform_norm(orientation_scale, norm_samples):
			orient_combs.append((t_rot, o_norm))
		for r_norm in uniform_norm(radial_scale, norm_samples):
			radial_combs.append((t_rot, r_norm))

	res = []
	for o_comb in orient_combs:
		for r_comb in radial_combs:
			for a_norm in np.arange(-axial_scale / 2, axial_scale / 2, axial_scale / norm_samples):
				for u_rot in np.arange(0, 360, 360 / up_rot_samples):
					res.append((*o_comb, *r_comb, a_norm, u_rot))
	return res

def uniform_sampling_depr(axial_scale, axial_samples, radial_scale, radial_dir_samples, radial_norm_samples, position_base,
	orientation_scale, orientation_dir_samples, orientation_norm_samples, orientation_base, up_rot_scale, up_rot_samples):
	samples = []
	orientation_dir_offset_base = arbitrary_perpendicular_vector(orientation_base)
	all_orientations = [orientation_base]
	for orientation_dir_offset in rotate_all_degrees(orientation_dir_offset_base, orientation_base, orientation_dir_samples):
		for orientation_norm_offset in uniform_norm(orientation_scale, orientation_norm_samples):
			t_orientation = orientation_base + orientation_dir_offset * orientation_norm_offset
			all_orientations.append(t_orientation)

	for t_orientation in all_orientations:
		all_radial_comb_offsets = [np.array([0, 0, 0])]
		radial_dir_offset_base = arbitrary_perpendicular_vector(t_orientation)
		for radial_dir_offset in rotate_all_degrees(radial_dir_offset_base, t_orientation, radial_dir_samples):
			for radial_norm_offset in uniform_norm(radial_scale, radial_norm_samples):
				all_radial_comb_offsets.append(radial_dir_offset * radial_norm_offset)
		for axial_offset in np.arange(-axial_scale / 2, axial_scale / 2, axial_scale / axial_samples):
			for radial_comb_offset in all_radial_comb_offsets:
				t_position = position_base + axial_offset * t_orientation + radial_comb_offset
				up_base = arbitrary_perpendicular_vector(t_orientation)
				for deg, t_up in enumerate(rotate_all_degrees(up_base, t_orientation, up_rot_samples)):
					samples.append((t_position, t_orientation, t_up, deg * 360 / up_rot_samples))
	return samples
