import numpy as np

def project_point_to_line(x, a, b, fix_oob = True):
	if np.linalg.norm(a - b) == 0:
		return a
	x_ = np.dot(x - a, b - a) / np.dot(b - a, b - a) * (b - a) + a
	if not fix_oob or np.allclose(np.linalg.norm(x_ - a) + np.linalg.norm(x_ - b), np.linalg.norm(a - b)):
		return x_
	else:
		return a if np.linalg.norm(x - a) < np.linalg.norm(x - b) else b

def random_points_in_sphere(num_points, radius = 1, in_sphere = True):
	vec = np.random.randn(num_points, 3)
	norms = (np.random.rand(num_points) if in_sphere else np.ones(num_points)) * radius
	norm_rates = norms / np.linalg.norm(vec, axis = 1)
	vec = vec * norm_rates.reshape(-1, 1)
	return vec

# Vector: input unit, output unit

def random_perpendicular_vectors(num_vectors, ref_vector):
	r = random_points_in_sphere(num_vectors, radius = 1, in_sphere = False)
	res_vector = r - np.sum(r * ref_vector, axis = 1).reshape(-1, 1) * ref_vector.reshape(1, 3)
	res_vector /= np.linalg.norm(res_vector, axis = 1).reshape(-1, 1)
	return res_vector

def random_perpendicular_offsets(num_offsets, ref_vector, max_offset):
	vectors = random_perpendicular_vectors(num_offsets, ref_vector)
	vectors *= (np.random.rand(num_offsets, 1) * max_offset)
	return vectors

def arbitrary_perpendicular_vector(ref_vector):
	r = np.array([1, 0, 0])
	res_vector = r - np.dot(ref_vector, r) * ref_vector
	return res_vector / np.linalg.norm(res_vector)

def get_rotation_matrix(axis, theta2):
	a = np.cos(theta2)
	b, c, d = -axis * np.sin(theta2)
	return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)], [2 * (b * c + a * d),
		a * a + c * c - b * b - d * d, 2 * (c * d - a * b)], [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

def rotate_all_degrees(base, axis, num_degrees, max_degree = 360):
	degrees = np.deg2rad(np.arange(num_degrees) * max_degree / num_degrees) / 2
	return [np.dot(get_rotation_matrix(axis, deg), base) for deg in degrees]

def rotate_single_vector(base, axis, theta_deg):
	res = np.dot(get_rotation_matrix(axis, np.deg2rad(theta_deg) / 2), base)
	return res / np.linalg.norm(res)

def get_vector_angle(a, b):
	return np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1, 1)))

def get_rotate_angle(src, tgt, axis):
	a = get_vector_angle(src, tgt)
	a1 = get_vector_angle(rotate_single_vector(src, axis, a), tgt)
	a2 = get_vector_angle(rotate_single_vector(src, axis, -a), tgt)
	return a if a1 < a2 else -a
