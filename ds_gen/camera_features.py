import numpy as np
from scipy.stats import fatiguelife

# Camera parameters
camera_params = {}
camera_params["focal_length"] = 10
camera_params["view_angle"] = 90
camera_params["up_direction"] = np.array([0, 1, 0])
camera_params["forward_direction"] = np.array([0, 0, 1])
camera_params["clipping_range"] = (3, 100)

# Velocity
# Velocity norm
vn_loc = -0.0604376084806407
vn_scale = 0.4474117306519036
vn_const = 1.1016112671704779
def get_velocity_norm(size):
	return fatiguelife.rvs(vn_const, loc = vn_loc, scale = vn_scale, size = size)


# Not decided yet
def get_max_radial_offset(radius):
	return radius * 0.9

def get_max_orient_offset(radius):
	return 10
