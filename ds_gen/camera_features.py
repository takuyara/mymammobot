import csv
import numpy as np
from scipy.stats import fatiguelife

from utils.misc import randu_gen, str_to_arr

# Camera parameters
camera_params = {}
camera_params["focal_length"] = 10
camera_params["view_angle"] = 90
camera_params["up_direction"] = np.array([0, 1, 0])
camera_params["forward_direction"] = np.array([0, 0, 1])
camera_params["clipping_range"] = (3, 100)

rotation_velocity_list = []
with open("velocity_res.csv", newline = "") as f:
	reader = csv.DictReader(f)
	for row in reader:
		rotation_velocity_list.append(str_to_arr(row["rotation_velocity"]))

# Velocity
# Velocity norm
vn_loc = -0.0604376084806407
vn_scale = 0.4474117306519036
vn_const = 1.1016112671704779
def get_velocity_norm(size):
	return fatiguelife.rvs(vn_const, loc = vn_loc, scale = vn_scale, size = size)

# Not decided yet
def static_radial_offset_gen(radius):
	return randu_gen(0, radius * 0.9)

def static_focal_radial_offset_gen(radius):
	return randu_gen(0, 10)

def dynamic_total_velocity_gen(radius):
	return randu_gen(0, 10)

def dynamic_radial_norm_velocity_gen(radius):
	return randu_gen(0, radius * 1.3)

def dynamic_axial_direction_gen():
	neg_rate = 0.2
	def dir_gen():
		if np.random.rand() < neg_rate:
			return -1
		else:
			return 1
	return dir_gen

def dynamic_rotation_velocity_gen():
	def r_vel_gen():
		idx = np.random.randint(len(rotation_velocity_list))
		return rotation_velocity_list[idx]
	return r_vel_gen

def is_valid_focal_radial_norm(focal_radial_norm, lumen_radius):
	return focal_radial_norm < 10

def is_valid_radial_norm(radial_norm, lumen_radius):
	#return radial_norm < lumen_radius
	return True