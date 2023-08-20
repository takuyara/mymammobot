import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
import cv2
import csv
import argparse
import time
from multiprocessing import Pool

from utils.camera_motion import camera_params
from utils.cl_utils import project_to_cl, load_all_cls, in_mesh_bounds, get_cl_direction
from utils.geometry import random_points_in_sphere, arbitrary_perpendicular_vector, rotate_single_vector, rotate_all_degrees, get_vector_angle, random_perpendicular_offsets
from domain_transfer.similarity import comb_corr_sim
#from domain_transfer.alignment import reg_depth_maps

default_params = {
	"coarse": {
		"position_scale": 7,
		"position_samples": 30,
		"orientation_scale": 0.3,
		"orientation_samples": 30,
		"up_samples": 10,
	},
	"fine": {
		"position_scale": 2,
		"position_samples": 15,
		"orientation_scale": 0.1,
		"orientation_samples": 15,
		"up_samples": 30,
	},
}

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--em-base-path", type = str, default = "./depth-images")
	parser.add_argument("--cl-base-path", type = str, default = "./CL")
	parser.add_argument("--output-metadata", type = str, default = "register_params.csv")
	parser.add_argument("--filter-oob", action = "store_true", default = False)
	parser.add_argument("--angle-threshold", type = float, default = 20)
	parser.add_argument("--em-idx", type = int, default = 0)
	parser.add_argument("--try-idx", type = int, default = 0)
	parser.add_argument("--init-idx", type = int, default = 0)
	parser.add_argument("--pool-size", type = int, default = 7)
	parser.add_argument("--step-size", type = int, default = 8)
	parser.add_argument("--window-size", type = int, default = 224)
	parser.add_argument("--focal-scale", type = float, default = 0)
	parser.add_argument("--position-scale", type = float, default = 7)
	parser.add_argument("--orientation-scale", type = float, default = 0.3)
	parser.add_argument("--focal-samples", type = int, default = 1)
	parser.add_argument("--position-samples", type = int, default = 30)
	parser.add_argument("--orientation-samples", type = int, default = 30)
	parser.add_argument("--up-samples", type = int, default = 10)
	parser.add_argument("--uses_new_rotation", action = "store_true", default = False)
	return parser.parse_args()

def get_depth_map(p, focal_length, camera_position, camera_orientation, up_direction, get_outputs = False, zoom = 1.0):
	camera = pv.Camera()
	#print(focal_length)
	camera.position = camera_position
	# camera.focal_point = camera_position + focal_length * camera_orientation / np.linalg.norm(camera_orientation)
	# camera.up = up_direction / np.linalg.norm(up_direction)
	camera.focal_point = camera_position + focal_length * camera_orientation
	#print("Before: ", focal_length)
	camera.up = up_direction
	camera.view_angle = camera_params["view_angle"]
	camera.clipping_range = camera_params["clipping_range"]
	camera.zoom(zoom)
	p.camera = camera
	p.show(auto_close = False)
	if get_outputs:
		return p.screenshot(None, return_img = True), -p.get_image_depth()
	else:
		return -p.get_image_depth()

def get_fixed_corr(ref_depth_map, p, focal_length, camera_position, camera_orientation, up_direction):
	depth_map = get_depth_map(p, focal_length, camera_position, camera_orientation, up_direction)
	corr = comb_corr_sim(ref_depth_map, depth_map)
	return corr, focal_length, camera_position, camera_orientation, up_direction

def randomise_params(
	focal_base,
	axial_scale, radial_scale, position_base,
	orientation_scale, orientation_base,
	num_samples):
	all_sampled_params = []
	orientation_offsets = random_points_in_sphere(num_samples, orientation_scale)
	axial_offsets = np.random.rand(num_samples) * axial_scale - axial_scale / 2
	for orientation_offset, axial_offset in zip(orientation_offsets, axial_offsets):
		t_orientation = orientation_base + orientation_offset
		t_orientation = t_orientation / np.linalg.norm(t_orientation)
		radial_offset = random_perpendicular_offsets(1, t_orientation, radial_scale)[0, ...]
		up_vector_base = arbitrary_perpendicular_vector(t_orientation)
		t_position = position_base + axial_offset * t_orientation + radial_offset
		all_sampled_params.append((focal_base, t_position, t_orientation, up_vector_base))
	return all_sampled_params


def fix_camera_pose(frame_idx, em_path, em_depth_path, output_path, args):
	surface = pv.read(args.mesh_path)
	zoom_scale = 2 ** -0.5
	half_angle = np.deg2rad(camera_params["view_angle"] / 2)
	size_change_rate = np.tan(half_angle / zoom_scale) / np.tan(half_angle)
	zoomed_size = int(args.window_size * size_change_rate)
	p1 = pv.Plotter(off_screen = True, window_size = (zoomed_size, zoomed_size))
	p1.add_mesh(surface)

	p = pv.Plotter(off_screen = True, window_size = (args.window_size, args.window_size))
	p.add_mesh(surface)
	
	real_depth_map = np.load(os.path.join(em_depth_path, f"{frame_idx:06d}.npy"))

	num_samples = 50
	angle_step_size = 20
	t_position = np.array([-33.79384209, -28.94875791, -160.28407521])
	t_orientation = np.array([-0.89581517, 0.32385322, -0.30435879])
	up_vector_base = np.array([0.43895517, 0.53761082, -0.71992567])

	"""
	t_up = rotate_single_vector(up_vector_base, t_orientation, 270)
	t_left = np.cross(t_up, t_orientation)
	t_left = t_left / np.linalg.norm(t_left)
	"""


	axial_scale = 0.5
	radial_scale = 7
	orientation_scale = 0.4
	
	
	all_sampled_params = randomise_params(camera_params["focal_length"],
		axial_scale, radial_scale, t_position, orientation_scale, t_orientation,
		num_samples)
	#best_corr_params = get_fixed_corr(real_depth_map, p, camera_params["focal_length"], t_position, t_orientation, up_vector_base)
	"""


	#plt.figure(figsize = (20, 15))

	left_offset = -2
	up_offset = 3
	axial_offset = 7
	up_rotate = -7
	ori_left_offset = -0.3
	ori_up_offset = 0.5
	new_position = t_position + left_offset * t_left + up_offset * t_up + axial_offset * t_orientation
	new_up = rotate_single_vector(t_up, t_orientation, up_rotate)
	new_orientation = t_orientation + t_up * ori_up_offset + t_left * ori_left_offset

	from sklearn.linear_model import LinearRegression as LR

	def plot_it(rgb, virtual_depth_map, capping_value = 40):
		rd1, vd1 = real_depth_map.reshape(-1, 1), virtual_depth_map.reshape(-1, 1)
		vd1 = np.minimum(vd1, capping_value)
		w = np.ones(rd1.shape[0])
		w[rd1.flatten() < 3] = 0.3
		reg = LR().fit(rd1, vd1, w)
		score = reg.score(rd1, vd1, w)
		err = (vd1 - reg.predict(rd1)) ** 2
		err = err.reshape(*virtual_depth_map.shape)
		plt.clf()
		plt.subplot(2, 2, 1)
		plt.imshow(real_depth_map, cmap = "gray")
		plt.subplot(2, 2, 2)
		plt.imshow(virtual_depth_map, cmap = "gray")
		plt.subplot(2, 2, 3)
		#plt.imshow(rgb)
		plt.imshow(err)
		plt.colorbar()
		plt.subplot(2, 2, 4)
		plt.scatter(real_depth_map.flatten(), virtual_depth_map.flatten())
		#plt.savefig(os.path.join(output_path, f"res_{left_offset}_{up_offset}_{this_corr}.png"))
		plt.suptitle(f"{score:.6f}")
		plt.show()
		return err
	"""

	from scipy import ndimage



	def centre_crop(orig_dep):
		centre_x, centre_y, half_size = orig_dep.shape[0] // 2, orig_dep.shape[1] // 2, args.window_size // 2
		rot_dep = orig_dep[centre_x - half_size : centre_x + half_size, centre_y - half_size : centre_y + half_size, ...]
		return rot_dep

	def get_rotation_plain(position, orientation, up, degree):
		return get_depth_map(p, camera_params["focal_length"], position, orientation, rotate_single_vector(up, orientation, 360 - degree))

	def get_rotation_new(orig_dep, degree):
		return centre_crop(ndimage.rotate(orig_dep, degree, reshape = False))

	"""
	orig_dep = get_depth_map(p1, camera_params["focal_length"], new_position, new_orientation, new_up, zoom = zoom_scale)
	for degree in range(0, 360, 10):
		p_dep = get_rotation_plain(new_position, new_orientation, new_up, degree)
		n_dep = get_rotation_new(orig_dep, degree)
		print(np.mean((p_dep - n_dep) ** 2))
	"""

	"""
	rgb, virtual_depth_map = get_depth_map(p, camera_params["focal_length"], new_position, new_orientation, new_up, get_outputs = True)
	err1 = plot_it(rgb, virtual_depth_map)
	rgb, virtual_depth_map = get_depth_map(p, camera_params["focal_length"], t_position, t_orientation, up_vector_base, get_outputs = True)
	err2 = plot_it(rgb, virtual_depth_map)
	plt.subplot(1, 2, 1)
	plt.hist(err1.flatten())
	plt.title("New pose error")
	plt.subplot(1, 2, 2)
	plt.hist(err2.flatten())
	plt.title("Old pose error")
	plt.show()

	exit()
	"""


	#this_corr = comb_corr_sim(real_depth_map, virtual_depth_map)

	"""
	n_trys = 50

	st_time = time.time()
	for i in range(n_trys):
		orig_dep = get_depth_map(p1, camera_params["focal_length"], t_position, t_orientation, up_vector_base)
	print("Renderring time: ", (time.time() - st_time) / n_trys)

	orig_dep = get_depth_map(p1, camera_params["focal_length"], t_position, t_orientation, up_vector_base, zoom = zoom_scale)
	st_time = time.time()
	for i in range(n_trys):
		cur_dep = get_rotation_new(orig_dep, 35)
	print("Rotation time: ", (time.time() - st_time) / n_trys)

	st_time = time.time()
	for i in range(n_trys):
		this_corr = comb_corr_sim(real_depth_map, cur_dep)
	print("Similarity time: ", (time.time() - st_time) / n_trys)

	exit()
	"""


	st_time = time.time()
	for i, (t_focal, t_position, t_orientation, t_up) in enumerate(all_sampled_params):
		orig_dep = get_depth_map(p1, camera_params["focal_length"], t_position, t_orientation, t_up, zoom = zoom_scale)
		for degree in range(0, 360, angle_step_size):
			#cur_rgb, cur_dep = get_rotation_new(orig_rgb, degree), get_rotation_new(orig_dep, degree)
			if args.uses_new_rotation:
				cur_dep = get_rotation_new(orig_dep, degree)
			else:
				cur_dep = get_depth_map(p, camera_params["focal_length"], t_position, t_orientation, t_up)
			this_corr = comb_corr_sim(real_depth_map, cur_dep)
			"""
			if this_corr > 0.7:
				plt.clf()
				plt.subplot(2, 2, 1)
				plt.imshow(real_depth_map, cmap = "gray")
				plt.subplot(2, 2, 2)
				plt.imshow(cur_dep, cmap = "gray")
				plt.subplot(2, 2, 3)
				plt.imshow(get_rotation_new(orig_rgb, degree))
				plt.subplot(2, 2, 4)
				plt.scatter(real_depth_map.flatten(), cur_dep.flatten())
				plt.savefig(os.path.join(output_path, f"{frame_idx:06d}-{this_corr:.4f}.png"))
			"""
	
	print("Average time: ", (time.time() - st_time) / (len(all_sampled_params) * (360 // angle_step_size)))
	#p1.show()


def main():
	args = get_args()
	args.em_idx = 1
	frame_idx = 1416
	args.try_idx = "non-specific-tryidx"
	output_path = os.path.join(args.em_base_path, f"EM-virtual-autofix-{args.em_idx}-{frame_idx}-{args.try_idx}")
	em_path = os.path.join(args.em_base_path, f"EM-{args.em_idx}")
	em_depth_path = os.path.join(args.em_base_path, f"EM-rawdep-{args.em_idx}")
	os.makedirs(output_path, exist_ok = True)

	"""
	focal_length = 10
	translation = np.array([-68.30139041, -36.53458994, -163.48089302])
	orientation = np.array([-0.9871122, 0.12990399, -0.09345835])
	up_direction = np.array([-0.0494519, -0.80303439, -0.59387732])
	surface = pv.read(args.mesh_path)
	p = pv.Plotter(off_screen = True, window_size = (args.window_size, args.window_size))
	p.add_mesh(surface)
	rgb_img, __ = get_depth_map(p, focal_length, translation, orientation, up_direction, get_outputs = True)
	plt.imshow(rgb_img)
	plt.show()
	"""

	for frame_idx in range(20):
		fix_camera_pose(frame_idx, em_path, em_depth_path, output_path, args)

	#fix_single_frame(0, em_path, em_depth_path, output_path, args)

if __name__ == '__main__':
	main()
