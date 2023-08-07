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
from utils.cl_geometry import project_to_cl, load_all_cls, random_points_in_sphere, random_perpendicular_vector, in_mesh_bounds
from domain_transfer.similarity import comb_corr_sim

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--em-base-path", type = str, default = "./depth-images")
	parser.add_argument("--cl-base-path", type = str, default = "./CL")
	parser.add_argument("--output-metadata", type = str, default = "register_params.csv")
	parser.add_argument("--ignore-oob", action = "store_true", default = False)
	parser.add_argument("--try-idx", type = int, default = 0)
	parser.add_argument("--pool-size", type = int, default = 10)
	parser.add_argument("--em-idx", type = int, default = 0)
	parser.add_argument("--step-size", type = int, default = 8)
	parser.add_argument("--window-size", type = int, default = 224)
	parser.add_argument("--focal-scale", type = float, default = 0)
	parser.add_argument("--position-scale", type = float, default = 7)
	parser.add_argument("--orientation-scale", type = float, default = 0.3)
	parser.add_argument("--focal-samples", type = int, default = 1)
	parser.add_argument("--position-samples", type = int, default = 40)
	parser.add_argument("--orientation-samples", type = int, default = 25)
	parser.add_argument("--up-samples", type = int, default = 20)
	return parser.parse_args()

def get_depth_map(p, focal_length, camera_position, camera_orientation, up_direction, get_outputs = False):
	camera = pv.Camera()
	camera.position = camera_position
	camera.focal_point = camera_position + focal_length * camera_orientation / np.linalg.norm(camera_orientation)
	camera.up = up_direction / np.linalg.norm(up_direction)
	camera.view_angle = camera_params["view_angle"]
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
	focal_scale, focal_samples, focal_base,
	position_scale, position_samples, position_base,
	orientation_scale, orientation_samples, orientation_base,
	up_samples):
	all_sampled_params = []
	focal_offsets = np.random.rand(focal_samples) * focal_scale - focal_scale / 2
	position_offsets = random_points_in_sphere(position_samples, position_scale)
	orientation_offsets = random_points_in_sphere(orientation_samples, orientation_scale)
	for focal_offset in focal_offsets:
		for position_offset in position_offsets:
			for orientation_offset in orientation_offsets:
				t_focal = focal_offset + focal_base
				t_position = position_offset + position_base
				t_orientation = orientation_base + orientation_offset
				t_orientation = t_orientation / np.linalg.norm(t_orientation)
				up_vectors = random_perpendicular_vector(up_samples, t_orientation)
				for t_up in up_vectors:
					all_sampled_params.append((t_focal, t_position, t_orientation, t_up))
	return all_sampled_params

def fix_single_frame(frame_idx, em_path, em_depth_path, output_path, args):
	surface = pv.read(args.mesh_path)
	p = pv.Plotter(off_screen = True, window_size = (args.window_size, args.window_size))
	p.add_mesh(surface)
	all_cls = load_all_cls(args.cl_base_path)

	raw_position = np.loadtxt(os.path.join(em_path, f"{frame_idx:06d}.txt")).reshape(-1)
	real_depth_map = np.load(os.path.join(em_depth_path, f"{frame_idx:06d}.npy"))
	translation, quaternion = raw_position[ : 3], raw_position[3 : ]
	translation, cl_indices = project_to_cl(translation, all_cls, return_cl_indices = True)
	orientation = R.from_quat(quaternion).apply(camera_params["forward_direction"])
	orientation = orientation / np.linalg.norm(orientation)
	up_direction = R.from_quat(quaternion).apply(camera_params["up_direction"])
	up_direction = up_direction / np.linalg.norm(up_direction)

	all_sampled_params = randomise_params(args.focal_scale, args.focal_samples, camera_params["focal_length"],
		args.position_scale, args.position_samples, translation, args.orientation_scale, args.orientation_samples, orientation, args.up_samples)
	best_corr_params = get_fixed_corr(real_depth_map, p, camera_params["focal_length"], translation, orientation, up_direction)
	better_params_found = False

	#p1 = pv.Plotter()
	#p1.add_mesh(surface, opacity = 0.5)

	st_time = time.time()
	n_oob = 0

	for i, (t_focal, t_position, t_orientation, t_up) in enumerate(all_sampled_params):
		#p1.add_mesh(pv.Arrow(t_position, t_orientation), color = "red")
		#p1.add_mesh(pv.Arrow(t_position, t_up), color = "green")
		if not args.ignore_oob and not in_mesh_bounds(t_position, all_cls, cl_indices):
			n_oob += 1
			continue
		this_corr_params = get_fixed_corr(real_depth_map, p, t_focal, t_position, t_orientation, t_up)
		if this_corr_params[0] > best_corr_params[0]:
			best_corr_params = this_corr_params
			better_params_found = True
	
	print("{:.2f} trys per second.".format(len(all_sampled_params) / (time.time() - st_time), n_oob / len(all_sampled_params)))
	# 185 trys per second
	#p1.show()

	rgb_img, dep_img = get_depth_map(p, *best_corr_params[1 : ], get_outputs = True)
	cv2.imwrite(os.path.join(output_path, f"{frame_idx:06d}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
	np.save(os.path.join(output_path, f"{frame_idx:06d}.npy"), dep_img)

	with open(args.output_metadata, "a", newline = "") as f:
		writer = csv.writer(f)
		writer.writerow([args.em_idx, frame_idx] + list(best_corr_params) + ["Y" if better_params_found else "N"])

	print(f"Frame {frame_idx} done.", flush = True)

def main():
	args = get_args()
	output_path = os.path.join(args.em_base_path, f"EM-virtual-autofix-{args.em_idx}-{args.try_idx}")
	em_path = os.path.join(args.em_base_path, f"EM-{args.em_idx}")
	em_depth_path = os.path.join(args.em_base_path, f"EM-rawdep-{args.em_idx}")
	os.makedirs(output_path, exist_ok = True)

	pool = Pool(args.pool_size)
	for i in range(0, len(os.listdir(em_path)) // 2, args.step_size):
		pool.apply_async(fix_single_frame, args = (i, em_path, em_depth_path, output_path, args))
	pool.close()
	pool.join()

	# fix_single_frame(0, em_path, em_depth_path, output_path, args)

if __name__ == '__main__':
	main()
