import os
import csv
import cv2
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R

from pose_fixing.evo_strat import EvolutionStrategy
from utils.cl_utils import load_all_cls, project_to_cl, get_cl_direction
from ds_gen.camera_features import camera_params
#from utils.geometry import rotate_single_vector, arbitrary_perpendicular_vector, get_vector_angle

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--em-base-path", type = str, default = "./depth-images")
	parser.add_argument("--cl-base-path", type = str, default = "./CL")
	parser.add_argument("--output-metadata", type = str, default = "new_reg_params.csv")
	#parser.add_argument("--angle-threshold", type = float, default = 20)
	
	parser.add_argument("--em-idx", type = int, default = 0)
	parser.add_argument("--try-idx", type = int, default = 0)
	parser.add_argument("--init-idx", type = int, default = 0)
	parser.add_argument("--pool-size", type = int, default = 7)
	parser.add_argument("--step-size", type = int, default = 8)
	parser.add_argument("--img-size", type = int, default = 224)
	
	parser.add_argument("--axial-scale", type = float, default = 6)
	parser.add_argument("--radial-scale-rate", type = float, default = 0.8)
	parser.add_argument("--orientation-scale", type = float, default = 0.5)
	parser.add_argument("--rot-scale", type = float, default = 40)

	parser.add_argument("--es-learning-rate", type = float, default = 0.28)
	parser.add_argument("--contour-tolerance", type = float, default = 5)
	#parser.add_argument("--fitness-func", type = str, default = "weighted_corr", choices = ["weighted_corr", "reg_mse"])
	#parser.add_argument("--parent-selection", type = str, default = "biased_roulette", choices = ["biased_roulette", "best"])

	parser.add_argument("--num-parents", type = int, default = 400)
	parser.add_argument("--num-offsprings", type = int, default = 2000)
	parser.add_argument("--num-generations", type = int, default = 12)

	parser.add_argument("--rot-samples", type = int, default = 6)
	parser.add_argument("--norm-samples", type = int, default = 3)
	parser.add_argument("--up-rot-samples", type = int, default = 6)
	parser.add_argument("--to-norm-scale", type = float, default = 0.4)

	return parser.parse_args()

def fix_single_image(args, img_idx, output_path):
	st_time = time.time()

	old_pose = np.loadtxt(os.path.join(args.em_base_path, f"EM-{args.em_idx}", f"{img_idx:06d}.txt")).reshape(-1)
	real_depth_map = np.load(os.path.join(args.em_base_path, f"EM-rawdep-{args.em_idx}", f"{img_idx:06d}.npy"))
	all_cls = load_all_cls(args.cl_base_path)

	base_position, quaternion = old_pose[ : 3], old_pose[3 : ]
	base_position, cl_indices = project_to_cl(base_position, all_cls, return_cl_indices = True)
	base_orientation = R.from_quat(quaternion).apply(camera_params["forward_direction"])
	base_orientation = base_orientation / np.linalg.norm(base_orientation)
	cl_orientation = get_cl_direction(all_cls, cl_indices)
	lumen_radius = all_cls[cl_indices[0]][1][cl_indices[1]]

	"""
	base_up = R.from_quat(quaternion).apply(camera_params["up_direction"])
	base_up = base_up / np.linalg.norm(base_up)
	arbit_up = arbitrary_perpendicular_vector(base_orientation)
	base_up_rot = get_vector_angle(arbit_up, base_up)
	angle_pos = get_vector_angle(rotate_single_vector(arbit_up, base_orientation, base_up_rot), base_up)
	angle_neg = get_vector_angle(rotate_single_vector(arbit_up, base_orientation, -base_up_rot), base_up)
	if angle_neg < angle_pos:
		base_up_rot = -base_up_rot
	"""
	
	es = EvolutionStrategy(args.mesh_path, args.img_size, real_depth_map, args.num_parents, args.num_offsprings,
		args.num_generations, args.axial_scale, lumen_radius * args.radial_scale_rate, args.orientation_scale, args.rot_scale,
		learning_rate = args.es_learning_rate, contour_tolerance = args.contour_tolerance, to_norm_scale = args.to_norm_scale)

	#es.show_sigma_samples()
	es.init_population(base_position, cl_orientation, args.norm_samples, args.rot_samples, args.up_rot_samples)
	global_optim, rgb_img, dep_img = es.run()

	cv2.imwrite(os.path.join(output_path, f"{img_idx:06d}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
	np.save(os.path.join(output_path, f"{img_idx:06d}.npy"), dep_img)

	with open(args.output_metadata, "a", newline = "") as f:
		writer = csv.writer(f)
		writer.writerow([args.em_idx, img_idx, args.try_idx, *global_optim[-1], *global_optim[ : 3]])

	print(f"Frame {img_idx} done. Estimated per frame parallel time in {((time.time() - st_time) / args.pool_size / 60):.2f} minutes.", flush = True)

def main():
	args = get_args()
	
	em_path = os.path.join(args.em_base_path, f"EM-{args.em_idx}")
	output_path = os.path.join(args.em_base_path, f"EM-newfix-{args.em_idx}-{args.try_idx}")
	os.makedirs(output_path, exist_ok = True)

	"""
	pool = Pool(args.pool_size)
	for i in range(args.init_idx, len(os.listdir(em_path)) // 2, args.step_size):
		pool.apply_async(fix_single_image, args = (args, i, output_path))
	pool.close()
	pool.join()
	"""
	
	fix_single_image(args, 1416, output_path)

if __name__ == '__main__':
	main()
