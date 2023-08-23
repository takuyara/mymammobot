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

	parser.add_argument("--light-mask-scales", type = float, nargs = "+", default = [0.1, 0.25, 0.4])

	parser.add_argument("--input-csv-path", type = str, default = None)

	parser.add_argument("--no-gpu", action = "store_true", default = False)

	return parser.parse_args()

def fix_single_image(args, em_idx, img_idx):
	st_time = time.time()

	output_path = os.path.join(args.em_base_path, f"EM-newfix-{em_idx}-{args.try_idx}")
	os.makedirs(output_path, exist_ok = True)

	old_pose = np.loadtxt(os.path.join(args.em_base_path, f"EM-{em_idx}", f"{img_idx:06d}.txt")).reshape(-1)
	real_depth_map = np.load(os.path.join(args.em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
	all_cls = load_all_cls(args.cl_base_path)

	base_position, quaternion = old_pose[ : 3], old_pose[3 : ]
	base_position, cl_indices = project_to_cl(base_position, all_cls, return_cl_indices = True)
	cl_orientation = get_cl_direction(all_cls, cl_indices)
	lumen_radius = all_cls[cl_indices[0]][1][cl_indices[1]]

	es = EvolutionStrategy(args.mesh_path, args.img_size, real_depth_map, args.num_parents, args.num_offsprings,
		args.num_generations, args.axial_scale, lumen_radius * args.radial_scale_rate, args.orientation_scale, args.rot_scale,
		learning_rate = args.es_learning_rate, contour_tolerance = args.contour_tolerance, to_norm_scale = args.to_norm_scale,
		light_mask_scales = args.light_mask_scales)

	es.init_population(base_position, cl_orientation, args.norm_samples, args.rot_samples, args.up_rot_samples)
	global_optim, rgb_img, dep_img = es.run()

	cv2.imwrite(os.path.join(output_path, f"{img_idx:06d}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
	np.save(os.path.join(output_path, f"{img_idx:06d}.npy"), dep_img)

	with open(args.output_metadata, "a", newline = "") as f:
		writer = csv.writer(f)
		writer.writerow([em_idx, img_idx, args.try_idx, *global_optim[-1], *global_optim[ : 3]])

	print(f"Frame {img_idx} done. Estimated per frame parallel time in {((time.time() - st_time) / args.pool_size / 60):.2f} minutes.", flush = True)

def main():
	args = get_args()

	em_img_indices = []
	if args.input_csv_path is None:
		em_path = os.path.join(args.em_base_path, f"EM-{args.em_idx}")
		for i in range(args.init_idx, len(os.listdir(em_path)) // 2, args.step_size):
			em_img_indices.append((args.em_idx, i))
	else:
		with open(args.input_csv_path, newline = "") as f:
			reader = csv.reader(f)
			for row in reader:
				em_img_indices.append((int(row[0]), int(row[1])))

	pool = Pool(args.pool_size)
	for em_idx, img_idx in em_img_indices:
		pool.apply_async(fix_single_image, args = (args, em_idx, img_idx))
	pool.close()
	pool.join()

if __name__ == '__main__':
	main()
