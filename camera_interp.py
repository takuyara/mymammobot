import os
import csv
import cv2
import time
import math
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R

from pose_fixing.evo_strat import EvolutionStrategy
from utils.cl_utils import load_all_cls, project_to_cl, get_cl_direction
from utils.misc import str_to_arr
from ds_gen.camera_features import camera_params

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--em-base-path", type = str, default = "./depth-images")
	parser.add_argument("--cl-base-path", type = str, default = "./CL")
	parser.add_argument("--output-metadata", type = str, default = "new_reg_params_interp.csv")
	#parser.add_argument("--angle-threshold", type = float, default = 20)
	
	parser.add_argument("--try-idx", type = int, default = 0)
	parser.add_argument("--pool-size", type = int, default = 7)
	parser.add_argument("--img-size", type = int, default = 224)
	
	parser.add_argument("--rot-scale", type = float, default = 180)

	parser.add_argument("--es-learning-rate", type = float, default = 0.28)
	parser.add_argument("--contour-tolerance", type = float, default = 5)
	#parser.add_argument("--fitness-func", type = str, default = "weighted_corr", choices = ["weighted_corr", "reg_mse"])
	#parser.add_argument("--parent-selection", type = str, default = "biased_roulette", choices = ["biased_roulette", "best"])

	parser.add_argument("--num-parents", type = int, default = 200)
	parser.add_argument("--num-offsprings", type = int, default = 1000)
	parser.add_argument("--num-generations", type = int, default = 5)
	parser.add_argument("--to-norm-scale", type = float, default = 0.4)

	parser.add_argument("--light-mask-scales", type = float, nargs = "+", default = [0.1, 0.25, 0.4])

	parser.add_argument("--input-csv-path", type = str, default = "interp_data.csv")

	parser.add_argument("--split-parts", type = int, default = 4)
	parser.add_argument("--cur-part", type = int, default = 0)


	return parser.parse_args()

def fix_single_image(args, em_idx, img_idx, position, orientation, up_rot, orientation_scale, radial_scale, axial_scale, up_rot_scale):
	st_time = time.time()

	output_path = os.path.join(args.em_base_path, f"EM-interpfix-{em_idx}-{args.try_idx}")
	os.makedirs(output_path, exist_ok = True)

	real_depth_map = np.load(os.path.join(args.em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))

	es = EvolutionStrategy(args.mesh_path, args.img_size, real_depth_map, args.num_parents, args.num_offsprings,
		args.num_generations, axial_scale, radial_scale, orientation_scale, args.rot_scale, up_rot_scale = up_rot_scale,
		learning_rate = args.es_learning_rate, contour_tolerance = args.contour_tolerance, to_norm_scale = args.to_norm_scale,
		light_mask_scales = args.light_mask_scales)

	es.init_population_norm(position, orientation, up_rot)
	global_optim, rgb_img, dep_img = es.run()

	cv2.imwrite(os.path.join(output_path, f"{img_idx:06d}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
	np.save(os.path.join(output_path, f"{img_idx:06d}.npy"), dep_img)

	with open(args.output_metadata, "a", newline = "") as f:
		writer = csv.writer(f)
		writer.writerow([em_idx, img_idx, args.try_idx, *global_optim[-1], *global_optim[ : 3]])

	print(f"Frame {img_idx} done. Estimated per frame parallel time in {((time.time() - st_time) / args.pool_size / 60):.2f} minutes.", flush = True)

def main():
	args = get_args()

	all_data = []
	with open(args.input_csv_path, newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			this_data = {"em_idx": int(row["em_idx"]), "img_idx": int(row["img_idx"]), "position": str_to_arr(row["position"]),
			"orientation": str_to_arr(row["orientation"]), "up_rot": float(row["up_rot"]), "orientation_scale": float(row["orientation_scale"]),
			"radial_scale": float(row["radial_scale"]), "axial_scale": float(row["axial_scale"]), "up_rot_scale": float(row["up_rot_scale"])}
			all_data.append(this_data)

	num_work_per = math.ceil(len(all_data) / args.split_parts)
	st = int(num_work_per * args.cur_part)
	ed = int(st + num_work_per)
	all_data = all_data[st : ed]

	pool = Pool(args.pool_size)
	for this_data in all_data:
		pool.apply_async(fix_single_image, args = (args, ), kwds = this_data)
	pool.close()
	pool.join()

	"""
	this_data = all_data[0]
	fix_single_image(args, **this_data)
	"""

if __name__ == '__main__':
	main()
