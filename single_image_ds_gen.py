import os
import argparse

from ds_gen.rotatable_single_images import generate_rotatable_images
from ds_gen.camera_features import load_from_args

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--output-path", type = str, default = "./virtual_dataset/single_image")
	parser.add_argument("--seg-cl-path", type = str, default = "./seg_cl_1")
	parser.add_argument("--img-size", type = int, default = 224)
	parser.add_argument("--num-samples", type = int, default = 50000)
	parser.add_argument("--partition", type = str, default = "train")
	parser.add_argument("--out-pose-only", action = "store_true", default = False)
	parser.add_argument("--norm-img", action = "store_true", default = False)
	parser.add_argument("--velocity-path", type = str, default = "velocity_res.csv")
	parser.add_argument("--static-stats-path", type = str, default = "./ds_gen/static_distrib.json")
	parser.add_argument("--focal-length", type = float, default = 100)
	parser.add_argument("--rotatable", action = "store_true", default = False)
	parser.add_argument("--view-angle", type = float, default = 90)
	parser.add_argument("--suffix", type = str, default = "")
	parser.add_argument("--max-axial-len", type = float, default = 120)
	parser.add_argument("--max-radius", type = float, default = 10)

	return parser.parse_args()

def main():
	args = get_args()
	load_from_args(args)

	output_path = os.path.join(args.output_path, args.partition)
	os.makedirs(output_path, exist_ok = True)
	reference_path = os.path.join(args.output_path + "_ref", args.partition)
	os.makedirs(reference_path, exist_ok = True)

	generate_rotatable_images(args.mesh_path, args.seg_cl_path, output_path, reference_path, args.num_samples, args.img_size, args.max_axial_len, args.max_radius, args.out_pose_only, args.norm_img, (2 ** -0.5) if args.rotatable else 1.0, args.suffix)

if __name__ == '__main__':
	main()
