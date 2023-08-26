import os
import argparse

from ds_gen.rotatable_single_images import generate_rotatable_images

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--output-path", type = str, default = "./virtual_dataset/single_image")
	parser.add_argument("--cl-path", type = str, default = "./CL")
	parser.add_argument("--img-size", type = int, default = 224)
	parser.add_argument("--num-samples", type = int, default = 50000)
	parser.add_argument("--partition", type = str, default = "train")
	return parser.parse_args()

def main():
	args = get_args()
	output_path = os.path.join(args.output_path, args.partition)
	os.makedirs(output_path, exist_ok = True)

	generate_rotatable_images(args.mesh_path, args.cl_path, output_path, args.num_samples, args.img_size)

if __name__ == '__main__':
	main()
